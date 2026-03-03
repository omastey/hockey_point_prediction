import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# =====================
# CONFIG
# =====================
BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025"]
GAME_TYPE = 2  # 2 = regular season

MAX_WORKERS = 8
REQUEST_SLEEP = 0.15
MIN_GAMES_FILTER = 10

OUTPUT_PATH = "edge_data/nhl_full_stats.parquet"

LIMIT = 0  # Set to >0 to limit number of players (for testing)


# =====================
# HELPERS
# =====================

def safe_get(url, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            time.sleep(0.5)
    return None


def get_player_ids_from_edge_parquet():
    """Use existing EDGE dataset if available."""
    parquet_path = "edge_data/nhl_edge_model_dataset.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path, columns=["playerId"])
        return sorted(df["playerId"].dropna().astype(int).unique().tolist())
    return None


def get_player_ids_from_rosters():
    standings = safe_get(f"{BASE}/standings/now")
    team_abbrevs = {t["teamAbbrev"]["default"] for t in standings["standings"]}

    player_ids = set()
    for season in SEASONS:
        for team in team_abbrevs:
            roster = safe_get(f"{BASE}/roster/{team}/{season}")
            if roster:
                for group in roster.values():
                    if isinstance(group, list):
                        for p in group:
                            player_ids.add(p["id"])
            time.sleep(0.1)

    return sorted(player_ids)


def extract_profile_features(data):
    """Return one row per (player, season) with season stats and cumulative
    career stats up to (but not including) that season."""
    player_id = data.get("playerId")
    first = (data.get("firstName", {}) or {}).get("default")
    last = (data.get("lastName", {}) or {}).get("default")
    full_name = " ".join([n for n in [first, last] if n]) or None

    # All NHL regular-season entries from the player's history
    nhl_reg_totals = [
        e for e in data.get("seasonTotals", [])
        if e.get("gameTypeId") == GAME_TYPE and e.get("leagueAbbrev") == "NHL"
    ]

    rows = []
    for season in SEASONS:
        season_int = int(season)

        # Season stats for this specific season
        season_entry = next(
            (e for e in nhl_reg_totals if e.get("season") == season_int), {}
        )

        gp_this_season = season_entry.get("gamesPlayed", 0) or 0
        if gp_this_season < MIN_GAMES_FILTER:
            continue

        # Career stats = cumulative of all NHL regular seasons BEFORE this one
        prior = [e for e in nhl_reg_totals if (e.get("season") or 0) < season_int]

        def _sum(field):
            return sum(e.get(field) or 0 for e in prior)

        career_goals = _sum("goals")
        career_shots = _sum("shots")
        career_shooting_pctg = (career_goals / career_shots * 100) if career_shots > 0 else None

        row = {
            "playerId": player_id,
            "fullName": full_name,
            "season": season,
            "pp_goals": season_entry.get("powerPlayGoals"),
            "pp_points": season_entry.get("powerPlayPoints"),
            "toi": season_entry.get("avgToi"),
            "pp_toi": season_entry.get("avgPowerPlayToi"),
            "career_points": _sum("points"),
            "career_games_played": _sum("gamesPlayed"),
            "career_goals": career_goals,
            "career_assists": _sum("assists"),
            "career_pp_points": _sum("powerPlayPoints"),
            "career_pp_goals": _sum("powerPlayGoals"),
            "career_shooting_pctg": career_shooting_pctg,
        }
        rows.append(row)

    return rows


def fetch_player_profile(pid):
    url = f"{BASE}/player/{pid}/landing"
    data = safe_get(url)
    if not data:
        return []

    rows = extract_profile_features(data)
    time.sleep(REQUEST_SLEEP)
    return rows


# =====================
# MAIN
# =====================
print("Resolving player IDs...")
player_ids = get_player_ids_from_edge_parquet()
if not player_ids:
    player_ids = get_player_ids_from_rosters()

if LIMIT > 0:
    player_ids = player_ids[:LIMIT]

print(f"Fetching profiles for {len(player_ids)} players...")

rows = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_player_profile, pid): pid for pid in player_ids}
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        rows.extend(result)
        if i % 25 == 0:
            print(f"Processed {i}/{len(player_ids)} players")

profile_df = pd.DataFrame(rows)

print(f"Final profile dataset size: {profile_df.shape}")

# Save parquet
os.makedirs("edge_data", exist_ok=True)
profile_df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Saved to {OUTPUT_PATH}")
