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

FEATURE_SEASON = 20232024  # for 23-24 season stats
GAME_TYPE = 2  # 2 = regular season

MAX_WORKERS = 8
REQUEST_SLEEP = 0.15
MIN_GAMES_FILTER = 1

OUTPUT_PATH = "edge_data/nhl_full_stats.parquet"

# Optional: limit number of players fetched (for quick tests)
# LIMIT = int(os.environ.get("PLAYER_LIMIT", "0"))
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
    for team in team_abbrevs:
        roster = safe_get(f"{BASE}/roster/{team}/{FEATURE_SEASON}")
        if roster:
            for group in roster.values():
                if isinstance(group, list):
                    for p in group:
                        player_ids.add(p["id"])
        time.sleep(0.1)

    return sorted(player_ids)


def find_season_totals(data, season, game_type):
    """Find NHL season totals entry for a given season and game type."""
    for entry in data.get("seasonTotals", []):
        if (
            entry.get("season") == season
            and entry.get("gameTypeId") == game_type
            and entry.get("leagueAbbrev") == "NHL"
        ):
            return entry
    return {}


def extract_profile_features(data):
    """Extract requested profile features for 23-24 and career."""
    player_id = data.get("playerId")
    first = (data.get("firstName", {}) or {}).get("default")
    last = (data.get("lastName", {}) or {}).get("default")
    full_name = " ".join([n for n in [first, last] if n]) or None

    # 23-24 season totals (NHL regular season)
    season_entry = find_season_totals(data, FEATURE_SEASON, GAME_TYPE)

    # career totals (regular season)
    career = data.get("careerTotals", {}).get("regularSeason", {})

    row = {
        "playerId": player_id,
        "fullName": full_name,

        # 23-24
        "pp_goals_2324": season_entry.get("powerPlayGoals"),
        "pp_points_2324": season_entry.get("powerPlayPoints"),
        "toi_2324": season_entry.get("avgToi"),
        "pp_toi_2324": season_entry.get("avgPowerPlayToi"),
        "gamesPlayed_2324": season_entry.get("gamesPlayed"),

        # career (regular season)
        "career_points": career.get("points"),
        "career_games_played": career.get("gamesPlayed"),
        "career_pp_points": career.get("powerPlayPoints"),
        "career_pp_assists": career.get("powerPlayAssists"),
        "career_pp_goals": career.get("powerPlayGoals"),
        "career_shooting_pctg": career.get("shootingPctg"),
    }

    return row


def fetch_player_profile(pid):
    url = f"{BASE}/player/{pid}/landing"
    data = safe_get(url)
    if not data:
        return None

    row = extract_profile_features(data)
    time.sleep(REQUEST_SLEEP)
    return row


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
        row = future.result()
        if row:
            rows.append(row)
        if i % 25 == 0:
            print(f"Processed {i}/{len(player_ids)} players")

# Build DataFrame
profile_df = pd.DataFrame(rows)

# Filter to players with some games in 23-24 (optional)
if "gamesPlayed_2324" in profile_df.columns:
    profile_df = profile_df[profile_df["gamesPlayed_2324"].fillna(0) >= MIN_GAMES_FILTER]

print(f"Final profile dataset size: {profile_df.shape}")

# Save parquet
os.makedirs("edge_data", exist_ok=True)
profile_df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Saved to {OUTPUT_PATH}")
