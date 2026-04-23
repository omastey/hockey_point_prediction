import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
try:
    from .constants import SEASONS, GAME_TYPE, MIN_GAMES_FILTER
except ImportError:
    from constants import SEASONS, GAME_TYPE, MIN_GAMES_FILTER

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_WORKERS = 8      # Parallel requests
REQUEST_SLEEP = 0.15  # Soft rate limit

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


def extract_season_stats(data, season):
    p = data["player"]
    first = p.get("firstName", {}).get("default")
    last = p.get("lastName", {}).get("default")

    row = {
        "playerId": p["id"],
        "fullName": f"{first} {last}" if first and last else None,
        "season": season,
        "position": p.get("position"),
        "shoots": p.get("shootsCatches"),
        "gamesPlayed": p.get("gamesPlayed", 0),
        "goals": p.get("goals", 0),
        "assists": p.get("assists", 0),
        "points": p.get("points", 0),
        "team": p.get("team", {}).get("abbrev"),
        "height_in": p.get("heightInInches"),
        "weight_lb": p.get("weightInPounds"),
    }

    season_year = int(season[:4])
    birth_year = int(p["birthDate"][:4])
    row["age"] = season_year - birth_year

    gp = max(row["gamesPlayed"], 1)
    row["ppg"] = row["points"] / gp
    row["gpg"] = row["goals"] / gp
    row["apg"] = row["assists"] / gp

    row["topShotSpeed"] = data.get("topShotSpeed", {}).get("imperial")
    row["topShotSpeed_pct"] = data.get("topShotSpeed", {}).get("percentile")

    speed = data.get("skatingSpeed", {}).get("speedMax", {})
    row["speedMax"] = speed.get("imperial")
    row["speedMax_pct"] = speed.get("percentile")

    bursts = data.get("skatingSpeed", {}).get("burstsOver20", {})
    row["burstsOver20"] = bursts.get("value")

    row["totalDistance"] = data.get("totalDistanceSkated", {}).get("imperial")

    z = data.get("zoneTimeDetails", {})
    row["oz_pct"] = z.get("offensiveZonePctg")
    row["dz_pct"] = z.get("defensiveZonePctg")

    row["shoots_left"] = 1 if row["shoots"] == "L" else 0
    row["shoots_right"] = 1 if row["shoots"] == "R" else 0

    pos = row["position"]
    row["is_center"] = 1 if pos == "C" else 0
    row["is_winger"] = 1 if pos in ["L", "R"] else 0
    row["is_defenseman"] = 1 if pos == "D" else 0

    sog = data.get("sogSummary", [])
    if isinstance(sog, list):
        all_entry = next((e for e in sog if e.get("locationCode") == "all"), None)
        if all_entry:
            row["shots"] = all_entry.get("shots")
            row["shotsPercentile"] = all_entry.get("shotsPercentile")
            row["shootingPercentage"] = all_entry.get("shootingPctg")
            row["shootingPctgPercentile"] = all_entry.get("shootingPctgPercentile")

    return row


def build_player_rows(pid):
    """Fetch EDGE stats for a single player across all seasons.
    Returns a list of row dicts, one per season with >= MIN_GAMES_FILTER games.
    switched_teams reflects whether the player changed teams vs the prior season.
    """
    rows = []
    for season in SEASONS:
        url = f"{BASE}/edge/skater-detail/{pid}/{season}/{GAME_TYPE}"
        data = safe_get(url)
        if not data:
            time.sleep(REQUEST_SLEEP)
            continue
        row = extract_season_stats(data, season)
        if row["gamesPlayed"] >= MIN_GAMES_FILTER:
            rows.append(row)
        time.sleep(REQUEST_SLEEP)

    # Compare each season's team to the prior qualifying season's team
    for i, row in enumerate(rows):
        row["switched_teams"] = False if i == 0 else row["team"] != rows[i - 1]["team"]

    return rows


# =====================
# GET TEAM ABBREVS
# =====================
print("Fetching team list...")
standings = safe_get(f"{BASE}/standings/now")
team_abbrevs = {t["teamAbbrev"]["default"] for t in standings["standings"]}

# =====================
# GET PLAYER IDS (from all seasons)
# =====================
print("Fetching player IDs across all seasons...")
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

player_ids = sorted(player_ids)
print(f"Found {len(player_ids)} unique players across {len(SEASONS)} seasons")

# =====================
# PARALLEL DATA COLLECTION
# =====================
print("Downloading player data in parallel...")
model_rows = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(build_player_rows, pid): pid for pid in player_ids}

    for i, future in enumerate(as_completed(futures), 1):
        rows = future.result()
        model_rows.extend(rows)

        if i % 25 == 0:
            print(f"Processed {i}/{len(player_ids)} players")

# =====================
# BUILD DATAFRAME
# =====================
df = pd.DataFrame(model_rows)

print(f"Final dataset size: {df.shape}")

# =====================
# SAVE PARQUET
# =====================
import os
out_path = "edge_data/nhl_edge_model_dataset.parquet"
os.makedirs("edge_data", exist_ok=True)
df.to_parquet(out_path, index=False)

print(f"✅ Dataset saved to {out_path}")
