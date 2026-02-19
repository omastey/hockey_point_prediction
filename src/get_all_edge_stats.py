import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# CONFIG
# =====================
BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

FEATURE_SEASON = "20232024"
TARGET_SEASON = "20242025"
GAME_TYPE = 2

MAX_WORKERS = 8          # Parallel requests
REQUEST_SLEEP = 0.15     # Soft rate limit
MIN_GAMES_FILTER = 10

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


def extract_features(data):
    p = data["player"]
    first = p.get("firstName", {}).get("default")
    last = p.get("lastName", {}).get("default")

    row = {
        "playerId": p["id"],
        "fullName": f"{first} {last}" if first and last else None,
        "position": p.get("position"),
        "shoots": p.get("shootsCatches"),
        "gamesPlayed_2324": p.get("gamesPlayed", 0),
        "goals_2324": p.get("goals", 0),
        "assists_2324": p.get("assists", 0),
        "points_2324": p.get("points", 0),
        "team_2324": p.get("team", {}).get("abbrev"),
        "shoots": p.get("shootsCatches"),
        "height_in": p.get("heightInInches"),
        "weight_lb": p.get("weightInPounds"),
    }

    birth_year = int(p["birthDate"][:4])
    row["age_2324"] = 2023 - birth_year

    gp = max(row["gamesPlayed_2324"], 1)
    row["ppg_2324"] = row["points_2324"] / gp
    row["gpg_2324"] = row["goals_2324"] / gp
    row["apg_2324"] = row["assists_2324"] / gp

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

        # --- Handedness Encoding ---
    row["shoots_left"] = 1 if row["shoots"] == "L" else 0
    row["shoots_right"] = 1 if row["shoots"] == "R" else 0

        # --- Position Encoding ---
    pos = row["position"]

    row["is_center"] = 1 if pos == "C" else 0
    row["is_winger"] = 1 if pos in ["L", "R"] else 0
    row["is_defenseman"] = 1 if pos == "D" else 0

    sog = data.get("sogSummary", [])
    if isinstance(sog, list):
        all_entry = next((e for e in sog if e.get("locationCode") == "all"), None)
        if all_entry:
            row["shots_2324"] = all_entry.get("shots")
            row["shotsPercentile_2324"] = all_entry.get("shotsPercentile")
            # Map shootingPctg -> shootingPercentage for clarity
            row["shootingPercentage_2324"] = all_entry.get("shootingPctg")
            row["shootingPctgPercentile_2324"] = all_entry.get("shootingPctgPercentile")
    
    return row


def extract_target(data):
    p = data["player"]
    points = p.get("points", 0) or 0
    gp = p.get("gamesPlayed", 0) or 0
    # Protect against divide-by-zero
    target_ppg = (points / gp) if gp else 0.0

    return {
        "team_2425": p.get("team", {}).get("abbrev"),
        "target_points_2425": points,
        "target_goals_2425": p.get("goals", 0),
        "target_assists_2425": p.get("assists", 0),
        "gp_2425": gp,
        "target_ppg_2425": target_ppg,
    }


def build_player_row(pid):
    f_url = f"{BASE}/edge/skater-detail/{pid}/{FEATURE_SEASON}/{GAME_TYPE}"
    t_url = f"{BASE}/edge/skater-detail/{pid}/{TARGET_SEASON}/{GAME_TYPE}"

    data_f = safe_get(f_url)
    data_t = safe_get(t_url)

    if not data_f or not data_t:
        return None

    row = extract_features(data_f)
    row.update(extract_target(data_t))
    row["switched_teams"] = row["team_2324"] != row["team_2425"]

    time.sleep(REQUEST_SLEEP)
    return row


# =====================
# GET TEAM ABBREVS
# =====================
print("Fetching team list...")
standings = safe_get(f"{BASE}/standings/now")
team_abbrevs = {t["teamAbbrev"]["default"] for t in standings["standings"]}

# =====================
# GET PLAYER IDS
# =====================
print("Fetching player IDs...")
player_ids = set()

for team in team_abbrevs:
    roster = safe_get(f"{BASE}/roster/{team}/{FEATURE_SEASON}")
    if roster:
        for group in roster.values():
            if isinstance(group, list):
                for p in group:
                    player_ids.add(p["id"])
    time.sleep(0.1)

player_ids = sorted(player_ids)
print(f"Found {len(player_ids)} players")

# =====================
# PARALLEL DATA COLLECTION
# =====================
print("Downloading player data in parallel...")
model_rows = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(build_player_row, pid): pid for pid in player_ids}

    for i, future in enumerate(as_completed(futures), 1):
        row = future.result()
        if row:
            model_rows.append(row)

        if i % 25 == 0:
            print(f"Processed {i}/{len(player_ids)} players")

# =====================
# BUILD DATAFRAME
# =====================
df = pd.DataFrame(model_rows)

# Apply minimum games filter to both seasons (feature + target)
df = df[
    (df["gamesPlayed_2324"] > MIN_GAMES_FILTER)
    & (df["gp_2425"] > MIN_GAMES_FILTER)
]

print(f"Final dataset size: {df.shape}")

# =====================
# SAVE PARQUET
# =====================
import os
out_path = "edge_data/nhl_edge_model_dataset.parquet"
os.makedirs("edge_data", exist_ok=True)
df.to_parquet(out_path, index=False)

print(f"✅ Dataset saved to {out_path}")
