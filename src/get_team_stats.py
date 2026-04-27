import os
import time
import requests
import pandas as pd

# =====================
# CONFIG
# =====================
try:
    from .constants import SEASONS, GAME_TYPE
except ImportError:
    from constants import SEASONS, GAME_TYPE

STATS_BASE = "https://api.nhle.com/stats/rest/en/team"
HEADERS = {"User-Agent": "Mozilla/5.0"}

REQUEST_SLEEP = 0.15
OUTPUT_PATH = "data/nhl_team_stats.parquet"


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


def fetch_report(report, season):
    season_id = int(season)
    url = f"{STATS_BASE}/{report}?cayenneExp=seasonId={season_id}%20and%20gameTypeId={GAME_TYPE}"
    data = safe_get(url)
    return data.get("data", []) if data else []


# =====================
# TEAM ID LOOKUP
# =====================
print("Fetching team ID → abbreviation lookup...")
teams_data = safe_get(STATS_BASE)
id_to_abbrev = {t["id"]: t["triCode"] for t in teams_data.get("data", [])}
print(f"Found {len(id_to_abbrev)} teams")

# =====================
# FETCH STATS PER SEASON
# =====================
rows = []

for season in SEASONS:
    print(f"Fetching {season}...")

    pp_rows = fetch_report("powerplay", season)
    fo_rows = fetch_report("faceoffpercentages", season)
    time.sleep(REQUEST_SLEEP)

    fo_by_team = {r["teamId"]: r for r in fo_rows}

    for pp in pp_rows:
        team_id = pp.get("teamId")
        abbrev  = id_to_abbrev.get(team_id)
        if not abbrev:
            continue

        fo = fo_by_team.get(team_id, {})

        rows.append({
            "team":                abbrev,
            "season":              season,
            "team_pp_pct":         pp.get("powerPlayPct"),
            "team_pp_goals_pg":    pp.get("ppGoalsPerGame"),
            "team_pp_opps_pg":     pp.get("ppOpportunitiesPerGame"),
            "team_faceoff_pct":    fo.get("faceoffWinPct"),
            "team_oz_faceoff_pct": fo.get("offensiveZoneFaceoffPct"),
            "team_dz_faceoff_pct": fo.get("defensiveZoneFaceoffPct"),
        })

# =====================
# SAVE
# =====================
df = pd.DataFrame(rows)
print(f"Team stats shape: {df.shape}")

os.makedirs("data", exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Saved to {OUTPUT_PATH}")
