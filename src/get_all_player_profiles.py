import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# =====================
# CONFIG
# =====================
try:
    from .constants import SEASONS, GAME_TYPE, MIN_GAMES_FILTER
except ImportError:
    from constants import SEASONS, GAME_TYPE, MIN_GAMES_FILTER

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_WORKERS = 8
REQUEST_SLEEP = 0.15

OUTPUT_PATH = "data/nhl_full_stats.parquet"

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


def get_team_name_to_abbrev():
    """Build a mapping from full team name → abbreviation using standings.
    Includes historical team names for relocated/renamed franchises."""
    standings = safe_get(f"{BASE}/standings/now")
    mapping = {}
    for t in standings["standings"]:
        name = t.get("teamName", {}).get("default")
        abbrev = t.get("teamAbbrev", {}).get("default")
        if name and abbrev:
            mapping[name] = abbrev

    # Historical team names (2010-present era)
    historical = {
        "Atlanta Thrashers": "ATL",
        "Phoenix Coyotes": "PHX",
        "Utah Hockey Club": "UTA",
        "Arizona Coyotes": "ARI",
    }
    for name, abbrev in historical.items():
        if name not in mapping:
            mapping[name] = abbrev

    return mapping


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


def extract_profile_features(data, team_name_to_abbrev=None):
    """Return one row per (player, season) with full season stats and cumulative
    career stats up to (but not including) that season."""
    player_id = data.get("playerId")
    first = (data.get("firstName", {}) or {}).get("default")
    last = (data.get("lastName", {}) or {}).get("default")
    full_name = " ".join([n for n in [first, last] if n]) or None

    # Player-level attributes
    position = data.get("position")
    shoots_catches = data.get("shootsCatches")
    height_in = data.get("heightInInches")
    weight_lb = data.get("weightInPounds")
    birth_date = data.get("birthDate")  # "YYYY-MM-DD"

    # Draft details (static per player)
    draft = data.get("draftDetails") or {}
    draft_round = draft.get("round")
    draft_overall_pick = draft.get("overallPick")
    is_undrafted = 1 if not draft else 0

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

        # Basic counting stats
        goals = season_entry.get("goals", 0) or 0
        assists = season_entry.get("assists", 0) or 0
        points = season_entry.get("points", 0) or 0
        shots = season_entry.get("shots", 0) or 0
        shooting_pctg = season_entry.get("shootingPctg")
        pp_goals = season_entry.get("powerPlayGoals")
        pp_points = season_entry.get("powerPlayPoints")
        toi = season_entry.get("avgToi")

        # Team for this season (map full name → abbreviation)
        team_name_raw = season_entry.get("teamName", {})
        if isinstance(team_name_raw, dict):
            full_name_team = team_name_raw.get("default")
        else:
            full_name_team = None
        team = None
        if full_name_team and team_name_to_abbrev:
            team = team_name_to_abbrev.get(full_name_team)

        # Per-game rates
        gp = max(gp_this_season, 1)
        ppg = points / gp
        gpg = goals / gp
        apg = assists / gp

        # Age: season start year - birth year
        age = None
        if birth_date:
            try:
                birth_year = int(birth_date[:4])
                season_year = int(season[:4])
                age = season_year - birth_year
            except (ValueError, TypeError):
                pass

        # Position flags
        is_center = 1 if position == "C" else 0
        is_winger = 1 if position in ("L", "R") else 0
        is_defenseman = 1 if position == "D" else 0

        # Handedness flags
        shoots_left = 1 if shoots_catches == "L" else 0
        shoots_right = 1 if shoots_catches == "R" else 0

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
            # Season stats
            "gamesPlayed": gp_this_season,
            "goals": goals,
            "assists": assists,
            "points": points,
            "shots": shots,
            "shootingPercentage": (shooting_pctg * 100) if shooting_pctg is not None else None,
            "pp_goals": pp_goals,
            "pp_points": pp_points,
            "toi": toi,
            "team": team,
            # Per-game rates
            "ppg": ppg,
            "gpg": gpg,
            "apg": apg,
            # Player attributes
            "position": position,
            "shoots": shoots_catches,
            "height_in": height_in,
            "weight_lb": weight_lb,
            "age": age,
            # Position & handedness flags
            "is_center": is_center,
            "is_winger": is_winger,
            "is_defenseman": is_defenseman,
            "shoots_left": shoots_left,
            "shoots_right": shoots_right,
            # Draft
            "draft_round": draft_round,
            "draft_overall_pick": draft_overall_pick,
            "is_undrafted": is_undrafted,
            # Career stats (prior seasons)
            "career_points": _sum("points"),
            "career_games_played": _sum("gamesPlayed"),
            "career_goals": career_goals,
            "career_assists": _sum("assists"),
            "career_pp_points": _sum("powerPlayPoints"),
            "career_pp_goals": _sum("powerPlayGoals"),
            "career_shooting_pctg": career_shooting_pctg,
            "years_in_nhl": len([e for e in prior if (e.get("gamesPlayed") or 0) >= MIN_GAMES_FILTER]),
        }
        rows.append(row)

    return rows


def fetch_player_profile(pid, team_name_to_abbrev=None):
    url = f"{BASE}/player/{pid}/landing"
    data = safe_get(url)
    if not data:
        return []

    rows = extract_profile_features(data, team_name_to_abbrev)
    time.sleep(REQUEST_SLEEP)
    return rows


# =====================
# MAIN
# =====================
print("Building team name → abbreviation mapping...")
TEAM_NAME_TO_ABBREV = get_team_name_to_abbrev()
print(f"Mapped {len(TEAM_NAME_TO_ABBREV)} team names to abbreviations")

print("Resolving player IDs from rosters...")
player_ids = get_player_ids_from_rosters()

if LIMIT > 0:
    player_ids = player_ids[:LIMIT]

print(f"Fetching profiles for {len(player_ids)} players...")

rows = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_player_profile, pid, TEAM_NAME_TO_ABBREV): pid for pid in player_ids}
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        rows.extend(result)
        if i % 25 == 0:
            print(f"Processed {i}/{len(player_ids)} players")

profile_df = pd.DataFrame(rows)

print(f"Final profile dataset size: {profile_df.shape}")

# Save parquet
os.makedirs("data", exist_ok=True)
profile_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
