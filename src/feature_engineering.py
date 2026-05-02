"""Shared feature engineering for hockey prediction models.

All functions take a DataFrame and return it with new columns added.
Called by both train_xgb.py (regressor) and train_breakout.py (classifier).
"""

import numpy as np
import pandas as pd


# =====================
# TOI CONVERSION
# =====================

def toi_to_seconds(value):
    """Convert TOI string (MM:SS or HH:MM:SS) to seconds."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return np.nan


# =====================
# PER-GAME RATES
# =====================

def compute_per_game_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-game rate columns and dist_per_60.
    Edge tracking columns (burstsOver20, totalDistance) may be NaN for pre-2021 rows."""
    gp = df["gamesPlayed"].replace(0, np.nan)
    df["shots_pg"]     = df["shots"]        / gp
    df["pp_points_pg"] = df["pp_points"]    / gp
    if "burstsOver20" in df.columns:
        df["bursts_pg"]    = df["burstsOver20"] / gp
    if "totalDistance" in df.columns:
        df["distance_pg"]  = df["totalDistance"] / gp
        df["dist_per_60"]  = df["totalDistance"] / (df["toi"] / 3600).replace(0, np.nan)
    return df


# =====================
# TEAM CONTEXT FEATURES
# =====================

TEAM_STAT_COLS = [
    "team_pp_pct", "team_pp_goals_pg", "team_pp_opps_pg",
    "team_faceoff_pct", "team_oz_faceoff_pct", "team_dz_faceoff_pct",
]


def compute_next_team_features(df: pd.DataFrame, team_stats_path: str) -> pd.DataFrame:
    """Compute next-season team environment features for switchers/stayers."""
    df["next_team"] = df.groupby("playerId")["team"].shift(-1)
    df["switched_teams"] = (df["team"] != df["next_team"]).fillna(False).astype(int)

    team_stats = pd.read_parquet(team_stats_path)
    dest_lookup = team_stats.rename(
        columns={"team": "next_team"} | {c: f"dest_{c}" for c in TEAM_STAT_COLS}
    )
    df = df.merge(
        dest_lookup[["next_team", "season"] + [f"dest_{c}" for c in TEAM_STAT_COLS]],
        on=["next_team", "season"], how="left"
    )

    for col in TEAM_STAT_COLS:
        df[f"next_{col}"] = np.where(df["switched_teams"] == 0, df[col], df[f"dest_{col}"])

    df = df.drop(columns=[f"dest_{c}" for c in TEAM_STAT_COLS])
    return df


# =====================
# DELTA FEATURES (year-over-year changes)
# =====================

def compute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add YoY delta features, has_prior_season flag, and prev_season_gp."""
    rate_delta_cols  = ["toi", "ppg", "gpg", "apg", "oz_pct"]
    count_delta_cols = ["shots_pg", "pp_points_pg", "bursts_pg", "distance_pg", "gamesPlayed"]
    all_delta_cols   = [c for c in rate_delta_cols + count_delta_cols if c in df.columns]

    prev_vals = df.groupby("playerId")[all_delta_cols].shift(1)
    for col in all_delta_cols:
        df[f"delta_{col}"] = df[col] - prev_vals[col]

    # 1 if player has a qualifying prior season in dataset, 0 otherwise
    df["has_prior_season"] = (~prev_vals["ppg"].isna()).astype(int)

    # Games played in prior qualifying season (context for delta magnitude)
    df["prev_season_gp"] = df.groupby("playerId")["gamesPlayed"].shift(1)

    return df


# =====================
# CAREER TRAJECTORY FEATURES
# =====================

def compute_career_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add career-high PPG tracking, age curve, and career stage."""
    # Best PPG in any prior qualifying season
    df["prev_career_high_ppg"] = df.groupby("playerId")["ppg"].transform(
        lambda x: x.shift(1).expanding().max()
    )
    # How far above/below career best
    df["career_high_gap"] = df["ppg"] - df["prev_career_high_ppg"]

    # How close to career ceiling — players near their high are more likely to break through
    df["pct_of_career_high"] = df["ppg"] / df["prev_career_high_ppg"].replace(0, np.nan)

    # Career PPG trajectory — slope of PPG across all prior seasons per player
    # Positive slope = player trending upward; negative = declining
    def _ppg_slope(series):
        slopes = []
        for i in range(len(series)):
            window = series.iloc[:i+1]
            if len(window) < 2:
                slopes.append(np.nan)
            else:
                x = np.arange(len(window))
                slopes.append(np.polyfit(x, window.values, 1)[0])
        return pd.Series(slopes, index=series.index)

    df["career_ppg_slope"] = df.groupby("playerId")["ppg"].transform(_ppg_slope)

    # Age curve
    df["age_squared"] = df["age"] ** 2

    # Discrete career stage: 0=developing, 1=entering prime, 2=prime, 3=declining
    def _career_stage(age):
        if age <= 22:
            return 0
        if age <= 26:
            return 1
        if age <= 32:
            return 2
        return 3

    df["career_stage"] = df["age"].apply(_career_stage)

    return df


# =====================
# REGRESSION-TO-MEAN SIGNALS
# =====================

def compute_regression_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add shooting % and PPG regression-to-mean signals."""
    df["shooting_pct_vs_career"] = df["shootingPercentage"] - df["career_shooting_pctg"]

    career_ppg_rate = df["career_points"] / df["career_games_played"].replace(0, np.nan)
    df["ppg_vs_career_rate"] = df["ppg"] - career_ppg_rate

    return df


# =====================
# TEAM ROSTER DEPTH FEATURES
# =====================

def compute_roster_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team roster context features from the player dataset itself.

    These measure a player's position within their team's depth chart
    and identify opportunity signals (aging teammates, departures, etc.).
    """
    # --- Players ahead: how many teammates had higher PPG this season ---
    def _count_ahead(group):
        ppg_vals = group["ppg"].values
        counts = [(ppg_vals > v).sum() for v in ppg_vals]
        return pd.Series(counts, index=group.index)

    df["players_ahead_on_team"] = df.groupby(["team", "season"]).apply(_count_ahead).droplevel([0, 1])

    # Same but position-specific: how many teammates at same position group had higher PPG
    df["_pos_group"] = np.where(df["is_defenseman"] == 1, "D", "F")

    def _count_ahead_pos(group):
        ppg_vals = group["ppg"].values
        counts = [(ppg_vals > v).sum() for v in ppg_vals]
        return pd.Series(counts, index=group.index)

    df["pos_players_ahead"] = df.groupby(["team", "season", "_pos_group"]).apply(
        _count_ahead_pos
    ).droplevel([0, 1, 2])

    # --- Team's top PPG player age (is the star aging out?) ---
    def _top_player_age(group):
        top_idx = group["ppg"].idxmax()
        return pd.Series(group.loc[top_idx, "age"], index=group.index)

    df["team_top_player_age"] = df.groupby(["team", "season"]).apply(_top_player_age).droplevel([0, 1])

    # --- Team PP concentration: what fraction of team PP points go to top 2 players ---
    def _pp_concentration(group):
        pp = group["pp_points"].values
        total_pp = pp.sum()
        if total_pp == 0:
            return pd.Series(0.0, index=group.index)
        top2 = np.sort(pp)[-2:].sum() if len(pp) >= 2 else pp.sum()
        return pd.Series(top2 / total_pp, index=group.index)

    df["team_pp_concentration"] = df.groupby(["team", "season"]).apply(
        _pp_concentration
    ).droplevel([0, 1])

    # --- Player's share of team PP points ---
    team_pp_total = df.groupby(["team", "season"])["pp_points"].transform("sum").replace(0, np.nan)
    df["player_pp_share"] = df["pp_points"] / team_pp_total

    # --- Team roster turnover: count of players on this team last season who are gone ---
    # Build a set of (team, season) -> set of playerIds
    # Then for each (team, season), count how many of last season's players left
    prev_season_map = dict(zip(
        df["season"].unique()[1:],
        df["season"].unique()[:-1]
    ))

    team_rosters = df.groupby(["team", "season"])["playerId"].apply(set).to_dict()

    def _roster_turnover(row):
        prev_season = prev_season_map.get(row["season"])
        if prev_season is None:
            return np.nan
        prev_roster = team_rosters.get((row["team"], prev_season), set())
        curr_roster = team_rosters.get((row["team"], row["season"]), set())
        if len(prev_roster) == 0:
            return np.nan
        departed = len(prev_roster - curr_roster)
        return departed / len(prev_roster)

    df["team_roster_turnover"] = df.apply(_roster_turnover, axis=1)

    # --- Position interaction features ---
    df["defenseman_x_delta_ppg"] = df["is_defenseman"] * df.get("delta_ppg", 0)
    df["defenseman_x_pp_points_pg"] = df["is_defenseman"] * df["pp_points_pg"]

    # Clean up temp column
    df = df.drop(columns=["_pos_group"])

    return df


# =====================
# ORCHESTRATOR
# =====================

def engineer_features(df: pd.DataFrame, team_stats_path: str) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Call order matters — later steps depend on columns created by earlier ones.
    """
    # Drop goalies — skater PPG models should not see them as trivially-zero rows
    if "position" in df.columns:
        n_before = len(df)
        df = df[df["position"] != "G"].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Dropped {n_dropped} goalie rows (position == 'G')")

    # TOI string → seconds
    if "toi" in df.columns:
        df["toi"] = df["toi"].apply(toi_to_seconds)

    # Numeric season year
    if "season" in df.columns:
        df["season_year"] = df["season"].astype(str).str[:4].astype(int)

    # Sort before any shift-based computations
    df = df.sort_values(["playerId", "season"]).reset_index(drop=True)

    # Per-game rates (needed by delta features)
    df = compute_per_game_rates(df)

    # Team context
    df = compute_next_team_features(df, team_stats_path)

    # Year-over-year deltas
    df = compute_delta_features(df)

    # Career trajectory
    df = compute_career_trajectory_features(df)

    # Regression signals
    df = compute_regression_signals(df)

    # Team roster depth & position interactions
    df = compute_roster_depth_features(df)

    # Physical features
    df["over_6ft"] = (df["height_in"] >= 72).astype(int)

    # Breakout-signal features

    # Points per minute of ice time — efficiency metric
    # High ppg_per_min + low TOI = upside if role expands
    toi_minutes = (df["toi"] / 60).replace(0, np.nan)
    df["ppg_per_minute"] = df["ppg"] / toi_minutes

    # Interaction: young player getting more ice time = breakout signal
    df["age_x_delta_toi"] = df["age"] * df.get("delta_toi", 0)

    # PPG league percentile — where this player ranks within their season
    # Elite players (99th percentile) have no room to "break out"
    df["ppg_league_percentile"] = df.groupby("season")["ppg"].rank(pct=True)

    # Interaction: PPG * career_high_gap — separates elite players peaking
    # (high PPG + positive gap = already at ceiling) from emerging players
    # (low PPG + positive gap = trending upward with room to grow)
    df["ppg_x_career_high_gap"] = df["ppg"] * df["career_high_gap"]

    # Years since draft — late bloomers (4-5 years post-draft) have breakout potential
    if "draft_overall_pick" in df.columns and "season_year" in df.columns:
        draft_year = df.get("draft_year")
        if draft_year is None and "draft_overall_pick" in df.columns:
            # Approximate draft year from age and season: drafted at ~18
            df["years_since_draft"] = df["season_year"] - (df["season_year"] - df["age"] + 18)
            # Simpler: years_since_draft ≈ age - 18 for drafted players, NaN for undrafted
            df["years_since_draft"] = np.where(
                df["is_undrafted"] == 0,
                df["age"] - 18,
                np.nan
            )

    return df
