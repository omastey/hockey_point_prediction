import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DISABLE_BASE_STATS = "--no-base-stats" in sys.argv

# Try XGBoost first; fall back to RandomForest if unavailable (e.g., missing libomp on macOS)
try:
    from xgboost import XGBRegressor
    _USE_XGB = True
except Exception as e:
    print("XGBoost unavailable, falling back to RandomForestRegressor:", e)
    from sklearn.ensemble import RandomForestRegressor
    _USE_XGB = False


def show_player_predictions(df, player_names,
                            player_col="fullName",
                            actual_col="target_ppg_next",
                            pred_col="Predicted_PPG_Next"):
    """Lookup predicted vs actual PPG for player(s) across all their seasons."""
    if isinstance(player_names, str):
        player_names = [player_names]

    result = df[df[player_col].isin(player_names)][
        [player_col, "season", actual_col, pred_col, "Prediction_Error"]
    ].sort_values([player_col, "season"])

    if result.empty:
        print("No matching players found.")
        return None

    print(result.to_string(index=False))
    return result


# =====================
# LOAD DATA
# =====================
df = pd.read_parquet("edge_data/nhl_merged_dataset.parquet")
print(f"Dataset shape: {df.shape}")

# =====================
# FEATURE PREPROCESSING
# =====================
def toi_to_seconds(value):
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

if "toi" in df.columns:
    df["toi"] = df["toi"].apply(toi_to_seconds)

# Encode season as a numeric year (e.g. "20232024" -> 2023)
if "season" in df.columns:
    df["season_year"] = df["season"].astype(str).str[:4].astype(int)

# Sort before any shift-based computations
df = df.sort_values(["playerId", "season"]).reset_index(drop=True)

# Distance per 60 min of ice time (toi is already in seconds)
df["dist_per_60"] = df["totalDistance"] / (df["toi"] / 3600).replace(0, np.nan)

# =====================
# TEAM CONTEXT FEATURES
# Goal: describe the offensive environment the player will play in next season
# without using next-season game results (no leakage).
# =====================

# --- Step A: Per-game rate stats (also reused by DELTA FEATURES below) ---
gp = df["gamesPlayed"].replace(0, np.nan)
df["shots_pg"]     = df["shots"]        / gp
df["pp_points_pg"] = df["pp_points"]    / gp
df["bursts_pg"]    = df["burstsOver20"] / gp
df["distance_pg"]  = df["totalDistance"] / gp

# --- Step B: Leave-one-out team environment averages ---
# Subtract each player's own contribution before averaging so a star player's
# stats don't dominate their own team context signal.
team_totals = df.groupby(["team", "season"]).agg(
    _sum_pp=("pp_points_pg", "sum"),
    _sum_goals=("gpg", "sum"),
    _sum_shots=("shots_pg", "sum"),
    _count=("playerId", "count"),
).reset_index()
df = df.merge(team_totals, on=["team", "season"], how="left")
df["team_pp_pts_pg"] = (df["_sum_pp"]    - df["pp_points_pg"]) / (df["_count"] - 1).clip(lower=1)
df["team_goals_pg"]  = (df["_sum_goals"] - df["gpg"])          / (df["_count"] - 1).clip(lower=1)
df["team_shots_pg"]  = (df["_sum_shots"] - df["shots_pg"])     / (df["_count"] - 1).clip(lower=1)
df = df.drop(columns=["_sum_pp", "_sum_goals", "_sum_shots", "_count"])

# --- Step C: Rolling 3-season team PP quality (more stable than single-season) ---
team_pp_history = df.groupby(["team", "season"])["team_pp_pts_pg"].first().reset_index()
team_pp_history = team_pp_history.sort_values(["team", "season"])
team_pp_history["team_pp_pts_pg_3yr"] = (
    team_pp_history.groupby("team")["team_pp_pts_pg"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)
df = df.merge(team_pp_history[["team", "season", "team_pp_pts_pg_3yr"]], on=["team", "season"], how="left")

# --- Step D: Attach destination team's current-season environment stats ---
# Use next season's team + current season's stats for that team — no leakage.
df["next_team"] = df.groupby("playerId")["team"].shift(-1)

team_env_lookup = df.groupby(["team", "season"]).agg(
    dest_team_pp_pts_pg=("team_pp_pts_pg", "first"),
    dest_team_pp_3yr=("team_pp_pts_pg_3yr", "first"),
    dest_team_goals_pg=("team_goals_pg", "first"),
).reset_index().rename(columns={"team": "next_team"})
df = df.merge(team_env_lookup, on=["next_team", "season"], how="left")

# --- Step E: PP quality delta — destination team vs current team ---
# Positive = moving to a better PP system (upside). Negative = worse (fall-off risk).
# Near zero for players staying on the same team.
df["team_pp_quality_delta"] = df["dest_team_pp_3yr"] - df["team_pp_pts_pg_3yr"]

# --- Step F: switched_teams — use existing column if present, else compute ---
if "switched_teams" not in df.columns:
    df["switched_teams"] = (df["team"] != df["next_team"]).astype(int)

# =====================
# DELTA FEATURES (year-over-year changes)
# =====================
# shots_pg, pp_points_pg, bursts_pg, distance_pg already computed in Step A above.

# Rate stats (already per-game or per-season-rate): delta directly
# Counting stats: delta on the per-game version computed above
rate_delta_cols  = ["toi", "ppg", "gpg", "apg", "oz_pct"]
count_delta_cols = ["shots_pg", "pp_points_pg", "bursts_pg", "distance_pg", "gamesPlayed"]
all_delta_cols   = rate_delta_cols + count_delta_cols

prev_vals = df.groupby("playerId")[all_delta_cols].shift(1)
for col in all_delta_cols:
    df[f"delta_{col}"] = df[col] - prev_vals[col]

# has_prior_season flag — 1 if player has a qualifying prior season in dataset,
# 0 if this is their first row. Lets the model learn to discount delta values
# that were filled from NaN (no prior data) vs genuine zero-change rows.
df["has_prior_season"] = (~prev_vals["ppg"].isna()).astype(int)

# How many games the player played last season (context for delta magnitude)
df["prev_season_gp"] = df.groupby("playerId")["gamesPlayed"].shift(1)

# =====================
# CAREER TRAJECTORY FEATURES
# =====================
# Best PPG a player achieved in any prior qualifying season
df["prev_career_high_ppg"] = df.groupby("playerId")["ppg"].transform(
    lambda x: x.shift(1).expanding().max()
)
# How far above/below their career best is the player this season?
# Positive = at or above career best (peak/breakout); Negative = in a slump or decline
df["career_high_gap"] = df["ppg"] - df["prev_career_high_ppg"]

# Age curve features
df["age_squared"] = df["age"] ** 2

def _career_stage(age):
    if age <= 23: return 0   # developing
    if age <= 31: return 1   # prime
    return 2                 # declining

df["career_stage"] = df["age"].apply(_career_stage)

# =====================
# REGRESSION SIGNALS
# =====================
# How far above/below career shooting % is this season?
# Positive = lucky year (regression risk down); Negative = unlucky (recovery upside)
df["shooting_pct_vs_career"] = df["shootingPercentage"] - df["career_shooting_pctg"]

# How far above/below the player's own career PPG rate is this season?
# Positive = performing above career average (regression risk); Negative = below (recovery upside)
career_ppg_rate = df["career_points"] / df["career_games_played"].replace(0, np.nan)
df["ppg_vs_career_rate"] = df["ppg"] - career_ppg_rate

# =====================
# DERIVE TARGET (next season's PPG per player)
# =====================
df["target_ppg_next"] = df.groupby("playerId")["ppg"].shift(-1)
df["target_points_next"] = df.groupby("playerId")["points"].shift(-1)
df["target_gp_next"] = df.groupby("playerId")["gamesPlayed"].shift(-1)

# Drop last-season rows (no future season to predict)
df = df.dropna(subset=["target_ppg_next"]).reset_index(drop=True)

# Filter out short seasons — removes injured/part-time players whose stats
# are too noisy to use as features or reliable targets
df = df[(df["gamesPlayed"] >= 40) & (df["target_gp_next"] >= 40)].reset_index(drop=True)

TARGET = "target_ppg_next"
print(f"Dataset shape after target derivation: {df.shape}")

# =====================
# TEMPORAL TRAIN / TEST SPLIT
# =====================
# Test on rows where season="20232024" — these predict 24-25 performance.
# Fall back to "20222023" (predicting 23-24) if 24-25 target data is unavailable.
TEST_SEASON = "20232024" if "20232024" in df["season"].values else "20222023"
train_mask = df["season"] < TEST_SEASON
test_mask  = df["season"] == TEST_SEASON

print(f"Temporal split — Train: {train_mask.sum()} rows (seasons before {TEST_SEASON}), "
      f"Test: {test_mask.sum()} rows (season {TEST_SEASON})")

# =====================
# DROP NON-FEATURE COLUMNS
# =====================
drop_cols = [
    "playerId",
    "fullName",
    "position",         # already one-hot encoded
    "shoots",           # already encoded
    "team",             # prevent leakage / too many categories
    "season",           # replaced by season_year

    "target_ppg_next",
    "target_points_next",
    "target_gp_next",

    "points",           # direct source of ppg target, always excluded
    "shotsPercentile",
    "season_year",
    "totalDistance",
    "next_team",        # helper column, environment captured in dest_team_* features
]

# When --no-base-stats is passed, drop persistence stats so the model is forced
# to rely on delta (YoY change) features and career/contextual signals instead.
# Delta versions of these features (delta_ppg, delta_apg, etc.) are still kept.
BASE_STATS = ["ppg", "apg", "gpg", "pp_points", "points", "assists", "goals", "pp_goals", "gamesPlayed", "shots"]
if DISABLE_BASE_STATS:
    drop_cols += BASE_STATS
    print(f">> --no-base-stats: excluding {BASE_STATS}")

X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [TARGET])
y = df[TARGET]

# Selective NaN filling:
#   - Career stats: fill 0 (rookies have no prior history, 0 is correct)
#   - Regression signals: fill 0 (no career baseline = no deviation)
#   - Delta features: leave as NaN — XGBoost handles NaN natively and learns
#     the best split direction for "no prior season" rows separately from
#     genuine zero-change rows. Filling with 0 conflates the two signals.
#   - Everything else: fill 0
fill_zero_candidates = (
    [c for c in X.columns if c.startswith("career_")]
    + ["shooting_pct_vs_career", "ppg_vs_career_rate", "prev_career_high_ppg"]
)
# Deduplicate while preserving order (career_high_gap already caught by career_ prefix)
seen = set()
fill_zero_cols = [c for c in fill_zero_candidates if c in X.columns and not (seen.add(c) or c in seen)]
delta_cols_in_X  = [c for c in X.columns if c.startswith("delta_")]
other_cols       = [c for c in X.columns if c not in fill_zero_cols + delta_cols_in_X]

X[fill_zero_cols] = X[fill_zero_cols].fillna(0)
X[other_cols]     = X[other_cols].fillna(0)
# delta_cols_in_X intentionally left as NaN for XGBoost to handle

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

# =====================
# MODEL
# =====================
if _USE_XGB:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2,
        min_child_weight=5,
        random_state=42,
    )
else:
    print("Using RandomForestRegressor instead of XGBoost")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_pred     = model.predict(X_test)
train_preds = model.predict(X_train)

rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
mae       = mean_absolute_error(y_test, y_pred)
bias      = float(np.mean(y_pred - y_test))
r2        = r2_score(y_test, y_pred)
train_r2  = r2_score(y_train, train_preds)
r2_gap    = train_r2 - r2

r2_gap = train_r2 - r2
if train_r2 >= 0.99 and r2 <= 0.80:
    fit_note = "Extreme overfitting"
elif train_r2 >= 0.95 and r2 <= 0.80:
    fit_note = "Overfitting"
elif train_r2 >= 0.83 and r2 >= 0.80 and r2_gap <= 0.05:
    fit_note = "Healthy model"
else:
    fit_note = "Inconclusive"

bias_dir = "overshooting" if bias > 0 else "undershooting"

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE     : {rmse:.3f}")
print(f"MAE      : {mae:.3f}  (avg error in PPG units)")
print(f"Bias     : {bias:+.3f}  (model is systematically {bias_dir})")
print(f"R²       : {r2:.3f}")
print(f"Train R² : {train_r2:.3f}")
print(f"Overfitting check: {fit_note} (train-test R² gap: {r2_gap:.3f})")

# =====================
# FEATURE IMPORTANCE
# =====================
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 15 Features:")
print(importances.head)

# =====================
# PREDICTIONS
# =====================
df["Predicted_PPG_Next"]    = model.predict(X)
df["Predicted_Points_Next"] = df["Predicted_PPG_Next"] * df["target_gp_next"].clip(lower=0)
df["Prediction_Error"]      = df["Predicted_PPG_Next"] - df[TARGET]
df["Abs_Error"]             = df["Prediction_Error"].abs()
df["Points_Error"]          = df["Predicted_Points_Next"] - df["target_points_next"]

cols_to_show = [
    "fullName",
    "season",
    "target_ppg_next",
    "Predicted_PPG_Next",
    "Prediction_Error",
    "Abs_Error",
    "target_points_next",
    "Predicted_Points_Next",
    "Points_Error",
    "target_gp_next",
]

# Best/worst filtered to test set only
test_df = df[test_mask].copy()

print(f"\nBest predictions — test set (season {TEST_SEASON}):")
print(test_df.sort_values("Abs_Error", ascending=True)[cols_to_show].head(10).to_string(index=False))

print(f"\nWorst predictions — test set (season {TEST_SEASON}):")
print(test_df.sort_values("Abs_Error", ascending=False)[cols_to_show].head(10).to_string(index=False))

# =====================
# ERROR BY POSITION
# =====================
test_df["pos_group"] = np.where(test_df["is_defenseman"] == 1, "Defenseman", "Forward")

pos_mae  = test_df.groupby("pos_group")["Abs_Error"].agg(MAE="mean", Count="count").round(3)
pos_bias = test_df.groupby("pos_group")["Prediction_Error"].mean().round(3).rename("Bias")

print("\nError by position (test set):")
print(pd.concat([pos_mae, pos_bias], axis=1).to_string())

# =====================
# ERROR BY PLAYER TIER
# =====================
# Split on median career PPG rate entering this season
career_ppg_rate_test = test_df["career_points"] / test_df["career_games_played"].clip(lower=1)
tier_median = career_ppg_rate_test.median()
test_df["tier"] = np.where(career_ppg_rate_test >= tier_median, "High-tier", "Low-tier")

tier_mae  = test_df.groupby("tier")["Abs_Error"].agg(MAE="mean", Count="count").round(3)
tier_bias = test_df.groupby("tier")["Prediction_Error"].mean().round(3).rename("Bias")

print(f"\nError by player tier (career PPG median split @ {tier_median:.3f}, test set):")
print(pd.concat([tier_mae, tier_bias], axis=1).to_string())

# =====================
# SPECIFIC PLAYERS
# =====================
PLAYERS_TO_SHOW = [
    "Elias Pettersson",
    "Nikita Kucherov",
]

for name in PLAYERS_TO_SHOW:
    rows = df[df["fullName"] == name][cols_to_show]
    print(f"\n{name} — Predicted vs Actual PPG:")
    if rows.empty:
        print(f"  {name} not found in dataset.")
    else:
        print(rows.to_string(index=False))

# =====================
# PLOT
# =====================
plt.figure(figsize=(10, 6))
importances.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
