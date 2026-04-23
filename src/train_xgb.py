import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

try:
    from .feature_engineering import engineer_features
except ImportError:
    from feature_engineering import engineer_features

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
# LOAD DATA & ENGINEER FEATURES
# =====================
df = pd.read_parquet("edge_data/nhl_merged_dataset.parquet")
print(f"Dataset shape: {df.shape}")

df = engineer_features(df, team_stats_path="edge_data/nhl_team_stats.parquet")

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
    "next_team",        # helper column, environment captured in dest_team_* features
    "season",           # replaced by season_year

    "target_ppg_next",
    "target_points_next",
    "target_gp_next",

    "points",           # direct source of ppg target, always excluded
    "shotsPercentile",
    "season_year",
    "totalDistance",
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
fill_zero_cols   = [c for c in dict.fromkeys(fill_zero_candidates) if c in X.columns]
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

print("\nAll Features by Importance:")
print(importances.to_string())

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
