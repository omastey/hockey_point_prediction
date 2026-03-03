import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
    """
    Lookup predicted vs actual PPG for player(s) across all their seasons.
    """
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

# =====================
# DELTA FEATURES (year-over-year changes)
# =====================
delta_cols = [
    "toi", "ppg", "gpg", "apg",
    "shots", "pp_points", "gamesPlayed",
    "oz_pct", "totalDistance", "burstsOver20",
]
prev_vals = df.groupby("playerId")[delta_cols].shift(1)
for col in delta_cols:
    df[f"delta_{col}"] = df[col] - prev_vals[col]

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
# DERIVE TARGET (Option A: next season's PPG per player)
# =====================

df["target_ppg_next"] = df.groupby("playerId")["ppg"].shift(-1)
df["target_points_next"] = df.groupby("playerId")["points"].shift(-1)
df["target_gp_next"] = df.groupby("playerId")["gamesPlayed"].shift(-1)

# Drop last-season rows (no future season to predict)
df = df.dropna(subset=["target_ppg_next"]).reset_index(drop=True)

TARGET = "target_ppg_next"

print(f"Dataset shape after target derivation: {df.shape}")

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

    # ADDING THIS FOR TESTING - MAY RE-ADD LATER
    "points",
    "assists",
    "goals",

    # "ppg",
    # "apg",
    # "gpg",
    # "pp_points",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [TARGET])
y = df[TARGET]

# Replace any remaining nulls
X = X.fillna(0)

# =====================
# TRAIN / TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# MODEL
# =====================
if _USE_XGB:
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,

        reg_alpha=0.5,      # L1 regularization
        reg_lambda=2,       # L2 regularization

        min_child_weight=3,

        random_state=42,
    )
else:
    print(80, "Using RandomForestRegressor instead of XGBoost")
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
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

train_preds = model.predict(X_train)

train_r2 = r2_score(y_train, train_preds)
r2 = r2_score(y_test, y_pred)

print("Train R²:", train_r2)
print("Test R²:", r2)

# =====================
# OVERFITTING CHECK (heuristic)
# =====================
r2_gap = train_r2 - r2
if train_r2 >= 0.99 and r2 <= 0.80:
    fit_note = "Extreme overfitting"
elif train_r2 >= 0.95 and r2 <= 0.80:
    fit_note = "Overfitting"
elif train_r2 >= 0.83 and r2 >= 0.80 and r2_gap <= 0.05:
    fit_note = "Healthy model"
else:
    fit_note = "Inconclusive"

print(f"Overfitting check: {fit_note} (train-test R² gap: {r2_gap:.3f})")

print("\n===== MODEL PERFORMANCE =====")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# =====================
# FEATURE IMPORTANCE
# =====================
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 15 Features:")
print(importances.head(15))


df["Predicted_PPG_Next"] = model.predict(X)
df["Predicted_Points_Next"] = df["Predicted_PPG_Next"] * df["target_gp_next"].clip(lower=0)
df["Prediction_Error"] = df["Predicted_PPG_Next"] - df[TARGET]
df["Abs_Error"] = df["Prediction_Error"].abs()
df["Points_Error"] = df["Predicted_Points_Next"] - df["target_points_next"]

# Show best and worst predictions across the dataset, sorted by absolute error
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

print("\nBest predictions (smallest absolute error):")
best = df.sort_values("Abs_Error", ascending=True)[cols_to_show].head(10)
print(best.to_string(index=False))

print("\nWorst predictions (largest absolute error):")
worst = df.sort_values("Abs_Error", ascending=False)[cols_to_show].head(10)
print(worst.to_string(index=False))

# Elias Pettersson predicted vs actual points
print("\nElias Pettersson — Predicted vs Actual PPG:")
ep_row = df[df["fullName"] == "Elias Pettersson"][cols_to_show]
if ep_row.empty:
    print("Elias Pettersson not found in dataset.")
else:
    print(ep_row.to_string(index=False))


# Optional plot
plt.figure(figsize=(10, 6))
importances.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
