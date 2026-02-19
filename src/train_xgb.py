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
                            actual_col="target_points_2425",
                            pred_col="Predicted_Points_24_25"):
    """
    Lookup predicted vs actual points for player(s).
    No model invocation required.
    """

    if isinstance(player_names, str):
        player_names = [player_names]

    result = df[df[player_col].isin(player_names)][
        [player_col, actual_col, pred_col, "Prediction_Error"]
    ].sort_values(pred_col, ascending=False)

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

if "toi_2324" in df.columns:
    df["toi_2324"] = df["toi_2324"].apply(toi_to_seconds)

# =====================
# DEFINE TARGET (PPG instead of total points)
# =====================
TARGET = "target_ppg_2425"

# =====================
# DROP NON-FEATURE COLUMNS
# =====================
drop_cols = [
    "playerId",
    "fullName",
    "position",     # already one-hot encoded
    "shoots",       # already encoded
    "team_2425",     # prevent leakage
    "team_2324",

    "target_points_2425",  # keep out of features
    "target_ppg_2425",
    "target_goals_2425",
    "target_assists_2425",
    "gp_2425",

    # ADDING THIS FOR TESTING - MAY RE-ADD LATER
    "points_2324",
    "assists_2324",
    "goals_2324",

    # "ppg_2324",
    # "apg_2324",
    # "gpg_2324",
    # "pp_points_2324",
]

X = df.drop(columns=drop_cols + [TARGET])
# X = X.select_dtypes(include=["number", "bool"])
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
# model = XGBRegressor(
#     n_estimators=400,
#     max_depth=4,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

if _USE_XGB:
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,

        reg_alpha=0.5,      # L1 regularization
        reg_lambda=2,     # L2 regularization

        min_child_weight=3,

        # objective="reg:squarederror",
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





df["Predicted_PPG_24_25"] = model.predict(X)
df["Predicted_Points_24_25"] = df["Predicted_PPG_24_25"] * df["gp_2425"].clip(lower=0)
df["Prediction_Error"] = df["Predicted_PPG_24_25"] - df[TARGET]
df["Abs_Error"] = df["Prediction_Error"].abs()
df["Points_Error"] = df["Predicted_Points_24_25"] - df["target_points_2425"]

# Show best and worst predictions across the dataset, sorted by absolute error
cols_to_show = [
    "fullName",
    "target_ppg_2425",
    "Predicted_PPG_24_25",
    "Prediction_Error",
    "Abs_Error",
    "target_points_2425",
    "Predicted_Points_24_25",
    "Points_Error",
    "gp_2425",
]

print("\nBest predictions (smallest absolute error):")
best = df.sort_values("Abs_Error", ascending=True)[cols_to_show].head(10)
print(best.to_string(index=False))

print("\nWorst predictions (largest absolute error):")
worst = df.sort_values("Abs_Error", ascending=False)[cols_to_show].head(10)
print(worst.to_string(index=False))

# Elias Pettersson predicted vs actual points
print("\nElias Pettersson — Predicted vs Actual Points:")
ep_cols = [
    "fullName",
    "target_ppg_2425",
    "Predicted_PPG_24_25",
    "Prediction_Error",
    "Abs_Error",
    "target_points_2425",
    "Predicted_Points_24_25",
    "Points_Error",
    "gp_2425",
]
ep_row = df[df["fullName"] == "Elias Pettersson"][ep_cols]
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


