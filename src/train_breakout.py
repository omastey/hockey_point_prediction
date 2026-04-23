"""XGBClassifier for predicting breakout seasons.

A breakout is defined as a season where a player:
  1. Exceeds their prior career-high PPG by a configurable factor (default 1.2x)
  2. AND improves their PPG by at least a configurable threshold (default 0.15)

Usage:
  python -m src.train_breakout
  python -m src.train_breakout --career-high-factor 1.3 --yoy-jump 0.20 --min-gp 30
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
import matplotlib.pyplot as plt

try:
    from .feature_engineering import engineer_features
except ImportError:
    from feature_engineering import engineer_features

try:
    from xgboost import XGBClassifier
    _USE_XGB = True
except Exception as e:
    print("XGBoost unavailable, falling back to RandomForestClassifier:", e)
    from sklearn.ensemble import RandomForestClassifier
    _USE_XGB = False


# =====================
# CLI ARGS
# =====================
parser = argparse.ArgumentParser(description="Train breakout season classifier")
parser.add_argument("--career-high-factor", type=float, default=1.2,
                    help="PPG next must exceed career-high PPG * this factor (default: 1.2)")
parser.add_argument("--yoy-jump", type=float, default=0.15,
                    help="Minimum PPG increase from current to next season (default: 0.15)")
parser.add_argument("--min-gp", type=int, default=40,
                    help="Minimum games played filter (default: 40)")
parser.add_argument("--threshold", type=float, default=0.30,
                    help="Classification probability threshold (default: 0.30, lower = more recall)")
parser.add_argument("--no-base-stats", action="store_true",
                    help="Exclude raw counting/rate stats, keep only deltas and context")
args = parser.parse_args()

CAREER_HIGH_FACTOR = args.career_high_factor
YOY_JUMP_THRESHOLD = args.yoy_jump
MIN_GP = args.min_gp
THRESHOLD = args.threshold

print(f"Breakout definition: PPG_next > {CAREER_HIGH_FACTOR}x career-high PPG "
      f"AND PPG_next - PPG >= {YOY_JUMP_THRESHOLD}")
print(f"Minimum games played: {MIN_GP}")
print(f"Classification threshold: {THRESHOLD}")

# =====================
# LOAD DATA & ENGINEER FEATURES
# =====================
df = pd.read_parquet("edge_data/nhl_merged_dataset.parquet")
print(f"Dataset shape: {df.shape}")

df = engineer_features(df, team_stats_path="edge_data/nhl_team_stats.parquet")

# =====================
# DERIVE TARGET
# =====================
df["target_ppg_next"] = df.groupby("playerId")["ppg"].shift(-1)
df["target_gp_next"] = df.groupby("playerId")["gamesPlayed"].shift(-1)

# Drop last-season rows (no future season to predict)
df = df.dropna(subset=["target_ppg_next"]).reset_index(drop=True)

# Filter short seasons
df = df[(df["gamesPlayed"] >= MIN_GP) & (df["target_gp_next"] >= MIN_GP)].reset_index(drop=True)

# Breakout target: both conditions must be met
# For players with no prior career high (first season), use their current PPG as baseline
career_high_baseline = df["prev_career_high_ppg"].fillna(df["ppg"])
condition_career_high = df["target_ppg_next"] > CAREER_HIGH_FACTOR * career_high_baseline
condition_yoy_jump = (df["target_ppg_next"] - df["ppg"]) >= YOY_JUMP_THRESHOLD

df["target_breakout"] = (condition_career_high & condition_yoy_jump).astype(int)

TARGET = "target_breakout"
n_breakout = df[TARGET].sum()
n_total = len(df)
print(f"Dataset shape after target derivation: {df.shape}")
print(f"Breakout rate: {n_breakout}/{n_total} ({n_breakout/n_total*100:.1f}%)")

# =====================
# TEMPORAL TRAIN / TEST SPLIT
# =====================
TEST_SEASON = "20232024" if "20232024" in df["season"].values else "20222023"
train_mask = df["season"] < TEST_SEASON
test_mask  = df["season"] == TEST_SEASON

print(f"Temporal split — Train: {train_mask.sum()} rows, Test: {test_mask.sum()} rows")
print(f"  Train breakouts: {df.loc[train_mask, TARGET].sum()}, "
      f"Test breakouts: {df.loc[test_mask, TARGET].sum()}")

# =====================
# DROP NON-FEATURE COLUMNS
# =====================
drop_cols = [
    "playerId",
    "fullName",
    "position",
    "shoots",
    "team",
    "next_team",
    "season",

    "target_breakout",
    "target_ppg_next",
    "target_gp_next",

    "points",
    "shotsPercentile",
    "season_year",
    "totalDistance",
]

BASE_STATS = ["ppg", "apg", "gpg", "pp_points", "points", "assists", "goals", "pp_goals", "gamesPlayed", "shots"]
if args.no_base_stats:
    drop_cols += BASE_STATS
    print(f">> --no-base-stats: excluding {BASE_STATS}")

X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [TARGET])
y = df[TARGET]

# NaN filling (same strategy as regressor)
fill_zero_candidates = (
    [c for c in X.columns if c.startswith("career_")]
    + ["shooting_pct_vs_career", "ppg_vs_career_rate", "prev_career_high_ppg"]
)
fill_zero_cols   = [c for c in dict.fromkeys(fill_zero_candidates) if c in X.columns]
delta_cols_in_X  = [c for c in X.columns if c.startswith("delta_")]
other_cols       = [c for c in X.columns if c not in fill_zero_cols + delta_cols_in_X]

X[fill_zero_cols] = X[fill_zero_cols].fillna(0)
X[other_cols]     = X[other_cols].fillna(0)

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

# =====================
# MODEL
# =====================
# Handle class imbalance with scale_pos_weight
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"Class balance — negative: {n_neg}, positive: {n_pos}, scale_pos_weight: {scale_pos_weight:.1f}")

if _USE_XGB:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
else:
    print("Using RandomForestClassifier instead of XGBoost")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

print("\n===== BREAKOUT CLASSIFIER PERFORMANCE =====")
print(f"\nClassification Report (test set, season {TEST_SEASON}, threshold={THRESHOLD}):")
print(classification_report(y_test, y_pred, target_names=["No Breakout", "Breakout"], zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

if y_test.sum() > 0:
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    brier = np.mean((y_prob - y_test) ** 2)
    logloss = -np.mean(y_test * np.log(y_prob + 1e-15) + (1 - y_test) * np.log(1 - y_prob + 1e-15))
    print(f"\nROC-AUC:            {roc_auc:.3f}")
    print(f"Average Precision:  {avg_prec:.3f}")
    print(f"Brier Score:        {brier:.3f}  (lower = better calibrated, 0 = perfect)")
    print(f"Log Loss:           {logloss:.3f}  (lower = better)")

    # Threshold sweep — shows recall/precision tradeoff at different cutoffs
    print(f"\nThreshold Sweep (test set):")
    print(f"  {'Thresh':>6}  {'Prec':>5}  {'Recall':>6}  {'F1':>5}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        preds_t = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_test, preds_t)
        tp = cm_t[1, 1] if cm_t.shape[0] > 1 else 0
        fp = cm_t[0, 1]
        fn = cm_t[1, 0] if cm_t.shape[0] > 1 else y_test.sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {t:>6.2f}  {prec:>5.2f}  {rec:>6.2f}  {f1:>5.2f}  {tp:>3}  {fp:>3}  {fn:>3}")
else:
    print("\nNo breakout examples in test set — cannot compute AUC metrics.")

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
# PREDICTIONS — BREAKOUT CANDIDATES
# =====================
df["breakout_prob"] = model.predict_proba(X)[:, 1]
df["breakout_pred"] = (df["breakout_prob"] >= THRESHOLD).astype(int)

test_df = df[test_mask].copy()

cols_to_show = [
    "fullName",
    "season",
    "ppg",
    "target_ppg_next",
    "breakout_prob",
    "breakout_pred",
    "target_breakout",
    "age",
    "career_stage",
]

print(f"\nTop breakout candidates — test set (season {TEST_SEASON}):")
print(test_df.sort_values("breakout_prob", ascending=False)[cols_to_show].head(15).to_string(index=False))

# Show actual breakouts in test set
actual_breakouts = test_df[test_df["target_breakout"] == 1]
if not actual_breakouts.empty:
    print(f"\nActual breakouts in test set ({len(actual_breakouts)} players):")
    print(actual_breakouts.sort_values("breakout_prob", ascending=False)[cols_to_show].to_string(index=False))
else:
    print("\nNo actual breakouts in test set.")

# =====================
# PLOT
# =====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature importance
importances.head(15).sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Top 15 Feature Importances — Breakout Classifier")

# Precision-Recall curve
if y_test.sum() > 0:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve (AP={avg_prec:.3f})")
else:
    axes[1].text(0.5, 0.5, "No breakouts in test set", ha="center", va="center")
    axes[1].set_title("Precision-Recall Curve")

plt.tight_layout()
plt.show()
