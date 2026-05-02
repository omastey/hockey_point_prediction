"""Classifier for predicting breakout seasons.

A breakout is defined as a season where a player:
  1. Exceeds their prior career-high PPG by an additive margin (default +0.15)
  2. AND improves their PPG by at least a threshold vs current season (default 0.15)
  3. AND reaches a minimum PPG floor (default 0.45, ~37 points over 82 games)

Usage:
  python -m src.train_breakout
  python -m src.train_breakout --model logistic
  python -m src.train_breakout --career-high-margin 0.12 --yoy-jump 0.20 --ppg-floor 0.50
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
    from .tuning import tune_logistic
except ImportError:
    from feature_engineering import engineer_features
    from tuning import tune_logistic

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
parser.add_argument("--career-high-margin", type=float, default=0.15,
                    help="PPG next must exceed career-high PPG by this amount (default: 0.15)")
parser.add_argument("--yoy-jump", type=float, default=0.15,
                    help="Minimum PPG increase from current to next season (default: 0.15)")
parser.add_argument("--ppg-floor", type=float, default=0.45,
                    help="Minimum PPG next season to qualify as breakout (default: 0.45)")
parser.add_argument("--min-gp", type=int, default=40,
                    help="Minimum games played filter (default: 40)")
parser.add_argument("--threshold", type=float, default=0.40,
                    help="Classification probability threshold (default: 0.40, lower = more recall, higher = more precision)")
parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "logistic"],
                    help="Model type: 'xgb' (XGBClassifier) or 'logistic' (LogisticRegression) (default: xgb)")
parser.add_argument("--no-base-stats", action="store_true",
                    help="Exclude raw counting/rate stats (included by default)")
parser.add_argument("--test-season", type=str, default="20242025",
                    help="Season to use as test set (default: 20242025)")
parser.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation (for headless/CI runs)")
parser.add_argument("--tune", action="store_true",
                    help="Run time-series CV to find best C, l1_ratio, class_weight before training")
parser.add_argument("--calibrate", type=str, default="none",
                    choices=["none", "sigmoid", "isotonic"],
                    help="Probability calibration method (XGB only). "
                         "'sigmoid'=Platt scaling (robust, 2 params); 'isotonic'=non-parametric (more flexible). "
                         "Calibrates on --calibration-season; trains on earlier seasons only.")
parser.add_argument("--calibration-season", type=str, default="20232024",
                    help="Season to use as calibration holdout (default: 20232024). "
                         "Must be earlier than --test-season.")
args = parser.parse_args()

CAREER_HIGH_MARGIN = args.career_high_margin
YOY_JUMP_THRESHOLD = args.yoy_jump
PPG_FLOOR = args.ppg_floor
MIN_GP = args.min_gp
THRESHOLD = args.threshold

print(f"Breakout definition: PPG_next > career-high PPG + {CAREER_HIGH_MARGIN} "
      f"AND PPG_next - PPG >= {YOY_JUMP_THRESHOLD} "
      f"AND PPG_next >= {PPG_FLOOR}")
print(f"Minimum games played: {MIN_GP}")
print(f"Classification threshold: {THRESHOLD}")

# =====================
# LOAD DATA & ENGINEER FEATURES
# =====================
df = pd.read_parquet("data/nhl_merged_dataset.parquet")
print(f"Dataset shape: {df.shape}")

df = engineer_features(df, team_stats_path="data/nhl_team_stats.parquet")

# =====================
# DERIVE TARGET
# =====================
df["target_ppg_next"] = df.groupby("playerId")["ppg"].shift(-1)
df["target_gp_next"] = df.groupby("playerId")["gamesPlayed"].shift(-1)

# Drop last-season rows (no future season to predict)
df = df.dropna(subset=["target_ppg_next"]).reset_index(drop=True)

# Filter short seasons
df = df[(df["gamesPlayed"] >= MIN_GP) & (df["target_gp_next"] >= MIN_GP)].reset_index(drop=True)

# Breakout target: all three conditions must be met
# For players with no prior career high (first season), use their current PPG as baseline
career_high_baseline = df["prev_career_high_ppg"].fillna(df["ppg"])
condition_career_high = df["target_ppg_next"] > career_high_baseline + CAREER_HIGH_MARGIN
condition_yoy_jump = (df["target_ppg_next"] - df["ppg"]) >= YOY_JUMP_THRESHOLD
condition_floor = df["target_ppg_next"] >= PPG_FLOOR

df["target_breakout"] = (condition_career_high & condition_yoy_jump & condition_floor).astype(int)

TARGET = "target_breakout"
n_breakout = df[TARGET].sum()
n_total = len(df)
print(f"Dataset shape after target derivation: {df.shape}")
print(f"Breakout rate: {n_breakout}/{n_total} ({n_breakout/n_total*100:.1f}%)")

# =====================
# TEMPORAL TRAIN / TEST SPLIT
# =====================
TEST_SEASON = args.test_season
train_mask = df["season"] < TEST_SEASON
test_mask  = df["season"] == TEST_SEASON

print(f"Temporal split — Train: {train_mask.sum()} rows, Test: {test_mask.sum()} rows")
print(f"  Train breakouts: {df.loc[train_mask, TARGET].sum()}, "
      f"Test breakouts: {df.loc[test_mask, TARGET].sum()}")

# =====================
# DROP NON-FEATURE COLUMNS
# =====================
drop_cols = [
    # Identity / target columns
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

# Archived features — removed to reduce noise for breakout prediction.
# See src/archived_features.py for rationale on each removal.
ARCHIVED_FEATURES = [
    # Next-season team stats (leakage-adjacent)
    "next_team_pp_pct", "next_team_pp_goals_pg", "next_team_pp_opps_pg",
    "next_team_faceoff_pct", "next_team_oz_faceoff_pct", "next_team_dz_faceoff_pct",
    # Current team aggregate stats (low importance)
    "team_pp_pct", "team_pp_goals_pg", "team_pp_opps_pg",
    "team_faceoff_pct", "team_oz_faceoff_pct", "team_dz_faceoff_pct",
    # Physical (consistently 0.0 importance)
    "height_in", "weight_lb", "over_6ft",
    # Low-signal categoricals
    "has_prior_season", "is_undrafted", "is_center", "is_winger",
    "shoots_left", "shoots_right",
    # Skating speed/distance (low importance)
    "speedMax", "speedMax_pct", "topShotSpeed", "topShotSpeed_pct",
    "delta_distance_pg", "distance_pg",
    # Redundant shooting metric (raw % kept, percentile dropped)
    "shootingPctgPercentile",
    # Redundant career counting stats (career_points + career_games_played kept)
    "career_goals", "career_assists",
    "career_pp_points", "career_pp_goals",
    # Other low-signal
    "defenseman_x_delta_ppg", "years_since_draft",
    # Subset of players_ahead_on_team — causes opposite-sign collinearity
    "pos_players_ahead",
    # L1 zeroed out, redundant with age + career_stage
    "age_squared",
]
drop_cols += ARCHIVED_FEATURES

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
X[delta_cols_in_X] = X[delta_cols_in_X].fillna(0)
X[other_cols]     = X[other_cols].fillna(0)

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

# =====================
# MODEL
# =====================
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"Class balance — negative: {n_neg}, positive: {n_pos}, scale_pos_weight: {scale_pos_weight:.1f}")

MODEL_TYPE = args.model

# Tuning or defaults
LOGISTIC_C = 0.5
LOGISTIC_L1_RATIO = 0.7
LOGISTIC_CLASS_WEIGHT = {0: 1, 1: 5}

if args.tune and MODEL_TYPE == "logistic":
    best = tune_logistic(X, y, df, train_mask, TARGET)
    LOGISTIC_C = best["C"]
    LOGISTIC_L1_RATIO = best["l1_ratio"]
    LOGISTIC_CLASS_WEIGHT = best["class_weight"]

if MODEL_TYPE == "logistic":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Logistic regression needs scaled features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=LOGISTIC_L1_RATIO,
        C=LOGISTIC_C,
        class_weight=LOGISTIC_CLASS_WEIGHT,
        max_iter=5000,
        random_state=42,
    )
    cw_str = f"{LOGISTIC_CLASS_WEIGHT[1]}:1"
    print(f"Model: LogisticRegression (ElasticNet, l1_ratio={LOGISTIC_L1_RATIO}, C={LOGISTIC_C}, class_weight {cw_str})")
    model.fit(X_train_scaled, y_train)

elif _USE_XGB:
    def _build_xgb(spw):
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2,
            min_child_weight=3,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=42,
        )

    if args.calibrate != "none":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator

        CALIB_SEASON = args.calibration_season
        if CALIB_SEASON >= TEST_SEASON:
            raise SystemExit(
                f"--calibration-season ({CALIB_SEASON}) must be earlier than "
                f"--test-season ({TEST_SEASON})."
            )

        fit_mask = train_mask & (df["season"] != CALIB_SEASON)
        calib_mask = train_mask & (df["season"] == CALIB_SEASON)
        X_fit, y_fit = X[fit_mask], y[fit_mask]
        X_calib, y_calib = X[calib_mask], y[calib_mask]

        if y_calib.sum() == 0:
            raise SystemExit(
                f"Calibration season {CALIB_SEASON} has no positive examples — "
                f"pick a different --calibration-season."
            )

        # Recompute scale_pos_weight on the smaller fit set
        spw_fit = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)

        base = _build_xgb(spw_fit)
        base.fit(X_fit, y_fit)
        xgb_for_importance = base

        # FrozenEstimator (sklearn 1.6+) replaces the old cv="prefit" pattern
        model = CalibratedClassifierCV(FrozenEstimator(base), method=args.calibrate)
        model.fit(X_calib, y_calib)
        print(f"Model: CalibratedClassifierCV(XGB, method={args.calibrate}) "
              f"— fit on {fit_mask.sum()} rows ({y_fit.sum()} pos), "
              f"calibrated on {calib_mask.sum()} rows ({y_calib.sum()} pos) from season {CALIB_SEASON}")
    else:
        model = _build_xgb(scale_pos_weight)
        xgb_for_importance = model
        print(f"Model: XGBClassifier (max_depth=4, lr=0.08, n_est=300)")
        model.fit(X_train, y_train)

else:
    print("XGBoost unavailable, using RandomForestClassifier")
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
if MODEL_TYPE == "logistic":
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
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
    print(f"  {'Thresh':>6}  {'Prec':>5}  {'Recall':>6}  {'F1':>5}  {'F0.5':>5}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    for t in [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        preds_t = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_test, preds_t)
        tp = cm_t[1, 1] if cm_t.shape[0] > 1 else 0
        fp = cm_t[0, 1]
        fn = cm_t[1, 0] if cm_t.shape[0] > 1 else y_test.sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        # F0.5 weights precision 2x more than recall
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
        print(f"  {t:>6.2f}  {prec:>5.2f}  {rec:>6.2f}  {f1:>5.2f}  {f05:>5.2f}  {tp:>3}  {fp:>3}  {fn:>3}")

    # Precision@k and Lift@k — measures top-of-list quality
    # P@k = fraction of top k predictions that are actual breakouts
    # Lift@k = P@k / base_rate (how much better than random)
    base_rate = y_test.sum() / len(y_test)
    sorted_idx = np.argsort(y_prob)[::-1]
    print(f"\nPrecision@k (ranked by predicted probability):")
    print(f"  Base rate: {base_rate:.3f} ({y_test.sum()}/{len(y_test)} actual breakouts)")
    print(f"  {'k':>3}  {'P@k':>5}  {'Lift':>5}  {'TP@k':>4}")
    for k in [5, 10, 15, 20, 30]:
        top_k_targets = y_test.iloc[sorted_idx[:k]]
        tp_at_k = int(top_k_targets.sum())
        p_at_k = tp_at_k / k
        lift_at_k = p_at_k / base_rate if base_rate > 0 else 0
        print(f"  {k:>3}  {p_at_k:>5.2f}  {lift_at_k:>5.2f}  {tp_at_k:>4}")
else:
    print("\nNo breakout examples in test set — cannot compute AUC metrics.")

# =====================
# FEATURE IMPORTANCE
# =====================
if MODEL_TYPE == "logistic":
    # Logistic regression: show coefficients (absolute value = importance, sign = direction)
    coefs = pd.Series(model.coef_[0], index=X.columns)
    importances = coefs.abs().sort_values(ascending=False)
    print("\nFeature Coefficients (logistic regression, sorted by |coef|):")
    print(coefs.reindex(importances.index).to_string(float_format="{:.4f}".format))
    n_nonzero = (coefs != 0).sum()
    print(f"\nL1 selected {n_nonzero}/{len(coefs)} features (zeroed out {len(coefs) - n_nonzero})")
else:
    # CalibratedClassifierCV wraps the base XGB; pull importances from the prefit base.
    importance_source = xgb_for_importance if "xgb_for_importance" in dir() else model
    importances = pd.Series(
        importance_source.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    print("\nAll Features by Importance:")
    print(importances.to_string())

# =====================
# PREDICTIONS — BREAKOUT CANDIDATES
# =====================
if MODEL_TYPE == "logistic":
    X_all_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    df["breakout_prob"] = model.predict_proba(X_all_scaled)[:, 1]
else:
    df["breakout_prob"] = model.predict_proba(X)[:, 1]
df["breakout_pred"] = (df["breakout_prob"] >= THRESHOLD).astype(int)

# Compute current and next season total points for display
df["current_points"] = (df["ppg"] * df["gamesPlayed"]).round(0).astype(int)
df["next_points"] = (df["target_ppg_next"] * df["target_gp_next"]).round(0).astype(int)
df["points_jump"] = df["next_points"] - df["current_points"]

test_df = df[test_mask].copy()

cols_to_show = [
    "fullName",
    "season",
    "ppg",
    "current_points",
    "target_ppg_next",
    "next_points",
    "points_jump",
    "breakout_prob",
    "breakout_pred",
    "target_breakout",
    "age",
    "career_stage",
]

# All players the model predicted as breakout
predicted_breakouts = test_df[test_df["breakout_pred"] == 1].copy()
print(f"\nAll predicted breakouts — test set (season {TEST_SEASON}, prob >= {THRESHOLD}):")
print(f"  {len(predicted_breakouts)} predicted, "
      f"{predicted_breakouts['target_breakout'].sum()} actual breakouts among them")
print(predicted_breakouts.sort_values("breakout_prob", ascending=False)[cols_to_show].to_string(index=False))

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
if not args.no_plot:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Feature importance / coefficients
    importances.head(15).sort_values().plot(kind="barh", ax=axes[0])
    title_suffix = "Coefficients" if MODEL_TYPE == "logistic" else "Importances"
    axes[0].set_title(f"Top 15 Feature {title_suffix} — Breakout Classifier")

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
