"""Hyperparameter tuning for breakout classifier via time-series CV.

Exports `tune_logistic()` for use by train_breakout.py.
Can also be run standalone: python -m src.tuning
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


# =====================
# SEARCH GRID
# =====================
C_VALUES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
L1_RATIOS = [0.5, 0.7, 0.85, 1.0]
CLASS_WEIGHTS = [
    {0: 1, 1: 3},
    {0: 1, 1: 5},
    {0: 1, 1: 8},
    {0: 1, 1: 10},
]


def tune_logistic(X, y, df, train_mask, target_col, n_folds=4):
    """Run time-series expanding-window CV to find best hyperparameters.

    Args:
        X: Feature DataFrame (already cleaned/filled)
        y: Target Series
        df: Full DataFrame (needed for season column)
        train_mask: Boolean mask for training rows
        target_col: Name of target column in df
        n_folds: Number of CV folds (last N training seasons)

    Returns:
        dict with keys: C, l1_ratio, class_weight, mean_AP, std_AP
    """
    print("\n===== HYPERPARAMETER TUNING (time-series CV) =====")

    train_seasons = sorted(df.loc[train_mask, "season"].unique())
    fold_seasons = train_seasons[-n_folds:]
    print(f"CV folds: validating on {fold_seasons}")

    total = len(C_VALUES) * len(L1_RATIOS) * len(CLASS_WEIGHTS)
    print(f"Searching {total} hyperparameter combinations...")

    results = []
    i = 0

    for c_val in C_VALUES:
        for l1_r in L1_RATIOS:
            for cw in CLASS_WEIGHTS:
                fold_aps = []
                for fold_season in fold_seasons:
                    fold_train_mask = (train_mask) & (df["season"] < fold_season)
                    fold_val_mask = (train_mask) & (df["season"] == fold_season)

                    if df.loc[fold_val_mask, target_col].sum() == 0:
                        continue

                    X_ft = X.loc[fold_train_mask.values]
                    y_ft = y.loc[fold_train_mask.values]
                    X_fv = X.loc[fold_val_mask.values]
                    y_fv = y.loc[fold_val_mask.values]

                    sc = StandardScaler()
                    X_ft_s = sc.fit_transform(X_ft)
                    X_fv_s = sc.transform(X_fv)

                    lr = LogisticRegression(
                        penalty="elasticnet", solver="saga",
                        l1_ratio=l1_r, C=c_val, class_weight=cw,
                        max_iter=5000, random_state=42,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lr.fit(X_ft_s, y_ft)

                    probs = lr.predict_proba(X_fv_s)[:, 1]
                    fold_aps.append(average_precision_score(y_fv, probs))

                if fold_aps:
                    mean_ap = np.mean(fold_aps)
                    std_ap = np.std(fold_aps)
                    cw_label = f"{cw[1]}:1"
                    results.append({
                        "C": c_val, "l1_ratio": l1_r,
                        "class_weight": cw, "class_weight_label": cw_label,
                        "mean_AP": mean_ap, "std_AP": std_ap,
                        "folds": len(fold_aps),
                    })

                i += 1
                if i % 28 == 0:
                    print(f"  Searched {i}/{total} combinations...")

    results.sort(key=lambda x: -x["mean_AP"])

    print(f"\nTop 10 configurations by mean CV Average Precision:")
    print(f"  {'C':>5}  {'l1_ratio':>8}  {'weight':>6}  {'mean_AP':>7}  {'std_AP':>6}  {'folds':>5}")
    for r in results[:10]:
        print(f"  {r['C']:>5.1f}  {r['l1_ratio']:>8.2f}  {r['class_weight_label']:>6}"
              f"  {r['mean_AP']:>7.3f}  {r['std_AP']:>6.3f}  {r['folds']:>5}")

    best = results[0]
    print(f"\n>>> Best: C={best['C']}, l1_ratio={best['l1_ratio']}, "
          f"class_weight={best['class_weight_label']}, "
          f"mean_AP={best['mean_AP']:.4f} +/- {best['std_AP']:.4f}")
    print("=" * 55)

    return {
        "C": best["C"],
        "l1_ratio": best["l1_ratio"],
        "class_weight": best["class_weight"],
        "mean_AP": best["mean_AP"],
        "std_AP": best["std_AP"],
    }
