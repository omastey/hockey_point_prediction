# XGBoost Hyperparameter Reference

Configuration choices for the two XGBoost models in this repo and the reasoning behind each one. Update this file whenever a hyperparameter is changed in `train_breakout.py` or `train_xgb.py`.

---

## PPG Regressor — `src/train_xgb.py`

Predicts next-season points-per-game. Continuous target, ~6,000 training rows.

```python
XGBRegressor(
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
```

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Many trees compensate for the very low learning rate. Together with `lr=0.03`, gives ~15 effective passes — slow, smooth fitting. |
| `learning_rate` | 0.03 | Low LR is appropriate because PPG is largely persistence-driven (current PPG is the strongest predictor). Slow learning lets the model build up a smooth additive correction on top of that baseline rather than overshooting. |
| `max_depth` | 2 | **Very shallow.** PPG prediction is dominated by additive signals (current PPG + age + role). Depth-2 trees ≈ stumps with one interaction. Deeper trees overfit interactions that don't generalize across seasons (sample size per leaf becomes tiny). |
| `subsample` | 0.8 | Use 80% of rows per tree. Standard stochastic GBM regularization — lowers variance without giving up much signal. |
| `colsample_bytree` | 0.7 | Use 70% of features per tree. With 80+ correlated features, this prevents the same dominant features (ppg, career_ppg) from being selected in every tree. |
| `reg_alpha` | 0.5 | Mild L1 regularization. Encourages sparsity at the leaf weight level. |
| `reg_lambda` | 2.0 | L2 regularization on leaf weights. Shrinks predictions toward zero, smoother decision surface. |
| `min_child_weight` | 5 | Each leaf needs at least 5 samples of weight. Prevents the model from carving out fragile rules that rely on a single rare player-season. |
| `random_state` | 42 | Reproducibility. |

### Performance baseline (after goalie filter)
- RMSE: 0.137 · MAE: 0.107 · R²: 0.795 · Train R²: 0.774

---

## Breakout Classifier — `src/train_breakout.py`

Predicts whether a player will have a breakout next season. Binary target, ~6% positive rate.

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2,
    min_child_weight=3,
    scale_pos_weight=auto,   # n_neg / n_pos ≈ 14.7
    eval_metric="aucpr",
    random_state=42,
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 300 | Fewer than the regressor because LR is higher (0.08 × 300 ≈ 0.03 × 800 in effective passes). 300 was found empirically — going to 500 didn't improve AP and increased overfitting on the 379 training positives. |
| `learning_rate` | 0.08 | Higher than the regressor because breakouts are rare and signals are noisier — a slightly faster learner finds the discriminative patterns without burying them in over-shrinkage. |
| `max_depth` | 4 | Deeper than the regressor because breakout prediction depends on **interactions**: `young × Δtoi`, `defenseman × pp_points_pg`, `low ppg × high career_high_gap`. Depth-4 captures these. Depth ≥6 overfit on training breakouts (we have only 379 positive examples — too few to support more complex trees). |
| `subsample` | 0.8 | Same logic as regressor. |
| `colsample_bytree` | 0.7 | Same logic as regressor. With many correlated features (delta_*, career_*, ppg variants), random feature subsetting forces diversity across trees. |
| `reg_alpha` | 0.5 | L1 leaf-weight regularization. Helps suppress noisy splits when class imbalance is amplified by `scale_pos_weight`. |
| `reg_lambda` | 2.0 | L2 leaf-weight regularization. Smooths probability estimates that would otherwise be pushed to extremes by `scale_pos_weight`. |
| `min_child_weight` | 3 | Lower than the regressor (5) because positives are rare — requiring 5 samples per leaf would prevent the model from learning rare-pattern signals. 3 is the floor where leaves still represent real subgroups. |
| `scale_pos_weight` | `n_neg / n_pos` (~14.7) | Re-weights positive class to match negative-class loss contribution. Without this, gradient boosting would barely move the loss for positives and the model would degenerate to "always negative." Auto-set from training data each run. **Note:** this value isn't grid-searched — could be a tuning lever (try {4, 8, 14}). |
| `eval_metric` | "aucpr" | Area under the **precision-recall** curve, not ROC. PR-AUC is the right metric for imbalanced binary classification because ROC-AUC overstates performance when negatives dominate (the FPR denominator is huge). |
| `random_state` | 42 | Reproducibility. |

### Decision threshold
- Default: **0.40** (raised from 0.30 in April 2026 to prioritize precision over recall — see `LEADERBOARD.md` and the precision@k discussion).
- Configurable via `--threshold`. Threshold sweep is printed every run from 0.10 to 0.70.
- **When `--calibrate sigmoid` is used, the threshold scale shifts** — calibrated probabilities are typically much smaller (e.g. uncalibrated 0.40 ≈ calibrated 0.10). Reach for the threshold sweep + Precision@k output to pick a threshold on the calibrated scale.

### Probability calibration (`--calibrate {sigmoid|isotonic}`)
Wraps the trained XGB in `sklearn.calibration.CalibratedClassifierCV` (with `FrozenEstimator` since sklearn 1.6+) so that output probabilities reflect actual frequencies.

**Why it's needed:** XGBoost with `scale_pos_weight=14.7` produces miscalibrated probabilities — Run 17's top-5 picks were all False Positives at 0.85+ confidence. Calibration learns a post-hoc map from raw scores → calibrated probabilities using a held-out season.

**Setup:**
- `--calibration-season` (default `20232024`) is held out from training and used as the calibration set.
- Base XGB is fit on all training seasons except the calibration season.
- `CalibratedClassifierCV` then learns the calibration map on the held-out season.
- Test set is unchanged (`--test-season`, default `20242025`).

**Method choice:**
| Method | Pros | Cons | When to use |
|---|---|---|---|
| `sigmoid` (Platt) | 2 parameters, robust with small data, smooth | Limited shape — only fixes monotonic distortion via logistic curve | Default — works well with our ~22 calibration positives |
| `isotonic` | Non-parametric, can correct any monotonic distortion | Overfits with few calibration points | Use only when calibration set has ≥100 positives |

**Empirical results (Run 19 sigmoid vs Run 17 uncalibrated baseline):**
- Brier 0.044 → **0.028** (much better calibrated)
- AP 0.274 → **0.298** (slight improvement)
- P@10: 30% → **40%** (the main win — top-of-list quality)
- ROC-AUC: 0.892 → 0.883 (slight regression — calibration trades a tiny bit of ranking quality for far better probability quality)

**Tradeoff:** training set shrinks by one season (5934 → 5462 rows; 379 → 357 positives). Acceptable for the calibration gains.

### Performance baseline (Run 17 — XGB, after goalie filter + base stats restored, threshold 0.30)
- TP=10 · FP=42 · FN=6 · ROC-AUC: 0.892 · Avg Precision: 0.274 · Brier: 0.044
- Precision@10: 30% · Lift@10: ~9x

---

## Logistic Regression Classifier — `src/train_breakout.py --model logistic`

Alternative classifier with much higher recall but lower precision-at-top. Used as a complement / second opinion to XGB.

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.7,
    C=0.5,
    class_weight={0: 1, 1: 5},
    max_iter=5000,
    random_state=42,
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `penalty` | "elasticnet" | Combines L1 (sparsity) and L2 (shrinkage). Pure L1 (Run 16) was sparser but lost recall; pure L2 wouldn't zero out redundant features. ElasticNet is the middle ground. |
| `l1_ratio` | 0.7 | 70% L1, 30% L2. Confirmed by CV (`tuning.py`) — top configurations cluster around 0.5-0.85. |
| `C` | 0.5 | Inverse regularization strength. Smaller C = stronger regularization. CV-tuned: 0.2-0.5 win consistently. |
| `class_weight` | `{0: 1, 1: 5}` | Re-weights breakouts as 5x more important. CV explored 3:1, 5:1, 8:1, 10:1 — 5:1 is the most balanced (3:1 underweights, 8:1+ explodes FPs). |
| `max_iter` | 5000 | SAGA solver needs more iterations for elasticnet convergence. |
| `solver` | "saga" | The only sklearn solver that supports elasticnet. Stochastic Average Gradient with L1+L2. |

### Performance baseline (Run 18 — logistic, after goalie filter + base stats restored, threshold 0.30)
- TP=14 · FP=85 · FN=2 · ROC-AUC: 0.894 · Avg Precision: 0.343 · Brier: 0.067
- Precision@10: ~10% · catches more breakouts overall but worse top-of-list quality than XGB

---

## Tuning Path

Hyperparameters can be re-tuned via time-series cross-validation:

```bash
python -m src.train_breakout --model logistic --tune
```

This calls `tune_logistic()` in `src/tuning.py`, which does expanding-window CV across the last 4 training seasons and grid-searches over `(C, l1_ratio, class_weight)`. There is **no XGB tuning path yet** — adding one is on the roadmap (see LEADERBOARD comments).

---

## When to Update This File

- Any change to `XGBRegressor(...)` or `XGBClassifier(...)` constructor args
- Default `--threshold` change in `train_breakout.py`
- Class-weight or `scale_pos_weight` strategy change
- New tuning path added (e.g., XGB CV)
- New ranking objective (e.g., switch to `rank:pairwise`)
