# PPG Regressor — Run Leaderboard

Test set: season noted per run | Target: `target_ppg_next` (next season's points-per-game)

## TL;DR — Current Best

| Best by | Run | Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|---|---|
| **All metrics** | **#1** | XGBRegressor (baseline) | **0.137** | **0.107** | **0.796** | First tracked baseline. No tuning yet. |

---

## Runs

| # | Date | Model | Key Config | RMSE | MAE | R² | Train R² | Bias | Forward MAE | D MAE | Notes |
|---|------|-------|------------|------|-----|-----|----------|------|-------------|-------|-------|
| 1 | 2026-05-27 | XGBRegressor | `n_estimators=500`, `lr=0.03`, `max_depth=2`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.5`, `reg_lambda=2`, `min_child_weight=5`. Goalies filtered upstream. Test=2024-25, train=all prior seasons (5934 rows). | 0.137 | 0.107 | 0.796 | 0.774 | -0.012 | 0.111 | 0.098 | Baseline. Slight undershoot bias (~0.012 PPG). No overfitting (train-test R² gap -0.022). High-tier players harder (MAE 0.125) than low-tier (MAE 0.089). Top features: `ppg_league_percentile`, `players_ahead_on_team`, `ppg`. |

---

## Prior configuration history (metrics not recorded)

Reconstructed from git history. These versions of the regressor existed but their RMSE/MAE/R² were never logged — they are listed here only for context on how the model evolved.

| Commit | Date | Key change | Why it matters |
|---|---|---|---|
| `6396d95` | 2026-02-18 | Initial scaffold. Target = `target_ppg_2425` (single hard-coded season). Random 80/20 train/test split. | Pre-temporal split — leaks information across seasons. |
| `fffd403` | 2026-03-03 | Switched to `target_ppg_next` (per-player next-season shift). Added multiple training seasons. | Made the task well-defined and multi-season. |
| `ddc6cf4` | 2026-03-15 | Added min-games filter (`gp ≥ 40` for current + next season). Switched to strict temporal split (test = 2023-24). | Eliminated noise from cup-of-coffee players and prevented temporal leakage. |
| `7dda85b` | 2026-04-27 | Bumped test season to 2024-25 once those games existed. | Latest holdout. |
| `59de410` | 2026-05-08 | Goalies filtered upstream during merge. | Stops the model from wasting capacity on rows it should never score. |

---

## How to add a run

After training, capture these numbers from the script's output:

```
RMSE     : 0.137
MAE      : 0.107
Bias     : -0.012
R²       : 0.796
Train R² : 0.774
Error by position (test set):
  Defenseman  MAE 0.098
  Forward     MAE 0.111
```

Add one row with: run #, date, model, key config diff vs prior run, the headline metrics, and a one-line note on what was tried / what moved.

Regenerate the embedded README plot with:

```bash
python -m src.train_xgb --no-plot --save-plot assets/regressor_feature_importance.png
```
