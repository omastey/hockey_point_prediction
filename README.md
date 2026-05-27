# NHL Player Performance Prediction

End-to-end ML pipeline that pulls 16 seasons of NHL data from three public APIs, engineers 80+ player and team features, and trains two models:

- **PPG Regressor** (XGBoost) — predicts a skater's points-per-game next season
- **Breakout Classifier** (Logistic Regression w/ ElasticNet) — flags players poised for a career-best leap

Built to be reproducible from a fresh clone: data is committed, training runs in under a minute on a laptop.

---

## Headline Results

Evaluated on the **2024–25 NHL season** (468 player-seasons), trained on all prior seasons back to 2010–11.

### Breakout Classifier — Run 14 (current best)
| Metric | Value |
|---|---|
| ROC-AUC | **0.902** |
| Average Precision | **0.39** |
| Recall @ threshold 0.40 | 12 / 16 actual breakouts (75%) |
| Precision @ threshold 0.40 | 12 / 64 predictions (19%) |
| Brier Score | 0.065 |

A "breakout" is defined as a season where a player both exceeds their prior career-high PPG by at least 0.15 *and* clears a 0.45 PPG floor (~37 points over 82 games). 16 such seasons occurred in 2024–25; the model surfaced 12 of them in its 64 highest-probability predictions.

![Breakout classifier — top 15 coefficients and precision-recall curve](assets/breakout_classifier.png)

### PPG Regressor
| Metric | Value |
|---|---|
| RMSE | 0.137 |
| MAE | 0.107 (PPG units, ~8 points / 82 games) |
| R² | 0.796 |
| Train R² | 0.774 (no overfitting) |

Defensemen MAE: 0.098 · Forwards MAE: 0.111 — model handles both positions cleanly without separate heads.

![PPG regressor — top 15 feature importances](assets/regressor_feature_importance.png)

See [`LEADERBOARD_BREAKOUT.md`](LEADERBOARD_BREAKOUT.md) for the full run history (20+ experiments) and [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for tuning details.

---

## Sample Predictions

A few highlights from the 2024–25 test set ([full CSV](assets/sample_predictions_ppg.csv)):

| Player | Actual PPG | Predicted | Error |
|---|---|---|---|
| Nikita Kucherov | 1.71 | 1.37 | −0.34 |
| Auston Matthews | 0.88 | 1.21 | +0.33 |
| Elias Pettersson | 0.69 | 0.82 | +0.13 |
| Leo Carlsson | 0.96 | 0.63 | −0.33 |
| Lucas Raymond | 0.95 | 0.95 | +0.00 |

Top breakout-probability predictions for 2024–25 ([full CSV](assets/sample_predictions_breakout.csv)) — bold = actual breakout:

| Rank | Player | Breakout Prob | Actual |
|---|---|---|---|
| 1 | **Macklin Celebrini** | 0.87 | ✓ |
| 2 | **Cutter Gauthier** | 0.84 | ✓ |
| 3 | **Leo Carlsson** | 0.81 | ✓ |
| 4 | **Connor Bedard** | 0.77 | ✓ |
| 5 | **Will Smith** | 0.70 | ✓ |
| 6 | **Matt Boldy** | 0.70 | ✓ |

---

## Quickstart

```bash
git clone https://github.com/omastey/hockey_point_prediction.git
cd hockey_point_prediction

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the PPG regressor (~15s)
python -m src.train_xgb

# Train the breakout classifier (best config: Run 14)
python -m src.train_breakout --model logistic --no-base-stats
```

Data is already committed (`data/*.parquet`), so no API calls are needed to reproduce results.

Useful flags:

```bash
# Headless / CI — save plot to file, skip display
python -m src.train_breakout --no-plot --save-plot assets/breakout.png

# Run hyperparameter CV (time-series split) before training
python -m src.train_breakout --model logistic --tune

# Different test season
python -m src.train_breakout --test-season 20232024

# XGBoost variant with calibration
python -m src.train_breakout --model xgb --calibrate sigmoid
```

---

## How It Works

```
NHL APIs              feature engineering        models
─────────             ───────────────────        ──────
Landing API ─┐
             ├─→ merge on (playerId, season) ─→ engineer ─→ XGB Regressor (PPG)
Edge API ────┤        + (team, season)          80+ feats
             │                                          └─→ Logistic Classifier (breakout)
Stats API ───┘
```

1. **Fetch** — `get_all_player_profiles.py`, `get_all_edge_stats.py`, `get_team_stats.py` pull from the NHL's Landing, Edge, and Stats APIs respectively.
2. **Merge** — `merge_datasets.py` joins profile data (16 seasons, primary) with Edge tracking stats (2021–22+, left-joined) and team context. XGBoost handles the NaNs from the partial Edge coverage natively.
3. **Engineer** — `feature_engineering.py` computes 80+ features: per-game rates, percentile ranks, career trajectory deltas (Δ TOI, Δ PPG), interaction terms (`ppg × career_high_gap`), team context (PP opportunity, faceoff share), and roster depth (`players_ahead_on_team`, `player_pp_share`).
4. **Train** — strict temporal split (train on all seasons before test season). Logistic models use ElasticNet for feature selection; XGB models support sigmoid/isotonic calibration via `CalibratedClassifierCV`.
5. **Evaluate** — TP/FP/FN at threshold, ROC-AUC, Average Precision, Brier score, Precision@K, lift, and per-position / per-tier MAE breakdowns.

For the full feature catalogue with formulas and rationale, see [`FEATURES.md`](FEATURES.md). For decisions about removed features (and why), see [`src/archived_features.py`](src/archived_features.py).

---

## Tech Stack

- **Python 3.11** · pandas · NumPy
- **scikit-learn** — LogisticRegression (ElasticNet), CalibratedClassifierCV, time-series CV
- **XGBoost** 2.x — regressor + classifier
- **matplotlib** — feature-importance bars, precision-recall curves
- **Parquet (pyarrow)** — efficient on-disk dataset storage
- **NHL public APIs** — Landing, Edge tracking, Stats

---

## Project Structure

```
src/                          Pipeline code (run as modules: python -m src.X)
  constants.py                Season list, edge-season cutoff, filters
  feature_engineering.py      80+ engineered features (per-game, deltas, interactions)
  tuning.py                   Time-series CV hyperparameter search
  archived_features.py        Removed features with rationale
  data.py                     Merge utilities (profile-primary, edge left-joined)

  get_all_player_profiles.py  NHL Landing API → player season stats
  get_all_edge_stats.py       NHL Edge API → puck/skating tracking
  get_team_stats.py           NHL Stats API → team PP / faceoff context
  merge_datasets.py           Combines all three into final training set

  train_xgb.py                PPG regressor (XGBoost)
  train_breakout.py           Breakout classifier (logistic / XGB)

data/                         Committed parquet datasets (no API calls needed)
models/                       Saved model artifacts (.joblib)
assets/                       Plots + sample predictions used in this README
scripts/                      One-off debug utilities
```

---

## Documentation

| Doc | What's in it |
|---|---|
| [`FEATURES.md`](FEATURES.md) | All 80+ features grouped by source (Landing / Edge / Team / Derived), with formulas and rationale |
| [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) | Hyperparameter tuning runs, CV configurations, calibration experiments |
| [`LEADERBOARD_BREAKOUT.md`](LEADERBOARD_BREAKOUT.md) | Chronological log of 20+ breakout-classifier experiments — what was tried, what worked, what regressed |
| [`.claude/CLAUDE.md`](.claude/CLAUDE.md) | Developer workflow notes (data pipeline commands, conventions for adding features) |
