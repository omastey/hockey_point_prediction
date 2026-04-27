# HockeyModel

NHL player performance prediction system with two models:
- **PPG Regressor** (XGBoost) — predicts next-season points per game
- **Breakout Classifier** (Logistic Regression / XGBoost) — predicts breakout seasons

## Project Structure

```
src/                             # Main pipeline code
  constants.py                   # SEASONS, EDGE_SEASONS, GAME_TYPE, MIN_GAMES_FILTER
  feature_engineering.py         # Shared feature computation (80+ features)
  tuning.py                      # Hyperparameter CV search (called by train_breakout)
  archived_features.py           # Removed features with rationale
  data.py                        # Merge utilities (profile-primary, edge left-joined)

  # Data fetching
  get_all_player_profiles.py     # NHL Landing API → player stats (all 16 seasons)
  get_all_edge_stats.py          # NHL Edge API → tracking stats (EDGE_SEASONS only)
  get_team_stats.py              # NHL Stats API → team PP/faceoff stats
  merge_datasets.py              # Combines profile + edge + team → merged dataset

  # Model training
  train_breakout.py              # Breakout classifier (logistic or xgb)
  train_xgb.py                   # PPG regressor

data/                            # All datasets (parquet)
  nhl_merged_dataset.parquet     # Final training data
  nhl_full_stats.parquet         # Player profiles (primary dataset)
  nhl_edge_model_dataset.parquet # Edge tracking stats (2021-22+)
  nhl_team_stats.parquet         # Team context stats
  examples/                      # Sample player JSONs, CSVs
  archived/                      # Old dataset versions

scripts/                         # One-off / debug tools
models/                          # Trained model artifacts (.joblib)
```

## How to Run

Always activate the venv first:
```bash
source .venv/bin/activate
```

### Data Pipeline (re-fetch from NHL APIs)
```bash
python -m src.get_all_edge_stats          # Edge tracking (EDGE_SEASONS)
python -m src.get_all_player_profiles     # Player profiles (all 16 seasons)
python -m src.get_team_stats              # Team stats (all 16 seasons)
python -m src.merge_datasets              # Merge into final dataset
```

### Training
```bash
# Breakout classifier (default: logistic, test season 2024-25)
python -m src.train_breakout --model logistic
python -m src.train_breakout --model xgb
python -m src.train_breakout --model logistic --tune     # With CV hyperparameter search
python -m src.train_breakout --model logistic --no-plot   # Skip plot (for headless/CI)
python -m src.train_breakout --test-season 20232024       # Different test season

# PPG regressor
python -m src.train_xgb
python -m src.train_xgb --no-plot

# Both models
./run_all.sh
```

### Key CLI args for train_breakout
- `--model logistic|xgb` — model type (default: xgb)
- `--test-season XXXXXXXX` — test season (default: 20242025)
- `--career-high-margin 0.15` — PPG above career high to qualify as breakout
- `--yoy-jump 0.15` — min PPG increase year-over-year
- `--ppg-floor 0.45` — min PPG next season
- `--min-gp 40` — min games played filter
- `--threshold 0.30` — classification probability threshold
- `--base-stats` — include raw counting stats (excluded by default)
- `--tune` — run time-series CV before training
- `--no-plot` — skip plot generation

## Workflow Rules

### When adding or removing features:
1. Update `src/feature_engineering.py` with the computation
2. Update `FEATURES.md` with the feature name, formula, and rationale
3. If removing: add to `ARCHIVED_FEATURES` in `src/train_breakout.py` with a comment explaining why, and document in `src/archived_features.py`

### When running a new model experiment:
1. Run the model and note the results
2. Add a row to `LEADERBOARD.md` with: run #, model type, key changes, TP/FP/FN, precision, recall, ROC-AUC, avg precision, brier score, and notes

### Data architecture:
- **Profile dataset is primary** (all 16 seasons, 2010-2026)
- **Edge tracking data is left-joined** (only available 2021-22+, NaN for older seasons)
- XGBoost handles NaN natively; logistic fills NaN with 0
- All parquet files go in `data/`, never `src/`
