# HockeyModel

A minimal, no-fuss scaffold for predicting an NHL player's points with a simple XGBoost baseline. It keeps the structure straightforward so you can iterate quickly.

## Project Structure

- data/
  - raw/: place input CSVs here (e.g., `sample_players.csv`).
  - processed/: store cleaned/engineered CSVs here (optional).
- models/: saved model artifacts (e.g., `xgb_points_model.joblib`).
- src/: small Python package with modules used by your scripts.
  - data.py: CSV load/save helpers.
  - features.py: basic feature selection (`games_played`, `shots`, `time_on_ice`, `goals`, `assists`).
  - train_xgb.py: minimal training script using XGBoost regressor.
- requirements.txt: Python dependencies.

## Getting Started

1. Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put your player data CSVs in `data/raw`. A sample file is already provided at `data/raw/sample_players.csv`.

3. Train the baseline XGBoost model:

```bash
python -m src.train_xgb
```

This prints an RMSE and saves the model to `models/xgb_points_model.joblib`.

## How to Evolve This

- Replace `data/raw/sample_players.csv` with real data and adjust columns.
- Edit `src/features.py` to add richer features (per-60 stats, power-play time, linemates, team strength, recent form).
- Swap the model: try `RandomForestRegressor` or linear models to compare.
- Add evaluation beyond RMSE (MAE, R^2, cross-validation). 

## Notes

- Keep it simple: CSVs in `data/raw`, models in `models`, small reusable functions in `src`.
- If you later want a notebook, add `notebooks/` for exploration; the current setup stays lightweight.
