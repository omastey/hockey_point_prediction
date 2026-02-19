from typing import Tuple
import pandas as pd


def select_basic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return simple features and target from a raw player DataFrame."""
    features = ["games_played", "shots", "time_on_ice", "goals", "assists"]
    X = df[features].copy()
    y = df["points"].copy()
    return X, y
