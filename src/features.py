from typing import Tuple
import pandas as pd


def select_basic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return simple features and target from a raw player DataFrame."""
    features = ["gamesPlayed", "shots", "toi", "goals", "assists"]
    X = df[features].copy()
    y = df["ppg"].copy()
    return X, y
