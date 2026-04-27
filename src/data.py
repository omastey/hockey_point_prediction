from pathlib import Path
import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load a raw CSV file into a DataFrame."""
    return pd.read_csv(path)


def save_processed_csv(df: pd.DataFrame, path: str) -> None:
    """Save a processed DataFrame to CSV, ensuring parent dirs exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# Edge tracking columns that only exist from 2021-22 onward
EDGE_ONLY_COLS = [
    "topShotSpeed", "topShotSpeed_pct",
    "speedMax", "speedMax_pct",
    "burstsOver20", "totalDistance",
    "oz_pct", "dz_pct",
    "shotsPercentile", "shootingPctgPercentile",
]


def merge_parquet_datasets(
    profile_path: str,
    edge_path: str,
    output_path: str,
    join_on: list = None,
) -> pd.DataFrame:
    """Merge profile (primary) and EDGE datasets.

    Profile is the primary dataset (all 16 seasons).
    Edge tracking columns are left-joined (only available 2021-22+).
    Pre-2021 rows will have NaN for edge columns.
    """
    if join_on is None:
        join_on = ["playerId", "season"]

    profile_df = pd.read_parquet(profile_path)
    edge_df = pd.read_parquet(edge_path)

    # Only keep edge-specific columns (not duplicated in profile)
    edge_cols_to_join = [c for c in EDGE_ONLY_COLS if c in edge_df.columns]
    edge_subset = edge_df[join_on + edge_cols_to_join].copy()

    merged = profile_df.merge(edge_subset, on=join_on, how="left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    return merged
