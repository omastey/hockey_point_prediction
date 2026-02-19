from pathlib import Path
import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load a raw CSV file into a DataFrame."""
    return pd.read_csv(path)


def save_processed_csv(df: pd.DataFrame, path: str) -> None:
    """Save a processed DataFrame to CSV, ensuring parent dirs exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def merge_parquet_datasets(
    edge_path: str,
    profile_path: str,
    output_path: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge EDGE and profile datasets on playerId and save as parquet."""
    edge_df = pd.read_parquet(edge_path)
    profile_df = pd.read_parquet(profile_path)

    merged = edge_df.merge(profile_df, on="playerId", how=how, suffixes=("_edge", "_profile"))

    # Resolve duplicate columns (e.g., fullName_edge/fullName_profile)
    if "fullName_edge" in merged.columns and "fullName_profile" in merged.columns:
        merged["fullName"] = merged["fullName_edge"].fillna(merged["fullName_profile"])
        merged = merged.drop(columns=["fullName_edge", "fullName_profile"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    return merged
