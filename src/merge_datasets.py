from pathlib import Path

import pandas as pd

try:
    from .data import merge_parquet_datasets
except ImportError:
    from data import merge_parquet_datasets


EDGE_PATH        = "edge_data/nhl_edge_model_dataset.parquet"
PROFILE_PATH     = "edge_data/nhl_full_stats.parquet"
TEAM_STATS_PATH  = "edge_data/nhl_team_stats.parquet"
OUTPUT_PATH      = "edge_data/nhl_merged_dataset.parquet"
MERGE_HOW = "inner"  # use "left" to keep all edge rows


def main() -> None:
    merged = merge_parquet_datasets(
        edge_path=EDGE_PATH,
        profile_path=PROFILE_PATH,
        output_path=OUTPUT_PATH,
        how=MERGE_HOW,
    )

    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    merged = merged.merge(team_stats, on=["team", "season"], how="left")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print(f"Merged dataset shape: {merged.shape}")
    print(f"Saved: {Path(OUTPUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
