from pathlib import Path

import pandas as pd

try:
    from .data import merge_parquet_datasets
except ImportError:
    from data import merge_parquet_datasets


PROFILE_PATH     = "data/nhl_full_stats.parquet"
EDGE_PATH        = "data/nhl_edge_model_dataset.parquet"
TEAM_STATS_PATH  = "data/nhl_team_stats.parquet"
OUTPUT_PATH      = "data/nhl_merged_dataset.parquet"


def main() -> None:
    merged = merge_parquet_datasets(
        profile_path=PROFILE_PATH,
        edge_path=EDGE_PATH,
        output_path=OUTPUT_PATH,
    )

    team_stats = pd.read_parquet(TEAM_STATS_PATH)
    merged = merged.merge(team_stats, on=["team", "season"], how="left")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print(f"Merged dataset shape: {merged.shape}")
    print(f"Saved: {Path(OUTPUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
