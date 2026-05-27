"""Load MoneyPuck skater CSVs into a parquet keyed on (playerId, season).

MoneyPuck data was downloaded manually (license-gated, no scraping). Files live
in data/moneypuck/. Schema notes:
  - 5 rows per player-season (situation = all|5on5|5on4|4on5|other) — we keep `all`
  - Season is start-year (2015 = 2015-16) — converted to our 8-digit format
  - Names have stripped/normalized diacritics — never join on name, only playerId
"""
from pathlib import Path

import pandas as pd

MONEYPUCK_DIR = Path("data/moneypuck")
HISTORICAL_CSV = MONEYPUCK_DIR / "skaters_2008_to_2024.csv"
CURRENT_CSV = MONEYPUCK_DIR / "skaters.csv"
OUTPUT_PATH = "data/nhl_moneypuck_skaters.parquet"

# Curated subset — see HYPERPARAMETERS.md / FEATURES.md for column rationale.
# Prefixed `mp_` on output to disambiguate from existing profile/edge columns.
KEEP_COLS = [
    "playerId",
    "season",
    "games_played",
    "icetime",
    "iceTimeRank",
    "gameScore",
    "I_F_xGoals",
    "I_F_flurryAdjustedxGoals",
    "I_F_highDangerxGoals",
    "I_F_oZoneShiftStarts",
    "I_F_dZoneShiftStarts",
    "onIce_xGoalsPercentage",
    "onIce_corsiPercentage",
    "OffIce_F_xGoals",
]

RENAME = {
    "games_played": "mp_gp",
    "icetime": "mp_icetime",
    "iceTimeRank": "mp_ice_time_rank",
    "gameScore": "mp_game_score",
    "I_F_xGoals": "mp_xg",
    "I_F_flurryAdjustedxGoals": "mp_xg_flurry",
    "I_F_highDangerxGoals": "mp_xg_high_danger",
    "I_F_oZoneShiftStarts": "mp_oz_starts",
    "I_F_dZoneShiftStarts": "mp_dz_starts",
    "onIce_xGoalsPercentage": "mp_on_ice_xg_pct",
    "onIce_corsiPercentage": "mp_on_ice_corsi_pct",
    "OffIce_F_xGoals": "mp_off_ice_xg_for",
}


def _load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=KEEP_COLS + ["situation"])
    df = df[df["situation"] == "all"].drop(columns=["situation"])
    return df


def build_moneypuck_dataset() -> pd.DataFrame:
    if not HISTORICAL_CSV.exists() or not CURRENT_CSV.exists():
        raise FileNotFoundError(
            f"Expected MoneyPuck CSVs in {MONEYPUCK_DIR}/ — got "
            f"{[p.name for p in MONEYPUCK_DIR.glob('*.csv')]}"
        )

    df = pd.concat([_load_one(HISTORICAL_CSV), _load_one(CURRENT_CSV)], ignore_index=True)

    # MoneyPuck uses start-year ints; profile dataset uses concatenated 8-digit strings
    df["season"] = df["season"].astype(int).apply(lambda s: f"{s}{s + 1}")

    df = df.rename(columns=RENAME)

    before = len(df)
    df = df.drop_duplicates(subset=["playerId", "season"], keep="last")
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate (playerId, season) rows")

    return df


def main() -> None:
    df = build_moneypuck_dataset()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"MoneyPuck skater dataset: {df.shape}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Saved: {Path(OUTPUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
