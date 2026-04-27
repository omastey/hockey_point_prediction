SEASONS = [
    "20102011",
    "20112012",  # Lockout-shortened 2012-13 excluded (48 games)
    "20122013",
    "20132014",
    "20142015",
    "20152016",
    "20162017",
    "20172018",
    "20182019",
    "20192020",  # COVID-interrupted regular season (~70 games before pause)
    "20202021",  # COVID-shortened season (56 games)
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026",
]

# Edge tracking data only available from 2021-22 onward
EDGE_SEASONS = [s for s in SEASONS if s >= "20212022"]

GAME_TYPE = 2  # 2 = regular season

MIN_GAMES_FILTER = 10
