# Feature Reference

All features used in the XGBoost model, grouped by source and type.
Features are computed from two NHL API sources merged on `(playerId, season)`.

---

## Identity Columns (not used as model features)

| Column | Description |
|---|---|
| `playerId` | NHL player ID |
| `fullName` | Player full name |
| `season` | Season string, e.g. `"20232024"` |
| `position` | Raw position code: `C`, `L`, `R`, `D` |
| `shoots` | Raw handedness: `L` or `R` |
| `team` | Team abbreviation for that season |

---

## EDGE Stats
*Source: NHL Edge API — `/edge/skater-detail/{playerId}/{season}/{gameType}`*

### Basic Season Stats

| Feature | Description |
|---|---|
| `gamesPlayed` | Games played that season |
| `goals` | Total goals |
| `assists` | Total assists |
| `points` | Total points (goals + assists) |
| `age` | Player age at the start of that season |

### Per-Game Rates

| Feature | Description |
|---|---|
| `ppg` | Points per game (`points / gamesPlayed`) |
| `gpg` | Goals per game (`goals / gamesPlayed`) |
| `apg` | Assists per game (`assists / gamesPlayed`) |

### Advanced Skating Metrics

| Feature | Description |
|---|---|
| `topShotSpeed` | Player's top recorded shot speed (mph) |
| `topShotSpeed_pct` | Percentile rank for top shot speed across all skaters |
| `speedMax` | Player's top recorded skating speed (mph) |
| `speedMax_pct` | Percentile rank for max skating speed across all skaters |
| `burstsOver20` | Number of skating bursts above 20 mph in the season |
| `totalDistance` | Total distance skated over the season (miles) |

### Zone Time

| Feature | Description |
|---|---|
| `oz_pct` | Percentage of zone time spent in the offensive zone |
| `dz_pct` | Percentage of zone time spent in the defensive zone |

### Shooting Metrics

| Feature | Description |
|---|---|
| `shots` | Total shots on goal |
| `shotsPercentile` | Percentile rank for shot volume |
| `shootingPercentage` | Season shooting percentage (`goals / shots`) |
| `shootingPctgPercentile` | Percentile rank for shooting percentage |

### Encoded Categorical Features

| Feature | Description |
|---|---|
| `shoots_left` | 1 if player shoots left, 0 otherwise |
| `shoots_right` | 1 if player shoots right, 0 otherwise |
| `is_center` | 1 if position is center (C) |
| `is_winger` | 1 if position is left wing (L) or right wing (R) |
| `is_defenseman` | 1 if position is defenseman (D) |
| `switched_teams` | 1 if player changed teams vs prior qualifying season |

---

## Profile Stats
*Source: NHL Landing API — `/player/{playerId}/landing` (`seasonTotals` array)*

### Season Power Play Stats

| Feature | Description |
|---|---|
| `pp_goals` | Power play goals that season |
| `pp_points` | Power play points that season |
| `toi` | Average time on ice per game that season (seconds, converted from MM:SS) |
| `pp_toi` | Average power play time on ice per game (mostly unavailable from API) |

### Career Cumulative Stats
*Cumulative totals from all prior NHL regular seasons — does NOT include the current season row.*

| Feature | Description |
|---|---|
| `career_points` | Career points entering this season |
| `career_games_played` | Career games played entering this season |
| `career_goals` | Career goals entering this season |
| `career_assists` | Career assists entering this season |
| `career_pp_points` | Career power play points entering this season |
| `career_pp_goals` | Career power play goals entering this season |
| `career_shooting_pctg` | Career shooting percentage entering this season (`career_goals / career_shots * 100`); null for players with no prior shot attempts |

---

## Engineered Features
*Derived during training preprocessing in `train_xgb.py`.*

### Season Encoding

| Feature | Description |
|---|---|
| `season_year` | Numeric encoding of the season start year (e.g. `"20232024"` → `2023`) |

### Year-Over-Year Delta Features
*Each delta is `current_season_value − previous_qualifying_season_value`. Null for a player's
first season in the dataset (filled to 0). Note: 20212022 deltas are noisy because the prior season
(20202021) was a COVID-shortened 56-game season.*

| Feature | Based On | What It Captures |
|---|---|---|
| `delta_toi` | `toi` | Change in avg ice time per game — coach confidence signal |
| `delta_ppg` | `ppg` | Change in points-per-game rate — momentum in overall production |
| `delta_gpg` | `gpg` | Change in goals-per-game rate |
| `delta_apg` | `apg` | Change in assists-per-game rate |
| `delta_shots` | `shots` | Change in total shot volume — offensive role / usage signal |
| `delta_pp_points` | `pp_points` | Change in power play production — PP role change signal |
| `delta_gamesPlayed` | `gamesPlayed` | Change in games played — health/availability trend |
| `delta_oz_pct` | `oz_pct` | Change in offensive zone deployment — coaching usage shift |
| `delta_totalDistance` | `totalDistance` | Change in total distance skated — workload/conditioning |
| `delta_burstsOver20` | `burstsOver20` | Change in high-speed burst count — athleticism trend |

### Regression-to-Mean Signals

| Feature | Formula | What It Captures |
|---|---|---|
| `shooting_pct_vs_career` | `shootingPercentage − career_shooting_pctg` | How far above/below a player's own career shooting % they performed this season. Positive = lucky year, likely to regress down. Negative = unlucky year, likely to recover. |
| `ppg_vs_career_rate` | `ppg − (career_points / career_games_played)` | How far above/below a player's own career PPG rate they are this season. Positive = above career average (regression risk). Negative = below career average (recovery upside). Null for rookies with no prior career data (filled to 0). |

---

## Target Variable

| Variable | Description |
|---|---|
| `target_ppg_next` | Points per game in the **following** season (`ppg` shifted back by one season per player). The last season row per player is dropped since no future season exists. |

Supporting target columns (used for evaluation, not model input):

| Variable | Description |
|---|---|
| `target_points_next` | Total points in the following season |
| `target_gp_next` | Games played in the following season (used to convert predicted PPG → projected points) |
