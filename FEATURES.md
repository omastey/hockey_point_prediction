# Feature Reference

All features used in the XGBoost model, grouped by source and type.
Features are computed from three NHL API sources merged on `(playerId, season)` for player stats
and `(team, season)` for team context stats.

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
| `next_team` | Team abbreviation for the following season (helper, dropped after computing destination features) |
| `season_year` | Numeric season start year — computed but dropped before model fit |

---

## EDGE Stats
*Source: NHL Edge API — `/edge/skater-detail/{playerId}/{season}/{gameType}`*

### Basic Season Stats

| Feature | Description |
|---|---|
| `gamesPlayed` | Games played that season |
| `goals` | Total goals |
| `assists` | Total assists |
| `age` | Player age at the start of that season |
| `height_in` | Player height in inches |
| `weight_lb` | Player weight in pounds |

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

### Zone Time

| Feature | Description |
|---|---|
| `oz_pct` | Percentage of zone time spent in the offensive zone |
| `dz_pct` | Percentage of zone time spent in the defensive zone |

### Shooting Metrics

| Feature | Description |
|---|---|
| `shots` | Total shots on goal |
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
| `switched_teams` | 1 if player changed teams vs the following season |

---

## Profile Stats
*Source: NHL Landing API — `/player/{playerId}/landing` (`seasonTotals` array)*

### Season Power Play Stats

| Feature | Description |
|---|---|
| `pp_goals` | Power play goals that season |
| `pp_points` | Power play points that season |
| `toi` | Average time on ice per game that season (seconds, converted from MM:SS) |

### Draft Information

| Feature | Description |
|---|---|
| `draft_round` | Round the player was drafted in (null if undrafted) |
| `draft_overall_pick` | Overall draft pick number (null if undrafted) |
| `is_undrafted` | 1 if the player was never drafted, 0 otherwise |

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
| `years_in_nhl` | Number of prior NHL regular seasons with at least `MIN_GAMES_FILTER` games played. 0 for rookies. |

---

## Team Context Stats
*Source: NHL Stats REST API — `api.nhle.com/stats/rest/en/team/{report}`. Joined onto the player dataset on `(team, season)` in `merge_datasets.py`.*

### Current Team Stats

| Feature | Description |
|---|---|
| `team_pp_pct` | Team power play percentage that season |
| `team_pp_goals_pg` | Team power play goals per game |
| `team_pp_opps_pg` | Team power play opportunities per game |
| `team_faceoff_pct` | Team overall faceoff win percentage |
| `team_oz_faceoff_pct` | Team offensive zone faceoff win percentage |
| `team_dz_faceoff_pct` | Team defensive zone faceoff win percentage |

### Next Season Team Environment
*Computed in `train_xgb.py`. For players staying on the same team, these equal the current team stats. For players switching teams, these reflect the destination team's stats — giving the model the actual environment the player will produce in next season.*

| Feature | Description |
|---|---|
| `next_team_pp_pct` | PP% of the team the player will play for next season |
| `next_team_pp_goals_pg` | PP goals per game of next season's team |
| `next_team_pp_opps_pg` | PP opportunities per game of next season's team |
| `next_team_faceoff_pct` | Faceoff win % of next season's team |
| `next_team_oz_faceoff_pct` | Offensive zone faceoff win % of next season's team |
| `next_team_dz_faceoff_pct` | Defensive zone faceoff win % of next season's team |

---

## Engineered Features
*Derived during training preprocessing in `train_xgb.py`.*

### Intermediate Per-Game Rates
*Computed from season totals; used directly as features and as inputs to delta calculations.*

| Feature | Formula | Description |
|---|---|---|
| `shots_pg` | `shots / gamesPlayed` | Shots on goal per game |
| `pp_points_pg` | `pp_points / gamesPlayed` | Power play points per game |
| `bursts_pg` | `burstsOver20 / gamesPlayed` | High-speed skating bursts per game |
| `distance_pg` | `totalDistance / gamesPlayed` | Miles skated per game |

### Derived Rate Features

| Feature | Formula | What It Captures |
|---|---|---|
| `dist_per_60` | `totalDistance / (toi / 3600)` | Miles skated per 60 minutes of ice time. Normalises total distance by TOI so a high-minute player isn't rewarded simply for playing more. A proxy for skating intensity and motor. |

### Year-Over-Year Delta Features
*Each delta is `current_season_value − previous_qualifying_season_value`. NaN for a player's
first season in the dataset (left as NaN for XGBoost; filled to 0 for RandomForest).*
*Counting stats are converted to per-game rates before differencing so that changes in
games played (e.g. a short season vs a full season) do not inflate or deflate the delta.*

| Feature | Based On | What It Captures |
|---|---|---|
| `delta_toi` | `toi` (avg sec/game) | Change in avg ice time per game — coach confidence signal |
| `delta_ppg` | `ppg` | Change in points-per-game rate — momentum in overall production |
| `delta_gpg` | `gpg` | Change in goals-per-game rate |
| `delta_apg` | `apg` | Change in assists-per-game rate |
| `delta_oz_pct` | `oz_pct` | Change in offensive zone deployment — coaching usage shift |
| `delta_shots_pg` | `shots / gamesPlayed` | Change in shots per game — offensive role / usage signal |
| `delta_pp_points_pg` | `pp_points / gamesPlayed` | Change in PP points per game — PP role change signal |
| `delta_bursts_pg` | `burstsOver20 / gamesPlayed` | Change in high-speed bursts per game — athleticism trend |
| `delta_distance_pg` | `totalDistance / gamesPlayed` | Change in distance skated per game — workload/conditioning |
| `delta_gamesPlayed` | `gamesPlayed` | Change in games played — health/availability trend |

### Delta Context Features

| Feature | Description |
|---|---|
| `has_prior_season` | 1 if the player has a qualifying prior season in the dataset, 0 if this is their first row. Allows tree models to learn to discount delta values for first-season rows rather than treating NaN-filled-to-0 as a genuine zero-change signal. |
| `prev_season_gp` | Games played in the prior qualifying season. Provides context for interpreting delta magnitudes — a +5 shot delta means something different if the player went from 20 games to 82 games vs staying at 82 games. |

### Career Trajectory Features

| Feature | Formula | What It Captures |
|---|---|---|
| `prev_career_high_ppg` | `max(ppg across all prior qualifying seasons)` | The player's personal best PPG entering this season. Null for the first season in dataset (filled to 0). |
| `career_high_gap` | `ppg − prev_career_high_ppg` | How far above/below their own career best a player is performing. Positive = at or above career high (breakout or peak); Negative = in a slump or declining from prior peak. |
| `age_squared` | `age²` | Captures the non-linear age curve — production tends to ramp up through the mid-20s and fall off more steeply after ~31. |
| `career_stage` | `0 if age ≤ 23, 1 if age ≤ 31, 2 otherwise` | Discrete career stage bucket: 0 = developing, 1 = prime, 2 = declining. |

### Regression-to-Mean Signals

| Feature | Formula | What It Captures |
|---|---|---|
| `shooting_pct_vs_career` | `shootingPercentage − career_shooting_pctg` | How far above/below a player's own career shooting % they performed this season. Positive = lucky year, likely to regress down. Negative = unlucky year, likely to recover. |
| `ppg_vs_career_rate` | `ppg − (career_points / career_games_played)` | How far above/below a player's own career PPG rate they are this season. Positive = above career average (regression risk). Negative = below career average (recovery upside). Null for rookies with no prior career data (filled to 0). |

---

## Notes

**`--no-base-stats` flag:** When passed, the following persistence stats are excluded from the model, forcing it to rely on delta and contextual signals instead. Delta versions of these features (e.g. `delta_ppg`) are still kept.

Excluded features: `ppg`, `apg`, `gpg`, `pp_points`, `assists`, `goals`, `pp_goals`, `gamesPlayed`, `shots`

---

## Target Variables

### Regressor Target (`train_xgb.py`)

| Variable | Description |
|---|---|
| `target_ppg_next` | Points per game in the **following** season (`ppg` shifted back by one season per player). The last season row per player is dropped since no future season exists. |

Supporting target columns (used for evaluation, not model input):

| Variable | Description |
|---|---|
| `target_points_next` | Total points in the following season |
| `target_gp_next` | Games played in the following season (used to convert predicted PPG → projected points) |

### Classifier Target (`train_breakout.py`)

| Variable | Description |
|---|---|
| `target_breakout` | Binary: 1 if the player's next season qualifies as a breakout, 0 otherwise. |

A breakout requires **both** conditions to be met:
1. `target_ppg_next > CAREER_HIGH_FACTOR × prev_career_high_ppg` — exceeds personal best by a factor (default 1.2)
2. `target_ppg_next - ppg >= YOY_JUMP_THRESHOLD` — large year-over-year improvement (default 0.15)

Both thresholds are configurable via CLI: `--career-high-factor` and `--yoy-jump`.
