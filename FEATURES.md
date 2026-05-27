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
*Derived in `feature_engineering.py`, shared by both `train_xgb.py` and `train_breakout.py`.*

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
| `pct_of_career_high` | `ppg / prev_career_high_ppg` | How close the player is to their personal ceiling. Values near 1.0 mean the player is approaching a breakout threshold. Null for first season (filled to 0). |
| `career_ppg_slope` | Linear regression slope of PPG across all career seasons | The trend direction of a player's production. Positive = improving trajectory, negative = declining. Null for players with only one season. |
| `career_stage` | `0 if age ≤ 22, 1 if ≤ 26, 2 if ≤ 32, 3 otherwise` | Discrete career stage bucket: 0 = developing, 1 = entering prime, 2 = prime, 3 = declining. |

### Breakout-Signal Features

| Feature | Formula | What It Captures |
|---|---|---|
| `ppg_per_minute` | `ppg / (toi / 60)` | Scoring efficiency per minute of ice time. High efficiency + low TOI = upside if role expands. |
| `age_x_delta_toi` | `age × delta_toi` | Interaction feature: young player getting increasing ice time is a classic breakout setup. |
| `years_since_draft` | `age − 18` (drafted players only) | Approximation of years since draft. Late bloomers (4–5 years post-draft) have breakout potential. Null for undrafted players. |
| `over_6ft` | `1 if height_in >= 72, 0 otherwise` | Binary size flag — larger players may have different breakout profiles. |
| `ppg_league_percentile` | `rank(ppg) within season` | Where this player ranks league-wide. Elite players (99th pct) have no room to "break out" further. |
| `ppg_x_career_high_gap` | `ppg × career_high_gap` | Separates elite players peaking (high PPG + positive gap = at ceiling) from emerging players (low PPG + positive gap = room to grow). |

### Team Roster Depth Features
*Computed from the player dataset itself by grouping on `(team, season)`.
These measure a player's position within their team's depth chart and identify opportunity signals.*

| Feature | Formula | What It Captures |
|---|---|---|
| `players_ahead_on_team` | Count of teammates with higher PPG | How deep in the lineup this player sits. Fewer players ahead = closer to top role. |
| `pos_players_ahead` | Count of same-position teammates with higher PPG | Positional depth — a D-man with 0 defensemen ahead is about to be the #1. |
| `team_top_player_age` | Age of the team's highest-PPG player | If the team's star is 32+, role inheritance opportunity for younger players. |
| `team_pp_concentration` | Fraction of team PP points held by top 2 players | High concentration + aging top players = PP role about to open up. |
| `player_pp_share` | `player_pp_points / team_total_pp_points` | Player's current share of team PP production — low share + young age = upside. |
| `team_roster_turnover` | Fraction of prior season's roster that departed | High turnover = more ice time and roles available for remaining/new players. |

### Position Interaction Features

| Feature | Formula | What It Captures |
|---|---|---|
| `defenseman_x_delta_ppg` | `is_defenseman × delta_ppg` | Defenseman-specific PPG momentum — D-men break out differently than forwards. |
| `defenseman_x_pp_points_pg` | `is_defenseman × pp_points_pg` | Defenseman PP involvement — a D-man getting PP time has high breakout potential. |

### Regression-to-Mean Signals

| Feature | Formula | What It Captures |
|---|---|---|
| `shooting_pct_vs_career` | `shootingPercentage − career_shooting_pctg` | How far above/below a player's own career shooting % they performed this season. Positive = lucky year, likely to regress down. Negative = unlucky year, likely to recover. |
| `ppg_vs_career_rate` | `ppg − (career_points / career_games_played)` | How far above/below a player's own career PPG rate they are this season. Positive = above career average (regression risk). Negative = below career average (recovery upside). Null for rookies with no prior career data (filled to 0). |

### MoneyPuck-Derived Features
*Engineered from joined MoneyPuck columns (`mp_*`). Computed in `compute_moneypuck_features()`. Pre-2008 rows have NaN.*

| Feature | Formula | What It Captures |
|---|---|---|
| `xg_luck` | `goals − mp_xg_flurry` | Goals scored above/below expected. Negative = unlucky shooter due for upward regression — the canonical breakout signal. More principled than `shooting_pct_vs_career` because xG controls for shot quality, not just career-average %. |
| `oz_start_pct` | `mp_oz_starts / (mp_oz_starts + mp_dz_starts)` | Fraction of non-neutral shift starts that began in the offensive zone. Pure deployment signal — coaches give OZ starts to players they trust offensively. New signal not captured anywhere else. |
| `xg_per_60` | `mp_xg_flurry / (mp_icetime / 3600)` | Expected goals per 60 minutes of ice time. Strips out role/usage so a 4th-liner is comparable to a 1st-liner. Leading indicator (xG → goals next year), where `ppg_per_minute` is lagging. |
| `xg_high_danger_per_60` | `mp_xg_high_danger / (mp_icetime / 3600)` | Per-60 expected goals from the slot/net-front area only. Filters out perimeter volume — distinguishes real chance creators from point-shot defensemen. |

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

A breakout requires **all three** conditions to be met:
1. `target_ppg_next > prev_career_high_ppg + CAREER_HIGH_MARGIN` — exceeds personal best by an additive margin (default 0.15)
2. `target_ppg_next - ppg >= YOY_JUMP_THRESHOLD` — large year-over-year improvement (default 0.15)
3. `target_ppg_next >= PPG_FLOOR` — minimum production threshold to filter noise (default 0.45, ~37 points over 82 games)

All thresholds are configurable via CLI: `--career-high-margin`, `--yoy-jump`, `--ppg-floor`.

---

## Future Feature Ideas
*Features that could improve breakout prediction but require new data sources or additional research.*

### Requires New Data Sources

| Feature Idea | Data Source Needed | Why It Helps |
|---|---|---|
| Linemate quality (avg PPG of top linemates) | Line combination data (MoneyPuck, NaturalStatTrick, DailyFaceoff) | A player promoted to play with elite talent is primed to break out |
| PP unit rank (PP1 vs PP2) | Detailed PP unit tracking | Moving onto PP1 is one of the biggest breakout catalysts |
| Coaching changes | Manual tracking or external database | New coaches often promote different players and reshape lineups |
| Individual expected goals (ixG) | Advanced stats providers (MoneyPuck, Evolving Hockey) | Underlying process metrics that lead PPG — high ixG + low goals = due for breakout |
| Corsi/Fenwick shot attempt share | Advanced stats providers | Puck possession proxy — players on dominant possession teams outperform |
| Prospect pipeline / org depth charts | Prospect ranking sites (EliteProspects, HockeyProspecting) | High-pedigree prospects blocked by veterans are primed for breakout when opportunity opens |

### Archived Features (removed from breakout classifier)
*These features are still computed and used by the regressor, but excluded from `train_breakout.py` to reduce noise. See `src/archived_features.py` for full rationale.*

| Feature(s) | Reason Removed |
|---|---|
| `next_team_pp_pct`, `next_team_pp_goals_pg`, `next_team_pp_opps_pg`, `next_team_faceoff_pct`, `next_team_oz_faceoff_pct`, `next_team_dz_faceoff_pct` | Leakage-adjacent — next team unknown at prediction time |
| `team_pp_pct`, `team_pp_goals_pg`, `team_pp_opps_pg`, `team_faceoff_pct`, `team_oz_faceoff_pct`, `team_dz_faceoff_pct` | Consistently low importance; player-level role features capture the signal better |
| `height_in`, `weight_lb`, `over_6ft` | Consistently 0.0 importance across all iterations |
| `has_prior_season`, `is_undrafted`, `is_center`, `is_winger`, `shoots_left`, `shoots_right` | Low-signal categoricals; `is_defenseman` kept via interaction features |
| `speedMax`, `speedMax_pct`, `topShotSpeed`, `topShotSpeed_pct`, `delta_distance_pg`, `distance_pg` | Raw skating metrics showed low breakout importance; burst features kept |
| `shootingPercentage`, `shootingPctgPercentile` | Regression signal (`shooting_pct_vs_career`) captures this better |
| `career_points`, `career_goals`, `career_assists`, `career_games_played`, `career_pp_points`, `career_pp_goals`, `career_shooting_pctg` | Redundant with derived rate/trajectory features; counting stats add noise |
| `defenseman_x_delta_ppg`, `years_since_draft`, `career_ppg_slope` | Unstable or zero importance across iterations |

### Research / Modeling Ideas

| Idea | Description |
|---|---|
| Position-specific breakout thresholds | Defensemen and forwards break out differently — separate definitions or models could improve accuracy |
| Calibrated probability output | `CalibratedClassifierCV` to make predicted probabilities match actual breakout rates (requires more training data) |
| Separate young vs prime-age models | Train two classifiers: one for career_stage 0-1, another for 2-3, since the breakout drivers are fundamentally different |
| Next-season role prediction | Two-stage model: first predict TOI/PP role change, then use predicted role as input to breakout model |
