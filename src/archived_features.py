"""Archived features — removed from breakout classifier to reduce noise.

These features are still computed by feature_engineering.py and used by the
regressor (train_xgb.py), but are excluded from the breakout classifier's
feature set via BREAKOUT_DROP_FEATURES in train_breakout.py.

Kept here for reference and potential future use.
"""

# =====================
# Next-Season Team Environment Stats
# =====================
# Removed: leakage-adjacent — we don't know a player's next team at prediction time.
# These are useful for the regressor (which has access to next-season info) but
# misleading for the breakout classifier.
#
# Features:
#   next_team_pp_pct, next_team_pp_goals_pg, next_team_pp_opps_pg,
#   next_team_faceoff_pct, next_team_oz_faceoff_pct, next_team_dz_faceoff_pct

# =====================
# Current Team Aggregate Stats
# =====================
# Removed: team-level PP and faceoff stats showed consistently low importance
# for breakout prediction. The player's own role within the team (captured by
# player_pp_share, players_ahead_on_team, etc.) is more predictive.
#
# Features:
#   team_pp_pct, team_pp_goals_pg, team_pp_opps_pg,
#   team_faceoff_pct, team_oz_faceoff_pct, team_dz_faceoff_pct

# =====================
# Physical Attributes
# =====================
# Removed: consistently 0.0 importance across all breakout model iterations.
# Height and weight do not appear to differentiate breakout candidates.
#
# Features:
#   height_in, weight_lb, over_6ft

# =====================
# Low-Signal Categoricals
# =====================
# Removed: binary encoded features that added noise without breakout signal.
# is_defenseman is kept via the defenseman_x_pp_points_pg interaction instead.
#
# Features:
#   has_prior_season, is_undrafted, is_center, is_winger,
#   shoots_left, shoots_right

# =====================
# Skating Speed / Distance Metrics
# =====================
# Removed: raw skating metrics (speed, shot speed, distance) showed low
# importance for breakout prediction. The per-game burst features are kept.
#
# Features:
#   speedMax, speedMax_pct, topShotSpeed, topShotSpeed_pct,
#   delta_distance_pg, distance_pg

# =====================
# Raw Shooting Metrics
# =====================
# Removed: shooting percentage and its percentile rank had low importance.
# The regression signal (shooting_pct_vs_career) captures the useful
# information from shooting % in a more breakout-relevant way.
#
# Features:
#   shootingPercentage, shootingPctgPercentile

# =====================
# Career Counting Stats
# =====================
# Removed: raw career totals are redundant with the derived rate/trajectory
# features (career_high_gap, ppg_vs_career_rate, years_in_nhl, etc.).
# Counting stats scale with games played, adding noise.
#
# Features:
#   career_points, career_goals, career_assists, career_games_played,
#   career_pp_points, career_pp_goals, career_shooting_pctg

# =====================
# Other Low-Signal Features
# =====================
# defenseman_x_delta_ppg: consistently 0.0 importance; the PP interaction
#   (defenseman_x_pp_points_pg) captures the D-man signal better.
# years_since_draft: approximation (age - 18) was too coarse; draft_round
#   and draft_overall_pick capture pedigree more directly.
# career_ppg_slope: unstable importance across iterations (jumped from #1
#   to 0.0 between runs). May be overfitting to small sample.
#
# Features:
#   defenseman_x_delta_ppg, years_since_draft, career_ppg_slope
