# Breakout Classifier — Run Leaderboard

## TL;DR — Best Models

| Best by | Run | Model | Avg Prec | ROC-AUC | Notes |
|---|---|---|---|---|---|
| **CV-pooled (honest)** | **#25** | XGB, 3-fold rolling-origin CV (Phase A MoneyPuck features) | **0.261** | 0.863 | Pooled across 2022-23/23-24/24-25 (1428 rows, 54 breakouts). Per-fold APs 0.265-0.280 — very stable (range 0.015). |
| **Single-season P@K** | **#21** | XGB + raw MoneyPuck columns | 0.301 | 0.879 | P@10 jumped 30% → 60% (Lift@10 = 17.5×, the best ever). On_ice_corsi%, xg, on_ice_xg% all rank in top features. |
| **Calibration (Brier)** | **#19** | XGB + Sigmoid calibration | 0.298 | 0.883 | Brier 0.028 (best calibrated); P@10 = 40%. |
| ~~Single-season AP~~ | ~~#14~~ | ~~Logistic (ElasticNet, no base stats)~~ | ~~0.411~~ | ~~0.902~~ | **Run #26 (Logistic CV) revealed pooled AP 0.203 vs single-season 0.411 — the gap was largely test-season luck.** Run 14's headline number is not reproducible across seasons. |

**Current production recommendation: Run #25** (XGB + Phase A MoneyPuck features, validated by 3-fold rolling-origin CV). The earlier Run #14 logistic model looked best on a single test season but CV exposed it as overfit to 2024-25 specifically.

See full run details below for what was tried, what worked, and what regressed.

---

Test set: season noted per run | Threshold: 0.30 (unless noted)

| # | Model | Key Changes | TP | FP | FN | Prec | Recall | ROC-AUC | Avg Prec | Brier | Notes |
|---|-------|-------------|----|----|-----|------|--------|---------|----------|-------|-------|
| 1 | XGB | Initial classifier (multiplicative 1.2x breakout def, 33 actual breakouts) | 4 | — | 29 | — | 0.12 | 0.658 | — | — | is_undrafted dominated importance |
| 2 | XGB | career_stage 4-bucket, eval_metric=aucpr, max_depth=4, lr=0.08 | 4 | — | 29 | — | 0.12 | ~0.66 | — | — | is_undrafted still #1 importance |
| 3 | XGB | Switched to additive breakout def (+0.15 over career high, PPG floor 0.45, 26 actual breakouts) | 7 | 15 | 19 | 0.32 | 0.27 | 0.742 | — | — | Big improvement, first good run |
| 4 | XGB | Added roster depth features (players_ahead, pp_share, roster_turnover, etc.) | 3 | — | 23 | — | 0.12 | ~0.70 | — | — | Roster features added noise, much worse |
| 5 | XGB | Aggressive feature cut (~40 features archived) | 5 | 17 | 21 | 0.23 | 0.19 | ~0.71 | — | — | MacKinnon + McDavid as FPs, lost elite context |
| 6 | XGB | Restored career_points, career_games_played, career_shooting_pctg, shootingPct, career_ppg_slope | 7 | 22 | 19 | 0.24 | 0.27 | 0.727 | 0.155 | 0.057 | Back to 7 TP, elites still FPs |
| 7 | XGB | Added ppg_league_percentile + ppg_x_career_high_gap | 8 | 22 | 18 | 0.27 | 0.31 | 0.738 | 0.171 | 0.054 | Best TP count, ppg_x_career_high_gap at #3 importance |
| 8 | XGB | Added ppg_x_league_pctile interaction | 4 | 27 | 22 | 0.13 | 0.15 | 0.749 | 0.141 | 0.059 | Two PPG interactions competed, tanked recall |
| 9 | XGB | Removed ppg_x_league_pctile (back to run 7 features) | 8 | 22 | 18 | 0.27 | 0.31 | 0.738 | 0.171 | 0.054 | Same as run 7 baseline |
| 10 | Logistic | L1 Lasso, C=0.5, class_weight=balanced | 23 | 230 | 3 | 0.09 | 0.88 | 0.724 | 0.119 | 0.204 | 88% recall but 253 predicted! Finds Type B breakouts. Poorly calibrated. |
| 11 | Logistic | L1 Lasso, C=0.5, class_weight 5:1 | 16 | 141 | 10 | 0.10 | 0.62 | 0.732 | 0.125 | 0.125 | 157 predicted, finds Necas/Werenski/Hagel/Protas. Better calibrated. |
| 12 | Logistic | **16 seasons (2010-2026)**, test=20242025, profile-primary merge, edge cols left-joined | 13 | 81 | 3 | 0.14 | 0.81 | 0.896 | 0.397 | 0.060 | Massive jump: ROC-AUC .73→.90, AvgPrec .13→.40. 13/16 breakouts caught. Missed: Brazeau, Samuelsson, Raddysh. Top features: career_high_gap (1.88), gpg (-1.11), players_ahead (-0.83). L1 selected 46/56. |
| 13 | XGB | 16 seasons, test=20242025, same data as run 12 | 11 | 32 | 5 | 0.26 | 0.69 | 0.885 | 0.276 | 0.037 | Better calibrated (Brier .037, LogLoss .132) and more precise (26%) but worse discrimination (AP .28 vs .40). Misses Necas (0.004!) and Carlsson. Over-indexes on career_stage (0.16) and age_squared (0.07). |
| 14 | Logistic | **ElasticNet** (l1=0.7), no base stats, drop pos_players_ahead + age_squared | 13 | 79 | 3 | 0.14 | 0.81 | **0.902** | **0.411** | 0.059 | Best ROC-AUC and AP yet. Cleaner coefficients: no opposite-sign collinearity. ppg_per_minute emerges (-0.65). At t=0.50: 10 TP / 28 FP (tighter than run 12's 10/32). |
| 15 | XGB | No base stats, drop pos_players_ahead + age_squared | 9 | 34 | 7 | 0.21 | 0.56 | 0.899 | 0.259 | 0.040 | Worse than run 13 — removing base stats hurt XGB more. Misses Bedard (0.29), Carlsson (0.22), Slafkovsky (0.22), Necas (0.009). Still age-dominated. |
| 16 | Logistic | CV-tuned: C=0.2, l1_ratio=1.0 (pure L1), class_weight 3:1 | 11 | 42 | 5 | 0.21 | 0.69 | 0.897 | 0.370 | 0.039 | More conservative — stronger reg, sparser (12/45 zeroed). Better calibrated but catches fewer breakouts. CV AP flat across top configs (~0.235), confirming run 14 defaults are in the right neighborhood. |
| 17 | XGB | Goalies filtered upstream + base stats restored as default | 10 | 42 | 6 | 0.19 | 0.62 | 0.892 | 0.274 | 0.044 | +1 TP vs run 15. Base stats clearly help XGB (AP 0.259→0.274). Goalie filter dropped 1083 rows. |
| 18 | Logistic | Goalies filtered upstream + base stats restored as default | 14 | 85 | 2 | 0.14 | 0.88 | 0.894 | 0.343 | 0.067 | Catches 14/16 (best recall yet) but AP regressed 0.411→0.343 — base stats add noise that elasticnet doesn't fully filter. Consider running logistic with --no-base-stats going forward. |
| 19 | XGB | **Sigmoid calibration** (FrozenEstimator + CalibratedClassifierCV, calibrated on 2023-24, threshold 0.40 calibrated) | 4 | 6 | 12 | 0.40 | 0.25 | 0.883 | **0.298** | **0.028** | At threshold 0.40 calibrated, only 10 predictions made (very tight). At threshold 0.10 calibrated: 10 TP / 29 FP (matches uncalibrated 0.40 default but with way better Brier). **P@10: 30%→40%**, AP 0.274→0.298. Threshold scale shifted — calibrated probs are smaller. |
| 20 | XGB | **Isotonic calibration** (same setup as run 19, method=isotonic) | 6 | 12 | 10 | 0.33 | 0.38 | 0.869 | 0.234 | 0.029 | **P@5: 0%→40%** (huge top-5 win), but AP regressed 0.274→0.234. With only 22 calibration positives, isotonic's piecewise function overfits — sigmoid is more robust here. |
| 21 | XGB | **+MoneyPuck raw cols** (xg, xg_flurry, xg_high_danger, on_ice_xg/corsi%, oz/dz starts, ice_time_rank, game_score, off_ice_xg_for, icetime — no goals/assists). Threshold 0.40. | 10 | 27 | 6 | 0.27 | 0.62 | 0.879 | **0.301** | 0.041 | Same TP/FN as run 17 but **15 fewer FPs** (42→27). AP 0.274→0.301. **P@10 jumped 30%→60% (Lift@10 = 17.5x — best ever)**. Top MP features by importance: on_ice_corsi%, xg, on_ice_xg%, xg_high_danger. ROC-AUC slipped 0.892→0.879 (XGB ranks tighter at the top, looser in the tail). |
| 22 | Logistic | **+MoneyPuck raw cols, --no-base-stats** (run 14 defaults: ElasticNet, l1=0.7, C=0.5, 5:1) | 13 | 56 | 3 | 0.19 | 0.81 | 0.894 | 0.281 | 0.067 | TP unchanged vs run 14 (13/16) but FP 79→56 (better). However AP regressed 0.411→0.281 — raw MP cols add noise that elasticnet doesn't filter cleanly, and `mp_xg_high_danger` picked up a *negative* coefficient (likely the elite-ceiling effect). Logistic needs engineered MP features (puck-luck residual, OZ start %), not raw counts. |
| 23 | XGB | **+Phase A engineered MP features**: xg_luck, oz_start_pct, xg_per_60, xg_high_danger_per_60 (raw MP cols still included) | 9 | 24 | 7 | 0.27 | 0.56 | 0.894 | 0.265 | 0.039 | Slightly *worse* than run 21 (P@10 60%→40%, AP 0.301→0.265, TP 10→9). FP count keeps improving (27→24). **`oz_start_pct` is XGB's #1 MP feature (importance 0.016)** — clear win for that one feature. `xg_luck` ranks low (0.0097) — likely swamped by collinearity with raw `goals`. Per-60 features dilute signal already present in raw `mp_xg` etc. |
| 24 | Logistic | **+Phase A, --no-base-stats** (run 14 defaults) | 12 | 62 | 4 | 0.16 | 0.75 | 0.895 | 0.275 | 0.069 | Marginally worse than run 22 across the board. **`oz_start_pct` is the strongest engineered feature (coef +0.585) — second only to career_high_gap and ppg_per_minute family.** `xg_luck` only gets coef −0.07 (right direction, very weak — collinearity with `goals`). Pattern emerging: Phase A *adds* features instead of *replacing* base counterparts; consider a run that swaps `goals`/`shootingPercentage` for `xg_luck`. |
| 25 | XGB | **3-fold rolling-origin CV** across 22-23, 23-24, 24-25 (Phase A features, base stats included) | 28 | 68 | 26 | 0.29 | 0.52 | 0.863 | 0.261 | 0.042 | Pooled across 3 test seasons (1428 rows, 54 breakouts). Per-fold AP: 22-23=0.280, 23-24=0.272, 24-25=0.265 — **very stable (range 0.015)**. Pooled AP 0.261 ≈ single-season Run 23's 0.265 → XGB was *not* fooled by single-season noise. P@10 pooled=30% (note: pooled P@k ≠ per-season P@k — different denominator). |
| 26 | Logistic | **3-fold rolling-origin CV** (--no-base-stats, Phase A features) | 38 | 171 | 16 | 0.18 | 0.70 | 0.862 | 0.203 | 0.072 | Per-fold AP: 22-23=0.238, 23-24=0.236, 24-25=0.275 — range 0.039 (less stable than XGB). **Pooled AP 0.203 vs single-season Run 14's 0.411 — the headline gap was largely test-season luck.** CV reveals Logistic actually performs *worse* than XGB under honest evaluation, reversing prior conclusion from runs 14/17. |
