# A-share Model Research Archive (2026-08-10)

## 1. Purpose and status

This document is the consolidated audit record for the 14:00 intraday, daily-plus-minute, and pure-daily optimization work completed through 2026-08-10.

Final status: `human_review_required`.

No experiment in this archive is approved for production. The current active model remains unchanged. Routine scheduler jobs remain disabled with `SCHEDULER_DISABLED=1`; realtime operation is isolated from this research.

## 2. Frozen execution standard

Unless a section explicitly says "native diagnostic", comparable strategy results use:

- A-share point-in-time data.
- Signal cutoff appropriate to the experiment: 13:55/14:00 for intraday studies, or T close for next-open daily studies.
- TopN without refill.
- Unfilled slots remain cash with zero return.
- Fixed-capital, cash-normalized portfolio returns.
- Round-trip cost: 20bp (`0.002`).
- Daily Top2 minimum fill: at least 1 filled name per day on average.
- Daily Top10 minimum fill: at least 2 filled names per day on average.
- Purged expanding or rolling walk-forward OOS evaluation.
- Open-label horizon 1 uses purge span 2 because entry is T+1 open and exit is T+2 open.
- No future-label column may be used as a feature.

Intraday adaptive T+3 studies additionally use:

- Entry around 14:50 after a 13:55/14:00 signal.
- Stop loss: -5%.
- Take profit: +9%.
- Trailing arm: +3%.
- Exit after a 2% drawdown from the running high.
- Unresolved/unexecutable exits receive the registered stress return.

## 3. Data and feature safety changes

The research exposed two independent causes of misleading backtests: execution-timing mismatch and insufficiently universal future-label filtering.

Implemented safety rules:

- Block all `target_*`, `label_*`, `adaptive_*`, `open_ret_*`, `tradable_ret_*`, `future_*`, `forward_*`, `fwd_*`, `next_*`, `realized_*`, `pnl_*`, `sell_return_*`, `exit_return_*`, and `holding_return_*` columns.
- Block `_target` and `_label` suffixes.
- Block raw execution columns such as `entry_open_next`, `exit_open_h`, `buyable_next`, and `buyable_close`.
- Preserve causal historical return features such as `ret_3d` and `ret_5d`.
- Enforce the rules both during feature discovery and at model boundaries.
- Bind open-target caches to the price-source signature.
- Bump the window-cache schema version after the safety change.

The active model's native close-based diagnostic had reported approximately +1.62% per day and Sharpe 264. This is not a valid production estimate: it uses information and execution assumptions unavailable at the real 14:00/14:50 decision point. A no-leak retrain reproduced the unrealistic native score, confirming that execution semantics, not only feature-schema contamination, were the dominant issue.

## 4. Intraday direct-return and target redesign

### 4.1 Direct-return immutable holdout

Protocol: `intraday_1400_direct_return_v1`.

Selected development recipe: `adaptive_realized_net_ret_t3`, Top5.

Untouched holdout result:

- Mean return: approximately -30.06bp/day.
- Compound return: -17.56%.
- Maximum drawdown: -24.22%.
- Positive 20-day blocks: 0 of 3.

Decision: failed. The holdout was consumed and must not be reused for selection.

### 4.2 Prospective target redesign

Protocol requires 123 fully matured trading days, exact eligible-key coverage, complete outcomes, append-only prefixes, and stable universe hashes.

Target families:

- Downside quantile.
- Cross-sectional rank.
- Conditional payoff.

This protocol remains a prospective framework; historical acceleration was performed in a separate backfill protocol.

### 4.3 Historical target-redesign backfill

Protocol: `intraday_1400_target_redesign_backfill_v2`.

Coverage:

- 2025-07-01 through 2026-08-03.
- 266 trading days.
- Four registered folds.
- 842,871 label rows.
- 798,743 final matched panel rows.
- 44,128 label keys lacked a daily prepared row and were excluded consistently from every candidate.

Results:

- Conditional payoff: approximately -21.96bp/day.
- Cross-sectional rank: approximately -41.17bp/day.
- Downside q20: near flat, but only 15.1% fill.

Decision: no production candidate. Conditional payoff was retained only as a development-only forward candidate before later daily studies superseded it.

## 5. Daily plus minute experiments

### 5.1 Raw minute feature enhancement

Protocol: `intraday_1400_daily_minute_enhancement_v1`.

Candidates included daily baseline plus all-minute, speed, path, volume/VWAP, risk, dependence, and context groups.

Representative strict results:

- Daily as-of baseline: approximately -8.50bp/day; fill about 1.19/10.
- Volume/VWAP variant: approximately -1.35bp/day; fill about 0.46/10.

Decision: no candidate met both return and absolute fill requirements. Low-fill apparent improvements were rejected.

### 5.2 Train-only minute residualization

Protocol: `intraday_1400_minute_feature_residualization_v1`.

Candidates:

- Daily plus OLS-residualized minute features.
- Daily plus Ridge-residualized minute features.

The cross-sectional-rank residual candidate was removed because it ranked over a label-conditioned OOS universe. Residualizer means, scales, and coefficients were fit on fold training dates only.

Results:

- Both variants reduced the loss magnitude relative to the daily baseline.
- Fill remained approximately 0.48-0.53/10.

Decision: failed the absolute fill gate. Measured dry-run peak RSS was about 9.65GiB; the registered memory budget was corrected from a static 2.63GiB estimate to 12GiB.

### 5.3 H1 buyability enhancement

Candidates:

- H1/buyability z-blend 50/50.
- H1/buyability z-blend 75/25.
- H1 Top50 constrained by buyability.

Results:

- 50/50: about -13.20bp/day; fill 8.96/10.
- 75/25: about -13.23bp/day; fill 3.64/10.
- Top50 buyability rerank: about -37.51bp/day; fill 9.58/10.

Decision: failed. Minute features predicted buyability, but higher buyability did not translate into better executable returns.

### 5.4 Executable-payoff heads

The v1 score combined buy probability, liquidation probability, conditional return, and a -10% unresolved-position stress payoff.

V1 result:

- Approximately -5.04bp/day.
- Fill about 1.39/10.
- Improved 3 of 4 folds, but failed fill and positive-return requirements.

V2 rerank results:

- H1 Top50 then payoff: about -15.86bp/day; fill 5.01/10.
- Payoff Top50 then buyability: about -22.16bp/day; fill 9.98/10.

Decision: failed. V2 is development-only because the v1 controller source was later overwritten; stored v1 artifacts remain hash-verifiable, but exact source replay is incomplete.

## 6. Pure-daily target and execution experiments

### 6.1 Close-target strict replay

The active close-target native diagnostic is invalid for 14:00/14:50 execution. Strict adaptive replay was approximately -4.78bp/day with very low fill.

Decision: retain active operationally, but do not treat its native backtest as comparable research evidence.

### 6.2 Tradability target variants

Strict results:

- `buyin-mask`: approximately +1.48bp/day; fill 0.36/2 (18.2%).
- `tradable-label`: approximately +0.45bp/day; fill 0.05/2 (2.4%).

Decision: both failed the Top2 requirement of at least 1 filled name per day on average.

### 6.3 Open-label

Strict result:

- 467 days.
- Mean return: -33.03bp/day.
- Compound return: -79.58%.
- Maximum drawdown: -80.87%.
- Mean fill: 1.48/2 (74.2%).

Decision: fill passed; return and drawdown failed.

### 6.4 Open-buyin-mask ensemble

Strict result:

- 466 days.
- Mean return: -9.21bp/day.
- Compound return: -36.45%.
- Maximum drawdown: -43.39%.
- Mean fill: 1.98/2 (99.0%).

Leg diagnostics:

- Ridge: approximately -4.96bp/day.
- LightGBM: approximately -11.28bp/day.
- IC leg: approximately -9.65bp/day.

Decision: this target solved fill but not alpha. Ridge was the least-bad model and became the controller baseline.

## 7. Automatic pure-daily branch sequence

Controller: `quant/daily_optimization_pipeline.py`.

Frozen development gates:

- Mean return greater than zero.
- Mean Top2 filled names at least 1.
- Maximum drawdown no worse than -60%.
- At least 3 positive months.
- A passing branch routes only to independent reproduction, never directly to production.

The controller implements unique run IDs, isolated caches, explicit next-open environment, SHA256 state and artifact hashes, nonblocking cycle locks, process-group timeout cleanup, durable failures, and branch stopping rules. Some remote runs below were launched directly before the controller was fully deployed; their artifacts therefore require this consolidated archive to connect logs, predictions, and decisions.

### 7.1 Ridge-only open target screen

Task: `eo940v`.

Native diagnostic over 12 recent windows:

- 675,682 predictions.
- Total return: -27.27%.
- Annualized return: -30.11%.
- Sharpe: -1.71.
- Maximum drawdown: -33.82%.

Decision: failed. Native diagnostics were not used to override the strict open-buyin evidence.

### 7.2 Ridge 80% plus LightGBM 20%

Remote artifact:

`/app/quant_data/full_a_2018_wide/daily_auto_open_regularized_lgbm_20260810T050200Z_v3_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet`

Strict next-open Top2 result:

- 225 days.
- Mean return: -25.1973bp/day.
- Compound return: -50.93%.
- Maximum drawdown: -67.51%.
- Mean filled names: 1.991/2.
- Fill rate: 99.56%.
- Positive months: 4.

Decision: failed return and drawdown gates. Adding only 20% LightGBM substantially worsened Ridge, providing strong OOS evidence of nonlinear overfit or instability under this target.

Operational note: two earlier launch attempts failed before training because the remote trainer lacked open-target support and a paired feature-safety dependency. A shell `tee` initially masked one failure code. The trainer and dependency were synchronized, and subsequent commands returned the actual Python exit status.

### 7.3 Ridge plus ExtraTrees 20%

Remote artifact:

`/app/quant_data/full_a_2018_wide/daily_auto_open_extratrees_20260810T051942Z_v1_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet`

Strict next-open Top2 result:

- 225 days.
- Mean return: -27.2487bp/day.
- Compound return: -53.00%.
- Maximum drawdown: -67.74%.
- Mean filled names: 1.982/2.
- Fill rate: 99.11%.
- Positive months: 4.

Decision: failed return and drawdown gates.

### 7.4 Ridge 80% plus RandomForest 20%

Remote artifact:

`/app/quant_data/full_a_2018_wide/daily_auto_open_random_forest_20260810T055029Z_v1_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet`

Configuration:

- 120 trees.
- Maximum depth 12.
- Minimum leaf size 20.
- `max_features=0.7`.
- Bootstrap enabled.
- Deterministic seed 42.
- Time-decay-weighted deterministic cap of 300,000 training rows for the initial screen.

Strict next-open Top2 result:

- 225 days.
- Mean return: -28.9737bp/day.
- Compound return: -55.38%.
- Maximum drawdown: -62.84%.
- Mean filled names: 1.982/2.
- Fill rate: 99.11%.
- Positive months: 4.

Decision: failed return and drawdown gates. RandomForest did not replace LightGBM successfully and was worse than the Ridge baseline.

Limitation: the legacy trainer still trained LightGBM even when `lgbm_weight=0`; its predictions had zero contribution to the final score. The final score was 80% Ridge plus 20% RandomForest, so the result is valid, but training time was unnecessarily increased. A future confirmation runner should skip LightGBM entirely when its weight is zero.

## 8. Final comparison and decision

| Candidate | Mean return | Compound | Max drawdown | Mean filled | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Open-buyin Ridge baseline | -9.21bp/day | -36.45% | -43.39% | 1.98/2 | Failed return |
| Ridge 80% + LightGBM 20% | -25.20bp/day | -50.93% | -67.51% | 1.991/2 | Failed return/drawdown |
| Ridge + ExtraTrees 20% | -27.25bp/day | -53.00% | -67.74% | 1.982/2 | Failed return/drawdown |
| Ridge 80% + RandomForest 20% | -28.97bp/day | -55.38% | -62.84% | 1.982/2 | Failed return/drawdown |

Final route: `human_review_required`.

Human-review conclusions:

1. Fill is no longer the primary problem under `open-buyin-mask`; ranking quality is.
2. LightGBM, ExtraTrees, and RandomForest all degrade strict OOS performance relative to Ridge.
3. More nonlinear model complexity is not justified under the current target and feature set.
4. No branch qualifies for independent reproduction, forward shadow, active publication, or production replacement.
5. Active remains untouched. A new research cycle requires an explicitly approved change to target semantics, execution assumptions, or causal feature information.

## 9. Reproduction and artifact notes

Relevant source entry points:

- `quant/daily_optimization_pipeline.py`
- `quant/full_train_batched.py`
- `quant/model.py`
- `quant/factors/engineering.py`
- `intraday_1400/direct_return_experiment.py`
- `intraday_1400/target_redesign.py`
- `intraday_1400/target_redesign_backfill.py`
- `intraday_1400/daily_minute_enhancement.py`
- `intraday_1400/minute_feature_residualization.py`
- `intraday_1400/daily_h1_buyability_enhancement.py`
- `intraday_1400/daily_h1_executable_payoff.py`

Test evidence at archive time:

- `quant.test_daily_optimization_pipeline`: 12 passed before the RandomForest branch registration; branch-order assertions were then updated.
- `quant.test_model_expansion_experiment`: 130 passed during the pipeline hardening cycle.
- `intraday_1400.test_intraday_1400`: 54 passed.
- Combined focused regression: 196 passed before the RandomForest addition.
- RandomForest pipeline regression task `ga4mgm`: passed after the addition.

Remote production isolation:

- Research host: `192.140.180.77`.
- Training container: `a-scheduler-1`.
- Research outputs live under `/app/quant_data/full_a_2018_wide/` with unique prefixes.
- `SCHEDULER_DISABLED=1` was explicitly set for research launches.
- Realtime was not stopped by these experiments.
- No active publication command was run.
