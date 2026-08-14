# Opus 5 Review Request: Rolling Ridge Provenance v2

## Review scope

- GitHub commit `60b0b8a`: reproducible quant audit source, C42-C44 implementation and validation evidence, and clean-clone verification.
- Earlier provenance commits: `ab8e684` and `d93451c`.
- Remote validation host: `root@192.140.180.77`.
- Production `/www/A` remained read-only.

## Remote validation

- Rolling module suite: `138/138` passed.
- Repository suite: `354/354` passed.
- Strict rolling Ridge-only OOS: exit code 0.
- LightGBM skipped in all 9 windows.

## Rolling v2 provenance

- 9 purged rolling windows.
- Candidate factors: 67 per window.
- Selected factors: 30 per window.
- Label: `tradable_ret_1d`.
- Purge span: 3 sessions.
- Selection manifest: 9 rows, 9 unique manifest hashes.
- Prediction rows: 115,165.
- `lgbm_pred` non-null rows: 0.
- Selection manifest SHA256: `5916e762f800b2c80c9695cc90ba12adb9214d417a9d44223c2014450a20fe18`.

## Metrics, explicitly not promotion evidence

Rolling v2:

- Total return: 49.84%
- Sharpe: 3.53
- Average turnover: 25.51%
- Max drawdown: -10.38%

Fixed 33-factor comparison:

- Total return: 38.23%
- Sharpe: 2.87
- Average turnover: 19.55%
- Max drawdown: -9.25%

The result-level pairing audit is now complete. Rolling v2 and fixed-33 have the same 176 return dates, the same 115,165 `(code,date)` prediction keys, the same 9 PIT universe manifests, the same `tradable_ret_1d` label, and the same 20bp cost contract.

Rolling minus fixed averages `+4.60bp/day` net and `+5.79bp/day` gross, with `+1.19bp/day` extra cost and `+0.05966` extra turnover. Circular moving-block bootstrap confidence intervals cross zero for block lengths 3, 5, 10 and 20. At block length 5, net and gross Holm-adjusted p-values are both `0.4112`; neither rejects zero gain.

Cost ablation does not support a production parameter change. `stride=2/3` materially reduces performance. Buffer 2 has little turnover effect; buffers 20/50 reduce turnover but also reduce return. The mechanical unsellable-holding risk had zero occurrences among 21 real cross-day holding overlaps, so no production code change was made for that issue.

Machine evidence: `audit_20260812_rolling_fixed_paired_validation.json`.

The correct conclusion remains directional evidence only, not confirmed alpha or promotion readiness. The fixed 33-factor selection still lacks producer/window/label provenance. The legacy `factor_selection_lh1000_cont.parquet` cannot be reconstructed from authoritative producer evidence, so no sidecar will be fabricated for it. The next control is a newly generated strict fixed-33 selection using the same 67-factor candidate pool, `tradable_ret_1d`, purge span 3, PIT universe and frozen first-window protocol as rolling v2; it must emit its own selection list, candidate/selected hashes, generator hash and manifest before being used for promotion inference.

## C11 relative baseline probe

A minimal remote real-data probe is now complete. The same strict prediction panel contains 177 dates and 115,165 prediction rows with `tradable_ret_1d` and `buyable_close`. The market control is same-panel equal-weight daily `tradable_ret_1d`, not an external index total-return series. Strict PIT industry history is `snapshots/sw_industry_history_pit.parquet`; applying `valid_from <= date < valid_to` and `available_from <= date` yields industry coverage on all 177 dates, 110,509 mapped rows (95.9571%) and 246 industries. The market equal-weight mean minus the PIT industry equal-weight mean is `+0.590481bp/day` over the common dates.

This is a coverage and baseline-definition probe only. It is not model excess-return evidence, does not establish alpha, and does not change promotion status. A research-only comparison module now exists at `quant/rolling_fixed_relative_comparison.py`; it requires equal rolling/fixed prediction keys, strict labels and buyability, return artifacts, and PIT industry history, and emits common-key hashes, coverage, absolute returns, market excess returns and PIT industry-relative returns. Its first remote execution was blocked because no fixed-33 prediction/return artifacts exist under `/www/A/research_runs`; the legacy fixed selection will not be substituted. The next step is to generate the strict fixed-33 control and then run this module once against both real artifacts.

## Review questions

1. Does `d93451c` correctly separate the full candidate pool hash from the selected factor hash?
2. Does strict rolling mode avoid fixed historical selection and avoid window-cache hits that could omit manifests?
3. Does the 176-day paired bootstrap and block-length sensitivity support only a directional, non-significant conclusion?
4. Is fixed-33 producer/window/label provenance the next required blocker to resolve before any promotion discussion?
5. Are there remaining P0/P1/P2 implementation risks?

## Current minute-validation review request (2026-08-14)

The current branch includes research-only minute runner changes that must be reviewed together with the code, not treated as minute alpha evidence:

- `intraday_1400/daily_minute_enhancement.py` now supports `--only-fold`, `--only-candidate`, `--gate-only`, and `--max-train-rows` for bounded engineering gates. The default full four-fold protocol remains unchanged.
- `_fit_daily_head()` writes a candidate model panel to an isolated parquet artifact, releases the parent panel, runs Ridge and LightGBM in separate subprocesses, writes prediction/metrics artifacts, verifies SHA256, and merges predictions in the parent.
- The current real remote runs have no valid minute return artifacts. The original and single-fold 12GiB runs were killed with `exit=137`; worker anon RSS was about 11.56GiB. A 20k training-row gate showed the same memory peak, so the primary hotspot may be screening/panel materialization or feature-matrix construction rather than training-row count.
- The research container memory limit was subsequently removed for diagnosis, but this does not create a valid result: host-wide OOM remains possible on the 16GiB machine, and the unrestricted run was still in progress when this request was written.
- Prepared provenance is complete for 44 months, labels cover 842,871 rows and 266 dates, and registered folds have OOS lengths 40/40/47/47. These are input-gate facts only.

Please review especially:

1. Whether `_model_worker()` has a complete and auditable subprocess boundary, including failure propagation and artifact cleanup.
2. Whether the research-only row/screening caps can ever affect the default formal protocol or be mistaken for OOS evidence.
3. Whether sampling in the gate screening path preserves the intended causal screening semantics; if not, require a separately named gate protocol.
4. Whether formal four-fold runs must use the original 100,000-row recipe and independently regenerate all artifacts.
5. Whether the remaining OOM is caused by `load_joined_prepared`, screening panel retention, or model matrix construction, and what bounded disk-first design should be used next.
6. Whether subprocess intermediate artifacts and their input hashes are sufficiently represented in the final protocol/provenance manifest.

Current minute conclusion: **no minute feature return conclusion exists**. Do not promote, publish, modify active manifests, change realtime, or enable scheduler based on any partial or capped run.

## Current gate acceleration and failure handoff (2026-08-14)

The current working tree contains research-only acceleration changes for the minute runner:

- `intraday_1400/pipeline.py` vectorizes the daily Spearman IC calculation while preserving the old equal-weight daily semantics; random-data equivalence was within `2.6e-18`, and a 24-feature benchmark improved from `3.275s` to `0.337s`.
- `intraday_1400/daily_minute_enhancement.py` limits causal screening to the two controls actually consumed by the gate, passes an explicit `align_controls=False` for incomplete variant subsets, and reuses the selected feature projection for execution-universe loading.
- `intraday_1400/fair_race_pipeline.py` keeps the default full-variant control alignment unchanged and adds the explicit opt-out only for the bounded research gate.

Remote evidence that needs review:

- v1 unrestricted gate eventually exited `137` after long single-core screening; no artifact was produced. This is separate from the older 12 GiB memcg OOM evidence.
- v2 exited `1` before screening because its actual container mount list lacked `/app/quant_data`; the same configuration with explicit mounts passed `provenance_ok 44`.
- v4 exited `1` in `align_control_feature_selections()` because the reduced variant subset violated an implicit assumption that all aligned variant keys exist.
- v5 exited `1` after screening, when execution-universe loading passed empty feature lists and `merge_prepared_frames()` rejected empty daily/asof/minute groups.
- v6 reached the `wf1 / daily_asof_baseline` candidate panel path but exited `1` at the same non-empty feature-group check. It produced no checkpoint, prediction, metrics, daily return, or report artifact. `OOMKilled=false`; host memory was healthy.
- After v6, further remote fixes and gate restarts were paused for this Opus5 review. The failed containers and isolated output directories are retained for diagnosis.

Review these specific contracts before any next run:

1. Define the contract for empty feature lists in `load_joined_prepared()` and `_build_panel()` separately for execution-universe and candidate-model panels; verify whether both paths should use the same projection.
2. Check per-month schema filtering and whether a valid selected feature can become an empty group after `intraday_columns` projection.
3. Review the new `align_controls` escape hatch and require tests for partial variant subsets without weakening the default full protocol.
4. Confirm that the optimized IC implementation is numerically identical to the previous daily Spearman definition under missing values, ties, constant columns, and insufficient dates.
5. Recommend a disk-first batch design for screening/panel loading before restarting real validation.

Tests currently available: full intraday regression `147/147`, minute enhancement targeted tests `13/13`, optimized IC equivalence, compileall, and `git diff --check`. These are engineering checks only; there is still no minute OOS return conclusion.
