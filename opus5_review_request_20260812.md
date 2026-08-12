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

## Review questions

1. Does `d93451c` correctly separate the full candidate pool hash from the selected factor hash?
2. Does strict rolling mode avoid fixed historical selection and avoid window-cache hits that could omit manifests?
3. Does the 176-day paired bootstrap and block-length sensitivity support only a directional, non-significant conclusion?
4. Is fixed-33 producer/window/label provenance the next required blocker to resolve before any promotion discussion?
5. Are there remaining P0/P1/P2 implementation risks?
