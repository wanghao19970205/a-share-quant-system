from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from intraday_1400 import pipeline
from quant import model as quant_model
from intraday_1400.collector import _codes_file_manifest, _factor_fingerprint, _read_codes
from intraday_1400.evaluation import (
    _common_labels,
    _evaluate_negative_controls,
    _evaluate_source,
    _holm_adjust,
    _metrics as evaluation_metrics,
    _paired_block_bootstrap,
)
from intraday_1400.fair_race_pipeline import (
    DEFAULT_TRAINED_VARIANTS,
    RaceVariant,
    _build_return_fusion_frame,
    _build_safety_constrained_frame,
    _cap_training_panel,
    _cash_complete_targets,
    _causal_eligible_predictions,
    _exit_expected_value_frame,
    _evaluate_recipe_frames,
    _inner_window_spec,
    _select_inner_recipe,
    _top50_risk_constrained_frame,
    align_control_feature_selections,
    load_joined_prepared,
    merge_prepared_frames,
    panel_for_variant,
    rolling_window_specs,
    source_features_for_selection,
    variant_features,
)
from intraday_1400.adaptive_exit_replay import (
    build_fetch_plan,
    replay_selected_trades,
    summarize_replay,
)
from intraday_1400.adaptive_label_pipeline import (
    adaptive_records_to_labels,
    deterministic_code_sample,
    label_diagnostics,
)
from intraday_1400.auto_research import (
    _cycle_lock,
    claim_holdout,
    evaluate_holdout_report,
    initialize_research_state,
    run_frozen_holdout_cycle,
)
from intraday_1400 import (
    daily_h1_buyability_enhancement,
    daily_minute_enhancement,
    direct_return_experiment,
    minute_feature_residualization,
    target_redesign,
    target_redesign_backfill,
)
from intraday_1400.direct_return_experiment import (
    CALIBRATION_DAYS as DIRECT_CALIBRATION_DAYS,
    TARGETS as DIRECT_TARGETS,
    TOP_NS as DIRECT_TOP_NS,
    select_recipe as select_direct_return_recipe,
    validate_frozen_recipe as validate_direct_return_frozen_recipe,
    validate_holdout_labels as validate_direct_return_holdout_labels,
    validate_selection_labels as validate_direct_return_selection_labels,
)
from intraday_1400.dual_paper_evaluation import (    capture_daily_publication,
    evaluate_prediction_pair,
    initialize_dual_paper_evaluation,
    run_dual_paper_evaluation_once,
    run_retrospective_bootstrap,
)
from intraday_1400.features import add_asof_base_features, aggregate_many, aggregate_symbol, feature_columns
from intraday_1400.forward_evaluation import (
    _mature_forward_dates,
    available_common_dates,
    forward_status,
    initialize_forward_evaluation,
    run_forward_evaluation_once,
    validate_forward_manifest,
)
from intraday_1400.structural_combo_experiment import (
    _fit_liquidation_head,
    _selected_feature_window,
)
from intraday_1400.structural_combo_holdout import (
    cash_normalized_execution_records,
    first_holdout_dates,
    validated_holdout_dates,
)
from intraday_1400.structural_combo import (    apply_probability_calibrator,
    build_daily_execution_filter_scores,
    build_e0_e4_staged_scores,
    build_structural_combo_scores,
    e4_coverage_diagnostics,
    fit_probability_calibrator,
    shift_daily_prior_to_signal,
)
from intraday_1400.offline_race import (
    ExecutionConfig,
    common_prediction_universe,
    simulate_adaptive_exit_race,
    simulate_fixed_exit_race,
)
from intraday_1400.storage import (
    atomic_parquet_if_changed,
    read_parts,
    upsert_partition_part,
    write_partition_part,
)


def _day_bars(date: str, future_shift: float = 0.0) -> pd.DataFrame:
    morning = pd.date_range(f"{date} 09:30", f"{date} 11:25", freq="5min")
    afternoon = pd.date_range(f"{date} 13:00", f"{date} 14:55", freq="5min")
    timestamps = morning.append(afternoon)
    base = np.linspace(10.0, 10.48, len(timestamps))
    after_1400 = timestamps.time >= pd.Timestamp("2000-01-01 14:00").time()
    close = base + after_1400.astype(float) * future_shift
    return pd.DataFrame({
        "date": timestamps,
        "open": close - 0.01,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        "volume": np.arange(len(timestamps), dtype=float) + 100.0,
        "amount": (np.arange(len(timestamps), dtype=float) + 100.0) * close,
    })


class IntradayFeatureTest(unittest.TestCase):
    def test_codes_file_manifest_records_source_hash_and_effective_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "codes.txt"
            path.write_text("600001\n000001\n600001\n300001\n", encoding="utf-8")
            codes = _read_codes(str(path), limit=2)
            manifest = _codes_file_manifest(str(path), codes, limit=2)

        self.assertEqual(codes, ["600001", "000001"])
        self.assertEqual(manifest["source_code_count"], 3)
        self.assertEqual(manifest["effective_code_count"], 2)
        self.assertEqual(manifest["limit"], 2)
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertTrue(manifest["path"].endswith("codes.txt"))

    def test_features_do_not_read_bars_after_cutoff(self):
        normal = aggregate_symbol(_day_bars("2026-07-01"), "600000")
        changed_future = aggregate_symbol(_day_bars("2026-07-01", future_shift=20.0), "600000")
        minute_columns = [column for column in normal if column.startswith("m5_")]
        pd.testing.assert_series_equal(
            normal.iloc[0][minute_columns],
            changed_future.iloc[0][minute_columns],
            check_names=False,
        )
        self.assertEqual(normal.iloc[0]["close"], changed_future.iloc[0]["close"])
        self.assertNotEqual(
            normal.iloc[0]["label_entry_price_1450"],
            changed_future.iloc[0]["label_entry_price_1450"],
        )

    def test_qfq_vwap_uses_adjusted_bar_values(self):
        bars = _day_bars("2026-07-01")
        bars["bar_vwap_qfq"] = bars["close"] * 0.5
        result = aggregate_symbol(bars, "600000")
        cutoff = bars[bars["date"].dt.time <= pd.Timestamp("2000-01-01 13:55").time()]
        expected = float((cutoff["bar_vwap_qfq"] * cutoff["volume"]).sum() / cutoff["volume"].sum())
        self.assertAlmostEqual(float(result.iloc[0]["vwap"]), expected)

    def test_cutoff_uses_36_completed_bars(self):
        result = aggregate_symbol(_day_bars("2026-07-01"), "600000")
        self.assertEqual(int(result.iloc[0]["bar_count"]), 36)
        self.assertEqual(result.iloc[0]["cutoff_bar_time"], "13:55")
        self.assertTrue(bool(result.iloc[0]["is_complete"]))

    def test_signal_eligibility_supports_legacy_asof_cache(self):
        raw = pd.concat([
            aggregate_symbol(_day_bars("2026-07-01"), "600000"),
            aggregate_symbol(_day_bars("2026-07-02"), "600000"),
        ], ignore_index=True).drop(columns=[
            "signal_last_bar_high", "signal_last_bar_low", "signal_last_bar_volume",
        ])
        result = add_asof_base_features(raw)
        self.assertFalse(bool(result.iloc[0]["signal_eligible"]))
        self.assertTrue(bool(result.iloc[1]["signal_eligible"]))

    def test_signal_eligibility_does_not_filter_strong_1400_return(self):
        normal = _day_bars("2026-07-01")
        next_day = _day_bars("2026-07-02")
        locked = _day_bars("2026-07-03")
        cutoff = locked["date"].dt.time == pd.Timestamp("2000-01-01 13:55").time()
        for column in ("open", "high", "low", "close"):
            locked[column] += 0.6
        locked.loc[cutoff, "high"] = locked.loc[cutoff, "close"]
        locked.loc[cutoff, "low"] = locked.loc[cutoff, "close"]
        raw = aggregate_symbol(pd.concat([normal, next_day, locked], ignore_index=True), "600000")
        result = add_asof_base_features(raw)
        self.assertFalse(bool(result.iloc[0]["signal_eligible"]))
        self.assertTrue(bool(result.iloc[1]["signal_eligible"]))
        self.assertTrue(bool(result.iloc[2]["signal_locked_up_1400"]))
        self.assertTrue(bool(result.iloc[2]["signal_eligible"]))

    def test_spawn_workers_match_serial_aggregation(self):
        items = [("600000", _day_bars("2026-07-01")), ("600001", _day_bars("2026-07-01"))]
        serial = aggregate_many(items, workers=1).sort_values("code").reset_index(drop=True)
        parallel = aggregate_many(items, workers=2).sort_values("code").reset_index(drop=True)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_base_features_use_same_time_across_dates(self):
        raw = pd.concat([
            aggregate_symbol(_day_bars(f"2026-07-{day:02d}"), "600000")
            for day in range(1, 22)
        ], ignore_index=True)
        raw = raw.drop(columns=[column for column in raw if column.startswith("label_")])
        enriched = add_asof_base_features(raw)
        self.assertIn("ret_20d", enriched)
        self.assertIn("m5_volume_vs_20d_median", enriched)
        self.assertIn("risk_gap_event_count_20", enriched)
        self.assertIn("risk_near_limit_down_count_20", enriched)
        self.assertIn("risk_amount_vs_median_20", enriched)
        self.assertTrue(np.isfinite(enriched.iloc[-1]["ret_20d"]))
        # The current day's amount must not enter its own historical median baseline.
        expected_amount_risk = raw.iloc[-1]["amount"] / raw.iloc[-21:-1]["amount"].median() - 1.0
        self.assertAlmostEqual(
            float(enriched.iloc[-1]["risk_amount_vs_median_20"]),
            float(expected_amount_risk),
        )
        self.assertNotIn("close", feature_columns(enriched))

    def test_partition_part_round_trip(self):
        frame = aggregate_symbol(_day_bars("2026-07-01"), "600000")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_partition_part(frame, root, "2026-07", "part-a")
            self.assertTrue(path.exists())
            loaded = read_parts(root)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]["code"], "600000")

    def test_prepared_loader_projects_filters_and_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for month, dates in {
                "2026-06": ["2026-06-29", "2026-06-30"],
                "2026-07": ["2026-07-01", "2026-07-02"],
            }.items():
                pd.DataFrame({
                    "code": ["600000", "600001"],
                    "date": pd.to_datetime(dates),
                    "target_net_ret_t1": [0.01, 0.03],
                    "entry_buyable": [True, True],
                    "selected_feature": [1.0, 2.0],
                    "unused_feature": [3.0, 4.0],
                }).to_parquet(root / f"{month}.parquet", index=False)
            with mock.patch.object(pipeline.config, "PREPARED_DIR", root):
                loaded = pipeline._load_prepared(
                    columns=["entry_buyable", "selected_feature"],
                    end_date="2026-07-01",
                    max_rows=3,
                )
                cutoff_month = pipeline._load_prepared(
                    max_months=1,
                    columns=["selected_feature"],
                    end_date="2026-06-30",
                    exclude_dates=["2026-06-29"],
                )
        self.assertLessEqual(len(loaded), 3)
        self.assertLessEqual(loaded["date"].max(), pd.Timestamp("2026-07-01"))
        self.assertIn("target_excess_ret_t1", loaded)
        self.assertNotIn("unused_feature", loaded)
        self.assertEqual(cutoff_month["date"].dt.strftime("%Y-%m").unique().tolist(), ["2026-06"])
        self.assertEqual(cutoff_month["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-06-30"])

    def test_rolling_validation_rejects_nonpositive_inputs(self):
        for kwargs in ({"windows": 0}, {"valid_days": 0}, {"top_n": 0}):
            with self.assertRaises(ValueError):
                pipeline.rolling_validate(**kwargs)

    def test_parquet_write_is_content_idempotent(self):
        frame = pd.DataFrame({
            "code": ["600000"],
            "date": pd.to_datetime(["2026-07-01"]),
            "value": [1.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "part.parquet"
            self.assertTrue(atomic_parquet_if_changed(frame, path))
            first_mtime = path.stat().st_mtime_ns
            self.assertFalse(atomic_parquet_if_changed(frame.copy(), path))
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)
            changed = frame.assign(value=2.0)
            self.assertTrue(atomic_parquet_if_changed(changed, path))

    def test_partition_upsert_keeps_unchanged_file_mtime(self):
        frame = pd.DataFrame({
            "code": ["600000"],
            "date": pd.to_datetime(["2026-07-01"]),
            "schema_version": [2],
            "value": [1.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = upsert_partition_part(
                frame, root, "2026-07", "part-a", ["code", "date", "schema_version"],
            )
            first_mtime = path.stat().st_mtime_ns
            upsert_partition_part(
                frame.copy(), root, "2026-07", "part-a", ["code", "date", "schema_version"],
            )
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)

    def test_daily_topn_metrics_weight_dates_equally(self):
        selected = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01"] * 2 + ["2026-07-02"] * 2),
            "code": ["600000", "600001", "600000", "600001"],
            "gross_return": [0.01, 0.03, -0.02, 0.0],
        })
        metrics = evaluation_metrics(selected, "gross_return", 0.002)
        self.assertAlmostEqual(metrics["mean_return"], 0.003)
        self.assertEqual(metrics["days"], 2)

    def test_daily_topn_metrics_penalize_selected_unsellable_names(self):
        selected = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "code": ["600000", "600001"],
            "gross_return": [np.nan, 0.02],
        })
        with mock.patch.dict("os.environ", {"INTRADAY_1400_UNSELLABLE_RETURN": "-0.10"}):
            metrics = evaluation_metrics(selected, "gross_return", 0.002)
        self.assertAlmostEqual(metrics["mean_return"], -0.041)
        self.assertEqual(metrics["missing_targets"], 1)
        self.assertEqual(metrics["mean_names"], 2.0)

    def test_daily_topn_metrics_treat_unbuyable_slot_as_cash(self):
        selected = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01"]),
            "code": ["600000"],
            "gross_return": [np.nan],
            "entry_buyable": [False],
            "target_outcome_observed_t1": [False],
        })
        metrics = evaluation_metrics(selected, "gross_return", 0.002)
        self.assertAlmostEqual(metrics["mean_return"], 0.0)
        self.assertEqual(metrics["missing_targets"], 0)
        self.assertEqual(metrics["immature_targets"], 0)

    def test_daily_topn_metrics_exclude_immature_targets_from_penalty(self):
        selected = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "code": ["600000", "600001"],
            "gross_return": [np.nan, 0.02],
            "target_outcome_observed_t1": [False, True],
        })
        with mock.patch.dict("os.environ", {"INTRADAY_1400_UNSELLABLE_RETURN": "-0.10"}):
            metrics = evaluation_metrics(selected, "gross_return", 0.002)
        self.assertAlmostEqual(metrics["mean_return"], 0.018)
        self.assertEqual(metrics["missing_targets"], 0)
        self.assertEqual(metrics["immature_targets"], 1)
        self.assertEqual(metrics["mean_names"], 1.0)

    def test_daily_topn_metrics_scale_sharpe_for_rebalance_stride(self):
        selected = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=4, freq="D"),
            "code": ["600000"] * 4,
            "gross_return": [0.012, 0.022, 0.032, 0.042],
        })
        metrics = evaluation_metrics(selected, "gross_return", 0.002, rebalance_stride=2)
        expected_daily = pd.Series([0.01, 0.03])
        expected = expected_daily.mean() / expected_daily.std() * np.sqrt(252 / 2)
        self.assertEqual(metrics["days"], 2)
        self.assertAlmostEqual(metrics["sharpe"], expected)

    def test_rebalance_stride_keeps_calendar_phase_when_middle_date_is_immature(self):
        selected = pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=4, freq="D"),
            "code": ["600000"] * 4,
            "gross_return": [0.012, np.nan, 0.032, 0.102],
            "target_outcome_observed_t1": [True, False, True, True],
        })
        metrics = evaluation_metrics(selected, "gross_return", 0.002, rebalance_stride=2)
        self.assertEqual(metrics["days"], 2)
        self.assertAlmostEqual(metrics["mean_return"], 0.02)
        self.assertEqual(metrics["immature_targets"], 0)

    def test_paired_block_bootstrap_aligns_dates_and_detects_gain(self):
        model_index = pd.date_range("2026-01-01", periods=14, freq="D")
        baseline_index = pd.date_range("2026-01-03", periods=14, freq="D")
        model = pd.Series(0.01, index=model_index)
        baseline = pd.Series(0.0, index=baseline_index)
        first = _paired_block_bootstrap(model, baseline, samples=199, block_length=3, seed=11)
        second = _paired_block_bootstrap(model, baseline, samples=199, block_length=3, seed=11)
        self.assertEqual(first, second)
        self.assertTrue(first["available"])
        self.assertEqual(first["days"], 12)
        self.assertAlmostEqual(first["paired_mean_gain"], 0.01)
        np.testing.assert_allclose(first["ci95"], [0.01, 0.01])
        self.assertAlmostEqual(first["p_value_one_sided"], 0.005)

    def test_paired_block_bootstrap_does_not_reject_null_gain(self):
        index = pd.date_range("2026-01-01", periods=12, freq="D")
        values = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
        result = _paired_block_bootstrap(values, values, samples=99, block_length=4, seed=17)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["paired_mean_gain"], 0.0)
        self.assertAlmostEqual(result["p_value_one_sided"], 1.0)

    def test_holm_adjustment_preserves_input_order_and_step_down_monotonicity(self):
        raw = [0.01, 0.04, 0.03]
        adjusted = _holm_adjust(raw)
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
        ordered = sorted(zip(raw, adjusted))
        self.assertEqual([value for _, value in ordered], sorted(value for _, value in ordered))

    def test_evaluate_source_ranks_before_unsellable_target_handling(self):
        date = pd.Timestamp("2026-07-01")
        labels = pd.DataFrame({
            "code": ["600000", "600001"],
            "date": [date, date],
            "entry_buyable": [True, True],
            "target_net_ret_t1": [np.nan, 0.02],
            "gross__target_net_ret_t1": [np.nan, 0.022],
        })
        predictions = pd.DataFrame({
            "code": ["600000", "600001"],
            "date": [date, date],
            "score": [2.0, 1.0],
        })
        metadata = {"gross_columns": {"target_net_ret_t1": "gross__target_net_ret_t1"}}
        with mock.patch.dict("os.environ", {"INTRADAY_1400_UNSELLABLE_RETURN": "-0.10"}):
            report = _evaluate_source(
                labels, predictions, Path("predictions.parquet"), True, (1,), (0.002,), metadata,
            )
        metrics = report["top"]["1"]["target_net_ret_t1"]["cost_20bp"]
        self.assertAlmostEqual(metrics["mean_return"], -0.10)
        self.assertEqual(metrics["missing_targets"], 1)
        self.assertEqual(metrics["mean_names"], 1.0)

    def test_negative_controls_are_deterministic_and_include_market_leg(self):
        dates = pd.to_datetime(["2026-07-01"] * 4 + ["2026-07-02"] * 4)
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002", "600003"] * 2,
            "date": dates,
            "entry_buyable": [True] * 8,
            "gross__target_net_ret_t1": [0.01, 0.02, 0.03, 0.04, -0.01, 0.0, 0.01, 0.02],
        })
        predictions = pd.DataFrame({
            "code": labels["code"],
            "date": dates,
            "score": [4.0, 3.0, 2.0, 1.0] * 2,
        })
        metadata = {"gross_columns": {"target_net_ret_t1": "gross__target_net_ret_t1"}}
        first = _evaluate_negative_controls(
            labels, predictions, (1, 2), (0.002,), metadata, samples=5, seed=7,
        )
        second = _evaluate_negative_controls(
            labels, predictions, (1, 2), (0.002,), metadata, samples=5, seed=7,
        )
        parallel = _evaluate_negative_controls(
            labels, predictions, (1, 2), (0.002,), metadata, samples=5, seed=7, workers=2,
        )
        strided = _evaluate_negative_controls(
            labels, predictions, (1, 2), (0.002,), metadata,
            samples=5, seed=7, workers=2, rebalance_stride=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["top"], parallel["top"])
        market = first["market_equal_weight"]["target_net_ret_t1"]["cost_20bp"]
        self.assertAlmostEqual(market["mean_return"], 0.013)
        strided_market = strided["market_equal_weight"]["target_net_ret_t1"]["cost_20bp"]
        self.assertEqual(strided["rebalance_stride"], 2)
        self.assertEqual(strided_market["days"], 1)
        self.assertAlmostEqual(strided_market["mean_return"], 0.023)
        for top_n in ("1", "2"):
            controls = first["top"][top_n]["target_net_ret_t1"]["cost_20bp"]
            self.assertEqual(controls["random_topn"]["samples"], 5)
            self.assertEqual(controls["score_shuffle"]["samples"], 5)

    def test_negative_controls_report_paired_holm_family(self):
        dates = pd.date_range("2026-01-01", periods=12, freq="D").repeat(4)
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002", "600003"] * 12,
            "date": dates,
            "entry_buyable": [True] * 48,
            "gross__target_net_ret_t1": [0.04, 0.02, 0.0, -0.02] * 12,
        })
        predictions = pd.DataFrame({
            "code": labels["code"],
            "date": dates,
            "score": [4.0, 3.0, 2.0, 1.0] * 12,
        })
        metadata = {"gross_columns": {"target_net_ret_t1": "gross__target_net_ret_t1"}}
        serial = _evaluate_negative_controls(
            labels, predictions, (1,), (0.002,), metadata,
            samples=7, seed=19, workers=1, bootstrap_samples=99, block_length=3,
        )
        parallel = _evaluate_negative_controls(
            labels, predictions, (1,), (0.002,), metadata,
            samples=7, seed=19, workers=2, bootstrap_samples=99, block_length=3,
        )
        self.assertEqual(serial["top"], parallel["top"])
        comparisons = serial["top"]["1"]["target_net_ret_t1"]["cost_20bp"][
            "paired_comparisons"
        ]
        self.assertEqual(set(comparisons), {"market_equal_weight", "random_topn", "score_shuffle"})
        for comparison in comparisons.values():
            self.assertTrue(comparison["available"])
            self.assertEqual(comparison["days"], 12)
            self.assertEqual(comparison["holm_family_size"], 3)
            self.assertGreaterEqual(comparison["p_value_holm"], comparison["p_value_one_sided"])

    def test_common_universe_uses_identical_date_code_keys(self):
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "target_net_ret_t1": [0.01, 0.02, 0.03],
        })
        predictions = {
            "base": pd.DataFrame({
                "code": ["600000", "600001"],
                "date": pd.to_datetime(["2026-07-01"] * 2),
                "score": [1.0, 2.0],
            }),
            "plus": pd.DataFrame({
                "code": ["600001", "600002"],
                "date": pd.to_datetime(["2026-07-01"] * 2),
                "score": [3.0, 4.0],
            }),
        }
        common = _common_labels(labels, predictions)
        self.assertEqual(common["code"].tolist(), ["600001"])

    def test_factor_fingerprint_ignores_appended_unchanged_days(self):
        short = pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-07-01", "2026-07-02"]))
        appended = pd.Series([1.0, 1.0, 1.0], index=pd.to_datetime([
            "2026-07-01", "2026-07-02", "2026-07-03",
        ]))
        changed = pd.Series([1.0, 1.0, 1.2], index=pd.to_datetime([
            "2026-07-01", "2026-07-02", "2026-07-03",
        ]))
        self.assertEqual(_factor_fingerprint(short), _factor_fingerprint(appended))
        self.assertNotEqual(_factor_fingerprint(short), _factor_fingerprint(changed))

    def test_nonprice_features_are_lagged_one_session(self):
        frame = pd.DataFrame({
            "code": ["600000", "600000"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "m5_ret_30m": [0.0, 0.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "test_parts" / "prepared_monthly"
            prepared.mkdir(parents=True)
            pd.DataFrame({
                "code": ["600000"], "date": pd.to_datetime(["2026-06-30"]),
                "income_signal": [1.0],
            }).to_parquet(prepared / "2026-06.parquet", index=False)
            pd.DataFrame({
                "code": ["600000", "600000"],
                "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
                "income_signal": [2.0, 999.0],
            }).to_parquet(prepared / "2026-07.parquet", index=False)
            with mock.patch.dict("os.environ", {"QUANT_DATA_DIR": str(root)}):
                result = pipeline._merge_lagged_nonprice(frame, "2026-07")
        self.assertEqual(result["income_signal"].tolist(), [1.0, 2.0])

    def test_market_context_uses_same_date_cross_section(self):
        frame = pd.DataFrame({
            "code": ["600000", "600001", "600000", "600001"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"]),
            "m5_ret_30m": [0.01, 0.03, -0.02, 0.02],
        })
        with mock.patch.dict("os.environ", {"SNAPSHOT_DIR": "/missing"}):
            result = pipeline._add_market_industry_context(frame)
        self.assertAlmostEqual(float(result.iloc[0]["m5_ret_30m_market_excess"]), -0.01)
        self.assertAlmostEqual(float(result.iloc[2]["m5_ret_30m_market_excess"]), -0.02)

    def test_cross_sectional_target_removes_daily_market_mean(self):
        frame = pd.DataFrame({
            "code": ["600000", "600001", "600000", "600001"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02"]),
            "target_net_ret_t1": [-0.03, -0.01, 0.01, 0.05],
            "target_penalty_net_ret_t1": [-0.10, -0.01, 0.01, -0.10],
        })
        result = pipeline._add_cross_sectional_training_target(frame)
        for column in ("target_excess_ret_t1", "target_penalty_excess_ret_t1"):
            daily_mean = result.groupby("date")[column].mean()
            self.assertTrue(np.allclose(daily_mean.to_numpy(), 0.0))
            self.assertNotIn(column, feature_columns(result))
        for _, day in result.groupby("date"):
            self.assertEqual(
                day["target_net_ret_t1"].rank().tolist(),
                day["target_excess_ret_t1"].rank().tolist(),
            )
            self.assertEqual(
                day["target_penalty_net_ret_t1"].rank().tolist(),
                day["target_penalty_excess_ret_t1"].rank().tolist(),
            )

    def test_training_target_mode_rejects_unknown_value(self):
        self.assertEqual(
            pipeline._training_target_columns("penalty_aware"),
            ("target_penalty_excess_ret_t1", "target_penalty_net_ret_t1"),
        )
        with self.assertRaises(ValueError):
            pipeline._training_target_columns("unknown")

    def test_close_baseline_ranks_before_unsellable_target_handling(self):
        date = pd.Timestamp("2026-07-01")
        panel = pd.DataFrame({
            "code": ["600000", "600001"],
            "date": [date, date],
            "target_net_ret_t1": [np.nan, 0.02],
            "entry_buyable": [True, True],
        })
        predictions = pd.DataFrame({
            "code": ["600000", "600001"],
            "date": [date, date],
            "pred": [2.0, 1.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            predictions.to_parquet(Path(temporary) / "active_quant_short_predictions.parquet")
            with mock.patch.dict("os.environ", {
                "QUANT_DATA_DIR": temporary,
                "INTRADAY_1400_UNSELLABLE_RETURN": "-0.10",
            }):
                metrics = pipeline._evaluate_close_baseline(
                    panel, "2026-06-30", "2026-07-01", top_n=1,
                )
        self.assertAlmostEqual(metrics["mean_net_return"], -0.10)
        self.assertEqual(metrics["missing_targets"], 1)
        self.assertEqual(metrics["mean_names"], 1.0)

    def test_realized_leg_metrics_use_actual_net_return_without_unsellable_backfill(self):
        panel = pd.DataFrame({
            "code": ["600000", "600001", "600002", "600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3 + ["2026-07-02"] * 3),
            "target_net_ret_t1": [np.nan, 0.03, -0.01, -0.02, 0.04, 0.01],
            "target_excess_ret_t1": [100.0, 50.0, -100.0, 100.0, -100.0, 50.0],
            "entry_buyable": [True] * 6,
        })
        predictions = pd.DataFrame({
            "code": panel["code"], "date": panel["date"], "pred": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        })
        with mock.patch.dict("os.environ", {"INTRADAY_1400_UNSELLABLE_RETURN": "-0.10"}):
            metrics = pipeline._realized_leg_metrics(
                panel, predictions, "2026-06-30", "2026-07-02", top_n=1,
            )
        self.assertAlmostEqual(metrics["realized_top1_mean_net_return"], -0.06)
        self.assertEqual(metrics["realized_days"], 2)
        self.assertEqual(metrics["realized_missing_targets"], 1)
        self.assertEqual(metrics["realized_mean_names"], 1.0)

    def test_variant_proxy_labels_use_stop_first(self):
        frame = pd.DataFrame({
            "code": ["600000", "600000"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "high": [10.2, 10.2], "low": [9.8, 9.8], "close": [10.0, 10.0],
            "atr14": [0.2, 0.2], "m5_price_vwap_gap": [0.0, 0.0],
            "label_entry_price_1450": [10.0, 10.5],
            "target_net_ret_t1": [0.0, np.nan],
            "target_mfe_t1": [0.10, np.nan],
            "target_mae_t1": [-0.06, np.nan],
            "entry_buyable": [True, True],
        })
        result = pipeline._add_variant_proxy_labels(frame)
        self.assertAlmostEqual(float(result.iloc[0]["target_v1_proxy_net"]), -0.052)
        self.assertAlmostEqual(float(result.iloc[0]["target_v3_proxy_net"]), -0.042)
        self.assertTrue(bool(result.iloc[0]["target_proxy_intrabar_ambiguous"]))

    def test_t1_label_uses_global_calendar_when_partition_misses_session(self):
        missing_session = pd.DataFrame({
            "code": ["600000", "600000", "600000"],
            "date": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-03"]),
            "label_entry_price_1450": [10.0, 10.0, 11.0],
            "label_entry_high_1450": [10.01, 10.01, 11.01],
            "label_entry_low_1450": [9.99, 9.99, 10.99],
            "label_entry_volume_1450": [100.0, 100.0, 100.0],
            "label_close_1455": [10.0, 10.0, 11.0],
            "label_high_to_1450": [10.1, 10.1, 11.1],
            "label_low_to_1450": [9.9, 9.9, 10.9],
            "label_entry_bar_present": [True, True, True],
        })
        calendar_partition = missing_session.iloc[[0]].copy()
        calendar_partition["code"] = "600001"
        calendar_partition["date"] = pd.Timestamp("2026-07-02")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_partition_part(missing_session, root, "2026-07", "part-a")
            write_partition_part(calendar_partition, root, "2026-07", "part-b")
            with mock.patch.object(pipeline.config, "LABEL_DIR", root):
                result = pipeline._read_matching_labels("part-a.parquet")
        first = result[result["date"] == pd.Timestamp("2026-07-01")].iloc[0]
        latest = result[result["date"] == pd.Timestamp("2026-07-03")].iloc[0]
        self.assertTrue(pd.isna(first["target_net_ret_t1"]))
        self.assertTrue(bool(first["target_outcome_observed_t1"]))
        self.assertAlmostEqual(float(first["target_penalty_net_ret_t1"]), -0.10)
        self.assertAlmostEqual(float(first["target_cash_net_ret_t1"]), -0.10)
        self.assertEqual(float(first["target_entry_fill"]), 1.0)
        self.assertEqual(float(first["target_exit_missing_day_t1"]), 1.0)
        self.assertEqual(float(first["target_exit_sellable_t1"]), 0.0)
        self.assertFalse(bool(latest["target_outcome_observed_t1"]))
        self.assertTrue(pd.isna(latest["target_penalty_net_ret_t1"]))
        self.assertEqual(float(latest["target_cash_net_ret_t1"]), 0.0)
        self.assertEqual(float(latest["target_entry_fill"]), 0.0)
        self.assertTrue(pd.isna(latest["target_exit_sellable_t1"]))
        self.assertTrue(pd.isna(latest["target_exit_missing_day_t1"]))

    def test_t1_outcome_waits_for_completed_next_session_labels(self):
        labels = pd.DataFrame({
            "code": ["600000", "600000"],
            "date": pd.to_datetime(["2026-06-30", "2026-07-01"]),
            "label_entry_price_1450": [10.0, 10.1],
            "label_entry_high_1450": [10.01, 10.11],
            "label_entry_low_1450": [9.99, 10.09],
            "label_entry_volume_1450": [100.0, 100.0],
            "label_close_1455": [10.0, 10.1],
            "label_high_to_1450": [10.1, 10.2],
            "label_low_to_1450": [9.9, 10.0],
            "label_entry_bar_present": [True, True],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_partition_part(labels, root, "2026-07", "part-a")
            with mock.patch.object(pipeline.config, "LABEL_DIR", root):
                result = pipeline._read_matching_labels(
                    "part-a.parquet",
                    trade_dates=list(pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"])),
                    observed_trade_dates=list(pd.to_datetime(["2026-06-30", "2026-07-01"])),
                )
        latest = result[result["date"] == pd.Timestamp("2026-07-01")].iloc[0]
        self.assertTrue(bool(latest["entry_buyable"]))
        self.assertFalse(bool(latest["target_outcome_observed_t1"]))
        self.assertTrue(pd.isna(latest["target_penalty_net_ret_t1"]))

    def test_exit_causes_distinguish_zero_volume_and_flat_limit_down(self):
        dates = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"] * 2)
        labels = pd.DataFrame({
            "code": ["600000"] * 3 + ["600001"] * 3,
            "date": dates,
            "label_entry_price_1450": [10.0, 10.0, 10.1, 20.0, 20.0, 19.0],
            "label_entry_high_1450": [10.01, 10.01, 10.11, 20.01, 20.01, 19.0],
            "label_entry_low_1450": [9.99, 9.99, 10.09, 19.99, 19.99, 19.0],
            "label_entry_volume_1450": [100.0, 100.0, 0.0, 100.0, 100.0, 100.0],
            "label_close_1455": [10.0, 10.0, 10.1, 20.0, 20.0, 19.0],
            "label_high_to_1450": [10.1, 10.1, 10.2, 20.1, 20.1, 19.1],
            "label_low_to_1450": [9.9, 9.9, 10.0, 19.9, 19.9, 18.9],
            "label_entry_bar_present": [True] * 6,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_partition_part(labels, root, "2026-07", "part-a")
            with mock.patch.object(pipeline.config, "LABEL_DIR", root):
                result = pipeline._read_matching_labels("part-a.parquet")
        zero = result[(result["code"] == "600000") & (result["date"] == pd.Timestamp("2026-07-01"))].iloc[0]
        locked = result[(result["code"] == "600001") & (result["date"] == pd.Timestamp("2026-07-01"))].iloc[0]
        self.assertEqual(float(zero["target_exit_zero_volume_t1"]), 1.0)
        self.assertEqual(float(zero["target_exit_sellable_t1"]), 0.0)
        self.assertAlmostEqual(float(zero["target_penalty_net_ret_t1"]), -0.10)
        self.assertEqual(float(locked["target_exit_flat_limit_down_t1"]), 1.0)
        self.assertEqual(float(locked["target_exit_sellable_t1"]), 0.0)

    def test_t1_label_does_not_skip_suspended_market_day(self):
        labels = pd.DataFrame({
            "code": ["600000", "600000", "600001", "600001", "600001"],
            "date": pd.to_datetime([
                "2026-07-01", "2026-07-03",
                "2026-07-01", "2026-07-02", "2026-07-03",
            ]),
            "label_entry_price_1450": [10.0, 11.0, 20.0, 20.2, 20.4],
            "label_entry_high_1450": [10.01, 11.01, 20.01, 20.21, 20.41],
            "label_entry_low_1450": [9.99, 10.99, 19.99, 20.19, 20.39],
            "label_entry_volume_1450": [100.0] * 5,
            "label_close_1455": [10.0, 11.0, 20.0, 20.2, 20.4],
            "label_high_to_1450": [10.1, 11.1, 20.1, 20.3, 20.5],
            "label_low_to_1450": [9.9, 10.9, 19.9, 20.1, 20.3],
            "label_entry_bar_present": [True] * 5,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_partition_part(labels, root, "2026-07", "part-a")
            with mock.patch.object(pipeline.config, "LABEL_DIR", root):
                result = pipeline._read_matching_labels("part-a.parquet")
        suspended = result[(result["code"] == "600000") & (result["date"] == pd.Timestamp("2026-07-01"))]
        self.assertTrue(pd.isna(suspended.iloc[0]["target_net_ret_t1"]))
        self.assertTrue(bool(suspended.iloc[0]["target_outcome_observed_t1"]))
        self.assertFalse(bool(suspended.iloc[0]["entry_buyable"]))
        self.assertTrue(pd.isna(suspended.iloc[0]["target_penalty_net_ret_t1"]))
        self.assertEqual(float(suspended.iloc[0]["target_cash_net_ret_t1"]), 0.0)
        self.assertEqual(float(suspended.iloc[0]["target_entry_fill"]), 0.0)
        normal = result[(result["code"] == "600001") & (result["date"] == pd.Timestamp("2026-07-02"))]
        expected = 20.4 / 20.2 - 1.0 - 0.002
        self.assertAlmostEqual(float(normal.iloc[0]["target_net_ret_t1"]), expected)
        self.assertAlmostEqual(float(normal.iloc[0]["target_cash_net_ret_t1"]), expected)


class FairRacePipelineTest(unittest.TestCase):
    @staticmethod
    def _prepared_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02"])
        daily = pd.DataFrame({
            "code": ["600000", "600001", "600000"],
            "date": dates,
            "target_ret_1d": [0.01, 0.03, -0.02],
            "ret_5d": [0.1, 0.2, 0.3],
            "volatility_10": [1.0, 2.0, 3.0],
        })
        intraday = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02"]),
            "entry_buyable": [True, True, True],
            "signal_eligible": [True, False, True],
            "target_net_ret_t1": [0.02, np.nan, 0.01],
            "target_penalty_net_ret_t1": [0.02, -0.10, 0.01],
            "target_cash_net_ret_t1": [0.02, -0.10, 0.01],
            "target_entry_fill": [1.0, 1.0, 1.0],
            "target_outcome_observed_t1": [True, True, True],
            "ret_5d": [0.4, 0.5, 0.6],
            "m5_ret_30m": [1.1, 1.2, 1.3],
        })
        return daily, intraday

    def test_prepared_merge_uses_common_keys_and_namespaces_features(self):
        daily, intraday = self._prepared_frames()
        panel, groups = merge_prepared_frames(daily, intraday)
        self.assertEqual(panel[["date", "code"]].values.tolist(), [
            [pd.Timestamp("2026-07-01"), "600000"],
            [pd.Timestamp("2026-07-01"), "600001"],
        ])
        self.assertIn("daily__ret_5d", groups["daily"])
        self.assertIn("asof__ret_5d", groups["asof"])
        self.assertEqual(groups["minute"], ["minute__m5_ret_30m"])
        self.assertEqual(groups["daily_matched"], ["daily__ret_5d"])
        self.assertEqual(groups["asof_matched"], ["asof__ret_5d"])
        self.assertTrue(bool(panel.iloc[0]["signal_eligible"]))
        self.assertFalse(bool(panel.iloc[1]["signal_eligible"]))
        self.assertNotIn("target_penalty_net_ret_t1", sum(groups.values(), []))

    def test_recipe_evaluation_uses_one_common_prediction_universe(self):
        dates = pd.to_datetime(["2026-07-01", "2026-07-02"])
        predictions = {
            "recipe_a": pd.DataFrame({
                "code": ["600000", "600000"],
                "date": dates,
                "score": [2.0, 1.0],
            }),
            "recipe_b": pd.DataFrame({
                "code": ["600000"],
                "date": dates[1:],
                "score": [1.0],
            }),
        }
        labels = pd.DataFrame({
            "code": ["600000", "600000"],
            "date": dates,
            "entry_buyable": [True, True],
            "target_outcome_observed_t1": [True, True],
            "target_net_ret_t1": [0.01, 0.02],
        })

        records, metrics = _evaluate_recipe_frames(predictions, labels, top_n=1)

        self.assertEqual(set(records["signal_date"]), {dates[1]})
        self.assertEqual(metrics["recipe_a"]["selected"], 1)
        self.assertEqual(metrics["recipe_b"]["selected"], 1)

    def test_binary_classifier_returns_sellable_probability(self):
        rows = []
        dates = pd.bdate_range("2026-06-01", periods=12)
        for date in dates:
            for index in range(20):
                rows.append({
                    "code": f"{600000 + index:06d}",
                    "date": date,
                    "risk_feature": float(index),
                    "target_exit_sellable_t1": float(index >= 2),
                })
        panel = pd.DataFrame(rows)
        result = quant_model.train_binary_classifier(
            panel,
            ["risk_feature"],
            "target_exit_sellable_t1",
            "ridge",
            train_end=str(dates[7].date()),
            valid_end=str(dates[-1].date()),
            predict_start=str(dates[8].date()),
            minority_weight=10.0,
            n_jobs=1,
        )
        self.assertTrue(result.ok, result.message)
        self.assertTrue(result.predictions["pred"].between(0.0, 1.0).all())
        self.assertIn("risk_pr_auc", result.metrics)

    def test_training_cap_is_deterministic_and_keeps_full_validation(self):
        panel = pd.DataFrame({
            "code": [f"{600000 + i:06d}" for i in range(12)],
            "date": pd.to_datetime(["2026-07-01"] * 8 + ["2026-07-02"] * 4),
            "value": np.arange(12),
        })
        first = _cap_training_panel(panel, pd.Timestamp("2026-07-01"), max_train_rows=3)
        second = _cap_training_panel(panel, pd.Timestamp("2026-07-01"), max_train_rows=3)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(int((first["date"] <= pd.Timestamp("2026-07-01")).sum()), 3)
        self.assertEqual(int((first["date"] > pd.Timestamp("2026-07-01")).sum()), 4)

    def test_training_cap_excludes_purge_without_copying_full_panel_first(self):
        dates = pd.to_datetime(["2026-07-01"] * 4 + ["2026-07-02"] * 4 + ["2026-07-03"] * 4)
        panel = pd.DataFrame({
            "code": [f"{600000 + i:06d}" for i in range(12)],
            "date": dates,
            "value": np.arange(12),
        })
        result = _cap_training_panel(
            panel,
            pd.Timestamp("2026-07-01"),
            max_train_rows=3,
            exclude_dates=(pd.Timestamp("2026-07-02"),),
        )
        self.assertNotIn(pd.Timestamp("2026-07-02"), result["date"].tolist())
        self.assertEqual(int((result["date"] <= pd.Timestamp("2026-07-01")).sum()), 3)
        self.assertEqual(int((result["date"] > pd.Timestamp("2026-07-01")).sum()), 4)

    def test_cash_complete_target_keeps_unbuyable_as_cash(self):
        panel = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "entry_buyable": [False, True, True],
            "target_penalty_net_ret_t1": [np.nan, -0.10, 0.03],
            "target_net_ret_t1": [np.nan, np.nan, 0.03],
            "target_outcome_observed_t1": [True, True, True],
        })
        result = _cash_complete_targets(panel)
        self.assertEqual(result["target_cash_net_ret_t1"].tolist(), [0.0, -0.10, 0.03])
        self.assertEqual(result["target_entry_fill"].tolist(), [0.0, 1.0, 1.0])
        self.assertTrue(pd.isna(result.iloc[0]["target_exit_sellable_t1"]))
        self.assertEqual(result["target_exit_sellable_t1"].iloc[1:].tolist(), [0.0, 1.0])
        self.assertAlmostEqual(float(result["target_cash_excess_ret_t1"].mean()), 0.0)

    def test_exit_expected_value_combines_sellability_and_penalty(self):
        dates = pd.to_datetime(["2026-07-01", "2026-07-01"])
        sellable = pd.DataFrame({
            "code": ["600000", "600001"], "date": dates, "raw_pred": [0.9, 0.5]
        })
        returns = pd.DataFrame({
            "code": ["600000", "600001"], "date": dates, "raw_pred": [0.02, 0.04]
        })
        result = _exit_expected_value_frame(sellable, returns, -0.10, "exit_risk")
        expected = [0.9 * 0.02 - 0.1 * 0.10, 0.5 * 0.04 - 0.5 * 0.10]
        self.assertGreater(float(result.iloc[0]["score"]), 0.0)
        self.assertLess(float(result.iloc[1]["score"]), 0.0)
        self.assertEqual(result["model_variant"].unique().tolist(), ["exit_risk"])
        self.assertGreater(expected[0], expected[1])

    def test_top50_risk_constraint_removes_highest_risk_candidates(self):
        codes = [f"{600000 + index:06d}" for index in range(60)]
        dates = pd.to_datetime(["2026-07-01"] * 60)
        returns = pd.DataFrame({
            "code": codes, "date": dates, "raw_pred": np.arange(60, dtype=float)
        })
        sellable = pd.DataFrame({
            "code": codes, "date": dates, "raw_pred": np.ones(60, dtype=float)
        })
        sellable.loc[50:59, "raw_pred"] = 0.0
        result = _top50_risk_constrained_frame(
            sellable, returns, "risk_constrained", candidate_n=50, risk_exclusions=10
        )
        selected = result.nlargest(10, "score")["code"].tolist()
        self.assertEqual(selected, list(reversed(codes[40:50])))
        self.assertFalse(set(codes[50:60]) & set(selected))

    def test_safety_constraint_is_deterministic_and_emits_only_survivors(self):
        codes = [f"{600000 + index:06d}" for index in range(20)]
        date = pd.Timestamp("2026-07-01")
        returns = pd.DataFrame({
            "code": codes,
            "date": date,
            "score": np.ones(20),
        })
        safety = pd.DataFrame({
            "code": codes,
            "date": date,
            "raw_pred": np.arange(20, dtype=float) / 20.0,
        })
        first = _build_safety_constrained_frame(
            safety, returns, "safe", candidate_n=15, risk_remove_n=5, top_n=10
        )
        second = _build_safety_constrained_frame(
            safety.sample(frac=1.0, random_state=7),
            returns.sample(frac=1.0, random_state=8),
            "safe",
            candidate_n=15,
            risk_remove_n=5,
            top_n=10,
        )
        self.assertEqual(first["code"].tolist(), codes[5:15])
        self.assertEqual(second["code"].tolist(), codes[5:15])
        self.assertTrue(first["safety_eligible"].all())
        self.assertTrue(np.isfinite(first["score"]).all())

    def test_safety_constraint_validates_capacity(self):
        frame = pd.DataFrame({
            "code": ["600000"],
            "date": [pd.Timestamp("2026-07-01")],
            "score": [1.0],
            "raw_pred": [0.9],
        })
        with self.assertRaises(ValueError):
            _build_safety_constrained_frame(
                frame, frame, "safe", candidate_n=10, risk_remove_n=1, top_n=10
            )

    def test_return_fusion_uses_fixed_h1_weight(self):
        keys = {"code": ["600000", "600001"], "date": pd.to_datetime(["2026-07-01"] * 2)}
        e0 = pd.DataFrame({**keys, "score": [2.0, -2.0]})
        h1 = pd.DataFrame({**keys, "score": [-2.0, 2.0]})
        result = _build_return_fusion_frame(e0, h1, 0.25, "fusion")
        self.assertEqual(result["score"].tolist(), [1.0, -1.0])

    def test_inner_window_uses_only_mature_dates_before_outer_cutoff(self):
        dates = pd.bdate_range("2026-01-01", periods=70)
        panel = pd.DataFrame({
            "date": dates,
            "target_outcome_observed_t1": True,
        })
        panel.loc[panel.index[-1], "target_outcome_observed_t1"] = False
        spec = _inner_window_spec(panel, dates[-1], valid_days=10)
        self.assertEqual(spec["valid_end"], dates[-2])
        self.assertEqual(spec["valid_start"], dates[-11])
        self.assertEqual(spec["purge_date"], dates[-12])
        self.assertEqual(spec["train_end"], dates[-13])

    def test_inner_recipe_requires_risk_improvement_without_return_damage(self):
        baseline = {
            "mean_return": -0.001,
            "mean_filled_names": 10.0,
            "unsellable": 5,
            "unsellable_rate": 0.01,
            "normal_sellable_mean": 0.002,
        }
        metrics = {
            "R00_e0": baseline,
            "R01_e0_safe05": {
                "mean_return": 0.001,
                "mean_filled_names": 10.0,
                "unsellable": 2,
                "unsellable_rate": 0.004,
                "normal_sellable_mean": 0.00195,
            },
            "R02_e0_safe10": {
                "mean_return": 0.002,
                "mean_filled_names": 10.0,
                "unsellable": 0,
                "unsellable_rate": 0.0,
                "normal_sellable_mean": 0.0010,
            },
        }
        selected = _select_inner_recipe(metrics)
        self.assertEqual(selected["name"], "R01_e0_safe05")

    def test_inner_recipe_rejects_cash_advantage_from_low_fill_rate(self):
        metrics = {
            "R00_e0": {
                "mean_return": -0.002,
                "mean_filled_names": 10.0,
                "unsellable": 5,
                "unsellable_rate": 0.01,
                "normal_sellable_mean": -0.001,
            },
            "R20_h150": {
                "mean_return": -0.0001,
                "mean_filled_names": 3.0,
                "unsellable": 1,
                "unsellable_rate": 0.002,
                "normal_sellable_mean": 0.001,
            },
        }
        selected = _select_inner_recipe(metrics)
        self.assertEqual(selected["name"], "R00_e0")

    def test_causal_eligibility_filters_before_ranking(self):
        panel = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "signal_eligible": [True, False, True],
        })
        predictions = {
            name: pd.DataFrame({
                "code": panel["code"],
                "date": panel["date"],
                "score": values,
            })
            for name, values in {
                "left": [1.0, 9.0, 2.0],
                "right": [2.0, 8.0, 1.0],
            }.items()
        }
        result = _causal_eligible_predictions(
            predictions, panel, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-01")
        )
        self.assertEqual(result["left"]["code"].tolist(), ["600000", "600002"])
        self.assertEqual(result["right"]["code"].tolist(), ["600000", "600002"])

    def test_joined_loader_filters_dates_and_excludes_purge_day(self):
        daily, intraday = self._prepared_frames()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily_dir = root / "daily"
            intraday_dir = root / "intraday"
            daily_dir.mkdir()
            intraday_dir.mkdir()
            daily.to_parquet(daily_dir / "2026-07.parquet", index=False)
            intraday.to_parquet(intraday_dir / "2026-07.parquet", index=False)
            panel, groups = load_joined_prepared(
                daily_dir,
                intraday_dir,
                pd.Timestamp("2026-07-01"),
                pd.Timestamp("2026-07-02"),
                daily_features=["ret_5d"],
                asof_features=["ret_5d"],
                minute_features=["m5_ret_30m"],
                exclude_dates=[pd.Timestamp("2026-07-02")],
            )
        self.assertEqual(panel["date"].unique().tolist(), [pd.Timestamp("2026-07-01")])
        self.assertEqual(len(panel), 2)
        self.assertEqual(groups["daily"], ["daily__ret_5d"])

    def test_daily_target_variant_does_not_overwrite_source_execution_target(self):
        daily, intraday = self._prepared_frames()
        panel, _ = merge_prepared_frames(daily, intraday)
        original = panel["target_penalty_net_ret_t1"].copy()
        variant = RaceVariant(
            "daily_plus_minute", ("daily", "minute"), "daily_h1", False, "diagnostic",
        )
        result = panel_for_variant(panel, variant)
        self.assertEqual(result["target_penalty_net_ret_t1"].tolist(), [0.01, 0.03])
        self.assertEqual(panel["target_penalty_net_ret_t1"].tolist(), original.tolist())
        source_feature = "daily__ret_5d"
        source_values = panel[source_feature].copy()
        result.loc[result.index[0], source_feature] = 999.0
        pd.testing.assert_series_equal(panel[source_feature], source_values)
        self.assertTrue(np.allclose(
            result.groupby("date")["target_penalty_excess_ret_t1"].mean(), 0.0,
        ))

    def test_legacy_execution_variant_preserves_observed_return_target(self):
        daily, intraday = self._prepared_frames()
        panel, _ = merge_prepared_frames(daily, intraday)
        variant = RaceVariant(
            "minute_legacy", ("asof", "minute"), "execution_legacy", True, "candidate",
        )
        result = panel_for_variant(panel, variant)
        pd.testing.assert_series_equal(
            result["target_net_ret_t1"],
            panel["target_net_ret_t1"],
        )
        self.assertTrue(result["target_excess_ret_t1"].notna().any())

    def test_rolling_specs_purge_one_complete_session(self):
        dates = pd.bdate_range("2025-01-01", periods=130)
        calendar = pd.DataFrame({
            "date": dates,
            "target_net_ret_t1": np.arange(len(dates), dtype=float),
        })
        train_end, purge, valid_start, valid_end = rolling_window_specs(
            calendar, windows=1, valid_days=5, min_training_days=120,
        )[0]
        self.assertEqual((train_end, purge, valid_start, valid_end), tuple(dates[-7:][[0, 1, 2, 6]]))

    def test_fair_race_includes_retrained_daily_baseline(self):
        names = [variant.name for variant in DEFAULT_TRAINED_VARIANTS]
        self.assertIn("daily_current_retrained", names)
        self.assertIn("daily_plus_minute_current_target", names)

    def test_control_matrix_reuses_identical_base_and_minute_features(self):
        selected = {
            "daily_current_retrained": {"daily": ["daily__ret_5d", "daily__ret_10d"]},
            "daily_plus_minute_current_target": {
                "daily": ["daily__ret_5d"],
                "minute": ["minute__m5_ret_30m"],
            },
            "daily_close_control": {"daily_matched": ["daily__ret_5d", "daily__ret_10d"]},
            "daily_close_plus_minute_control": {
                "daily_matched": ["daily__ret_5d"],
                "minute": ["minute__m5_ret_30m"],
            },
            "daily_asof_control": {"asof_matched": ["asof__ret_10d"]},
            "daily_asof_plus_minute_control": {
                "asof_matched": ["asof__ret_10d"],
                "minute": ["minute__m5_ret_60m"],
            },
        }
        groups = {
            "asof_matched": ["asof__ret_5d", "asof__ret_10d"],
        }
        result = align_control_feature_selections(selected, groups)
        self.assertEqual(
            result["daily_close_plus_minute_control"]["daily_matched"],
            result["daily_close_control"]["daily_matched"],
        )
        self.assertEqual(
            result["daily_asof_control"]["asof_matched"],
            ["asof__ret_5d", "asof__ret_10d"],
        )
        self.assertEqual(
            result["daily_asof_plus_minute_control"]["minute"],
            ["minute__m5_ret_30m"],
        )
        self.assertEqual(
            result["daily_plus_minute_current_target"]["daily"],
            result["daily_current_retrained"]["daily"],
        )

    def test_selected_feature_names_map_back_to_source_columns(self):
        selected = {
            "variant": {
                "daily_matched": ["daily__ret_5d"],
                "asof": ["asof__ret_5d"],
                "minute": ["minute__m5_ret_30m"],
            },
        }
        self.assertEqual(
            source_features_for_selection(selected),
            (["ret_5d"], ["ret_5d"], ["m5_ret_30m"]),
        )

    def test_variant_features_use_only_requested_groups(self):
        groups = {"daily": ["daily__ret"], "asof": ["asof__ret"], "minute": ["minute__ret"]}
        variant = RaceVariant(
            "daily_plus_minute", ("daily", "minute"), "execution", False, "diagnostic",
        )
        self.assertEqual(
            variant_features(groups, variant),
            ["daily__ret", "minute__ret"],
        )


class ForwardEvaluationTest(unittest.TestCase):
    @staticmethod
    def _screening(path: Path) -> Path:
        report = {
            "windows": [{
                "window": 1,
                "train_end": "2026-07-01",
                "selected": {
                    "daily_asof_plus_minute_control": {
                        "asof_matched": ["asof__ret_5d"],
                        "minute": ["minute__m5_ret_30m"],
                    }
                },
            }]
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    @staticmethod
    def _prepared(directory: Path, dates, observed=None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"date": pd.to_datetime(dates), "code": "600000"})
        if observed is not None:
            frame["target_outcome_observed_t1"] = observed
        frame.to_parquet(directory / "2026-07.parquet", index=False)

    def test_available_common_dates_intersects_prepared_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "daily"
            intraday = root / "intraday"
            self._prepared(daily, ["2026-07-01", "2026-07-02"])
            self._prepared(intraday, ["2026-07-02", "2026-07-03"], [True, True])
            self.assertEqual(
                available_common_dates(daily, intraday).tolist(),
                [pd.Timestamp("2026-07-02")],
            )

    def test_initialize_freezes_latest_common_cutoff_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "daily"
            intraday = root / "intraday"
            state = root / "state"
            screening = self._screening(root / "screening.json")
            self._prepared(daily, ["2026-07-01", "2026-07-02"])
            self._prepared(intraday, ["2026-07-01", "2026-07-02"], [True, False])
            first = initialize_forward_evaluation(screening, state, daily, intraday)
            second = initialize_forward_evaluation(screening, state, daily, intraday)
            self.assertEqual(first, second)
            self.assertEqual(first["cutoff_date"], "2026-07-02")
            self.assertEqual(forward_status(state)["status"], "initialized_waiting_for_new_mature_dates")

    def test_validate_rejects_immutable_configuration_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "daily"
            intraday = root / "intraday"
            screening = self._screening(root / "screening.json")
            self._prepared(daily, ["2026-07-01"])
            self._prepared(intraday, ["2026-07-01"], [True])
            manifest = initialize_forward_evaluation(screening, root / "state", daily, intraday)
            manifest["penalty"] = -0.05
            with self.assertRaises(RuntimeError):
                validate_forward_manifest(manifest, screening)

    def test_mature_forward_dates_require_post_cutoff_observed_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "daily"
            intraday = root / "intraday"
            dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
            self._prepared(daily, dates)
            self._prepared(intraday, dates, [True, False, True])
            manifest = {"cutoff_date": "2026-07-01", "processed_dates": []}
            self.assertEqual(
                _mature_forward_dates(manifest, daily, intraday),
                [pd.Timestamp("2026-07-03")],
            )

    def test_run_without_new_dates_waits_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            daily = root / "daily"
            intraday = root / "intraday"
            state = root / "state"
            screening = self._screening(root / "screening.json")
            self._prepared(daily, ["2026-07-01"])
            self._prepared(intraday, ["2026-07-01"], [True])
            initialize_forward_evaluation(screening, state, daily, intraday)
            result = run_forward_evaluation_once(screening, state, model_threads=1)
            self.assertEqual(result["status"], "waiting_for_new_mature_dates")
            self.assertFalse((state / "execution_records.parquet").exists())


class StructuralComboTest(unittest.TestCase):
    def test_feature_screening_window_never_exceeds_training_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "screening.json"
            path.write_text(json.dumps({"windows": [
                {"train_end": "2025-08-06", "name": "causal"},
                {"train_end": "2025-11-06", "name": "future"},
                {"train_end": "2026-05-11", "name": "latest"},
            ]}), encoding="utf-8")
            selected = _selected_feature_window(path)
            self.assertEqual(selected["name"], "causal")

    def test_holdout_uses_exactly_first_sixty_dates(self):
        dates = pd.bdate_range("2025-11-03", periods=65)
        labels = pd.DataFrame({"date": list(reversed(dates))})
        selected = first_holdout_dates(labels)
        self.assertEqual(len(selected), 60)
        self.assertEqual(selected[0], dates[0])
        self.assertEqual(selected[-1], dates[59])
        with self.assertRaises(ValueError):
            first_holdout_dates(labels.iloc[:59])

    def test_holdout_must_begin_after_training_labels_and_calibration(self):
        training = pd.DataFrame({"date": pd.bdate_range("2025-07-01", periods=70)})
        overlapping = pd.DataFrame({"date": pd.bdate_range("2025-09-01", periods=60)})
        with self.assertRaises(ValueError):
            validated_holdout_dates(training, overlapping)
        future = pd.DataFrame({"date": pd.bdate_range("2025-11-03", periods=60)})
        selected = validated_holdout_dates(training, future)
        self.assertEqual(selected[0], pd.Timestamp("2025-11-03"))

    def test_probability_calibration_is_fitted_only_from_supplied_rows(self):
        calibrator = fit_probability_calibrator(
            [-2.0, -1.0, 1.0, 2.0], [0.0, 0.0, 1.0, 1.0]
        )
        probability = apply_probability_calibrator([-1.0, 1.0], calibrator)
        self.assertLess(probability.iloc[0], probability.iloc[1])
        constant = fit_probability_calibrator([0.0, 1.0], [1.0, 1.0])
        self.assertEqual(constant["kind"], "constant")
        self.assertAlmostEqual(apply_probability_calibrator([99.0], constant).iloc[0], 0.9999)

    def test_daily_prior_maps_only_to_next_trading_session(self):
        calendar = pd.to_datetime(["2026-07-03", "2026-07-06", "2026-07-07"])
        daily = pd.DataFrame({
            "code": ["600000", "600000", "600000"],
            "date": calendar,
            "pred": [1.0, 2.0, 3.0],
        })
        shifted = shift_daily_prior_to_signal(daily, calendar)
        self.assertEqual(shifted["source_date"].tolist(), calendar[:2].tolist())
        self.assertEqual(shifted["date"].tolist(), calendar[1:].tolist())

    def test_e4_coverage_rejects_missing_expected_date(self):
        daily = pd.DataFrame({
            "code": ["600000"],
            "date": pd.to_datetime(["2026-07-06"]),
            "pred": [1.0],
        })
        minute = pd.DataFrame({
            "code": ["600000"],
            "date": pd.to_datetime(["2026-07-06"]),
            "score": [1.0],
        })
        with self.assertRaisesRegex(ValueError, "2026-07-07"):
            e4_coverage_diagnostics(
                daily,
                minute,
                pd.to_datetime(["2026-07-06", "2026-07-07"]),
                candidate_n=10,
            )

    def test_e4_coverage_reports_top_candidate_retention(self):
        date = pd.Timestamp("2026-07-06")
        codes = [f"{600000 + index:06d}" for index in range(20)]
        daily = pd.DataFrame({
            "code": codes,
            "date": date,
            "pred": np.arange(20, dtype=float),
        })
        minute = pd.DataFrame({
            "code": codes[:5],
            "date": date,
            "score": np.arange(5, dtype=float),
        })
        diagnostics = e4_coverage_diagnostics(
            daily, minute, pd.DatetimeIndex([date]), candidate_n=10
        )
        self.assertEqual(diagnostics["candidate_rows"], 10)
        self.assertEqual(diagnostics["minute_matched_candidate_rows"], 0)
        self.assertEqual(diagnostics["top_candidate_retention"], 0.0)

    def test_daily_execution_filter_keeps_daily_rank_after_buy_gate(self):
        date = pd.Timestamp("2026-07-06")
        codes = [f"{600000 + index:06d}" for index in range(20)]
        daily = pd.DataFrame({
            "code": codes,
            "date": date,
            "raw_pred": np.arange(20, 0, -1, dtype=float),
        })
        buy = pd.DataFrame({
            "code": codes,
            "date": date,
            "raw_pred": [0.1] * 10 + [0.9] * 10,
        })
        result = build_daily_execution_filter_scores(
            daily, buy, candidate_n=20, minimum_buy_probability=0.50
        )
        self.assertEqual(result["code"].tolist(), codes[10:])
        selected = result.nlargest(3, "score")["code"].tolist()
        self.assertEqual(selected, codes[10:13])

    def test_staged_combo_never_promotes_outside_daily_top100(self):
        date = pd.Timestamp("2026-07-06")
        codes = [f"{600000 + index:06d}" for index in range(120)]
        daily = pd.DataFrame({
            "code": codes,
            "date": date,
            "raw_pred": np.arange(120, 0, -1, dtype=float),
        })
        minute = pd.DataFrame({
            "code": codes,
            "date": date,
            "score": np.arange(120, dtype=float),
        })
        outputs = build_e0_e4_staged_scores(daily, minute, candidate_n=100)
        for frame in outputs.values():
            self.assertEqual(len(frame), 100)
            self.assertFalse(set(codes[100:]) & set(frame["code"]))
        selected = outputs["e4_top100_minute_block"].nlargest(10, "score")["code"]
        self.assertEqual(set(selected), set(codes[90:100]))

    def test_liquidation_head_uses_constant_when_train_has_no_open_cases(self):
        panel = pd.DataFrame({
            "code": ["600000", "600001", "600000"],
            "date": pd.to_datetime(["2025-08-01", "2025-08-01", "2025-09-01"]),
            "feature": [1.0, 2.0, 3.0],
            "target": [1.0, 1.0, 0.0],
        })
        result = _fit_liquidation_head(panel, ["feature"], "target", model_threads=1)
        self.assertIn("constant_empirical", result["metrics"])
        self.assertEqual(result["metrics"]["constant_empirical"]["open_samples"], 0)
        self.assertTrue((result["predictions"]["raw_pred"] == 1.0).all())

    def test_structural_combo_uses_fixed_execution_formula(self):
        dates = pd.to_datetime(["2026-07-01"] * 3 + ["2026-07-02"] * 3)
        codes = ["600000", "600001", "600002"] * 2
        keys = {"code": codes, "date": dates}
        direct = pd.DataFrame({**keys, "raw_pred": [-0.01, 0.0, 0.01] * 2})
        buy = pd.DataFrame({**keys, "raw_pred": [1.0, 0.5, 0.0] * 2})
        liquidate = pd.DataFrame({**keys, "raw_pred": [1.0, 0.8, 0.5] * 2})
        conditional = pd.DataFrame({**keys, "raw_pred": [0.02, 0.03, 0.04] * 2})
        result = build_structural_combo_scores(direct, buy, liquidate, conditional)
        structural = result["c1_structural"]
        first_day = structural[structural["date"] == pd.Timestamp("2026-07-01")]
        self.assertAlmostEqual(first_day.iloc[0]["c1_structural"], 0.02)
        self.assertAlmostEqual(first_day.iloc[1]["c1_structural"], 0.5 * (0.8 * 0.03 + 0.2 * -0.10))
        self.assertAlmostEqual(first_day.iloc[2]["c1_structural"], 0.0)
        self.assertTrue(np.isfinite(result["c2_fixed_50_50"]["score"]).all())


class AdaptiveExitReplayTest(unittest.TestCase):
    def test_adaptive_label_code_sample_is_order_independent(self):
        codes = ["600003", "600001", "600002", "600000"]
        first = deterministic_code_sample(codes, 3)
        second = deterministic_code_sample(list(reversed(codes)), 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_adaptive_labels_keep_realized_and_stress_returns_separate(self):
        calendar = pd.bdate_range("2026-07-01", periods=6)
        records = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "signal_date": [calendar[1]] * 3,
            "entry_buyable": [False, True, True],
            "exit_sellable": [False, True, False],
            "net_return": [0.0, 0.02, np.nan],
            "penalty_net_return": [0.0, 0.02, -0.10],
            "entry_reason": ["locked", "filled", "filled"],
            "exit_reason": ["not_entered", "time_cap", "blocked"],
            "exit_timestamp": [pd.NaT, calendar[2] + pd.Timedelta(hours=14, minutes=50), pd.NaT],
            "entry_price": [np.nan, 10.0, 10.0],
            "exit_price": [np.nan, 10.2, np.nan],
        })
        labels = adaptive_records_to_labels(records, calendar)
        diagnostics = label_diagnostics(labels)
        self.assertTrue(pd.isna(labels.loc[0, "adaptive_liquidated_by_t3"]))
        self.assertAlmostEqual(labels.loc[1, "adaptive_realized_net_ret_t3"], 0.02)
        self.assertTrue(pd.isna(labels.loc[2, "adaptive_realized_net_ret_t3"]))
        self.assertAlmostEqual(labels.loc[2, "adaptive_stress_net_ret_t3"], -0.10)
        self.assertEqual(diagnostics["open_t3"], 1)

    def test_fetch_plan_includes_prior_and_three_exit_sessions(self):
        calendar = pd.bdate_range("2026-07-01", periods=7)
        trades = pd.DataFrame({
            "code": ["600000", "600000", "600001"],
            "signal_date": [calendar[2], calendar[2], calendar[-1]],
        })
        plan, incomplete = build_fetch_plan(trades, calendar, max_exit_sessions=3)
        self.assertEqual(incomplete, [calendar[-1]])
        self.assertEqual(plan["session"].drop_duplicates().tolist(), calendar[1:6].tolist())
        self.assertEqual(len(plan), 5)

    @staticmethod
    def _trade(signal_date, fixed_exit_sellable=True):
        return pd.DataFrame({
            "model": ["minute_e3_minus_10"],
            "signal_date": [signal_date],
            "code": ["600000"],
            "rank": [1],
            "score": [1.0],
            "entry_buyable": [True],
            "exit_sellable": [fixed_exit_sellable],
        })

    def test_replay_executes_stop_signal_on_next_bar(self):
        calendar = pd.bdate_range("2026-07-01", periods=5)
        signal = calendar[1]
        bars = pd.DataFrame([
            (calendar[0] + pd.Timedelta(hours=14, minutes=50), 9.5, 9.6, 9.4, 9.5, 100),
            (signal + pd.Timedelta(hours=14, minutes=50), 10.0, 10.02, 9.98, 10.0, 100),
            (calendar[2] + pd.Timedelta(hours=9, minutes=35), 9.6, 9.7, 9.5, 9.6, 100),
            (calendar[2] + pd.Timedelta(hours=9, minutes=40), 9.4, 9.5, 9.3, 9.4, 100),
            (calendar[2] + pd.Timedelta(hours=9, minutes=45), 9.35, 9.4, 9.3, 9.35, 100),
        ], columns=["timestamp", "open", "high", "low", "close", "volume"])
        bars["code"] = "600000"
        bars["amount"] = bars["close"] * bars["volume"]
        result = replay_selected_trades(self._trade(signal), bars, calendar)
        self.assertEqual(result.iloc[0]["exit_reason"], "stop_loss")
        self.assertEqual(result.iloc[0]["exit_timestamp"], calendar[2] + pd.Timedelta(hours=9, minutes=45))
        self.assertAlmostEqual(result.iloc[0]["net_return"], 9.35 / 10.0 - 1.0 - 0.002)

    def test_replay_recovers_fixed_t1_block_on_t2(self):
        calendar = pd.bdate_range("2026-07-01", periods=5)
        signal = calendar[1]
        bars = pd.DataFrame([
            (calendar[0] + pd.Timedelta(hours=14, minutes=50), 9.5, 9.6, 9.4, 9.5, 100),
            (signal + pd.Timedelta(hours=14, minutes=50), 10.0, 10.02, 9.98, 10.0, 100),
            (calendar[2] + pd.Timedelta(hours=14, minutes=45), 10.0, 10.01, 9.99, 10.0, 100),
            (calendar[2] + pd.Timedelta(hours=14, minutes=50), 9.5, 9.5, 9.5, 9.5, 0),
            (calendar[3] + pd.Timedelta(hours=9, minutes=35), 9.4, 9.5, 9.3, 9.4, 100),
        ], columns=["timestamp", "open", "high", "low", "close", "volume"])
        bars["code"] = "600000"
        bars["amount"] = bars["close"] * bars["volume"]
        result = replay_selected_trades(
            self._trade(signal, fixed_exit_sellable=False), bars, calendar
        )
        summary = summarize_replay(result)
        self.assertTrue(result.iloc[0]["exit_sellable"])
        self.assertEqual(result.iloc[0]["exit_timestamp"], calendar[3] + pd.Timedelta(hours=9, minutes=35))
        self.assertEqual(
            summary["breakdown"]["minute_e3_minus_10"]["fixed_t1_blocked_recovered"],
            1,
        )


class DualPaperEvaluationTest(unittest.TestCase):
    @staticmethod
    def _active_files(root: Path, published_at: str = "2026-07-01T12:00:00"):
        predictions = root / "active.parquet"
        manifest = root / "active.json"
        pd.DataFrame({
            "code": ["600000", "600001"],
            "date": pd.to_datetime(["2026-07-01"] * 2),
            "score": [2.0, 1.0],
        }).to_parquet(predictions, index=False)
        manifest.write_text(json.dumps({"published_at": published_at}), encoding="utf-8")
        return predictions, manifest

    def test_daily_publication_capture_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions, manifest = self._active_files(root)
            ledger = root / "ledger"
            first = capture_daily_publication(predictions, manifest, ledger)
            second = capture_daily_publication(predictions, manifest, ledger)
            self.assertEqual(first, second)
            changed = pd.read_parquet(predictions)
            changed.loc[0, "score"] = 99.0
            changed.to_parquet(predictions, index=False)
            with self.assertRaises(RuntimeError):
                capture_daily_publication(predictions, manifest, ledger)

    def test_daily_publication_after_entry_cutoff_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions, manifest = self._active_files(root, "2026-07-01T15:05:00")
            with self.assertRaises(RuntimeError):
                capture_daily_publication(predictions, manifest, root / "ledger")

    def test_main_race_preserves_independent_universes(self):
        date = pd.Timestamp("2026-07-01")
        daily = pd.DataFrame({
            "code": ["600000", "600001"], "date": date, "score": [2.0, 1.0],
        })
        minute = pd.DataFrame({
            "code": ["600001", "600002"], "date": date, "score": [2.0, 1.0],
        })
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": date,
            "entry_buyable": True,
            "target_net_ret_t1": [0.01, 0.02, 0.03],
            "target_outcome_observed_t1": True,
        })
        main, diagnostic = evaluate_prediction_pair(daily, minute, labels)
        self.assertEqual(
            set(main.loc[main["model"] == "daily_actual", "code"]),
            {"600000", "600001"},
        )
        self.assertEqual(
            set(main.loc[main["model"] == "minute_e3_minus_10", "code"]),
            {"600001", "600002"},
        )
        self.assertEqual(set(diagnostic["code"]), {"600001"})

    def test_retrospective_bootstrap_preserves_main_and_common_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = pd.to_datetime(["2026-07-01", "2026-07-02"])
            active = pd.DataFrame([
                {"code": code, "date": date, "pred": score}
                for date in dates
                for code, score in (("600000", 2.0), ("600001", 1.0))
            ])
            active_path = root / "active.parquet"
            active.to_parquet(active_path, index=False)
            e3_dir = root / "e3"
            e3_dir.mkdir()
            minute = pd.DataFrame([
                {"code": code, "date": date, "score": score}
                for date in dates
                for code, score in (("600001", 2.0), ("600002", 1.0))
            ])
            minute.to_parquet(
                e3_dir / "window_1_exec_e3_exit_risk_10_predictions.parquet",
                index=False,
            )
            prepared = root / "prepared"
            prepared.mkdir()
            labels = pd.DataFrame([
                {
                    "code": code,
                    "date": date,
                    "entry_buyable": True,
                    "target_net_ret_t1": value,
                    "target_outcome_observed_t1": True,
                }
                for date in dates
                for code, value in (("600000", 0.01), ("600001", 0.02), ("600002", 0.03))
            ])
            labels.to_parquet(prepared / "2026-07.parquet", index=False)
            report = run_retrospective_bootstrap(
                active_path, e3_dir, prepared, root / "output"
            )
            self.assertEqual(report["windows"][0]["days"], 2)
            self.assertEqual(
                set(report["main_comparison"]["models"]),
                {"daily_actual", "minute_e3_minus_10"},
            )
            self.assertTrue((root / "output" / "retrospective_report.json").exists())

    def test_dual_race_waits_until_both_snapshots_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minute_state = root / "minute"
            minute_state.mkdir()
            (minute_state / "manifest.json").write_text(json.dumps({
                "cutoff_date": "2026-07-01",
                "immutable_sha256": "minute-hash",
                "intraday_prepared_dir": str(root / "prepared"),
            }), encoding="utf-8")
            predictions, active_manifest = self._active_files(root)
            state = root / "dual"
            initialize_dual_paper_evaluation(
                minute_state, state, predictions, active_manifest
            )
            result = run_dual_paper_evaluation_once(state)
            self.assertEqual(result["status"], "waiting_for_paired_mature_dates")
            self.assertEqual(result["processed_dates"], [])


class OfflineRaceTest(unittest.TestCase):
    @staticmethod
    def _predictions() -> dict[str, pd.DataFrame]:
        dates = pd.to_datetime(["2026-07-01"] * 3)
        return {
            "daily": pd.DataFrame({
                "code": ["600000", "600001", "600002"],
                "date": dates,
                "score": [3.0, 2.0, 1.0],
            }),
            "minute": pd.DataFrame({
                "code": ["600000", "600001", "600002"],
                "date": dates,
                "score": [1.0, 3.0, 2.0],
            }),
        }

    def test_common_universe_excludes_keys_missing_from_either_model(self):
        predictions = self._predictions()
        predictions["minute"] = predictions["minute"].iloc[1:].copy()
        common = common_prediction_universe(predictions)
        self.assertEqual(common["daily"]["code"].tolist(), ["600001", "600002"])
        self.assertEqual(common["minute"]["code"].tolist(), ["600001", "600002"])

    def test_fixed_race_keeps_unbuyable_top_name_without_backfill(self):
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "entry_buyable": [False, True, True],
            "target_outcome_observed_t1": [True, True, True],
            "target_net_ret_t1": [np.nan, np.nan, 0.03],
            "label_entry_price_1450": [10.0, 20.0, 30.0],
        })
        records, report = simulate_fixed_exit_race(
            self._predictions(), labels, ExecutionConfig(top_n=1),
        )
        daily = records[records["model"] == "daily"].iloc[0]
        minute = records[records["model"] == "minute"].iloc[0]
        self.assertEqual(daily["code"], "600000")
        self.assertFalse(bool(daily["entry_buyable"]))
        self.assertEqual(float(daily["net_return"]), 0.0)
        self.assertEqual(minute["code"], "600001")
        self.assertAlmostEqual(float(minute["net_return"]), -0.10)
        self.assertEqual(report["models"]["daily"]["mean_names"], 1.0)

    def test_fixed_race_reports_all_pairwise_model_comparisons(self):
        predictions = self._predictions()
        predictions["daily_plus_minute"] = predictions["minute"].assign(
            score=[2.0, 1.0, 3.0]
        )
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600002"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "entry_buyable": [True, True, True],
            "target_outcome_observed_t1": [True, True, True],
            "target_net_ret_t1": [0.01, 0.02, 0.03],
        })
        _, report = simulate_fixed_exit_race(
            predictions, labels, ExecutionConfig(top_n=1),
        )
        self.assertEqual(len(report["pairwise"]), 3)
        self.assertIn("minute_minus_daily", report["pairwise"])
        self.assertIn("daily_plus_minute_minus_daily", report["pairwise"])

    def test_adaptive_exit_uses_next_bar_after_completed_signal(self):
        predictions = {
            name: frame[frame["code"] == "600000"].copy()
            for name, frame in self._predictions().items()
        }
        bars = pd.DataFrame({
            "code": ["600000"] * 5,
            "timestamp": pd.to_datetime([
                "2026-06-30 14:55",
                "2026-07-01 14:50",
                "2026-07-02 09:30",
                "2026-07-02 09:35",
                "2026-07-02 14:50",
            ]),
            "open": [10.0, 10.0, 10.0, 9.4, 9.8],
            "high": [10.1, 10.1, 10.0, 9.6, 9.9],
            "low": [9.9, 9.9, 9.3, 9.3, 9.7],
            "close": [10.0, 10.0, 9.4, 9.5, 9.8],
            "volume": [100.0] * 5,
            "bar_vwap_qfq": [10.0, 10.0, 9.6, 9.45, 9.8],
        })
        records, _ = simulate_adaptive_exit_race(
            predictions,
            bars,
            ExecutionConfig(top_n=1, stop_loss=0.05),
        )
        trade = records.iloc[0]
        self.assertEqual(trade["exit_reason"], "stop_loss")
        self.assertEqual(trade["exit_timestamp"], pd.Timestamp("2026-07-02 09:35"))
        self.assertAlmostEqual(float(trade["exit_price"]), 9.45)
        self.assertAlmostEqual(float(trade["net_return"]), -0.057)

    def test_adaptive_exit_rolls_suspended_t1_to_next_session(self):
        predictions = {
            name: frame[frame["code"] == "600000"].copy()
            for name, frame in self._predictions().items()
        }
        bars = pd.DataFrame({
            "code": ["600000", "600000", "600999", "600000"],
            "timestamp": pd.to_datetime([
                "2026-06-30 14:55",
                "2026-07-01 14:50",
                "2026-07-02 14:50",
                "2026-07-03 09:30",
            ]),
            "open": [10.0, 10.0, 20.0, 10.2],
            "high": [10.1, 10.1, 20.1, 10.4],
            "low": [9.9, 9.9, 19.9, 10.1],
            "close": [10.0, 10.0, 20.0, 10.3],
            "volume": [100.0] * 4,
            "bar_vwap_qfq": [10.0, 10.0, 20.0, 10.25],
        })
        records, _ = simulate_adaptive_exit_race(
            predictions, bars, ExecutionConfig(top_n=1),
        )
        trade = records.iloc[0]
        self.assertEqual(trade["exit_reason"], "suspended_t1_roll")
        self.assertEqual(trade["exit_timestamp"], pd.Timestamp("2026-07-03 09:30"))
        self.assertAlmostEqual(float(trade["net_return"]), 0.023)

    def test_adaptive_exit_rolls_locked_limit_down_to_next_session(self):
        predictions = {
            name: frame[frame["code"] == "600000"].copy()
            for name, frame in self._predictions().items()
        }
        bars = pd.DataFrame({
            "code": ["600000"] * 6,
            "timestamp": pd.to_datetime([
                "2026-06-30 14:55",
                "2026-07-01 14:50",
                "2026-07-02 09:30",
                "2026-07-02 14:45",
                "2026-07-02 14:50",
                "2026-07-03 09:30",
            ]),
            "open": [10.0, 10.0, 9.5, 9.5, 9.5, 9.4],
            "high": [10.1, 10.1, 9.5, 9.5, 9.5, 9.6],
            "low": [9.9, 9.9, 9.5, 9.5, 9.5, 9.3],
            "close": [10.0, 10.0, 9.5, 9.5, 9.5, 9.5],
            "volume": [100.0] * 6,
            "bar_vwap_qfq": [10.0, 10.0, 9.5, 9.5, 9.5, 9.45],
        })
        records, _ = simulate_adaptive_exit_race(
            predictions,
            bars,
            ExecutionConfig(top_n=1, stop_loss=0.0, take_profit=0.0),
        )
        trade = records.iloc[0]
        self.assertEqual(trade["exit_reason"], "time_cap")
        self.assertEqual(trade["exit_timestamp"], pd.Timestamp("2026-07-03 09:30"))
        self.assertAlmostEqual(float(trade["net_return"]), -0.057)


class AutoResearchTest(unittest.TestCase):
    @staticmethod
    def _report(start: str = "2025-11-03", end: str = "2026-01-23") -> dict:
        models = {
            "e4_daily_top10": {
                "days": 60,
                "mean_return": -0.001,
                "compound_return": -0.06,
                "max_drawdown": -0.10,
                "mean_names": 10.0,
                "mean_filled_names": 10.0,
            },
            "h2_daily_top100_buy_filter": {
                "days": 60,
                "mean_return": 0.001,
                "compound_return": 0.06,
                "max_drawdown": -0.08,
                "mean_names": 10.0,
                "mean_filled_names": 9.0,
            },
        }
        block_starts = ["2025-11-03", "2025-12-01", "2025-12-29"]
        block_ends = ["2025-11-28", "2025-12-26", "2026-01-23"]
        blocks = [{
            "block": index + 1,
            "start": block_starts[index],
            "end": block_ends[index],
            "comparison": {
                "models": {
                    name: {**metrics, "days": 20}
                    for name, metrics in models.items()
                }
            },
            "account_comparison": {
                "models": {
                    name: {**metrics, "days": 20}
                    for name, metrics in models.items()
                }
            },
        } for index in range(3)]
        return {
            "experiment": "synthetic_holdout",
            "untouched_holdout": True,
            "daily_history_causal": True,
            "holdout_days": 60,
            "holdout_start": start,
            "holdout_end": end,
            "comparison": {"models": models},
            "account_comparison": {"models": models},
            "twenty_day_blocks": blocks,
        }

    def test_cash_normalization_preserves_fixed_account_denominator(self):
        records = pd.DataFrame({
            "model": ["minute"],
            "signal_date": [pd.Timestamp("2026-01-05")],
            "code": ["600000"],
            "entry_buyable": [True],
            "exit_sellable": [True],
            "outcome_observed": [True],
            "net_return": [0.10],
        })
        dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
        normalized = cash_normalized_execution_records(
            records, dates, top_n=2, models=["minute", "empty"]
        )
        daily = normalized.groupby(["model", "signal_date"])["net_return"].mean()
        self.assertEqual(len(normalized), 8)
        self.assertAlmostEqual(
            float(daily.loc[("minute", pd.Timestamp("2026-01-05"))]), 0.05
        )
        self.assertAlmostEqual(
            float(daily.loc[("minute", pd.Timestamp("2026-01-06"))]), 0.0
        )
        self.assertAlmostEqual(
            float(daily.loc[("empty", pd.Timestamp("2026-01-05"))]), 0.0
        )

    def test_positive_h2_routes_to_execution_filter_forward_shadow(self):
        decision = evaluate_holdout_report(self._report())
        self.assertEqual(decision["winner"], "h2_daily_top100_buy_filter")
        self.assertEqual(decision["next_branch"], "execution_filter")
        self.assertTrue(decision["research_winner"])
        self.assertTrue(decision["forward_shadow_eligible"])
        self.assertFalse(decision["production_candidate"])

    def test_mutable_daily_history_cannot_select_next_branch(self):
        report = self._report()
        minute_metrics = {
            "days": 60,
            "mean_return": 0.0005,
            "compound_return": 0.03,
            "max_drawdown": -0.08,
            "mean_names": 10.0,
            "mean_filled_names": 9.0,
        }
        report["account_comparison"]["models"]["e1_e3_structural"] = minute_metrics
        report["comparison"]["models"]["e1_e3_structural"] = minute_metrics
        for block in report["twenty_day_blocks"]:
            block_metrics = {**minute_metrics, "days": 20}
            block["comparison"]["models"]["e1_e3_structural"] = block_metrics
            block["account_comparison"]["models"]["e1_e3_structural"] = block_metrics
        report["daily_history_causal"] = False
        decision = evaluate_holdout_report(report)
        self.assertEqual(decision["diagnostic_overall_winner"], "h2_daily_top100_buy_filter")
        self.assertEqual(decision["winner"], "e1_e3_structural")
        self.assertEqual(decision["next_branch"], "independent_structural")
        self.assertFalse(decision["research_winner"])
        self.assertFalse(decision["forward_shadow_eligible"])

    def test_low_fill_relative_winner_continues_research_without_shadow_promotion(self):
        report = self._report()
        report["account_comparison"]["models"]["h2_daily_top100_buy_filter"][
            "mean_filled_names"
        ] = 4.0
        decision = evaluate_holdout_report(report)
        self.assertEqual(decision["next_branch"], "execution_filter")
        self.assertTrue(decision["research_winner"])
        self.assertFalse(decision["forward_shadow_eligible"])

    def test_branch_decision_requires_all_three_complete_blocks(self):
        report = self._report()
        report["twenty_day_blocks"] = report["twenty_day_blocks"][:2]
        with self.assertRaises(ValueError):
            evaluate_holdout_report(report)
        report = self._report()
        report["twenty_day_blocks"][1]["account_comparison"]["models"].pop(
            "h2_daily_top100_buy_filter"
        )
        with self.assertRaises(ValueError):
            evaluate_holdout_report(report)

    def test_controlled_cycle_is_idempotent_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_labels = root / "training.parquet"
            holdout_labels = root / "holdout.parquet"
            screening = root / "screening.json"
            output = root / "output"
            state = root / "state"
            daily_dir = root / "daily"
            intraday_dir = root / "intraday"
            quant_dir = root / "quant"
            daily_dir.mkdir()
            intraday_dir.mkdir()
            quant_dir.mkdir()
            pd.DataFrame({"date": pd.bdate_range("2025-07-01", periods=60)}).to_parquet(
                training_labels, index=False
            )
            pd.DataFrame({"date": pd.bdate_range("2025-11-03", periods=60)}).to_parquet(
                holdout_labels, index=False
            )
            pd.DataFrame({"value": [1]}).to_parquet(daily_dir / "data.parquet", index=False)
            pd.DataFrame({"value": [1]}).to_parquet(intraday_dir / "data.parquet", index=False)
            pd.DataFrame({"value": [1]}).to_parquet(
                quant_dir / "active_quant_short_predictions.parquet", index=False
            )
            screening.write_text("{}", encoding="utf-8")
            stale_labels = root / "stale-labels.parquet"
            stale_labels.write_bytes(b"stale")
            claim_holdout(
                state,
                "stale-crashed-cycle",
                "2025-11-03",
                str(pd.bdate_range("2025-11-03", periods=60)[-1].date()),
                stale_labels,
            )
            fail_once = {"value": True}

            def fake_holdout(*args, **kwargs):
                if fail_once["value"]:
                    fail_once["value"] = False
                    raise RuntimeError("transient training failure")
                report = self._report(
                    start="2025-11-03",
                    end=str(pd.bdate_range("2025-11-03", periods=60)[-1].date()),
                )
                report["twenty_day_blocks"][0]["start"] = report["holdout_start"]
                report["twenty_day_blocks"][-1]["end"] = report["holdout_end"]
                report["input_hashes"] = kwargs["expected_input_hashes"]
                output.mkdir(parents=True, exist_ok=True)
                (output / "holdout_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                return report

            with mock.patch(
                "intraday_1400.fair_race_pipeline.default_daily_prepared_dir",
                return_value=daily_dir,
            ), mock.patch(
                "intraday_1400.auto_research.config.PREPARED_DIR", intraday_dir
            ), mock.patch(
                "quant.config.QUANT_DIR", quant_dir
            ), mock.patch(
                "intraday_1400.structural_combo_holdout.run_frozen_holdout",
                side_effect=fake_holdout,
            ) as runner:
                with self.assertRaises(RuntimeError):
                    run_frozen_holdout_cycle(
                        training_labels, holdout_labels, screening, output, state
                    )
                failed_manifest = initialize_research_state(state)
                self.assertTrue(all(
                    claim["status"] == "abandoned"
                    for claim in failed_manifest["holdout_claims"]
                ))
                first = run_frozen_holdout_cycle(
                    training_labels, holdout_labels, screening, output, state
                )
                second = run_frozen_holdout_cycle(
                    training_labels, holdout_labels, screening, output, state
                )
            manifest = initialize_research_state(state)
            self.assertEqual(first, second)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(len(manifest["experiments"]), 1)
            self.assertEqual(len(manifest["consumed_holdouts"]), 1)
            self.assertEqual(
                [claim["status"] for claim in manifest["holdout_claims"]].count("consumed"),
                1,
            )
            self.assertEqual(
                [claim["status"] for claim in manifest["holdout_claims"]].count("abandoned"),
                1,
            )
            self.assertTrue(manifest["production_isolated"])
            self.assertFalse(manifest["next_experiment"]["production_publication"])

    def test_invalid_holdout_dates_fail_before_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.parquet"
            holdout = root / "holdout.parquet"
            pd.DataFrame({"date": pd.bdate_range("2025-07-01", periods=80)}).to_parquet(
                training, index=False
            )
            pd.DataFrame({"date": pd.bdate_range("2025-09-01", periods=60)}).to_parquet(
                holdout, index=False
            )
            with self.assertRaises(ValueError):
                run_frozen_holdout_cycle(
                    training,
                    holdout,
                    root / "missing-screening.json",
                    root / "output",
                    root / "state",
                )
            manifest = initialize_research_state(root / "state")
            self.assertEqual(manifest["holdout_claims"], [])

    def test_cycle_lock_rejects_concurrent_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            with _cycle_lock(state):
                with self.assertRaises(RuntimeError):
                    with _cycle_lock(state):
                        pass

    def test_no_standalone_report_consumption_api_is_exposed(self):
        from intraday_1400 import auto_research

        self.assertFalse(hasattr(auto_research, "consume_holdout_report"))
        self.assertFalse(hasattr(auto_research, "_CONTROLLED_CYCLE_TOKEN"))

    def test_overlapping_holdout_claim_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.parquet"
            labels_path.write_bytes(b"immutable-label-artifact")
            first = claim_holdout(
                root / "state", "first", "2025-11-03", "2026-01-23", labels_path
            )
            repeated = claim_holdout(
                root / "state", "first", "2025-11-03", "2026-01-23", labels_path
            )
            self.assertEqual(first, repeated)
            with self.assertRaises(RuntimeError):
                claim_holdout(
                    root / "state", "second", "2026-01-01", "2026-03-27", labels_path
                )


class DirectReturnExperimentTest(unittest.TestCase):
    @staticmethod
    def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame({"date": dates, "code": "600000"})

    @staticmethod
    def _selection_dates() -> pd.DatetimeIndex:
        development = pd.bdate_range("2026-01-28", "2026-04-13")
        development = development[:46].append(pd.DatetimeIndex([pd.Timestamp("2026-04-13")]))
        calibration = pd.bdate_range("2026-04-17", "2026-04-30")
        return development.append(calibration)

    def test_consumed_holdout_cannot_enter_recipe_selection(self):
        base = self._frame(pd.DatetimeIndex([pd.Timestamp("2025-11-03")]))
        selection = self._frame(self._selection_dates())
        with self.assertRaisesRegex(ValueError, "previously consumed holdout"):
            validate_direct_return_selection_labels(base, selection, self._selection_dates())

    def test_selection_artifact_rejects_holdout_or_purge_dates(self):
        base = self._frame(pd.DatetimeIndex([pd.Timestamp("2025-09-12")]))
        selection = self._frame(
            self._selection_dates().append(pd.DatetimeIndex([pd.Timestamp("2026-05-11")]))
        )
        with self.assertRaisesRegex(ValueError, "only registered development and calibration"):
            validate_direct_return_selection_labels(base, selection, self._selection_dates())

    def test_direct_return_features_use_only_frozen_screening_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "screening.json"
            payload = {"windows": [
                {
                    "train_end": "2025-08-06",
                    "selected": {"daily_asof_plus_minute_control": {
                        "asof_matched": ["asof__ret_5d"],
                        "minute": ["minute__m5_return"],
                    }},
                },
                {
                    "train_end": "2026-01-27",
                    "selected": {"daily_asof_plus_minute_control": {
                        "asof_matched": ["asof__future_choice"],
                        "minute": [],
                    }},
                },
            ]}
            report.write_text(json.dumps(payload), encoding="utf-8")
            window, _, _, features = direct_return_experiment._selected_features(report)
            self.assertEqual(window["train_end"], "2025-08-06")
            self.assertEqual(features, ["asof__ret_5d", "minute__m5_return"])

    def test_recipe_selection_requires_exact_registered_grid(self):
        candidates = []
        for target in DIRECT_TARGETS:
            for top_n in DIRECT_TOP_NS:
                candidates.append({
                    "target": target,
                    "top_n": top_n,
                    "metrics": {
                        "days": DIRECT_CALIBRATION_DAYS,
                        "mean_return": 0.001 if top_n == 10 else 0.0,
                        "compound_return": 0.01,
                        "max_drawdown": -0.02,
                    },
                })
        selected = select_direct_return_recipe(candidates)
        self.assertEqual(selected["top_n"], 10)
        with self.assertRaisesRegex(ValueError, "exact six"):
            select_direct_return_recipe(candidates[:-1])

    def test_holdout_requires_exact_registered_sixty_dates(self):
        dates = pd.bdate_range("2026-05-11", "2026-08-03")
        dates = dates[:59].append(pd.DatetimeIndex([pd.Timestamp("2026-08-03")]))
        self.assertEqual(len(validate_direct_return_holdout_labels(self._frame(dates), dates)), 60)
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            validate_direct_return_holdout_labels(self._frame(dates[:-1]), dates)
        wrong_calendar = dates.delete(10).append(pd.DatetimeIndex([pd.Timestamp("2026-05-24")])).sort_values()
        with self.assertRaisesRegex(ValueError, "prepared A-share trading calendar"):
            validate_direct_return_holdout_labels(self._frame(dates), wrong_calendar)

    def test_adaptive_labels_are_not_reported_as_fixed_t1_exits(self):
        predictions = {"minute": pd.DataFrame({
            "code": ["600000"], "date": [pd.Timestamp("2026-05-11")], "score": [1.0]
        })}
        labels = pd.DataFrame({
            "code": ["600000"],
            "date": [pd.Timestamp("2026-05-11")],
            "entry_buyable": [True],
            "target_net_ret_t1": [0.02],
            "target_outcome_observed_t1": [True],
        })
        records = direct_return_experiment._simulate_adaptive_label_race(
            predictions, labels, ExecutionConfig(top_n=1)
        )
        self.assertEqual(records.iloc[0]["exit_reason"], "adaptive_t3_exit")

    def test_split_manifest_rejects_replaced_selection_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.parquet"
            holdout = root / "holdout.parquet"
            source = root / "source.parquet"
            pd.DataFrame({"value": [1]}).to_parquet(selection, index=False)
            pd.DataFrame({"value": [2]}).to_parquet(holdout, index=False)
            pd.DataFrame({"value": [3]}).to_parquet(source, index=False)
            manifest = {
                "protocol": direct_return_experiment.PROTOCOL,
                "source": {"path": str(source.resolve()), "sha256": direct_return_experiment.artifact_hash(source)},
                "selection": {
                    "path": str(selection.resolve()),
                    "sha256": direct_return_experiment.artifact_hash(selection),
                },
                "holdout": {
                    "path": str(holdout.resolve()),
                    "sha256": direct_return_experiment.artifact_hash(holdout),
                },
            }
            manifest["manifest_hash"] = direct_return_experiment._canonical_hash(manifest)
            manifest_path = root / "label_split_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            direct_return_experiment.validate_split_manifest(
                manifest_path, selection, holdout
            )
            pd.DataFrame({"value": [99]}).to_parquet(selection, index=False)
            with self.assertRaisesRegex(RuntimeError, "selection labels"):
                direct_return_experiment.validate_split_manifest(
                    manifest_path, selection, holdout
                )

    def test_frozen_recipe_hash_rejects_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "development_report.json"
            report.write_text("{}", encoding="utf-8")
            code_paths = {
                "base_labels": report,
                "selection_labels": report,
                "split_manifest": report,
                "screening_report": report,
                "daily_prepared": report,
                "intraday_prepared": report,
                "controller_code": Path(direct_return_experiment.__file__),
                "model_head_code": Path(direct_return_experiment.__file__).with_name(
                    "fair_race_pipeline.py"
                ),
                "quant_model_code": Path(direct_return_experiment.__file__).parent.parent
                / "quant" / "model.py",
            }
            manifest = {
                "protocol": direct_return_experiment.PROTOCOL,
                "protocol_hash": direct_return_experiment._canonical_hash(
                    direct_return_experiment._protocol_payload()
                ),
                "selected_recipe": {
                    "target": DIRECT_TARGETS[0], "top_n": 10, "name": "stress_top10"
                },
                "features": ["asof__ret_5d", "minute__m5_return"],
                "feature_hash": direct_return_experiment._canonical_hash(
                    ["asof__ret_5d", "minute__m5_return"]
                ),
                "training_recipe": direct_return_experiment.TRAINING_RECIPE,
                "training_recipe_hash": direct_return_experiment._canonical_hash(
                    direct_return_experiment.TRAINING_RECIPE
                ),
                "selection_input_hashes": {
                    name: {"path": str(path), "sha256": direct_return_experiment.artifact_hash(path)}
                    for name, path in code_paths.items()
                },
                "development_report": str(report),
                "development_report_hash": direct_return_experiment.artifact_hash(report),
                "production_isolated": True,
                "human_approval_required": True,
                "production_publication": False,
            }
            manifest["freeze_hash"] = direct_return_experiment._canonical_hash(manifest)
            validate_direct_return_frozen_recipe(manifest)
            manifest["selected_recipe"]["top_n"] = 15
            with self.assertRaisesRegex(RuntimeError, "manifest was modified"):
                validate_direct_return_frozen_recipe(manifest)


class TargetRedesignTest(unittest.TestCase):
    def test_lightgbm_quantile_objective_is_explicit_and_bounded(self):
        self.assertTrue(callable(quant_model.train_ridge))
        self.assertTrue(callable(quant_model.train_elastic_net))
        self.assertTrue(callable(quant_model.train_ic_weighted))
        self.assertEqual(
            quant_model._lightgbm_objective_params("quantile", 0.20),
            {"objective": "quantile", "alpha": 0.20, "eval_metric": "quantile"},
        )
        self.assertEqual(
            quant_model._lightgbm_objective_params("regression", None),
            {"objective": "regression", "eval_metric": "l2"},
        )
        with self.assertRaises(ValueError):
            quant_model._lightgbm_objective_params("quantile", 1.0)
        with self.assertRaises(ValueError):
            quant_model._lightgbm_objective_params("regression", 0.2)

    def test_ridge_and_quantile_lightgbm_execute_changed_paths(self):
        dates = pd.date_range("2026-01-01", periods=70, freq="D")
        panel = pd.DataFrame({
            "code": [f"{index % 5:06d}" for index in range(70)],
            "date": dates,
            "feature": np.linspace(-1.0, 1.0, 70),
            "target": np.linspace(-0.03, 0.04, 70),
        })
        ridge = quant_model.train_ridge(
            panel, ["feature"], label_col="target",
            train_end="2026-03-01", valid_end="2026-03-11", predict_start="2026-03-02",
        )
        self.assertTrue(ridge.ok, ridge.message)
        captured = {}

        class FakeRegressor:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.best_iteration_ = 0

            def fit(self, x, y, **kwargs):
                return self

            def predict(self, x):
                return np.zeros(len(x), dtype=float)

        fake_lightgbm = mock.Mock(LGBMRegressor=FakeRegressor, early_stopping=mock.Mock())
        with mock.patch.dict("sys.modules", {"lightgbm": fake_lightgbm}):
            result = quant_model.train_lightgbm(
                panel, ["feature"], label_col="target",
                train_end="2026-03-01", valid_end="2026-03-11", predict_start="2026-03-02",
                objective="quantile", alpha=0.20, early_stopping_rounds=0,
            )
        self.assertTrue(result.ok, result.message)
        self.assertEqual(captured["objective"], "quantile")
        self.assertEqual(captured["alpha"], 0.20)

    def test_registered_split_uses_first_123_new_dates_by_position(self):
        dates = pd.bdate_range("2026-08-04", periods=130)
        split = target_redesign.registered_split(dates)
        self.assertEqual(len(split["development"]), 47)
        self.assertEqual(len(split["purge_1"]), 3)
        self.assertEqual(len(split["calibration"]), 10)
        self.assertEqual(len(split["purge_2"]), 3)
        self.assertEqual(len(split["holdout"]), 60)
        self.assertEqual(split["development"][0], dates[0])
        self.assertEqual(split["holdout"][-1], dates[122])
        with self.assertRaisesRegex(ValueError, "requires 123"):
            target_redesign.registered_split(dates[:122])
        calendar_days = pd.date_range("2026-08-04", periods=123, freq="D")
        with self.assertRaisesRegex(ValueError, "prepared trading sessions"):
            target_redesign.split_manifest(
                target_redesign.registered_split(calendar_days), dates
            )

    def test_partial_label_date_is_not_counted_as_mature(self):
        labels = pd.DataFrame({
            "code": ["600000"],
            "date": [pd.Timestamp("2026-08-04")],
            "adaptive_entry_buyable": [True],
            "adaptive_liquidated_by_t3": [True],
            "adaptive_realized_net_ret_t3": [0.01],
            "adaptive_stress_net_ret_t3": [0.01],
            "adaptive_horizon_observed_t3": [True],
        })
        expected = pd.DataFrame({
            "code": ["600000", "600001"],
            "date": [pd.Timestamp("2026-08-04"), pd.Timestamp("2026-08-04")],
        })
        dates = target_redesign.mature_dates_after_consumed(
            labels, pd.DatetimeIndex([pd.Timestamp("2026-08-04")]), expected
        )
        self.assertEqual(len(dates), 0)

    def test_target_columns_are_date_local_and_conditionally_masked(self):
        labels = pd.DataFrame({
            "code": ["600000", "600001", "600000", "600001"],
            "date": pd.to_datetime(["2026-08-04", "2026-08-04", "2026-08-05", "2026-08-05"]),
            "adaptive_entry_buyable": [True, False, True, True],
            "adaptive_liquidated_by_t3": [True, False, False, True],
            "adaptive_realized_net_ret_t3": [0.01, np.nan, np.nan, 0.03],
            "adaptive_stress_net_ret_t3": [0.01, 0.0, -0.10, 0.03],
            "adaptive_horizon_observed_t3": [True, True, True, True],
        })
        result = target_redesign.build_target_columns(labels)
        ranks = result.groupby("date")["target_cross_sectional_rank"].agg(["min", "max"])
        self.assertTrue((ranks["min"] == 0.0).all())
        self.assertTrue((ranks["max"] == 0.5).all())
        self.assertEqual(int(result["target_exit_t3_given_entry"].notna().sum()), 3)
        self.assertEqual(int(result["target_conditional_return"].notna().sum()), 2)

    def test_conditional_payoff_uses_entry_exit_and_stress_tail(self):
        score = target_redesign.conditional_payoff_scores(
            pd.Series([1.0, 0.5]),
            pd.Series([1.0, 0.0]),
            pd.Series([0.02, 0.08]),
        )
        self.assertAlmostEqual(float(score.iloc[0]), 0.02)
        self.assertAlmostEqual(float(score.iloc[1]), -0.05)

    def test_ready_split_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            calendar = pd.bdate_range("2026-08-04", periods=123)
            split = target_redesign.registered_split(calendar)
            manifest = target_redesign.split_manifest(split, calendar)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            state = {"split_manifest": {
                "path": str(path.resolve()),
                "sha256": target_redesign.artifact_hash(path),
            }}
            target_redesign._validate_ready_split(state)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                target_redesign._validate_ready_split(state)

    def test_readiness_waits_without_new_mature_labels_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_state = root / "parent_state.json"
            parent_report = root / "parent_report.json"
            execution = root / "execution.parquet"
            daily = root / "daily.parquet"
            input_hashes = {"labels": {"sha256": "abc"}}
            freeze_hash = "frozen-parent"
            parent_report.write_text(json.dumps({
                "next_branch": "target_redesign",
                "holdout_end": "2026-08-03",
                "freeze_hash": freeze_hash,
                "input_hashes": input_hashes,
            }), encoding="utf-8")
            pd.DataFrame({"value": [1]}).to_parquet(execution, index=False)
            pd.DataFrame({"value": [2]}).to_parquet(daily, index=False)
            parent = {
                "protocol": "intraday_1400_direct_return_v1",
                "status": "consumed",
                "start": "2026-05-11",
                "end": "2026-08-03",
                "input_hashes": input_hashes,
                "claim_hash": direct_return_experiment._canonical_hash(input_hashes),
                "freeze_hash": freeze_hash,
                "production_publication": False,
                "artifacts": {
                    "report": {
                        "path": str(parent_report.resolve()),
                        "sha256": target_redesign.artifact_hash(parent_report),
                    },
                    "execution_records": {
                        "path": str(execution.resolve()),
                        "sha256": target_redesign.artifact_hash(execution),
                    },
                    "account_daily_returns": {
                        "path": str(daily.resolve()),
                        "sha256": target_redesign.artifact_hash(daily),
                    },
                },
            }
            parent["state_hash"] = target_redesign._canonical_hash(parent)
            parent_state.write_text(json.dumps(parent), encoding="utf-8")
            state = target_redesign.initialize_or_refresh(
                root / "state", parent_state, parent_report
            )
            self.assertEqual(state["status"], "awaiting_123_mature_dates")
            self.assertEqual(state["available_mature_dates"], 0)
            self.assertFalse(state["production_publication"])
            repeated = target_redesign.initialize_or_refresh(
                root / "state", parent_state, parent_report
            )
            self.assertEqual(state, repeated)


class TargetRedesignBackfillTest(unittest.TestCase):
    @staticmethod
    def _labels(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
        frame = index.to_frame(index=False)
        frame["adaptive_entry_buyable"] = True
        frame["adaptive_liquidated_by_t3"] = True
        frame["adaptive_realized_net_ret_t3"] = 0.01
        frame["adaptive_stress_net_ret_t3"] = 0.01
        frame["adaptive_horizon_observed_t3"] = True
        return frame

    def test_backfill_folds_are_purged_and_cover_174_oos_days(self):
        dates = pd.bdate_range("2025-07-01", periods=266)
        folds = target_redesign_backfill.registered_folds(dates)
        self.assertEqual([len(fold["oos"]) for fold in folds], [40, 40, 47, 47])
        self.assertEqual(sum(len(fold["oos"]) for fold in folds), 174)
        for fold in folds:
            self.assertEqual(len(fold["purge"]), 3)
            self.assertLess(fold["train"][-1], fold["purge"][0])
            self.assertLess(fold["purge"][-1], fold["oos"][0])

    def test_backfill_family_selection_requires_complete_fixed_grid(self):
        comparison = {
            family: {
                "days": 174,
                "mean_return": 0.001 if family == "cross_sectional_rank" else 0.0,
                "compound_return": 0.01,
                "max_drawdown": -0.02,
                "mean_filled_names": 10.0,
            }
            for family in target_redesign_backfill.FAMILY_ORDER
        }
        self.assertEqual(
            target_redesign_backfill.select_family(comparison), "cross_sectional_rank"
        )
        comparison.pop("conditional_payoff")
        with self.assertRaisesRegex(ValueError, "all three"):
            target_redesign_backfill.select_family(comparison)

    def test_backfill_overlap_must_agree_and_is_not_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates_a = pd.bdate_range("2025-07-01", periods=143)
            dates_b = pd.bdate_range(dates_a[-20], periods=143)
            first = root / "first.parquet"
            second = root / "second.parquet"
            self._labels(dates_a, ["600000"]).to_parquet(first, index=False)
            self._labels(dates_b, ["600000"]).to_parquet(second, index=False)
            combined = target_redesign_backfill.combine_historical_labels([first, second])
            self.assertEqual(combined["date"].nunique(), 266)
            payload = target_redesign_backfill.protocol_payload()
            self.assertFalse(payload["historical_intervals_are_untouched"])
            self.assertFalse(payload["historical_results_may_promote_production"])
            changed = self._labels(dates_b, ["600000"])
            changed.loc[changed.index[0], "adaptive_stress_net_ret_t3"] = -0.01
            changed.to_parquet(second, index=False)
            combined_changed = target_redesign_backfill.combine_historical_labels([first, second])
            self.assertGreater(combined_changed.attrs["overlap_conflict_count"], 0)
            self.assertAlmostEqual(
                float(combined_changed.loc[
                    (combined_changed["date"] == dates_b[0])
                    & (combined_changed["code"] == "600000"),
                    "adaptive_stress_net_ret_t3",
                ].iloc[0]),
                -0.01,
            )


class DailyMinuteEnhancementTest(unittest.TestCase):
    def test_candidate_grid_keeps_identical_daily_base(self):
        base = ["asof__ret_5d", "asof__volatility_20"]
        minute = [
            "minute__m5_ret_15",
            "minute__m5_path_length",
            "minute__m5_volume_hhi",
            "minute__m5_downside_semivar",
            "minute__m5_return_autocorr_1",
            "minute__m5_realized_vol_market_rank",
        ]
        grid = daily_minute_enhancement.candidate_features(base, minute)
        expected = {
            daily_minute_enhancement.BASELINE,
            daily_minute_enhancement.ALL_MINUTE,
            *[f"daily_plus_{name}" for name in daily_minute_enhancement.MINUTE_FAMILIES],
        }
        self.assertEqual(set(grid), expected)
        self.assertEqual(grid[daily_minute_enhancement.BASELINE], base)
        for name, features in grid.items():
            self.assertEqual(features[:len(base)], base, name)

    @staticmethod
    def _metrics(mean_return: float, drawdown: float = -0.10, filled: float = 10.0) -> dict:
        return {
            "mean_return": mean_return,
            "compound_return": mean_return * 100,
            "max_drawdown": drawdown,
            "mean_filled_names": filled,
        }

    def test_selection_requires_three_fold_wins_and_fill_gate(self):
        names = {
            daily_minute_enhancement.BASELINE,
            daily_minute_enhancement.ALL_MINUTE,
            *[f"daily_plus_{name}" for name in daily_minute_enhancement.MINUTE_FAMILIES],
        }
        aggregate = {name: self._metrics(0.0) for name in names}
        aggregate["daily_plus_risk"] = self._metrics(0.001)
        aggregate["daily_plus_speed"] = self._metrics(0.002, filled=4.0)
        folds = []
        for index in range(4):
            models = {name: self._metrics(0.0) for name in names}
            models["daily_plus_risk"] = self._metrics(0.001 if index < 3 else -0.001)
            models["daily_plus_speed"] = self._metrics(0.002)
            folds.append({"name": f"wf{index + 1}", "models": models})
        decision = daily_minute_enhancement.select_enhancement(aggregate, folds)
        self.assertEqual(decision["status"], "enhancement_selected")
        self.assertEqual(decision["selected"], "daily_plus_risk")

    def test_build_panel_freezes_the_common_labeled_universe(self):
        prepared = pd.DataFrame({
            "date": pd.to_datetime(["2025-07-01"]),
            "code": ["000001"],
            "signal_eligible": [True],
            "daily_target_ret_1d": [0.01],
        })
        labels = pd.DataFrame({
            "date": pd.to_datetime(["2025-07-01", "2025-07-01"]),
            "code": ["000001", "000002"],
            "adaptive_stress_net_ret_t3": [0.02, -0.01],
        })
        intraday_keys = labels[["date", "code"]].copy()
        with mock.patch.object(
            daily_minute_enhancement, "load_joined_prepared", return_value=(prepared, {})
        ), mock.patch.object(
            daily_minute_enhancement, "_intraday_eligible_keys", return_value=intraday_keys
        ):
            panel, hashes = daily_minute_enhancement._build_panel(
                labels, Path("daily"), Path("minute"), [], []
            )
        self.assertEqual(panel["code"].tolist(), ["000001"])
        self.assertEqual(
            panel.attrs["prepared_vs_label_key_counts"]["joined_prepared_vs_labels"]["right_only"],
            1,
        )
        self.assertEqual(
            hashes["final_matched"]["2025-07-01"],
            daily_minute_enhancement._canonical_hash(["000001"]),
        )
        self.assertIn("final_all_keys", hashes)

    def test_build_panel_rejects_labels_outside_intraday_eligibility(self):
        prepared = pd.DataFrame({
            "date": pd.to_datetime(["2025-07-01"]),
            "code": ["000001"],
            "signal_eligible": [True],
            "daily_target_ret_1d": [0.01],
        })
        labels = pd.DataFrame({
            "date": pd.to_datetime(["2025-07-01"]),
            "code": ["000001"],
            "adaptive_stress_net_ret_t3": [0.02],
        })
        with mock.patch.object(
            daily_minute_enhancement, "load_joined_prepared", return_value=(prepared, {})
        ), mock.patch.object(
            daily_minute_enhancement,
            "_intraday_eligible_keys",
            return_value=pd.DataFrame({"date": pd.to_datetime([]), "code": []}),
        ):
            with self.assertRaisesRegex(ValueError, "outside intraday signal eligibility"):
                daily_minute_enhancement._build_panel(
                    labels, Path("daily"), Path("minute"), [], []
                )

    def test_causal_screening_recomputes_features_from_verified_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "feature_manifest.json"
            manifest.write_text(
                json.dumps({"features": ["ret_5d", "m5_ret_15m"]}), encoding="utf-8"
            )
            panel = pd.DataFrame({
                "date": pd.to_datetime(["2025-08-06"]),
                "code": ["000001"],
            })
            selected = {
                "daily_asof_plus_minute_control": {
                    "asof_matched": ["asof__ret_5d"],
                    "minute": ["minute__m5_ret_15m"],
                }
            }
            old_report_features = (["asof__old"], ["minute__old"])
            with mock.patch.object(
                daily_minute_enhancement,
                "_selected_features",
                return_value=({"train_end": "2025-08-06"}, *old_report_features, []),
            ), mock.patch.object(
                daily_minute_enhancement, "load_joined_prepared", return_value=(panel, {})
            ), mock.patch.object(
                daily_minute_enhancement, "screen_window_features", return_value=selected
            ):
                _, base, minute, evidence = daily_minute_enhancement._causal_screened_features(
                    Path("old-screening.json"),
                    Path("daily"),
                    Path("minute"),
                    {"feature_manifest": {"path": str(manifest)}},
                )
            self.assertEqual(base, ["asof__ret_5d"])
            self.assertEqual(minute, ["minute__m5_ret_15m"])
            self.assertEqual(evidence["method"], "recomputed_from_verified_1355_prepared")

    def test_selection_rejects_incomplete_registered_folds(self):
        names = {
            daily_minute_enhancement.BASELINE,
            daily_minute_enhancement.ALL_MINUTE,
            *[f"daily_plus_{name}" for name in daily_minute_enhancement.MINUTE_FAMILIES],
        }
        aggregate = {name: self._metrics(0.001) for name in names}
        folds = [
            {"name": f"wf{index + 1}", "models": aggregate}
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "ordered registered four folds"):
            daily_minute_enhancement.select_enhancement(aggregate, folds)

    def test_prepared_provenance_rejects_wrong_cutoff_and_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_dir = root / "prepared_monthly"
            feature_dir = root / "features" / "2025-07"
            (root / "checkpoints").mkdir(parents=True)
            (root / "models").mkdir(parents=True)
            prepared_dir.mkdir()
            feature_dir.mkdir(parents=True)
            pd.DataFrame({"value": [1]}).to_parquet(prepared_dir / "2025-07.parquet")
            pd.DataFrame({"value": [1]}).to_parquet(feature_dir / "part.parquet")
            (root / "checkpoints" / "prepare_state.json").write_text(
                json.dumps({"2025-07": "forged"}), encoding="utf-8"
            )
            (root / "models" / "feature_manifest.json").write_text(
                json.dumps({"features": []}), encoding="utf-8"
            )
            with mock.patch.object(daily_minute_enhancement.config, "CUTOFF_TIME", "14:00"):
                with self.assertRaisesRegex(RuntimeError, "requires cutoff 13:55"):
                    daily_minute_enhancement._prepared_provenance(prepared_dir)
            with mock.patch.object(pipeline, "_nonprice_sources", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "does not attest"):
                    daily_minute_enhancement._prepared_provenance(prepared_dir)

    def test_research_paths_reject_production_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intraday = root / "prepared_monthly"
            with self.assertRaisesRegex(RuntimeError, "production or scheduler"):
                daily_minute_enhancement._validate_research_paths(
                    root / "active_quant" / "output", root / "state", intraday
                )
            with self.assertRaisesRegex(RuntimeError, "isolated intraday research root"):
                daily_minute_enhancement._validate_research_paths(
                    root.parent / "outside", root / "state", intraday
                )

    def test_existing_state_rejects_changed_invocation_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            state_dir = root / "state"
            intraday = root / "prepared_monthly"
            output.mkdir()
            state_dir.mkdir()
            intraday.mkdir()
            artifacts = {}
            for name in ("report", "execution_records", "daily_returns"):
                path = output / f"{name}.bin"
                path.write_bytes(name.encode("ascii"))
                artifacts[name] = {
                    "path": str(path.resolve()),
                    "sha256": daily_minute_enhancement.artifact_hash(path),
                }
            state = {
                "protocol": daily_minute_enhancement.PROTOCOL,
                "protocol_hash": daily_minute_enhancement._canonical_hash(
                    daily_minute_enhancement.protocol_payload()
                ),
                "untouched_holdout": False,
                "eligible_for_production": False,
                "human_approval_required": True,
                "production_publication": False,
                "input_hashes": {"version": "old"},
                **artifacts,
            }
            state["state_hash"] = daily_minute_enhancement._canonical_hash(state)
            (state_dir / "manifest.json").write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(
                daily_minute_enhancement, "_input_hashes", return_value={"version": "new"}
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match the frozen inputs"):
                    daily_minute_enhancement._run_enhancement_race_unlocked(
                        [], root / "screening.json", output, state_dir,
                        root / "daily", intraday,
                    )

    def test_validate_state_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for name in ("report", "execution_records", "daily_returns"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                artifacts[name] = {
                    "path": str(path.resolve()),
                    "sha256": daily_minute_enhancement.artifact_hash(path),
                }
            state = {
                "protocol": daily_minute_enhancement.PROTOCOL,
                "protocol_hash": daily_minute_enhancement._canonical_hash(
                    daily_minute_enhancement.protocol_payload()
                ),
                "untouched_holdout": False,
                "eligible_for_production": False,
                "human_approval_required": True,
                "production_publication": False,
                "input_hashes": {},
                **artifacts,
            }
            state["state_hash"] = daily_minute_enhancement._canonical_hash(state)
            (root / "report").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "report artifact changed"):
                daily_minute_enhancement.validate_state(state, verify_inputs=False)

    def test_no_incremental_candidate_keeps_daily_baseline(self):
        names = {
            daily_minute_enhancement.BASELINE,
            daily_minute_enhancement.ALL_MINUTE,
            *[f"daily_plus_{name}" for name in daily_minute_enhancement.MINUTE_FAMILIES],
        }
        aggregate = {name: self._metrics(-0.001) for name in names}
        aggregate[daily_minute_enhancement.BASELINE] = self._metrics(0.001)
        folds = [{"name": f"wf{index + 1}", "models": aggregate} for index in range(4)]
        decision = daily_minute_enhancement.select_enhancement(aggregate, folds)
        self.assertEqual(decision["status"], "no_enhancement_passed")
        self.assertEqual(decision["selected"], daily_minute_enhancement.BASELINE)
        self.assertEqual(decision["next_branch"], "minute_feature_residualization")


class MinuteFeatureResidualizationTest(unittest.TestCase):
    @staticmethod
    def _panel() -> pd.DataFrame:
        dates = pd.to_datetime(["2025-07-01"] * 30 + ["2025-07-02"] * 30)
        control = np.linspace(-1.0, 1.0, 60)
        return pd.DataFrame({
            "date": dates,
            "code": [f"{index:06d}" for index in range(60)],
            "asof__control": control,
            "minute__feature": 2.0 * control + np.sin(np.arange(60)) * 0.01,
        })

    def test_residualizer_uses_train_dates_only(self):
        panel = self._panel()
        train_dates = pd.DatetimeIndex(pd.to_datetime(["2025-07-01"]))
        _, features, evidence = minute_feature_residualization._fit_transform_residuals(
            panel, train_dates, ["asof__control"], ["minute__feature"], 10.0
        )
        changed = panel.copy()
        changed.loc[changed["date"] > train_dates[-1], "asof__control"] = 1000.0
        changed.loc[changed["date"] > train_dates[-1], "minute__feature"] = -1000.0
        _, changed_features, changed_evidence = (
            minute_feature_residualization._fit_transform_residuals(
                changed, train_dates, ["asof__control"], ["minute__feature"], 10.0
            )
        )
        self.assertEqual(features, ["minute_resid__feature"])
        self.assertEqual(changed_features, features)
        self.assertEqual(evidence["coefficient_hashes"], changed_evidence["coefficient_hashes"])
        self.assertEqual(evidence["control_mean_hash"], changed_evidence["control_mean_hash"])
        self.assertEqual(evidence["control_scale_hash"], changed_evidence["control_scale_hash"])

    def test_residualizer_preserves_raw_minute_features(self):
        panel = self._panel()
        original = panel["minute__feature"].copy()
        transformed, features, _ = minute_feature_residualization._fit_transform_residuals(
            panel,
            pd.DatetimeIndex(pd.to_datetime(["2025-07-01"])),
            ["asof__control"],
            ["minute__feature"],
            0.0,
        )
        pd.testing.assert_series_equal(transformed["minute__feature"], original)
        self.assertTrue(np.isfinite(transformed[features[0]]).all())

    @staticmethod
    def _metrics(mean_return: float, filled: float = 10.0, drawdown: float = -0.10) -> dict:
        return {
            "mean_return": mean_return,
            "compound_return": mean_return * 100,
            "max_drawdown": drawdown,
            "mean_filled_names": filled,
        }

    def test_residual_selection_keeps_fill_and_four_fold_gates(self):
        names = {
            daily_minute_enhancement.BASELINE,
            *minute_feature_residualization.RESIDUAL_CANDIDATES,
        }
        aggregate = {name: self._metrics(0.0) for name in names}
        aggregate["daily_plus_resid_all_ridge"] = self._metrics(0.002, filled=4.0)
        folds = []
        for index in range(4):
            models = {name: self._metrics(0.0) for name in names}
            models["daily_plus_resid_all_ridge"] = self._metrics(0.002)
            folds.append({"name": f"wf{index + 1}", "models": models})
        decision = minute_feature_residualization.select_residual_enhancement(aggregate, folds)
        self.assertEqual(decision["status"], "no_residual_enhancement_passed")
        self.assertEqual(decision["selected"], daily_minute_enhancement.BASELINE)
        with self.assertRaisesRegex(ValueError, "ordered registered four folds"):
            minute_feature_residualization.select_residual_enhancement(aggregate, folds[:3])
        incomplete = [{**fold, "models": dict(fold["models"])} for fold in folds]
        incomplete[0]["models"].pop("daily_plus_resid_all_ols")
        with self.assertRaisesRegex(ValueError, "incomplete candidate grid"):
            minute_feature_residualization.select_residual_enhancement(aggregate, incomplete)
        nonfinite = [{**fold, "models": {k: dict(v) for k, v in fold["models"].items()}} for fold in folds]
        nonfinite[0]["models"][daily_minute_enhancement.BASELINE]["mean_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            minute_feature_residualization.select_residual_enhancement(aggregate, nonfinite)

    def test_residualizer_rejects_estimate_above_memory_budget(self):
        panel = self._panel()
        recipe = minute_feature_residualization.RESIDUALIZER_RECIPE
        original = recipe["maximum_estimated_peak_bytes"]
        recipe["maximum_estimated_peak_bytes"] = 1
        try:
            with self.assertRaisesRegex(RuntimeError, "exceeds protocol budget"):
                minute_feature_residualization._fit_transform_residuals(
                    panel,
                    pd.DatetimeIndex(pd.to_datetime(["2025-07-01"])),
                    ["asof__control"],
                    ["minute__feature"],
                    10.0,
                )
        finally:
            recipe["maximum_estimated_peak_bytes"] = original


class DailyH1BuyabilityEnhancementTest(unittest.TestCase):
    def test_score_construction_is_train_label_and_outcome_invariant(self):
        date = pd.Timestamp("2026-08-04")
        daily = pd.DataFrame({"code": ["600001", "600002", "600003"], "date": date,
                              "score": [3.0, 2.0, 1.0]})
        buy = pd.DataFrame({"code": daily["code"], "date": date, "pred": [0.2, 0.5, 0.9]})
        first = daily_h1_buyability_enhancement.build_buyability_scores(daily, buy)
        changed = daily.copy()
        changed["realized_fill"] = [False, True, False]
        second = daily_h1_buyability_enhancement.build_buyability_scores(changed, buy)
        for name in daily_h1_buyability_enhancement.CANDIDATES:
            pd.testing.assert_frame_equal(first[name], second[name])
        merged = daily.rename(columns={"score": "daily_h1"}).merge(
            buy.rename(columns={"pred": "p_buy"}), on=["code", "date"]
        )
        h1_z = daily_h1_buyability_enhancement._zscore_by_date(merged, "daily_h1")
        merged["buy_logit"] = daily_h1_buyability_enhancement.clipped_logit(merged["p_buy"])
        buy_z = daily_h1_buyability_enhancement._zscore_by_date(merged, "buy_logit")
        expected_75 = 0.75 * h1_z + 0.25 * buy_z
        self.assertTrue(np.allclose(first["h1_buy_zblend_75"]["score"], expected_75))
        self.assertFalse(daily_h1_buyability_enhancement.protocol_payload()["four_point_five_percent_filter"])

    def test_constrained_top50_uses_deterministic_h1_and_code_ties_without_refill(self):
        date = pd.Timestamp("2026-08-04")
        daily = pd.DataFrame({"code": ["600002", "600001", "600003"], "date": date,
                              "score": [1.0, 1.0, 1.0]})
        buy = pd.DataFrame({"code": daily["code"], "date": date, "pred": [0.5, 0.5, 0.5]})
        result = daily_h1_buyability_enhancement.build_buyability_scores(daily, buy)["h1_buy_constrained_top50"]
        self.assertEqual(result["code"].tolist(), ["600001", "600002", "600003"])
        self.assertEqual(len(result), 3)
        codes = [f"{index:06d}" for index in range(60)]
        daily_60 = pd.DataFrame({
            "code": codes,
            "date": date,
            "score": np.arange(60, 0, -1, dtype=float),
        })
        buy_60 = pd.DataFrame({
            "code": codes,
            "date": date,
            "pred": [0.1] * 50 + [0.999] * 10,
        })
        constrained = daily_h1_buyability_enhancement.build_buyability_scores(
            daily_60, buy_60
        )["h1_buy_constrained_top50"]
        self.assertEqual(len(constrained), 50)
        self.assertFalse(set(codes[50:]) & set(constrained["code"]))

    def test_selection_has_exact_grid_and_three_fold_gate(self):
        names = set(daily_h1_buyability_enhancement.CANDIDATES)
        def metrics(value, filled=10.0):
            return {"mean_return": value, "compound_return": value * 10, "max_drawdown": -0.10,
                    "mean_filled_names": filled}
        aggregate = {name: metrics(0.0) for name in names}
        aggregate["h1_buy_zblend_50"] = metrics(0.01)
        folds = []
        for i, fold in enumerate(target_redesign_backfill.FOLD_POSITIONS):
            models = {name: metrics(0.0) for name in names}
            models["h1_buy_zblend_50"] = metrics(0.01 if i < 3 else -0.01)
            folds.append({"name": fold["name"], "models": models})
        decision = daily_h1_buyability_enhancement.select_enhancement(aggregate, folds)
        self.assertEqual(decision["selected"], "h1_buy_zblend_50")
        with self.assertRaisesRegex(ValueError, "exact registered candidate grid"):
            daily_h1_buyability_enhancement.select_enhancement({**aggregate, "extra": metrics(0)}, folds)

    def test_classifier_api_preserves_existing_positional_order(self):
        parameters = list(inspect.signature(quant_model.train_binary_classifier).parameters)
        self.assertEqual(parameters[:14], [
            "panel", "features", "label_col", "classifier", "train_end", "valid_end",
            "predict_start", "decay_half_life_days", "min_weight", "minority_weight",
            "n_estimators", "learning_rate", "max_train_rows", "n_jobs",
        ])
        self.assertEqual(parameters[14:], ["predict_end", "enforce_max_train_rows"])

    def test_classifier_prediction_scope_is_current_oos_only(self):
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "code": ["000001"] * 4,
        })
        _, _, predict = quant_model._split_train_valid_predict(
            panel,
            train_end="2026-01-01",
            valid_end="2026-01-04",
            predict_start="2026-01-02",
            predict_end="2026-01-03",
        )
        self.assertEqual(
            predict["date"].tolist(),
            pd.to_datetime(["2026-01-02", "2026-01-03"]).tolist(),
        )

    def test_classifier_recipe_is_fixed_and_train_fold_only(self):
        recipe = daily_h1_buyability_enhancement.MODEL_RECIPE["buyability_head"]
        self.assertEqual(recipe["minority_weight"], 1.0)
        self.assertEqual(recipe["n_estimators"], 160)
        self.assertEqual(recipe["learning_rate"], 0.02)
        self.assertEqual(recipe["max_train_rows"], 400000)
        self.assertTrue(recipe["enforce_max_train_rows"])
        self.assertEqual(recipe["predict_scope"], "current_fold_oos_only")
        self.assertFalse(recipe["oos_calibration"])
        self.assertFalse(recipe["oos_threshold_selection"])
        self.assertEqual(daily_h1_buyability_enhancement.BUY_LABEL, "adaptive_entry_buyable")


if __name__ == "__main__":
    unittest.main()
