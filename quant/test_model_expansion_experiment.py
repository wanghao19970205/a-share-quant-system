from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from quant import daily_update
from quant import full_train_batched
from quant import model_expansion_experiment as experiment
from quant import scheduled_workflow
from quant import shadow_leg_evaluation
from quant import warehouse
from quant import watchlist_grid
from quant.factors import engineering
from stock_analyzer import amazingdata_source


class IntradaySourcePriorityTest(unittest.TestCase):
    def test_intraday_prefers_amazingdata(self):
        frames = {
            "600001": pd.DataFrame({
                "date": pd.to_datetime(["2026-07-21"]),
                "open": [10.0], "high": [10.2], "low": [9.9], "close": [10.1],
            }),
            "600002": pd.DataFrame({
                "date": pd.to_datetime(["2026-07-21"]),
                "open": [20.0], "high": [20.2], "low": [19.9], "close": [20.1],
            }),
        }
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_daily_prices", return_value=frames) as batch, \
                mock.patch.object(daily_update, "_run_batch", return_value={
                    "ok": 2, "fail": 0, "rows": 2, "failures": [],
                }), \
                mock.patch.object(daily_update.datafeed, "market_spot") as free_spot:
            result = daily_update.update_intraday_spot(["600001", "600002"], workers=2)

        batch.assert_called_once()
        free_spot.assert_not_called()
        self.assertEqual(result["source"], "AmazingData")

    def test_intraday_falls_back_when_amazingdata_coverage_is_low(self):
        spot = pd.DataFrame({
            "代码": ["600001", "600002"],
            "最新价": [10.0, 20.0],
        })
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_daily_prices", return_value={}), \
                mock.patch.object(daily_update.datafeed, "market_spot", return_value=spot) as free_spot, \
                mock.patch.object(daily_update, "_run_batch", side_effect=[
                    {"ok": 0, "fail": 2, "rows": 0, "failures": ["600001:EmptyBrokerData"]},
                    {"ok": 2, "fail": 0, "rows": 2, "failures": []},
                ]):
            result = daily_update.update_intraday_spot(["600001", "600002"], workers=2)

        free_spot.assert_called_once()
        self.assertEqual(result["ok"], 2)

    def test_intraday_falls_back_when_amazingdata_is_unavailable(self):
        spot = pd.DataFrame({
            "代码": ["600001", "600002"],
            "最新价": [10.0, 20.0],
        })
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=False), \
                mock.patch.object(daily_update.datafeed, "market_spot", return_value=spot) as free_spot, \
                mock.patch.object(daily_update, "_run_batch", return_value={
                    "ok": 2, "fail": 0, "rows": 2,
                }):
            result = daily_update.update_intraday_spot(["600001", "600002"], workers=2)

        free_spot.assert_called_once()
        self.assertEqual(result["ok"], 2)
    def test_intraday_corrupt_local_price_only_fails_that_code(self):
        row = pd.Series({
            "最新价": 10.0,
            "今开": 9.9,
            "最高": 10.1,
            "最低": 9.8,
        })
        with mock.patch.object(
            daily_update.warehouse,
            "load_price",
            side_effect=ValueError("corrupt parquet"),
        ), mock.patch.object(daily_update.warehouse, "save_price") as save_price:
            result = daily_update._update_one_intraday_spot(
                "600001",
                row,
                pd.Timestamp("2026-07-21").date(),
            )

        self.assertEqual(result, ("600001", "ValueError", 0))
        save_price.assert_not_called()

    def test_amazingdata_daily_batch_uses_bounded_chunks(self):
        symbols = [f"{600000 + index:06d}" for index in range(5)]
        calls: list[list[str]] = []

        def query(codes, **kwargs):
            calls.append(codes)
            return {
                code: pd.DataFrame({
                    "date": ["2026-07-21"],
                    "open": [10.0], "high": [10.2], "low": [9.8], "close": [10.1],
                })
                for code in codes
            }

        fake_ad = mock.Mock()
        fake_ad.constant.Period.day.value = "day"
        with mock.patch.object(amazingdata_source, "_ensure_login", return_value=True), \
                mock.patch.object(amazingdata_source, "_market") as market, \
                mock.patch.object(amazingdata_source, "_ad", fake_ad), \
                mock.patch.object(amazingdata_source, "sdk_call", side_effect=lambda fn, codes, **kwargs: query(codes, **kwargs)), \
                mock.patch.dict(os.environ, {"AMAZINGDATA_KLINE_BATCH_SIZE": "2"}):
            result = amazingdata_source.fetch_daily_batch(symbols, "20260720", "20260721")

        self.assertEqual([len(chunk) for chunk in calls], [2, 2, 1])
        self.assertEqual(set(result), set(symbols))

    def test_price_save_replaces_file_atomically(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(warehouse.config, "PRICE_DIR", directory), \
                mock.patch.object(warehouse.config, "ensure_dirs"):
            path = Path(directory) / "600001.parquet"
            path.write_bytes(b"old")
            frame = pd.DataFrame({"code": ["600001"], "date": pd.to_datetime(["2026-07-21"]), "close": [10.0]})

            warehouse.save_price("600001", frame)

            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class PublishAccelerationTest(unittest.TestCase):
    def test_daily_update_explicit_force_latest_reaches_price_command(self):
        args = types.SimpleNamespace(
            skip_daily_update=False,
            universe="mainboard_active",
            update_workers=12,
            lookback_days=5,
            event_window_days=30,
            snapshot_dir="/tmp/snapshots",
            skip_valuation=True,
            skip_events=False,
            skip_fundamentals=True,
            skip_snapshots=True,
            force_latest_price=True,
            intraday_spot=False,
            dry_run=True,
        )

        with mock.patch("builtins.print") as output:
            scheduled_workflow.run_daily_update(args, {})

        command = output.call_args.args[0]
        self.assertIn("--force-latest", command)
        self.assertNotIn("--intraday-spot", command)

    def test_price_freshness_is_restricted_to_update_universe(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(scheduled_workflow.config, "PRICE_DIR", directory):
            for code, date in {
                "600001": "2026-07-22",
                "600002": "2026-07-22",
                "300001": "2026-07-21",
            }.items():
                pd.DataFrame({"date": pd.to_datetime([date])}).to_parquet(
                    Path(directory) / f"{code}.parquet", index=False,
                )

            latest, coverage, count = scheduled_workflow._price_freshness(
                ["600001", "600002"],
            )

        self.assertEqual(latest, pd.Timestamp("2026-07-22"))
        self.assertEqual(coverage, 1.0)
        self.assertEqual(count, 2)

    def test_daily_update_sigsegv_recovers_only_missing_codes_with_free_sources(self):
        args = types.SimpleNamespace(
            skip_daily_update=False,
            universe="mainboard_active",
            update_workers=12,
            lookback_days=5,
            event_window_days=30,
            snapshot_dir="/tmp/snapshots",
            skip_valuation=True,
            skip_events=False,
            skip_fundamentals=True,
            skip_snapshots=True,
            force_latest_price=True,
            intraday_spot=False,
            dry_run=False,
        )
        freshness = [
            (pd.Timestamp("2026-07-22"), 0.5, 2),
            (pd.Timestamp("2026-07-22"), 1.0, 2),
        ]
        frames = {
            "600001": pd.DataFrame({"date": pd.to_datetime(["2026-07-22"])}),
            "600002": pd.DataFrame({"date": pd.to_datetime(["2026-07-21"])}),
        }
        calls: list[tuple[list[str], dict[str, str]]] = []

        def run(command, **kwargs):
            calls.append((command, kwargs["env"]))
            return types.SimpleNamespace(
                returncode=-11 if len(calls) == 1 else 0,
            )

        with mock.patch.object(scheduled_workflow, "_latest_price_date", return_value=pd.Timestamp("2026-07-21")), \
                mock.patch.object(scheduled_workflow, "_price_freshness", side_effect=freshness), \
                mock.patch.object(scheduled_workflow.datafeed, "refresh_mainboard_universe"), \
                mock.patch.object(scheduled_workflow.datafeed, "universe", return_value=list(frames)), \
                mock.patch.object(scheduled_workflow.pd, "read_parquet", side_effect=lambda path, **_: frames[Path(path).stem]), \
                mock.patch.object(scheduled_workflow.subprocess, "run", side_effect=run), \
                mock.patch.object(scheduled_workflow, "_write_recovery_codes", return_value=mock.Mock(unlink=mock.Mock())):
            scheduled_workflow.run_daily_update(args, {"BASE": "1"})

        recovery_command, recovery_env = calls[1]
        self.assertIn("--codes-file", recovery_command)
        self.assertEqual(recovery_command[recovery_command.index("--workers") + 1], "4")
        self.assertEqual(recovery_env["AMAZINGDATA_AUTO_LOGIN"], "0")
        self.assertIn("--skip-events", recovery_command)

    def test_parquet_max_date_uses_footer_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "price.parquet"
            pd.DataFrame({
                "date": pd.to_datetime(["2026-07-18", "2026-07-21"]),
                "close": [10.0, 10.2],
            }).to_parquet(path, index=False, row_group_size=1)

            with mock.patch.object(
                scheduled_workflow.pd,
                "read_parquet",
            ) as read_parquet:
                latest = scheduled_workflow._parquet_max_date(path)

        self.assertEqual(latest, pd.Timestamp("2026-07-21"))
        read_parquet.assert_not_called()

    def test_parquet_max_date_falls_back_without_statistics(self):
        path = Path("price.parquet")
        parquet = mock.Mock()
        parquet.schema_arrow.get_field_index.return_value = -1
        frame = pd.DataFrame({"date": pd.to_datetime(["2026-07-21"])})

        with mock.patch.object(
            scheduled_workflow.pq,
            "ParquetFile",
            return_value=parquet,
        ), mock.patch.object(
            scheduled_workflow.pd,
            "read_parquet",
            return_value=frame,
        ) as read_parquet:
            latest = scheduled_workflow._parquet_max_date(path)

        self.assertEqual(latest, pd.Timestamp("2026-07-21"))
        read_parquet.assert_called_once_with(path, columns=["date"])

    def test_merge_active_predictions_reuses_fresh_frame_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active.parquet"
            source = root / "source.parquet"
            fresh = pd.DataFrame({
                "code": ["600001"],
                "date": pd.to_datetime(["2026-07-21"]),
                "prediction": [0.2],
            })
            old = pd.DataFrame({
                "code": ["600001"],
                "date": pd.to_datetime(["2026-07-18"]),
                "prediction": [0.1],
                "legacy": [1.0],
            })
            old.to_parquet(active, index=False)

            scheduled_workflow.merge_active_predictions(
                active,
                source,
                active,
                fresh_frame=fresh,
            )

            result = pd.read_parquet(active)

        self.assertEqual(len(result), 2)
        self.assertNotIn("legacy", fresh.columns)
        self.assertEqual(fresh["code"].tolist(), ["600001"])

    def test_active_checks_can_reuse_price_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            active = Path(directory) / "active.parquet"
            pd.DataFrame({
                "date": pd.to_datetime(["2026-07-21"]),
            }).to_parquet(active, index=False)

            with mock.patch.object(
                scheduled_workflow,
                "_latest_price_date",
            ) as latest_price:
                scheduled_workflow.assert_active_is_latest(
                    active,
                    price_latest=pd.Timestamp("2026-07-21"),
                )

        latest_price.assert_not_called()


class DailyPanelAccelerationTest(unittest.TestCase):
    def test_incremental_price_factors_match_full_history(self):
        dates = pd.bdate_range("2018-01-02", periods=1900)
        steps = np.arange(len(dates))
        close = pd.Series(
            10.0
            + 0.002 * steps
            + np.sin(steps / 17)
            + 0.3 * np.cos(steps / 7)
        )
        volume = 1_000_000 + 1_000 * steps + 50_000 * np.sin(steps / 9)
        price = pd.DataFrame({
            "code": "600001",
            "date": dates,
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "turnover": 2.0 + np.sin(steps / 30),
        })
        output_start = dates[-22]

        with mock.patch.object(
            engineering.warehouse,
            "load_price",
            return_value=price,
        ) as load_price, mock.patch.object(
            engineering.warehouse,
            "load_price_tail",
        ) as load_price_tail:
            full = engineering._price_factors("600001")
            incremental = engineering._price_factors(
                "600001",
                start_date=output_start,
                warmup_rows=260,
            )

        self.assertEqual(load_price.call_count, 2)
        load_price_tail.assert_not_called()

        factor_columns = [
            column for column in full.columns
            if column not in {"code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"}
        ]
        expected = full[full["date"] >= output_start].reset_index(drop=True)
        actual = incremental.reset_index(drop=True)
        pd.testing.assert_frame_equal(
            actual[["code", "date", *factor_columns]],
            expected[["code", "date", *factor_columns]],
        )
        self.assertGreaterEqual(incremental["date"].min(), output_start)
        self.assertLess(len(incremental), len(full))

    def test_incremental_price_tail_uses_exact_sparse_warmup_rows(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(warehouse.config, "PRICE_DIR", directory):
            early = pd.bdate_range("2020-01-02", periods=270)
            recent = pd.bdate_range("2026-06-01", periods=35)
            dates = early.append(recent)
            close = pd.Series(np.linspace(8.0, 18.0, len(dates)))
            price = pd.DataFrame({
                "code": "600001",
                "date": dates,
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
                "amount": np.linspace(8_000_000, 36_000_000, len(dates)),
                "turnover": np.linspace(1.0, 3.0, len(dates)),
            })
            output_start = recent[-15]
            warehouse.save_price("600001", price)

            full = engineering._price_factors("600001")
            incremental = engineering._price_factors(
                "600001",
                start_date=output_start,
                warmup_rows=260,
            )

        expected = full[full["date"] >= output_start].reset_index(drop=True)
        actual = incremental.reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_incremental_price_tail_handles_less_than_warmup_history(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(warehouse.config, "PRICE_DIR", directory):
            dates = pd.bdate_range("2026-01-02", periods=80)
            close = pd.Series(np.linspace(8.0, 12.0, len(dates)))
            price = pd.DataFrame({
                "code": "600001",
                "date": dates,
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": np.linspace(1_000_000, 1_500_000, len(dates)),
                "amount": np.linspace(8_000_000, 18_000_000, len(dates)),
                "turnover": np.linspace(1.0, 2.0, len(dates)),
            })
            output_start = dates[-15]
            warehouse.save_price("600001", price)

            full = engineering._price_factors("600001")
            incremental = engineering._price_factors(
                "600001",
                start_date=output_start,
                warmup_rows=260,
            )

        expected = full[full["date"] >= output_start].reset_index(drop=True)
        actual = incremental.reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_incremental_panel_reuses_shared_factor_frames(self):
        dates = pd.bdate_range("2026-07-01", periods=5)
        price = pd.DataFrame({
            "code": "600001", "date": dates,
            "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
            "volume": 1000.0, "amount": 10000.0, "turnover": 1.0,
        })
        shared = {
            name: pd.DataFrame()
            for name in (
                "financial_yjbb", "income", "cashflow", "balance",
                "performance_forecast", "margin_underlying_szse",
                "block_trades", "lhb",
            )
        }
        with mock.patch.object(engineering.warehouse, "load_price", return_value=price), \
                mock.patch.object(engineering.warehouse, "load_valuation", return_value=pd.DataFrame()), \
                mock.patch.object(engineering, "_asof_report_factor") as report_loader, \
                mock.patch.object(engineering, "_forecast_events") as forecast_loader, \
                mock.patch.object(engineering, "_margin_underlying") as margin_loader, \
                mock.patch.object(engineering, "_event_counts") as event_loader:
            panel = engineering.build_panel(
                codes=["600001"],
                horizon=3,
                shared_factors=shared,
            )

        self.assertFalse(panel.empty)
        report_loader.assert_not_called()
        forecast_loader.assert_not_called()
        margin_loader.assert_not_called()
        event_loader.assert_not_called()

    def test_incremental_panel_filters_shared_factors_to_batch_codes(self):
        dates = pd.bdate_range("2026-07-01", periods=5)
        price = pd.DataFrame({
            "code": "600001", "date": dates,
            "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
            "volume": 1000.0, "amount": 10000.0, "turnover": 1.0,
        })
        shared = {
            name: pd.DataFrame()
            for name in (
                "financial_yjbb", "income", "cashflow", "balance",
                "performance_forecast", "margin_underlying_szse",
                "block_trades", "lhb",
            )
        }
        shared["block_trades"] = pd.DataFrame({
            "code": ["600001", "600002"],
            "date": [dates[-1], dates[-1]],
            "block_trade_cnt_1d": [1, 9],
        })
        with mock.patch.object(engineering.warehouse, "load_price", return_value=price), \
                mock.patch.object(engineering.warehouse, "load_valuation", return_value=pd.DataFrame()):
            panel = engineering.build_panel(
                codes=["600001"],
                horizon=3,
                shared_factors=shared,
            )

        self.assertEqual(set(panel["code"]), {"600001"})
        self.assertEqual(float(panel.iloc[-1]["block_trade_cnt_1d"]), 1.0)

    def test_recipe_signature_ignores_universe_file_churn(self):
        first = full_train_batched._recipe_signature(
            factors=["ret_20d", "volatility_20"],
            horizon=3,
            universe_codes=["600001", "600002"],
        )
        second = full_train_batched._recipe_signature(
            factors=["volatility_20", "ret_20d"],
            horizon=3,
            universe_codes=["600001", "600002", "001220"],
        )

        self.assertEqual(first, second)

    def test_legacy_recipe_signature_retains_universe_membership(self):
        first = full_train_batched._legacy_recipe_signature(
            factors=["ret_20d"],
            horizon=3,
            universe_codes=["600001"],
        )
        second = full_train_batched._legacy_recipe_signature(
            factors=["ret_20d"],
            horizon=3,
            universe_codes=["600001", "001220"],
        )

        self.assertNotEqual(first, second)


class ModelExpansionExperimentTest(unittest.TestCase):
    def test_score_pred_treats_missing_optional_ic_as_zero(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "base_pred": [0.25, -0.10],
        })

        result = watchlist_grid._score_pred(frame, ic_weight=0.5)

        pd.testing.assert_series_equal(
            result["pred"],
            frame["base_pred"],
            check_names=False,
        )

    def test_topk_residual_rerank_never_promotes_outside_champion_pool(self):
        date = pd.Timestamp("2026-01-05")
        rows = []
        for index in range(20):
            rows.append({
                "code": f"{600000 + index:06d}",
                "date": date,
                "base_pred": float(20 - index),
                "ridge_pred": float(20 - index),
                "residual_pred": 1000.0 if index == 19 else float(index),
            })
        frame = pd.DataFrame(rows)
        params = {
            "top_n": 2,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }

        scored = experiment._topk_residual_rerank_scores(
            frame,
            "residual_pred",
            params,
            pool_size=10,
            residual_weight=1.0,
        )

        pool = set(scored.loc[scored["in_champion_pool"], "code"])
        selected = set(scored.nlargest(2, "pred")["code"])
        self.assertEqual(len(pool), 10)
        self.assertTrue(selected.issubset(pool))
        self.assertNotIn("600019", selected)
        self.assertTrue(pd.isna(scored.loc[scored["code"] == "600019", "pred"]).all())

    def test_topk_residual_rerank_zero_weight_preserves_champion_order(self):
        frame = pd.DataFrame({
            "code": [f"{600000 + index:06d}" for index in range(12)],
            "date": pd.Timestamp("2026-01-05"),
            "base_pred": np.arange(12, dtype=float),
            "ridge_pred": np.arange(12, dtype=float),
            "residual_pred": np.arange(12, dtype=float)[::-1],
        })
        params = {
            "top_n": 2,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }

        scored = experiment._topk_residual_rerank_scores(
            frame,
            "residual_pred",
            params,
            pool_size=10,
            residual_weight=0.0,
        )

        self.assertEqual(scored.nlargest(2, "pred")["code"].tolist(), ["600011", "600010"])
        self.assertTrue(scored["residual_rerank_weight"].eq(0.0).all())

    def test_topk_residual_state_features_ignore_realized_targets(self):
        rows = []
        for date_index, date in enumerate(pd.to_datetime(["2026-01-05", "2026-01-06"])):
            for code_index in range(12):
                rows.append({
                    "code": f"{600000 + code_index:06d}",
                    "date": date,
                    "base_pred": float(code_index),
                    "ridge_pred": float(code_index),
                    "residual_pred": float(code_index + date_index),
                    "target_ret_3d": float(code_index - 6) / 100.0,
                })
        frame = pd.DataFrame(rows)
        params = {
            "top_n": 2,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }

        first = experiment.topk_residual_state_features(frame, "residual_pred", params)
        changed = frame.copy()
        changed["target_ret_3d"] = changed["target_ret_3d"].sample(
            frac=1.0, random_state=19
        ).to_numpy()
        second = experiment.topk_residual_state_features(changed, "residual_pred", params)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["topk_residual_coverage"].eq(1.0).all())

    def test_topk_confidence_gate_freezes_selection_threshold_and_daily_weights(self):
        dates = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-08-05", "2025-08-06"])
        rows = []
        for date_index, date in enumerate(dates):
            for code_index in range(12):
                rows.append({
                    "code": f"{600000 + code_index:06d}",
                    "date": date,
                    "base_pred": float(code_index + 1),
                    "ridge_pred": float(code_index + 1),
                    "residual_pred": float((code_index + date_index) % 12),
                    "target_ret_3d": float(code_index - 5) / 100.0,
                })
        frame = pd.DataFrame(rows)
        states = pd.DataFrame({
            "date": dates,
            "topk_residual_coverage": [0.2, 0.8, 0.4, 0.9],
            "topk_residual_dispersion": [1.0, 1.0, 100.0, 100.0],
            "topk_residual_top_gap_z": [0.5, 0.5, 50.0, 50.0],
        })
        params = {
            "top_n": 2,
            "gross_exposure": 0.3,
            "slot_weight": 0.15,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }
        evaluated_weights = []

        def fake_evaluation(*args, **_kwargs):
            weight = args[4]
            evaluated_weights.append(weight.copy() if isinstance(weight, pd.Series) else weight)
            sharpe = 2.0 if isinstance(weight, pd.Series) else 1.0
            metrics = pd.DataFrame({
                "horizon": [3],
                "sharpe": [sharpe],
                "max_drawdown": [-0.10],
            })
            return metrics, {}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "shadow"
            frame.to_parquet(source, index=False)
            with mock.patch.object(
                experiment, "topk_residual_state_features", return_value=states
            ), mock.patch.object(
                experiment, "_topk_rerank_evaluation", side_effect=fake_evaluation
            ), mock.patch.object(
                watchlist_grid, "promotion_decision", return_value={"promote": False}
            ), mock.patch.object(
                watchlist_grid, "stability_decision", return_value={"passed": False}
            ):
                report = experiment.evaluate_topk_residual_confidence_gate_shadow(
                    source,
                    "residual_pred",
                    params,
                    set(frame["code"]),
                    output,
                    state_quantiles=(0.5,),
                    horizons=(3,),
                )
            holdout = pd.read_parquet(output / "holdout_predictions.parquet")

            self.assertFalse(report["publishable"])
            self.assertEqual(report["selected_recipe"]["feature"], "topk_residual_coverage")
            self.assertAlmostEqual(report["selected_recipe"]["threshold"], 0.5)
            self.assertEqual(report["requested_horizons"], [3])
            self.assertEqual(report["evaluated_horizons"], [3])
            daily = holdout.groupby("date")["residual_rerank_weight"].first()
            self.assertEqual(float(daily.loc[pd.Timestamp("2025-08-05")]), 0.0)
            self.assertEqual(float(daily.loc[pd.Timestamp("2025-08-06")]), 0.05)
            selected = {
                date: set(group.nlargest(2, "pred")["code"])
                for date, group in holdout.dropna(subset=["pred"]).groupby("date")
            }
            pools = holdout[holdout["in_champion_pool"]].groupby("date")["code"].agg(set)
            self.assertTrue(all(codes.issubset(pools.loc[date]) for date, codes in selected.items()))
            self.assertFalse((output / "active_quant_model.json").exists())

        selection_weight = next(
            value for value in evaluated_weights
            if isinstance(value, pd.Series) and value.index.max() < pd.Timestamp("2025-08-01")
        )
        self.assertEqual(selection_weight.to_dict(), {
            pd.Timestamp("2025-01-06"): 0.0,
            pd.Timestamp("2025-01-07"): 0.05,
        })

    def test_topk_shadow_report_is_never_publishable(self):
        dates = pd.bdate_range("2024-01-02", periods=320)
        rows = []
        for date_index, date in enumerate(dates):
            for code_index in range(12):
                score = float(code_index)
                target = (score - 5.5) / 100.0
                rows.append({
                    "code": f"{600000 + code_index:06d}",
                    "date": date,
                    "base_pred": score,
                    "ridge_pred": score,
                    "residual_pred": score + (date_index % 3) * 0.01,
                    "target_ret_1d": target,
                    "target_ret_2d": target,
                    "target_ret_3d": target,
                })
        frame = pd.DataFrame(rows)
        params = {
            "top_n": 2,
            "gross_exposure": 0.3,
            "slot_weight": 0.15,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "shadow"
            frame.to_parquet(source, index=False)
            report = experiment.evaluate_topk_residual_rerank_shadow(
                source,
                "residual_pred",
                params,
                set(frame["code"]),
                output,
                weights=(0.0, 0.05),
                holdout_months=6,
            )

            self.assertFalse(report["publishable"])
            self.assertTrue((output / "shadow_evaluation.json").exists())
            self.assertTrue((output / "recipe.json").exists())
            self.assertFalse((output / "active_quant_model.json").exists())

    def test_neutralize_target_shrinks_small_industry_effects(self):
        rng = np.random.default_rng(23)
        rows = []
        industries = []
        for index in range(80):
            industry = "small" if index < 4 else "large"
            industries.append(industry)
            rows.append({
                "date": pd.Timestamp("2026-01-05"),
                "target": (1.0 if industry == "small" else 0.0) + rng.normal(scale=0.05),
            })
        frame = pd.DataFrame(rows)
        industry = pd.Series(industries, index=frame.index)

        unshrunk = experiment.neutralize_target_cross_section(
            frame,
            "target",
            industry=industry,
            exposure_columns=(),
            industry_shrinkage=0.0,
        )
        shrunk = experiment.neutralize_target_cross_section(
            frame,
            "target",
            industry=industry,
            exposure_columns=(),
            industry_shrinkage=20.0,
        )

        self.assertLess(abs(float(unshrunk[industry == "small"].mean())), 1e-10)
        self.assertGreater(abs(float(shrunk[industry == "small"].mean())), 0.1)

    def test_residual_state_features_ignore_realized_targets(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05"] * 4 + ["2026-01-06"] * 4),
            "residual_pred": [0.1, 0.2, 0.3, 1.0, -0.5, 0.0, 0.5, 2.0],
            "target_ret_3d": [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2],
        })

        first = experiment.residual_state_features(frame, "residual_pred")
        changed = frame.copy()
        changed["target_ret_3d"] = changed["target_ret_3d"].sample(
            frac=1.0, random_state=11
        ).to_numpy()
        second = experiment.residual_state_features(changed, "residual_pred")

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first["residual_coverage"] == 1.0).all())
        self.assertGreater(float(first.iloc[1]["residual_top_gap_z"]), 0.0)
        self.assertTrue(first["residual_dispersion_ratio"].isna().all())

    def test_lightgbm_ranker_forwards_top2_configuration(self):
        rows = []
        for date in pd.date_range("2026-01-01", periods=6):
            for index in range(20):
                rows.append({
                    "code": f"{600000 + index:06d}",
                    "date": date,
                    "factor": float(index),
                    "target_ret_3d": float(index) / 100.0,
                })
        panel = pd.DataFrame(rows)

        class FakeRanker:
            best_iteration_ = 4
            fit_kwargs = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, _x, _y, **kwargs):
                FakeRanker.fit_kwargs = kwargs
                return self

            def predict(self, values):
                return np.arange(len(values), dtype=float)

        fake_lightgbm = mock.Mock()
        fake_lightgbm.LGBMRanker = FakeRanker
        fake_lightgbm.early_stopping.return_value = object()
        with mock.patch.dict("sys.modules", {"lightgbm": fake_lightgbm}):
            result = experiment.model.train_lightgbm_ranker(
                panel,
                ["factor"],
                horizon=3,
                train_end="2026-01-03",
                valid_end="2026-01-04",
                predict_start="2026-01-05",
                rank_bins=10,
                eval_at=(2,),
            )

        self.assertTrue(result.ok)
        self.assertEqual(FakeRanker.fit_kwargs["eval_at"], [2])
        self.assertEqual(result.metrics["rank_bins"], 10)
        self.assertEqual(result.metrics["eval_at"], [2])

    def test_neutralize_target_removes_industry_and_size_exposure(self):
        rng = np.random.default_rng(42)
        rows = []
        for date in pd.date_range("2026-01-01", periods=3):
            for index in range(80):
                industry = "A" if index < 40 else "B"
                size = rng.normal()
                target = (0.2 if industry == "A" else -0.2) + 0.3 * size + rng.normal(scale=0.01)
                rows.append({
                    "date": date,
                    "target": target,
                    "log_mv_total": size,
                    "industry": industry,
                })
        frame = pd.DataFrame(rows)

        residual = experiment.neutralize_target_cross_section(
            frame,
            "target",
            industry=frame["industry"],
            exposure_columns=("log_mv_total",),
        )

        self.assertLess(residual.groupby(frame["date"]).mean().abs().max(), 1e-10)
        self.assertLess(abs(residual.corr(frame["log_mv_total"])), 1e-10)
        industry_means = residual.groupby([frame["date"], frame["industry"]]).mean()
        self.assertLess(industry_means.abs().max(), 1e-10)

    def test_pit_industry_mapping_obeys_valid_and_available_dates(self):
        frame = pd.DataFrame({
            "code": ["600001", "600001", "600001"],
            "date": pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"]),
        })
        history = pd.DataFrame({
            "code": ["600001", "600001"],
            "industry": ["Old", "New"],
            "valid_from": pd.to_datetime(["2024-01-01", "2025-02-01"]),
            "valid_to": pd.to_datetime(["2025-02-01", "2099-01-01"]),
            "available_from": pd.to_datetime(["2024-01-02", "2025-03-01"]),
        })

        mapped = experiment._pit_industry_for_frame(frame, history)

        self.assertEqual(mapped.iloc[0], "Old")
        self.assertTrue(pd.isna(mapped.iloc[1]))
        self.assertEqual(mapped.iloc[2], "New")

    def test_pit_industry_mapping_keeps_open_interval_after_publication(self):
        frame = pd.DataFrame({
            "code": ["600001", "600001"],
            "date": pd.to_datetime(["2025-02-05", "2025-03-05"]),
        })
        history = pd.DataFrame({
            "code": ["600001"],
            "industry": ["New"],
            "valid_from": pd.to_datetime(["2025-02-01"]),
            "valid_to": pd.to_datetime([None]),
            "available_from": pd.to_datetime(["2025-03-01"]),
        })

        mapped = experiment._pit_industry_for_frame(frame, history)

        self.assertTrue(pd.isna(mapped.iloc[0]))
        self.assertEqual(mapped.iloc[1], "New")

    def test_pit_industry_history_is_publishable_only_with_required_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "industry_history.parquet"
            pd.DataFrame({
                "code": ["600001"],
                "industry": ["Bank"],
                "valid_from": ["2024-01-01"],
                "valid_to": [None],
                "available_from": ["2024-01-02"],
            }).to_parquet(path, index=False)
            mode, publishable, mapping, history, updated = experiment._load_industry_metadata(
                None, path
            )

        self.assertEqual(mode, "strict_pit_industry")
        self.assertTrue(publishable)
        self.assertIsNone(mapping)
        self.assertEqual(len(history), 1)
        self.assertEqual(updated, "2024-01-02T00:00:00")

    def test_future_available_industry_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "industry_history.parquet"
            pd.DataFrame({
                "code": ["600001"],
                "industry": ["Bank"],
                "valid_from": ["2024-02-01"],
                "available_from": ["2024-01-01"],
            }).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "available_from predates valid_from"):
                experiment._load_industry_metadata(None, path)

    def test_retroactively_published_industry_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "industry_history.parquet"
            pd.DataFrame({
                "code": ["600001"],
                "industry": ["Bank"],
                "valid_from": ["2024-01-01"],
                "available_from": ["2024-01-02"],
                "source_updated_at": ["2025-01-01"],
            }).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "published after available_from"):
                experiment._load_industry_metadata(None, path)

    def test_confidence_uses_model_agreement_without_realized_targets(self):
        frame = pd.DataFrame({
            "code": ["600001", "600002", "600003", "600004"],
            "date": pd.to_datetime(["2026-01-05"] * 4),
            "ridge_pred": [1.0, 2.0, 3.0, 4.0],
            "lgbm_pred": [1.0, 2.0, 4.0, 3.0],
            "elastic_pred": [1.0, 2.0, 3.0, 4.0],
            "base_pred": [1.0, 2.0, 3.5, 3.5],
            "ic_z": [0.0] * 4,
            "target_ret_3d": [0.9, -0.8, 0.7, -0.6],
        })
        params = {
            "lgbm_weight": 0.5,
            "elastic_weight": 0.2,
            "ic_weight": 0.0,
        }

        first = experiment.confidence_from_model_agreement(frame, params)
        shuffled = frame.copy()
        shuffled["target_ret_3d"] = shuffled["target_ret_3d"].sample(
            frac=1.0, random_state=7
        ).to_numpy()
        second = experiment.confidence_from_model_agreement(shuffled, params)

        pd.testing.assert_series_equal(first, second)
        self.assertGreater(float(first.iloc[0]), float(first.iloc[2]))

    def test_orthogonal_increment_removes_daily_champion_rank_exposure(self):
        rng = np.random.default_rng(17)
        rows = []
        for date in pd.to_datetime(["2026-01-05", "2026-01-06"]):
            for index in range(60):
                champion = float(index)
                rows.append({
                    "date": date,
                    "champion_score": champion,
                    "candidate": champion + rng.normal(scale=15.0),
                })
        frame = pd.DataFrame(rows)

        residual = experiment.orthogonalize_increment_cross_section(
            frame,
            "candidate",
            controls=("champion_score",),
        )

        self.assertEqual(int(residual.notna().sum()), len(frame))
        for _, index in frame.groupby("date").groups.items():
            champion_rank = frame.loc[index, "champion_score"].rank(pct=True)
            self.assertLess(abs(float(residual.loc[index].corr(champion_rank))), 1e-10)
            self.assertAlmostEqual(float(residual.loc[index].std(ddof=0)), 1.0)

    def test_optional_leg_freezes_selection_weight_and_reports_all_horizons(self):
        dates = pd.bdate_range("2025-01-02", periods=180)
        rows = []
        for date_index, date in enumerate(dates):
            for code_index in range(20):
                rows.append({
                    "code": f"{code_index + 1:06d}",
                    "date": date,
                    "base_pred": float(code_index),
                    "ridge_pred": float(code_index),
                    "orthogonal_increment_pred": float((code_index + date_index) % 20),
                    "target_ret_1d": float(code_index - 10) / 1000.0,
                    "target_ret_2d": float(code_index - 10) / 900.0,
                    "target_ret_3d": float(code_index - 10) / 800.0,
                })
        frame = pd.DataFrame(rows)
        params = {
            "top_n": 2,
            "gross_exposure": 0.3,
            "slot_weight": 0.15,
            "pred_quantile": None,
            "ridge_quantile": None,
        }
        selection_calls = []
        holdout_calls = []

        def fake_metrics(_prepared, called_params, horizons, *_args):
            weight = float(called_params["catboost_weight"])
            selection_calls.append(weight)
            sharpe = 2.0 if weight == 0.05 else 1.0
            return pd.DataFrame({
                "horizon": list(horizons),
                "sharpe": [sharpe] * len(horizons),
                "max_drawdown": [-0.10] * len(horizons),
            })

        def fake_returns(_frame, called_params, horizons, *_args):
            weight = float(called_params["catboost_weight"])
            holdout_calls.append(weight)
            return {
                int(horizon): pd.DataFrame({
                    "date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
                    "ret": [0.01, 0.02],
                })
                for horizon in horizons
            }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "shadow.json"
            frame.to_parquet(source, index=False)
            with mock.patch.object(
                watchlist_grid, "evaluate_prepared_params", side_effect=fake_metrics
            ), mock.patch.object(
                watchlist_grid, "evaluate_prepared_returns", side_effect=fake_returns
            ), mock.patch.object(
                watchlist_grid, "promotion_decision", return_value={"promote": False}
            ), mock.patch.object(
                watchlist_grid, "stability_decision", return_value={"passed": False}
            ):
                report = shadow_leg_evaluation.evaluate_optional_leg(
                    source,
                    "orthogonal_increment_pred",
                    params,
                    set(frame["code"]),
                    output,
                    weights=(0.0, 0.03, 0.05, 0.10),
                    horizons=(1, 2, 3),
                )

        self.assertEqual(report["selected_weight"], 0.05)
        self.assertEqual(selection_calls[:4], [0.0, 0.03, 0.05, 0.10])
        self.assertEqual(selection_calls[4:], [0.0, 0.05])
        self.assertEqual(holdout_calls, [0.0, 0.05])
        self.assertEqual(report["requested_horizons"], [1, 2, 3])
        self.assertEqual(report["evaluated_horizons"], [1, 2, 3])
        self.assertEqual(
            [row["horizon"] for row in report["independent_signal"]],
            [1, 2, 3],
        )

    def test_orthogonal_shadow_is_never_publishable(self):
        dates = pd.bdate_range("2025-01-02", periods=130)
        rows = []
        rng = np.random.default_rng(31)
        for date in dates:
            for code_index in range(30):
                score = float(code_index)
                rows.append({
                    "code": f"{code_index + 1:06d}",
                    "date": date,
                    "base_pred": score,
                    "ridge_pred": score,
                    "residual_pred": score + rng.normal(scale=8.0),
                    "target_ret_1d": score / 1000.0,
                    "target_ret_2d": score / 900.0,
                    "target_ret_3d": score / 800.0,
                })
        frame = pd.DataFrame(rows)
        params = {
            "top_n": 2,
            "gross_exposure": 0.3,
            "slot_weight": 0.15,
            "pred_quantile": None,
            "ridge_quantile": None,
            "positive_only": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            output = Path(directory) / "orthogonal"
            frame.to_parquet(source, index=False)
            report = experiment.build_orthogonal_increment_shadow(
                source,
                "residual_pred",
                params,
                set(frame["code"]),
                output,
                weights=(0.0, 0.05),
                holdout_months=2,
            )

            self.assertFalse(report["publishable"])
            self.assertTrue((output / "predictions.parquet").exists())
            self.assertTrue((output / "shadow_evaluation.json").exists())
            self.assertTrue((output / "recipe.json").exists())
            self.assertFalse((output / "active_quant_model.json").exists())

    def test_industry_research_recipe_is_explicitly_not_publishable(self):
        source = Path("source.parquet")
        audit = Path("audit.parquet")
        prepared = Path("prepared")
        industry = Path("industry.parquet")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(
                    experiment,
                    "_load_industry_metadata",
                    return_value=(
                        "industry_research",
                        False,
                        pd.Series({"600001": "Bank"}),
                        None,
                        "2026-07-16T00:00:00",
                    ),
                ), \
                mock.patch.object(pd, "read_parquet", side_effect=RuntimeError("stop after recipe")):
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                experiment.build_residual_shadow(
                    source, audit, prepared, output, industry_meta=industry
                )
            recipe = json.loads((output / "recipe.json").read_text(encoding="utf-8"))

        self.assertEqual(recipe["mode"], "industry_research")
        self.assertFalse(bool(recipe["publishable"]))
        self.assertEqual(recipe["industry_meta"], str(industry))

    def test_daily_market_from_prices_uses_same_day_and_trailing_closes(self):
        dates = pd.date_range("2026-01-01", periods=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for code, closes in (("600001", [10, 11, 10, 12]), ("600002", [20, 18, 19, 19])):
                pd.DataFrame({"date": dates, "close": closes}).to_parquet(root / f"{code}.parquet")
            market = experiment._daily_market_from_prices(
                root, {"600001", "600002"}, dates.min(), dates.max() + pd.Timedelta(days=1)
            )

        self.assertEqual(len(market), 3)
        self.assertEqual(market["n_stocks"].tolist(), [2, 2, 2])
        self.assertAlmostEqual(float(market.iloc[0]["breadth"]), 0.5)

    def test_market_regime_requires_trailing_history(self):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=300)
        market = pd.DataFrame({
            "date": dates,
            "median_return": rng.normal(0, 0.01, len(dates)),
            "breadth": rng.uniform(0, 1, len(dates)),
        })

        result = experiment.build_market_regimes(
            market,
            trend_window=20,
            volatility_window=20,
            history_window=252,
        )

        self.assertTrue(result.iloc[:62]["regime"].eq("insufficient_history").all())
        self.assertTrue(result.iloc[-1]["regime"] != "insufficient_history")

    def test_regime_weight_selection_scores_each_horizon_separately(self):
        dates = pd.date_range("2026-01-01", periods=25)
        rows = []
        for horizon in (1, 2, 3):
            for weight in (0.7, 0.85):
                for index, date in enumerate(dates):
                    ret = (0.012 if index % 2 else 0.008) if weight == 0.7 else (0.005 if index % 2 else -0.004)
                    rows.append({
                        "date": date, "regime": "up", "horizon": horizon,
                        "weight": weight, "ret": ret,
                    })
        selected = experiment.select_regime_weights(pd.DataFrame(rows), [0.7, 0.85], min_observations=20)

        self.assertEqual(selected, {"up": 0.7})

    def test_regime_weight_selection_uses_supplied_returns(self):
        dates = pd.date_range("2026-01-01", periods=40)
        returns = pd.DataFrame({
            "date": list(dates) * 2,
            "regime": ["up"] * 80,
            "weight": [0.0] * 40 + [0.1] * 40,
            "ret": [0.0] * 40 + [0.01, 0.02] * 20,
        })

        selected = experiment.select_regime_weights(returns, [0.0, 0.1], min_observations=20)

        self.assertEqual(selected, {"up": 0.1})


if __name__ == "__main__":
    unittest.main()
