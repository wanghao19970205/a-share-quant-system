from __future__ import annotations

import inspect
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from quant import backtest
from quant import check_daily_update
from quant import daily_update
from quant import full_train_batched
from quant import model as quant_model
from quant import pipeline as quant_pipeline
from quant import model_expansion_experiment as experiment
from quant import scheduled_workflow
from quant import select as factor_select
from quant import tradability
from quant import shadow_leg_evaluation
from quant import warehouse
from quant import watchlist_grid
from quant.factors import engineering
from stock_analyzer import amazingdata_source, sentiment_signal


class SentimentCalendarTest(unittest.TestCase):
    def test_weekend_article_moves_to_next_session_without_calendar_day_decay(self):
        articles = pd.DataFrame({
            "publish_dt": pd.to_datetime(["2026-01-09 10:00", "2026-01-11 10:00"]),
            "category": ["news", "news"],
            "sentiment": [1.0, -1.0],
            "llm_score": [np.nan, np.nan],
        })
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"]),
            "ret_1d": [0.0, 0.0, np.nan],
            "ret_3d": [0.0, np.nan, np.nan],
        })
        with tempfile.TemporaryDirectory() as temporary:
            pd.DataFrame({
                "date": pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"])
                .astype("datetime64[ns]")
            }).to_parquet(Path(temporary) / "trading_calendar.parquet", index=False)
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": temporary}), \
                    mock.patch.object(sentiment_signal, "_articles", return_value=articles), \
                    mock.patch.object(sentiment_signal, "_price_forward_returns", return_value=prices):
                result = sentiment_signal._candidate_daily(
                    "600001", 1.0, {"news": 1.0},
                    pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-13"),
                )

        monday = result.loc[result["date"] == pd.Timestamp("2026-01-12"), "sentiment_score"]
        self.assertEqual(len(monday), 1)
        self.assertAlmostEqual(float(monday.iloc[0]), 0.0)

    def test_strict_runtime_cutoff_excludes_close_and_date_only_articles(self):
        stored = pd.DataFrame({
            "publish_time": [
                "2026-01-09 14:59:00",
                "2026-01-09 15:00:00",
                "2026-01-09 15:01:00",
                "2026-01-09",
            ],
            "sentiment": [1.0, -1.0, -1.0, -1.0],
            "category": ["news"] * 4,
        })
        base_model = {
            "name": "strict-cutoff-test",
            "half_life_days": 7.0,
            "lookback_days": 30,
            "category_weights": {"news": 1.0},
            "signal_source": "lexicon",
            "enabled": False,
        }
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-09", "2026-01-12",
        ]))
        with mock.patch.object(sentiment_signal.news_store, "read_store", return_value=stored), \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar):
            legacy = sentiment_signal.score_at(
                "600001", pd.Timestamp("2026-01-09"), base_model,
            )
            strict = sentiment_signal.score_at(
                "600001", pd.Timestamp("2026-01-09"),
                {**base_model, "strict_announcement_lag": True},
            )
        self.assertEqual(legacy.article_count, 3)
        self.assertEqual(strict.article_count, 1)
        self.assertEqual(
            sentiment_signal._strict_publish_timestamp("2026-01-09"),
            pd.Timestamp("2026-01-09 15:00:00"),
        )
        self.assertEqual(
            sentiment_signal._strict_publish_timestamp(
                "2026-01-09T07:00:00+00:00"
            ),
            pd.Timestamp("2026-01-09 15:00:00"),
        )

    def test_strict_candidate_lags_close_articles_to_next_session(self):
        stored = pd.DataFrame({
            "publish_time": [
                "2026-01-09 14:59:00",
                "2026-01-09 15:00:00",
                "2026-01-09 15:01:00",
                "2026-01-09",
                "2026-01-10 10:00:00",
            ],
            "sentiment": [1.0, -1.0, -1.0, -1.0, -1.0],
            "category": ["news"] * 5,
        })
        articles = stored.copy()
        articles["publish_dt"] = pd.to_datetime(
            articles["publish_time"], format="mixed",
        )
        articles["strict_publish_dt"] = articles["publish_time"].map(
            sentiment_signal._strict_publish_timestamp
        )
        articles["llm_score"] = np.nan
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"]),
            "ret_1d": [0.01, 0.02, np.nan],
            "ret_3d": [0.03, np.nan, np.nan],
        })
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-09", "2026-01-12", "2026-01-13",
        ]))
        with mock.patch.object(sentiment_signal, "_articles", return_value=articles), \
                mock.patch.object(sentiment_signal, "_price_forward_returns", return_value=prices), \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar):
            legacy = sentiment_signal._candidate_daily(
                "600001", 1.0, {"news": 1.0},
                pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-13"),
            )
            strict = sentiment_signal._candidate_daily(
                "600001", 1.0, {"news": 1.0},
                pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-13"),
                strict_announcement_lag=True,
            )
        legacy_friday = legacy.loc[legacy["date"] == pd.Timestamp("2026-01-09"), "sentiment_score"].iloc[0]
        strict_friday = strict.loc[strict["date"] == pd.Timestamp("2026-01-09"), "sentiment_score"].iloc[0]
        strict_monday = strict.loc[strict["date"] == pd.Timestamp("2026-01-12"), "sentiment_score"].iloc[0]
        self.assertGreater(strict_friday, legacy_friday)
        self.assertGreater(strict_friday, 0.0)
        self.assertLess(strict_monday, strict_friday)

    def test_strict_runtime_uses_exchange_session_decay(self):
        stored = pd.DataFrame({
            "publish_time": [
                "2026-01-09 14:00:00",
                "2026-01-12 14:00:00",
            ],
            "sentiment": [1.0, -1.0],
            "category": ["news", "news"],
        })
        model = {
            "name": "strict-session-decay",
            "half_life_days": 1.0,
            "lookback_days": 30,
            "category_weights": {"news": 1.0},
            "signal_source": "lexicon",
            "strict_announcement_lag": True,
            "enabled": False,
        }
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-09", "2026-01-12", "2026-01-13",
        ]))
        with mock.patch.object(sentiment_signal.news_store, "read_store", return_value=stored), \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar):
            signal = sentiment_signal.score_at(
                "600001", pd.Timestamp("2026-01-12"), model,
            )
            invalid = sentiment_signal.score_at("600001", "not-a-date", model)
        self.assertEqual(signal.article_count, 2)
        self.assertAlmostEqual(signal.raw_score, -1.0 / 3.0, places=3)
        self.assertEqual(invalid.note, "无效 asof")
        self.assertFalse(invalid.available)

    def test_strict_runtime_matches_candidate_session_aggregation(self):
        stored = pd.DataFrame({
            "publish_time": [
                "2026-01-09 14:00:00",
                "2026-01-12 14:00:00",
            ],
            "sentiment": [1.0, -1.0],
            "category": ["news", "news"],
        })
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-09", "2026-01-12", "2026-01-13",
        ]))
        prices = pd.DataFrame({
            "date": calendar,
            "ret_1d": [0.01, 0.02, np.nan],
            "ret_3d": [0.03, np.nan, np.nan],
        })
        model = {
            "name": "strict-session-parity",
            "half_life_days": 1.0,
            "lookback_days": 30,
            "category_weights": {"news": 1.0},
            "signal_source": "lexicon",
            "strict_announcement_lag": True,
            "enabled": False,
        }
        with mock.patch.object(sentiment_signal.news_store, "read_store", return_value=stored), \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar):
            articles = sentiment_signal._articles("600001")
            runtime = sentiment_signal.score_at(
                "600001", pd.Timestamp("2026-01-12"), model,
            )
        with mock.patch.object(sentiment_signal, "_articles", return_value=articles), \
                mock.patch.object(sentiment_signal, "_price_forward_returns", return_value=prices), \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar):
            candidate = sentiment_signal._candidate_daily(
                "600001", 1.0, {"news": 1.0},
                pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-13"),
                strict_announcement_lag=True,
            )
        monday = candidate.loc[
            candidate["date"] == pd.Timestamp("2026-01-12"),
            "sentiment_score",
        ].iloc[0]
        self.assertAlmostEqual(runtime.score, monday, places=3)

    def test_strict_sentiment_purges_three_sessions_before_holdout(self):
        calendar = pd.bdate_range("2026-01-01", periods=20)
        frame = pd.DataFrame({
            "date": calendar,
            "sentiment_score": np.linspace(-1.0, 1.0, len(calendar)),
            "ret_1d": np.linspace(0.01, 0.02, len(calendar)),
            "ret_3d": np.linspace(0.02, 0.03, len(calendar)),
            "code": ["600001"] * len(calendar),
        })
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(sentiment_signal, "_authoritative_calendar", return_value=calendar), \
                mock.patch.object(sentiment_signal, "_articles", return_value=pd.DataFrame()) as article_load, \
                mock.patch.object(sentiment_signal, "_price_forward_returns", return_value=pd.DataFrame()) as price_load, \
                mock.patch.object(sentiment_signal, "_candidate_daily", return_value=frame):
            model = sentiment_signal.train(
                ["600001"], calendar[0], calendar[-1],
                str(Path(temporary) / "strict_sentiment.json"),
                strict_announcement_lag=True,
                strict_label_purge=True,
            )
        raw_split = calendar[0] + (calendar[-1] - calendar[0]) * 0.75
        split_position = int(calendar.searchsorted(raw_split, side="left"))
        self.assertEqual(model["holdout_start"], str(calendar[split_position].date()))
        self.assertEqual(model["fit_end_exclusive"], str(calendar[split_position - 3].date()))
        self.assertEqual(model["purge_sessions"], 3)
        self.assertTrue(model["strict_label_purge"])
        article_load.assert_called_once_with("600001")
        price_load.assert_called_once_with("600001")

    def test_candidate_daily_requires_authoritative_calendar(self):
        articles = pd.DataFrame({
            "publish_dt": pd.to_datetime(["2026-01-09 10:00"]),
            "category": ["news"], "sentiment": [1.0], "llm_score": [np.nan],
        })
        prices = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-09"]), "ret_1d": [0.0], "ret_3d": [0.0],
        })
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.dict(os.environ, {"QUANT_DATA_DIR": temporary}), \
                mock.patch.object(sentiment_signal, "_articles", return_value=articles), \
                mock.patch.object(sentiment_signal, "_price_forward_returns", return_value=prices), \
                self.assertRaisesRegex(RuntimeError, "authoritative trading calendar unavailable"):
            sentiment_signal._candidate_daily(
                "600001", 1.0, {"news": 1.0},
                pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-09"),
            )


class IntradaySourcePriorityTest(unittest.TestCase):
    def test_trading_calendar_refresh_saves_authoritative_dates(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]).astype("datetime64[ns]")
        })
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_trading_calendar", return_value=calendar), \
                mock.patch.object(daily_update.warehouse, "save") as save:
            result = daily_update.refresh_trading_calendar()

        save.assert_called_once()
        self.assertEqual(save.call_args.args[0], "trading_calendar")
        pd.testing.assert_frame_equal(save.call_args.args[1], calendar)
        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(result["rows"], 2)

    def test_trading_calendar_refresh_rejects_non_increasing_dates(self):
        calendar = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-02"])})
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_trading_calendar", return_value=calendar), \
                mock.patch.object(daily_update.warehouse, "save") as save, \
                self.assertRaisesRegex(ValueError, "unique and increasing"):
            daily_update.refresh_trading_calendar()
        save.assert_not_called()

    def test_pit_reference_refresh_saves_normalized_sources_without_activation(self):
        security_master = pd.DataFrame({
            "code": ["600000"],
            "list_date": pd.to_datetime(["1999-11-10"]).astype("datetime64[ns]"),
        })
        index_history = pd.DataFrame({
            "index_code": ["000300.SH"],
            "code": ["600000"],
            "in_date": pd.to_datetime(["2005-04-08"]).astype("datetime64[ns]"),
        })
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_security_master", return_value=security_master), \
                mock.patch.object(daily_update.datafeed, "broker_index_constituent_history", return_value=index_history) as history, \
                mock.patch.object(daily_update.warehouse, "save") as save:
            result = daily_update.refresh_pit_reference_data(["000300.SH"])

        history.assert_called_once_with(["000300.SH"])
        self.assertEqual([call.args[0] for call in save.call_args_list], [
            "security_master", "index_constituent_history",
        ])
        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(result["security_master_rows"], 1)
        self.assertEqual(result["index_history_rows"], 1)

    def test_trading_status_refresh_is_explicit_and_not_activated(self):
        history = pd.DataFrame({
            "code": ["600000", "600000"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]).astype("datetime64[ns]"),
            "is_st": [False, True],
            "is_suspended": [False, True],
            "is_withdrawal": [False, False],
        })
        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), \
                mock.patch.object(daily_update.datafeed, "broker_history_stock_status", return_value=history) as fetch, \
                mock.patch.object(daily_update.warehouse, "save") as save:
            result = daily_update.refresh_trading_status_reference(["600000"])

        fetch.assert_called_once_with(["600000"])
        save.assert_called_once_with("trading_status_history", history)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["st_rows"], 1)
        self.assertEqual(result["suspended_rows"], 1)
        self.assertEqual(result["withdrawal_rows"], 0)
        self.assertEqual(result["batches"], 1)
        self.assertEqual(result["batch_size"], 0)

    def test_refresh_trading_status_reference_supports_explicit_batches(self):
        def status_batch(codes):
            return pd.DataFrame({
                "code": codes,
                "date": pd.to_datetime(["2026-07-21"] * len(codes)),
                "is_st": [False] * len(codes),
                "is_suspended": [False] * len(codes),
                "is_withdrawal": [False] * len(codes),
            })

        with mock.patch.object(daily_update.datafeed, "broker_available", return_value=True), mock.patch.object(
            daily_update.datafeed, "broker_history_stock_status", side_effect=status_batch,
        ) as fetch, mock.patch.object(daily_update.warehouse, "save") as save:
            result = daily_update.refresh_trading_status_reference(
                ["600003", "600001", "600002"], batch_size=2,
            )

        self.assertEqual(
            [call.args[0] for call in fetch.call_args_list],
            [["600001", "600002"], ["600003"]],
        )
        saved = save.call_args.args[1]
        self.assertEqual(saved["code"].tolist(), ["600001", "600002", "600003"])
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["batch_size"], 2)

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
                mock.patch.dict(os.environ, {"AMAZINGDATA_KLINE_BATCH_SIZE": "2"}), \
                mock.patch("builtins.print") as output:
            result = amazingdata_source.fetch_daily_batch(
                symbols, "20260720", "20260721",
                progress_offset=1,
                progress_total=4,
            )

        progress = [str(call.args[0]).split()[1] for call in output.call_args_list]
        self.assertEqual([len(chunk) for chunk in calls], [2, 2, 1])
        self.assertEqual(progress, ["2/4", "3/4", "4/4"])
        self.assertEqual(set(result), set(symbols))

    def test_latest_price_probe_uses_one_unadjusted_batch(self):
        frames = {
            "000001": pd.DataFrame({"date": pd.to_datetime(["2026-08-05"])}),
            "600000": pd.DataFrame({"date": pd.to_datetime(["2026-08-06"])}),
        }
        with mock.patch.object(
            daily_update.datafeed,
            "broker_daily_prices",
            return_value=frames,
        ) as batch:
            latest = daily_update._probe_latest_price_date(
                ["000001", "600000"],
                lookback_days=5,
            )

        self.assertEqual(latest, pd.Timestamp("2026-08-06").date())
        self.assertEqual(batch.call_count, 1)
        self.assertEqual(batch.call_args.kwargs["adjust"], "")

    def test_force_latest_skips_remote_date_probe(self):
        with mock.patch.object(daily_update.config, "ensure_dirs"), \
                mock.patch.object(daily_update, "_refresh_mainboard_universe_isolated"), \
                mock.patch.object(daily_update.datafeed, "universe", return_value=["600001"]), \
                mock.patch.object(daily_update, "_probe_latest_price_date") as probe, \
                mock.patch.object(daily_update, "_stale_codes", return_value=["600001"]), \
                mock.patch.object(daily_update, "_update_prices_batched", return_value={
                    "ok": 1, "fail": 0, "rows": 1, "failures": [],
                }):
            result = daily_update.run(
                force_latest=True,
                skip_valuation=True,
                skip_events=True,
                skip_snapshots=True,
            )

        probe.assert_not_called()
        self.assertEqual(result["n_codes"], 1)
        self.assertEqual(result["price"]["source_latest"], "")

    def test_price_batch_reuses_dates_from_stale_scan(self):
        today = pd.Timestamp.today().normalize()
        old = pd.DataFrame({
            "code": ["600001"],
            "date": [today],
            "open": [10.0], "high": [10.2], "low": [9.9], "close": [10.1],
        })
        fresh = old.drop(columns=["code"])
        with mock.patch.object(daily_update, "_warmup_broker", return_value=True), \
                mock.patch.object(
                    daily_update.datafeed,
                    "broker_daily_prices",
                    return_value={"600001": fresh},
                ), \
                mock.patch.object(
                    daily_update.warehouse,
                    "load_price",
                    return_value=old,
                ) as load_price, \
                mock.patch.object(daily_update.warehouse, "save_price"):
            result = daily_update._update_prices_batched(
                ["600001"],
                batch_size=800,
                local_dates={"600001": today.date()},
            )

        self.assertEqual(result["ok"], 1)
        self.assertEqual(load_price.call_count, 1)

    def test_price_save_replaces_file_atomically(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(warehouse.config, "PRICE_DIR", directory), \
                mock.patch.object(warehouse.config, "ensure_dirs"):
            path = Path(directory) / "600001.parquet"
            path.write_bytes(b"old")
            frame = pd.DataFrame({
                "code": ["600001"],
                "date": pd.to_datetime(["2026-07-21"]).astype("datetime64[ns]"),
                "close": [10.0],
            })

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

    def test_recovery_codes_round_trip_through_daily_update(self):
        path = scheduled_workflow._write_recovery_codes(["600001", "000001"])
        try:
            self.assertEqual(path.read_text(encoding="utf-8"), "600001\n000001\n")
            with mock.patch.object(daily_update.config, "ensure_dirs"), \
                    mock.patch.object(daily_update, "refresh_trading_calendar", return_value={}):
                result = daily_update.run(
                    codes_file=str(path),
                    skip_price=True,
                    skip_valuation=True,
                    skip_events=True,
                    skip_fundamentals=True,
                    skip_snapshots=True,
                )
            self.assertEqual(result["n_codes"], 2)
        finally:
            path.unlink(missing_ok=True)

    def test_recovery_codes_reject_empty_valid_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codes.txt"
            path.write_text("invalid\n", encoding="utf-8")
            with mock.patch.object(daily_update.config, "ensure_dirs"), \
                    mock.patch.object(daily_update, "refresh_trading_calendar", return_value={}), \
                    self.assertRaisesRegex(ValueError, "no valid six-digit codes"):
                daily_update.run(codes_file=str(path))

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

    def test_short_legacy_merge_reuses_identical_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = root / "short.parquet"
            legacy = root / "legacy.parquet"
            source = root / "source.parquet"
            short.write_bytes(b"same-history")
            legacy.write_bytes(b"same-history")
            fresh = pd.DataFrame()

            with mock.patch.object(
                scheduled_workflow, "merge_active_predictions",
            ) as merge, mock.patch.object(
                scheduled_workflow, "_atomic_copy",
            ) as copy:
                mode = scheduled_workflow._merge_short_and_legacy(
                    short, legacy, source, fresh)

        self.assertEqual(mode, "shared-history")
        merge.assert_called_once_with(
            short, source, short, fresh_frame=fresh)
        copy.assert_called_once_with(short, legacy)

    def test_short_legacy_merge_preserves_different_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = root / "short.parquet"
            legacy = root / "legacy.parquet"
            source = root / "source.parquet"
            short.write_bytes(b"short-history")
            legacy.write_bytes(b"legacy-history")
            fresh = pd.DataFrame()

            with mock.patch.object(
                scheduled_workflow, "merge_active_predictions",
            ) as merge, mock.patch.object(
                scheduled_workflow, "_atomic_copy",
            ) as copy:
                mode = scheduled_workflow._merge_short_and_legacy(
                    short, legacy, source, fresh)

        self.assertEqual(mode, "independent-history")
        self.assertEqual(merge.call_count, 2)
        copy.assert_not_called()

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
    def test_pit_price_row_gate_uses_nth_historical_observation(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(engineering.config, "PRICE_DIR", temporary):
            pd.DataFrame({
                "date": pd.to_datetime([
                    "2020-01-02", "2020-01-03", "2020-01-06", "2026-08-10",
                ]),
            }).to_parquet(Path(temporary) / "600001.parquet", index=False)
            pd.DataFrame({
                "date": pd.to_datetime(["2020-01-02", "2026-08-10"]),
            }).to_parquet(Path(temporary) / "600002.parquet", index=False)

            eligible = engineering._price_row_eligibility_dates(
                ["600001", "600002"], min_price_rows=3,
            )

        self.assertEqual(eligible, {"600001": pd.Timestamp("2020-01-06")})

    def test_strict_announcement_lag_maps_to_next_exchange_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calendar_path = root / "trading_calendar.parquet"
            pd.DataFrame({
                "date": pd.to_datetime([
                    "2026-01-30", "2026-02-02", "2026-02-03",
                ]).astype("datetime64[ns]"),
            }).to_parquet(calendar_path, index=False)
            report = pd.DataFrame({
                "code": ["600001", "600002"],
                "ann_date": ["2026-01-30", "2026-02-01"],
                "ROE": [10.0, 12.0],
            })
            with mock.patch.object(engineering.warehouse, "load", return_value=report), \
                    mock.patch.object(engineering.config, "TRADING_CALENDAR_FILE", str(calendar_path)):
                strict = engineering._asof_report_factor(
                    "income", "income", {"600001", "600002"},
                    strict_announcement_lag=True,
                )
                legacy = engineering._asof_report_factor(
                    "income", "income", {"600001", "600002"},
                )

        self.assertEqual(
            strict["date"].tolist(),
            [pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-02")],
        )
        self.assertEqual(strict["income_roe"].tolist(), [10.0, 12.0])
        self.assertEqual(
            legacy["date"].tolist(),
            [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-01")],
        )

    def test_report_factor_never_falls_back_to_reporting_period(self):
        report = pd.DataFrame({
            "code": ["600001"],
            "report_date": ["2025-12-31"],
            "ROE": [12.0],
        })
        with mock.patch.object(engineering.warehouse, "load", return_value=report):
            result = engineering._asof_report_factor("income", "income", {"600001"})
        self.assertTrue(result.empty)

    def test_report_factor_drops_rows_without_announcement_date(self):
        report = pd.DataFrame({
            "code": ["600001", "600001"],
            "ann_date": [None, "2026-04-30"],
            "report_date": ["2025-09-30", "2025-12-31"],
            "ROE": [10.0, 12.0],
        })
        with mock.patch.object(engineering.warehouse, "load", return_value=report):
            result = engineering._asof_report_factor("income", "income", {"600001"})
        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2026-04-30")])
        self.assertEqual(result["income_roe"].tolist(), [12.0])

    def test_forecast_never_falls_back_to_reporting_period(self):
        forecast = pd.DataFrame({
            "code": ["600001"],
            "report_date": ["2025-12-31"],
            "预告类型": ["预增"],
        })
        with mock.patch.object(engineering.warehouse, "load", return_value=forecast):
            result = engineering._forecast_events({"600001"})
        self.assertTrue(result.empty)

    def test_trading_gap_risk_uses_exchange_sessions_across_calendar_boundaries(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime([
                "2025-09-30", "2025-10-09", "2025-12-31",
                "2026-01-02", "2026-01-05",
            ]).astype("datetime64[ns]"),
        })
        panel = pd.DataFrame({
            "code": ["600001"] * 4 + ["600002"] * 2,
            "date": pd.to_datetime([
                "2025-09-30", "2025-10-09", "2025-12-31", "2026-01-05",
                "2025-12-31", "2026-01-02",
            ]).astype("datetime64[ns]"),
            "close": [10.0, 10.1, 10.2, 10.3, 20.0, 20.1],
        })
        with mock.patch.object(engineering.warehouse, "load", return_value=calendar):
            result = engineering._add_trading_gap_risk(panel)

        first = result[result["code"] == "600001"]["risk_trading_gap_days"].tolist()
        second = result[result["code"] == "600002"]["risk_trading_gap_days"].tolist()
        self.assertTrue(np.isnan(first[0]))
        self.assertEqual(first[1:], [1.0, 1.0, 2.0])
        self.assertTrue(np.isnan(second[0]))
        self.assertEqual(second[1:], [1.0])

    def test_trading_gap_risk_rejects_price_date_missing_from_calendar(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02"]).astype("datetime64[ns]"),
        })
        panel = pd.DataFrame({
            "code": ["600001", "600001"],
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        })
        with mock.patch.object(engineering.warehouse, "load", return_value=calendar), \
                self.assertRaisesRegex(RuntimeError, "2026-01-05"):
            engineering._add_trading_gap_risk(panel)

    def test_trading_gap_risk_is_diagnostic_not_model_feature(self):
        panel = pd.DataFrame({
            "ret_5d": [0.01],
            "risk_trading_gap_days": [2.0],
        })
        self.assertEqual(engineering.feature_columns(panel), ["ret_5d"])

    def test_incremental_trading_gap_retains_previous_stock_session(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime([
                "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07",
            ]).astype("datetime64[ns]"),
        })
        price = pd.DataFrame({
            "code": ["600001"] * 3,
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-07"]),
            "open": [10.0, 10.1, 10.2], "high": [10.1, 10.2, 10.3],
            "low": [9.9, 10.0, 10.1], "close": [10.0, 10.1, 10.2],
            "volume": [1000.0] * 3, "amount": [10000.0, 10100.0, 10200.0],
            "turnover": [1.0, 1.1, 1.2],
        })
        output_start = pd.Timestamp("2026-01-07")
        with mock.patch.object(engineering.warehouse, "load_price", return_value=price), \
                mock.patch.object(engineering.warehouse, "load", return_value=calendar):
            full = engineering._add_trading_gap_risk(
                engineering._price_factors("600001")
            )
            incremental = engineering._add_trading_gap_risk(
                engineering._price_factors(
                    "600001", output_start, retain_previous_row=True,
                )
            )

        expected = full[full["date"] >= output_start].reset_index(drop=True)
        actual = incremental[incremental["date"] >= output_start].reset_index(drop=True)
        self.assertEqual(actual["risk_trading_gap_days"].tolist(), [2.0])
        pd.testing.assert_frame_equal(actual, expected)

    def test_strict_calendar_factors_count_missing_market_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calendar_path = root / "trading_calendar.parquet"
            sessions = pd.to_datetime([
                "2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04",
            ]).astype("datetime64[ns]")
            pd.DataFrame({"date": sessions}).to_parquet(calendar_path, index=False)
            observed = sessions[[0, 1, 3, 4]]
            price = pd.DataFrame({
                "code": ["600001"] * 4, "date": observed,
                "open": [10.0, 10.1, 10.3, 10.4],
                "high": [10.1, 10.2, 10.4, 10.5],
                "low": [9.9, 10.0, 10.2, 10.3],
                "close": [10.0, 10.1, 10.3, 10.4],
                "volume": [1000.0] * 4, "amount": [10000.0] * 4,
                "turnover": [1.0] * 4,
            })
            with mock.patch.object(engineering.warehouse, "load_price", return_value=price),                     mock.patch.object(engineering.config, "TRADING_CALENDAR_FILE", str(calendar_path)):
                strict = engineering._price_factors(
                    "600001", strict_calendar_factors=True,
                )
                row_based = engineering._price_factors("600001")

        strict = strict.set_index("date")
        row_based = row_based.set_index("date")
        self.assertTrue(np.isnan(strict.loc[sessions[3], "ret_1d"]))
        self.assertAlmostEqual(row_based.loc[sessions[3], "ret_1d"], 10.3 / 10.1 - 1)
        self.assertAlmostEqual(strict.loc[sessions[4], "ret_3d"], 10.4 / 10.1 - 1)
        self.assertAlmostEqual(row_based.loc[sessions[4], "ret_3d"], 10.4 / 10.0 - 1)
        self.assertEqual(strict.index.tolist(), observed.tolist())

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
        calendar = pd.DataFrame({"date": dates.astype("datetime64[ns]")})
        with mock.patch.object(engineering.warehouse, "load_price", return_value=price), \
                mock.patch.object(engineering.warehouse, "load_valuation", return_value=pd.DataFrame()), \
                mock.patch.object(engineering.warehouse, "load", return_value=calendar), \
                mock.patch.object(engineering, "_asof_report_factor") as report_loader, \
                mock.patch.object(engineering, "_forecast_events") as forecast_loader, \
                mock.patch.object(engineering, "_margin_underlying") as margin_loader, \
                mock.patch.object(engineering, "_event_counts") as event_loader:
            panel = engineering.build_panel(
                codes=["600001"],
                horizon=3,
                shared_factors=shared,
                include_trading_gap_risk=True,
            )

        self.assertFalse(panel.empty)
        self.assertEqual(panel["risk_trading_gap_days"].iloc[1:].tolist(), [1.0] * 4)
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

    def test_pit_universe_resolver_uses_half_open_intervals_at_window_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master_path = root / "security_master.parquet"
            history_path = root / "index_constituent_history.parquet"
            pd.DataFrame({
                "code": ["600001", "600002", "600003", "600004"],
                "list_date": pd.to_datetime([
                    "2020-01-01", "2020-01-01", "2026-01-10", "2020-01-01",
                ]),
                "delist_date": pd.to_datetime([None, "2026-01-10", None, None]),
            }).to_parquet(master_path, index=False)
            pd.DataFrame({
                "index_code": ["000300.SH"] * 6,
                "code": ["600001", "600002", "600003", "600004", "600004", pd.NA],
                "is_standard_a_share": [True, True, True, True, True, False],
                "in_date": pd.to_datetime([
                    "2020-01-01", "2020-01-01", "2026-01-10",
                    "2020-01-01", "2026-01-09", "2020-01-01",
                ]),
                "out_date": pd.to_datetime([
                    None, "2026-01-10", None, "2026-01-05", None, None,
                ]),
            }).to_parquet(history_path, index=False)

            manifest = full_train_batched.resolve_pit_window_universe(
                "000300.SH", pd.Timestamp("2026-01-10T14:00:00"),
                master_path, history_path,
            )

        self.assertEqual(manifest["resolved_codes"], ["600001", "600003", "600004"])
        self.assertEqual(manifest["resolved_code_count"], 3)
        self.assertEqual(manifest["membership_interval"], "[in_date,out_date)")
        self.assertEqual(len(manifest["universe_hash"]), 64)
        self.assertEqual(len(manifest["manifest_hash"]), 64)
        self.assertEqual(len(manifest["sources"]["security_master"]["sha256"]), 64)

    def test_effective_pit_universe_intersects_static_codes(self):
        manifest = {
            "resolved_codes": ["600001", "600002"],
            "manifest_hash": "manifest",
        }
        with mock.patch.object(
            full_train_batched, "resolve_pit_window_universe", return_value=manifest,
        ):
            actual_manifest, codes = full_train_batched._resolve_effective_window_universe(
                "000300.SH", pd.Timestamp("2026-01-05"), {"600002", "600003"},
            )

        self.assertIs(actual_manifest, manifest)
        self.assertEqual(codes, {"600002"})

    def test_effective_pit_universe_rejects_empty_intersection(self):
        manifest = {"resolved_codes": ["600001"], "manifest_hash": "manifest"}
        with mock.patch.object(
            full_train_batched, "resolve_pit_window_universe", return_value=manifest,
        ), self.assertRaisesRegex(RuntimeError, "effective PIT universe is empty"):
            full_train_batched._resolve_effective_window_universe(
                "000300.SH", pd.Timestamp("2026-01-05"), {"600002"},
            )

    def test_pit_universe_resolver_fails_closed_on_missing_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master_path = root / "security_master.parquet"
            history_path = root / "index_constituent_history.parquet"
            pd.DataFrame({"code": ["600001"]}).to_parquet(master_path, index=False)
            pd.DataFrame({
                "index_code": ["000300.SH"], "code": ["600001"],
                "is_standard_a_share": [True], "in_date": pd.to_datetime(["2020-01-01"]),
                "out_date": pd.to_datetime([None]),
            }).to_parquet(history_path, index=False)
            with self.assertRaisesRegex(ValueError, "security master missing columns"):
                full_train_batched.resolve_pit_window_universe(
                    "000300.SH", pd.Timestamp("2026-01-10"), master_path, history_path,
                )

    def test_recipe_signature_tracks_pit_universe_manifest(self):
        first = full_train_batched._recipe_signature(
            factors=["ret_20d"], horizon=3, pit_universe_manifest_hash="first",
        )
        second = full_train_batched._recipe_signature(
            factors=["ret_20d"], horizon=3, pit_universe_manifest_hash="second",
        )
        self.assertNotEqual(first, second)

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
    def test_pipeline_requires_nonoverlapping_explicit_holdout_boundaries(self):
        result = quant_pipeline._validate_explicit_holdout_boundaries(
            "2024-12-31", "2025-06-30", "2025-07-01"
        )
        self.assertEqual(result[0], pd.Timestamp("2024-12-31"))
        with self.assertRaisesRegex(ValueError, "explicit train_end"):
            quant_pipeline._validate_explicit_holdout_boundaries(None, "2025-06-30", "2025-07-01")
        with self.assertRaisesRegex(ValueError, "train_end < valid_end < predict_start"):
            quant_pipeline._validate_explicit_holdout_boundaries(
                "2025-07-01", "2025-06-30", "2025-07-01"
            )

    def test_daily_ic_cache_persists_across_instances(self):
        rows = []
        for date in pd.to_datetime(["2026-01-05", "2026-01-06"]):
            for index in range(6):
                rows.append({
                    "date": date,
                    "factor": float(index),
                    "target_ret_1d": float(index),
                })
        panel = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            first = full_train_batched.DailyICCache(
                workers=1, cache_dir=Path(temporary)
            )
            _, hits, misses = first.get(panel, ["factor"], 1)
            self.assertEqual((hits, misses), (0, 2))
            second = full_train_batched.DailyICCache(
                workers=1, cache_dir=Path(temporary)
            )
            _, hits, misses = second.get(panel, ["factor"], 1)
        self.assertEqual((hits, misses), (2, 0))

    def test_rebalance_stride_uses_source_date_calendar_without_rephasing(self):
        dates = pd.bdate_range("2026-01-05", periods=5)
        pred = pd.DataFrame({
            "date": dates,
            "code": ["600001"] * len(dates),
            "pred": np.arange(len(dates), dtype=float),
        })
        selected = backtest._apply_rebalance_stride(pred, 2)
        self.assertEqual(selected["date"].tolist(), list(dates[::2]))
        self.assertEqual(backtest._apply_rebalance_stride(pred, 1).shape, pred.shape)

    def test_authoritative_stride_phase_survives_missing_prediction_date(self):
        dates = pd.bdate_range("2026-01-05", periods=6)
        calendar = pd.DataFrame({"date": dates})
        pred_dates = dates.delete(2)
        pred = pd.DataFrame({
            "date": pred_dates,
            "code": ["600001"] * len(pred_dates),
            "pred": np.arange(len(pred_dates), dtype=float),
        })

        selected = backtest._apply_rebalance_stride(
            pred, 2, trading_calendar=calendar,
        )

        self.assertEqual(selected["date"].tolist(), [dates[0], dates[4]])

    def test_watchlist_stride_defaults_daily_and_uses_authoritative_calendar(self):
        dates = pd.bdate_range("2026-01-05", periods=5)
        pred = pd.DataFrame({
            "date": dates,
            "code": ["600001"] * len(dates),
            "pred": np.arange(len(dates), dtype=float),
        })
        calendar = pd.DataFrame({"date": dates})

        with mock.patch.object(watchlist_grid.warehouse, "load") as load:
            daily = watchlist_grid._stride_predictions(
                pred, {"rebalance_stride": 1},
            )
            load.assert_not_called()
        with mock.patch.object(
            watchlist_grid.warehouse, "load", return_value=calendar,
        ) as load:
            strided = watchlist_grid._stride_predictions(
                pred, {"rebalance_stride": 2},
            )
            load.assert_called_once_with("trading_calendar")

        self.assertEqual(daily["date"].tolist(), dates.tolist())
        self.assertEqual(
            strided["date"].tolist(), [dates[0], dates[2], dates[4]],
        )

    def test_grid_selection_keeps_rebalance_stride_in_parameter_identity(self):
        grid = pd.DataFrame({
            "param_id": [0, 1],
            "source": ["template", "template"],
            "rebalance_stride": [1, 2],
            "horizon": [1, 1],
            "sharpe": [0.1, 0.2],
            "annual_return": [0.01, 0.02],
            "max_drawdown": [-0.1, -0.1],
            "win_rate": [0.5, 0.5],
            "direction_win_rate": [0.5, 0.5],
            "avg_turnover": [0.2, 0.2],
        })

        summary = watchlist_grid._selection_summary(grid)

        self.assertEqual(set(summary["rebalance_stride"]), {1, 2})
        self.assertEqual(len(summary), 2)

    def test_grid_selection_keeps_hold_buffer_in_parameter_identity(self):
        grid = pd.DataFrame({
            "param_id": [0, 0],
            "source": ["template", "template"],
            "hold_rank_buffer": [0, 1],
            "horizon": [1, 1],
            "sharpe": [0.1, 0.2],
            "annual_return": [0.01, 0.02],
            "max_drawdown": [-0.1, -0.1],
            "win_rate": [0.5, 0.5],
            "direction_win_rate": [0.5, 0.5],
            "avg_turnover": [0.2, 0.1],
        })
        summary = watchlist_grid._selection_summary(grid)
        self.assertEqual(len(summary), 2)
        self.assertEqual(set(summary["hold_rank_buffer"]), {0, 1})

    def test_weight_turnover_uses_stock_and_cash_weights(self):
        self.assertAlmostEqual(
            backtest._weight_turnover(
                np.zeros(2), np.asarray([0.15, 0.15]),
            ),
            0.30,
        )
        self.assertAlmostEqual(
            backtest._weight_turnover(
                np.asarray([0.15, 0.15, 0.0]),
                np.asarray([0.15, 0.0, 0.15]),
            ),
            0.15,
        )
        self.assertAlmostEqual(
            backtest._weight_turnover(
                np.asarray([0.20, 0.10]),
                np.asarray([0.10, 0.0]),
            ),
            0.20,
        )
        self.assertAlmostEqual(
            backtest._weight_turnover(
                np.asarray([0.15, 0.15]), np.zeros(2),
            ),
            0.30,
        )

    def test_portfolio_cost_tracks_partial_exposure_and_liquidation(self):
        dates = pd.bdate_range("2026-01-05", periods=3)
        rows = []
        scores = ((3.0, 2.0, 1.0), (3.0, 1.0, 2.0), (3.0, 2.0, 1.0))
        buyable = ((True, True, True), (True, True, True), (False, False, False))
        for date_index, date in enumerate(dates):
            for code_index, code in enumerate(("600001", "600002", "600003")):
                rows.append({
                    "date": date,
                    "code": code,
                    "pred": scores[date_index][code_index],
                    "target_ret_1d": 0.0,
                    "buyable_close": buyable[date_index][code_index],
                })
        returns, _ = backtest.portfolio_from_predictions(
            pd.DataFrame(rows),
            horizon=1,
            top_n=2,
            max_weight=0.15,
            filter_untradable=True,
            cost_roundtrip=0.002,
            no_refill=True,
        )

        np.testing.assert_allclose(returns["turnover"], [0.30, 0.15, 0.30])
        np.testing.assert_allclose(returns["cost"], [0.0006, 0.0003, 0.0006])
        np.testing.assert_allclose(returns["ret"], -returns["cost"])

    def test_portfolio_rank_hysteresis_retains_then_replaces_incumbents(self):
        dates = pd.date_range("2026-01-05", periods=3, freq="B")
        daily_scores = [
            {"600001": 4.0, "600002": 3.0, "600003": 2.0, "600004": 1.0},
            {"600003": 4.0, "600001": 3.0, "600002": 2.0, "600004": 1.0},
            {"600003": 4.0, "600001": 3.0, "600004": 2.0, "600002": 1.0},
        ]
        rows = [
            {
                "date": date,
                "code": code,
                "pred": score,
                "target_ret_1d": 0.0,
            }
            for date, scores in zip(dates, daily_scores)
            for code, score in scores.items()
        ]
        pred = pd.DataFrame(rows)
        implicit = backtest.portfolio_from_predictions(
            pred, horizon=1, top_n=2, max_weight=0.5,
            filter_untradable=False, cost_roundtrip=0.0,
        )
        explicit = backtest.portfolio_from_predictions(
            pred, horizon=1, top_n=2, max_weight=0.5,
            filter_untradable=False, cost_roundtrip=0.0,
            hold_rank_buffer=0,
        )
        pd.testing.assert_frame_equal(implicit[0], explicit[0])
        pd.testing.assert_frame_equal(implicit[1], explicit[1])

        _, holdings = backtest.portfolio_from_predictions(
            pred, horizon=1, top_n=2, max_weight=0.5,
            filter_untradable=False, cost_roundtrip=0.0,
            hold_rank_buffer=1,
        )
        held = {
            date: set(frame["code"])
            for date, frame in holdings.groupby("date")
        }
        self.assertEqual(held[dates[0]], {"600001", "600002"})
        self.assertEqual(held[dates[1]], {"600001", "600002"})
        self.assertEqual(held[dates[2]], {"600001", "600003"})

    def test_fast_and_formal_grid_share_rank_hysteresis(self):
        dates = pd.date_range("2026-01-05", periods=3, freq="B")
        daily_scores = [
            {"600001": 4.0, "600002": 3.0, "600003": 2.0, "600004": 1.0},
            {"600003": 4.0, "600001": 3.0, "600002": 2.0, "600004": 1.0},
            {"600003": 4.0, "600001": 3.0, "600004": 2.0, "600002": 1.0},
        ]
        pred = pd.DataFrame([
            {
                "date": date,
                "code": code,
                "pred": score,
                "base_pred": score,
                "ridge_pred": score,
                "lgbm_pred": score,
                "target_ret_1d": score / 100.0,
            }
            for date, scores in zip(dates, daily_scores)
            for code, score in scores.items()
        ])
        params = pd.Series({
            "top_n": 2,
            "gross_exposure": 0.3,
            "slot_weight": 0.15,
            "hold_rank_buffer": 1,
            "rebalance_stride": 1,
        })
        with mock.patch.object(backtest, "bt_filter_untradable", return_value=False), \
                mock.patch.object(backtest, "bt_use_open_fill", return_value=False), \
                mock.patch.object(backtest, "bt_cost_roundtrip", return_value=0.0):
            prepared = watchlist_grid._prepare_fast_grid(pred, [1])
            fast = watchlist_grid._fast_combo_metrics(
                prepared, params, "short", 1, positive_only=True,
            )
            slow = watchlist_grid._run_combo(
                pred, params, "short", 1, positive_only=True,
            )
        self.assertIsNotNone(fast)
        self.assertIsNotNone(slow)
        for key in ("avg_turnover", "sharpe", "total_return"):
            self.assertAlmostEqual(fast[key], slow[key])

    def test_fast_grid_cost_uses_weight_turnover(self):
        prepared = {
            "base": np.asarray([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0]]),
            "lgbm_z": np.zeros((2, 3)),
            "ridge_z": np.zeros((2, 3)),
            "elastic_z": None,
            "catboost_z": None,
            "extra_trees_z": None,
            "ic": np.zeros((2, 3)),
            "ridge": np.full((2, 3), np.nan),
            "rule_z": None,
            "targets": {1: np.zeros((2, 3))},
            "buyable": np.ones((2, 3), dtype=bool),
        }
        params = pd.Series({
            "top_n": 2,
            "slot_weight": 0.15,
            "pred_quantile": None,
            "ridge_quantile": None,
        })
        with mock.patch.object(backtest, "bt_cost_roundtrip", return_value=0.002):
            metrics = watchlist_grid._fast_combo_metrics(
                prepared, params, "short", 1, positive_only=False,
            )

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["avg_turnover"], 0.225)
        expected_total = (1.0 - 0.0006) * (1.0 - 0.0003) - 1.0
        self.assertAlmostEqual(metrics["total_return"], expected_total)

    def test_workflow_quant_data_dir_rebinds_parent_and_child_paths(self):
        previous_dir = scheduled_workflow.config.QUANT_DIR
        previous_env = os.environ.get("QUANT_DATA_DIR")
        child_env: dict[str, str] = {}
        try:
            with tempfile.TemporaryDirectory() as temporary:
                requested = str(Path(temporary) / "data" / ".." / "data")
                resolved = scheduled_workflow._configure_quant_data_dir(
                    requested, child_env,
                )
                expected = str((Path(temporary) / "data").resolve())
                self.assertEqual(resolved, expected)
                self.assertEqual(child_env["QUANT_DATA_DIR"], expected)
                self.assertEqual(str(scheduled_workflow._quant_dir()), expected)
                self.assertEqual(
                    scheduled_workflow.config.PRICE_DIR,
                    str(Path(expected) / "price"),
                )
        finally:
            scheduled_workflow.config.configure_quant_dir(previous_dir)
            if previous_env is None:
                os.environ.pop("QUANT_DATA_DIR", None)
            else:
                os.environ["QUANT_DATA_DIR"] = previous_env

    def test_daily_health_sample_is_deterministic_and_exchange_stratified(self):
        codes = ["600001", "600002", "000001", "300001", "830001"]
        first = check_daily_update._stratified_hash_sample(codes, 3, "seed")
        second = check_daily_update._stratified_hash_sample(codes, 3, "seed")
        self.assertEqual(first, second)
        self.assertEqual(
            {check_daily_update._market_bucket(code) for code in first},
            {"beijing", "shanghai", "shenzhen"},
        )

    def test_daily_health_stale_gate_exits_nonzero_when_enabled(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        })
        stale_price = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05"]),
        })

        def load_table(name):
            return calendar if name == "trading_calendar" else pd.DataFrame()

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(check_daily_update.datafeed, "universe", return_value=["600001"]), \
                mock.patch.object(check_daily_update.warehouse, "load_price", return_value=stale_price), \
                mock.patch.object(check_daily_update.warehouse, "load_valuation", return_value=pd.DataFrame()), \
                mock.patch.object(check_daily_update.warehouse, "load", side_effect=load_table), \
                mock.patch("builtins.print"), \
                mock.patch("sys.argv", [
                    "check_daily_update",
                    "--sample", "1",
                    "--snapshot-dir", temporary,
                    "--fail-on-stale-price",
                    "--as-of-date", "2026-01-06",
                ]), \
                self.assertRaises(SystemExit) as raised:
            check_daily_update.main()
        self.assertIn("stale sampled price data", str(raised.exception))

    def test_pipeline_selection_manifest_freezes_label_windows_and_hashes(self):
        manifest = quant_pipeline._selection_manifest(
            "factor_selection_test",
            "tradable_ret_3d",
            ["factor_b", "factor_a"],
            ["factor_a"],
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2025-06-30"),
            pd.Timestamp("2025-07-01"),
        ).iloc[0]
        self.assertEqual(manifest["label_col"], "tradable_ret_3d")
        self.assertEqual(manifest["candidate_count"], 2)
        self.assertEqual(manifest["selected_count"], 1)
        self.assertEqual(len(manifest["candidate_pool_sha256"]), 64)
        self.assertEqual(len(manifest["generator_code_sha256"]), 64)

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


class FutureLabelFeatureSafetyTest(unittest.TestCase):
    def test_feature_columns_exclude_all_horizon_targets_and_execution_labels(self):
        panel = pd.DataFrame({
            "code": ["600001"],
            "date": pd.to_datetime(["2026-08-07"]),
            "ret_5d": [0.01],
            "target_ret_1d": [0.02],
            "target_ret_3d": [0.03],
            "open_ret_3d": [0.04],
            "tradable_ret_1d": [0.01],
            "buyable_close": [True],
            "adaptive_entry_buyable": [True],
            "label_future_return": [0.02],
            "future_return": [0.03],
            "forward_return_5d": [0.04],
            "fwd_ret_5d": [0.05],
            "next_day_return": [0.06],
            "realized_return_5d": [0.07],
            "pnl_after_5d": [0.08],
            "e4_daily_prior": [0.09],
            "prior_model_score": [0.10],
            "daily_pred_rank": [0.11],
        })
        self.assertEqual(engineering.feature_columns(panel, horizon=1), ["ret_5d"])

    def test_training_boundary_rejects_unsafe_cached_factor(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe future-label factors"):
            full_train_batched._reject_unsafe_factors(
                ["ret_5d", "target_ret_3d"], "test window"
            )

    def test_selected_manifest_rejects_score_shaped_columns_but_allows_rule_score(self):
        selection = pd.DataFrame({"factor": ["rule_score", "e4_daily_prior"]})
        with mock.patch.object(warehouse, "load", return_value=selection), \
                self.assertRaisesRegex(RuntimeError, "e4_daily_prior"):
            full_train_batched._selected_factors("unsafe_selection", Path("unused"), 5)

    def test_strict_purge_uses_authoritative_sessions_across_holiday(self):
        calendar = pd.DataFrame({
            "date": pd.to_datetime([
                "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04",
            ]).astype("datetime64[ns]"),
        })
        with mock.patch.object(
            full_train_batched.warehouse, "load", return_value=calendar,
        ):
            h1 = full_train_batched._purged_end_by_calendar(
                pd.Timestamp("2026-02-04"), 1,
            )
            h2 = full_train_batched._purged_end_by_calendar(
                pd.Timestamp("2026-02-04"), 2,
            )

        self.assertEqual(h1, pd.Timestamp("2026-02-02"))
        self.assertEqual(h2, pd.Timestamp("2026-01-30"))

    def test_purge_boundary_uses_each_stock_trade_rows(self):
        frame = pd.DataFrame({
            "code": ["600001"] * 5 + ["600002"] * 3,
            "date": pd.to_datetime([
                "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05",
                "2026-01-01", "2026-01-02", "2026-01-05",
            ]),
        })
        result = full_train_batched._purged_end_by_code(
            frame, pd.Timestamp("2026-01-06"), 1
        )
        self.assertEqual(result, pd.Timestamp("2026-01-02"))

    def test_missing_ohlc_is_not_marked_buyable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            pd.DataFrame({
                "code": ["600001", "600001"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "close": [10.0, 10.1],
            }).to_parquet(price_dir / "600001.parquet", index=False)
            result = tradability.price_tradability(["600001"], [1], quant_dir=root)
        self.assertFalse(bool(result.iloc[0]["buyable_next"]))
        self.assertFalse(bool(result.iloc[0]["buyable_close"]))

    def test_strict_tradability_rejects_missing_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            pd.DataFrame({
                "code": ["600001"],
                "date": pd.to_datetime(["2026-01-01"]),
                "close": [10.0],
            }).to_parquet(price_dir / "600001.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "requires volume"):
                tradability.price_tradability(
                    ["600001"], [1], quant_dir=root, require_calendar=True,
                )

    def test_adv_gate_requires_strict_calendar_labels(self):
        with self.assertRaisesRegex(ValueError, "authoritative calendar"):
            full_train_batched._validate_execution_gate_params(
                "open-label", strict_execution_labels=False, min_adv20=1.0,
            )
        full_train_batched._validate_execution_gate_params(
            "open-label", strict_execution_labels=True, min_adv20=1.0,
        )

    def test_zero_volume_daily_bar_is_never_buyable_and_sell_rolls_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            pd.DataFrame({
                "code": ["600001"] * 3,
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "open": [10.0, 10.1, 10.2],
                "high": [10.0, 10.1, 10.2],
                "low": [10.0, 10.1, 10.2],
                "close": [10.0, 10.1, 10.2],
                "volume": [100.0, 0.0, 100.0],
            }).to_parquet(price_dir / "600001.parquet", index=False)
            result = tradability.price_tradability(["600001"], [1], quant_dir=root)

        self.assertFalse(bool(result.iloc[0]["buyable_next"]))
        self.assertFalse(bool(result.iloc[1]["buyable_close"]))
        self.assertAlmostEqual(result.iloc[0]["tradable_ret_1d"], 0.02)

    def test_sell_window_with_only_zero_volume_rows_has_no_fabricated_fill(self):
        result = tradability.rolled_sell_close(
            np.array([10.0, 9.9, 9.8]),
            np.array([False, False, False]),
            horizon=1,
            cap=2,
            sell_unavailable=np.array([False, True, True]),
        )
        self.assertTrue(np.isnan(result[0]))

    def test_pit_status_applies_dynamic_st_limit_and_suspension_roll(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            dates = pd.to_datetime([
                "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
            ]).astype("datetime64[ns]")
            pd.DataFrame({
                "code": ["600001"] * 4, "date": dates,
                "open": [10.0, 10.0, 10.5, 10.2],
                "high": [10.0, 10.0, 10.5, 10.2],
                "low": [10.0, 10.0, 10.4, 10.2],
                "close": [10.0, 10.0, 10.5, 10.2],
                "volume": [100.0] * 4,
            }).to_parquet(price_dir / "600001.parquet", index=False)
            pd.DataFrame({
                "code": ["600001"] * 4, "date": dates,
                "high_limit": [11.0, 11.0, 10.5, 11.55],
                "low_limit": [9.0, 9.0, 9.5, 9.45],
                "is_st": [False, False, True, False],
                "is_suspended": [False, True, False, False],
                "is_withdrawal": [False] * 4,
                "is_ex_right": [False] * 4,
            }).to_parquet(root / "trading_status_history.parquet", index=False)

            strict = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, require_status=True,
            )
            legacy = tradability.price_tradability(["600001"], [1], quant_dir=root)

        self.assertFalse(bool(strict.iloc[1]["buyable_close"]))
        self.assertFalse(bool(strict.iloc[2]["buyable_close"]))
        self.assertTrue(bool(legacy.iloc[2]["buyable_close"]))
        self.assertAlmostEqual(strict.iloc[0]["tradable_ret_1d"], 0.05)

    def test_pit_status_missing_date_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            dates = pd.to_datetime(["2026-01-05", "2026-01-06"]).astype("datetime64[ns]")
            pd.DataFrame({
                "code": ["600001"] * 2, "date": dates,
                "open": [10.0, 10.1], "high": [10.0, 10.1],
                "low": [10.0, 10.1], "close": [10.0, 10.1],
                "volume": [100.0, 100.0],
            }).to_parquet(price_dir / "600001.parquet", index=False)
            pd.DataFrame({
                "code": ["600001"], "date": dates[:1],
                "high_limit": [11.0], "low_limit": [9.0],
                "is_st": [False], "is_suspended": [False],
                "is_withdrawal": [False], "is_ex_right": [False],
            }).to_parquet(root / "trading_status_history.parquet", index=False)

            result = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, require_status=True,
            )

        self.assertTrue(bool(result.iloc[0]["status_present"]))
        self.assertFalse(bool(result.iloc[1]["status_present"]))
        self.assertFalse(bool(result.iloc[0]["buyable_next"]))
        self.assertFalse(bool(result.iloc[1]["buyable_close"]))
        self.assertTrue(np.isnan(result.iloc[0]["tradable_ret_1d"]))

    def test_listing_session_gate_uses_authoritative_calendar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            dates = pd.to_datetime([
                "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04",
            ]).astype("datetime64[ns]")
            pd.DataFrame({"date": dates}).to_parquet(
                root / "trading_calendar.parquet", index=False,
            )
            pd.DataFrame({
                "code": ["600001"], "list_date": pd.to_datetime(["2026-01-30"]),
            }).to_parquet(root / "security_master.parquet", index=False)
            pd.DataFrame({
                "code": ["600001"] * 4, "date": dates,
                "open": [10.0, 10.1, 10.2, 10.3],
                "high": [10.1, 10.2, 10.3, 10.4],
                "low": [9.9, 10.0, 10.1, 10.2],
                "close": [10.0, 10.1, 10.2, 10.3],
                "volume": [100.0] * 4, "amount": [1_000_000.0] * 4,
            }).to_parquet(price_dir / "600001.parquet", index=False)

            result = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, min_listing_sessions=3,
            )

        self.assertEqual(result["listing_sessions"].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(bool(result.iloc[1]["buyable_close"]))
        self.assertTrue(bool(result.iloc[2]["buyable_close"]))
        self.assertFalse(bool(result.iloc[0]["buyable_next"]))
        self.assertTrue(bool(result.iloc[1]["buyable_next"]))

    def test_absolute_adv_gate_uses_only_prior_twenty_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            dates = pd.bdate_range("2026-01-02", periods=22).astype("datetime64[ns]")
            pd.DataFrame({
                "code": ["600001"] * 22, "date": dates,
                "open": np.linspace(10.0, 10.21, 22),
                "high": np.linspace(10.0, 10.21, 22),
                "low": np.linspace(10.0, 10.21, 22),
                "close": np.linspace(10.0, 10.21, 22),
                "volume": [100.0] * 22,
                "amount": [1_000_000.0] * 20 + [100_000_000.0, 1_000_000.0],
            }).to_parquet(price_dir / "600001.parquet", index=False)

            blocked = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, min_adv20=2_000_000.0,
            )
            allowed = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, min_adv20=500_000.0,
            )

        self.assertAlmostEqual(blocked.iloc[20]["adv20"], 1_000_000.0)
        self.assertFalse(bool(blocked.iloc[20]["buyable_close"]))
        self.assertTrue(bool(allowed.iloc[20]["buyable_close"]))
        self.assertFalse(bool(blocked.iloc[19]["liquidity_pass"]))

    def test_calendar_sessions_do_not_skip_missing_suspension_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            sessions = pd.to_datetime([
                "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04",
            ]).astype("datetime64[ns]")
            pd.DataFrame({"date": sessions}).to_parquet(
                root / "trading_calendar.parquet", index=False,
            )
            observed = sessions[[0, 1, 3]]
            pd.DataFrame({
                "code": ["600001"] * 3, "date": observed,
                "open": [10.0, 10.1, 10.3], "high": [10.0, 10.1, 10.3],
                "low": [10.0, 10.1, 10.3], "close": [10.0, 10.1, 10.3],
                "volume": [100.0] * 3,
            }).to_parquet(price_dir / "600001.parquet", index=False)
            pd.DataFrame({
                "code": ["600001"] * 4, "date": sessions,
                "high_limit": [11.0, 11.0, 11.0, 11.0],
                "low_limit": [9.0, 9.0, 9.0, 9.0],
                "is_st": [False] * 4,
                "is_suspended": [False, False, True, False],
                "is_withdrawal": [False] * 4,
                "is_ex_right": [False] * 4,
            }).to_parquet(root / "trading_status_history.parquet", index=False)

            strict = tradability.price_tradability(
                ["600001"], [1], quant_dir=root,
                require_status=True, require_calendar=True,
            )
            row_based = tradability.price_tradability(
                ["600001"], [1], quant_dir=root, require_status=True,
            )

        self.assertTrue(np.isnan(strict.iloc[1]["target_ret_1d"]))
        self.assertAlmostEqual(strict.iloc[1]["tradable_ret_1d"], 10.3 / 10.1 - 1)
        self.assertAlmostEqual(row_based.iloc[1]["target_ret_1d"], 10.3 / 10.1 - 1)
        self.assertEqual(strict["date"].tolist(), observed.tolist())
        self.assertTrue(np.isnan(strict.iloc[-1]["target_ret_1d"]))

    def test_execution_targets_use_full_label_span_for_purge(self):
        self.assertEqual(full_train_batched._purge_span("baseline", 1), 1)
        self.assertEqual(full_train_batched._purge_span("tradable-label", 1), 3)
        with mock.patch.dict(os.environ, {"QUANT_BT_SELL_ROLL_MAX_DAYS": "5"}):
            self.assertEqual(full_train_batched._purge_span("tradable-label", 1), 5)
        self.assertEqual(full_train_batched._purge_span("open-label", 1), 2)
        self.assertEqual(full_train_batched._purge_span("open-buyin-mask", 1), 2)

    def test_tradable_label_recipe_tracks_sell_roll_days(self):
        with mock.patch.dict(os.environ, {"QUANT_BT_SELL_ROLL_MAX_DAYS": "2"}):
            first = full_train_batched._label_recipe_params("tradable-label")
        with mock.patch.dict(os.environ, {"QUANT_BT_SELL_ROLL_MAX_DAYS": "5"}):
            second = full_train_batched._label_recipe_params("tradable-label")
        self.assertEqual(first["sell_roll_max_days"], 2)
        self.assertEqual(second["sell_roll_max_days"], 5)
        self.assertNotEqual(
            full_train_batched._recipe_signature(**first),
            full_train_batched._recipe_signature(**second),
        )

    def test_c30_strict_gate_protects_baseline_cache_and_rejects_one_month(self):
        with mock.patch.object(
            full_train_batched, "_price_source_signature", return_value="price-hash",
        ):
            legacy = full_train_batched._label_recipe_params("baseline")
            strict = full_train_batched._label_recipe_params(
                "baseline", enforce_c30_gates=True,
            )

        self.assertEqual(legacy, {})
        self.assertEqual(strict["price_source_signature"], "price-hash")
        self.assertTrue(strict["c30_gates"])
        self.assertNotEqual(
            full_train_batched._recipe_signature(**legacy),
            full_train_batched._recipe_signature(**strict),
        )
        with self.assertRaisesRegex(ValueError, "refresh_months=0 or refresh_months>=2"):
            full_train_batched._validate_c30_refresh_months(True, 1)
        full_train_batched._validate_c30_refresh_months(True, 2)
        full_train_batched._validate_c30_refresh_months(False, 1)

    def test_c30_audit_reports_baseline_cache_and_refresh_month_risks(self):
        with mock.patch.object(
            full_train_batched, "_price_source_signature", return_value="price-hash",
        ):
            baseline = full_train_batched._c30_recipe_audit("baseline", 1)
            strict_label = full_train_batched._c30_recipe_audit("tradable-label", 2)

        self.assertEqual(baseline["price_source_signature"], "price-hash")
        self.assertFalse(baseline["baseline_price_signature_protected"])
        self.assertTrue(baseline["refresh_months_horizon_risk"])
        self.assertTrue(strict_label["baseline_price_signature_protected"])
        self.assertFalse(strict_label["refresh_months_horizon_risk"])

    def test_strict_execution_labels_isolate_cache_and_recipe(self):
        frame = pd.DataFrame({
            "code": ["600001"], "date": pd.to_datetime(["2026-01-05"]),
            "tradable_ret_1d": [0.01], "buyable_close": [True],
        })
        full_train_batched._TRAD_CACHE.clear()
        with mock.patch.object(
            full_train_batched.tradability, "price_tradability", return_value=frame,
        ) as loader, mock.patch.object(
            full_train_batched, "_price_source_signature", return_value="price-hash",
        ), mock.patch.object(
            full_train_batched, "_trading_status_source_signature", return_value="status-hash",
        ):
            full_train_batched._cached_price_tradability(["600001"], 1)
            full_train_batched._cached_price_tradability(
                ["600001"], 1, strict_execution_labels=True,
            )
            legacy = full_train_batched._label_recipe_params("tradable-label")
            strict = full_train_batched._label_recipe_params(
                "tradable-label", strict_execution_labels=True,
            )

        self.assertEqual(loader.call_count, 2)
        self.assertEqual(loader.call_args_list[0].kwargs["require_status"], False)
        self.assertEqual(loader.call_args_list[1].kwargs["require_status"], True)
        self.assertEqual(loader.call_args_list[1].kwargs["require_calendar"], True)
        self.assertNotIn("strict_execution_labels", legacy)
        self.assertNotIn("trading_status_source_signature", legacy)
        self.assertTrue(strict["strict_execution_labels"])
        self.assertEqual(strict["trading_status_source_signature"], "status-hash")
        self.assertNotEqual(
            full_train_batched._recipe_signature(**legacy),
            full_train_batched._recipe_signature(**strict),
        )
        full_train_batched._TRAD_CACHE.clear()

    def test_strict_label_recipe_tracks_status_artifact_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            status_path = root / "trading_status_history.parquet"
            status_path.write_bytes(b"status-a")
            with mock.patch.object(full_train_batched.config, "QUANT_DIR", root), mock.patch.object(
                full_train_batched, "_price_source_signature", return_value="price-hash",
            ):
                first = full_train_batched._label_recipe_params(
                    "tradable-label", strict_execution_labels=True,
                )
                status_path.write_bytes(b"status-b")
                second = full_train_batched._label_recipe_params(
                    "tradable-label", strict_execution_labels=True,
                )
                legacy = full_train_batched._label_recipe_params("tradable-label")
                first_recipe = full_train_batched._recipe_signature(**first)
                second_recipe = full_train_batched._recipe_signature(**second)
                dates = [pd.Timestamp(value) for value in (
                    "2025-01-01", "2025-07-01", "2025-08-01", "2025-09-01",
                )]
                first_window_key = full_train_batched._window_cache_key(
                    prepared, first_recipe, *dates,
                )
                second_window_key = full_train_batched._window_cache_key(
                    prepared, second_recipe, *dates,
                )

        self.assertNotEqual(
            first["trading_status_source_signature"],
            second["trading_status_source_signature"],
        )
        self.assertNotEqual(first_recipe, second_recipe)
        self.assertNotEqual(first_window_key, second_window_key)
        self.assertNotIn("trading_status_source_signature", legacy)

    def test_strict_window_cache_never_uses_legacy_fallback(self):
        self.assertTrue(full_train_batched._legacy_window_cache_allowed(False, ""))
        self.assertFalse(full_train_batched._legacy_window_cache_allowed(True, ""))
        self.assertFalse(
            full_train_batched._legacy_window_cache_allowed(False, "000852.SH")
        )

    def test_strict_tradability_cache_reloads_after_status_change(self):
        frame = pd.DataFrame({
            "code": ["600001"], "date": pd.to_datetime(["2026-01-05"]),
            "tradable_ret_1d": [0.01], "buyable_close": [True],
        })
        full_train_batched._TRAD_CACHE.clear()
        current_signature = ["status-a"]
        with mock.patch.object(
            full_train_batched.tradability, "price_tradability", return_value=frame,
        ) as loader, mock.patch.object(
            full_train_batched, "_trading_status_source_signature",
            side_effect=lambda: current_signature[0],
        ):
            full_train_batched._cached_price_tradability(
                ["600001"], 1, strict_execution_labels=True,
            )
            full_train_batched._cached_price_tradability(
                ["600001"], 1, strict_execution_labels=True,
            )
            current_signature[0] = "status-b"
            full_train_batched._cached_price_tradability(
                ["600001"], 1, strict_execution_labels=True,
            )

        self.assertEqual(loader.call_count, 2)
        full_train_batched._TRAD_CACHE.clear()

    def test_strict_tradability_cache_rejects_status_change_during_load(self):
        frame = pd.DataFrame({
            "code": ["600001"], "date": pd.to_datetime(["2026-01-05"]),
            "tradable_ret_1d": [0.01], "buyable_close": [True],
        })
        full_train_batched._TRAD_CACHE.clear()
        with mock.patch.object(
            full_train_batched.tradability, "price_tradability", return_value=frame,
        ), mock.patch.object(
            full_train_batched, "_trading_status_source_signature",
            side_effect=["status-a", "status-b"],
        ), self.assertRaisesRegex(RuntimeError, "changed during label loading"):
            full_train_batched._cached_price_tradability(
                ["600001"], 1, strict_execution_labels=True,
                expected_status_signature="status-a",
            )

        self.assertFalse(full_train_batched._TRAD_CACHE)

    def test_listing_gate_enters_training_recipe(self):
        with mock.patch.object(
            full_train_batched, "_price_source_signature", return_value="price-hash",
        ):
            first = full_train_batched._label_recipe_params(
                "tradable-label", min_listing_sessions=20,
            )
            second = full_train_batched._label_recipe_params(
                "tradable-label", min_listing_sessions=60,
            )

        self.assertEqual(first["min_listing_sessions"], 20)
        self.assertNotEqual(
            full_train_batched._recipe_signature(**first),
            full_train_batched._recipe_signature(**second),
        )

    def test_open_target_cache_signature_tracks_price_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            price_dir = root / "price"
            price_dir.mkdir()
            price = price_dir / "600001.parquet"
            price.write_bytes(b"before")
            with mock.patch.object(full_train_batched.config, "QUANT_DIR", root):
                before = full_train_batched._price_source_signature()
                price.write_bytes(b"after-price-update")
                after = full_train_batched._price_source_signature()
            self.assertNotEqual(before, after)

    def test_shared_model_boundary_rejects_unknown_future_label(self):
        frame = pd.DataFrame({
            "code": ["600001"],
            "date": pd.to_datetime(["2026-08-07"]),
            "target_ret_1d": [0.01],
            "future_return": [0.02],
        })
        with self.assertRaisesRegex(ValueError, "unsafe future-label model features"):
            quant_model._xy(frame, ["future_return"], "target_ret_1d")

    def test_factor_ic_accepts_explicit_label_column(self):
        rows = []
        for date in pd.to_datetime(["2026-01-02", "2026-01-05"]):
            for index in range(6):
                rows.append({
                    "date": date,
                    "factor": float(index),
                    "target_ret_1d": float(5 - index),
                    "tradable_ret_1d": float(index),
                })
        panel = pd.DataFrame(rows)
        result = factor_select.daily_ic(
            panel, ["factor"], horizon=1, label_col="tradable_ret_1d"
        )
        self.assertEqual(len(result), 2)
        self.assertTrue((result["ic"] > 0).all())

    def test_catboost_ranker_exposes_explicit_label_arguments(self):
        parameters = inspect.signature(quant_model.train_catboost_ranker).parameters
        self.assertIn("label_col", parameters)
        self.assertIn("train_mask_col", parameters)

    def test_purge_can_be_enabled_without_rolling_factor_selection(self):
        args = types.SimpleNamespace(
            strategy_mode="incumbent-refresh",
            purge_horizon=True,
            incumbent_purge_horizon=False,
            incumbent_rolling_factor_select=False,
        )
        self.assertTrue(scheduled_workflow._uses_purge_training(args))
        self.assertFalse(scheduled_workflow._uses_rolling_training(args))

    def test_long_horizons_require_larger_validation_windows(self):
        self.assertEqual(scheduled_workflow._minimum_validation_months(1), 1)
        self.assertEqual(scheduled_workflow._minimum_validation_months(3), 2)
        self.assertEqual(scheduled_workflow._minimum_validation_months(10), 3)
        self.assertEqual(scheduled_workflow._minimum_validation_months(15), 3)

    def test_short_grid_failure_preserves_incumbent_parameters(self):
        args = types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prefix_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet"
            source.write_bytes(b"prediction-artifact")
            with mock.patch.object(scheduled_workflow, "_quant_dir", return_value=root), \
                    mock.patch.object(scheduled_workflow, "_optimization_universe", return_value=["600001"]), \
                    mock.patch.object(
                        scheduled_workflow.watchlist_grid,
                        "run_grid",
                        side_effect=RuntimeError("grid unavailable"),
                    ) as run_grid:
                result = scheduled_workflow.run_short_grid(
                    {"short_1_3": "prefix"}, args, {"top_n": 2},
                )
        self.assertIsNone(result)
        self.assertEqual(
            run_grid.call_args.kwargs["fixed_params"],
            {"rebalance_stride": 1, "hold_rank_buffer": 0},
        )

    def test_ridge_only_keeps_ridge_cohort_without_lightgbm_join(self):
        dates = pd.to_datetime(["2026-01-05", "2026-01-05"])
        ridge = types.SimpleNamespace(
            ok=True,
            predictions=pd.DataFrame({
                "code": ["600001", "600002"],
                "date": dates,
                "pred": [0.2, 0.1],
                "target_ret_1d": [0.01, 0.02],
            }),
        )
        lgbm = types.SimpleNamespace(
            ok=True,
            predictions=pd.DataFrame({
                "code": ["600001"],
                "date": dates[:1],
                "pred": [0.3],
                "target_ret_1d": [0.01],
            }),
        )
        current = pd.Timestamp("2026-01-05")
        test_end = pd.Timestamp("2026-01-06")
        ensemble = full_train_batched._primary_window_predictions(
            ridge, lgbm, current, test_end, ridge_only=False,
        )
        ridge_only = full_train_batched._primary_window_predictions(
            ridge, None, current, test_end, ridge_only=True,
        )
        self.assertEqual(list(ensemble["code"]), ["600001"])
        self.assertEqual(set(ridge_only["code"]), {"600001", "600002"})
        self.assertTrue(ridge_only["lgbm_pred"].isna().all())
        self.assertEqual(
            ridge_only.set_index("code")["ridge_pred"].to_dict(),
            {"600001": 0.2, "600002": 0.1},
        )

    def test_ridge_only_changes_window_recipe_identity(self):
        base = {
            "factors": ["factor_a", "factor_b"],
            "horizon": 1,
            "lgbm_weight": 0.0,
        }
        ensemble = full_train_batched._recipe_signature(
            **base, ridge_only=False,
        )
        ridge_only = full_train_batched._recipe_signature(
            **base, ridge_only=True,
        )
        self.assertNotEqual(ensemble, ridge_only)
        self.assertIn(
            "ridge_only",
            inspect.signature(full_train_batched.train_batched).parameters,
        )
        with mock.patch.object(
            full_train_batched.qmodel, "train_lightgbm_ranker",
        ) as train_lightgbm:
            result = full_train_batched._train_lightgbm_window(True)
        self.assertIsNone(result)
        train_lightgbm.assert_not_called()

    def test_strict_execution_label_join_fails_closed_on_sparse_status(self):
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        window = pd.DataFrame({
            "code": ["600001"] * len(dates),
            "date": dates,
        })
        sparse = pd.DataFrame({
            "code": ["600001"],
            "date": [dates[0]],
            "tradable_ret_1d": [0.01],
            "buyable_close": [True],
        })
        with mock.patch.object(
            full_train_batched, "_cached_price_tradability",
            return_value=sparse,
        ):
            legacy = full_train_batched._join_tradability(
                window, 1, strict_execution_labels=False,
            )
            with self.assertRaisesRegex(RuntimeError, "tradable_ret NaN rate"):
                full_train_batched._join_tradability(
                    window, 1, strict_execution_labels=True,
                )
        self.assertEqual(legacy["tradable_ret_1d"].notna().sum(), 1)

    def test_join_tradability_forwards_expected_status_signature(self):
        window = pd.DataFrame({
            "code": ["600001"], "date": pd.to_datetime(["2026-01-05"]),
        })
        tradability_frame = window.assign(
            tradable_ret_1d=0.01, buyable_close=True,
        )
        with mock.patch.object(
            full_train_batched, "_cached_price_tradability",
            return_value=tradability_frame,
        ) as cached:
            full_train_batched._join_tradability(
                window, 1, strict_execution_labels=True,
                expected_status_signature="status-hash",
            )

        self.assertEqual(
            cached.call_args.kwargs["expected_status_signature"], "status-hash"
        )

    def test_selection_provenance_validates_target_and_factor_hash(self):
        selected = ["factor_a", "factor_b"]
        manifest = quant_pipeline._selection_manifest(
            "selection_demo", "tradable_ret_1d", ["factor_a", "factor_b", "factor_c"],
            selected, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01"),
            pd.Timestamp("2025-07-01"),
        )
        with mock.patch.object(
            full_train_batched.warehouse, "load", return_value=manifest,
        ):
            signature = full_train_batched._validate_selection_provenance(
                "selection_demo", selected, "tradable_ret_1d",
            )
        self.assertEqual(len(signature), 64)
        with mock.patch.object(
            full_train_batched.warehouse, "load", return_value=manifest,
        ), self.assertRaisesRegex(RuntimeError, "label mismatch"):
            full_train_batched._validate_selection_provenance(
                "selection_demo", selected, "target_ret_1d",
            )

    def test_rolling_selection_uses_prepared_features_not_fixed_selection(self):
        sample = pd.DataFrame({
            "code": ["600001"], "date": pd.to_datetime(["2025-01-02"]),
            "factor_from_panel": [1.0], "target_ret_1d": [0.01],
        })
        with mock.patch.object(
            full_train_batched, "_prepared_files", return_value=[Path("2025-01.parquet")],
        ), mock.patch.object(pd, "read_parquet", return_value=sample), mock.patch.object(
            full_train_batched.warehouse, "load",
        ) as load:
            factors = full_train_batched._selected_factors(
                "legacy_selection", Path("prepared"), 1,
                require_selection_provenance=True, rolling_factor_select=True,
            )
        load.assert_not_called()
        self.assertIn("factor_from_panel", factors)

    def test_rolling_selection_manifest_binds_purged_window_and_factor_hashes(self):
        row = full_train_batched._rolling_selection_manifest_row(
            3,
            pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-25"),
            pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-25"),
            pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01"),
            "tradable_ret_1d", ["factor_b", "factor_a"], ["factor_a"], 5,
        )
        self.assertEqual(row["train_end"], "2025-06-25")
        self.assertEqual(row["label_col"], "tradable_ret_1d")
        self.assertEqual(row["purge_span"], 5)
        self.assertEqual(row["candidate_count"], 2)
        self.assertEqual(row["selected_count"], 1)
        self.assertEqual(len(row["candidate_pool_sha256"]), 64)
        self.assertEqual(len(row["selected_sha256"]), 64)
        self.assertEqual(len(row["generator_code_sha256"]), 64)
        self.assertEqual(len(row["manifest_hash"]), 64)

    def test_strict_rolling_provenance_disables_window_cache(self):
        self.assertTrue(
            full_train_batched._rolling_cache_allowed(
                require_selection_provenance=True, rolling_factor_select=True,
            ) is False
        )
        self.assertTrue(
            full_train_batched._rolling_cache_allowed(
                require_selection_provenance=False, rolling_factor_select=True,
            ) is True
        )

    def test_selection_provenance_is_required_only_when_enabled(self):
        empty = pd.DataFrame()
        with mock.patch.object(full_train_batched.warehouse, "load", return_value=empty), \
                self.assertRaisesRegex(RuntimeError, "provenance unavailable"):
            full_train_batched._validate_selection_provenance(
                "missing_selection", ["factor_a"], "target_ret_1d",
            )

    def test_strict_panel_options_are_reachable_from_training_entry(self):
        parameters = inspect.signature(full_train_batched.build_monthly_panel).parameters
        self.assertIn("include_trading_gap_risk", parameters)
        self.assertIn("strict_calendar_factors", parameters)
        self.assertIn("strict_announcement_lag", parameters)

        command = ["python", "-m", "quant.full_train_batched"]
        args = types.SimpleNamespace(
            include_trading_gap_risk=True,
            strict_calendar_factors=True,
            strict_announcement_lag=True,
            strict_pit_min_price_rows=True,
            strict_execution_labels=True,
            require_selection_provenance=True,
            enforce_c30_gates=True,
            train_target_mode="tradable-label",
            min_adv20=50_000_000.0,
            min_listing_sessions=60,
        )
        scheduled_workflow._append_research_training_args(command, args)
        self.assertIn("--include-trading-gap-risk", command)
        self.assertIn("--strict-calendar-factors", command)
        self.assertIn("--strict-announcement-lag", command)
        self.assertIn("--strict-pit-min-price-rows", command)
        self.assertIn("--strict-execution-labels", command)
        self.assertIn("--require-selection-provenance", command)
        self.assertIn("--enforce-c30-gates", command)
        self.assertEqual(command[command.index("--train-target-mode") + 1], "tradable-label")
        self.assertEqual(command[command.index("--min-listing-sessions") + 1], "60")

    def test_research_training_options_remain_default_off(self):
        command = ["python", "-m", "quant.full_train_batched"]
        scheduled_workflow._append_research_training_args(
            command, types.SimpleNamespace(),
        )
        self.assertEqual(command, ["python", "-m", "quant.full_train_batched"])

    def test_derived_style_reports_actual_source_prediction_horizon(self):
        requested = {"short_1_3": 1, "swing_7_15": 10}

        published = scheduled_workflow._published_prediction_horizons(requested)

        self.assertEqual(published, {"short_1_3": 1, "swing_7_15": 1})
        self.assertEqual(requested, {"short_1_3": 1, "swing_7_15": 10})
        self.assertEqual(
            scheduled_workflow.TRAINING_PROFILE["source"],
            "short_model_predictions_reused_by_derived_styles",
        )


if __name__ == "__main__":
    unittest.main()
