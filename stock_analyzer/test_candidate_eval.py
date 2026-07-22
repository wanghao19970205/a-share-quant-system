import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from stock_analyzer import candidate_eval, moneyflow


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"])


class MoneyFlowHistoryTest(unittest.TestCase):
    def test_price_volume_history_matches_pc_proxy_semantics(self):
        dates = pd.date_range("2026-07-01", periods=16, freq="B")
        closes = [10, 11, 10, 10, 12, 11, 13, 14, 13, 15, 16, 15, 17, 16, 18, 19]
        frame = pd.DataFrame({
            "date": dates,
            "close": closes,
            "volume": [100] * 16,
            "amount": [1000 + index * 10 for index in range(16)],
        })

        result = moneyflow.price_volume_history(frame, days=14)

        self.assertEqual(len(result), 14)
        self.assertGreater(result.iloc[-1]["net_amount"], 0)
        self.assertLess(result.iloc[-3]["net_amount"], 0)
        self.assertEqual(result.iloc[0]["date"], dates[2])


class BrokerSourcePriorityTest(unittest.TestCase):
    def test_broker_sdk_is_used_before_free_sources(self):
        broker = _frame([[pd.Timestamp("2026-07-18"), 10, 11, 9, 10.5, 100, 1000, 1.0, 5.0]])
        fallback = mock.Mock(return_value=broker.assign(close=9.5))
        with mock.patch.object(candidate_eval.data.amazingdata_source, "available", return_value=True), \
                mock.patch.object(candidate_eval.data, "_from_amazingdata", return_value=broker) as sdk, \
                mock.patch.object(candidate_eval.data, "_SOURCES", (("免费兜底", fallback, 1),)):
            result = candidate_eval.data.fetch_daily.__wrapped__("600001", days=10)

        sdk.assert_called_once()
        fallback.assert_not_called()
        self.assertEqual(float(result.iloc[-1]["close"]), 10.5)
        self.assertEqual(candidate_eval.data.last_source("600001"), "银河AmazingData")

    def test_free_source_is_used_only_after_broker_failure(self):
        fallback_frame = _frame([[pd.Timestamp("2026-07-18"), 10, 11, 9, 9.5, 100, 1000, 1.0, -5.0]])
        fallback = mock.Mock(return_value=fallback_frame)
        with mock.patch.object(candidate_eval.data.amazingdata_source, "available", return_value=True), \
                mock.patch.object(candidate_eval.data, "_from_amazingdata", side_effect=RuntimeError("broker offline")) as sdk, \
                mock.patch.object(candidate_eval.data, "_SOURCES", (("免费兜底", fallback, 1),)):
            result = candidate_eval.data.fetch_daily.__wrapped__("600002", days=10)

        sdk.assert_called_once()
        fallback.assert_called_once()
        self.assertEqual(float(result.iloc[-1]["close"]), 9.5)
        self.assertEqual(candidate_eval.data.last_source("600002"), "免费兜底")


class CandidateMarketMergeTest(unittest.TestCase):
    def test_quant_intraday_bar_is_appended_when_network_stops_previous_day(self):
        today = pd.Timestamp(dt.date.today())
        previous = today - pd.Timedelta(days=1)
        network = _frame([[previous, 1.70, 1.86, 1.69, 1.86, 100, 1000, 1.0, 10.0]])
        local = _frame([
            [previous, 1.70, 1.86, 1.69, 1.86, 100, 1000, 1.0, 10.0],
            [today, 1.98, 2.05, 1.81, 1.99, 200, 2000, 2.0, 6.99],
        ])
        with tempfile.TemporaryDirectory() as directory:
            price_dir = Path(directory) / "price"
            price_dir.mkdir()
            local.to_parquet(price_dir / "600881.parquet", index=False)
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}), \
                    mock.patch.object(candidate_eval.data, "fetch_daily", return_value=network):
                merged, source = candidate_eval._merged_market_frame("600881")

        self.assertEqual(merged.iloc[-1]["date"], today)
        self.assertEqual(float(merged.iloc[-1]["close"]), 1.99)
        self.assertEqual(source, "午间/收盘日更仓")

    def test_quant_bar_is_used_when_network_request_fails(self):
        today = pd.Timestamp(dt.date.today())
        local = _frame([
            [today - pd.Timedelta(days=1), 1.70, 1.86, 1.69, 1.86, 100, 1000, 1.0, 10.0],
            [today, 1.98, 2.05, 1.81, 1.99, 200, 2000, 2.0, 6.99],
        ])
        with tempfile.TemporaryDirectory() as directory:
            price_dir = Path(directory) / "price"
            price_dir.mkdir()
            local.to_parquet(price_dir / "600881.parquet", index=False)
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}), \
                    mock.patch.object(candidate_eval.data, "fetch_daily", side_effect=RuntimeError("offline")):
                merged, source = candidate_eval._merged_market_frame("600881")

        self.assertEqual(len(merged), 2)
        self.assertEqual(float(merged.iloc[-1]["close"]), 1.99)
        self.assertEqual(source, "午间/收盘日更仓")

    def test_network_bar_overrides_quant_bar_on_same_date(self):
        today = pd.Timestamp(dt.date.today())
        network = _frame([[today, 1.98, 2.05, 1.81, 1.93, 300, 3000, 3.0, 3.76]])
        local = _frame([[today, 1.98, 2.05, 1.81, 1.99, 200, 2000, 2.0, 6.99]])
        with tempfile.TemporaryDirectory() as directory:
            price_dir = Path(directory) / "price"
            price_dir.mkdir()
            local.to_parquet(price_dir / "600881.parquet", index=False)
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}), \
                    mock.patch.object(candidate_eval.data, "fetch_daily", return_value=network):
                merged, source = candidate_eval._merged_market_frame("600881")

        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[-1]["close"]), 1.93)
        self.assertEqual(source, "行情接口")

    def test_newer_network_bar_is_not_overwritten_by_old_quant_bar(self):
        today = pd.Timestamp(dt.date.today())
        previous = today - pd.Timedelta(days=1)
        network = _frame([[today, 1.98, 2.05, 1.81, 1.93, 300, 3000, 3.0, 3.76]])
        local = _frame([[previous, 1.70, 1.86, 1.69, 1.86, 100, 1000, 1.0, 10.0]])
        with tempfile.TemporaryDirectory() as directory:
            price_dir = Path(directory) / "price"
            price_dir.mkdir()
            local.to_parquet(price_dir / "600881.parquet", index=False)
            with mock.patch.dict(os.environ, {"QUANT_DATA_DIR": directory}), \
                    mock.patch.object(candidate_eval.data, "fetch_daily", return_value=network):
                merged, source = candidate_eval._merged_market_frame("600881")

        self.assertEqual(float(merged.iloc[-1]["close"]), 1.93)
        self.assertEqual(source, "行情接口")

    def test_candidate_uses_last_positive_close_for_percent_change(self):
        today = pd.Timestamp(dt.date.today())
        market = _frame([
            [today - pd.Timedelta(days=2), 10, 10, 10, 10, 100, 1000, 1.0, 0.0],
            [today - pd.Timedelta(days=1), 0, 0, 0, 0, 0, 0, 0.0, 0.0],
            [today, 11, 11, 11, 11, 100, 1100, 1.0, 10.0],
        ])
        prediction_result = mock.Mock(
            direction="偏多", level="bullish", confidence="中", composite=1.0,
            logic="ok", action="hold", engine="test",
        )
        with mock.patch.object(candidate_eval, "_merged_market_frame", return_value=(market, "test")), \
                mock.patch.object(candidate_eval.data, "get_stock_name", return_value="测试"), \
                mock.patch.object(candidate_eval.advisor, "advise", return_value=None), \
                mock.patch.object(candidate_eval.overseas, "analyze", return_value=None), \
                mock.patch.object(candidate_eval.sectors, "analyze_linkage", return_value=None), \
                mock.patch.object(candidate_eval.news, "analyze", return_value=None), \
                mock.patch.object(candidate_eval.moneyflow, "analyze", return_value=None), \
                mock.patch.object(candidate_eval.broker_extra, "analyze", return_value=None), \
                mock.patch.object(candidate_eval.quant_signal, "get", return_value=None), \
                mock.patch.object(candidate_eval.sentiment_signal, "score_at", return_value=None), \
                mock.patch.object(candidate_eval.prediction, "predict", return_value=prediction_result):
            result = candidate_eval.evaluate_candidate("600001")

        self.assertEqual(result["pct"], 10.0)

    def test_evaluate_top_normalizes_shared_llm_base_url(self):
        calls = []

        def evaluate(code, **kwargs):
            calls.append((kwargs["base_url"], kwargs["freshness_bucket"]))
            return {"code": code, "available": True, "rank_score": 1.0}

        with mock.patch.object(
            candidate_eval.llm,
            "get_base_url",
            return_value="https://example.test/v1",
        ), mock.patch.object(
            candidate_eval,
            "evaluate_candidate",
            side_effect=evaluate,
        ), mock.patch.object(
            candidate_eval.overseas,
            "analyze",
            return_value=None,
        ), mock.patch.object(
            candidate_eval.sectors,
            "analyze_sectors",
            return_value=None,
        ), mock.patch.object(
            candidate_eval.news,
            "analyze_sector_news",
            return_value=None,
        ) as sector_news:
            candidate_eval.evaluate_top(
                ["600001"],
                base_url="",
                freshness_bucket=123,
            )

        sector_news.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            "https://example.test/v1",
        )
        self.assertEqual(calls, [("https://example.test/v1", 123)])

    def test_candidate_failure_does_not_discard_successful_batch_rows(self):
        def evaluate(code, **_kwargs):
            if code == "600001":
                raise RuntimeError("upstream failed")
            return {"code": code, "available": True, "rank_score": 1.0}

        with mock.patch.object(candidate_eval, "evaluate_candidate", side_effect=evaluate), \
                mock.patch.object(candidate_eval.overseas, "analyze", return_value=None), \
                mock.patch.object(candidate_eval.sectors, "analyze_sectors", return_value=None), \
                mock.patch.object(candidate_eval.news, "analyze_sector_news", return_value=None):
            result = candidate_eval.evaluate_top(["600001", "600002"], max_workers=2)

        rows = {row["code"]: row for row in result["rows"]}
        self.assertTrue(rows["600002"]["available"])
        self.assertFalse(rows["600001"]["available"])
        self.assertIn("RuntimeError", rows["600001"]["note"])

    def test_action_does_not_call_reference_price_today_limit_without_evidence(self):
        latest = pd.Series({"pct_change": 6.99, "close": 1.99, "high": 2.05})

        action = candidate_eval._sanitize_limit_claims(
            "跌破1.86（今日涨停价）则考虑减仓", "亚泰集团", latest)

        self.assertEqual(action, "跌破1.86（参考价）则考虑减仓")

    def test_action_keeps_limit_word_when_quote_supports_it(self):
        latest = pd.Series({"pct_change": 10.02, "close": 2.05, "high": 2.05})

        action = candidate_eval._sanitize_limit_claims(
            "跌破2.05（今日涨停价）则考虑减仓", "亚泰集团", latest)

        self.assertIn("今日涨停价", action)

    def test_logic_does_not_claim_limit_down_without_matching_low(self):
        latest = pd.Series({"pct_change": -9.7, "close": 45.10, "high": 48.80, "low": 44.91})

        logic = candidate_eval._sanitize_limit_claims(
            "虽当日跌停，但抛压衰竭", "科伦药业", latest)

        self.assertEqual(logic, "虽当日下跌，但抛压衰竭")

    def test_logic_keeps_limit_down_when_quote_supports_it(self):
        latest = pd.Series({"pct_change": -10.0, "close": 44.91, "high": 48.80, "low": 44.91})

        logic = candidate_eval._sanitize_limit_claims(
            "虽当日跌停，但抛压衰竭", "科伦药业", latest)

        self.assertIn("当日跌停", logic)

    def test_logic_replaces_stale_limit_down_claim_on_up_day(self):
        latest = pd.Series({"pct_change": 6.44, "close": 47.80, "high": 49.40, "low": 42.69})

        logic = candidate_eval._sanitize_limit_claims(
            "虽当日跌停，但抛压衰竭", "科伦药业", latest)

        self.assertEqual(logic, "虽当日波动，但抛压衰竭")


if __name__ == "__main__":
    unittest.main()
