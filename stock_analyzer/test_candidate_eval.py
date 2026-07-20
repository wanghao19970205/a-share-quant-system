import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from stock_analyzer import candidate_eval


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"])


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

    def test_quant_bar_overrides_network_bar_on_same_date(self):
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
        self.assertEqual(float(merged.iloc[-1]["close"]), 1.99)
        self.assertEqual(source, "午间/收盘日更仓")

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
