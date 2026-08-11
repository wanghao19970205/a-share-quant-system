"""复权换算单测：验证 amazingdata_source._apply_adjust 的 qfq/hfq 数学。

不依赖网络/券商 SDK：直接把构造的因子序列传给 _apply_adjust，
校验前复权(归一化到最新日)、后复权、不复权三种口径的价格换算是否正确。
容器内跑：docker exec a-scheduler-1 python3 -m unittest stock_analyzer.test_amazingdata_adjust
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from stock_analyzer import amazingdata_source as a


def _kline(dates, closes):
    """构造标准化后的最小 K 线（open=high=low=close，便于核对）。"""
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100] * len(closes),
    })


class TradingCalendarTest(unittest.TestCase):
    def test_calendar_normalizes_sequence_to_sorted_dates(self):
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_base", mock.Mock()), \
                mock.patch.object(a, "sdk_call", return_value=["20240103", "20240102"]):
            result = a.fetch_trading_calendar()

        self.assertEqual(result["date"].tolist(), [
            pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
        ])

    def test_calendar_rejects_duplicate_or_invalid_dates(self):
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_base", mock.Mock()), \
                mock.patch.object(a, "sdk_call", return_value=["20240102", "20240102"]), \
                self.assertRaisesRegex(ValueError, "duplicate"):
            a.fetch_trading_calendar()
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_base", mock.Mock()), \
                mock.patch.object(a, "sdk_call", return_value=["not-a-date"]), \
                self.assertRaisesRegex(ValueError, "invalid"):
            a.fetch_trading_calendar()

    def test_security_master_normalizes_listing_intervals(self):
        basic = pd.DataFrame({
            "MARKET_CODE": ["600000.SH", "000001.SZ"],
            "SECURITY_NAME": ["A", "B"],
            "LISTDATE": ["19991110", "19910403"],
            "DELISTDATE": [None, "20240105"],
            "LISTPLATE_NAME": ["main", "main"],
            "IS_LISTED": [1, 0],
        })
        base = mock.Mock()
        info = mock.Mock()
        ad = mock.Mock()
        ad.InfoData.return_value = info
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_base", base), \
                mock.patch.object(a, "_ad", ad), \
                mock.patch.object(a, "sdk_call", side_effect=[["600000.SH", "000001.SZ"], basic]):
            result = a.fetch_security_master()

        self.assertEqual(result["code"].tolist(), ["000001", "600000"])
        self.assertEqual(result.loc[0, "delist_date"], pd.Timestamp("2024-01-05"))
        self.assertEqual(str(result["list_date"].dtype), "datetime64[ns]")

    def test_index_history_normalizes_membership_intervals(self):
        raw = {"000300.SH": pd.DataFrame({
            "INDEX_CODE": ["000300.SH", "000300.SH"],
            "CON_CODE": ["600000.SH", "000001.SZ"],
            "INDATE": ["20050408", "20050408"],
            "OUTDATE": [None, "20061019"],
            "INDEX_NAME": ["HS300", "HS300"],
        })}
        info = mock.Mock()
        ad = mock.Mock()
        ad.InfoData.return_value = info
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_ad", ad), \
                mock.patch.object(a, "sdk_call", return_value=raw):
            result = a.fetch_index_constituent_history(["000300.SH"])

        self.assertEqual(result["code"].tolist(), ["000001", "600000"])
        self.assertEqual(result.loc[0, "out_date"], pd.Timestamp("2006-10-19"))
        self.assertEqual(str(result["in_date"].dtype), "datetime64[ns]")

    def test_index_history_preserves_nonstandard_historical_members(self):
        raw = pd.DataFrame({
            "INDEX_CODE": ["000300.SH"], "CON_CODE": ["T00018.SH"],
            "INDATE": ["20050408"], "OUTDATE": ["20061019"],
            "INDEX_NAME": ["HS300"],
        })
        ad = mock.Mock()
        ad.InfoData.return_value = mock.Mock()
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_ad", ad), \
                mock.patch.object(a, "sdk_call", return_value=raw):
            result = a.fetch_index_constituent_history(["000300.SH"])

        self.assertEqual(result.loc[0, "market_code"], "T00018.SH")
        self.assertTrue(pd.isna(result.loc[0, "code"]))
        self.assertFalse(result.loc[0, "is_standard_a_share"])

    def test_index_history_rejects_reversed_interval(self):
        raw = pd.DataFrame({
            "INDEX_CODE": ["000300.SH"], "CON_CODE": ["600000.SH"],
            "INDATE": ["20240105"], "OUTDATE": ["20240104"],
            "INDEX_NAME": ["HS300"],
        })
        ad = mock.Mock()
        ad.InfoData.return_value = mock.Mock()
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_ad", ad), \
                mock.patch.object(a, "sdk_call", return_value=raw), \
                self.assertRaisesRegex(ValueError, "before in_date"):
            a.fetch_index_constituent_history(["000300.SH"])

    def test_history_stock_status_normalizes_pit_limits_and_flags(self):
        raw = {"600000.SH": pd.DataFrame({
            "MARKET_CODE": ["600000.SH", "600000.SH"],
            "TRADE_DATE": ["20240102", "20240103"],
            "PRECLOSE": [10.0, 10.1],
            "HIGH_LIMITED": [11.0, 11.11], "LOW_LIMITED": [9.0, 9.09],
            "PRICE_HIGH_LMT_RATE": [0.1, 0.1], "PRICE_LOW_LMT_RATE": [0.1, 0.1],
            "IS_ST_SEC": ["0", "1"], "IS_SUSP_SEC": ["0", "1"],
            "IS_WD_SEC": ["0", "0"], "IS_XR_SEC": ["0", "1"],
        })}
        ad = mock.Mock()
        ad.InfoData.return_value = mock.Mock()
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_ad", ad), \
                mock.patch.object(a, "sdk_call", return_value=raw):
            result = a.fetch_history_stock_status(["600000"])

        self.assertEqual(str(result["date"].dtype), "datetime64[ns]")
        self.assertEqual(result["code"].tolist(), ["600000", "600000"])
        self.assertEqual(result["is_st"].tolist(), [False, True])
        self.assertEqual(result["is_suspended"].tolist(), [False, True])
        self.assertEqual(result["high_limit"].tolist(), [11.0, 11.11])


class NormalizeKlineTest(unittest.TestCase):
    @staticmethod
    def _frame(**columns):
        base = {
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
        }
        base.update(columns)
        return pd.DataFrame(base)

    def test_preserves_unnamed_datetime_index(self):
        frame = self._frame()
        frame.index = pd.to_datetime(["2024-01-02", "2024-01-03"])

        result = a._normalize_kline(frame)

        self.assertEqual(result["date"].tolist(), list(frame.index))

    def test_parses_compact_integer_dates_as_calendar_dates(self):
        result = a._normalize_kline(self._frame(date=[20240103, 20240102]))

        self.assertEqual(result["date"].tolist(), [
            pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
        ])

    def test_rejects_missing_date_source(self):
        with self.assertRaisesRegex(ValueError, "date column unavailable"):
            a._normalize_kline(self._frame())

    def test_rejects_columns_that_both_normalize_to_date(self):
        frame = self._frame(
            date=["2024-01-02", "2024-01-03"],
            kline_time=["2024-01-02", "2024-01-03"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate normalized K-line columns"):
            a._normalize_kline(frame)

    def test_rejects_invalid_or_duplicate_dates(self):
        with self.assertRaisesRegex(ValueError, "invalid values"):
            a._normalize_kline(self._frame(date=["2024-01-02", "not-a-date"]))
        with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
            a._normalize_kline(self._frame(date=["2024-01-02", "2024-01-02"]))


class ApplyAdjustTest(unittest.TestCase):
    def setUp(self):
        # 3 个交易日，含一次除权：原始价机械跳空（10 -> 5），因子在除权日翻倍。
        self.dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        self.raw = _kline(self.dates, [10.0, 5.0, 5.5])
        # 后复权因子：除权前 1.0，除权后 2.0（价格砍半 → 因子翻倍以还原连续性）
        self.factor = pd.Series(
            [1.0, 2.0, 2.0], index=pd.to_datetime(self.dates))

    def test_qfq_normalizes_to_latest(self):
        """前复权：归一化到最新日因子(2.0)，最新价保持真实，历史被缩放。"""
        out = a._apply_adjust(self.raw, self.factor, "qfq")
        c = out["close"].tolist()
        # base=2.0: day1 10*1/2=5.0, day2 5*2/2=5.0, day3 5.5*2/2=5.5
        self.assertAlmostEqual(c[0], 5.0, places=6)
        self.assertAlmostEqual(c[1], 5.0, places=6)
        self.assertAlmostEqual(c[2], 5.5, places=6)  # 最新日 == 原始价
        # 前复权后除权跳空被抹平：day1->day2 不再是 -50% 的假跌
        self.assertAlmostEqual(c[0], c[1], places=6)

    def test_hfq_multiplies_factor(self):
        """后复权：raw * factor，最早价保持真实。"""
        out = a._apply_adjust(self.raw, self.factor, "hfq")
        c = out["close"].tolist()
        self.assertAlmostEqual(c[0], 10.0, places=6)   # 10*1
        self.assertAlmostEqual(c[1], 10.0, places=6)   # 5*2
        self.assertAlmostEqual(c[2], 11.0, places=6)   # 5.5*2

    def test_no_adjust_passthrough(self):
        """adjust="" 原样返回，不动价格。"""
        out = a._apply_adjust(self.raw, self.factor, "")
        self.assertEqual(out["close"].tolist(), [10.0, 5.0, 5.5])

    def test_all_price_cols_scaled_volume_untouched(self):
        """OHLC 全部按同一因子缩放，成交量不变。"""
        out = a._apply_adjust(self.raw, self.factor, "qfq")
        for col in ("open", "high", "low", "close"):
            self.assertAlmostEqual(out[col].iloc[0], 5.0, places=6)
        self.assertEqual(out["volume"].tolist(), [100, 100, 100])

    def test_missing_factor_falls_back_to_raw(self):
        """取不到因子时回退原始价，不报错、不返回错价。"""
        out = a._apply_adjust(self.raw, None, "qfq")
        self.assertEqual(out["close"].tolist(), [10.0, 5.0, 5.5])

    def test_strict_adjustment_rejects_missing_factor(self):
        with self.assertRaisesRegex(RuntimeError, "adjustment factor unavailable for 600000.SH"):
            a._apply_adjust(
                self.raw,
                None,
                "qfq",
                require_adjustment_factor=True,
                symbol="600000.SH",
            )

    def test_strict_adjustment_rejects_uncovered_price_dates(self):
        factor = pd.Series(
            [2.0, 2.0], index=pd.to_datetime(self.dates[1:]),
        )
        with self.assertRaisesRegex(RuntimeError, "does not cover 1/3 price dates"):
            a._apply_adjust(
                self.raw,
                factor,
                "qfq",
                require_adjustment_factor=True,
                symbol="600000.SH",
            )

    def test_fetch_daily_propagates_strict_adjustment_policy(self):
        with mock.patch.object(a, "raw_kline", return_value=self.raw), \
                mock.patch.object(a, "_backward_factor", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "adjustment factor unavailable for 600000.SH"):
            a.fetch_daily(
                "600000", "20240101", "20240103",
                require_adjustment_factor=True,
            )

    def test_fetch_daily_batch_rejects_failed_factor_batch_in_strict_mode(self):
        sdk = mock.Mock()
        sdk.constant.Period.day.value = "day"
        with mock.patch.object(a, "_ensure_login", return_value=True), \
                mock.patch.object(a, "_ad", sdk), \
                mock.patch.object(a, "_market", mock.Mock()), \
                mock.patch.object(a, "sdk_call", return_value={"600000.SH": self.raw}), \
                mock.patch.object(a, "_get_factor_frame", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "adjustment factor unavailable for 600000.SH"):
            a.fetch_daily_batch(
                ["600000"], "20240101", "20240103",
                require_adjustment_factor=True,
            )

    def test_ffill_on_missing_trading_day(self):
        """K 线某日在因子表缺失时按前值 ffill，不产生 NaN 价。"""
        raw = _kline(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                     [10.0, 5.0, 5.5, 6.0])
        out = a._apply_adjust(raw, self.factor, "qfq")
        self.assertFalse(out["close"].isna().any())
        # 01-04 无因子 → ffill 用 2.0，base 仍是因子表最新(2.0)：6.0*2/2=6.0
        self.assertAlmostEqual(out["close"].iloc[-1], 6.0, places=6)

    def test_factor_series_extracts_column(self):
        """_factor_series 从宽表按券商代码取列并转 DatetimeIndex。"""
        wide = pd.DataFrame(
            {"600000.SH": [1.0, 2.0], "000001.SZ": [3.0, 4.0]},
            index=["2024-01-01", "2024-01-02"])
        s = a._factor_series(wide, "600000.SH")
        self.assertEqual(s.tolist(), [1.0, 2.0])
        self.assertIsInstance(s.index, pd.DatetimeIndex)
        self.assertIsNone(a._factor_series(wide, "999999.SH"))  # 缺列返回 None


if __name__ == "__main__":
    unittest.main(verbosity=2)

