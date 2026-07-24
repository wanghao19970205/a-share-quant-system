"""复权换算单测：验证 amazingdata_source._apply_adjust 的 qfq/hfq 数学。

不依赖网络/券商 SDK：直接把构造的因子序列传给 _apply_adjust，
校验前复权(归一化到最新日)、后复权、不复权三种口径的价格换算是否正确。
容器内跑：docker exec a-scheduler-1 python3 -m unittest stock_analyzer.test_amazingdata_adjust
"""
from __future__ import annotations

import unittest

import pandas as pd

from stock_analyzer import amazingdata_source as a


def _kline(dates, closes):
    """构造标准化后的最小 K 线（open=high=low=close，便于核对）。"""
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100] * len(closes),
    })


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

