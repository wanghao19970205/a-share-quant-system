"""技术指标计算：均线系统、KDJ、RSI、BIAS、成交量/量能、换手率。

所有函数接收 ``fetch_daily`` 返回的标准化 DataFrame（含 open/high/low/close/volume/
amount/turnover 列），返回带新增指标列的副本，不修改入参。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 均线周期
MA_WINDOWS = (5, 10, 20, 60)


def add_ma(df: pd.DataFrame, windows: tuple[int, ...] = MA_WINDOWS) -> pd.DataFrame:
    """均线系统：对收盘价计算多条简单移动平均线，列名如 ma5、ma10。"""
    df = df.copy()
    for n in windows:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
    return df


def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ 随机指标。

    RSV = (C - Ln) / (Hn - Ln) * 100
    K = EMA(RSV, 1/m1)，D = EMA(K, 1/m2)，J = 3K - 2D。
    """
    df = df.copy()
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = (df["close"] - low_n) / rng * 100
    df["kdj_k"] = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / m2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def add_rsi(df: pd.DataFrame, windows: tuple[int, ...] = (6, 12, 24)) -> pd.DataFrame:
    """RSI 相对强弱指标，采用 Wilder 平滑（ewm alpha=1/n）。"""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for n in windows:
        avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi{n}"] = 100 - 100 / (1 + rs)
        df[f"rsi{n}"] = df[f"rsi{n}"].fillna(100)  # 全涨无回撤时 RSI=100
    return df


def add_bias(df: pd.DataFrame, windows: tuple[int, ...] = (6, 12, 24)) -> pd.DataFrame:
    """BIAS 乖离率：(收盘价 - N日均价) / N日均价 * 100。"""
    df = df.copy()
    for n in windows:
        ma = df["close"].rolling(n).mean()
        df[f"bias{n}"] = (df["close"] - ma) / ma * 100
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标。

    DIF = EMA(fast) - EMA(slow)，DEA = EMA(DIF, signal)，MACD 柱 = 2*(DIF - DEA)。
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_dif"] = ema_fast - ema_slow
    df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2
    return df


def add_cci(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """CCI 顺势指标：(TP - MA(TP)) / (0.015 * 平均绝对偏差)，TP=(H+L+C)/3。"""
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f"cci{n}"] = (tp - ma_tp) / (0.015 * mad.replace(0, np.nan))
    return df


def add_wr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """威廉指标 %R：(Hn - C) / (Hn - Ln) * -100，取值 -100~0，越低越超卖。"""
    df = df.copy()
    high_n = df["high"].rolling(n).max()
    low_n = df["low"].rolling(n).min()
    rng = (high_n - low_n).replace(0, np.nan)
    df[f"wr{n}"] = (high_n - df["close"]) / rng * -100
    return df


def add_roc(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """ROC 变动率：(C - C_n前) / C_n前 * 100，衡量动量。"""
    df = df.copy()
    df[f"roc{n}"] = df["close"].pct_change(n) * 100
    return df


def add_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """ATR 真实波幅（归一化为占收盘价百分比 atr_pct），衡量波动率。"""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / n, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100
    return df


def add_volume_energy(df: pd.DataFrame) -> pd.DataFrame:
    """成交量与量能分析。

    - vol_ma5 / vol_ma10：成交量均线，判断放量/缩量。
    - vol_ratio：当日成交量 / 近5日均量，>1 放量，<1 缩量。
    - obv：能量潮，累计资金流向，反映量能强弱。
    """
    df = df.copy()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma10"] = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]

    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算全部指标，返回带所有指标列的 DataFrame。"""
    df = add_ma(df)
    df = add_kdj(df)
    df = add_rsi(df)
    df = add_bias(df)
    df = add_macd(df)
    df = add_cci(df)
    df = add_wr(df)
    df = add_roc(df)
    df = add_atr(df)
    df = add_volume_energy(df)
    return df
