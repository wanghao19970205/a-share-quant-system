"""个股资金流向分析。

默认使用本地已加载行情做量价资金代理，避免等待东财主力资金流接口。
该结果反映成交额、换手率、涨跌和短线量价结构，不等同于真实主力净流入。

只做资金面倾向研判，不构成投资建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import akshare as ak
import pandas as pd

from stock_analyzer import data, dcache, net

_COL_MAIN = "主力净流入-净额"
_COL_MAIN_R = "主力净流入-净占比"
_COL_SUPER = "超大单净流入-净额"


@dataclass
class MoneyFlowSignal:
    name: str
    score: int
    detail: str


@dataclass
class MoneyFlowAnalysis:
    available: bool
    score: float                 # -2~2，正=资金流入偏多
    level: str                   # bullish / bearish / neutral
    label: str
    signals: list = field(default_factory=list)
    net_main_1d: float = 0.0     # 当日主力净流入（元）
    net_main_5d: float = 0.0     # 近5日主力累计净流入（元）
    ratio_main_1d: float = 0.0   # 当日主力净占比（%）
    pos_days_5: int = 0          # 近5日主力净流入为正的天数
    recent: object = None        # 近N日明细 DataFrame（展示用）
    note: str = ""
    is_proxy: bool = False       # True=本地量价代理（非真实主力资金流）


def _market(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


@dcache.disk_cache(dcache.kline_ttl, name="moneyflow")
def _fetch(code: str) -> pd.DataFrame:
    """东财个股历史资金流（走代理路由；被墙且无代理时抛异常，由上层降级）。"""
    with net.akshare_proxied():
        return ak.stock_individual_fund_flow(stock=code, market=_market(code))


def _unavailable(note: str) -> MoneyFlowAnalysis:
    return MoneyFlowAnalysis(False, 0.0, "neutral", "资金流向不可用", note=note)


def price_volume_history(price_df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """按 PC 端口径生成量价资金代理：涨日成交额为正，跌日为负。"""
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date", "net_amount"])
    px = price_df.sort_values("date").tail(max(int(days), 20)).copy()
    for column in ("close", "volume", "amount"):
        if column in px.columns:
            px[column] = pd.to_numeric(px[column], errors="coerce")
    if "close" not in px.columns or "volume" not in px.columns:
        return pd.DataFrame(columns=["date", "net_amount"])
    amount = px["amount"] if "amount" in px.columns else px["close"] * px["volume"]
    if amount.isna().all() or float(amount.tail(5).fillna(0).sum()) <= 0:
        amount = px["close"] * px["volume"]
    direction = px["close"].pct_change().fillna(0.0).apply(
        lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)
    )
    result = px[["date"]].copy()
    result["net_amount"] = amount * direction
    return result.dropna(subset=["date", "net_amount"]).tail(max(int(days), 0))


def _price_volume_analysis(symbol: str, note: str = "", price_df: pd.DataFrame | None = None) -> MoneyFlowAnalysis:
    """用本地行情做量价资金代理。"""
    if price_df is not None:
        px = price_df.copy()
    else:
        try:
            px = data.fetch_daily(symbol, days=80).copy()
        except Exception as e:  # noqa: BLE001
            return _unavailable(f"{note or '本地行情量价代理'}不可用（{type(e).__name__}）")
    if px is None or px.empty or len(px) < 10:
        return _unavailable(f"{note or '本地行情量价代理'}样本不足")
    px = px.sort_values("date").tail(20).copy()
    for c in ("close", "volume", "amount", "turnover"):
        if c in px.columns:
            px[c] = pd.to_numeric(px[c], errors="coerce")
    amount = px["amount"] if "amount" in px.columns else (px["close"] * px["volume"])
    if amount.isna().all() or float(amount.tail(5).sum()) <= 0:
        amount = px["close"] * px["volume"]
    ret1 = float(px["close"].iloc[-1] / px["close"].iloc[-2] - 1) if len(px) >= 2 else 0.0
    ret5 = float(px["close"].iloc[-1] / px["close"].iloc[-6] - 1) if len(px) >= 6 else 0.0
    avg5 = float(amount.tail(5).mean())
    avg20 = float(amount.tail(20).mean()) if len(amount) >= 20 else float(amount.mean())
    vol_ratio = avg5 / avg20 if avg20 else 1.0
    turnover1 = float(px["turnover"].iloc[-1]) if "turnover" in px.columns and pd.notna(px["turnover"].iloc[-1]) else 0.0
    pos_days = int((px["close"].pct_change().tail(5) > 0).sum())

    signals: list[MoneyFlowSignal] = []
    score = 0
    if ret5 > 0 and vol_ratio >= 1.15:
        score += 1
        signals.append(MoneyFlowSignal("放量上涨", +1, f"近5日涨幅 {ret5*100:+.2f}%，成交额为20日均值 {vol_ratio:.2f}倍"))
    elif ret5 < 0 and vol_ratio >= 1.15:
        score -= 1
        signals.append(MoneyFlowSignal("放量下跌", -1, f"近5日跌幅 {ret5*100:+.2f}%，成交额为20日均值 {vol_ratio:.2f}倍"))
    else:
        signals.append(MoneyFlowSignal("量价结构", 0, f"近5日涨跌 {ret5*100:+.2f}%，成交额为20日均值 {vol_ratio:.2f}倍"))

    if ret1 > 0 and turnover1 >= 3:
        score += 1
        signals.append(MoneyFlowSignal("当日承接活跃", +1, f"当日涨跌 {ret1*100:+.2f}%，换手率 {turnover1:.2f}%"))
    elif ret1 < 0 and turnover1 >= 3:
        score -= 1
        signals.append(MoneyFlowSignal("当日抛压活跃", -1, f"当日涨跌 {ret1*100:+.2f}%，换手率 {turnover1:.2f}%"))
    else:
        signals.append(MoneyFlowSignal("当日量价", 0, f"当日涨跌 {ret1*100:+.2f}%，换手率 {turnover1:.2f}%"))

    if pos_days >= 4:
        score += 1
        signals.append(MoneyFlowSignal("短线趋势延续", +1, f"近5日 {pos_days}/5 天上涨"))
    elif pos_days <= 1:
        score -= 1
        signals.append(MoneyFlowSignal("短线趋势偏弱", -1, f"近5日仅 {pos_days}/5 天上涨"))

    score = max(-2, min(2, score))
    if score >= 1:
        level, label = "bullish", "资金面偏多（量价代理）"
    elif score <= -1:
        level, label = "bearish", "资金面偏空（量价代理）"
    else:
        level, label = "neutral", "资金面中性（量价代理）"

    # 量价资金代理：按当日涨跌方向对成交额赋号，作为资金净流入/流出的近似，
    # 使其可正可负（流出为负），而非仅展示无向的成交额。
    proxy_history = price_volume_history(px, days=20)
    signed_amount = pd.Series(proxy_history["net_amount"].to_numpy(), index=px.index)
    net1 = float(signed_amount.iloc[-1]) if len(signed_amount) else 0.0
    net5 = float(signed_amount.tail(5).sum())
    ratio1_signed = turnover1 * (1.0 if ret1 > 0 else (-1.0 if ret1 < 0 else 0.0))

    recent = px.tail(5).copy()
    recent["净额代理"] = signed_amount.tail(5).to_numpy()
    return MoneyFlowAnalysis(
        available=True, score=float(score), level=level, label=label, signals=signals,
        net_main_1d=net1,
        net_main_5d=net5,
        ratio_main_1d=ratio1_signed,
        pos_days_5=pos_days,
        recent=recent,
        is_proxy=True,
        note=(note or "本栏用本地行情量价代理：按当日涨跌方向对成交额赋号近似资金净流入/流出，不请求东财主力资金流接口")
             + "；不等同于真实主力净流入。",
    )


def analyze(symbol: str, price_df: pd.DataFrame | None = None, prefer_remote: bool = False) -> MoneyFlowAnalysis:
    """分析个股资金流向，默认直接使用本地量价代理（-2~2）。"""
    if not prefer_remote:
        return _price_volume_analysis(symbol, price_df=price_df)

    code = data._normalize_symbol(symbol)
    try:
        df = _fetch(code)
    except Exception as e:  # noqa: BLE001
        return _price_volume_analysis(symbol, f"东财资金流接口不可用（{type(e).__name__}）", price_df=price_df)
    if df is None or df.empty or _COL_MAIN not in df.columns:
        return _price_volume_analysis(symbol, "未获取到东财主力资金流数据", price_df=price_df)

    df = df.copy()
    for c in (_COL_MAIN, _COL_MAIN_R, _COL_SUPER):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[_COL_MAIN])
    if df.empty:
        return _unavailable("资金流数据为空。")

    recent = df.tail(5)
    net1 = float(df[_COL_MAIN].iloc[-1])
    net5 = float(recent[_COL_MAIN].sum())
    ratio1 = float(df[_COL_MAIN_R].iloc[-1]) if _COL_MAIN_R in df.columns else 0.0
    pos_days = int((recent[_COL_MAIN] > 0).sum())

    signals: list[MoneyFlowSignal] = []
    score = 0

    # 近5日主力累计净流入方向
    if net5 > 0:
        score += 1
        signals.append(MoneyFlowSignal("主力5日净流入", +1, f"近5日主力累计净流入 {net5 / 1e8:+.2f}亿元"))
    else:
        score -= 1
        signals.append(MoneyFlowSignal("主力5日净流出", -1, f"近5日主力累计净流出 {net5 / 1e8:+.2f}亿元"))

    # 当日主力净占比（相对成交额，去规模化）
    if ratio1 >= 5:
        score += 1
        signals.append(MoneyFlowSignal("当日主力大幅流入", +1, f"当日主力净占比 {ratio1:+.2f}%"))
    elif ratio1 <= -5:
        score -= 1
        signals.append(MoneyFlowSignal("当日主力大幅流出", -1, f"当日主力净占比 {ratio1:+.2f}%"))
    else:
        signals.append(MoneyFlowSignal("当日主力资金", 0, f"当日主力净占比 {ratio1:+.2f}%（{net1 / 1e8:+.2f}亿）"))

    # 近5日流入连续性
    if pos_days >= 4:
        score += 1
        signals.append(MoneyFlowSignal("资金持续流入", +1, f"近5日 {pos_days}/5 天主力净流入，趋势向上"))
    elif pos_days <= 1:
        score -= 1
        signals.append(MoneyFlowSignal("资金持续流出", -1, f"近5日仅 {pos_days}/5 天主力净流入，抛压为主"))

    score = max(-2, min(2, score))
    if score >= 1:
        level, label = "bullish", "资金面偏多"
    elif score <= -1:
        level, label = "bearish", "资金面偏空"
    else:
        level, label = "neutral", "资金面中性"

    return MoneyFlowAnalysis(
        available=True, score=float(score), level=level, label=label, signals=signals,
        net_main_1d=net1, net_main_5d=net5, ratio_main_1d=ratio1, pos_days_5=pos_days,
        recent=recent,
    )
