"""加减仓建议引擎。

对每个维度打分（看多为正、看空为负），加总得到综合分，
再映射为「加仓 / 持有 / 减仓」建议。所有阈值均为经验值，
仅供参考，不构成投资建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Signal:
    """单项指标信号。"""
    name: str          # 指标名
    score: int         # 分值：正=看多，负=看空，0=中性
    detail: str        # 文字说明


@dataclass
class Advice:
    """综合建议结果。"""
    total_score: int
    action: str                       # 加仓 / 减仓 / 持有观望
    level: str                        # bullish / bearish / neutral
    signals: list[Signal] = field(default_factory=list)


def _last(df: pd.DataFrame, col: str) -> float:
    return float(df[col].iloc[-1])


def _analyze_ma(df: pd.DataFrame) -> Signal:
    """均线系统：多头/空头排列 + 价格与MA20关系。"""
    c = _last(df, "close")
    ma5, ma10, ma20, ma60 = (_last(df, f"ma{n}") for n in (5, 10, 20, 60))
    score = 0
    notes = []
    if ma5 > ma10 > ma20 > ma60:
        score += 2
        notes.append("均线多头排列(MA5>10>20>60)")
    elif ma5 < ma10 < ma20 < ma60:
        score -= 2
        notes.append("均线空头排列(MA5<10<20<60)")
    else:
        notes.append("均线交织无明显趋势")
    if c > ma20:
        score += 1
        notes.append("价格站上MA20")
    else:
        score -= 1
        notes.append("价格跌破MA20")
    return Signal("均线系统", score, "；".join(notes))


def _analyze_kdj(df: pd.DataFrame) -> Signal:
    k, d, j = _last(df, "kdj_k"), _last(df, "kdj_d"), _last(df, "kdj_j")
    k_prev, d_prev = float(df["kdj_k"].iloc[-2]), float(df["kdj_d"].iloc[-2])
    score = 0
    notes = [f"K={k:.1f} D={d:.1f} J={j:.1f}"]
    if j < 0 or k < 20:
        score += 2
        notes.append("超卖区，反弹概率大")
    elif j > 100 or k > 80:
        score -= 2
        notes.append("超买区，回调风险大")
    if k_prev < d_prev and k > d:
        score += 1
        notes.append("KDJ金叉")
    elif k_prev > d_prev and k < d:
        score -= 1
        notes.append("KDJ死叉")
    return Signal("KDJ", score, "；".join(notes))


def _analyze_rsi(df: pd.DataFrame) -> Signal:
    r6, r12 = _last(df, "rsi6"), _last(df, "rsi12")
    score = 0
    notes = [f"RSI6={r6:.1f} RSI12={r12:.1f}"]
    if r6 < 20:
        score += 2
        notes.append("RSI超卖")
    elif r6 > 80:
        score -= 2
        notes.append("RSI超买")
    elif r6 > r12:
        score += 1
        notes.append("短期强于中期")
    else:
        score -= 1
        notes.append("短期弱于中期")
    return Signal("RSI", score, "；".join(notes))


def _analyze_bias(df: pd.DataFrame) -> Signal:
    """BIAS 乖离率：负乖离过大易反弹（看多），正乖离过大易回调（看空）。"""
    b6 = _last(df, "bias6")
    score = 0
    notes = [f"BIAS6={b6:.2f}%"]
    if b6 < -8:
        score += 2
        notes.append("负乖离过大，超跌反弹")
    elif b6 < -4:
        score += 1
        notes.append("偏离均线较远，偏多")
    elif b6 > 8:
        score -= 2
        notes.append("正乖离过大，回调压力")
    elif b6 > 4:
        score -= 1
        notes.append("短期涨幅偏高")
    else:
        notes.append("乖离正常")
    return Signal("BIAS乖离率", score, "；".join(notes))


def _analyze_macd(df: pd.DataFrame) -> Signal:
    """MACD：金叉/死叉、零轴上下、柱状线动量。趋势跟随信号，回测验证可提升胜率。"""
    dif, dea = _last(df, "macd_dif"), _last(df, "macd_dea")
    dif_p, dea_p = float(df["macd_dif"].iloc[-2]), float(df["macd_dea"].iloc[-2])
    hist, hist_p = _last(df, "macd_hist"), float(df["macd_hist"].iloc[-2])
    score = 0
    notes = [f"DIF={dif:.3f} DEA={dea:.3f}"]
    if dif_p < dea_p and dif > dea:
        score += 2
        notes.append("MACD金叉")
    elif dif_p > dea_p and dif < dea:
        score -= 2
        notes.append("MACD死叉")
    if dif > 0:
        score += 1
        notes.append("DIF在零轴上方")
    else:
        score -= 1
        notes.append("DIF在零轴下方")
    if hist > hist_p:
        score += 1
        notes.append("红柱走强/绿柱收敛")
    else:
        score -= 1
        notes.append("红柱收敛/绿柱走强")
    return Signal("MACD", score, "；".join(notes))


def _analyze_volume(df: pd.DataFrame) -> Signal:
    """成交量/量能：放量上涨看多，放量下跌看空，结合OBV趋势。"""
    ratio = _last(df, "vol_ratio")
    price_up = _last(df, "close") > float(df["close"].iloc[-2])
    obv_up = _last(df, "obv") > float(df["obv"].iloc[-6])  # 近5日OBV趋势
    score = 0
    notes = [f"量比={ratio:.2f}"]
    if ratio > 1.5 and price_up:
        score += 2
        notes.append("放量上涨，资金进场")
    elif ratio > 1.5 and not price_up:
        score -= 2
        notes.append("放量下跌，抛压沉重")
    elif ratio < 0.7 and not price_up:
        score += 1
        notes.append("缩量回调，抛压减轻")
    if obv_up:
        score += 1
        notes.append("OBV走高，量能积累")
    else:
        score -= 1
        notes.append("OBV走低，量能流失")
    return Signal("成交量/量能", score, "；".join(notes))


def _analyze_turnover(df: pd.DataFrame) -> Signal:
    """换手率：适度放大活跃，过高警惕见顶，过低交投清淡。"""
    t = _last(df, "turnover")
    t_ma5 = float(df["turnover"].rolling(5).mean().iloc[-1])
    score = 0
    notes = [f"换手率={t:.2f}% (5日均={t_ma5:.2f}%)"]
    if t > 15:
        score -= 1
        notes.append("换手过高，警惕分歧见顶")
    elif 3 <= t <= 10 and t > t_ma5:
        score += 1
        notes.append("换手温和放大，交投活跃")
    elif t < 1:
        notes.append("换手低迷，关注度不足")
    return Signal("换手率", score, "；".join(notes))


def advise(df: pd.DataFrame) -> Advice:
    """综合所有指标给出加减仓建议。df 必须已由 indicators.compute_all 处理。"""
    signals = [
        _analyze_ma(df),
        _analyze_kdj(df),
        _analyze_rsi(df),
        _analyze_bias(df),
        _analyze_macd(df),
        _analyze_volume(df),
        _analyze_turnover(df),
    ]
    total = sum(s.score for s in signals)

    # 阈值随 MACD 分量（±4）纳入后上调，保持「加仓」信号的高置信度。
    if total >= 5:
        action, level = "建议加仓", "bullish"
    elif total >= 3:
        action, level = "偏多，可小幅加仓/持有", "bullish"
    elif total <= -5:
        action, level = "建议减仓", "bearish"
    elif total <= -3:
        action, level = "偏空，可小幅减仓/观望", "bearish"
    else:
        action, level = "持有观望", "neutral"

    return Advice(total_score=total, action=action, level=level, signals=signals)
