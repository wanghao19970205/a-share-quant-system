"""次日涨跌预估：汇总技术面/外围/板块/新闻/基本面·资金面多维信号，
组织成 prompt 交给 Qwen 综合研判；无 Qwen key 时用规则法加权兜底。

仅做概率性倾向研判，不构成投资建议。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from stock_analyzer import llm


@dataclass
class Prediction:
    direction: str          # 偏多 / 偏空 / 震荡
    level: str              # bullish / bearish / neutral
    confidence: str         # 高 / 中 / 低
    composite: float        # 规则法综合分（-2~2），供参考
    logic: str
    risks: str
    action: str
    engine: str             # Qwen大模型 / 本地规则
    summary: str = ""       # 喂给模型的多维信号摘要（可展示）


def _clip(x, lo=-2.0, hi=2.0):
    return max(lo, min(hi, x))


def _to_float(value):
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return None
    return v if math.isfinite(v) else None


def _fmt_num(value, digits: int = 2, signed: bool = False, suffix: str = "") -> str:
    v = _to_float(value)
    if v is None:
        return "-"
    text = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return f"{text}{suffix}"


def _fmt_pct(value, digits: int = 2, already_pct: bool = False, signed: bool = True) -> str:
    v = _to_float(value)
    if v is None:
        return "-"
    if not already_pct:
        v *= 100
    text = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return f"{text}%"


def _fmt_large(value) -> str:
    v = _to_float(value)
    if v is None:
        return "-"
    av = abs(v)
    if av >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if av >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:.0f}"


def _row_value(row, key: str):
    try:
        return row.get(key)
    except Exception:  # noqa: BLE001
        return None


def _last_row(tech_df):
    if tech_df is None:
        return None
    try:
        if tech_df.empty:
            return None
        return tech_df.iloc[-1]
    except Exception:  # noqa: BLE001
        return None


def _return_line(tech_df) -> str:
    if tech_df is None:
        return ""
    parts = []
    try:
        close_s = tech_df["close"]
        latest = _to_float(close_s.iloc[-1])
        for days in (1, 3, 5, 10, 20):
            if latest is None or len(close_s) <= days:
                continue
            prev = _to_float(close_s.iloc[-days - 1])
            if prev:
                parts.append(f"近{days}日{(latest / prev - 1) * 100:+.2f}%")
    except Exception:  # noqa: BLE001
        return ""
    return "【近几日收益】" + "；".join(parts) if parts else ""


def _ma_line(row) -> str:
    close = _to_float(_row_value(row, "close"))
    mas = {n: _to_float(_row_value(row, f"ma{n}")) for n in (5, 10, 20, 60)}
    valid = [mas[n] for n in (5, 10, 20, 60)]
    if close is None or any(v is None for v in valid):
        return ""
    if mas[5] > mas[10] > mas[20] > mas[60]:
        state = "多头排列"
    elif mas[5] < mas[10] < mas[20] < mas[60]:
        state = "空头排列"
    else:
        state = "均线交织"
    dist = "，".join(f"距MA{n}{(close / mas[n] - 1) * 100:+.2f}%" for n in (5, 10, 20, 60) if mas[n])
    ma_vals = " / ".join(f"MA{n}={mas[n]:.2f}" for n in (5, 10, 20, 60))
    return f"【均线系统】{state}；{ma_vals}；{dist}。"


def _tech_detail_lines(tech_df) -> list[str]:
    row = _last_row(tech_df)
    if row is None:
        return []
    date = _row_value(row, "date")
    date_text = f"{date.strftime('%Y-%m-%d')}，" if hasattr(date, "strftime") else (f"{date}，" if date else "")
    lines = [
        f"【行情】{date_text}开{_fmt_num(_row_value(row, 'open'))} / 高{_fmt_num(_row_value(row, 'high'))} / "
        f"低{_fmt_num(_row_value(row, 'low'))} / 收{_fmt_num(_row_value(row, 'close'))}，"
        f"涨跌{_fmt_pct(_row_value(row, 'pct_change'), already_pct=True)}，"
        f"振幅{_fmt_pct(_row_value(row, 'amplitude'), already_pct=True, signed=False)}。"
    ]
    ret = _return_line(tech_df)
    if ret:
        lines.append(ret)
    ma = _ma_line(row)
    if ma:
        lines.append(ma)
    lines.extend([
        f"【MACD】DIF={_fmt_num(_row_value(row, 'macd_dif'), 3)}，DEA={_fmt_num(_row_value(row, 'macd_dea'), 3)}，"
        f"柱={_fmt_num(_row_value(row, 'macd_hist'), 3, signed=True)}。",
        f"【KDJ】K={_fmt_num(_row_value(row, 'kdj_k'), 1)}，D={_fmt_num(_row_value(row, 'kdj_d'), 1)}，"
        f"J={_fmt_num(_row_value(row, 'kdj_j'), 1)}。",
        f"【RSI/BIAS】RSI6={_fmt_num(_row_value(row, 'rsi6'), 1)}，RSI12={_fmt_num(_row_value(row, 'rsi12'), 1)}，"
        f"RSI24={_fmt_num(_row_value(row, 'rsi24'), 1)}；BIAS6={_fmt_pct(_row_value(row, 'bias6'), already_pct=True)}，"
        f"BIAS12={_fmt_pct(_row_value(row, 'bias12'), already_pct=True)}，BIAS24={_fmt_pct(_row_value(row, 'bias24'), already_pct=True)}。",
        f"【量能/换手】成交量{_fmt_large(_row_value(row, 'volume'))}，成交额{_fmt_large(_row_value(row, 'amount'))}，"
        f"5日均量{_fmt_large(_row_value(row, 'vol_ma5'))}，10日均量{_fmt_large(_row_value(row, 'vol_ma10'))}，"
        f"量比{_fmt_num(_row_value(row, 'vol_ratio'), 2)}，换手{_fmt_pct(_row_value(row, 'turnover'), already_pct=True, signed=False)}。",
    ])
    return lines


def _advice_signal_text(advice) -> str:
    sigs = []
    for s in getattr(advice, "signals", []) or []:
        detail = getattr(s, "detail", "") or ""
        sigs.append(f"{s.name}{s.score:+d}" + (f"（{detail}）" if detail else ""))
    return "；".join(sigs)


def _quant_line(quant) -> str:
    if quant is None or not getattr(quant, "available", True):
        return ""
    rank = f"全A排名 {quant.rank}/{quant.universe_size}，全A分位 {_fmt_pct(quant.rank_pct, digits=1, signed=False)}"
    watch_rank = getattr(quant, "watch_rank", None)
    watch_universe = getattr(quant, "watch_universe_size", None)
    if watch_rank is not None and watch_universe:
        watch = f"白名单排名 {watch_rank}/{watch_universe}，白名单分位 {_fmt_pct(getattr(quant, 'watch_rank_pct', None), digits=1, signed=False)}"
    else:
        watch = "白名单排名暂无（可能不在白名单或白名单未命中）"
    horizon = getattr(quant, "expected_return_horizon", None)
    est_label = f"{horizon}日" if horizon else "模型"
    risk_parts = [
        f"参考价={_fmt_num(getattr(quant, 'entry_price', None))}",
        f"止损={_fmt_num(getattr(quant, 'stop_loss', None))}",
        f"止盈1={_fmt_num(getattr(quant, 'take_profit_1', None))}",
        f"止盈2={_fmt_num(getattr(quant, 'take_profit_2', None))}",
        f"盈亏比1/2={_fmt_num(getattr(quant, 'risk_reward_1', None), 2)}/{_fmt_num(getattr(quant, 'risk_reward_2', None), 2)}",
        f"ATR14={_fmt_num(getattr(quant, 'atr_14', None), 3)}（{_fmt_pct(getattr(quant, 'atr_pct', None), digits=2, signed=False)}）",
    ]
    note = getattr(quant, "risk_note", "") or ""
    return (
        f"【量化选股模型】模型口径：{quant.model}；预测日 {quant.date}；方向 {getattr(quant, 'direction', '中性')}；"
        f"量化分 {quant.score:+.4f}；{rank}；{watch}；"
        f"{est_label}预估收益 {_fmt_pct(getattr(quant, 'expected_return', None))}；"
        f"止损止盈：{'，'.join(risk_parts)}。" + (f"风控口径：{note}" if note else "")
    )


def build_summary(symbol, name, close, pct, advice=None, sent=None,
                  link=None, nws=None, fund=None, mf=None, quant=None, tech_df=None,
                  sentiment=None) -> str:
    """把各模块结论汇总成可读文本（也用于喂给大模型）。"""
    lines = [f"标的：{name}（{symbol}），最新价 {close:.2f}，当日涨跌 {pct:+.2f}%"]
    lines.extend(_tech_detail_lines(tech_df))
    lines.append("")

    if advice is not None:
        sig = _advice_signal_text(advice)
        lines.append(f"【技术信号明细】结论：{advice.action}（评分 {advice.total_score:+d}）。分项：{sig}")

    if sent is not None:
        lines.append(f"【外围·美股】{sent.label}（评分 {sent.weighted_score:+.2f}）。{sent.summary}")

    if link is not None and getattr(link, "matched", None):
        concl = (link.link_conclusion or "").replace("<b>", "").replace("</b>", "")
        lines.append(f"【外围板块联动】{concl}")

    if nws is not None:
        lines.append(f"【新闻情绪】{nws.overall_label}（评分 {nws.overall_score:+.2f}）。"
                     f"市场：{nws.market_summary}")
        if getattr(nws, "stock_summary", ""):
            lines.append(f"　个股消息：{nws.stock_summary}")

    if sentiment is not None and getattr(sentiment, "available", False):
        state = "已纳入最终分" if getattr(sentiment, "enabled", False) else "验证未过，仅展示"
        lines.append(
            f"【历史舆情模型】{sentiment.model}；{sentiment.lookback_days}日衰减分 {sentiment.score:+.3f}；"
            f"文章 {sentiment.article_count} 条（正{sentiment.positive_count}/负{sentiment.negative_count}）；{state}。")

    if mf is not None and getattr(mf, "available", False):
        ms = "；".join(f"{s.name}{s.score:+d}（{s.detail}）" for s in mf.signals)
        lines.append(f"【资金流向】{mf.label}（评分 {mf.score:+.2f}）。{ms}")

    if fund is not None and getattr(fund, "available", False) and getattr(fund, "signals", None):
        fs = "；".join(f"{s.name}{s.score:+d}（{s.detail}）" for s in fund.signals)
        lines.append(f"【基本面·资金面】{fund.label}（评分 {fund.score:+.2f}）。{fs}")
        if getattr(fund, "revenue_text", ""):
            lines.append(f"　财务：{fund.revenue_text}")

    qline = _quant_line(quant)
    if qline:
        lines.append(qline)

    return "\n".join(lines)


def _rule_based(advice, sent, link, nws, fund, mf=None, quant=None, sentiment=None):
    """无大模型时的加权兜底。返回 (composite, direction, level, confidence)。"""
    parts, weights = [], []
    if advice is not None:
        parts.append(_clip(advice.total_score / 3.0)); weights.append(0.4)
    if sent is not None:
        parts.append(_clip(sent.weighted_score)); weights.append(0.2)
    if nws is not None:
        parts.append(_clip(nws.overall_score)); weights.append(0.2)
    if mf is not None and getattr(mf, "available", False):
        parts.append(_clip(mf.score)); weights.append(0.2)
    if fund is not None and getattr(fund, "available", False) and getattr(fund, "signals", None):
        parts.append(_clip(fund.score)); weights.append(0.2)
    if quant is not None and getattr(quant, "available", True):
        parts.append(_clip((quant.rank_pct - 0.5) * 4.0)); weights.append(0.25)
    if (sentiment is not None and getattr(sentiment, "available", False)
            and getattr(sentiment, "enabled", False) and getattr(sentiment, "blend_weight", 0) > 0):
        parts.append(_clip(sentiment.score)); weights.append(float(sentiment.blend_weight))
    comp = round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 2) if parts else 0.0

    if comp >= 0.4:
        direction, level = "偏多", "bullish"
    elif comp <= -0.4:
        direction, level = "偏空", "bearish"
    else:
        direction, level = "震荡", "neutral"
    conf = "高" if abs(comp) >= 1.0 else ("中" if abs(comp) >= 0.4 else "低")
    return comp, direction, level, conf


_SYSTEM = (
    "你是资深A股短线交易研判助手。基于用户给出的数值证据（行情、近几日收益、均线、MACD、KDJ、"
    "RSI、BIAS、量能、换手、技术信号明细、外围美股、外围板块联动、新闻情绪、资金流向、"
    "基本面/资金面、量化排名、白名单排名、预估收益、止损止盈、模型口径），研判该股<b>下一交易日</b>的涨跌倾向。"
    "回答时要优先解释最关键的2-4个证据，若技术面与量化模型矛盾，要明确哪个证据权重更高以及原因。"
    "必须以【行情】中的明确日期为准：不得把历史收盘价、参考价或量化止损价称为今日/当日涨停价或跌停价；"
    "只有该日期涨幅达到对应股票涨停阈值且收盘/最新价等于最高价时，才可称为涨停；"
    "只有跌幅达到对应跌停阈值且收盘/最新价等于最低价时，才可称为跌停。"
    "严格只输出JSON，格式："
    '{"direction":"偏多/偏空/震荡","confidence":"高/中/低","logic":"核心逻辑2-3句",'
    '"risks":"主要风险1-2句","action":"操作/仓位建议1句"}。'
    "注意：这是概率性研判，不构成投资建议。"
)


def predict(symbol, name, close, pct, advice=None, sent=None, link=None,
            nws=None, fund=None, mf=None, quant=None, tech_df=None, sentiment=None,
            key="", model="", base_url="") -> Prediction:
    summary = build_summary(symbol, name, close, pct, advice, sent, link, nws, fund, mf, quant, tech_df, sentiment)
    comp, r_dir, r_level, r_conf = _rule_based(advice, sent, link, nws, fund, mf, quant, sentiment)

    key = llm.get_key(key)
    if key:
        data = llm.chat_json(_SYSTEM, summary, key=key,
                             model=llm.get_model(model), base_url=llm.get_base_url(base_url))
        if data and data.get("direction"):
            d = str(data["direction"])
            level = "bullish" if "多" in d else ("bearish" if "空" in d else "neutral")
            return Prediction(
                direction=d, level=level,
                confidence=str(data.get("confidence", r_conf)),
                composite=comp, logic=str(data.get("logic", "")),
                risks=str(data.get("risks", "")), action=str(data.get("action", "")),
                engine="Qwen大模型", summary=summary)

    # 规则法兜底
    return Prediction(
        direction=r_dir, level=r_level, confidence=r_conf, composite=comp,
        logic="综合技术面/外围/新闻/基本面加权得到的倾向（未启用大模型）。",
        risks="规则法未考虑消息面细节与市场情绪突变，仅供参考。",
        action="偏多可小幅加仓、偏空减仓、震荡观望，结合自身风险控制。",
        engine="本地规则", summary=summary)


# ------------------------- 纯技术面 Qwen 打分（可回测） -------------------------
# 只用「截至当日」的技术指标（因果、可复现），故可放进回测；不含新闻/外围（无历史逐日快照）。
def technical_snapshot(sub) -> str:
    """由 df 切片（截至当日）生成紧凑的技术面快照文本。"""
    r = sub.iloc[-1]
    c = float(r["close"])
    ma = {n: float(r.get(f"ma{n}", float("nan"))) for n in (5, 10, 20, 60)}
    if ma[5] > ma[10] > ma[20] > ma[60]:
        ma_state = "多头排列"
    elif ma[5] < ma[10] < ma[20] < ma[60]:
        ma_state = "空头排列"
    else:
        ma_state = "交织"
    ret5 = ""
    if len(sub) >= 6:
        ret5 = f"{(c / float(sub['close'].iloc[-6]) - 1) * 100:+.1f}%"
    return (
        f"收盘{c:.2f}，当日{float(r.get('pct_change', 0)):+.2f}%，近5日{ret5}；"
        f"均线{ma_state}，价{'上' if c > ma[20] else '下'}穿MA20；"
        f"MACD DIF={float(r.get('macd_dif', 0)):.3f}/DEA={float(r.get('macd_dea', 0)):.3f}/柱={float(r.get('macd_hist', 0)):+.3f}；"
        f"KDJ K={float(r.get('kdj_k', 0)):.0f}/D={float(r.get('kdj_d', 0)):.0f}/J={float(r.get('kdj_j', 0)):.0f}；"
        f"RSI6={float(r.get('rsi6', 0)):.0f}/RSI12={float(r.get('rsi12', 0)):.0f}；"
        f"BIAS6={float(r.get('bias6', 0)):+.1f}%；"
        f"量比={float(r.get('vol_ratio', 0)):.2f}，换手={float(r.get('turnover', 0)):.2f}%"
    )


_TECH_SYS = (
    "你是A股短线交易员。根据给定的『截至当日』技术指标，预测该股<b>下一交易日</b>涨或跌。"
    "综合均线形态、KDJ/RSI超买超卖、BIAS乖离、量能。严格只输出JSON："
    '{"dir":"涨/跌/平","score":整数}，score 取 -2~2，正数看涨、负数看跌、0中性。'
)


@lru_cache(maxsize=4096)
def _qwen_tech_cached(snapshot: str, key: str, model: str, base_url: str) -> int:
    data = llm.chat_json(_TECH_SYS, snapshot, key=key, model=model, base_url=base_url)
    if not data:
        return 0
    try:
        s = int(data.get("score", 0))
    except Exception:  # noqa: BLE001
        s = 0
    return max(-2, min(2, s))


def qwen_tech_score(sub, key: str = "", model: str = "", base_url: str = "") -> int:
    """用 Qwen 对某时点技术快照给出 -2~2 的涨跌打分（缓存，无 key 返回 0）。"""
    key = llm.get_key(key)
    if not key:
        return 0
    return _qwen_tech_cached(technical_snapshot(sub), key,
                             llm.get_model(model), llm.get_base_url(base_url))
