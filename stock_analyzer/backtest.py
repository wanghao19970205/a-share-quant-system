"""回测：验证「加减仓/技术面信号」的历史胜率。

方法：在最近一段区间内，对每个交易日用**截至当日**的数据计算加减仓综合评分，
当 |评分| ≥ 阈值时视为一次方向性信号，检查其后 N 个交易日的实际涨跌方向是否一致，
统计胜率（1天 / 3天）。

说明：均线/KDJ/RSI/BIAS 等指标均为因果（只依赖当日及之前），在全量数据上一次算好、
按日切片取最后一行即为「时点值」，无未来函数。

注意：「次日涨跌预估」还叠加了新闻/外围/基本面等**当前快照**类信号，缺乏历史逐日快照，
无法忠实回测，故这里回测其技术面主干（加减仓信号）——它也是预估的核心。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from stock_analyzer import advisor, data, indicators, prediction


def run(symbol: str, lookback: int = 65, horizons=(1, 3, 5), threshold: int = 3,
        engine: str = "rule", trend_filter: bool = True,
        key: str = "", model: str = "", base_url: str = "") -> dict:
    """回测单只股票。

    Args:
        lookback: 回看的交易日数（约 65 ≈ 近3个月）。
        horizons: 持有天数列表，检验信号后 N 日方向。
        threshold: |综合评分| ≥ 该值才算一次方向性信号（含 MACD 后经回测取 3 最优）。
        engine: 'rule'=加减仓规则信号；'qwen'=用大模型读技术快照打分作为信号。
        trend_filter: 仅采纳与 MA20 趋势方向一致的信号（回测显示可提升胜率）。
    """
    raw = data.fetch_daily(symbol, days=300)
    df = indicators.compute_all(raw)
    n = len(df)
    max_h = max(horizons)
    # 起点：保证指标有效(≥60) 且给评估区间与前瞻留足空间
    start = max(60, n - lookback - max_h)
    if start >= n - max_h:
        return {"symbol": data._normalize_symbol(symbol), "n_signals": 0,
                "winrate": {}, "trades": [], "note": "历史数据不足"}

    trades = []
    for i in range(start, n - max_h):
        sub = df.iloc[:i + 1]
        if engine == "qwen":
            sc = prediction.qwen_tech_score(sub, key=key, model=model, base_url=base_url)
        else:
            sc = advisor.advise(sub).total_score
        if abs(sc) < threshold:
            continue
        direction = 1 if sc > 0 else -1
        # 趋势过滤：只做与 MA20 方向一致的信号（顺势），逆势信号胜率偏低予以剔除
        if trend_filter:
            up = float(sub["close"].iloc[-1]) > float(sub["ma20"].iloc[-1])
            if (direction > 0) != up:
                continue
        rec = {"date": df["date"].iloc[i].strftime("%Y-%m-%d"), "score": sc,
               "dir": "看多" if direction > 0 else "看空"}
        for h in horizons:
            fwd = df["close"].iloc[i + h] / df["close"].iloc[i] - 1
            rec[f"ret{h}"] = round(fwd * 100, 2)
            rec[f"win{h}"] = bool((fwd > 0) == (direction > 0))
        trades.append(rec)

    winrate = {}
    for h in horizons:
        wins = [t[f"win{h}"] for t in trades]
        winrate[h] = round(sum(wins) / len(wins) * 100, 1) if wins else None

    return {
        "symbol": data._normalize_symbol(symbol),
        "name": data.get_stock_name(symbol),
        "n_signals": len(trades),
        "winrate": winrate,       # {1: 胜率%, 3: 胜率%}
        "trades": trades,
        "note": "",
    }


def run_many(codes: list, lookback: int = 65, horizons=(1, 3, 5), threshold: int = 3,
             engine: str = "rule", trend_filter: bool = True,
             key: str = "", model: str = "", base_url: str = "") -> list:
    """批量回测。多线程并发拉取行情（主要耗时是网络 I/O），保持输入顺序返回。

    Qwen 引擎逐日调用大模型，并发度适当降低以规避限流。
    """
    seen, uniq = set(), []
    for c in codes:
        c = data._normalize_symbol(c)
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return []

    def _one(c):
        try:
            return run(c, lookback=lookback, horizons=horizons, threshold=threshold,
                       engine=engine, trend_filter=trend_filter,
                       key=key, model=model, base_url=base_url)
        except Exception as e:  # noqa: BLE001
            return {"symbol": c, "n_signals": 0, "winrate": {},
                    "trades": [], "note": type(e).__name__}

    workers = min(4 if engine == "qwen" else 8, len(uniq))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, uniq))
