"""多维信号「每日快照落盘」，为将来回测**完整预估**（含新闻/外围/基本面）打基础。

原理：新闻/外围/基本面等信号没有历史逐日数据，无法事后还原。因此从现在起，
每次分析时把当日各维度评分 + 综合研判落盘保存；日积月累后，即可用这些历史快照
join 后续真实涨跌，回测完整预估的胜率。

存储：每只股票一个 CSV（snapshots/{code}.csv），按日期去重。
容器中建议把 SNAPSHOT_DIR（默认 ./snapshots）挂载到宿主机卷以持久化。
"""
from __future__ import annotations

import os

import pandas as pd

from stock_analyzer import data

_FIELDS = ["date", "close", "tech", "overseas", "sector", "news", "moneyflow", "fund",
           "news_count", "news_pos_count", "news_neg_count", "sector_matched_count",
           "hot_keywords", "qwen_event_tags", "quant_score", "quant_rank_pct", "quant_model",
           "sentiment_score", "sentiment_model", "sentiment_count",
           "pred_composite", "pred_level", "engine"]


def _dir() -> str:
    d = os.environ.get("SNAPSHOT_DIR", "snapshots")
    os.makedirs(d, exist_ok=True)
    return d


def _path(symbol: str) -> str:
    return os.path.join(_dir(), f"{data._normalize_symbol(symbol)}.csv")


def history(symbol: str) -> pd.DataFrame:
    p = _path(symbol)
    if not os.path.exists(p):
        return pd.DataFrame(columns=_FIELDS)
    try:
        return pd.read_csv(p, dtype={"date": str})
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_FIELDS)


def count(symbol: str) -> int:
    return len(history(symbol))


def save(symbol: str, record: dict) -> int:
    """按日期去重保存一条快照，返回累计天数。record 至少含 date。"""
    df = history(symbol)
    row = {k: record.get(k) for k in _FIELDS}
    date = str(row.get("date") or "")
    if not date:
        return len(df)
    df = df[df["date"].astype(str) != date]  # 覆盖同日
    df = df.reindex(columns=_FIELDS).reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame([row], columns=_FIELDS)
    else:
        df.loc[len(df)] = row
    df = df.reindex(columns=_FIELDS).sort_values("date").reset_index(drop=True)
    try:
        df.to_csv(_path(symbol), index=False)
    except Exception:  # noqa: BLE001
        pass
    return len(df)


def backtest(symbol: str, horizons=(1, 3, 5), threshold: float = 0.4) -> dict:
    """用已积累的历史快照回测「完整预估」综合方向的胜率。

    仅统计快照日期在价格序列中、且其后有 N 个交易日的样本；数据不足时结果很少。
    """
    hist = history(symbol)
    if len(hist) == 0:
        return {"n_days": 0, "n_signals": 0, "winrate": {}, "note": "尚无历史快照"}
    px = data.fetch_daily(symbol, days=400).sort_values("date").reset_index(drop=True)
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(px["date"])}
    max_h = max(horizons)
    trades = []
    for _, r in hist.iterrows():
        d = str(r["date"])
        if d not in idx:
            continue
        i = idx[d]
        if i + max_h >= len(px):
            continue  # 前瞻不足
        try:
            comp = float(r.get("pred_composite"))
        except Exception:  # noqa: BLE001
            continue
        if abs(comp) < threshold:
            continue
        direction = 1 if comp > 0 else -1
        rec = {"date": d, "comp": comp}
        for h in horizons:
            fwd = px["close"].iloc[i + h] / px["close"].iloc[i] - 1
            rec[f"win{h}"] = bool((fwd > 0) == (direction > 0))
        trades.append(rec)
    winrate = {}
    for h in horizons:
        wins = [t[f"win{h}"] for t in trades]
        winrate[h] = round(sum(wins) / len(wins) * 100, 1) if wins else None
    return {"n_days": len(hist), "n_signals": len(trades), "winrate": winrate,
            "note": "" if trades else "已积累快照，但尚无满足前瞻天数的样本（需再等几天）"}
