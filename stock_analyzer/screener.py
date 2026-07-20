"""多股票批量打分选股。

候选池两种来源：
- A. 用户粘贴的自选股代码列表
- B. 某行业板块的成分股（akshare 东财，走 net 代理以应对屏蔽）

打分依据：加减仓综合评分（技术面，快）。按分值排序，正分靠前。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import akshare as ak

from stock_analyzer import advisor, data, indicators, net, quant_signal, stock_meta


@lru_cache(maxsize=512)
def score_one(code: str, profile: str | None = None) -> dict:
    """对单只股票打分。失败返回 available=False。"""
    try:
        df = indicators.compute_all(data.fetch_daily(code, days=250))
        adv = advisor.advise(df)
        latest = df.iloc[-1]
        q = quant_signal.score_for_screener(code, profile=profile)
        meta = stock_meta.get_stock_meta(code)
        quant_component = ((q.get("quant_effective_rank_pct", q.get("quant_rank_pct", 0.5)) - 0.5) * 6.0
                           if q.get("quant_available") else 0.0)
        blended_score = round(float(adv.total_score) + quant_component, 2)
        return {
            "code": data._normalize_symbol(code),
            "name": meta.get("name") or data.get_stock_name(code),
            "industry": meta.get("industry", ""),
            "sector": meta.get("sector", ""),
            "business": meta.get("business", ""),
            "score": adv.total_score,
            "blended_score": blended_score,
            "action": adv.action,
            "close": round(float(latest["close"]), 2),
            "pct": round(float(latest["pct_change"]), 2),
            "available": True,
            **q,
        }
    except Exception as e:  # noqa: BLE001
        return {"code": data._normalize_symbol(code), "name": "", "available": False,
                "error": type(e).__name__}


def score_many(codes: list, profile: str | None = None) -> list:
    """批量打分，按综合评分降序。codes 为 6 位或带前缀代码列表。

    多线程并发拉取（行情获取是网络 I/O，为主要耗时），显著缩短批量时间；
    单只结果 lru 缓存，重复查询近乎瞬时。
    """
    seen, uniq = set(), []
    for c in codes:
        c = data._normalize_symbol(c)
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(uniq))) as ex:
        results = list(ex.map(lambda c: score_one(c, profile), uniq))
    ok = [r for r in results if r.get("available")]
    ok.sort(key=lambda r: r.get("blended_score", r["score"]), reverse=True)
    bad = [r for r in results if not r.get("available")]
    return ok + bad


@lru_cache(maxsize=1)
def industry_list() -> list:
    """东财行业板块名称列表（用于 B 选择）。"""
    try:
        with net.akshare_proxied():
            df = ak.stock_board_industry_name_em()
        col = "板块名称" if "板块名称" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()
    except Exception:  # noqa: BLE001
        return []


@lru_cache(maxsize=32)
def industry_constituents(name: str, limit: int = 50) -> list:
    """某行业板块的成分股代码（最多 limit 只，避免过慢）。"""
    try:
        with net.akshare_proxied():
            df = ak.stock_board_industry_cons_em(symbol=name)
        col = "代码" if "代码" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()[:limit]
    except Exception:  # noqa: BLE001
        return []


@lru_cache(maxsize=32)
def concept_constituents(name: str, limit: int = 50) -> list:
    """某概念板块的成分股代码（最多 limit 只）。"""
    try:
        with net.akshare_proxied():
            df = ak.stock_board_concept_cons_em(symbol=name)
        col = "代码" if "代码" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()[:limit]
    except Exception:  # noqa: BLE001
        return []
