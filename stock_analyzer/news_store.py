"""白名单新闻库：把个股公告/研报/新闻/市场要闻结构化落盘，供盘中新闻联动与事件回测。

阶段一目标：从现在起把白名单相关新闻按「文章发布时间」结构化存储，日积月累建库；
可回溯的历史（公告/研报带发布日期）一次性回填最近一年。

数据源（akshare，均免费）：
- 个股公告：ak.stock_zh_a_disclosure_report_cninfo（巨潮，支持日期区间，可回溯历史）。
- 个股研报：ak.stock_research_report_em（东财，含发布日期，历史较深）。
- 个股新闻：ak.stock_news_em（东财搜索，仅近端，尽力而为）。
- 市场要闻/快讯：ak.stock_info_global_sina + ak.stock_info_cjzc_em（仅近端，往后每日积累）。
  （财联社电报 stock_info_global_cls 当前 akshare 版本返回 404，暂不可用。）

存储：每只股票一份 parquet（NEWS_DIR/{code}.parquet），市场级要闻存 NEWS_DIR/_market.parquet；
按 (category, url) 去重（无 url 时退化为 category+title+publish_time）。发布时间即文章时间。
情绪：回填阶段用本地利好/利空词典打分（免费、可复现）；LLM 情绪可后续增量补。
"""
from __future__ import annotations

import os

import pandas as pd

from stock_analyzer import data, news

# 结构化字段（发布时间即文章时间）。
NEWS_FIELDS = [
    "code", "publish_time", "date", "category", "source",
    "title", "summary", "url", "sectors", "event_tags", "sentiment",
    # 离线 Qwen 结构化标注。保留词典 sentiment 作为可复现的回退分，不被覆盖。
    "llm_impact", "llm_relevance", "llm_horizon", "llm_event_types",
    "llm_novelty", "llm_certainty", "llm_reason", "llm_model",
    "llm_annotated_at", "llm_content_hash",
]

# 新闻类别：公告 / 研报 / 个股新闻 / 市场要闻快讯。
CATEGORIES = ("announcement", "research", "stock_news", "flash")

# 事件标签词典（与 snapshot_batch._EVENT_KEYWORDS 对齐；此处自带一份以保持模块轻量、无副作用导入）。
_EVENT_KEYWORDS = {
    "业绩": ["预增", "预减", "预盈", "预亏", "扭亏", "业绩", "利润", "营收"],
    "订单": ["中标", "合同", "订单", "签约", "采购"],
    "产能": ["扩产", "投产", "量产", "产能", "项目建设"],
    "并购": ["并购", "收购", "重组", "注入", "股权转让"],
    "融资": ["定增", "融资", "发债", "可转债", "募资"],
    "回购增持": ["回购", "增持", "员工持股"],
    "减持": ["减持", "清仓", "套现"],
    "监管风险": ["处罚", "立案", "问询", "调查", "违规", "诉讼"],
    "政策热点": ["政策", "补贴", "规划", "试点", "支持"],
    "价格变化": ["涨价", "提价", "降价", "价格", "供需"],
}


def _dir() -> str:
    # 新闻库与快照共享持久卷；显式 NEWS_DIR 仍可覆盖（训练/测试时使用）。
    d = os.environ.get("NEWS_DIR", os.path.join(os.environ.get("SNAPSHOT_DIR", "snapshots"), "news_data"))
    os.makedirs(d, exist_ok=True)
    return d


def _path(code: str) -> str:
    name = "_market" if not code else data._normalize_symbol(code)
    return os.path.join(_dir(), f"{name}.parquet")


def _event_tags(text: str) -> list:
    tags = []
    for tag, words in _EVENT_KEYWORDS.items():
        if any(w in text for w in words):
            tags.append(tag)
    return tags


def enrich(item: dict) -> dict:
    """把原始抓取项补全为规整记录：派生 date、sectors、event_tags、sentiment。"""
    title = str(item.get("title") or "")
    summary = str(item.get("summary") or "")
    text = (title + " " + summary).strip()
    pt = str(item.get("publish_time") or "").strip()
    ts = pd.to_datetime(pt, errors="coerce")
    date = ts.strftime("%Y-%m-%d") if pd.notna(ts) else pt[:10]

    sectors = item.get("sectors")
    if not sectors:
        sectors = news._map_to_a_topics(text)
    if isinstance(sectors, (list, tuple)):
        sectors = "|".join(str(s) for s in sectors if str(s))
    else:
        sectors = str(sectors or "")

    sent = item.get("sentiment")
    if sent is None:
        sent = news._lexicon_score(text)

    return {
        "code": str(item.get("code") or ""),
        "publish_time": pt,
        "date": date,
        "category": str(item.get("category") or ""),
        "source": str(item.get("source") or ""),
        "title": title,
        "summary": summary,
        "url": str(item.get("url") or ""),
        "sectors": sectors,
        "event_tags": "|".join(_event_tags(text)),
        "sentiment": int(sent),
        "llm_impact": item.get("llm_impact"),
        "llm_relevance": item.get("llm_relevance"),
        "llm_horizon": str(item.get("llm_horizon") or ""),
        "llm_event_types": str(item.get("llm_event_types") or ""),
        "llm_novelty": item.get("llm_novelty"),
        "llm_certainty": item.get("llm_certainty"),
        "llm_reason": str(item.get("llm_reason") or ""),
        "llm_model": str(item.get("llm_model") or ""),
        "llm_annotated_at": str(item.get("llm_annotated_at") or ""),
        "llm_content_hash": str(item.get("llm_content_hash") or ""),
    }


def read_store(code: str) -> pd.DataFrame:
    p = _path(code)
    if not os.path.exists(p):
        return pd.DataFrame(columns=NEWS_FIELDS)
    try:
        df = pd.read_parquet(p)
        return df.reindex(columns=NEWS_FIELDS)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=NEWS_FIELDS)


def _dedup_key(df: pd.DataFrame) -> pd.Series:
    url = df["url"].fillna("").astype(str)
    fallback = (df["category"].astype(str) + "|" + df["title"].astype(str)
                + "|" + df["publish_time"].astype(str))
    return df["category"].astype(str) + "||" + url.where(url != "", fallback)


def save_store(code: str, df: pd.DataFrame) -> None:
    """原样写回某只股票新闻库，供离线标注补充字段时使用。"""
    out = df.copy().reindex(columns=NEWS_FIELDS)
    out = out.sort_values("publish_time").reset_index(drop=True)
    out.to_parquet(_path(code), index=False)


def save_items(code: str, items: list) -> tuple:
    """去重合并落盘，返回 (新增条数, 落盘后总条数)。已存在的记录保留原值（幂等）。"""
    old = read_store(code)
    if not items:
        return 0, len(old)
    new = pd.DataFrame([enrich(it) for it in items]).reindex(columns=NEWS_FIELDS)
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new
    combined = combined.reindex(columns=NEWS_FIELDS)
    combined = combined[~_dedup_key(combined).duplicated(keep="first")].copy()
    combined = combined.sort_values("publish_time").reset_index(drop=True)
    added = max(len(combined) - len(old), 0)
    try:
        combined.to_parquet(_path(code), index=False)
    except Exception:  # noqa: BLE001
        pass
    return added, len(combined)


def stats() -> dict:
    """汇总当前新闻库：文件数、总条数、各类别条数、时间跨度。"""
    d = _dir()
    total = 0
    by_cat: dict = {}
    codes = 0
    tmin = tmax = ""
    for fn in os.listdir(d):
        if not fn.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(d, fn), columns=["category", "date"])
        except Exception:  # noqa: BLE001
            continue
        if fn != "_market.parquet":
            codes += 1
        total += len(df)
        for cat, n in df["category"].value_counts().items():
            by_cat[cat] = by_cat.get(cat, 0) + int(n)
        dates = df["date"].astype(str)
        dates = dates[dates.str.len() >= 8]
        if len(dates):
            lo, hi = dates.min(), dates.max()
            tmin = lo if not tmin else min(tmin, lo)
            tmax = hi if not tmax else max(tmax, hi)
    return {"codes": codes, "total": total, "by_category": by_cat,
            "date_min": tmin, "date_max": tmax}
