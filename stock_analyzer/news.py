"""新闻资讯分析：板块新闻总结 + 个股映射 + 个股动作消息 + 综合情绪。

数据源（均不依赖被屏蔽的东财行情子域名）：
- 个股新闻/动作消息：ak.stock_news_em(symbol)（东财搜索接口，含公告/龙虎榜/研报等）。
- 市场快讯（用于板块新闻分类）：ak.stock_info_global_sina（新浪快讯）+
  ak.stock_info_cjzc_em（财经早餐）。

情绪判断：优先用 Qwen 大模型（llm.score_news），无 key 或失败时降级到本地利好/利空词典。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import akshare as ak

from stock_analyzer import data, dcache, llm, stock_meta

# 利好/利空词典（词典兜底打分用）
_POS = ["涨停", "大涨", "创新高", "新高", "中标", "签约", "订单", "突破", "回购", "增持",
        "分红", "盈利", "扭亏", "利好", "超预期", "合作", "获批", "量产", "提价", "受益",
        "补贴", "政策支持", "增长", "预增", "业绩预喜", "龙头", "放量", "涨价", "扩产", "并购"]
_NEG = ["跌停", "大跌", "暴跌", "亏损", "下滑", "减持", "质押", "违规", "处罚", "退市",
        "立案", "预亏", "商誉", "风险警示", "问询", "诉讼", "爆雷", "减产", "裁员", "利空",
        "低于预期", "下调", "解禁", "停牌", "债务", "下跌", "承压", "警示", "被查"]

_A_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "商业航天": ("商业航天", "卫星", "火箭", "航天", "北斗", "遥感", "空间站"),
    "军工": ("军工", "国防", "航空装备", "军贸", "雷达", "导弹"),
    "半导体": ("半导体", "芯片", "集成电路", "封测", "晶圆", "光刻"),
    "人工智能": ("人工智能", "AI", "大模型", "算力", "机器人"),
    "算力": ("算力", "服务器", "数据中心", "云计算", "GPU"),
    "光通信": ("光模块", "光通信", "光纤", "光缆", "通信设备"),
    "低空经济": ("低空经济", "无人机", "eVTOL", "通航", "飞行器"),
    "机器人": ("机器人", "伺服", "减速器", "自动化"),
    "新能源车": ("新能源车", "新能源汽车", "汽车", "充电桩", "电驱"),
    "锂电池": ("锂电", "锂电池", "动力电池", "正极", "负极", "电解液"),
    "光伏": ("光伏", "太阳能", "硅片", "组件", "逆变器"),
    "储能": ("储能", "钠电", "钒电池"),
    "风电": ("风电", "海上风电", "风机"),
    "消费电子": ("消费电子", "手机", "折叠屏", "苹果", "华为", "显示"),
    "信创": ("信创", "国产软件", "操作系统", "数据库"),
    "数据要素": ("数据要素", "数据资产", "数据交易", "大数据"),
    "医药": ("医药", "创新药", "生物", "疫苗", "医疗器械", "中药"),
    "券商": ("券商", "证券", "资本市场", "并购重组"),
    "银行": ("银行", "息差", "存款", "贷款"),
    "白酒": ("白酒", "酒企", "茅台", "五粮液"),
    "化工新材料": ("新材料", "化工", "树脂", "高分子", "碳纤维"),
    "超硬材料": ("超硬材料", "金刚石", "培育钻石", "砂轮", "磨具", "磨料"),
}


def _map_to_a_topics(text: str) -> list[str]:
    hits: list[str] = []
    for topic, kws in _A_TOPIC_KEYWORDS.items():
        if any(kw and kw in text for kw in kws):
            hits.append(topic)
    return hits


def _split_topics(text: str) -> list[str]:
    out: list[str] = []
    for raw in str(text or "").replace("，", "、").replace(",", "、").replace(";", "、").split("、"):
        val = raw.strip()
        if val and val.lower() != "nan" and val not in out:
            out.append(val)
    return out


@dataclass
class NewsItem:
    time: str
    title: str
    summary: str = ""
    source: str = ""
    url: str = ""
    sectors: list = field(default_factory=list)
    sentiment: int = 0
    is_stock: bool = False   # 是否个股新闻


@dataclass
class SectorNews:
    name: str
    score: float
    tone: str
    count: int
    samples: list = field(default_factory=list)   # 代表标题


@dataclass
class NewsAnalysis:
    market_summary: str
    sector_news: list            # list[SectorNews]
    stock_items: list            # list[NewsItem]（个股动作消息）
    stock_summary: str
    matched_sectors: list        # 个股主营映射到的板块名
    matched_sector_news: list    # list[SectorNews]（A股行业/概念新闻）
    overall_score: float
    overall_level: str           # bullish/bearish/neutral
    overall_label: str
    conclusion: str
    engine: str                  # "Qwen大模型" 或 "本地词典"
    available: bool = True
    note: str = ""


# ------------------------- 词典打分 -------------------------
def _lexicon_score(text: str) -> int:
    if not text:
        return 0
    pos = sum(text.count(w) for w in _POS)
    neg = sum(text.count(w) for w in _NEG)
    raw = pos - neg
    return max(-2, min(2, raw))


def _tone(score: float) -> str:
    if score >= 0.6:
        return "偏多"
    if score <= -0.6:
        return "偏空"
    return "中性"


def _level(score: float):
    if score >= 0.6:
        return "bullish", "新闻面偏多"
    if score <= -0.6:
        return "bearish", "新闻面偏空"
    return "neutral", "新闻面中性"


# ------------------------- 数据拉取 -------------------------
@dcache.disk_cache(dcache.news_ttl, name="market_news")
def fetch_market_news(limit: int = 40) -> tuple:
    """市场级快讯（新浪快讯 + 财经早餐），返回 tuple[NewsItem]（可哈希以便缓存）。"""
    items: list[NewsItem] = []
    try:
        sina = ak.stock_info_global_sina()
        for _, r in sina.head(25).iterrows():
            content = str(r.get("内容", ""))
            items.append(NewsItem(time=str(r.get("时间", "")), title=content[:50],
                                  summary=content, source="新浪快讯"))
    except Exception:  # noqa: BLE001
        pass
    try:
        cj = ak.stock_info_cjzc_em()
        for _, r in cj.head(25).iterrows():
            items.append(NewsItem(time=str(r.get("发布时间", "")), title=str(r.get("标题", "")),
                                  summary=str(r.get("摘要", "")), source="财经早餐",
                                  url=str(r.get("链接", ""))))
    except Exception:  # noqa: BLE001
        pass
    return tuple(items[:limit])


@dcache.disk_cache(dcache.news_ttl, name="stock_news")
def fetch_stock_news(symbol: str, limit: int = 10) -> tuple:
    code = data._normalize_symbol(symbol)
    try:
        df = ak.stock_news_em(symbol=code)
    except Exception:  # noqa: BLE001
        return tuple()
    items = []
    for _, r in df.head(limit).iterrows():
        items.append(NewsItem(
            time=str(r.get("发布时间", "")), title=str(r.get("新闻标题", "")),
            summary=str(r.get("新闻内容", ""))[:120], source=str(r.get("文章来源", "")),
            url=str(r.get("新闻链接", "")), is_stock=True))
    return tuple(items)


# ------------------------- 情绪打分（LLM 或词典） -------------------------
def _score_items(items: list, key: str, model: str, context: str = "",
                 base_url: str = "") -> tuple:
    """给 items 逐条打分，返回 (engine, summary)。就地写入 item.sentiment。"""
    if not items:
        return "本地词典", ""
    titles = [it.title for it in items]
    if key:
        res = llm.score_news(titles, key=key, model=model, context=context, base_url=base_url)
        if res:
            for it, s in zip(items, res["scores"]):
                it.sentiment = s
            return "Qwen大模型", res.get("summary", "")
    # 词典兜底
    for it in items:
        it.sentiment = _lexicon_score(it.title + " " + it.summary)
    return "本地词典", ""


# ------------------------- 板块新闻 -------------------------
@lru_cache(maxsize=8)
def analyze_sector_news(key: str = "", model: str = llm.DEFAULT_MODEL,
                        base_url: str = "") -> tuple:
    """市场快讯按板块分类 + 情绪。返回 (list[SectorNews], market_summary, engine)。"""
    items = list(fetch_market_news())
    for it in items:
        it.sectors = _map_to_a_topics(it.title + " " + it.summary)
    engine, summary = _score_items(items, key, model, context="以下为A股市场财经快讯：",
                                   base_url=base_url)

    buckets: dict[str, list] = {}
    for it in items:
        for sec in it.sectors:
            buckets.setdefault(sec, []).append(it)

    sector_news = []
    for sec, its in buckets.items():
        avg = sum(i.sentiment for i in its) / len(its)
        sector_news.append(SectorNews(
            name=sec, score=round(avg, 2), tone=_tone(avg), count=len(its),
            samples=[i.title for i in its[:3]]))
    sector_news.sort(key=lambda s: s.score, reverse=True)

    if not summary:
        strong = [s.name for s in sector_news if s.score > 0][:4]
        weak = [s.name for s in sector_news if s.score < 0][:4]
        parts = []
        if strong:
            parts.append("新闻面偏暖：" + "、".join(strong))
        if weak:
            parts.append("新闻面偏冷：" + "、".join(weak))
        summary = "；".join(parts) or "市场新闻情绪总体平稳。"
    return tuple(sector_news), summary, engine


# ------------------------- 综合分析 -------------------------
def analyze(symbol: str, key: str = "", model: str = llm.DEFAULT_MODEL,
            base_url: str = "") -> NewsAnalysis:
    key = llm.get_key(key)
    model = llm.get_model(model)
    base_url = llm.get_base_url(base_url)
    sector_news, market_summary, engine = analyze_sector_news(key, model, base_url)
    sector_map = {s.name: s for s in sector_news}

    # 个股动作消息
    stock_items = list(fetch_stock_news(symbol))
    _, stock_summary = _score_items(stock_items, key, model,
                                    context="以下为该股相关新闻/公告：", base_url=base_url)
    stock_score = (sum(i.sentiment for i in stock_items) / len(stock_items)
                   if stock_items else 0.0)

    # 个股 -> A股行业/概念 映射
    try:
        meta = stock_meta.get_stock_meta(symbol)
    except Exception:  # noqa: BLE001
        meta = {}
    matched = []
    for val in [meta.get("a_industry", ""), *_split_topics(meta.get("a_concepts", ""))]:
        if val and val not in matched:
            matched.append(val)
    if not matched:
        text = " ".join(str(meta.get(k, "")) for k in ("business", "product_type", "product_name"))
        matched = _map_to_a_topics(text)
    matched_sector_news = [sector_map[m] for m in matched if m in sector_map]

    # 综合：个股动作消息权重更高，其次为A股行业/概念新闻
    parts, weights = [], []
    if stock_items:
        parts.append(stock_score); weights.append(2.0)
    if matched_sector_news:
        sec_avg = sum(s.score for s in matched_sector_news) / len(matched_sector_news)
        parts.append(sec_avg); weights.append(1.0)
    overall = sum(p * w for p, w in zip(parts, weights)) / sum(weights) if parts else 0.0
    overall = round(overall, 2)
    level, label = _level(overall)

    # 结论
    sec_txt = ("、".join(f"{s.name}({s.tone})" for s in matched_sector_news)
               if matched_sector_news else "无明确对应板块")
    stock_txt = (f"个股动作消息情绪 {stock_score:+.2f}" if stock_items else "暂无个股新闻")
    conclusion = (f"{stock_txt}；A股行业/概念新闻：{sec_txt}。综合新闻面 <b>{label}</b>"
                  f"（评分 {overall:+.2f}）。")

    return NewsAnalysis(
        market_summary=market_summary, sector_news=sector_news,
        stock_items=stock_items, stock_summary=stock_summary,
        matched_sectors=matched, matched_sector_news=matched_sector_news,
        overall_score=overall, overall_level=level, overall_label=label,
        conclusion=conclusion, engine=engine,
    )
