"""美股外围板块拆解 + 个股主营业务映射（仅美股，速度快、数据稳）。

思路：
1. 把美股按行业板块拆开：用行业 ETF 作为板块情绪代理 + 该板块龙头个股。
2. 取所选 A 股的主营业务（同花顺），映射到对应美股板块，给出关联结论。
   映射优先用 Qwen 大模型（能识别概念/上下游，如金刚石→半导体散热），否则关键词兜底。

美股代表个股选择标准：取该板块对应行业 ETF 的**前权重成分股**（业务纯度高的龙头），每板块 2 家。

数据源：
- 美股 ETF / 个股：新浪 stock_us_daily（国内直连稳定，全部并发拉取）。
- 个股主营业务：同花顺 stock_zyjs_ths。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import data, dcache, llm, overseas

# 板块 -> 美股行业 ETF（该板块「美股」情绪代理）
SECTOR_ETFS: dict[str, str] = {
    "半导体": "SMH", "科技": "XLK", "通信服务": "XLC", "金融": "XLF",
    "能源": "XLE", "医药健康": "XLV", "可选消费": "XLY", "必需消费": "XLP",
    "工业": "XLI", "原材料": "XLB", "公用事业": "XLU", "房地产": "XLRE",
    "太阳能光伏": "TAN",
}

# 板块 -> 美股代表龙头 [(代码, 中文名, 主营业务)]
# 取对应行业 ETF 的前权重龙头（业务纯度高、最能代表并驱动板块），每板块 2 家。
SECTOR_STOCKS: dict[str, list[tuple[str, str, str]]] = {
    "半导体": [("NVDA", "英伟达", "GPU与AI加速芯片"), ("TSM", "台积电", "晶圆代工")],
    "科技": [("MSFT", "微软", "软件与云计算"), ("AAPL", "苹果", "消费电子与服务")],
    "通信服务": [("META", "Meta", "社交媒体"), ("GOOGL", "谷歌", "搜索与广告")],
    "金融": [("JPM", "摩根大通", "综合银行"), ("V", "Visa", "支付网络")],
    "能源": [("XOM", "埃克森美孚", "石油天然气"), ("CVX", "雪佛龙", "石油天然气")],
    "医药健康": [("LLY", "礼来", "创新药(减肥/糖尿病)"), ("UNH", "联合健康", "医疗保险与服务")],
    "可选消费": [("AMZN", "亚马逊", "电商与云"), ("TSLA", "特斯拉", "电动车")],
    "必需消费": [("COST", "好市多", "仓储会员超市"), ("WMT", "沃尔玛", "商超零售")],
    "工业": [("GE", "通用电气", "航空发动机"), ("CAT", "卡特彼勒", "工程机械")],
    "原材料": [("LIN", "林德", "工业气体"), ("FCX", "自由港", "铜矿")],
    "公用事业": [("NEE", "新纪元能源", "电力与新能源"), ("SO", "南方电力", "电力")],
    "房地产": [("PLD", "安博", "物流地产"), ("AMT", "美国电塔", "通信基础设施")],
    "太阳能光伏": [("FSLR", "第一太阳能", "光伏组件"), ("ENPH", "Enphase", "光伏微逆变器")],
}

# 主营业务关键词 -> 对应外围板块
KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "半导体": ("半导体", "科技"), "芯片": ("半导体", "科技"), "集成电路": ("半导体", "科技"),
    "晶圆": ("半导体",), "封测": ("半导体",), "光刻": ("半导体",),
    "面板": ("半导体", "科技"), "显示": ("半导体", "科技"), "液晶": ("半导体", "科技"),
    "基板玻璃": ("半导体", "科技"), "光电": ("科技",), "消费电子": ("科技",),
    "电子": ("科技", "半导体"), "软件": ("科技",), "计算机": ("科技",), "人工智能": ("科技",),
    "云计算": ("科技",), "数据中心": ("科技",),
    "通信": ("通信服务",), "运营商": ("通信服务",), "传媒": ("通信服务",),
    "游戏": ("通信服务",), "互联网": ("通信服务", "科技"), "广告": ("通信服务",),
    "银行": ("金融",), "证券": ("金融",), "保险": ("金融",), "券商": ("金融",),
    "金融": ("金融",), "支付": ("金融",),
    "石油": ("能源",), "天然气": ("能源",), "煤": ("能源",), "油气": ("能源",), "炼化": ("能源",),
    "光伏": ("太阳能光伏", "科技"), "太阳能": ("太阳能光伏",), "锂电": ("太阳能光伏", "原材料"),
    "新能源": ("太阳能光伏",), "储能": ("太阳能光伏",), "风电": ("太阳能光伏",),
    "动力电池": ("太阳能光伏", "原材料"),
    "医药": ("医药健康",), "生物": ("医药健康",), "医疗": ("医药健康",),
    "疫苗": ("医药健康",), "创新药": ("医药健康",), "医院": ("医药健康",), "器械": ("医药健康",),
    "白酒": ("必需消费",), "食品": ("必需消费",), "饮料": ("必需消费",),
    "乳": ("必需消费",), "农": ("必需消费",), "养殖": ("必需消费",), "调味": ("必需消费",),
    "家电": ("可选消费",), "汽车": ("可选消费",), "服装": ("可选消费",),
    "零售": ("可选消费",), "旅游": ("可选消费",), "餐饮": ("可选消费",), "家居": ("可选消费",),
    "机械": ("工业",), "工程": ("工业",), "航空": ("工业",), "军工": ("工业",),
    "国防": ("工业",), "机器人": ("工业", "科技"), "装备": ("工业",), "电力设备": ("工业",),
    "钢": ("原材料",), "有色": ("原材料",), "化工": ("原材料",), "材料": ("原材料",),
    "水泥": ("原材料",), "铜": ("原材料",), "铝": ("原材料",), "稀土": ("原材料",),
    "金刚石": ("半导体", "原材料"), "超硬": ("原材料", "半导体"), "散热": ("半导体", "科技"),
    "磨料": ("原材料",), "磨具": ("原材料",), "石墨": ("原材料",), "碳材料": ("原材料",),
    "3d打印": ("工业",), "增材制造": ("工业",),
    "电力": ("公用事业",), "水务": ("公用事业",), "燃气": ("公用事业",), "环保": ("公用事业",),
    "地产": ("房地产",), "物业": ("房地产",),
}


# ------------------------- 数据结构 -------------------------
@dataclass
class StockQuote:
    symbol: str
    name: str
    business: str
    region: str
    available: bool = False
    pct: float = 0.0


@dataclass
class RegionSentiment:
    region: str
    available: bool
    pct: float = 0.0
    score: int = 0


@dataclass
class SectorDetail:
    name: str
    regions: dict
    overall_score: float
    trend: str
    stocks: list = field(default_factory=list)

    @property
    def available(self) -> bool:
        return any(r.available for r in self.regions.values())

    @property
    def us_pct(self) -> float:
        r = self.regions.get("美股")
        return r.pct if r and r.available else 0.0

    @property
    def us_available(self) -> bool:
        r = self.regions.get("美股")
        return bool(r and r.available)

    def us_stocks(self) -> list:
        return [q for q in self.stocks if q.region == "美股"]


@dataclass
class SectorAnalysis:
    business: dict
    all_sectors: list
    sector_summary: str
    matched: dict
    linked: list = field(default_factory=list)
    link_score: float = 0.0
    link_level: str = "neutral"
    link_conclusion: str = ""
    biz_available: bool = True
    biz_note: str = ""
    map_engine: str = "关键词"      # 关键词 / AI
    map_reason: str = ""           # LLM 映射理由


# ------------------------- 行情拉取（并发） -------------------------
def _latest_pct(close):
    if close is None:
        return None
    close = close.dropna().astype(float)
    if len(close) < 2:
        return None
    return (close.iloc[-1] / close.iloc[-2] - 1) * 100


@lru_cache(maxsize=256)
def _fetch_close(symbol: str):
    """统一取收盘价序列：日/韩走 Yahoo，其余(美股ETF/个股)走新浪。"""
    if symbol.endswith(".T") or symbol.endswith(".KS"):
        return overseas._fetch_yahoo(symbol, rng="1mo", retries=2)
    df = ak.stock_us_daily(symbol=symbol)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["close"]


def _safe_close(symbol: str):
    try:
        return _fetch_close(symbol)
    except Exception:  # noqa: BLE001
        return None


def _prefetch(symbols: set) -> dict:
    """并发拉取所有 symbol 的收盘价序列。"""
    out: dict = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, close in zip(symbols, ex.map(_safe_close, symbols)):
            out[sym] = close
    return out


def _signal(pct: float) -> int:
    if pct >= 1.0:
        return 2
    if pct > 0.1:
        return 1
    if pct <= -1.0:
        return -2
    if pct < -0.1:
        return -1
    return 0


# ------------------------- 板块分析 -------------------------
@dcache.disk_cache(dcache.market_ttl, name="sectors")
def analyze_sectors():
    """并发拉取全部美股板块（ETF + 龙头个股）情绪。返回 (list[SectorDetail], summary)。"""
    # 汇总所有待拉取代码（仅美股，全部走新浪，速度快）
    symbols: set = set(SECTOR_ETFS.values())
    for items in SECTOR_STOCKS.values():
        for sym, _, _ in items:
            symbols.add(sym)
    closes = _prefetch(symbols)

    details: list[SectorDetail] = []
    for name, etf in SECTOR_ETFS.items():
        regions: dict[str, RegionSentiment] = {}
        stocks: list[StockQuote] = []

        # 美股 ETF 作为板块情绪代理
        us_trend = "-"
        etf_close = closes.get(etf)
        pct = _latest_pct(etf_close)
        if pct is not None:
            regions["美股"] = RegionSentiment("美股", True, pct, _signal(pct))
            us_trend = overseas._trend(etf_close.dropna().astype(float))
        else:
            regions["美股"] = RegionSentiment("美股", False)

        # 美股代表龙头个股
        for sym, cn, biz in SECTOR_STOCKS.get(name, []):
            q = StockQuote(sym, cn, biz, "美股")
            p = _latest_pct(closes.get(sym))
            if p is not None:
                q.pct, q.available = p, True
            stocks.append(q)

        overall = float(regions["美股"].score) if regions["美股"].available else 0.0
        details.append(SectorDetail(name, regions, round(overall, 2), us_trend, stocks))

    avail_sec = [d for d in details if d.available]
    if not avail_sec:
        return details, "外围板块数据不可用。"

    strong = sorted([d for d in avail_sec if d.us_pct > 0.3],
                    key=lambda d: d.us_pct, reverse=True)
    weak = sorted([d for d in avail_sec if d.us_pct < -0.3], key=lambda d: d.us_pct)
    parts = []
    if strong:
        parts.append("走强：" + "、".join(f"{d.name}({d.us_pct:+.2f}%)" for d in strong[:5]))
    if weak:
        parts.append("走弱：" + "、".join(f"{d.name}({d.us_pct:+.2f}%)" for d in weak[:5]))
    avg = sum(d.us_pct for d in avail_sec) / len(avail_sec)
    tone = "整体偏多" if avg >= 0.3 else ("整体偏空" if avg <= -0.3 else "涨跌分化")
    summary = f"美股板块{tone}。" + "；".join(parts)
    return details, summary


# ------------------------- 个股主营 -------------------------
@lru_cache(maxsize=64)
def get_main_business(symbol: str) -> dict:
    code = data._normalize_symbol(symbol)
    df = ak.stock_zyjs_ths(symbol=code)
    if df is None or df.empty:
        return {}
    row = df.iloc[0]
    return {
        "主营业务": str(row.get("主营业务", "") or ""),
        "产品类型": str(row.get("产品类型", "") or ""),
        "产品名称": str(row.get("产品名称", "") or ""),
        "经营范围": str(row.get("经营范围", "") or ""),
    }


def map_to_sectors(text: str) -> dict:
    hits: dict[str, list[str]] = {}
    for kw, secs in KEYWORD_MAP.items():
        if kw in text:
            for sec in secs:
                hits.setdefault(sec, [])
                if kw not in hits[sec]:
                    hits[sec].append(kw)
    return hits


# ------------------------- 关联分析 -------------------------
def analyze_linkage(symbol: str, key: str = "", model: str = "",
                    base_url: str = "") -> SectorAnalysis:
    details, summary = analyze_sectors()
    sector_map = {d.name: d for d in details}

    try:
        biz = get_main_business(symbol)
        biz_ok, biz_note = bool(biz), ""
    except Exception as e:  # noqa: BLE001
        biz, biz_ok, biz_note = {}, False, f"主营业务获取失败：{type(e).__name__}"

    result = SectorAnalysis(business=biz, all_sectors=details, sector_summary=summary,
                            matched={}, biz_available=biz_ok, biz_note=biz_note)
    if not biz:
        result.link_conclusion = "未获取到主营业务，无法建立外围板块关联。"
        return result

    # 优先用 Qwen 智能映射（能识别概念/上下游，如金刚石→半导体散热）；否则关键词兜底
    matched: dict = {}
    key = llm.get_key(key)
    if key:
        res = llm.map_sectors(
            biz.get("主营业务", ""),
            biz.get("产品类型", "") or biz.get("产品名称", ""),
            list(SECTOR_ETFS.keys()), key=key,
            model=llm.get_model(model), base_url=llm.get_base_url(base_url))
        if res and res.get("sectors"):
            matched = {s: ["AI"] for s in res["sectors"]}
            result.map_engine = "AI"
            result.map_reason = res.get("reason", "")
    if not matched:
        # 关键词兜底：只匹配主营+产品（不含经营范围，避免宽泛业务误命中）
        text = " ".join([biz.get("主营业务", ""), biz.get("产品类型", ""),
                         biz.get("产品名称", "")])
        matched = map_to_sectors(text)
    result.matched = matched
    if not matched:
        result.link_conclusion = "主营业务未匹配到明确的外围板块，建议以个股技术面为主。"
        return result

    linked = [sector_map[name] for name in matched
              if name in sector_map and sector_map[name].available]
    result.linked = linked
    if not linked:
        result.link_conclusion = "已匹配到板块，但对应外围板块数据暂不可用。"
        return result

    score = sum(d.us_pct for d in linked) / len(linked)
    result.link_score = round(score, 2)
    if score >= 0.6:
        result.link_level, eff = "bullish", "偏利好"
    elif score <= -0.6:
        result.link_level, eff = "bearish", "偏利空"
    else:
        result.link_level, eff = "neutral", "影响中性"

    detail = "、".join(f"{d.name}(美股{d.us_pct:+.2f}%)" for d in linked)
    biz_short = biz.get("主营业务") or biz.get("产品类型") or "主营"
    reason_txt = f"（{result.map_reason}）" if result.map_reason else ""
    result.link_conclusion = (
        f"个股主营【{biz_short}】，对应外围板块：{detail}{reason_txt}。"
        f"综合隔夜美股对该股 <b>{eff}</b>（关联评分 {score:+.2f}）。"
    )
    return result
