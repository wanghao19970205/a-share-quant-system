"""Lightweight stock display metadata for UI tables."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import all_a_meta, data, dcache, net, sectors


_MARKET_HINTS = {
    "6": "沪市主板",
    "0": "深市主板",
    "2": "深市主板",
    "3": "创业板",
    "8": "北交所",
    "4": "北交所",
    "9": "沪市B股",
}


def _watchlist_path() -> str:
    return os.environ.get(
        "QUANT_WATCHLIST",
        os.path.join(os.environ.get("SNAPSHOT_DIR", "snapshots"), "watchlist.txt"),
    )


@lru_cache(maxsize=1)
def watchlist_info_map() -> dict[str, dict]:
    path = _watchlist_path()
    if not os.path.exists(path):
        return {}
    out: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                body, _, comment = line.partition("#")
                parts = body.strip().split(maxsplit=1)
                code = data._normalize_symbol(parts[0]) if parts else ""
                if len(code) != 6 or not code.isdigit():
                    continue
                name = parts[1].strip() if len(parts) > 1 else ""
                raw_hints = (comment.replace("，", ",").replace("、", ",")
                             .replace("；", ",").replace(";", ",").replace("／", "/"))
                hints = [x.strip() for part in raw_hints.split(",") for x in part.split("/") if x.strip()]
                out[code] = {"name": name, "concept_hints": hints}
    except Exception:  # noqa: BLE001
        return out
    return out


@lru_cache(maxsize=1)
def watchlist_name_map() -> dict[str, str]:
    return {code: str(info.get("name") or "") for code, info in watchlist_info_map().items()}


@dcache.disk_cache(lambda: 86400 * 14, name="stock_business")
def _business_from_ths(code: str) -> dict:
    df = ak.stock_zyjs_ths(symbol=code)
    if df is None or df.empty:
        return {}
    row = df.iloc[0]
    return {
        "business": str(row.get("主营业务", "") or ""),
        "product_type": str(row.get("产品类型", "") or ""),
        "product_name": str(row.get("产品名称", "") or ""),
    }


@dcache.disk_cache(lambda: 86400 * 14, name="stock_em_info")
def _stock_info_from_em(code: str) -> dict:
    with net.akshare_proxied():
        df = ak.stock_individual_info_em(symbol=code, timeout=8)
    if df is None or df.empty or not {"item", "value"}.issubset(df.columns):
        return {}
    info = {str(r.get("item") or ""): str(r.get("value") or "") for _, r in df.iterrows()}
    return {
        "em_name": info.get("股票简称", ""),
        "a_industry": info.get("行业", ""),
        "market_value": info.get("总市值", ""),
    }


_A_INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "半导体": ("半导体", "芯片", "集成电路", "封测", "晶圆", "光刻"),
    "国防军工": ("军工", "国防", "航空", "航天", "兵器", "导弹", "雷达"),
    "电子": ("电子", "消费电子", "显示", "面板", "光电", "传感器"),
    "计算机": ("软件", "计算机", "人工智能", "云计算", "数据中心", "信创"),
    "通信": ("通信", "光模块", "光通信", "5G", "卫星通信"),
    "机械设备": ("机械", "装备", "机器人", "机床", "自动化", "工程机械"),
    "电力设备": ("光伏", "风电", "储能", "电池", "电力设备", "新能源"),
    "汽车": ("汽车", "新能源车", "零部件", "电驱", "无人驾驶"),
    "医药生物": ("医药", "生物", "疫苗", "医疗", "创新药", "中药"),
    "基础化工": ("化工", "材料", "树脂", "纤维", "高分子", "新材料"),
    "有色金属": ("有色", "铜", "铝", "锂", "稀土", "金属粉末", "钛", "黄金", "珠宝"),
    "公用事业": ("电力", "发电", "热电", "能源"),
    "煤炭": ("煤炭", "煤矿", "能源"),
    "房地产": ("地产", "房地产", "开发"),
    "建筑装饰": ("建筑", "工程", "园林", "生态"),
    "轻工制造": ("家居", "家具", "包装"),
    "商贸零售": ("珠宝", "黄金", "零售"),
    "金融": ("银行", "证券", "保险", "金融", "期货"),
    "食品饮料": ("白酒", "食品", "饮料", "乳", "调味"),
}

_A_CONCEPT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "商业航天": ("商业航天", "卫星", "火箭", "航天", "空间站", "北斗", "遥感"),
    "军工": ("军工", "国防", "航空", "航天", "兵器", "导弹", "雷达", "军品"),
    "半导体": ("半导体", "芯片", "集成电路", "封测", "晶圆", "光刻", "IGBT", "MCU", "存储"),
    "PCB": ("PCB", "电路板", "覆铜板", "封装基板"),
    "光通信": ("光模块", "光通信", "光纤", "光缆", "通信设备"),
    "人工智能": ("人工智能", "AI", "大模型", "算力", "智能", "机器人"),
    "算力": ("算力", "服务器", "数据中心", "云计算", "GPU"),
    "机器人": ("机器人", "伺服", "减速器", "自动化", "智能装备"),
    "新能源车": ("新能源车", "新能源汽车", "汽车", "动力电池", "电驱", "充电桩"),
    "锂电池": ("锂电", "锂电池", "电池", "正极", "负极", "电解液", "隔膜"),
    "光伏": ("光伏", "太阳能", "硅片", "组件", "逆变器"),
    "储能": ("储能", "电池储能", "钠电", "钒电池"),
    "风电": ("风电", "风机", "叶片", "海上风电"),
    "低空经济": ("低空经济", "无人机", "通航", "飞行器", "eVTOL"),
    "消费电子": ("消费电子", "手机", "平板", "穿戴", "摄像头", "显示", "面板"),
    "信创": ("信创", "国产软件", "操作系统", "数据库", "中间件"),
    "数据要素": ("数据要素", "数据资产", "数据交易", "大数据"),
    "医药": ("医药", "创新药", "生物", "疫苗", "医疗器械", "中药"),
    "化工新材料": ("新材料", "化工", "树脂", "纤维", "膜材料", "高分子"),
    "有色金属": ("有色", "铜", "铝", "钛", "稀土", "黄金", "钽", "镁"),
    "超硬材料": ("超硬材料", "金刚石", "培育钻石", "砂轮", "磨具", "磨料"),
    "电子元件": ("电容", "电感", "元器件", "连接器", "继电器"),
    "PCB": ("PCB", "电路板", "覆铜板", "封装基板"),
    "电力": ("电力", "发电", "热电", "能源"),
    "煤炭": ("煤炭", "煤矿", "能源"),
    "房地产": ("地产", "房地产", "园区", "开发"),
    "黄金珠宝": ("黄金", "珠宝", "首饰"),
    "家居": ("家居", "家具", "办公椅"),
}

_THEME_FALLBACKS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("电子制造", ("600667", "002384", "603890", "002484", "002463", "002669", "002546", "002745", "600184",
                "002138", "002636", "002409"), "电子", ("电子元件", "PCB", "消费电子")),
    ("化工新材料", ("605589", "603823", "605020", "002741", "600378", "002149"), "基础化工", ("化工新材料", "新材料")),
    ("电力能源", ("605580", "600578", "600403", "600869", "600673"), "公用事业", ("电力", "新能源")),
    ("机器人装备", ("002747", "603082", "603203", "603166", "002046", "000519"), "机械设备", ("机器人", "智能装备")),
    ("超硬材料", ("600172",), "机械设备", ("超硬材料", "培育钻石", "磨料磨具")),
    ("通信补充", ("600522", "600498", "600487"), "通信", ("光通信", "通信设备", "光纤光缆")),
    ("信创数据", ("000032",), "计算机", ("信创", "数据要素")),
    ("房地产建筑", ("600683", "603316", "001267", "603389"), "房地产", ("地产", "建筑装饰")),
    ("有色黄金", ("002171", "600916", "002345"), "有色金属", ("有色金属", "黄金珠宝")),
    ("消费服务", ("600655", "603661", "603991"), "商贸零售", ("消费", "家居")),
    ("半导体", ("002185", "600584", "605358", "603986", "003026", "603005", "600206", "603061", "002156",
                "002371", "600520", "600877", "002213", "600360", "603160", "003043", "603501", "002119",
                "600171", "600460", "001309", "002049", "600745", "603690", "600641", "002077", "603290",
                "603893", "603375", "605111", "603068"), "半导体", ("芯片", "集成电路", "封测")),
    ("通信", ("002281", "000063", "601869", "600105", "603083", "002897", "003031", "002902", "002792",
              "000070", "600198", "600345", "002296", "002796", "002491", "000586", "002583", "603042",
              "600776", "603421", "002313", "002881", "002194", "600775", "002396", "003040", "603803",
              "002017", "603236", "603118", "002104"), "通信", ("光通信", "通信设备", "卫星通信")),
    ("显示电子", ("000725", "600707", "603773", "002845", "001399", "002456", "600703", "002222", "600552", "000100", "002654",
                 "002992", "002137", "002106", "603685", "003019", "002273", "002036", "000050", "002983",
                 "603297", "000020", "600071", "000536", "603703", "605588", "002289", "000509", "002449",
                 "001308", "000045", "002387", "605218", "002876", "002952", "002962", "002955", "003015",
                 "000727", "001373", "002587", "603679", "002808", "002217"), "电子", ("消费电子", "显示面板", "光电")),
    ("商业航天", ("000026", "000039", "000058", "000066", "000547", "000551", "000561", "000697", "000708",
               "000733", "000738", "000768", "000901", "000922", "000925", "000962", "001208", "001229",
               "001270", "001379", "001400", "002006", "002023", "002025", "002080", "002081", "002085",
               "002111", "002115", "002151", "002179", "002182", "002201", "002202", "002204", "002212",
               "002246", "002278", "002297", "002324", "002338", "002342", "002361", "002389", "002402",
               "002413", "002414", "002428", "002430", "002446", "002465", "002471", "002516", "002540",
               "002560", "002565", "002579", "002625", "002651", "002658", "002683", "002738", "002756",
               "002815", "002829", "002843", "002848", "002879", "002927", "002933", "002935", "002938",
               "002977", "002985", "003009", "003029", "600038", "600060", "600118", "600143", "600169",
               "600266", "600268", "600316", "600330", "600363", "600372", "600391", "600399", "600416",
               "600456", "600501", "600562", "600590", "600592", "600736", "600760", "600765", "600850",
               "600862", "600879", "600893", "600990", "601026", "601106", "601137", "601186", "601608",
               "601669", "601698", "601800", "601992", "603011", "603017", "603100", "603131", "603211",
               "603212", "603228", "603261", "603267", "603305", "603308", "603507", "603678", "603757",
               "603809", "603859", "603920", "603936", "603977", "605058", "605090", "605123", "605598"),
     "国防军工", ("商业航天", "航天航空", "航空装备")),
)

_THEME_BY_CODE: dict[str, tuple[str, tuple[str, ...]]] = {}
for _theme, _codes, _industry, _concepts in _THEME_FALLBACKS:
    for _code in _codes:
        _THEME_BY_CODE[_code] = (_industry, _concepts)


def _split_names(text: str) -> list[str]:
    out: list[str] = []
    for raw in str(text or "").replace("，", "、").replace(",", "、").replace(";", "、").split("、"):
        val = raw.strip()
        if val and val.lower() != "nan" and val not in out:
            out.append(val)
    return out


def _match_keywords(text: str, mapping: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    hits: list[str] = []
    for name, kws in mapping.items():
        for kw in kws:
            if kw and kw in text:
                if name not in names:
                    names.append(name)
                if kw not in hits:
                    hits.append(kw)
    return names, hits


def _concepts_from_text(text: str) -> tuple[list[str], list[str]]:
    return _match_keywords(text, _A_CONCEPT_KEYWORDS)


def _industries_from_text(text: str) -> list[str]:
    names, _ = _match_keywords(text, _A_INDUSTRY_KEYWORDS)
    return names


def _add_unique(items: list[str], values) -> None:
    for val in values or []:
        val = str(val or "").strip()
        if val and val.lower() != "nan" and val not in items:
            items.append(val)


def _base_meta(code: str) -> tuple[str, dict, list[str]]:
    watch_info = watchlist_info_map().get(code, {})
    name = str(watch_info.get("name") or "") or code
    concepts: list[str] = []
    _add_unique(concepts, watch_info.get("concept_hints") or [])
    return name, watch_info, concepts


@lru_cache(maxsize=2048)
def get_stock_meta_light(symbol: str) -> dict:
    code = data._normalize_symbol(symbol)
    name, watch_info, concepts = _base_meta(code)
    hint_text = " ".join([name, " ".join(watch_info.get("concept_hints") or [])])
    kw_concepts, a_keywords = _concepts_from_text(hint_text)
    _add_unique(concepts, kw_concepts)
    theme_industry, theme_concepts = _THEME_BY_CODE.get(code, ("", ()))
    _add_unique(concepts, theme_concepts)
    industry_hits = _industries_from_text(hint_text)
    a_industry = theme_industry or (industry_hits[0] if industry_hits else _MARKET_HINTS.get(code[:1], ""))
    a_concepts = "、".join(concepts[:8])
    return {
        "code": code,
        "name": name or code,
        "industry": a_industry,
        "sector": a_concepts,
        "a_industry": a_industry,
        "a_concepts": a_concepts,
        "overseas_sector": "",
        "business": "",
        "product_type": "",
        "product_name": "",
        "matched_keywords": "、".join(a_keywords),
    }


@lru_cache(maxsize=1024)
def get_stock_meta(symbol: str) -> dict:
    code = data._normalize_symbol(symbol)
    name, watch_info, concepts = _base_meta(code)
    if not name or name == code:
        try:
            name = data.get_stock_name(code)
        except Exception:  # noqa: BLE001
            name = code
    em_info: dict = {}
    try:
        em_info = _stock_info_from_em(code)
        if (not name or name == code) and em_info.get("em_name"):
            name = str(em_info["em_name"])
    except Exception:  # noqa: BLE001
        em_info = {}
    business = product_type = product_name = ""
    try:
        biz = _business_from_ths(code)
        business = str(biz.get("business") or "")
        product_type = str(biz.get("product_type") or "")
        product_name = str(biz.get("product_name") or "")
    except Exception:  # noqa: BLE001
        pass
    text = " ".join(x for x in (business, product_type, product_name, " ".join(watch_info.get("concept_hints") or [])) if x)
    overseas_matched = sectors.map_to_sectors(text) if text else {}
    overseas_sector = "、".join(overseas_matched.keys()) if overseas_matched else ""
    overseas_keywords = "、".join(dict.fromkeys(kw for kws in overseas_matched.values() for kw in kws)) if overseas_matched else ""

    kw_concepts, a_keywords = _concepts_from_text(text)
    _add_unique(concepts, kw_concepts)
    _add_unique(concepts, [val for val in _split_names(product_type) if 2 <= len(val) <= 12])
    theme_industry, theme_concepts = _THEME_BY_CODE.get(code, ("", ()))
    _add_unique(concepts, theme_concepts)
    industry_hits = _industries_from_text(text)
    raw_industry = str(em_info.get("a_industry") or "").strip()
    a_industry = raw_industry or theme_industry or (industry_hits[0] if industry_hits else _MARKET_HINTS.get(code[:1], ""))
    a_concepts = "、".join(concepts[:8])
    industry_display = a_industry or product_type or _MARKET_HINTS.get(code[:1], "")
    return {
        "code": code,
        "name": name or code,
        "industry": industry_display,
        "sector": a_concepts or overseas_sector,
        "a_industry": a_industry,
        "a_concepts": a_concepts,
        "overseas_sector": overseas_sector,
        "business": business,
        "product_type": product_type,
        "product_name": product_name,
        "matched_keywords": "、".join(dict.fromkeys(a_keywords + _split_names(overseas_keywords))),
    }


@dcache.disk_cache(lambda: 86400 * 7, name="watchlist_concept_reverse")
def reverse_concepts_for_codes(codes: tuple[str, ...], max_boards: int = 120) -> dict[str, list[str]]:
    """按概念板块成分反查股票所属概念。

    这个接口会遍历多个东财概念板块，首轮较慢；默认不在 UI 首屏调用，适合后台预热缓存。
    """
    targets = {data._normalize_symbol(c) for c in codes if str(c).strip()}
    out: dict[str, list[str]] = {c: [] for c in targets}
    if not targets:
        return out
    try:
        with net.akshare_proxied():
            boards = ak.stock_board_concept_name_em()
    except Exception:  # noqa: BLE001
        return out
    if boards is None or boards.empty:
        return out
    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    names = boards[name_col].dropna().astype(str).head(max_boards).tolist()

    def _one(board: str):
        try:
            with net.akshare_proxied():
                cons = ak.stock_board_concept_cons_em(symbol=board)
            if cons is None or cons.empty or "代码" not in cons.columns:
                return board, []
            got = [data._normalize_symbol(c) for c in cons["代码"].dropna().astype(str)]
            return board, [c for c in got if c in targets]
        except Exception:  # noqa: BLE001
            return board, []

    with ThreadPoolExecutor(max_workers=12) as ex:
        for board, got in ex.map(_one, names):
            for code in got:
                if board not in out[code]:
                    out[code].append(board)
    return out


def meta_for_codes(
    symbols,
    remote: bool = True,
    use_all_a_meta: bool = False,
) -> pd.DataFrame:
    codes = list(dict.fromkeys(
        data._normalize_symbol(str(symbol)) for symbol in symbols if str(symbol).strip()
    ))
    if not codes:
        return pd.DataFrame()

    persistent = all_a_meta.load_all_a_meta() if use_all_a_meta else pd.DataFrame()
    persistent_by_code = (
        persistent.set_index("code").to_dict("index") if not persistent.empty else {}
    )
    fn = get_stock_meta if remote else get_stock_meta_light
    rows: list[dict] = []
    for code in codes:
        fallback = fn(code)
        saved = persistent_by_code.get(code)
        if saved:
            name = str(saved.get("name") or "").strip()
            industry = str(saved.get("a_industry") or "").strip()
            concepts = str(saved.get("a_concepts") or "").strip()
            fallback.update({
                "name": name or fallback.get("name") or code,
                "market_board": saved.get("market_board") or all_a_meta.market_board(code),
                "industry": industry,
                "sector": concepts,
                "a_industry": industry,
                "a_industries": saved.get("a_industries") or industry,
                "a_concepts": concepts,
                "meta_updated_at": saved.get("meta_updated_at") or "",
            })
        else:
            fallback["market_board"] = all_a_meta.market_board(code)
        rows.append(fallback)
    return pd.DataFrame(rows).drop_duplicates("code")


def enrich_frame(
    df: pd.DataFrame,
    code_col: str = "code",
    remote: bool = True,
    use_all_a_meta: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or code_col not in df.columns:
        return df
    out = df.copy()
    out[code_col] = out[code_col].astype(str).map(data._normalize_symbol)
    meta = meta_for_codes(
        out[code_col].dropna().astype(str).unique().tolist(),
        remote=remote,
        use_all_a_meta=use_all_a_meta,
    )
    if meta.empty:
        return out
    return out.merge(meta, left_on=code_col, right_on="code", how="left", suffixes=("", "_meta"))
