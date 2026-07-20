"""基于 AKShare 的 A 股行情数据拉取模块（多数据源自动切换）。

AKShare 免费、无需 token。默认东方财富接口的 API 子域名
(push2his.eastmoney.com) 在部分网络下被屏蔽，因此这里按
东财 -> 新浪 -> 腾讯 的顺序自动切换，任一数据源可用即返回，
最大程度保证在受限网络下也能拉到数据。
"""
from __future__ import annotations

import datetime as _dt
import time
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import amazingdata_source, dcache, net

# AKShare 中文列名 -> 英文标准列名
_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",       # 单位：手
    "成交额": "amount",       # 单位：元
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",     # 单位：%
}


def _normalize_symbol(symbol: str) -> str:
    """去除市场前缀，返回 6 位纯数字代码。如 sh600519 -> 600519。"""
    symbol = symbol.strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if symbol.startswith(prefix):
            symbol = symbol[len(prefix):]
    return symbol.zfill(6) if symbol.isdigit() else symbol


# 最近一次成功的数据源（按 6 位代码记录），供 UI 展示实际用了哪个源
_LAST_SOURCE: dict = {}
_LAST_PROFILE: dict = {}


def last_source(symbol: str) -> str:
    return _LAST_SOURCE.get(_normalize_symbol(symbol), "-")


def last_profile(symbol: str) -> dict:
    return _LAST_PROFILE.get(_normalize_symbol(symbol), {})


def _prefixed_symbol(code: str) -> str:
    """给 6 位代码加交易所前缀（新浪/腾讯接口需要）。"""
    if code.startswith(("6", "9")):
        return "sh" + code
    if code.startswith(("0", "2", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


def _fmt(d) -> str:
    return d.strftime("%Y%m%d")


# ------------------------- 各数据源实现 -------------------------
def _from_eastmoney(code, start, end, adjust):
    """东方财富（push2his.eastmoney.com），字段最全含换手率/涨跌幅。

    该子域名在部分网络下被屏蔽；配置了代理时会通过 net.akshare_proxied()
    路由请求，尝试恢复东财数据源（未配置代理则直连，失败自动切到新浪/腾讯）。
    """
    with net.akshare_proxied():
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=_fmt(start), end_date=_fmt(end), adjust=adjust,
            timeout=8,
        )
    return df.rename(columns=_COL_MAP) if df is not None else df


def _from_sina(code, start, end, adjust):
    """新浪财经（finance.sina.com.cn），换手率由成交量/流通股本换算。"""
    df = ak.stock_zh_a_daily(
        symbol=_prefixed_symbol(code),
        start_date=_fmt(start), end_date=_fmt(end), adjust=adjust,
    )
    if df is None or df.empty:
        return df
    df = df.copy()
    if "outstanding_share" in df.columns and df["outstanding_share"].gt(0).any():
        df["turnover"] = df["volume"] / df["outstanding_share"] * 100
    return df


def _from_tencent(code, start, end, adjust):
    """腾讯（web.ifzq.gtimg.cn），仅含 OHLC 与成交量，无换手率。"""
    df = ak.stock_zh_a_hist_tx(
        symbol=_prefixed_symbol(code),
        start_date=_fmt(start), end_date=_fmt(end), adjust=adjust,
    )
    if df is None or df.empty:
        return df
    # 腾讯接口的 amount 列即成交量
    return df.rename(columns={"amount": "volume"})


def _from_amazingdata(code, start, end, adjust):
    """银河证券 AmazingData（券商官方 SDK，优先源；未安装/未登录则不可用）。"""
    return amazingdata_source.fetch_daily(code, _fmt(start), _fmt(end), adjust)


# 数据源优先级：(名称, 实现, 重试次数)
# UI 首屏优先速度：新浪/腾讯通常比东财 push2his 更快；东财保留作字段更全的兜底源。
_SOURCES = (
    ("新浪财经", _from_sina, 1),
    ("腾讯", _from_tencent, 1),
    ("东方财富", _from_eastmoney, 1),
)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列、补齐缺失字段（amount/turnover/pct_change），按日期升序。"""
    df = df.rename(columns=_COL_MAP).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = 0.0
    if "turnover" not in df.columns:
        df["turnover"] = 0.0
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100
    df["pct_change"] = df["pct_change"].fillna(0)
    df[["amount", "turnover"]] = df[["amount", "turnover"]].fillna(0)
    return df


@dcache.disk_cache(dcache.kline_ttl, name="kline")
def fetch_daily(symbol: str, days: int = 400, adjust: str = "qfq",
                freshness_bucket: int | None = None) -> pd.DataFrame:
    """拉取指定股票近 ``days`` 个自然日的日线数据（多数据源自动切换）。

    Args:
        symbol: 股票代码，支持 600519 / sh600519 等格式。
        days:   回溯的自然日天数（越大历史越长，用于计算长周期均线）。
        adjust: 复权方式，qfq=前复权，hfq=后复权，""=不复权。
        freshness_bucket: 可选的行情刷新分桶，仅参与缓存键；展示层按时间分桶传入可避免复用旧会话行情。

    Returns:
        标准化后的 DataFrame，含 date/open/high/low/close/volume/amount/
        pct_change/turnover 列，按日期升序。

    Raises:
        ConnectionError: 当所有数据源均获取失败时（附带各源错误详情）。
    """
    code = _normalize_symbol(symbol)
    end = _dt.date.today()
    start = end - _dt.timedelta(days=days)

    errors: list[str] = []
    profile = {"code": code, "sources": [], "success": "", "total_sec": 0.0}
    total_t0 = time.perf_counter()
    # 券商 SDK 可用时作为优先源（官方权威），否则用免费源
    sources = _SOURCES
    if amazingdata_source.available():
        sources = (("银河AmazingData", _from_amazingdata, 1),) + _SOURCES
    try:
        for name, fn, tries in sources:
            for attempt in range(tries):
                t0 = time.perf_counter()
                item = {"source": name, "attempt": attempt + 1, "ok": False, "sec": 0.0, "note": ""}
                try:
                    df = fn(code, start, end, adjust)
                    item["sec"] = round(time.perf_counter() - t0, 3)
                    if df is not None and not df.empty:
                        item["ok"] = True
                        item["rows"] = int(len(df))
                        profile["sources"].append(item)
                        profile["success"] = name
                        profile["total_sec"] = round(time.perf_counter() - total_t0, 3)
                        _LAST_SOURCE[code] = name
                        _LAST_PROFILE[code] = profile
                        return _standardize(df)
                    item["note"] = "返回空数据"
                    profile["sources"].append(item)
                    errors.append(f"{name}: 返回空数据({item['sec']:.2f}s)")
                    break
                except Exception as e:  # noqa: BLE001 逐源捕获，失败则切换下一个源
                    item["sec"] = round(time.perf_counter() - t0, 3)
                    item["note"] = f"{type(e).__name__}: {e}"
                    profile["sources"].append(item)
                    if attempt < tries - 1:
                        time.sleep(0.4 * (attempt + 1))
                        continue
                    errors.append(f"{name}: {e}({item['sec']:.2f}s)")
    finally:
        profile["total_sec"] = round(time.perf_counter() - total_t0, 3)
        _LAST_PROFILE[code] = profile

    raise ConnectionError(
        "所有行情数据源均获取失败：\n  - " + "\n  - ".join(errors) + "\n"
        "可能原因：网络不稳定 / 数据源被屏蔽（如公司网络封锁）/ akshare 版本过旧。\n"
        "建议：检查网络或代理后重试，或运行 `python3 -m pip install -U akshare` 升级。"
    )


@lru_cache(maxsize=1)
def stock_name_map() -> dict[str, str]:
    """返回 {代码: 名称} 映射，用于展示股票名称。"""
    try:
        with net.akshare_proxied():
            spot = ak.stock_zh_a_spot_em()
        return dict(zip(spot["代码"].astype(str), spot["名称"]))
    except Exception:
        pass
    try:
        listing = ak.stock_info_a_code_name()
        code_col = "code" if "code" in listing.columns else "代码"
        name_col = "name" if "name" in listing.columns else "名称"
        return dict(zip(listing[code_col].astype(str).str.zfill(6), listing[name_col].astype(str)))
    except Exception:
        return {}


def get_stock_name(symbol: str) -> str:
    code = _normalize_symbol(symbol)
    # 1. 本地自选股名称表（白名单，秒级、无网络；避免逐只等券商 15s 超时）
    try:
        from stock_analyzer import stock_meta
        n = stock_meta.watchlist_name_map().get(code)
        if n:
            return n
    except Exception:  # noqa: BLE001
        pass
    # 2. 券商官方(权威)
    try:
        if amazingdata_source.available():
            n = amazingdata_source.stock_name(code)
            if n:
                return n
    except Exception:  # noqa: BLE001
        pass
    # 3. 东财现货名称表  4. 兜底用代码
    return stock_name_map().get(code, code)
