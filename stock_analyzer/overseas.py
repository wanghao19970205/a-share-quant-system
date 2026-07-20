"""外围市场走势分析：美股三大指数（纳斯达克 / 标普500 / 道琼斯）。

用途：A 股开盘前，隔夜美股是重要的情绪先行指标，本模块拉取各指数近期走势
并量化为「外围情绪」评分，供预估 A 股次日涨跌时参考。
（日韩指数已按需求移除，只看美股；如需恢复见 ASIAN_INDICES 注释。）

数据源：美股走新浪 index_us_stock_sina（国内直连稳定）→ 失败回退 Yahoo。
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import dcache

try:
    import yfinance as _yf
except Exception:  # noqa: BLE001 未安装时回退到原始请求
    _yf = None

try:
    from curl_cffi import requests as _ccr  # yfinance 1.2+ 需要的会话类型
except Exception:  # noqa: BLE001
    _ccr = None

# 美股指数：中文名 -> {新浪代码, Yahoo代码}
US_INDICES = {
    "纳斯达克": {"sina": ".IXIC", "yahoo": "^IXIC"},
    "标普500": {"sina": ".INX", "yahoo": "^GSPC"},
    "道琼斯": {"sina": ".DJI", "yahoo": "^DJI"},
}
# 亚太指数：已按需求移除（只看美股）。如需恢复日韩，填回下方即可：
#   "日经225": {"yahoo": "^N225", "kw": ("日经",)},
#   "韩国KOSPI": {"yahoo": "^KS11", "kw": ("韩国", "KOSPI", "首尔")},
ASIAN_INDICES: dict = {}
# 各市场对 A 股次日的影响权重（隔夜美股科技股领先性最强）
WEIGHTS = {"纳斯达克": 1.5, "标普500": 1.5, "道琼斯": 1.0,
           "日经225": 1.0, "韩国KOSPI": 1.0}

_UA = {"User-Agent": "Mozilla/5.0"}

# ------------------------- 代理池（绕过 Yahoo 限流） -------------------------
# 通过环境变量 A_PROXIES 配置，逗号分隔，例如：
#   export A_PROXIES="http://ip1:port,http://user:pass@ip2:port"
# 每次请求轮换一个代理；命中限流时退避并切换到下一个代理重试。
def _load_proxies() -> list[str]:
    raw = os.environ.get("A_PROXIES", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


_PROXIES = _load_proxies()
_proxy_cycle = itertools.cycle(_PROXIES) if _PROXIES else None
_proxy_lock = threading.Lock()


def _next_proxy():
    """轮换取下一个代理；未配置则返回 None。"""
    if not _proxy_cycle:
        return None
    with _proxy_lock:
        return next(_proxy_cycle)


def set_proxies(proxies: list[str]) -> None:
    """运行时设置 Yahoo 代理列表（供 UI 侧边栏动态配置）。"""
    global _PROXIES, _proxy_cycle
    cleaned = [p.strip() for p in (proxies or []) if p and p.strip()]
    _PROXIES = cleaned
    _proxy_cycle = itertools.cycle(cleaned) if cleaned else None


def _make_yf_session(proxy: str | None):
    """构造 yfinance 需要的 curl_cffi 会话（可带代理）。"""
    if _ccr is None or proxy is None:
        return None
    return _ccr.Session(impersonate="chrome",
                        proxies={"http": proxy, "https": proxy})


@dataclass
class MarketTrend:
    """单个外围市场的走势与信号。"""
    name: str
    available: bool
    last_close: float = 0.0
    pct: float = 0.0        # 最新一日涨跌幅 %
    cum3: float = 0.0       # 近3日累计涨跌幅 %
    trend: str = "-"        # 短期趋势：上行/下行/走平
    score: int = 0          # 多空信号：+看多 / -看空
    source: str = ""        # 实际使用的数据源
    note: str = ""


@dataclass
class OverseasSentiment:
    """外围市场综合情绪。"""
    weighted_score: float
    level: str              # bullish / bearish / neutral
    label: str              # 外围偏多 / 外围偏空 / 外围中性
    summary: str
    markets: list[MarketTrend] = field(default_factory=list)


# ------------------------- 数据源 -------------------------
def _yahoo_via_urllib(symbol: str, rng: str, proxy: str | None = None) -> pd.Series | None:
    """直接请求 Yahoo chart 接口（yfinance 不可用时的回退，支持代理）。"""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(symbol)}?interval=1d&range={rng}")
    req = urllib.request.Request(url, headers=_UA)
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        raw = opener.open(req, timeout=15).read()
    else:
        raw = urllib.request.urlopen(req, timeout=15).read()
    res = json.loads(raw)["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    s = pd.Series(closes, index=pd.to_datetime(ts, unit="s")).dropna()
    return s if not s.empty else None


def _fetch_yahoo(symbol: str, rng: str = "3mo", retries: int | None = None) -> pd.Series | None:
    """Yahoo Finance 日线收盘价序列。

    优先用 yfinance（curl_cffi 会话，自动管理 cookie/crumb）；
    配置了 A_PROXIES 时，每次请求轮换代理、命中限流退避后切换代理重试。
    """
    attempts = retries or max(3, len(_PROXIES) + 1)
    last_err: Exception | None = None
    for attempt in range(attempts):
        proxy = _next_proxy()
        try:
            if _yf is not None:
                session = _make_yf_session(proxy)
                tk = _yf.Ticker(symbol, session=session) if session else _yf.Ticker(symbol)
                hist = tk.history(period=rng, auto_adjust=False)
                if hist is not None and not hist.empty and "Close" in hist:
                    s = hist["Close"].dropna()
                    s.index = pd.to_datetime(s.index)
                    if not s.empty:
                        return s
                raise ValueError("空数据")
            return _yahoo_via_urllib(symbol, rng, proxy)
        except Exception as e:  # noqa: BLE001 限流/网络异常 → 退避并切换代理重试
            last_err = e
            if attempt < attempts - 1:
                time.sleep(1.2 * (attempt + 1))
    if last_err:
        raise last_err
    return None


def _fetch_us_sina(sina_symbol: str) -> pd.Series | None:
    df = ak.index_us_stock_sina(symbol=sina_symbol)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["close"]


@lru_cache(maxsize=1)
def _global_spot():
    return ak.index_global_spot_em()


def _fetch_global_em(keywords: tuple[str, ...]) -> pd.Series | None:
    spot = _global_spot()
    if spot is None or spot.empty:
        return None
    name_col = "名称" if "名称" in spot.columns else spot.columns[1]
    mask = spot[name_col].astype(str).apply(lambda x: any(k in x for k in keywords))
    matched = spot.loc[mask, name_col]
    if matched.empty:
        return None
    df = ak.index_global_hist_em(symbol=str(matched.iloc[0]))
    if df is None or df.empty:
        return None
    close_col = "最新价" if "最新价" in df.columns else "收盘"
    df = df.rename(columns={"日期": "date", close_col: "close"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["close"]


# ------------------------- 分析 -------------------------
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


def _trend(close: pd.Series) -> str:
    if len(close) < 6:
        return "-"
    ma5 = close.rolling(5).mean()
    diff = ma5.iloc[-1] - ma5.iloc[-5]
    base = abs(close.iloc[-1]) or 1
    if diff / base > 0.002:
        return "上行"
    if diff / base < -0.002:
        return "下行"
    return "走平"


def _build(name: str, close: pd.Series, source: str) -> MarketTrend:
    close = close.dropna().astype(float)
    if len(close) < 2:
        return MarketTrend(name, available=False, note="历史数据不足")
    pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
    n = min(4, len(close))
    cum3 = (close.iloc[-1] / close.iloc[-n] - 1) * 100
    return MarketTrend(
        name=name, available=True, last_close=float(close.iloc[-1]),
        pct=pct, cum3=cum3, trend=_trend(close), score=_signal(pct),
        source=source,
    )


def _fetch_market(name: str, sources: list[tuple[str, callable]]) -> MarketTrend:
    """按优先级尝试多个数据源，返回首个成功的走势；全失败则标记不可用。"""
    errs = []
    for src_name, fn in sources:
        try:
            close = fn()
            if close is not None and len(close) >= 2:
                return _build(name, close, src_name)
            errs.append(f"{src_name}:空")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{src_name}:{type(e).__name__}")
    return MarketTrend(name, available=False, note="；".join(errs) or "无可用数据源")


@dcache.disk_cache(dcache.market_ttl, name="overseas")
def analyze() -> OverseasSentiment:
    """拉取并分析全部外围市场，返回综合情绪。单个市场失败不影响整体。"""
    markets: list[MarketTrend] = []

    for name, cfg in US_INDICES.items():
        markets.append(_fetch_market(name, [
            ("新浪", lambda c=cfg: _fetch_us_sina(c["sina"])),
            ("Yahoo", lambda c=cfg: _fetch_yahoo(c["yahoo"])),
        ]))

    for name, cfg in ASIAN_INDICES.items():
        markets.append(_fetch_market(name, [
            ("Yahoo", lambda c=cfg: _fetch_yahoo(c["yahoo"])),
            ("东财", lambda c=cfg: _fetch_global_em(c["kw"])),
        ]))

    avail = [m for m in markets if m.available]
    if not avail:
        return OverseasSentiment(0.0, "neutral", "外围数据不可用",
                                 "未能获取任何外围市场数据，请检查网络。", markets)

    total_w = sum(WEIGHTS.get(m.name, 1.0) for m in avail)
    weighted = sum(m.score * WEIGHTS.get(m.name, 1.0) for m in avail) / total_w

    if weighted >= 0.8:
        level, label = "bullish", "外围偏多"
    elif weighted <= -0.8:
        level, label = "bearish", "外围偏空"
    else:
        level, label = "neutral", "外围中性"

    ups = [m.name for m in avail if m.score > 0]
    downs = [m.name for m in avail if m.score < 0]
    parts = []
    if ups:
        parts.append("上涨：" + "、".join(ups))
    if downs:
        parts.append("下跌：" + "、".join(downs))
    summary = "；".join(parts) or "外围市场涨跌互现，方向不明。"

    return OverseasSentiment(round(weighted, 2), level, label, summary, markets)
