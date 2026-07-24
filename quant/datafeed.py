"""量化选股 · 数据接入层（akshare 免费源，已在本机网络实测可达）。

数据源可达性说明（本机/公司网络）：
- 东财**行情推送**服务器 push2/push2his（全市场快照、板块成分、资金流）被屏蔽 → 需代理；
  故本层核心数据**避开** push2，改用可直连的源：
  · 日线：复用 stock_analyzer.data.fetch_daily（新浪源，已验证）
  · 估值/财报/事件：东财**数据中心** datacenter-web（未被屏蔽，实测可用）
  · 成分/清单：中证 csindex、akshare code_name
- 少数仍走 push2 的接口用 net.akshare_proxied() 包裹，配置代理后可用。

所有适配器统一输出：含 6 位 ``code`` 列、日期列为 datetime 的整洁 DataFrame。
"""
from __future__ import annotations

import datetime as _dt

import akshare as ak
import pandas as pd

from stock_analyzer import amazingdata_source
from stock_analyzer import data as _sa_data
from stock_analyzer import net

_norm = _sa_data._normalize_symbol


# ------------------------- 股票池 / 清单 -------------------------
def _retry(fn, tries=3):
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.8 * (i + 1))
    raise last


def _csindex_members(index_code: str) -> list[str]:
    """指数成分：优先新浪(快/稳)，失败回退中证 csindex，均带轻量重试。"""
    def sina():
        df = ak.index_stock_cons(symbol=index_code)   # 列：品种代码/品种名称/纳入日期
        col = "品种代码" if "品种代码" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()

    def csi():
        df = ak.index_stock_cons_csindex(symbol=index_code)
        col = "成分券代码" if "成分券代码" in df.columns else df.columns[4]
        return df[col].astype(str).tolist()

    try:
        return _retry(sina)
    except Exception:  # noqa: BLE001
        return _retry(csi)


_MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


def _mainboard_active_universe() -> list[str]:
    """Read the quant-only active mainboard universe; fail closed if missing or empty."""
    from quant import config

    path = config.MAINBOARD_UNIVERSE_FILE
    try:
        with open(path, encoding="utf-8") as f:
            codes = sorted({_norm(c) for c in f.read().split() if _norm(c).isdigit()})
    except OSError as exc:
        raise RuntimeError(f"mainboard universe file unavailable: {path}") from exc
    if not codes:
        raise RuntimeError(f"mainboard universe file is empty: {path}")
    return codes


def _broker_mainboard_records() -> "list[tuple[str, str]] | None":
    """券商 SDK 取全 A 码表+名称，返回 [(6位code, name)]；不可用/失败返回 None（交由兜底）。

    两次批量单请求：get_code_list 一次返回沪深北全 A 代码，get_stock_basic 一次批量
    返回名称/上市状态；不逐票循环。用 IS_LISTED==1 排除退市（比"名称含退"更准），
    再按主板前缀过滤。全程走券商 SDK、绝不碰 akshare，故不会毒化进程网络态。
    """
    from stock_analyzer import amazingdata_source as _ad_src

    if not _ad_src.available() or not _ad_src._ensure_login():
        return None
    try:
        base = _ad_src._base
        broker_codes = _ad_src.sdk_call(base.get_code_list, "EXTRA_STOCK_A", timeout=60.0)
        if not broker_codes:
            return None
        info = _ad_src._ad.InfoData()
        basic = _ad_src.sdk_call(info.get_stock_basic, list(broker_codes), timeout=90.0)
    except Exception:  # noqa: BLE001 券商取码表失败则回退 akshare 子进程
        return None

    if isinstance(basic, dict):
        frames = [v for v in basic.values() if v is not None and len(v)]
        basic = pd.concat(frames, ignore_index=True) if frames else None
    if basic is None or len(basic) == 0:
        return None

    name_cols = ("SECURITY_NAME", "SEC_NAME", "NAME", "名称", "证券简称", "SECURITY_ABBR")
    name_col = next((c for c in name_cols if c in basic.columns), None)
    records: list[tuple[str, str]] = []
    for _, row in basic.iterrows():
        raw = str(row.get("MARKET_CODE", "") or "")
        code = _norm(raw.split(".")[0])
        name = str(row.get(name_col, "") or "").strip() if name_col else ""
        is_listed = row.get("IS_LISTED", 1)
        try:
            listed_ok = int(is_listed) == 1
        except (TypeError, ValueError):
            listed_ok = True  # 字段缺失/异常时不误杀，保持与旧口径一致
        if code.isdigit() and code.startswith(_MAINBOARD_PREFIXES) and listed_ok:
            records.append((code, name))
    return records or None


def _akshare_subprocess_records() -> "list[tuple[str, str]]":
    """兜底：在独立子进程内跑 akshare 取全 A 码表+名称，避免毒化本进程券商网络态。

    子进程只 import akshare、拉一次 stock_info_a_code_name、以 JSON 吐回，父进程解析。
    退市仍按"名称含退"排除（免费源无上市状态字段）。
    """
    import json
    import subprocess
    import sys as _sys

    child = (
        "import akshare as ak, json, sys\n"
        "df = ak.stock_info_a_code_name()\n"
        "cc = 'code' if 'code' in df.columns else df.columns[0]\n"
        "nc = next((c for c in ('name','名称','股票简称') if c in df.columns), None)\n"
        "out = [[str(r.get(cc,'')), (str(r.get(nc,'')) if nc else '')] "
        "for _, r in df.iterrows()]\n"
        "json.dump(out, sys.stdout, ensure_ascii=False)\n"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", child],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"akshare 子进程取码表失败 rc={proc.returncode}: {proc.stderr[-500:]}")
    records: list[tuple[str, str]] = []
    for raw_code, name in json.loads(proc.stdout):
        code = _norm(str(raw_code))
        name = str(name or "")
        if code.isdigit() and code.startswith(_MAINBOARD_PREFIXES) and "退" not in name:
            records.append((code, name))
    return records


def refresh_mainboard_universe() -> list[str]:
    """Refresh the quant-only active mainboard universe.

    券商优先（get_code_list + get_stock_basic，IS_LISTED 过滤，零 akshare 零毒化），
    券商不可用/失败时回退 akshare 子进程（隔离，不毒化本进程）。
    """
    from quant import config

    # 码表走 akshare 子进程：单次调用即得全量，简单快且已隔离防毒化；
    # 不再先试券商 get_code_list/get_stock_basic(每次白等 ~24s+ 才回退)。
    # 因子仍走券商(批量一次 vs akshare 逐票 3000+ 次，券商快几个数量级)。
    records = _akshare_subprocess_records()
    source = "akshare-subprocess"
    codes = sorted({code for code, _ in records})
    if not codes:
        raise RuntimeError("current A-share list produced no active mainboard codes")
    path = config.MAINBOARD_UNIVERSE_FILE
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Quant-only active Shanghai/Shenzhen mainboard universe; UI watchlist is separate.\n")
        f.writelines(f"{code} {name}\n" for code, name in sorted(set(records)))
    print(f"[universe] refreshed {len(codes)} mainboard codes via {source}", flush=True)
    return codes
def universe(kind: str = "all", arg: str | None = None) -> list[str]:
    """返回 6 位代码列表。kind: 'all'=全A；'mainboard_active'=量化主板池；'csindex'=指数成分。"""
    if kind == "csindex":
        codes = _csindex_members(arg)
    elif kind == "mainboard_active":
        codes = _mainboard_active_universe()
    else:
        codes = _retry(lambda: ak.stock_info_a_code_name()["code"].astype(str).tolist())
    return sorted({_norm(c) for c in codes if _norm(c).isdigit()})


# ------------------------- 日线行情（AmazingData 优先，多免费源兜底） -------------------------
def broker_available() -> bool:
    """Return whether the configured AmazingData SDK can be attempted."""
    return amazingdata_source.available()


def daily_price(code: str, start: str = "20180101") -> pd.DataFrame:
    """单只日线（前复权），复用项目已有多源拉取（新浪直连可用）。"""
    code = _norm(code)
    start_d = _dt.datetime.strptime(start, "%Y%m%d").date()
    days = (_dt.date.today() - start_d).days + 5
    df = _sa_data.fetch_daily(code, days=days)  # date/open/high/low/close/volume/amount/turnover/pct_change
    df = df[df["date"] >= pd.Timestamp(start_d)].copy()
    # 券商源(AmazingData)返回的行情已带 code 列（如 000001.SZ），需先移除，
    # 再统一插入 6 位规范化 code，避免 insert 撞列名报 ValueError。
    if "code" in df.columns:
        df = df.drop(columns=["code"])
    df.insert(0, "code", code)
    return df.reset_index(drop=True)


# ------------------------- 估值时序（东财数据中心，直连可用） -------------------------
_VAL_COLMAP = {
    "数据日期": "date", "当日收盘价": "close", "当日涨跌幅": "pct_change",
    "总市值": "mv_total", "流通市值": "mv_float", "总股本": "shares_total",
    "流通股本": "shares_float", "PE(TTM)": "pe_ttm", "PE(静)": "pe_static",
    "市净率": "pb", "PEG值": "peg", "市现率": "pcf", "市销率": "ps",
}


def valuation(code: str) -> pd.DataFrame:
    """个股每日估值时序（PE/PB/PS/PEG/市现率 + 市值/股本），2018 至今。"""
    code = _norm(code)
    df = ak.stock_value_em(symbol=code)
    df = df.rename(columns=_VAL_COLMAP)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.insert(0, "code", code)
    return df.reset_index(drop=True)


# ------------------------- 全市场按报告期财报（数据中心，直连可用） -------------------------
def _code_col(df: pd.DataFrame) -> str | None:
    for c in ("股票代码", "证券代码", "代码", "成分券代码", "品种代码"):
        if c in df.columns:
            return c
    return None


def _insert_code(df: pd.DataFrame) -> pd.DataFrame:
    code_col = _code_col(df)
    if code_col and "code" not in df.columns:
        df.insert(0, "code", df[code_col].astype(str).map(_norm))
    return df


def _tag_report(df: pd.DataFrame, report_date: str) -> pd.DataFrame:
    """规范代码列、补报告期列、把公告日期规范为 ann_date（防前视用）。"""
    df = _insert_code(df.copy())
    df["report_date"] = pd.to_datetime(report_date)
    for c in ("最新公告日期", "公告日期", "预告公告日"):
        if c in df.columns:
            df["ann_date"] = pd.to_datetime(df[c], errors="coerce")
            break
    return df


def financial_yjbb(report_date: str) -> pd.DataFrame:
    """业绩报表（EPS/营收/净利/ROE/毛利率 + 最新公告日期）。report_date 如 20251231。"""
    return _tag_report(ak.stock_yjbb_em(date=report_date), report_date)


def balance_sheet(report_date: str) -> pd.DataFrame:
    return _tag_report(ak.stock_zcfz_em(date=report_date), report_date)


def income(report_date: str) -> pd.DataFrame:
    return _tag_report(ak.stock_lrb_em(date=report_date), report_date)


def cashflow(report_date: str) -> pd.DataFrame:
    return _tag_report(ak.stock_xjll_em(date=report_date), report_date)


def performance_forecast(report_date: str) -> pd.DataFrame:
    """业绩预告（预告类型/预测指标/变动幅度 + 公告日期）。"""
    return _tag_report(ak.stock_yjyg_em(date=report_date), report_date)


# ------------------------- 事件类（数据中心，直连可用） -------------------------
def block_trades(start: str, end: str) -> pd.DataFrame:
    """大宗交易每日明细（成交价/折溢价/买卖营业部）。"""
    df = _insert_code(ak.stock_dzjy_mrmx(symbol="A股", start_date=start, end_date=end).copy())
    for c in ("交易日期", "日期"):
        if c in df.columns:
            df["date"] = pd.to_datetime(df[c], errors="coerce")
            break
    return df.reset_index(drop=True)


def lhb(start: str, end: str) -> pd.DataFrame:
    """龙虎榜明细。"""
    df = _insert_code(ak.stock_lhb_detail_em(start_date=start, end_date=end).copy())
    if "上榜日" in df.columns:
        df["date"] = pd.to_datetime(df["上榜日"], errors="coerce")
    return df.reset_index(drop=True)


def holder_num(report_date: str) -> pd.DataFrame:
    """全市场股东户数（report_date 如 20260331）。"""
    df = _insert_code(ak.stock_zh_a_gdhs(symbol=report_date).copy())
    df["report_date"] = pd.to_datetime(report_date)
    return df.reset_index(drop=True)


def dividend(report_date: str) -> pd.DataFrame:
    """分红送配（report_date 如 20251231）。"""
    df = _insert_code(ak.stock_fhps_em(date=report_date).copy())
    df["report_date"] = pd.to_datetime(report_date)
    return df.reset_index(drop=True)


# ------------------------- 两融（交易所公开源） -------------------------
def _safe_frame(fn, *, date: str | None = None) -> pd.DataFrame:
    try:
        df = fn()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    df = _insert_code(df.copy())
    if date and "date" not in df.columns:
        df["date"] = pd.to_datetime(date)
    return df.reset_index(drop=True)


def margin_sse(start: str, end: str) -> pd.DataFrame:
    """上交所融资融券市场汇总（日频）。"""
    df = _safe_frame(lambda: ak.stock_margin_sse(start_date=start, end_date=end))
    if not df.empty and "信用交易日期" in df.columns:
        df["date"] = pd.to_datetime(df["信用交易日期"], errors="coerce")
        df["market"] = "SSE"
    return df


def margin_szse(date: str) -> pd.DataFrame:
    """深交所融资融券市场汇总（日频）；当前 AkShare 偶发空表 schema 错误，失败返回空表。"""
    df = _safe_frame(lambda: ak.stock_margin_szse(date=date), date=date)
    if not df.empty:
        df["market"] = "SZSE"
    return df


def margin_detail_sse(date: str) -> pd.DataFrame:
    """上交所个股融资融券明细（日频）；接口不稳定，失败返回空表。"""
    df = _safe_frame(lambda: ak.stock_margin_detail_sse(date=date), date=date)
    if not df.empty:
        df["market"] = "SSE"
    return df


def margin_detail_szse(date: str) -> pd.DataFrame:
    """深交所个股融资融券明细（日频）；接口不稳定，失败返回空表。"""
    df = _safe_frame(lambda: ak.stock_margin_detail_szse(date=date), date=date)
    if not df.empty:
        df["market"] = "SZSE"
    return df


def margin_underlying_szse(date: str) -> pd.DataFrame:
    """深交所融资融券标的名单（日频）。"""
    df = _safe_frame(lambda: ak.stock_margin_underlying_info_szse(date=date), date=date)
    if not df.empty:
        df["market"] = "SZSE"
    return df


# 需代理（push2 行情推送服务器）——配置代理后可用
# AmazingData 批量 K 线接口；盘中主链路用单次请求覆盖整批代码。
def broker_daily_prices(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    return amazingdata_source.fetch_daily_batch(codes, start, end)


def market_spot() -> pd.DataFrame:
    """全市场实时快照：东财失败时回退到新浪批量快照。"""
    try:
        with net.akshare_proxied():
            return ak.stock_zh_a_spot_em()
    except Exception as first_error:  # noqa: BLE001
        try:
            return ak.stock_zh_a_spot()
        except Exception as second_error:  # noqa: BLE001
            raise RuntimeError(
                "all whole-market intraday snapshot sources failed: "
                f"eastmoney={type(first_error).__name__}; "
                f"legacy={type(second_error).__name__}"
            ) from second_error
