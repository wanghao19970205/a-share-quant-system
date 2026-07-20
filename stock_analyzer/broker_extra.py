"""券商(AmazingData)基本面 + 资金面分析。

复用 amazingdata_source 的登录会话（不额外占连接）。仅在券商 SDK 可用且登录成功时有效，
否则整体标记 unavailable，不影响其它模块。

信号（均为经验阈值，仅供参考）：
- 业绩预告 get_profit_notice：净利同比预增/预减 → 基本面前瞻
- 龙虎榜 get_long_hu_bang：最新上榜日净买入/卖出 → 资金异动
- 融资融券 get_margin_detail：融资余额环比 → 杠杆资金加/减仓
- 股东户数 get_holder_num：户数环比 → 筹码集中/分散
- 利润表 get_income：最新营收/归母净利（展示）
"""
from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import amazingdata_source as _ads
from stock_analyzer import data as _data
from stock_analyzer import net


@dataclass
class Signal:
    name: str
    score: int          # +看多 / -看空 / 0 中性
    detail: str
    date: str = ""


@dataclass
class BrokerAnalysis:
    available: bool
    note: str = ""
    signals: list = field(default_factory=list)   # list[Signal]
    revenue_text: str = ""                          # 最新财务展示
    score: float = 0.0
    level: str = "neutral"
    label: str = "基本面/资金面 中性"


def available() -> bool:
    return _ads.available()


def _info():
    if not _ads._ensure_login():
        raise RuntimeError(_ads._last_error or "券商未登录")
    return _ads._ad.InfoData()


def _as_df(r, code):
    """接口返回可能是 df 或 {code: df}，统一取出 df。"""
    if isinstance(r, dict):
        if code in r:
            return r[code]
        return next(iter(r.values())) if r else None
    return r


def _num(v, default=0.0):
    try:
        if isinstance(v, str):
            v = v.strip().replace(",", "").replace("%", "")
            if v in ("", "-", "--", "None", "nan"):
                return default
        f = float(v)
        return default if pd.isna(f) else f
    except Exception:  # noqa: BLE001
        return default


def _fmt_yi(x: float) -> str:
    if abs(x) >= 1e8:
        return f"{x / 1e8:.2f}亿"
    if abs(x) >= 1e4:
        return f"{x / 1e4:.2f}万"
    return f"{x:.0f}"


def _fmt_period(p) -> str:
    s = str(p)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else s


# ------------------------- 各信号 -------------------------
def _sig_profit_notice(info, code) -> Signal | None:
    df = _as_df(info.get_profit_notice([code]), code)
    if df is None or len(df) == 0 or "REPORTING_PERIOD" not in df.columns:
        return None
    row = df.sort_values("REPORTING_PERIOD").iloc[-1]
    cmin, cmax = _num(row.get("P_CHANGE_MIN")), _num(row.get("P_CHANGE_MAX"))
    chg = (cmin + cmax) / 2
    period = _fmt_period(row.get("REPORTING_PERIOD"))
    if chg >= 50:
        score, tag = 2, "大幅预增"
    elif chg > 0:
        score, tag = 1, "预增"
    elif chg <= -50:
        score, tag = -2, "大幅预减"
    elif chg < 0:
        score, tag = -1, "预减"
    else:
        score, tag = 0, "基本持平"
    detail = f"{period} 业绩预告：归母净利同比 {cmin:+.1f}%~{cmax:+.1f}%（{tag}）"
    return Signal("业绩预告", score, detail, str(row.get("ANN_DATE", "")))


def _sig_long_hu_bang(info, code) -> Signal | None:
    df = info.get_long_hu_bang([code])
    df = _as_df(df, code)
    if df is None or len(df) == 0 or "TRADE_DATE" not in df.columns:
        return None
    last_date = df["TRADE_DATE"].max()
    d = df[df["TRADE_DATE"] == last_date]
    buy = d["BUY_AMOUNT"].apply(_num).sum()
    sell = d["SELL_AMOUNT"].apply(_num).sum()
    net = buy - sell
    reason = str(d["REASON_TYPE_NAME"].iloc[0]) if "REASON_TYPE_NAME" in d.columns else ""
    if net > 5e7:
        score = 2
    elif net > 0:
        score = 1
    elif net < -5e7:
        score = -2
    elif net < 0:
        score = -1
    else:
        score = 0
    flow = "净买入" if net >= 0 else "净卖出"
    detail = f"{_fmt_period(last_date)} 上榜（{reason}）：{flow} {_fmt_yi(abs(net))}"
    return Signal("龙虎榜", score, detail, str(last_date))


def _sig_margin(info, code) -> Signal | None:
    df = _as_df(info.get_margin_detail([code]), code)
    if df is None or len(df) < 1 or "BORROW_MONEY_BALANCE" not in df.columns:
        return None
    df = df.sort_values("TRADE_DATE")
    bal = _num(df["BORROW_MONEY_BALANCE"].iloc[-1])
    date = df["TRADE_DATE"].iloc[-1]
    if len(df) >= 2:
        prev = _num(df["BORROW_MONEY_BALANCE"].iloc[-2])
        chg = (bal - prev) / prev * 100 if prev else 0.0
    else:
        chg = 0.0
    if chg > 3:
        score = 2
    elif chg > 0.3:
        score = 1
    elif chg < -3:
        score = -2
    elif chg < -0.3:
        score = -1
    else:
        score = 0
    detail = f"融资余额 {_fmt_yi(bal)}，环比 {chg:+.2f}%（杠杆资金{'加仓' if chg >= 0 else '减仓'}）"
    return Signal("融资融券", score, detail, str(date))


def _sig_holder(info, code) -> Signal | None:
    df = info.get_holder_num([code])
    df = _as_df(df, code)
    if df is None or len(df) < 2 or "HOLDER_NUM" not in df.columns:
        return None
    df = df.sort_values("HOLDER_ENDDATE")
    cur = _num(df["HOLDER_NUM"].iloc[-1])
    prev = _num(df["HOLDER_NUM"].iloc[-2])
    if not prev:
        return None
    chg = (cur - prev) / prev * 100
    # 户数减少=筹码集中(偏多)，增加=分散(偏空)
    if chg <= -10:
        score = 2
    elif chg < 0:
        score = 1
    elif chg >= 10:
        score = -2
    elif chg > 0:
        score = -1
    else:
        score = 0
    end = _fmt_period(df["HOLDER_ENDDATE"].iloc[-1])
    detail = f"股东户数 {cur:,.0f}（截至{end}），环比 {chg:+.1f}%（{'筹码集中' if chg < 0 else '趋于分散'}）"
    return Signal("股东户数", score, detail, str(df["ANN_DT"].iloc[-1]))


def _revenue_text(info, code) -> str:
    try:
        df = _as_df(info.get_income([code]), code)
        if df is None or len(df) == 0 or "REPORTING_PERIOD" not in df.columns:
            return ""
        latest = df["REPORTING_PERIOD"].max()
        d = df[df["REPORTING_PERIOD"] == latest]
        # 同期多行取营收最大者（一般为合并报表）
        d = d.sort_values("TOT_OPERA_REV").iloc[-1]
        rev = _num(d.get("TOT_OPERA_REV"))
        npf = _num(d.get("NET_PRO_EXCL_MIN_INT_INC"))
        eps = _num(d.get("BASIC_EPS"))
        return (f"{_fmt_period(latest)} 报告期：营业总收入 {_fmt_yi(rev)}，"
                f"归母净利 {_fmt_yi(npf)}，EPS {eps:.3f}")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------- 东财数据中心兜底（与训练数据同源） -------------------------
def _quarter_dates(years: int = 2, limit: int | None = None) -> list[str]:
    today = _dt.date.today()
    dates: list[str] = []
    for y in range(today.year, today.year - years - 1, -1):
        for md in ("1231", "0930", "0630", "0331"):
            d = f"{y}{md}"
            if d <= today.strftime("%Y%m%d"):
                dates.append(d)
    return dates[:limit] if limit else dates


def _norm_key(v) -> str:
    s = str(v).strip().lower()
    for ch in (" ", "-", "_", "/", "%", "（", "）", "(", ")"):
        s = s.replace(ch, "")
    return s


def _row_value(row, names: tuple[str, ...], default=""):
    for name in names:
        if name in row.index:
            val = row.get(name)
            if pd.notna(val) and str(val).strip() not in ("", "-", "--", "nan"):
                return val
    norm_cols = {_norm_key(col): col for col in row.index}
    for name in names:
        needle = _norm_key(name)
        for norm_col, col in norm_cols.items():
            if needle and needle in norm_col:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() not in ("", "-", "--", "nan"):
                    return val
    return default


def _code_col(df: pd.DataFrame) -> str | None:
    for col in ("股票代码", "证券代码", "代码", "code"):
        if col in df.columns:
            return col
    return None


def _code6(v) -> str:
    s = str(v).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.split(".", 1)[0]
    return s.zfill(6) if s.isdigit() else s


def _code_rows(df: pd.DataFrame, code6: str) -> pd.DataFrame:
    col = _code_col(df)
    if not col:
        return pd.DataFrame()
    return df[df[col].astype(str).map(_code6) == code6].copy()


@lru_cache(maxsize=16)
def _ak_yjyg(report_date: str) -> pd.DataFrame:
    with net.akshare_proxied():
        df = ak.stock_yjyg_em(date=report_date)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@lru_cache(maxsize=16)
def _ak_yjbb(report_date: str) -> pd.DataFrame:
    with net.akshare_proxied():
        df = ak.stock_yjbb_em(date=report_date)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@lru_cache(maxsize=16)
def _ak_lrb(report_date: str) -> pd.DataFrame:
    with net.akshare_proxied():
        df = ak.stock_lrb_em(date=report_date)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@lru_cache(maxsize=32)
def _ak_profit_notice(code6: str) -> Signal | None:
    for report_date in _quarter_dates(2, limit=6):
        try:
            df = _ak_yjyg(report_date)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        sub = _code_rows(df, code6)
        if sub.empty:
            continue
        row = sub.iloc[0]
        chg = _num(_row_value(row, ("业绩变动幅度", "变动幅度", "预测幅度")))
        notice_type = str(_row_value(row, ("预告类型", "类型"), ""))
        metric = str(_row_value(row, ("预测指标", "指标"), "业绩预告"))
        if chg >= 50 or any(k in notice_type for k in ("预增", "扭亏", "略增")):
            score = 2 if chg >= 50 or "扭亏" in notice_type else 1
        elif chg <= -50 or any(k in notice_type for k in ("预减", "首亏", "续亏", "略减")):
            score = -2 if chg <= -50 or "首亏" in notice_type else -1
        else:
            score = 0
        detail = f"{report_date} {metric}：{notice_type or '未分类'}，业绩变动幅度 {chg:+.2f}%"
        reason = str(_row_value(row, ("业绩变动原因", "变动原因"), ""))
        if reason and reason != "nan":
            detail += f"；{reason[:90]}"
        return Signal("业绩预告(东财数据中心)", score, detail, str(_row_value(row, ("公告日期", "预告公告日"), "")))
    return None


def _financial_row(code6: str):
    for report_date in _quarter_dates(2, limit=6):
        try:
            df = _ak_yjbb(report_date)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        sub = _code_rows(df, code6)
        if sub.empty:
            continue
        return report_date, sub.iloc[0]
    return "", None


@lru_cache(maxsize=32)
def _ak_financial_report(code6: str) -> Signal | None:
    report_date, row = _financial_row(code6)
    if row is None:
        return None
    npf_yoy = _num(_row_value(row, ("净利润同比", "净利润增长率", "归属净利润同比")))
    rev_yoy = _num(_row_value(row, ("营业收入同比", "营收同比", "营业总收入同比", "营业收入增长率")))
    roe = _num(_row_value(row, ("净资产收益率", "ROE")))
    eps = _num(_row_value(row, ("每股收益", "EPS")))
    if npf_yoy >= 50 or (npf_yoy > 0 and rev_yoy > 0 and roe >= 8):
        score = 2 if npf_yoy >= 50 else 1
    elif npf_yoy <= -50 or (npf_yoy < 0 and rev_yoy < 0):
        score = -2 if npf_yoy <= -50 else -1
    else:
        score = 0
    detail = (f"{report_date} 业绩报表：净利润同比 {npf_yoy:+.2f}%"
              f"，营收同比 {rev_yoy:+.2f}%")
    if roe:
        detail += f"，ROE {roe:.2f}%"
    if eps:
        detail += f"，EPS {eps:.3f}"
    return Signal("业绩报表(东财数据中心)", score, detail, str(_row_value(row, ("最新公告日期", "公告日期"), "")))


@lru_cache(maxsize=32)
def _ak_financial_text(code6: str) -> str:
    report_date, row = _financial_row(code6)
    if row is None:
        return ""
    rev = _num(_row_value(row, ("营业总收入", "营业收入", "营收")))
    npf = _num(_row_value(row, ("净利润", "归属净利润", "归母净利润")))
    rev_yoy = _num(_row_value(row, ("营业收入同比", "营收同比", "营业总收入同比", "营业收入增长率")))
    npf_yoy = _num(_row_value(row, ("净利润同比", "净利润增长率", "归属净利润同比")))
    eps = _num(_row_value(row, ("每股收益", "EPS")))
    text = (f"{report_date} 报告期：营业总收入 {_fmt_yi(rev)}（同比 {rev_yoy:+.2f}%），"
            f"净利润 {_fmt_yi(npf)}（同比 {npf_yoy:+.2f}%）")
    if eps:
        text += f"，EPS {eps:.3f}"
    return text


def _income_row(code6: str):
    for report_date in _quarter_dates(2, limit=6):
        try:
            df = _ak_lrb(report_date)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        sub = _code_rows(df, code6)
        if sub.empty:
            continue
        return report_date, sub.iloc[0]
    return "", None


@lru_cache(maxsize=32)
def _ak_income_signal(code6: str) -> Signal | None:
    report_date, row = _income_row(code6)
    if row is None:
        return None
    rev_yoy = _num(_row_value(row, ("营业总收入同比", "营业收入同比", "营收同比")))
    npf_yoy = _num(_row_value(row, ("净利润同比", "净利润增长率", "归属净利润同比")))
    rev = _num(_row_value(row, ("营业总收入", "营业收入", "营收")))
    npf = _num(_row_value(row, ("净利润", "归属净利润", "归母净利润")))
    if npf_yoy >= 50 or (npf > 0 and npf_yoy > 0 and rev_yoy > 0):
        score = 2 if npf_yoy >= 50 else 1
    elif npf_yoy <= -50 or (npf < 0 and npf_yoy < 0):
        score = -2 if npf_yoy <= -50 else -1
    else:
        score = 0
    detail = (f"{report_date} 利润表：营业总收入 {_fmt_yi(rev)}（同比 {rev_yoy:+.2f}%），"
              f"净利润 {_fmt_yi(npf)}（同比 {npf_yoy:+.2f}%）")
    return Signal("利润表(东财数据中心)", score, detail, str(_row_value(row, ("最新公告日期", "公告日期"), "")))


@lru_cache(maxsize=32)
def _ak_income_text(code6: str) -> str:
    report_date, row = _income_row(code6)
    if row is None:
        return ""
    rev = _num(_row_value(row, ("营业总收入", "营业收入", "营收")))
    npf = _num(_row_value(row, ("净利润", "归属净利润", "归母净利润")))
    rev_yoy = _num(_row_value(row, ("营业总收入同比", "营业收入同比", "营收同比")))
    npf_yoy = _num(_row_value(row, ("净利润同比", "净利润增长率", "归属净利润同比")))
    return (f"{report_date} 报告期：营业总收入 {_fmt_yi(rev)}（同比 {rev_yoy:+.2f}%），"
            f"净利润 {_fmt_yi(npf)}（同比 {npf_yoy:+.2f}%）")


@lru_cache(maxsize=32)
def _ak_lhb(code6: str) -> Signal | None:
    end = _dt.date.today()
    start = end - _dt.timedelta(days=180)
    try:
        with net.akshare_proxied():
            df = ak.stock_lhb_detail_em(start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "代码" not in df.columns:
        return None
    sub = df[df["代码"].astype(str).str.zfill(6) == code6].copy()
    if sub.empty:
        return None
    sub["上榜日"] = pd.to_datetime(sub["上榜日"], errors="coerce")
    row = sub.sort_values("上榜日").iloc[-1]
    net_amt = _num(row.get("龙虎榜净买额"))
    reason = str(row.get("上榜原因", ""))
    if net_amt > 5e7:
        score = 2
    elif net_amt > 0:
        score = 1
    elif net_amt < -5e7:
        score = -2
    elif net_amt < 0:
        score = -1
    else:
        score = 0
    flow = "净买入" if net_amt >= 0 else "净卖出"
    date_s = row.get("上榜日")
    date_s = date_s.strftime("%Y-%m-%d") if pd.notna(date_s) else ""
    detail = f"{date_s} 上榜（{reason[:32]}）：{flow} {_fmt_yi(abs(net_amt))}"
    return Signal("龙虎榜(东财数据中心)", score, detail, date_s)


def _free_fallback(symbol: str, note: str = "") -> BrokerAnalysis:
    code6 = _data._normalize_symbol(symbol)
    signals: list[Signal] = []
    rev = ""

    try:
        sig = _ak_financial_report(code6)
        if sig is not None:
            signals.append(sig)
            rev = _ak_financial_text(code6)
    except Exception:  # noqa: BLE001
        pass

    if not signals:
        for fn in (_ak_income_signal, _ak_profit_notice):
            try:
                sig = fn(code6)
                if sig is not None:
                    signals.append(sig)
            except Exception:  # noqa: BLE001
                continue
        rev = _ak_income_text(code6)

    if not signals and not rev:
        return BrokerAnalysis(False, note=(note or "券商和东财数据中心均未取到可用基本面/资金面数据"))
    score = round(sum(s.score for s in signals) / len(signals), 2) if signals else 0.0
    if score >= 0.6:
        level, label = "bullish", "基本面/资金面 偏多（东财数据中心）"
    elif score <= -0.6:
        level, label = "bearish", "基本面/资金面 偏空（东财数据中心）"
    else:
        level, label = "neutral", "基本面/资金面 中性（东财数据中心）"
    src_note = "已使用训练同源的 AKShare 东财数据中心快速兜底（优先业绩报表，必要时补利润表/业绩预告）。"
    if note:
        src_note = f"{note}；{src_note}"
    return BrokerAnalysis(True, note=src_note, signals=signals, revenue_text=rev,
                          score=score, level=level, label=label)


def clear_cache() -> None:
    """Clear broker and Eastmoney datacenter caches used by the UI retry buttons."""
    for fn in (
        analyze,
        _ak_profit_notice,
        _ak_financial_report,
        _ak_financial_text,
        _ak_income_signal,
        _ak_income_text,
        _ak_lhb,
        _ak_yjyg,
        _ak_yjbb,
        _ak_lrb,
    ):
        fn.cache_clear()


# ------------------------- 综合 -------------------------
def _broker_once(symbol: str) -> tuple[list[Signal], str]:
    code = _ads._to_broker_code(symbol)

    def _run() -> tuple[list[Signal], str]:
        signals: list[Signal] = []
        info = _info()
        for fn in (_sig_profit_notice, _sig_long_hu_bang, _sig_margin, _sig_holder):
            try:
                s = fn(info, code)
                if s is not None:
                    signals.append(s)
            except Exception:  # noqa: BLE001 单项失败不影响其余
                continue
        return signals, _revenue_text(info, code)

    # 无锁 + 整体超时：一次批量券商调用挂起时最多等待 _BROKER_TIMEOUT，
    # 超时抛 TimeoutError，由 analyze 立即走东财数据中心兜底，不再无限等待。
    return _ads.sdk_call(_run, timeout=_ads._BROKER_TIMEOUT)


@lru_cache(maxsize=64)
def analyze(symbol: str, retry: int = 6, retry_interval: float = 2.0) -> BrokerAnalysis:
    if not available():
        return _free_fallback(symbol, "券商 SDK 不可用或未配置")
    attempts = max(1, int(retry or 1))
    signals: list[Signal] = []
    rev = ""
    for i in range(attempts):
        try:
            signals, rev = _broker_once(symbol)
        except TimeoutError:
            # SDK 挂起无返回，重试无益，直接走兜底。
            return _free_fallback(symbol, "券商接口超时无响应，已跳过重试")
        except Exception as e:  # noqa: BLE001
            return _free_fallback(symbol, f"券商未登录或请求失败：{e}")
        if signals or rev:
            break
        if i < attempts - 1:
            time.sleep(max(0.0, float(retry_interval or 0.0)))

    if not signals:
        note = f"券商 AmazingData 已返回，但连续重试 {attempts} 次仍未取到可用的基本面/资金面信号"
        fallback = _free_fallback(symbol, note)
        if rev and not fallback.revenue_text:
            fallback.revenue_text = rev
        return fallback

    score = round(sum(s.score for s in signals) / len(signals), 2)
    if score >= 0.6:
        level, label = "bullish", "基本面/资金面 偏多"
    elif score <= -0.6:
        level, label = "bearish", "基本面/资金面 偏空"
    else:
        level, label = "neutral", "基本面/资金面 中性"
    note = f"使用券商 AmazingData；请求最多重试 {attempts} 次、间隔 {float(retry_interval or 0):.1f}s。券商已返回有效信号，未触发东财数据中心兜底。"
    return BrokerAnalysis(True, signals=signals, revenue_text=rev,
                          score=score, level=level, label=label, note=note)
