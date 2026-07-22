"""量化选股 · 因子工程。

从 ``quant_data`` parquet 仓读取价格、估值、财报、事件数据，生成按
``code/date`` 对齐的训练面板：
- 价量因子：动量、反转、波动、换手/成交额变化、均线乖离等。
- 估值因子：PE/PB/PS/PEG、市值及其对数。
- 财务因子：业绩报表/利润表里可匹配到的 ROE、毛利率、收入/利润增速等，按 ann_date 做 point-in-time 对齐。
- 事件因子：龙虎榜、大宗交易、业绩预告、两融标的等滚动计数/标记。
- 截面处理：去极值、zscore 标准化、可按行业/市值做中性化。
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from quant import config, warehouse


_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]
_PANEL_WORKERS = 12
_PRICE_PROCESS_MIN_CODES = 50
_PRICE_PROCESS_WORKERS = min(
    max(int(os.environ.get("PANEL_PRICE_PROCESS_WORKERS", "8") or 8), 1),
    os.cpu_count() or 1,
)


def _technical_rule_score(df: pd.DataFrame) -> pd.Series:
    """Vectorized version of the legacy technical advisor score."""
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce")
    turnover = pd.to_numeric(df.get("turnover"), errors="coerce")
    score = pd.Series(0.0, index=df.index)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    score += ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype(float) * 2
    score -= ((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)).astype(float) * 2
    score += (close > ma20).where(ma20.notna(), False).astype(float)
    score -= (close <= ma20).where(ma20.notna(), False).astype(float)

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    score += ((j < 0) | (k < 20)).where(k.notna(), False).astype(float) * 2
    score -= ((j > 100) | (k > 80)).where(k.notna(), False).astype(float) * 2
    score += ((k.shift(1) < d.shift(1)) & (k > d)).astype(float)
    score -= ((k.shift(1) > d.shift(1)) & (k < d)).astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi6 = 100 - 100 / (1 + gain.ewm(alpha=1 / 6, adjust=False).mean() / loss.ewm(alpha=1 / 6, adjust=False).mean().replace(0, np.nan))
    rsi12 = 100 - 100 / (1 + gain.ewm(alpha=1 / 12, adjust=False).mean() / loss.ewm(alpha=1 / 12, adjust=False).mean().replace(0, np.nan))
    rsi6 = rsi6.fillna(100)
    rsi12 = rsi12.fillna(100)
    score += (rsi6 < 20).astype(float) * 2
    score -= (rsi6 > 80).astype(float) * 2
    score += ((rsi6 >= 20) & (rsi6 <= 80) & (rsi6 > rsi12)).astype(float)
    score -= ((rsi6 >= 20) & (rsi6 <= 80) & (rsi6 <= rsi12)).astype(float)

    bias6 = (close - close.rolling(6).mean()) / close.rolling(6).mean() * 100
    score += (bias6 < -8).astype(float) * 2
    score += ((bias6 >= -8) & (bias6 < -4)).astype(float)
    score -= (bias6 > 8).astype(float) * 2
    score -= ((bias6 <= 8) & (bias6 > 4)).astype(float)

    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    score += ((dif.shift(1) < dea.shift(1)) & (dif > dea)).astype(float) * 2
    score -= ((dif.shift(1) > dea.shift(1)) & (dif < dea)).astype(float) * 2
    score += (dif > 0).astype(float)
    score -= (dif <= 0).where(dif.notna(), False).astype(float)
    score += (hist > hist.shift(1)).astype(float)
    score -= (hist <= hist.shift(1)).where(hist.shift(1).notna(), False).astype(float)

    vol_ratio = volume / volume.rolling(5).mean()
    price_up = close > close.shift(1)
    obv = (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
    score += ((vol_ratio > 1.5) & price_up).astype(float) * 2
    score -= ((vol_ratio > 1.5) & (~price_up)).astype(float) * 2
    score += ((vol_ratio < 0.7) & (~price_up)).astype(float)
    score += (obv > obv.shift(5)).astype(float)
    score -= (obv <= obv.shift(5)).where(obv.shift(5).notna(), False).astype(float)

    turnover_ma5 = turnover.rolling(5).mean()
    score -= (turnover > 15).astype(float)
    score += ((turnover >= 3) & (turnover <= 10) & (turnover > turnover_ma5)).astype(float)
    return score.replace([np.inf, -np.inf], np.nan)


def _read_codes(limit: int = 0) -> list[str]:
    if not os.path.isdir(config.PRICE_DIR):
        return []
    codes = sorted(p[:-8] for p in os.listdir(config.PRICE_DIR) if p.endswith(".parquet"))
    return codes[:limit] if limit else codes


def _num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False),
        errors="coerce",
    )


def _first_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for pat in patterns:
        rx = re.compile(pat)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    return None


def _price_factors(
    code: str,
    start_date: pd.Timestamp | None = None,
    warmup_rows: int = 0,
) -> pd.DataFrame:
    # EWM-based indicators carry recursive state, so compute from full history and
    # only trim the emitted rows. This preserves exact full-panel factor values.
    del warmup_rows
    df = warehouse.load_price(code)
    if df.empty:
        return df
    df = df.copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in _PRICE_COLS:
        if c in df.columns:
            df[c] = _num(df[c])

    close = df["close"]
    df["ret_1d"] = close.pct_change()
    for n in (3, 5, 10, 20, 60):
        df[f"ret_{n}d"] = close.pct_change(n)
        df[f"ma_gap_{n}"] = close / close.rolling(n).mean() - 1
    for n in (5, 10, 20):
        df[f"volatility_{n}"] = df["ret_1d"].rolling(n).std()
        if "volume" in df.columns:
            df[f"volume_ratio_{n}"] = df["volume"] / df["volume"].rolling(n).mean() - 1
    for n in (10, 20, 60):
        roll_high = close.rolling(n).max()
        roll_low = close.rolling(n).min()
        df[f"drawdown_{n}"] = close / roll_high - 1
        df[f"range_pos_{n}"] = (close - roll_low) / (roll_high - roll_low).replace(0, np.nan)
        df[f"ret_vol_adj_{n}"] = df[f"ret_{n}d"] / df["ret_1d"].rolling(n).std().replace(0, np.nan)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]
    df["macd_hist_chg_3"] = df["macd_hist"] - df["macd_hist"].shift(3)
    if "high" in df.columns and "low" in df.columns:
        df["intraday_range"] = (df["high"] - df["low"]) / close.replace(0, np.nan)
    if "turnover" in df.columns:
        df["turnover_chg_5"] = df["turnover"].pct_change(5)
    if "amount" in df.columns:
        df["amount_chg_5"] = df["amount"].pct_change(5)
    df["rule_score"] = _technical_rule_score(df)
    df["rule_score_chg_5"] = df["rule_score"] - df["rule_score"].shift(5)
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()

    return df[["code", "date", "close"] + [c for c in df.columns if c not in {"code", "date", "close"}]]


def _valuation_factors(code: str, start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = warehouse.load_valuation(code)
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    keep = ["code", "date"]
    for c in ("pe_ttm", "pe_static", "pb", "ps", "peg", "pcf", "mv_total", "mv_float"):
        if c in df.columns:
            df[c] = _num(df[c])
            keep.append(c)
    for c in ("mv_total", "mv_float"):
        if c in df.columns:
            df[f"log_{c}"] = np.log(df[c].where(df[c] > 0))
            keep.append(f"log_{c}")
    return df[keep].drop_duplicates(["code", "date"], keep="last")


def _asof_report_factor(name: str, prefix: str, codes: set[str]) -> pd.DataFrame:
    df = warehouse.load(name)
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    df = df[df["code"].astype(str).isin(codes)].copy()
    if df.empty:
        return pd.DataFrame()
    if "ann_date" not in df.columns:
        df["ann_date"] = df.get("report_date")
    df["date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df = df.dropna(subset=["code", "date"])

    aliases = {
        "roe": ["净资产收益率", "ROE"],
        "gross_margin": ["销售毛利率", "毛利率"],
        "net_profit_yoy": ["净利润同比", "净利润增长率", "归属净利润同比"],
        "revenue_yoy": ["营业收入同比", "营收同比", "营业收入增长率"],
        "eps": ["每股收益", "EPS"],
        "net_profit": ["净利润", "归属净利润"],
        "revenue": ["营业收入", "营收"],
    }
    out = df[["code", "date"]].copy()
    for key, pats in aliases.items():
        col = _first_col(df, pats)
        if col:
            out[f"{prefix}_{key}"] = _num(df[col])
    value_cols = [c for c in out.columns if c not in {"code", "date"}]
    if not value_cols:
        return pd.DataFrame()
    out = out.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    return out


def _merge_asof_panel(base: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    if factor.empty:
        return base
    parts = []
    factor_groups = {
        str(code): group.drop(columns=["code"]).sort_values("date")
        for code, group in factor.groupby("code", sort=False)
    }
    for code, group in base.groupby("code", sort=False):
        matched = factor_groups.get(str(code))
        if matched is None or matched.empty:
            parts.append(group)
            continue
        parts.append(
            pd.merge_asof(
                group.sort_values("date"),
                matched,
                on="date",
                direction="backward",
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else base


def _event_counts(name: str, codes: set[str], prefix: str) -> pd.DataFrame:
    df = warehouse.load(name)
    if df.empty or "code" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    df = df[df["code"].astype(str).isin(codes)].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out = df.groupby(["code", "date"]).size().rename(f"{prefix}_cnt_1d").reset_index()
    return out


def _forecast_events(codes: set[str]) -> pd.DataFrame:
    df = warehouse.load("performance_forecast")
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    df = df[df["code"].astype(str).isin(codes)].copy()
    if df.empty:
        return pd.DataFrame()
    if "ann_date" not in df.columns:
        df["ann_date"] = df.get("report_date")
    df["date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    out = df[["code", "date"]].copy()
    if "业绩变动幅度" in df.columns:
        out["forecast_yoy"] = _num(df["业绩变动幅度"])
    if "预告类型" in df.columns:
        good = df["预告类型"].astype(str).str.contains("预增|略增|扭亏|续盈", regex=True)
        bad = df["预告类型"].astype(str).str.contains("预减|略减|首亏|续亏|增亏", regex=True)
        out["forecast_signal"] = np.select([good, bad], [1, -1], default=0)
    out = out.dropna(subset=["date"])
    if len(out.columns) <= 2:
        return pd.DataFrame()
    return out.groupby(["code", "date"], as_index=False).mean(numeric_only=True)


def _margin_underlying(codes: set[str]) -> pd.DataFrame:
    df = warehouse.load("margin_underlying_szse")
    if df.empty or "code" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    df = df[df["code"].astype(str).isin(codes)].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out = df[["code", "date"]].copy()
    if "当日可融资" in df.columns:
        out["margin_buyable"] = (df["当日可融资"].astype(str).str.upper() == "Y").astype(float)
    if "当日可融券" in df.columns:
        out["margin_shortable"] = (df["当日可融券"].astype(str).str.upper() == "Y").astype(float)
    return out.drop_duplicates(["code", "date"], keep="last")


def _market_of_code(code: str) -> str:
    code = str(code)
    if code.startswith(("60", "68", "90")):
        return "SH"
    if code.startswith(("00", "30", "20")):
        return "SZ"
    if code.startswith(("43", "83", "87", "88")):
        return "BJ"
    return "UNK"


def _filter_codes_by_price_rows(codes: list[str], min_price_rows: int = 0) -> list[str]:
    if min_price_rows <= 0:
        return codes
    kept = []
    for code in codes:
        p = os.path.join(config.PRICE_DIR, f"{code}.parquet")
        if not os.path.exists(p):
            continue
        try:
            if len(pd.read_parquet(p, columns=["date"])) >= min_price_rows:
                kept.append(code)
        except Exception:  # noqa: BLE001
            continue
    return kept


def _factor_subset(factor: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    if factor.empty or "code" not in factor.columns:
        return factor
    return factor[factor["code"].astype(str).isin(codes)].copy()


def _price_factor_job(args: tuple[str, pd.Timestamp | None, int]) -> pd.DataFrame:
    return _price_factors(*args)


def _price_factor_parts(
    codes: list[str],
    start_date: pd.Timestamp | None,
    warmup_rows: int,
) -> list[pd.DataFrame]:
    jobs = [(code, start_date, warmup_rows) for code in codes]
    if len(codes) < _PRICE_PROCESS_MIN_CODES or _PRICE_PROCESS_WORKERS <= 1:
        with ThreadPoolExecutor(max_workers=_PANEL_WORKERS) as executor:
            return list(executor.map(_price_factor_job, jobs))
    with ProcessPoolExecutor(max_workers=_PRICE_PROCESS_WORKERS) as executor:
        return list(executor.map(_price_factor_job, jobs, chunksize=5))


def build_panel(codes: list[str] | None = None, limit: int = 0, horizon: int = 5,
                min_price_rows: int = 0, output_start: pd.Timestamp | None = None,
                warmup_rows: int = 260,
                shared_factors: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """生成训练面板。返回包含 ``target_ret_{horizon}d`` 的 DataFrame。

    ``output_start`` limits emitted rows for incremental refreshes. Price factors
    still use full history so recursive EWM indicators remain exactly stable.
    """
    panel_started = time.perf_counter()
    stage_started = panel_started
    codes = codes or _read_codes(0)
    codes = _filter_codes_by_price_rows(codes, min_price_rows)
    if limit and codes:
        codes = codes[:limit]
    factor_start = pd.Timestamp(output_start) if output_start is not None else None
    price_parts = _price_factor_parts(codes, factor_start, warmup_rows)
    price_parts = [p for p in price_parts if p is not None and not p.empty]
    if not price_parts:
        return pd.DataFrame()
    panel = pd.concat(price_parts, ignore_index=True).sort_values(["code", "date"])
    panel["market_cat"] = panel["code"].astype(str).map(_market_of_code)
    codes_set = set(panel["code"].astype(str))
    print(
        f"[panel:raw:timing] stage=price seconds={time.perf_counter() - stage_started:.2f} "
        f"codes={len(codes_set)} rows={len(panel)}",
        flush=True,
    )

    stage_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=_PANEL_WORKERS) as executor:
        val_parts = list(executor.map(
            lambda code: _valuation_factors(code, factor_start),
            codes,
        ))
    val_parts = [p for p in val_parts if p is not None and not p.empty]
    if val_parts:
        val = pd.concat(val_parts, ignore_index=True)
        panel = panel.merge(val, on=["code", "date"], how="left", suffixes=("", "_val"))
    print(
        f"[panel:raw:timing] stage=valuation seconds={time.perf_counter() - stage_started:.2f}",
        flush=True,
    )

    stage_started = time.perf_counter()
    shared_factors = shared_factors or {}
    for name, prefix in (("financial_yjbb", "yjbb"), ("income", "income"), ("cashflow", "cashflow"), ("balance", "balance")):
        factor = shared_factors.get(name)
        if factor is None:
            factor = _asof_report_factor(name, prefix, codes_set)
        else:
            factor = _factor_subset(factor, codes_set)
        panel = _merge_asof_panel(panel, factor)
    print(
        f"[panel:raw:timing] stage=financial_asof seconds={time.perf_counter() - stage_started:.2f}",
        flush=True,
    )

    stage_started = time.perf_counter()
    forecast = shared_factors.get("performance_forecast")
    if forecast is None:
        forecast = _forecast_events(codes_set)
    else:
        forecast = _factor_subset(forecast, codes_set)
    margin = shared_factors.get("margin_underlying_szse")
    if margin is None:
        margin = _margin_underlying(codes_set)
    else:
        margin = _factor_subset(margin, codes_set)
    for asof_ev in (forecast, margin):
        panel = _merge_asof_panel(panel, asof_ev)

    event_frames = [
        shared_factors.get("block_trades"),
        shared_factors.get("lhb"),
    ]
    if event_frames[0] is None:
        event_frames[0] = _event_counts("block_trades", codes_set, "block_trade")
    else:
        event_frames[0] = _factor_subset(event_frames[0], codes_set)
    if event_frames[1] is None:
        event_frames[1] = _event_counts("lhb", codes_set, "lhb")
    else:
        event_frames[1] = _factor_subset(event_frames[1], codes_set)
    for ev in event_frames:
        if not ev.empty:
            panel = panel.merge(ev, on=["code", "date"], how="left")
    print(
        f"[panel:raw:timing] stage=events seconds={time.perf_counter() - stage_started:.2f}",
        flush=True,
    )

    stage_started = time.perf_counter()
    event_cols = [c for c in panel.columns if c.endswith("_cnt_1d")]
    for c in event_cols:
        panel[c] = panel[c].fillna(0)
        for n in (5, 20):
            panel[f"{c[:-3]}{n}d"] = panel.groupby("code")[c].transform(lambda s: s.rolling(n, min_periods=1).sum())

    panel["target_ret_%dd" % horizon] = panel.groupby("code")["close"].shift(-horizon) / panel["close"] - 1

    # 更贴近实盘的成交口径：T 日出信号 -> T+1 开盘买入 -> 持有 horizon 个交易日 -> T+1+horizon 开盘卖出。
    # 同时标记 T+1 是否可买入（一字涨停当日买不进）。这些列含未来信息，仅供回测成交用，
    # 已在 feature_columns 中排除，不会作为训练特征。
    if "open" in panel.columns:
        g = panel.groupby("code", sort=False)
        entry_open = g["open"].shift(-1)                 # 次日开盘买入价
        exit_open = g["open"].shift(-(1 + horizon))      # 持有 horizon 日后的开盘卖出价
        panel["entry_open_next"] = entry_open
        panel["exit_open_h"] = exit_open
        panel["open_ret_%dd" % horizon] = exit_open / entry_open - 1
        if "high" in panel.columns and "low" in panel.columns:
            nxt_high = g["high"].shift(-1)
            nxt_low = g["low"].shift(-1)
            nxt_close = g["close"].shift(-1)
            one_word_up = (nxt_high == nxt_low) & (nxt_close > panel["close"])
            panel["buyable_next"] = (~one_word_up.fillna(False)) & entry_open.notna()
        else:
            panel["buyable_next"] = entry_open.notna()

    panel = panel.replace([np.inf, -np.inf], np.nan)
    if factor_start is not None:
        panel = panel[panel["date"] >= factor_start].copy()
    print(
        f"[panel:raw:timing] stage=targets seconds={time.perf_counter() - stage_started:.2f} "
        f"total_seconds={time.perf_counter() - panel_started:.2f}",
        flush=True,
    )
    return panel.reset_index(drop=True)


def categorical_columns(panel: pd.DataFrame) -> list[str]:
    cols = []
    for c in ("market_cat", "industry"):
        if c in panel.columns:
            cols.append(c)
    return cols


def feature_columns(panel: pd.DataFrame, horizon: int = 5) -> list[str]:
    banned = {"code", "date", "target_ret_%dd" % horizon}
    banned |= {"open", "high", "low", "close", "volume", "amount"}
    # 回测成交口径列含未来信息，禁止作为特征
    banned |= {"entry_open_next", "exit_open_h", "buyable_next", "open_ret_%dd" % horizon}
    return [c for c in panel.columns if c not in banned and pd.api.types.is_numeric_dtype(panel[c])]


def _feature_kind(name: str) -> str:
    if name.startswith("cross__"):
        return "cross_continuous"
    if name.endswith("_bin") or "_q" in name:
        return "discretized_onehot"
    if name.startswith("cat_"):
        return "categorical_onehot"
    if name.startswith("rule_score"):
        return "technical_rule_score"
    if name.startswith(("ret_", "ma_gap_", "volatility_", "volume_ratio_", "drawdown_", "range_pos_", "ret_vol_adj_", "macd_")) or name in {"intraday_range", "turnover_chg_5", "amount_chg_5"}:
        return "price_volume_continuous"
    if name in {"pe_ttm", "pe_static", "pb", "ps", "peg", "pcf", "mv_total", "mv_float", "log_mv_total", "log_mv_float"}:
        return "valuation_continuous"
    if name.startswith(("yjbb_", "income_", "cashflow_", "balance_")):
        return "financial_continuous"
    if name.startswith(("block_trade_", "lhb_", "forecast_", "margin_")):
        return "event_or_margin"
    return "numeric_continuous"


def winsorize_cross_section(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        ql = out.groupby("date")[c].transform(lambda s: s.quantile(lower))
        qh = out.groupby("date")[c].transform(lambda s: s.quantile(upper))
        out[c] = out[c].clip(ql, qh)
    return out


def zscore_cross_section(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mean = out.groupby("date")[c].transform("mean")
        std = out.groupby("date")[c].transform("std").replace(0, np.nan)
        out[c] = (out[c] - mean) / std
    return out


def neutralize_cross_section(df: pd.DataFrame, cols: list[str], neutral_cols: list[str] | None = None) -> pd.DataFrame:
    """按交易日做线性中性化。默认用 log_mv_total，若有 industry 列也会加入行业哑变量。"""
    neutral_cols = neutral_cols or (["log_mv_total"] if "log_mv_total" in df.columns else [])
    out = df.copy()
    if not neutral_cols and "industry" not in out.columns:
        return out
    for _, idx in out.groupby("date").groups.items():
        g = out.loc[idx]
        xs = []
        if neutral_cols:
            xs.append(g[neutral_cols].apply(pd.to_numeric, errors="coerce"))
        if "industry" in g.columns:
            xs.append(pd.get_dummies(g["industry"], prefix="ind", dummy_na=True, dtype=float))
        if not xs:
            continue
        x = pd.concat(xs, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x.insert(0, "const", 1.0)
        xmat = x.to_numpy(dtype=float)
        for c in cols:
            y = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y)
            if ok.sum() <= xmat.shape[1] + 2:
                continue
            beta = np.linalg.lstsq(xmat[ok], y[ok], rcond=None)[0]
            resid = y - xmat @ beta
            out.loc[idx, c] = resid
    return out


def discretize_cross_section(df: pd.DataFrame, cols: list[str], q: int = 5) -> pd.DataFrame:
    """连续值按交易日分位数离散化，并展开为 one-hot。"""
    frames = []
    idx = df.index
    for c in cols:
        bins = pd.Series(index=idx, dtype="float64")
        for _, gidx in df.groupby("date").groups.items():
            s = pd.to_numeric(df.loc[gidx, c], errors="coerce")
            if s.notna().sum() < q or s.nunique(dropna=True) < 2:
                continue
            try:
                b = pd.qcut(s.rank(method="first"), q=q, labels=False, duplicates="drop")
            except ValueError:
                continue
            bins.loc[gidx] = b.astype(float)
        dummies = pd.get_dummies(bins.astype("Int64"), prefix=f"{c}_q", dummy_na=False, dtype=float)
        frames.append(dummies)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=idx)


def onehot_categories(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frames = []
    for c in cols:
        dummies = pd.get_dummies(df[c].astype("string").fillna("UNK"), prefix=f"cat_{c}", dtype=float)
        frames.append(dummies)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)


def summarize_features(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"feature": c, "kind": _feature_kind(c)} for c in cols])


def prepare_features(panel: pd.DataFrame, horizon: int = 5, neutralize: bool = True,
                     add_discrete: bool = True, add_onehot: bool = True,
                     discrete_q: int = 5, drop_target_na: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """去极值、标准化、中性化后返回可训练样本和特征列。"""
    continuous = feature_columns(panel, horizon)
    cats = categorical_columns(panel)
    out = winsorize_cross_section(panel, continuous)
    if neutralize:
        out = neutralize_cross_section(out, continuous)
    out = zscore_cross_section(out, continuous)

    extra_parts = []
    if add_discrete and continuous:
        extra_parts.append(discretize_cross_section(out, continuous, q=discrete_q))
    if add_onehot and cats:
        extra_parts.append(onehot_categories(panel, cats))
    if extra_parts:
        extras = pd.concat(extra_parts, axis=1)
        out = pd.concat([out, extras], axis=1)

    encoded = []
    for part in extra_parts:
        encoded.extend(part.columns.tolist())
    feats = continuous + encoded
    target = "target_ret_%dd" % horizon
    keep = ["code", "date", target] + feats
    out = out[keep]
    if drop_target_na:
        out = out.dropna(subset=[target])
    return out.reset_index(drop=True), feats


def save_panel(name: str = "factor_panel", limit: int = 0, horizon: int = 5) -> pd.DataFrame:
    panel = build_panel(limit=limit, horizon=horizon)
    if panel.empty:
        return panel
    prepared, feats = prepare_features(panel, horizon=horizon)
    warehouse.save(name, prepared)
    warehouse.save(f"{name}_features", summarize_features(feats))
    return prepared


def main():
    import argparse

    ap = argparse.ArgumentParser(description="生成量化因子训练面板")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--name", default="factor_panel")
    args = ap.parse_args()
    df = save_panel(args.name, args.limit, args.horizon)
    print(f"保存 {args.name}: {len(df)} 行, {len(df.columns)} 列")


if __name__ == "__main__":
    main()
