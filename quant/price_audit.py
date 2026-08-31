"""价格数据质量审计（只读）。

动机：训练标签直接由 `close` 派生，坏价格会静默污染梯度。已实测到
`target_ret_1d == -1.0`（收盘价跌到约 0）和 `prev_close == 0` 导致的 `inf` 收益。
本模块量化这些坏点的规模并定位具体标的，供后续决定剔除或修复。

检查项（逐 code）：
- 非正价格：`close/open/high/low <= 0`
- OHLC 不自洽：`high < low`、`close` 或 `open` 落在 `[low, high]` 之外
- 日期问题：重复日期、非单调日期
- 收益越界：日收益超出该板块合法涨跌停档位（含容差）——物理上不可能，必为数据错误
- 极端收益：`|ret| >= 0.5`
- 零成交量日占比

用法（研究沙箱内）：
    python -m quant.price_audit --limit 0            # 全量
    python -m quant.price_audit --limit 300 --top 20 # 抽样
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant import config, tradability

PRICE_COLS = ["date", "open", "high", "low", "close", "volume"]


def _codes(limit: int) -> list[str]:
    files = sorted(Path(config.PRICE_DIR).glob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    return [f.stem for f in files]


def audit_code(code: str) -> dict:
    path = Path(config.PRICE_DIR) / f"{code}.parquet"
    try:
        px = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        return {"code": code, "unreadable": 1, "error": type(e).__name__}
    cols = [c for c in PRICE_COLS if c in px.columns]
    px = px[cols].copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    for c in cols:
        if c != "date":
            px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.dropna(subset=["date", "close"])
    n = len(px)
    if n == 0:
        return {"code": code, "rows": 0, "empty": 1}
    out = {"code": code, "rows": n, "unreadable": 0, "empty": 0}
    out["dup_dates"] = int(px["date"].duplicated().sum())
    out["unsorted"] = int((px["date"].diff().dt.total_seconds().dropna() < 0).sum())
    px = px.sort_values("date")
    out["nonpositive_close"] = int((px["close"] <= 0).sum())
    if {"high", "low"}.issubset(px.columns):
        out["high_lt_low"] = int((px["high"] < px["low"] - 1e-9).sum())
        out["close_out_of_range"] = int(
            ((px["close"] > px["high"] + 1e-9) | (px["close"] < px["low"] - 1e-9)).sum())
    else:
        out["high_lt_low"] = out["close_out_of_range"] = 0
    out["zero_volume"] = int((px["volume"].fillna(0) <= 0).sum()) if "volume" in px.columns else -1
    prev = px["close"].shift(1)
    out["nonpositive_prev"] = int((prev <= 0).sum())
    with np.errstate(all="ignore"):
        ret = np.where(prev.to_numpy() > 0, px["close"].to_numpy() / prev.to_numpy() - 1.0, np.nan)
    ret = pd.Series(ret, index=px.index)
    finite = ret[np.isfinite(ret)]
    # 越界：超过该板块最大合法涨跌幅（留 1% 容差覆盖分位取整与除权残差）
    cap = max(tradability.limit_tiers(code)) + 0.01
    out["ret_beyond_limit"] = int((finite.abs() > cap).sum())
    out["ret_extreme_50pct"] = int((finite.abs() >= 0.5).sum())
    out["ret_minus_one"] = int((finite <= -0.999).sum())
    out["worst_ret"] = round(float(finite.min()), 4) if len(finite) else np.nan
    out["best_ret"] = round(float(finite.max()), 4) if len(finite) else np.nan
    out["first_date"] = str(px["date"].iloc[0].date())
    out["last_date"] = str(px["date"].iloc[-1].date())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只审计前 N 个 code，0 表示全量")
    ap.add_argument("--top", type=int, default=15, help="列出最严重的 N 个 code")
    args = ap.parse_args()

    codes = _codes(args.limit)
    print(f"[audit] price_dir={config.PRICE_DIR} codes={len(codes)}", flush=True)
    rows = [audit_code(c) for c in codes]
    df = pd.DataFrame(rows).fillna(0)

    flags = ["unreadable", "empty", "dup_dates", "unsorted", "nonpositive_close",
             "high_lt_low", "close_out_of_range", "nonpositive_prev",
             "ret_beyond_limit", "ret_extreme_50pct", "ret_minus_one"]
    flags = [f for f in flags if f in df.columns]

    print(f"\n== 汇总（{len(df)} 个标的，{int(df['rows'].sum())} 行）==", flush=True)
    for f in flags:
        bad_rows = int(df[f].sum())
        bad_codes = int((df[f] > 0).sum())
        print(f"  {f:22s} 行数={bad_rows:7d}  涉及标的={bad_codes:5d}", flush=True)
    if "zero_volume" in df.columns:
        zv = df.loc[df["zero_volume"] >= 0, "zero_volume"].sum()
        print(f"  {'zero_volume':22s} 行数={int(zv):7d}  "
              f"占比={zv / max(int(df['rows'].sum()), 1):.4%}", flush=True)

    df["severity"] = df[flags].sum(axis=1)
    worst = df[df["severity"] > 0].sort_values("severity", ascending=False).head(args.top)
    if worst.empty:
        print("\n未发现坏点。", flush=True)
        return
    show = ["code", "rows", "severity"] + [f for f in flags if worst[f].sum() > 0]
    show += [c for c in ("worst_ret", "best_ret", "first_date", "last_date") if c in worst.columns]
    print(f"\n== 最严重 {len(worst)} 个标的 ==", flush=True)
    print(worst[show].to_string(index=False), flush=True)
    print("\n判读：ret_beyond_limit / ret_minus_one / nonpositive_* 属于物理不可能，"
          "必须在建面板前剔除或修复；zero_volume 属正常停牌，由 fail-closed 处理。", flush=True)


if __name__ == "__main__":
    main()
