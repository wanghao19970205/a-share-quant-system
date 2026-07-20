"""量化选股 · 因子筛选。

基于因子面板计算：
- IC / RankIC 均值、波动、ICIR、胜率；
- 分层收益，多空组合收益；
- 因子衰减，不同 forward horizon 的相关性稳定度。
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from quant import warehouse
from quant.factors import engineering


def _target_col(df: pd.DataFrame, horizon: int) -> str:
    col = f"target_ret_{horizon}d"
    if col not in df.columns:
        raise ValueError(f"缺少标签列 {col}")
    return col


def _daily_ic_rows(panel: pd.DataFrame, groups: list[tuple[object, np.ndarray]],
                   factors: list[str], target: str, method: str) -> list[dict]:
    rows = []
    for date, positions in groups:
        g = panel.iloc[positions]
        y = g[target]
        if y.notna().sum() < 5:
            continue
        for f in factors:
            x = g[f]
            ok = x.notna() & y.notna()
            if ok.sum() < 5 or x[ok].nunique() < 3:
                continue
            rows.append({"date": date, "factor": f, "ic": x[ok].corr(y[ok], method=method)})
    return rows


def daily_ic(panel: pd.DataFrame, factors: list[str], horizon: int = 5,
             method: str = "spearman", workers: int = 1) -> pd.DataFrame:
    target = _target_col(panel, horizon)
    groups = list(panel.groupby("date").indices.items())
    worker_count = min(max(int(workers), 1), len(groups))
    if worker_count <= 1:
        return pd.DataFrame(_daily_ic_rows(panel, groups, factors, target, method))

    chunk_size = math.ceil(len(groups) / worker_count)
    chunks = [groups[start:start + chunk_size] for start in range(0, len(groups), chunk_size)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        parts = executor.map(
            lambda chunk: _daily_ic_rows(panel, chunk, factors, target, method),
            chunks,
        )
    return pd.DataFrame(row for part in parts for row in part)


_IC_SUMMARY_COLUMNS = [
    "factor", "ic_mean", "ic_std", "ic_count", "icir", "ic_win_rate", "abs_ic_mean",
]


def ic_summary_from_daily_ic(ic: pd.DataFrame) -> pd.DataFrame:
    """Build the IC summary from an already calculated daily IC table."""
    if ic.empty:
        return pd.DataFrame(columns=_IC_SUMMARY_COLUMNS)
    grp = ic.groupby("factor")["ic"]
    out = grp.agg(ic_mean="mean", ic_std="std", ic_count="count").reset_index()
    out["icir"] = out["ic_mean"] / out["ic_std"].replace(0, np.nan)
    out["ic_win_rate"] = grp.apply(lambda s: (s > 0).mean()).values
    out["abs_ic_mean"] = out["ic_mean"].abs()
    return out.sort_values(["abs_ic_mean", "icir"], ascending=False).reset_index(drop=True)


def ic_summary(panel: pd.DataFrame, factors: list[str] | None = None, horizon: int = 5) -> pd.DataFrame:
    factors = factors or engineering.feature_columns(panel, horizon)
    return ic_summary_from_daily_ic(daily_ic(panel, factors, horizon=horizon))


def quantile_returns(panel: pd.DataFrame, factor: str, horizon: int = 5, q: int = 5) -> pd.DataFrame:
    target = _target_col(panel, horizon)
    rows = []
    for date, g in panel.groupby("date"):
        sub = g[[factor, target]].dropna()
        if len(sub) < q * 3 or sub[factor].nunique() < q:
            continue
        try:
            bucket = pd.qcut(sub[factor], q=q, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        tmp = sub.assign(bucket=bucket)
        ret = tmp.groupby("bucket")[target].mean()
        for b, v in ret.items():
            rows.append({"date": date, "factor": factor, "bucket": int(b), "ret": float(v)})
    return pd.DataFrame(rows)


def layer_summary(panel: pd.DataFrame, factors: list[str], horizon: int = 5, q: int = 5) -> pd.DataFrame:
    rows = []
    for f in factors:
        qr = quantile_returns(panel, f, horizon=horizon, q=q)
        if qr.empty:
            continue
        pivot = qr.pivot_table(index="date", columns="bucket", values="ret")
        if 1 not in pivot.columns or q not in pivot.columns:
            continue
        ls = pivot[q] - pivot[1]
        rows.append({
            "factor": f,
            "top_mean": pivot[q].mean(),
            "bottom_mean": pivot[1].mean(),
            "long_short_mean": ls.mean(),
            "long_short_win_rate": (ls > 0).mean(),
            "n_dates": len(ls.dropna()),
        })
    if not rows:
        return pd.DataFrame(columns=["factor", "top_mean", "bottom_mean", "long_short_mean", "long_short_win_rate", "n_dates"])
    return pd.DataFrame(rows).sort_values("long_short_mean", ascending=False).reset_index(drop=True)


def decay_summary(raw_panel: pd.DataFrame, factors: list[str], horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or [1, 3, 5, 10, 20]
    rows = []
    panel = raw_panel.copy().sort_values(["code", "date"])
    for h in horizons:
        target = f"target_ret_{h}d"
        if target not in panel.columns:
            panel[target] = panel.groupby("code")["close"].shift(-h) / panel["close"] - 1
        prepared, feats = engineering.prepare_features(panel, horizon=h)
        use = [f for f in factors if f in feats]
        summ = ic_summary(prepared, use, horizon=h)
        for _, r in summ.iterrows():
            rows.append({"factor": r["factor"], "horizon": h, "ic_mean": r["ic_mean"], "icir": r["icir"]})
    return pd.DataFrame(rows)


def select_factors(panel: pd.DataFrame, horizon: int = 5, min_abs_ic: float = 0.02, min_icir: float = 0.15,
                   top_n: int = 50) -> pd.DataFrame:
    factors = engineering.feature_columns(panel, horizon)
    summ = ic_summary(panel, factors, horizon=horizon)
    if summ.empty:
        return summ
    picked = summ[(summ["abs_ic_mean"] >= min_abs_ic) & (summ["icir"].abs() >= min_icir)].copy()
    if picked.empty:
        picked = summ.head(top_n).copy()
    return picked.head(top_n).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="筛选量化因子")
    ap.add_argument("--panel", default="factor_panel")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--output", default="factor_selection")
    args = ap.parse_args()

    panel = warehouse.load(args.panel)
    if panel.empty:
        raise SystemExit(f"面板不存在或为空：{args.panel}")
    picked = select_factors(panel, horizon=args.horizon, top_n=args.top)
    layers = layer_summary(panel, picked["factor"].tolist(), horizon=args.horizon)
    warehouse.save(args.output, picked)
    if not layers.empty:
        warehouse.save(f"{args.output}_layers", layers)
    print(f"保存 {args.output}: {len(picked)} 个因子")
    print(picked.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
