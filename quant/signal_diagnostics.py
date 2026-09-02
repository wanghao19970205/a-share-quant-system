"""信号分层诊断：IC / ICIR、十分组单调性、top_n 敏感性（只读，不训练）。

用途：在动模型和因子之前，先确认"有没有信号、信号在哪一层、需要多少只票"。
所有指标都在**可交易口径**下计算（`buyable_close` 过滤 + `tradable_ret`），
避免重复此前"给买不进的封板票发钱"的错误。

用法（容器内）：
    python -m quant.signal_diagnostics --prefix ab_baseline
    python -m quant.signal_diagnostics --prefix ab_baseline --horizons 1,2,5,10,20 --top-n 2,5,10,20,30

只读现有 `{prefix}_bt_{model}_predictions.parquet`，不重训、不写模型产物、不碰冠军。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config, lowfreq_backtest, tradability, warehouse

DEFAULT_MODEL = "ridge_lightgbm_ranker_ensemble"


def _load_predictions(prefix: str, model: str) -> pd.DataFrame:
    df = warehouse.load(f"{prefix}_bt_{model}_predictions")
    if df.empty:
        raise SystemExit(f"没有预测文件：{prefix}_bt_{model}_predictions.parquet")
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["code", "date", "pred"])


def _join_tradability(pred: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """重 join 可交易口径列，覆盖训练期带入的同名列，保证与评测尺子一致。"""
    trad = tradability.price_tradability(sorted(pred["code"].unique()), horizons)
    keep = ["code", "date", "buyable_close"]
    keep += [c for c in trad.columns if c.startswith(("tradable_ret_", "target_ret_"))]
    trad = trad[[c for c in keep if c in trad.columns]].drop_duplicates(["code", "date"])
    drop = [c for c in pred.columns if c.startswith(("tradable_ret_", "target_ret_"))]
    drop += [c for c in ("buyable_close",) if c in pred.columns]
    return pred.drop(columns=drop, errors="ignore").merge(trad, on=["code", "date"], how="left")


def _buyable(df: pd.DataFrame) -> pd.DataFrame:
    """只保留信号日真实可买入的样本；缺失视为不可买（fail-closed）。"""
    return df[df["buyable_close"].fillna(False).astype(bool)]


def restrict_vol_band(df: pd.DataFrame, lo: float, hi: float,
                      col: str = "vol_60") -> pd.DataFrame:
    """把样本限制在波动率分位带 ``(lo, hi]`` 内（分位按日度截面升序排名）。

    动机：全池 rank IC 被高波动段污染——分桶实测最高波动 20% 的年化超额是 -27%、
    t=-9.8，模型只要沾上这一段，IC 就被这段的噪声压平。这里回答的是"把高波动段
    先剔掉之后，模型在剩下的池子里还有没有信号"。

    默认取 ``vol_60``：窗口扫描实测 60 日排名最稳，10 日排名自身的抖动就是噪声源。
    波动率复用 ``lowfreq_backtest`` 的特征缓存；缓存缺失直接报错，不静默跳过过滤。
    """
    cache = Path(config.QUANT_DIR) / lowfreq_backtest.FEATURE_CACHE
    if not cache.exists():
        raise SystemExit(f"缺少波动率特征缓存 {cache}；先跑一次 quant.lowfreq_backtest 生成")
    vol = pd.read_parquet(cache, columns=["code", "date", col])
    vol["code"] = vol["code"].astype(str).str.zfill(6)
    vol["date"] = pd.to_datetime(vol["date"], errors="coerce")
    out = df.merge(vol.drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
    missing = float(out[col].isna().mean())
    out = out.dropna(subset=[col])
    q = out.groupby("date")[col].rank(pct=True, ascending=True)
    kept = out[(q > lo) & (q <= hi)]
    print(f"[diag] vol_band={col}({lo:.2f},{hi:.2f}] rows {len(df)} -> {len(kept)} "
          f"（波动率缺失剔除比例={missing:.4f}）", flush=True)
    if kept.empty:
        raise SystemExit("波动率带内没有样本，检查带宽或特征缓存覆盖范围")
    return kept.drop(columns=[col])


def ic_table(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """按 horizon 的日度截面 rank IC 及其稳定性。ICIR = mean/std，t = ICIR*sqrt(n_days)。"""
    rows = []
    pool = _buyable(df)
    for h in horizons:
        col = f"tradable_ret_{h}d"
        if col not in pool.columns:
            continue
        sub = pool.dropna(subset=["pred", col])
        ic = sub.groupby("date").apply(
            lambda g: g["pred"].corr(g[col], method="spearman") if len(g) > 20 else np.nan,
            include_groups=False,
        ).dropna()
        if ic.empty:
            continue
        mean, std = float(ic.mean()), float(ic.std(ddof=1))
        icir = mean / std if std > 0 else np.nan
        rows.append({
            "horizon": h, "ic_mean": round(mean, 5), "ic_median": round(float(ic.median()), 5),
            "ic_std": round(std, 5), "icir": round(icir, 4) if np.isfinite(icir) else np.nan,
            "t_stat": round(icir * np.sqrt(len(ic)), 3) if np.isfinite(icir) else np.nan,
            "ic_gt0_rate": round(float((ic > 0).mean()), 4), "n_days": len(ic),
        })
    return pd.DataFrame(rows)


def decile_table(df: pd.DataFrame, horizons: list[int], n_bins: int = 10) -> pd.DataFrame:
    """按日度 pred 十分组的前向平均收益，用于看单调性（第 1 组 = pred 最高）。"""
    pool = _buyable(df).copy()
    if pool.empty:
        return pd.DataFrame()
    pool["bucket"] = pool.groupby("date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first", ascending=False),
                          min(n_bins, max(s.notna().sum(), 1)),
                          labels=False, duplicates="drop") + 1
        if s.notna().sum() >= n_bins else np.nan
    )
    rows = []
    for h in horizons:
        col = f"tradable_ret_{h}d"
        if col not in pool.columns:
            continue
        g = pool.dropna(subset=["bucket", col]).groupby("bucket")[col]
        for bucket, mean_ret in g.mean().items():
            rows.append({"horizon": h, "bucket": int(bucket),
                         "mean_ret": round(float(mean_ret), 5), "n": int(g.size()[bucket])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 单调性：首组减末组的价差，以及 Spearman(组序, 平均收益)
    spreads = []
    for h, grp in out.groupby("horizon"):
        s = grp.sort_values("bucket")
        spread = float(s.iloc[0]["mean_ret"] - s.iloc[-1]["mean_ret"])
        mono = float(pd.Series(s["bucket"]).corr(pd.Series(s["mean_ret"]), method="spearman"))
        spreads.append({"horizon": h, "top_minus_bottom": round(spread, 5),
                        "monotonicity": round(mono, 4)})
    return out, pd.DataFrame(spreads)


def top_n_sweep(df: pd.DataFrame, horizon: int, top_ns: list[int],
                cost: float) -> pd.DataFrame:
    """在同一把可交易尺子下扫 top_n，看组合集中度对成本后收益的影响。"""
    rows = []
    for n in top_ns:
        r, h = backtest.portfolio_from_predictions(
            df, horizon=horizon, top_n=n, max_weight=1.0 / max(n, 1),
            filter_untradable=True, cost_roundtrip=cost)
        if r.empty:
            continue
        s = backtest.evaluate_returns(r["ret"], periods_per_year=max(1, 252 // horizon))
        rows.append({
            "top_n": n, "periods": s.get("periods"),
            "annual_return": round(float(s.get("annual_return", np.nan)), 4),
            "sharpe": round(float(s.get("sharpe", np.nan)), 3),
            "max_drawdown": round(float(s.get("max_drawdown", np.nan)), 4),
            "win_rate": round(float(s.get("win_rate", np.nan)), 4),
            "avg_turnover": round(float(r["turnover"].mean()), 4),
            "mean_pick_ret": round(float(h[f"tradable_ret_{horizon}d"].mean()), 5),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ab_baseline")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--horizons", default="1,2,5,10,20")
    ap.add_argument("--top-n", default="2,5,10,20,30")
    ap.add_argument("--cost", type=float, default=None,
                    help="单边往返成本，默认取 backtest.bt_cost_roundtrip()")
    ap.add_argument("--vol-band", default=None,
                    help="先把样本限制在波动率分位带内再做诊断，如 0.20-0.60")
    ap.add_argument("--vol-band-col", default="vol_60",
                    help="波动率带用哪一列，默认 vol_60")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    top_ns = [int(x) for x in args.top_n.split(",") if x.strip()]
    cost = args.cost if args.cost is not None else backtest.bt_cost_roundtrip()

    pred = _load_predictions(args.prefix, args.model)
    print(f"[diag] prefix={args.prefix} rows={len(pred)} dates={pred['date'].nunique()} "
          f"cost={cost}", flush=True)
    df = _join_tradability(pred, horizons)
    if args.vol_band:
        lo, hi = lowfreq_backtest.parse_band(args.vol_band)
        df = restrict_vol_band(df, lo, hi, col=args.vol_band_col)
    buyable_rate = float(df["buyable_close"].fillna(False).astype(bool).mean())
    print(f"[diag] buyable_close_rate={buyable_rate:.4f}", flush=True)

    print("\n== 1. 日度截面 rank IC（可交易样本，按 horizon）==", flush=True)
    print(ic_table(df, horizons).to_string(index=False), flush=True)

    print("\n== 2. pred 十分组前向收益（bucket 1 = pred 最高）==", flush=True)
    dec = decile_table(df, horizons)
    if isinstance(dec, tuple):
        table, spread = dec
        print(table.pivot(index="bucket", columns="horizon", values="mean_ret").to_string(),
              flush=True)
        print("\n-- 单调性（monotonicity 越接近 -1 越好：组序升高收益下降）--", flush=True)
        print(spread.to_string(index=False), flush=True)
    else:
        print("样本不足，无法分组", flush=True)

    print(f"\n== 3. top_n 敏感性（horizon={horizons[0]}，同尺可交易评测）==", flush=True)
    print(top_n_sweep(df, horizons[0], top_ns, cost).to_string(index=False), flush=True)

    print("\n判读：IC t 值需显著（|t|>2）且十分组接近单调才算有信号；"
          "top_n 扫参看集中度降低后 sharpe 是否改善。", flush=True)


if __name__ == "__main__":
    main()
