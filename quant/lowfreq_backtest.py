"""低频 + 缓冲带组合回测（只读研究）。

动机（均为实测结论，见 research/return-improvement 提交历史）：
- 日频重排本身就毁收益：同一段行情等权基准 h=1 为 -11.93%、h=10 为 +1.70%；
- 硬切 top_n 制造大量无意义换手，缓冲带可把换手压掉约 80%；
- 绝对收益无判读价值，必须看对等权基准的超额。

因此本模块把"降频 + 缓冲带 + 超额口径"固化为可复现流程，替代此前的一次性脚本。
所有收益都走修正后的可交易口径（`tradability.SEAL_VERSION>=3`）：
新进仓位必须当日可买（`buyable_close`），收益用 `tradable_ret_{h}d`（跌停顺延），
物理上不可能的坏价格行已在 tradability 层置空。

用法（研究沙箱内）：
    python -m quant.lowfreq_backtest --signal low_vol --horizons 1,2,5,10
    python -m quant.lowfreq_backtest --signal low_vol --bands 0.10/0.30,0.15/0.40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config, tradability

FEATURE_CACHE = "lowfreq_features.parquet"


def _universe(path: str | None) -> list[str]:
    """读股票池文件。每行形如 ``000001 平安银行``，允许 ``#`` 注释行。"""
    p = Path(path or config.MAINBOARD_UNIVERSE_FILE)
    codes = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        token = line.split()[0].strip()
        if token.isdigit():
            codes.append(token.zfill(6))
    if not codes:
        raise SystemExit(f"股票池为空或格式不符：{p}")
    return sorted(set(codes))


def _price_features(codes: list[str]) -> pd.DataFrame:
    """逐股读收盘价，算 PIT 安全的波动率/动量特征（只用到当日收盘为止的信息）。"""
    frames = []
    for code in codes:
        path = Path(config.PRICE_DIR) / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            px = pd.read_parquet(path, columns=["date", "close", "volume"])
        except Exception:  # noqa: BLE001
            try:
                px = pd.read_parquet(path, columns=["date", "close"])
            except Exception:  # noqa: BLE001
                continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px["close"] = pd.to_numeric(px["close"], errors="coerce")
        px = px.dropna(subset=["date", "close"]).sort_values("date")
        if len(px) < 30:
            continue
        r = px["close"].pct_change()
        out = pd.DataFrame({"code": code, "date": px["date"].to_numpy()})
        out["vol_10"] = r.rolling(10, min_periods=8).std().to_numpy()
        out["vol_20"] = r.rolling(20, min_periods=15).std().to_numpy()
        out["mom_20"] = (px["close"] / px["close"].shift(20) - 1).to_numpy()
        if "volume" in px.columns:
            v = pd.to_numeric(px["volume"], errors="coerce")
            out["dollar_vol_20"] = (v * px["close"]).rolling(
                20, min_periods=15).mean().to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_frame(codes: list[str], horizons: list[int], refresh: bool = False) -> pd.DataFrame:
    """特征 + 可交易口径的合并长表，缓存到研究数据目录避免重复全量扫盘。"""
    cache = Path(config.QUANT_DIR) / FEATURE_CACHE
    want = {f"tradable_ret_{h}d" for h in horizons}
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if want.issubset(df.columns):
            print(f"[lowfreq] 复用缓存 {cache.name} rows={len(df)}", flush=True)
            return df
    print(f"[lowfreq] 构建特征 codes={len(codes)} horizons={horizons}", flush=True)
    feat = _price_features(codes)
    trad = tradability.price_tradability(codes, horizons)
    keep = ["code", "date", "buyable_close"] + [f"tradable_ret_{h}d" for h in horizons]
    trad = trad[[c for c in keep if c in trad.columns]].drop_duplicates(["code", "date"])
    df = feat.merge(trad, on=["code", "date"], how="inner")
    df["buyable_close"] = df["buyable_close"].fillna(False).astype(bool)
    df.to_parquet(cache, index=False)
    print(f"[lowfreq] 缓存写入 {cache.name} rows={len(df)} "
          f"seal_v={tradability.SEAL_VERSION}", flush=True)
    return df


def rebalance_grid(df: pd.DataFrame, signal: str, h: int) -> set:
    """调仓日网格。策略与基准必须共用同一网格，否则超额无法对齐。"""
    dates = np.array(sorted(df.dropna(subset=[signal])["date"].unique()))
    return set(dates[::h])


def simulate(df: pd.DataFrame, signal: str, h: int, target_n: int,
             entry_q: float, exit_q: float, cost: float,
             ascending: bool = True) -> pd.DataFrame:
    """按 h 个交易日调仓、带进出缓冲带的等权组合。

    ``ascending=True`` 表示信号值越小越优（如低波动）。持仓跌出 ``exit_q`` 才卖出，
    新进标的必须落在 ``entry_q`` 内且当日 ``buyable_close``；已持仓不要求可买。
    """
    ret_col = f"tradable_ret_{h}d"
    if ret_col not in df.columns:
        raise KeyError(f"缺少收益列 {ret_col}；构建特征时需包含 horizon={h}")
    pool = df.dropna(subset=[signal]).copy()
    pool["q"] = pool.groupby("date")[signal].rank(pct=True, ascending=ascending)
    rb = rebalance_grid(df, signal, h)
    by_date = {d: g.set_index("code") for d, g in pool[pool["date"].isin(rb)].groupby("date")}
    held: list[str] = []
    rows = []
    for d in sorted(by_date):
        info = by_date[d]
        prev = list(held)
        stay = [c for c in prev if c in info.index and info.at[c, "q"] <= exit_q]
        need = target_n - len(stay)
        if need > 0:
            cand = info[(info["q"] <= entry_q) & info["buyable_close"]
                        & (~info.index.isin(stay))]
            stay += cand.sort_values("q").index[:need].tolist()
        cur = [c for c in stay if c in info.index and pd.notna(info.at[c, ret_col])]
        if not cur:
            continue
        w = 1.0 / len(cur)
        gross = float(sum(float(info.at[c, ret_col]) for c in cur) * w)
        sp, sc = set(prev), set(cur)
        turnover = 1.0 if not sp else len(sp ^ sc) / max(len(sp | sc), 1)
        rows.append({"date": d, "gross_ret": gross, "ret": gross - turnover * cost,
                     "turnover": turnover, "n": len(cur)})
        held = cur
    return pd.DataFrame(rows)


def benchmark(df: pd.DataFrame, h: int, signal: str) -> pd.Series:
    """同调仓日网格上的全可买样本等权收益，作为超额基准。"""
    ret_col = f"tradable_ret_{h}d"
    pool = df[df["buyable_close"]].dropna(subset=[ret_col])
    rb = rebalance_grid(df, signal, h)
    return pool[pool["date"].isin(rb)].groupby("date")[ret_col].mean()


def excess_series(r: pd.DataFrame, bench: pd.Series) -> pd.Series:
    """逐调仓日的净超额。只保留策略与基准都有值的日期，避免半边缺失污染均值。"""
    aligned = r.set_index("date")["ret"].reindex(bench.index).dropna()
    return aligned - bench.reindex(aligned.index)


def _excess_stats(ex: pd.Series, ppy: int) -> tuple[float, float, float]:
    """返回（年化超额, 信息比, t 值）。t 用超额均值的标准误，不做自相关修正。"""
    if len(ex) < 2:
        return np.nan, np.nan, np.nan
    sd = float(ex.std(ddof=1))
    ann = float((1 + ex.mean()) ** ppy - 1)
    if not sd > 0:
        return ann, np.nan, np.nan
    ir = float(ex.mean() / sd * np.sqrt(ppy))
    t = float(ex.mean() / sd * np.sqrt(len(ex)))
    return ann, ir, t


def report(label: str, r: pd.DataFrame, bench: pd.Series, h: int) -> dict:
    ppy = max(1, int(round(252 / h)))
    net = backtest.evaluate_returns(r["ret"], periods_per_year=ppy)
    gross = backtest.evaluate_returns(r["gross_ret"], periods_per_year=ppy)
    ex = excess_series(r, bench)
    ex_ann, ir, t = _excess_stats(ex, ppy)
    row = {
        "strategy": label, "h": h, "periods": int(net.get("periods", 0)),
        "net_annual": round(float(net.get("annual_return", np.nan)), 4),
        "gross_annual": round(float(gross.get("annual_return", np.nan)), 4),
        "excess_annual": round(ex_ann, 4) if np.isfinite(ex_ann) else np.nan,
        "info_ratio": round(ir, 3) if np.isfinite(ir) else np.nan,
        "t": round(t, 2) if np.isfinite(t) else np.nan,
        "sharpe": round(float(net.get("sharpe", np.nan)), 3),
        "max_dd": round(float(net.get("max_drawdown", np.nan)), 4),
        "win": round(float(net.get("win_rate", np.nan)), 4),
        "turnover": round(float(r["turnover"].mean()), 4),
        "holdings": round(float(r["n"].mean()), 1),
    }
    return row


def yearly_excess(ex: pd.Series, ppy: int) -> pd.DataFrame:
    """分年度超额。单年样本 < 10 个调仓日时 t 值无意义，仍照实列出供人工判读。"""
    if not len(ex):
        return pd.DataFrame()
    rows = []
    for year, seg in ex.groupby(pd.DatetimeIndex(ex.index).year):
        ann, ir, t = _excess_stats(seg, ppy)
        rows.append({"year": int(year), "periods": int(len(seg)),
                     "excess_annual": round(ann, 4) if np.isfinite(ann) else np.nan,
                     "info_ratio": round(ir, 3) if np.isfinite(ir) else np.nan,
                     "t": round(t, 2) if np.isfinite(t) else np.nan,
                     "win": round(float((seg > 0).mean()), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="low_vol",
                    choices=["low_vol", "low_vol20", "low_turnover"])
    ap.add_argument("--horizons", default="1,2,5,10", help="调仓间隔（交易日），上限 10")
    ap.add_argument("--target-n", default="50,100")
    ap.add_argument("--bands", default="hard,0.10/0.30,0.15/0.40",
                    help="缓冲带 entry/exit 分位；hard 表示无缓冲的硬切")
    ap.add_argument("--cost", type=float, default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--refresh", action="store_true", help="强制重建特征缓存")
    ap.add_argument("--reverse", action="store_true",
                    help="反转信号方向（低波动 → 高波动），用于检验超额是否来自真实截面价差")
    ap.add_argument("--yearly", action="store_true", help="额外输出分年度超额，检验稳定性")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if max(horizons) > 10:
        raise SystemExit("调仓间隔上限为 10 个交易日")
    target_ns = [int(x) for x in args.target_n.split(",") if x.strip()]
    cost = args.cost if args.cost is not None else backtest.bt_cost_roundtrip()
    sig_map = {"low_vol": ("vol_10", True), "low_vol20": ("vol_20", True),
               "low_turnover": ("dollar_vol_20", True)}
    signal, ascending = sig_map[args.signal]
    if args.reverse:
        ascending = not ascending

    codes = _universe(args.universe)
    df = build_frame(codes, horizons, refresh=args.refresh)
    print(f"[lowfreq] rows={len(df)} codes={df['code'].nunique()} "
          f"dates={df['date'].nunique()} signal={signal} ascending={ascending} "
          f"cost={cost}", flush=True)

    rows = []
    yearly: list[tuple[str, pd.DataFrame]] = []
    for h in horizons:
        bench = benchmark(df, h, signal)
        ppy = max(1, int(round(252 / h)))
        bs = backtest.evaluate_returns(bench, periods_per_year=ppy)
        rows.append({"strategy": "BENCH equal-weight", "h": h,
                     "periods": int(bs.get("periods", 0)),
                     "net_annual": round(float(bs.get("annual_return", np.nan)), 4),
                     "gross_annual": round(float(bs.get("annual_return", np.nan)), 4),
                     "excess_annual": 0.0, "info_ratio": 0.0, "t": 0.0,
                     "sharpe": round(float(bs.get("sharpe", np.nan)), 3),
                     "max_dd": round(float(bs.get("max_drawdown", np.nan)), 4),
                     "win": round(float(bs.get("win_rate", np.nan)), 4),
                     "turnover": 0.0, "holdings": np.nan})
        for n in target_ns:
            for band in args.bands.split(","):
                band = band.strip()
                if band == "hard":
                    q = min(1.0, n / max(df["code"].nunique(), 1))
                    entry_q = exit_q = q
                    label = "HARD-CUT"
                else:
                    entry_q, exit_q = (float(x) for x in band.split("/"))
                    label = f"BUFFER {entry_q:.2f}/{exit_q:.2f}"
                r = simulate(df, signal, h, n, entry_q, exit_q, cost, ascending)
                if r.empty:
                    continue
                row = report(f"{label} n={n}", r, bench, h)
                rows.append(row)
                if args.yearly:
                    yb = yearly_excess(excess_series(r, bench), ppy)
                    if not yb.empty:
                        yearly.append((f"{row['strategy']} h={h}", yb))
                print(f"  {row['strategy']:26s} h={h:2d} periods={row['periods']:4d} "
                      f"net={row['net_annual']:+.4f} gross={row['gross_annual']:+.4f} "
                      f"excess={row['excess_annual']:+.4f} IR={row['info_ratio']:+.3f} "
                      f"t={row['t']:+.2f} "
                      f"maxDD={row['max_dd']:+.4f} turnover={row['turnover']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    print("\n== 汇总（excess = 对同调仓日等权基准的年化超额，IR = 超额信息比）==", flush=True)
    print(out.to_string(index=False), flush=True)
    for label, yb in yearly:
        print(f"\n-- 分年度超额 {label} --", flush=True)
        print(yb.to_string(index=False), flush=True)
    print("\n判读：只有 excess_annual > 0 且 |t| > 2 才算跑赢基准；"
          "periods 太少时 sharpe/IR/t 无统计意义。", flush=True)


if __name__ == "__main__":
    main()
