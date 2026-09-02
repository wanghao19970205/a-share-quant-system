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
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config, tradability, warehouse

FEATURE_CACHE = "lowfreq_features.parquet"
# 波动率窗口 -> 最少观测数。窗口越长排名越稳、换手越低，实测 20 天优于 10 天。
VOL_WINDOWS = {10: 8, 20: 15, 40: 30, 60: 45, 120: 90}


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
        for w, mp in VOL_WINDOWS.items():
            out[f"vol_{w}"] = r.rolling(w, min_periods=mp).std().to_numpy()
        out["mom_20"] = (px["close"] / px["close"].shift(20) - 1).to_numpy()
        if "volume" in px.columns:
            v = pd.to_numeric(px["volume"], errors="coerce")
            out["dollar_vol_20"] = (v * px["close"]).rolling(
                20, min_periods=15).mean().to_numpy()
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_frame(codes: list[str], horizons: list[int], refresh: bool = False,
                need_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """特征 + 可交易口径的合并长表，缓存到研究数据目录避免重复全量扫盘。

    ``need_cols`` 里的列缺一个就重建：老缓存没有新加的波动率窗口，若沿用会在
    后面才以 KeyError 暴露，或更糟——被误当成"信号全缺"而静默跳过。
    """
    cache = Path(config.QUANT_DIR) / FEATURE_CACHE
    want = {f"tradable_ret_{h}d" for h in horizons} | set(need_cols)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if want.issubset(df.columns):
            print(f"[lowfreq] 复用缓存 {cache.name} rows={len(df)}", flush=True)
            return df
        print(f"[lowfreq] 缓存缺列 {sorted(want - set(df.columns))}，重建", flush=True)
    print(f"[lowfreq] 构建特征 codes={len(codes)} horizons={horizons}", flush=True)
    feat = _price_features(codes)
    trad = tradability.price_tradability(codes, horizons)
    keep = ["code", "date", "buyable_close"] + [f"tradable_ret_{h}d" for h in horizons]
    trad = trad[[c for c in keep if c in trad.columns]].drop_duplicates(["code", "date"])
    df = feat.merge(trad, on=["code", "date"], how="inner")
    df["buyable_close"] = df["buyable_close"].fillna(False).astype(bool)
    # 原子落盘：缓存有几百 MB，重建期间若有别的实验在读，非原子写会读到半个文件。
    tmp = cache.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    print(f"[lowfreq] 缓存写入 {cache.name} rows={len(df)} "
          f"seal_v={tradability.SEAL_VERSION}", flush=True)
    return df


def join_predictions(df: pd.DataFrame, prefix: str, model: str) -> pd.DataFrame:
    """把已有预测文件的 ``pred`` 并进长表，只保留有预测的 (code, date)。

    用途：模型信号也要过同一套「缓冲带 + 低频 + 可交易口径」的检验。日频 top_n
    评测里模型换手 0.85、成本吃掉一切，看不出信号本身有没有用。
    """
    name = f"{prefix}_bt_{model}_predictions"
    pred = warehouse.load(name)
    if pred.empty:
        raise SystemExit(f"没有预测文件：{name}.parquet")
    pred = pred[["code", "date", "pred"]].copy()
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred = pred.dropna(subset=["code", "date", "pred"]).drop_duplicates(["code", "date"])
    out = df.merge(pred, on=["code", "date"], how="inner")
    if out.empty:
        raise SystemExit(f"{name} 与特征表没有交集，检查股票池/日期范围")
    print(f"[lowfreq] join {name} rows={len(out)} dates={out['date'].nunique()}", flush=True)
    return out


def restrict_vol_band(df: pd.DataFrame, lo: float, hi: float,
                      col: str = "vol_10") -> pd.DataFrame:
    """按日度截面把样本限制在波动率分位带 ``(lo, hi]`` 内；缺波动率的样本剔除。"""
    if col not in df.columns:
        raise SystemExit(f"缺少波动率列 {col}")
    q = df.groupby("date")[col].rank(pct=True, ascending=True)
    kept = df[(q > lo) & (q <= hi)]
    if kept.empty:
        raise SystemExit(f"波动率带 ({lo},{hi}] 内没有样本")
    print(f"[lowfreq] vol_band={col}({lo:.2f},{hi:.2f}] rows {len(df)} -> {len(kept)}",
          flush=True)
    return kept


def smooth_signal(df: pd.DataFrame, signal: str, window: int,
                  ascending: bool) -> pd.DataFrame:
    """把信号换成「过去 window 个交易日截面分位的滚动均值」，用来压低换手。

    动机：模型 pred 逐日跳动，同样一套缓冲带下换手是波动率信号的 2.5 倍，
    成本把 gross 优势全部吃掉。先按日度截面转成分位再平滑，避免 pred 的量纲
    逐日漂移。只用当日及之前的分位，不引入未来信息。

    返回后信号语义统一为「越小越优」，调用方需用 ``ascending=True``。
    """
    q = df.groupby("date")[signal].rank(pct=True, ascending=ascending)
    out = df.assign(_q=q).sort_values(["code", "date"])
    out[signal] = out.groupby("code")["_q"].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    return out.drop(columns=["_q"])


def rebalance_grid(df: pd.DataFrame, signal: str, h: int) -> set:
    """调仓日网格。策略与基准必须共用同一网格，否则超额无法对齐。"""
    dates = np.array(sorted(df.dropna(subset=[signal])["date"].unique()))
    return set(dates[::h])


def prepare(df: pd.DataFrame, signal: str, h: int,
            ascending: bool = True) -> list[tuple]:
    """预计算各调仓日的截面，输出按日期升序的 ``(date, ids, q, buyable, ret)`` 列表。

    两处提速都在这里：同一 ``(signal, h, ascending)`` 下所有参数组合共用这份结果
    （分位排名要扫全表）；``ids`` 是全表统一的整数编号且逐日升序，回测循环即可用
    ``searchsorted`` 做集合运算，不必反复对字符串代码做哈希对齐。
    """
    ret_col = f"tradable_ret_{h}d"
    if ret_col not in df.columns:
        raise KeyError(f"缺少收益列 {ret_col}；构建特征时需包含 horizon={h}")
    rb = rebalance_grid(df, signal, h)
    pool = df.dropna(subset=[signal])
    pool = pool[pool["date"].isin(rb)][["code", "date", signal, "buyable_close", ret_col]]
    pool = pool.copy()
    # 分位是逐日截面内的排名，先筛调仓日再排名不改变取值，但省掉非调仓日的排序开销。
    pool["q"] = pool.groupby("date")[signal].rank(pct=True, ascending=ascending)
    pool["cid"] = pd.factorize(pool["code"])[0]
    out = []
    for d, g in pool.groupby("date", sort=True):
        g = g.sort_values("cid")
        out.append((d, g["cid"].to_numpy(np.int64), g["q"].to_numpy(float),
                    g["buyable_close"].to_numpy(bool), g[ret_col].to_numpy(float)))
    return out


def simulate(sections: list[tuple], target_n: int, entry_q: float, exit_q: float,
             cost: float, entry_lo: float = 0.0, exit_lo: float = 0.0,
             max_out: int | None = None) -> pd.DataFrame:
    """在 ``prepare`` 产出的截面上跑带进出缓冲带的等权组合。

    分位区间为半开的 ``(lo, q]``，这样相邻分桶不会在边界上重复收票；持仓落到
    ``(exit_lo, exit_q]`` 之外才卖出，新进标的必须落在 ``(entry_lo, entry_q]`` 内
    且当日 ``buyable_close``；已持仓不要求可买。``lo`` 默认 0，因分位数恒大于 0，
    等价于传统的取头部。信号方向由 ``prepare(ascending=...)`` 决定。

    ``max_out`` 限制每期主动卖出只数：掉出退出带的持仓按分位排序，只卖最差的
    那几只，其余暂留。卖出侧才是换手的源头——只限制补仓会让组合越卖越空、
    换手率（对称差/并集）反而升高，实测如此。当日退市/停牌的持仓不受上限保护。
    """
    held = np.empty(0, dtype=np.int64)
    seen: set[int] = set()
    rows = []
    for d, ids, q, buyable, ret in sections:
        prev = held
        # ids 已升序，用 searchsorted 定位持仓；当日退市/停牌的持仓自然落不到位。
        pos = np.searchsorted(ids, prev)
        ok = pos < ids.size
        pos = np.where(ok, pos, 0)
        ok &= ids[pos] == prev
        pq = np.where(ok, q[pos], np.nan)
        inside = ok & (pq > exit_lo) & (pq <= exit_q)
        if max_out is not None:
            dropping = np.flatnonzero(ok & ~inside)
            if dropping.size > max_out:
                # 分位越大越差，只卖最差的 max_out 只
                worst = dropping[np.argsort(-pq[dropping], kind="stable")[:max_out]]
                reprieve = np.zeros(prev.size, dtype=bool)
                reprieve[dropping] = True
                reprieve[worst] = False
                inside = inside | reprieve
        stay = prev[inside]
        need = target_n - stay.size
        if need > 0:
            cand = (q > entry_lo) & (q <= entry_q) & buyable
            if stay.size:
                cand &= ~np.isin(ids, stay, assume_unique=True)
            ci = np.flatnonzero(cand)
            if ci.size:
                take = ci[np.argsort(q[ci], kind="stable")[:need]]
                stay = np.concatenate([stay, ids[np.sort(take)]])
        spos = np.searchsorted(ids, stay)
        rv = ret[spos]
        good = np.isfinite(rv)
        if not good.any():
            continue
        cur = stay[good]
        gross = float(rv[good].mean())
        if prev.size:
            inter = np.intersect1d(prev, cur, assume_unique=True).size
            turnover = (prev.size + cur.size - 2 * inter) / max(
                prev.size + cur.size - inter, 1)
        else:
            turnover = 1.0
        # 记录首次出现的只数：换手压得越低，组合越接近一个固定篮子，
        # 而固定篮子恰恰是存活偏差最容易吃到的形态，必须能看见这个数。
        fresh = int(sum(1 for c in cur.tolist() if c not in seen))
        seen.update(cur.tolist())
        rows.append({"date": d, "gross_ret": gross, "ret": gross - turnover * cost,
                     "turnover": turnover, "n": int(cur.size), "new_names": fresh})
        held = np.sort(cur)
    return pd.DataFrame(rows)


def benchmark(df: pd.DataFrame, h: int, grid: set) -> pd.Series:
    """给定调仓日网格上的全可买样本等权收益，作为超额基准。

    网格必须由策略侧传入而不是在这里另算一遍：股票池被波动率带收窄后，策略的
    有效日期会比全池少（早期 vol_60 尚未成形），各自算网格就会错位到无法相减。
    """
    ret_col = f"tradable_ret_{h}d"
    pool = df[df["buyable_close"]].dropna(subset=[ret_col])
    return pool[pool["date"].isin(grid)].groupby("date")[ret_col].mean()


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
    # 退化序列（如常数超额）的 std 只剩浮点噪声，直接除会算出 1e16 量级的假 t 值。
    # 真实日度超额的 std 在 1e-3 量级，1e-12 作为下限足够安全。
    if not sd > 1e-12:
        return ann, np.nan, np.nan
    ir = float(ex.mean() / sd * np.sqrt(ppy))
    t = float(ex.mean() / sd * np.sqrt(len(ex)))
    return ann, ir, t


def report(label: str, r: pd.DataFrame, bench: pd.Series, h: int,
           split: pd.Timestamp | None = None) -> dict:
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
    if "new_names" in r.columns:
        uniq = int(r["new_names"].sum())
        row["uniq_codes"] = uniq
        # 平均持有期（单位：调仓期）。数值接近总期数说明组合近乎静态。
        row["avg_tenure"] = round(float(r["n"].sum() / uniq), 1) if uniq else np.nan
    if split is not None:
        # 选参只能看 _is 列，_oos 列是留出期；同时看两边就等于又一次全样本挑选。
        for tag, seg in (("is", ex[ex.index < split]), ("oos", ex[ex.index >= split])):
            s_ann, _, s_t = _excess_stats(seg, ppy)
            row[f"periods_{tag}"] = int(len(seg))
            row[f"excess_{tag}"] = round(s_ann, 4) if np.isfinite(s_ann) else np.nan
            row[f"t_{tag}"] = round(s_t, 2) if np.isfinite(s_t) else np.nan
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


def parse_band(spec: str) -> tuple[float, float]:
    """解析一侧分位区间。``0.40`` 表示 (0, 0.40]，``0.20-0.60`` 表示 (0.20, 0.60]。"""
    if "-" in spec:
        lo, hi = (float(x) for x in spec.split("-", 1))
    else:
        lo, hi = 0.0, float(spec)
    if not 0.0 <= lo < hi <= 1.0:
        raise SystemExit(f"分位区间非法：{spec}（需满足 0 <= lo < hi <= 1）")
    return lo, hi


def _specs(args, n_codes: int, target_ns: list[int]) -> list[tuple]:
    """生成 (label, target_n, entry_lo, entry_q, exit_lo, exit_q) 组合清单。"""
    if args.buckets:
        b = args.buckets
        pad = 0.5 / b   # 缓冲带宽度取桶宽的一半，进出对称
        out = []
        for i in range(b):
            lo, hi = i / b, (i + 1) / b
            out.append((f"VOLQ {lo:.2f}-{hi:.2f}", 10 ** 9,
                        lo, hi, max(0.0, lo - pad), min(1.0, hi + pad)))
        return out
    out = []
    for n in target_ns:
        for band in args.bands.split(","):
            band = band.strip()
            if band == "hard":
                q = min(1.0, n / max(n_codes, 1))
                out.append((f"HARD-CUT n={n}", n, 0.0, q, 0.0, q))
            else:
                e_spec, x_spec = band.split("/", 1)
                e_lo, e_hi = parse_band(e_spec)
                x_lo, x_hi = parse_band(x_spec)
                if x_lo > e_lo or x_hi < e_hi:
                    raise SystemExit(f"卖出区间必须包住买入区间：{band}")
                out.append((f"BAND {e_spec}/{x_spec} n={n}", n, e_lo, e_hi, x_lo, x_hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="low_vol",
                    choices=["low_vol", "low_vol20", "low_vol40", "low_vol60", "low_vol120",
                             "low_turnover", "mom", "pred"])
    ap.add_argument("--horizons", default="1,2,5,10", help="调仓间隔（交易日），上限 10")
    ap.add_argument("--target-n", default="50,100")
    ap.add_argument("--bands", default="hard,0.10/0.30,0.15/0.40",
                    help="进/出分位区间；hard 表示无缓冲的硬切。"
                         "一侧可写 0.30（即 (0,0.30]）或 0.20-0.30（即 (0.20,0.30]）")
    ap.add_argument("--cost", type=float, default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--refresh", action="store_true", help="强制重建特征缓存")
    ap.add_argument("--reverse", action="store_true",
                    help="反转信号方向（低波动 → 高波动），用于检验超额是否来自真实截面价差")
    ap.add_argument("--yearly", action="store_true", help="额外输出分年度超额，检验稳定性")
    ap.add_argument("--buckets", type=int, default=0,
                    help="改为按信号分位分桶（如 5），映射信号与收益的形状而非只取头部")
    ap.add_argument("--split", default=None,
                    help="样本内/样本外切分日（如 2023-01-01）；选参只许看 _is 列")
    ap.add_argument("--pred-prefix", default=None,
                    help="--signal pred 时的预测文件前缀，如 ab_tradable_mask")
    ap.add_argument("--pred-model", default="ridge_lightgbm_ranker_ensemble")
    ap.add_argument("--vol-band", default=None,
                    help="先按波动率分位带筛股票池再选股，如 0.20-0.60；基准仍为全可买池")
    ap.add_argument("--smooth", type=int, default=1,
                    help="把信号换成过去 N 日截面分位的滚动均值，用于压低换手（1 表示不平滑）")
    ap.add_argument("--vol-band-col", default="vol_10",
                    help="波动率带用哪一列做池过滤，默认 vol_10")
    ap.add_argument("--max-out-frac", type=float, default=None,
                    help="每期主动卖出上限占 target_n 的比例，如 0.05；不给则不限制")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if max(horizons) > 10:
        raise SystemExit("调仓间隔上限为 10 个交易日")
    split = pd.Timestamp(args.split) if args.split else None
    target_ns = [int(x) for x in args.target_n.split(",") if x.strip()]
    cost = args.cost if args.cost is not None else backtest.bt_cost_roundtrip()
    sig_map = {"low_vol": ("vol_10", True), "low_vol20": ("vol_20", True),
               "low_vol40": ("vol_40", True), "low_vol60": ("vol_60", True),
               "low_vol120": ("vol_120", True),
               "low_turnover": ("dollar_vol_20", True), "mom": ("mom_20", False),
               "pred": ("pred", False)}
    signal, ascending = sig_map[args.signal]
    if args.reverse:
        ascending = not ascending
    if args.signal == "pred" and not args.pred_prefix:
        raise SystemExit("--signal pred 需要同时给 --pred-prefix")

    need = {signal} | ({args.vol_band_col} if args.vol_band else set())
    codes = _universe(args.universe)
    df = build_frame(codes, horizons, refresh=args.refresh,
                     need_cols=tuple(c for c in sorted(need) if c != "pred"))
    if args.pred_prefix:
        df = join_predictions(df, args.pred_prefix, args.pred_model)
    # 基准始终是全可买池，只有选股池被波动率带收窄；两者共用同一调仓日网格。
    bench_df = df
    if args.vol_band:
        lo, hi = parse_band(args.vol_band)
        df = restrict_vol_band(df, lo, hi, col=args.vol_band_col)
    if args.smooth > 1:
        # 平滑后信号统一为「越小越优」，方向已折进分位里。
        df = smooth_signal(df, signal, args.smooth, ascending)
        ascending = True
        print(f"[lowfreq] smooth={args.smooth} 已把 {signal} 换成滚动分位均值", flush=True)
    print(f"[lowfreq] rows={len(df)} codes={df['code'].nunique()} "
          f"dates={df['date'].nunique()} signal={signal} ascending={ascending} "
          f"cost={cost}", flush=True)

    rows = []
    yearly: list[tuple[str, pd.DataFrame]] = []
    for h in horizons:
        # 网格由策略侧决定，基准跟随。基准可以缺最尾部的日期——那些日子还没有前向
        # 收益，策略同样交易不了；但缺在中间就是真错位（换排序信号时踩过），必须拦住。
        grid = rebalance_grid(df, signal, h)
        bench = benchmark(bench_df, h, grid)
        if bench.empty:
            raise SystemExit(f"h={h} 基准为空，检查 buyable_close 与前向收益列")
        gap = sorted(d for d in grid if d not in set(bench.index))
        if gap and min(gap) < bench.index.max():
            raise SystemExit(f"h={h} 基准在 {min(gap)} 等 {len(gap)} 个调仓日缺失且不在尾部，"
                             f"超额无法对齐")
        if gap:
            print(f"[lowfreq] h={h} 尾部 {len(gap)} 个调仓日无前向收益，已剔除", flush=True)
        ppy = max(1, int(round(252 / h)))
        sections = prepare(df, signal, h, ascending)
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
        for label, n, e_lo, e_hi, x_lo, x_hi in _specs(args, df["code"].nunique(), target_ns):
            max_out = None
            if args.max_out_frac is not None and n < 10 ** 8:
                max_out = max(1, int(round(args.max_out_frac * n)))
                label = f"{label} out<={max_out}"
            r = simulate(sections, n, e_hi, x_hi, cost, entry_lo=e_lo, exit_lo=x_lo,
                         max_out=max_out)
            if r.empty:
                continue
            row = report(label, r, bench, h, split=split)
            rows.append(row)
            if args.yearly:
                yb = yearly_excess(excess_series(r, bench), ppy)
                if not yb.empty:
                    yearly.append((f"{row['strategy']} h={h}", yb))
            tail = ""
            if split is not None:
                tail = (f" | IS excess={row['excess_is']:+.4f} t={row['t_is']:+.2f}"
                        f" | OOS excess={row['excess_oos']:+.4f} t={row['t_oos']:+.2f}")
            print(f"  {row['strategy']:26s} h={h:2d} periods={row['periods']:4d} "
                  f"net={row['net_annual']:+.4f} gross={row['gross_annual']:+.4f} "
                  f"excess={row['excess_annual']:+.4f} IR={row['info_ratio']:+.3f} "
                  f"t={row['t']:+.2f} maxDD={row['max_dd']:+.4f} "
                  f"turnover={row['turnover']:.4f} n={row['holdings']:.0f}{tail}", flush=True)

    out = pd.DataFrame(rows)
    print("\n== 汇总（excess = 对同调仓日等权基准的年化超额，IR = 超额信息比）==", flush=True)
    print(out.to_string(index=False), flush=True)
    for label, yb in yearly:
        print(f"\n-- 分年度超额 {label} --", flush=True)
        print(yb.to_string(index=False), flush=True)
    print("\n判读：只有 excess_annual > 0 且 |t| > 2 才算跑赢基准；"
          "periods 太少时 sharpe/IR/t 无统计意义。", flush=True)
    if split is not None:
        print(f"切分日={split.date()}：选参只许看 excess_is/t_is，"
              f"excess_oos/t_oos 是留出期结果，用来判定选出来的东西是否存活。", flush=True)


if __name__ == "__main__":
    main()
