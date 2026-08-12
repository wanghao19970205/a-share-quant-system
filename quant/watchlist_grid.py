"""Run watchlist-only multi-horizon grids on a prediction artifact."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quant import backtest, config, tradability, warehouse

# 回测成交口径由 backtest 的环境变量开关决定（默认：当日收盘成交、无成本、不过滤涨停/停牌）。


def _quant_dir() -> Path:
    return Path(config.QUANT_DIR)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _quant_dir() / p


def _read_watchlist(path: str | Path) -> set[str]:
    p = Path(path)
    codes: set[str] = set()
    if not p.exists():
        return codes
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].split(",")[0].strip()
        if len(token) >= 6 and token[:6].isdigit():
            codes.add(token[:6])
    return codes


def _parse_horizons(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _empty_to_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _rebalance_stride(params: pd.Series | dict) -> int:
    return max(int(_empty_to_none(params.get("rebalance_stride")) or 1), 1)


def _hold_rank_buffer(params: pd.Series | dict) -> int:
    return max(int(_empty_to_none(params.get("hold_rank_buffer")) or 0), 0)


def _stride_calendar(stride: int) -> pd.DataFrame | None:
    return warehouse.load("trading_calendar") if stride > 1 else None


def _stride_predictions(pred: pd.DataFrame, params: pd.Series | dict) -> pd.DataFrame:
    stride = _rebalance_stride(params)
    return backtest._apply_rebalance_stride(
        pred, stride, trading_calendar=_stride_calendar(stride),
    )


def _load_predictions(path: Path, watchlist: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["code", "date"])
    if watchlist:
        df = df[df["code"].isin(watchlist)].copy()
    if "base_pred" not in df.columns:
        df["base_pred"] = pd.to_numeric(df.get("pred"), errors="coerce")
    if "ic_z" not in df.columns:
        df["ic_z"] = 0.0
    return df


def _sell_roll_max_days() -> int:
    """跌停顺延卖出的上限交易日数（含预定卖出日）。单一真源在 tradability.sell_roll_max_days()。"""
    return tradability.sell_roll_max_days()


def _rolled_sell_close(close: np.ndarray, sell_blocked: np.ndarray, horizon: int, cap: int) -> np.ndarray:
    """跌停顺延卖出实现价。单一真源在 quant.tradability.rolled_sell_close()。"""
    return tradability.rolled_sell_close(close, sell_blocked, horizon, cap)


def _price_targets(codes: list[str], horizons: list[int]) -> pd.DataFrame:
    """可交易口径的价目标/掩码。单一真源在 quant.tradability.price_tradability()。"""
    return tradability.price_tradability(codes, horizons, quant_dir=_quant_dir())


def _ensure_targets(pred: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    need_open = [h for h in horizons if f"open_ret_{h}d" not in pred.columns]
    need_close = [h for h in horizons if f"target_ret_{h}d" not in pred.columns]
    need_tradable = [h for h in horizons if f"tradable_ret_{h}d" not in pred.columns]
    if not need_open and not need_close and not need_tradable \
            and "buyable_next" in pred.columns and "buyable_close" in pred.columns:
        return pred
    targets = _price_targets(sorted(pred["code"].dropna().unique()),
                             sorted(set(need_open) | set(need_close) | set(need_tradable) | set(horizons)))
    if targets.empty:
        return pred
    # 避免重复列冲突：只并入 pred 中尚未存在的列
    keep = ["code", "date"] + [c for c in targets.columns if c not in ("code", "date") and c not in pred.columns]
    return pred.merge(targets[keep], on=["code", "date"], how="left")


def _base_combos(path: Path, kind: str) -> pd.DataFrame:
    template = pd.DataFrame()
    path = Path(path)  # 容错：调用方可能传 str，统一成 Path
    if path.exists():
        loaded = pd.read_parquet(path)
        if kind == "short":
            cols = ["ic_weight", "top_n", "gross_exposure", "slot_weight", "ridge_quantile", "pred_quantile", "naive_weight", "rebalance_stride", "hold_rank_buffer"]
        else:
            cols = ["ic_weight", "max_weight", "ridge_quantile", "pred_quantile", "naive_weight", "rebalance_stride", "hold_rank_buffer"]
        cols = [c for c in cols if c in loaded.columns]
        if cols:
            template = loaded[cols].drop_duplicates().reset_index(drop=True)
            if "rebalance_stride" not in template.columns:
                template["rebalance_stride"] = 1
            else:
                template["rebalance_stride"] = (
                    pd.to_numeric(template["rebalance_stride"], errors="coerce")
                    .fillna(1).clip(lower=1).astype(int)
                )
            if "hold_rank_buffer" not in template.columns:
                template["hold_rank_buffer"] = 0
            else:
                template["hold_rank_buffer"] = (
                    pd.to_numeric(template["hold_rank_buffer"], errors="coerce")
                    .fillna(0).clip(lower=0).astype(int)
                )
    if kind == "short":
        rows = []
        for ic in (0.0, 0.03, 0.06, 0.10, 0.15):
            for top_n in (1, 2, 3, 4, 5):
                for gross in (0.18, 0.24, 0.30):
                    for rq in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
                        for pq in (None, 0.50, 0.55, 0.60, 0.65, 0.70):
                                for naive in (0.0, 0.1, 0.2):
                                    for buffer in (0, 1, 2):
                                        rows.append({
                                            "ic_weight": ic,
                                            "top_n": top_n,
                                            "gross_exposure": gross,
                                            "slot_weight": gross / top_n,
                                            "ridge_quantile": rq,
                                            "pred_quantile": pq,
                                            "naive_weight": naive,
                                            "rebalance_stride": 1,
                                            "hold_rank_buffer": buffer,
                                        })
        standard = pd.DataFrame(rows)
        if template.empty:
            return standard
        return pd.concat([standard, template], ignore_index=True).drop_duplicates().reset_index(drop=True)
    rows = []
    for ic in (0.0, 0.05, 0.10, 0.15):
        for mw in (0.07, 0.08, 0.09, 0.10):
            for rq in (0.30, 0.35, 0.40, 0.45):
                for pq in (None, 0.55, 0.60):
                    for buffer in (0, 1, 2):
                        rows.append({
                            "ic_weight": ic, "max_weight": mw,
                            "ridge_quantile": rq, "pred_quantile": pq,
                            "rebalance_stride": 1,
                            "hold_rank_buffer": buffer,
                        })
    standard = pd.DataFrame(rows)
    if template.empty:
        return standard
    return pd.concat([standard, template], ignore_index=True).drop_duplicates().reset_index(drop=True)


def _template_combos(path: Path, kind: str) -> pd.DataFrame:
    """在基础参数网格上叠加朴素腿权重 naive_weight（rule_score 当日横截面 z 的融合权重）。"""
    base = _base_combos(path, kind)
    if base.empty or "naive_weight" in base.columns:
        return base
    naive = pd.DataFrame({"naive_weight": [0.0, 0.1, 0.2]})
    out = base.assign(_k=1).merge(naive.assign(_k=1), on="_k").drop(columns="_k")
    return out.reset_index(drop=True)


def _evaluate_returns(
    ret: pd.Series, horizon: int, rebalance_stride: int = 1,
) -> dict:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty:
        return {}
    ret = ret.clip(lower=-0.99)
    nav = (1 + ret).cumprod()
    periods_per_year = 252 / max(int(horizon), int(rebalance_stride), 1)
    annual = nav.iloc[-1] ** (periods_per_year / max(len(ret), 1)) - 1 if nav.iloc[-1] > 0 else np.nan
    vol = ret.std() * np.sqrt(periods_per_year)
    peak = nav.cummax()
    return {
        "periods": int(len(ret)),
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": float(annual) if np.isfinite(annual) else None,
        "annual_vol": float(vol),
        "sharpe": float(annual / vol) if vol and np.isfinite(vol) and np.isfinite(annual) else None,
        "max_drawdown": float((nav / peak - 1).min()),
        "win_rate": float((ret > 0).mean()),
    }


def _apply_model_blend(df: pd.DataFrame, lgbm_weight: float | None = None) -> pd.DataFrame:
    out = df.copy()
    if lgbm_weight is None or "lgbm_pred" not in out.columns or "ridge_pred" not in out.columns:
        return out
    weight = float(lgbm_weight)
    lgbm_z = out.groupby("date")["lgbm_pred"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0.0)
    ridge_z = out.groupby("date")["ridge_pred"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0.0)
    out["base_pred"] = weight * lgbm_z.fillna(0.0) + (1.0 - weight) * ridge_z.fillna(0.0)
    return out


def _score_pred(df: pd.DataFrame, ic_weight: float, naive_weight: float = 0.0,
                lgbm_weight: float | None = None, elastic_weight: float = 0.0,
                catboost_weight: float = 0.0, extra_trees_weight: float = 0.0) -> pd.DataFrame:
    out = _apply_model_blend(df, lgbm_weight)
    base = pd.to_numeric(out["base_pred"], errors="coerce")
    if "ic_z" in out.columns:
        ic = pd.to_numeric(out["ic_z"], errors="coerce").fillna(0.0)
    else:
        ic = pd.Series(0.0, index=out.index)
    out["pred"] = base + float(ic_weight) * ic
    if elastic_weight and "elastic_z" in out.columns:
        out["pred"] = out["pred"] + float(elastic_weight) * pd.to_numeric(
            out["elastic_z"], errors="coerce").fillna(0.0)
    if catboost_weight and "catboost_z" in out.columns:
        out["pred"] = out["pred"] + float(catboost_weight) * pd.to_numeric(
            out["catboost_z"], errors="coerce").fillna(0.0)
    if extra_trees_weight and "extra_trees_z" in out.columns:
        out["pred"] = out["pred"] + float(extra_trees_weight) * pd.to_numeric(
            out["extra_trees_z"], errors="coerce").fillna(0.0)
    if naive_weight and "rule_score" in out.columns:
        rz = out.groupby("date")["rule_score"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0.0)
        out["pred"] = out["pred"] + float(naive_weight) * rz.fillna(0.0)
    return out


def _to_matrix(pred: pd.DataFrame, dates: pd.Index, codes: pd.Index, col: str) -> np.ndarray:
    if col not in pred.columns:
        return np.full((len(dates), len(codes)), np.nan, dtype=float)
    wide = pred.pivot_table(index="date", columns="code", values=col, aggfunc="last")
    wide = wide.reindex(index=dates, columns=codes)
    return wide.to_numpy(dtype=float)


def _prepare_fast_grid(pred: pd.DataFrame, horizons: list[int]) -> dict:
    dates = pd.Index(sorted(pred["date"].dropna().unique()))
    codes = pd.Index(sorted(pred["code"].dropna().unique()))
    use_open = backtest.bt_use_open_fill()
    filter_untradable = backtest.bt_filter_untradable()
    # 成交口径：默认收盘。QUANT_BT_FILL=next_open→次日开盘(open_ret)；
    # 收盘口径且开启可交易性(QUANT_BT_FILTER_UNTRADABLE=1,默认开)→用跌停顺延后的 tradable_ret；
    # 关闭可交易性→乐观 target_ret（信号日收盘→h日后收盘，不理会涨跌停）。
    target_mats = {}
    for h in horizons:
        col = None
        if use_open and f"open_ret_{h}d" in pred.columns:
            col = f"open_ret_{h}d"
        elif (not use_open) and filter_untradable and f"tradable_ret_{h}d" in pred.columns:
            col = f"tradable_ret_{h}d"
        elif f"target_ret_{h}d" in pred.columns:
            col = f"target_ret_{h}d"
        elif f"open_ret_{h}d" in pred.columns:
            col = f"open_ret_{h}d"
        if col:
            target_mats[h] = _to_matrix(pred, dates, codes, col)
    # 买入端可交易性：next_open 口径用 buyable_next(次日一字涨停买不进)；
    # 收盘口径用 buyable_close(当日涨停封板/一字涨停尾盘买不进)。
    buyable = None
    if filter_untradable:
        buy_col = "buyable_next" if use_open else "buyable_close"
        if buy_col not in pred.columns:
            raise ValueError(
                f"严格可交易网格缺少 {buy_col}；不能静默降级到乐观口径"
            )
        bm = _to_matrix(pred, dates, codes, buy_col)
        buyable = np.nan_to_num(bm, nan=0.0) > 0.5
    ridge = _to_matrix(pred, dates, codes, "ridge_pred")
    lgbm = _to_matrix(pred, dates, codes, "lgbm_pred")

    def _row_z(values: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            mu = np.nanmean(values, axis=1, keepdims=True)
            sd = np.nanstd(values, axis=1, keepdims=True)
        return np.where(np.isfinite(values) & (sd > 0), (values - mu) / np.where(sd > 0, sd, 1.0), 0.0)

    # 朴素腿：rule_score 的逐日横截面 z-score，供融合
    rule_z = _row_z(_to_matrix(pred, dates, codes, "rule_score"))
    return {
        "dates": dates,
        "codes": codes,
        "base": _to_matrix(pred, dates, codes, "base_pred"),
        "lgbm_z": _row_z(lgbm),
        "ridge_z": _row_z(ridge),
        "elastic_z": np.nan_to_num(_to_matrix(pred, dates, codes, "elastic_z"), nan=0.0),
        "catboost_z": np.nan_to_num(_to_matrix(pred, dates, codes, "catboost_z"), nan=0.0),
        "extra_trees_z": np.nan_to_num(_to_matrix(pred, dates, codes, "extra_trees_z"), nan=0.0),
        "ic": np.nan_to_num(_to_matrix(pred, dates, codes, "ic_z"), nan=0.0),
        "ridge": ridge,
        "rule_z": rule_z,
        "targets": target_mats,
        "buyable": buyable,
        "stride_indices": {},
    }


def _prepared_stride_indices(prepared: dict, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.arange(np.asarray(prepared["ridge"]).shape[0])
    cache = prepared.setdefault("stride_indices", {})
    if stride not in cache:
        dates = pd.DatetimeIndex(prepared["dates"])
        selected = backtest._apply_rebalance_stride(
            pd.DataFrame({"date": dates}),
            stride,
            trading_calendar=_stride_calendar(stride),
        )
        selected_dates = pd.DatetimeIndex(selected["date"])
        cache[stride] = np.flatnonzero(dates.isin(selected_dates))
    return cache[stride]


def _apply_row_quantile(mask: np.ndarray, values: np.ndarray, q: float | None, min_count: int = 5) -> np.ndarray:
    if q is None:
        return mask
    counts = mask.sum(axis=1)
    rows = counts >= min_count
    if not rows.any():
        return mask
    out = mask.copy()
    masked_values = np.where(mask[rows], values[rows], np.nan)
    with np.errstate(all="ignore"):
        threshold = np.nanquantile(masked_values, float(q), axis=1)
    out[rows] &= values[rows] >= threshold[:, None]
    return out


def _fast_combo_metrics(prepared: dict, row: pd.Series, kind: str, horizon: int, positive_only: bool) -> dict | None:
    target = prepared["targets"].get(horizon)
    if target is None:
        return None
    ic_weight = float(_empty_to_none(row.get("ic_weight")) or 0.0)
    naive_weight = float(_empty_to_none(row.get("naive_weight")) or 0.0)
    lgbm_weight = _empty_to_none(row.get("lgbm_weight"))
    if lgbm_weight is None:
        score = prepared["base"].copy()
    else:
        weight = float(lgbm_weight)
        score = weight * prepared["lgbm_z"] + (1.0 - weight) * prepared["ridge_z"]
    score = score + ic_weight * prepared["ic"]
    elastic_weight = float(_empty_to_none(row.get("elastic_weight")) or 0.0)
    if elastic_weight and prepared.get("elastic_z") is not None:
        score = score + elastic_weight * prepared["elastic_z"]
    catboost_weight = float(_empty_to_none(row.get("catboost_weight")) or 0.0)
    if catboost_weight and prepared.get("catboost_z") is not None:
        score = score + catboost_weight * prepared["catboost_z"]
    extra_trees_weight = float(_empty_to_none(row.get("extra_trees_weight")) or 0.0)
    if extra_trees_weight and prepared.get("extra_trees_z") is not None:
        score = score + extra_trees_weight * prepared["extra_trees_z"]
    if naive_weight and prepared.get("rule_z") is not None:
        score = score + naive_weight * prepared["rule_z"]
    stride = _rebalance_stride(row)
    stride_indices = _prepared_stride_indices(prepared, stride)
    score = score[stride_indices]
    target = target[stride_indices]
    evaluable_dates = (np.isfinite(score) & np.isfinite(target)).any(axis=1)
    if not evaluable_dates.any():
        return None
    score = score[evaluable_dates]
    target = target[evaluable_dates]
    top_n = int(row.get("top_n", 3)) if kind == "short" else 3
    raw_max_weight = row.get("slot_weight", row.get("max_weight", 1.0 / max(top_n, 1))) if kind == "short" else row.get("max_weight", 1.0 / max(top_n, 1))
    max_weight = float(raw_max_weight) if _empty_to_none(raw_max_weight) is not None else 1.0 / max(top_n, 1)
    gross_exposure = float(
        _empty_to_none(row.get("gross_exposure"))
        or min(max_weight * max(top_n, 1), 1.0)
    )
    no_refill = True
    pred_quantile = _empty_to_none(row.get("pred_quantile"))
    ridge_quantile = _empty_to_none(row.get("ridge_quantile"))

    mask = np.isfinite(score) & np.isfinite(target)
    if positive_only:
        mask &= score > 0
    buyable = prepared.get("buyable")
    if buyable is not None:
        buyable = buyable[stride_indices][evaluable_dates]
    mask = _apply_row_quantile(mask, score, float(pred_quantile) if pred_quantile is not None else None)
    ridge = prepared["ridge"][stride_indices][evaluable_dates]
    if ridge_quantile is not None and np.isfinite(ridge).any():
        ridge_mask = mask & np.isfinite(ridge)
        mask = _apply_row_quantile(ridge_mask, ridge, float(ridge_quantile))
    if not np.isfinite(score).any() or not np.isfinite(target).any():
        return None

    n_pick = min(max(top_n, 1), score.shape[1])
    hold_rank_buffer = _hold_rank_buffer(row)
    pick_bool = np.zeros_like(mask, dtype=bool)
    if hold_rank_buffer > 0:
        previous_codes: set[str] = set()
        codes = np.asarray(prepared["codes"]).astype(str)
        for date_index in range(mask.shape[0]):
            indices = backtest._select_with_rank_hysteresis(
                codes,
                score[date_index],
                mask[date_index],
                top_n,
                hold_rank_buffer,
                previous_codes,
            )
            actual = np.zeros(mask.shape[1], dtype=bool)
            actual[indices] = True
            if no_refill and buyable is not None:
                actual &= buyable[date_index]
            pick_bool[date_index] = actual
            previous_codes = set(codes[actual])
    else:
        ranked_score = np.where(mask, score, -np.inf)
        order = np.argpartition(-ranked_score, kth=n_pick - 1, axis=1)[:, :n_pick]
        top_score = np.take_along_axis(ranked_score, order, axis=1)
        sort_idx = np.argsort(-top_score, axis=1)
        order = np.take_along_axis(order, sort_idx, axis=1)
        top_score = np.take_along_axis(top_score, sort_idx, axis=1)
        picked = np.isfinite(top_score)
        if no_refill and buyable is not None:
            picked &= np.take_along_axis(buyable, order, axis=1)
        row_idx = np.arange(mask.shape[0])[:, None]
        pick_bool[row_idx, order] = picked
    counts = pick_bool.sum(axis=1)
    weights = np.minimum(max_weight, gross_exposure / np.maximum(counts, 1))
    stock_weights = pick_bool.astype(float) * weights[:, None]
    ret = np.nansum(target * stock_weights, axis=1)
    previous_weights = np.vstack([
        np.zeros((1, stock_weights.shape[1]), dtype=float),
        stock_weights[:-1],
    ])
    turnover = backtest._weight_turnover(previous_weights, stock_weights)

    # 计提调仓成本（按换手比例）；QUANT_BT_COST_ROUNDTRIP=0 时忽略
    _cost = backtest.bt_cost_roundtrip()
    if _cost:
        ret = ret - turnover * _cost

    metrics = _evaluate_returns(pd.Series(ret), horizon, rebalance_stride=stride)
    if not metrics:
        return None
    all_targets = target[pick_bool]
    metrics["direction_win_rate"] = float((all_targets > 0).mean()) if all_targets.size else np.nan
    metrics["avg_turnover"] = float(turnover.mean()) if turnover.size else np.nan
    metrics["avg_holdings"] = float(counts.mean())
    return metrics


def _run_combo(pred: pd.DataFrame, row: pd.Series, kind: str, horizon: int, positive_only: bool) -> dict | None:
    ic_weight = float(_empty_to_none(row.get("ic_weight")) or 0.0)
    naive_weight = float(_empty_to_none(row.get("naive_weight")) or 0.0)
    data = _score_pred(
        pred,
        ic_weight,
        naive_weight,
        lgbm_weight=_empty_to_none(row.get("lgbm_weight")),
        elastic_weight=float(_empty_to_none(row.get("elastic_weight")) or 0.0),
        catboost_weight=float(_empty_to_none(row.get("catboost_weight")) or 0.0),
        extra_trees_weight=float(_empty_to_none(row.get("extra_trees_weight")) or 0.0),
    )
    top_n = int(row.get("top_n", 3)) if kind == "short" else 3
    if kind == "short":
        max_weight = float(row.get("slot_weight", row.get("max_weight", 1.0 / max(top_n, 1))))
    else:
        max_weight = float(row.get("max_weight", 1.0 / max(top_n, 1)))
    pred_quantile = _empty_to_none(row.get("pred_quantile"))
    ridge_quantile = _empty_to_none(row.get("ridge_quantile"))
    stride = _rebalance_stride(row)
    data = _stride_predictions(data, row)
    returns, holdings = backtest.portfolio_from_predictions(
        data,
        horizon=horizon,
        top_n=top_n,
        max_weight=max_weight,
        positive_only=positive_only,
        pred_quantile=float(pred_quantile) if pred_quantile is not None else None,
        ridge_quantile=float(ridge_quantile) if ridge_quantile is not None else None,
        filter_untradable=True,
        no_refill=True,
        require_tradability=True,
        hold_rank_buffer=_hold_rank_buffer(row),
    )
    if returns.empty or holdings.empty:
        return None
    metrics = _evaluate_returns(
        returns["ret"], horizon, rebalance_stride=stride,
    )
    if not metrics:
        return None
    target = f"target_ret_{horizon}d"
    metrics["direction_win_rate"] = float((pd.to_numeric(holdings[target], errors="coerce") > 0).mean()) if target in holdings else np.nan
    metrics["avg_turnover"] = float(returns["turnover"].mean()) if "turnover" in returns else np.nan
    metrics["avg_holdings"] = float(returns["n_holdings"].mean()) if "n_holdings" in returns else np.nan
    return metrics


def _selection_summary(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    numeric = grid.copy()
    for col in ("sharpe", "annual_return", "max_drawdown", "win_rate", "direction_win_rate", "avg_turnover"):
        numeric[col] = pd.to_numeric(numeric.get(col), errors="coerce")
    group_cols = [c for c in ("param_id", "source", "lgbm_weight", "ic_weight", "elastic_weight", "catboost_weight", "extra_trees_weight", "top_n", "gross_exposure", "slot_weight", "max_weight", "ridge_quantile", "pred_quantile", "naive_weight", "rebalance_stride", "hold_rank_buffer") if c in numeric.columns]
    out = numeric.groupby(group_cols, dropna=False).agg(
        horizons=("horizon", lambda s: "/".join(str(int(x)) for x in sorted(s.dropna().unique()))),
        avg_sharpe=("sharpe", "mean"),
        avg_annual_return=("annual_return", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        avg_win_rate=("win_rate", "mean"),
        avg_direction_win_rate=("direction_win_rate", "mean"),
        avg_turnover=("avg_turnover", "mean"),
    ).reset_index()
    gross = pd.to_numeric(out.get("gross_exposure"), errors="coerce")
    if gross is None:
        gross = pd.Series(np.nan, index=out.index)
    max_weight = pd.to_numeric(out.get("max_weight"), errors="coerce")
    top_n = pd.to_numeric(out.get("top_n"), errors="coerce")
    if max_weight is not None:
        if top_n is None:
            top_n = pd.Series(3.0, index=out.index)
        gross = gross.fillna(max_weight * top_n)
    gross = gross.where(gross > 0)
    normalized_annual_return = out["avg_annual_return"] / gross
    normalized_drawdown = out["worst_drawdown"] / gross
    normalized_turnover = out["avg_turnover"] / gross
    out["selection_score"] = (
        out["avg_sharpe"].fillna(0.0)
        + (out["avg_win_rate"].fillna(0.5) - 0.5) * 2.0
        + (out["avg_direction_win_rate"].fillna(0.5) - 0.5)
        + normalized_annual_return.fillna(0.0) * 0.15
        + normalized_drawdown.fillna(-1.0) * 0.25
        - normalized_turnover.fillna(1.0) * 0.05
    )
    return out.sort_values("selection_score", ascending=False).reset_index(drop=True)


def prepare_fixed_context(predictions: Path, horizons: list[int], watchlist: set[str],
                          start_date: str | pd.Timestamp | None = None,
                          end_date: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Load and prepare one holdout dataset for evaluating multiple parameter sets."""
    pred = _load_predictions(predictions, watchlist)
    if start_date is not None:
        pred = pred[pred["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        pred = pred[pred["date"] < pd.Timestamp(end_date)].copy()
    if pred.empty:
        return pred, {}
    pred = _ensure_targets(pred, horizons)
    return pred, _prepare_fast_grid(pred, horizons)


def evaluate_prepared_params(prepared: dict, params: dict, horizons: list[int],
                             kind: str, positive_only: bool) -> pd.DataFrame:
    """Evaluate fixed parameters using an already prepared holdout matrix."""
    if not prepared:
        return pd.DataFrame()
    normalized = dict(params)
    if kind == "short" and normalized.get("slot_weight") is None and normalized.get("max_weight") is None:
        top_n = max(int(normalized.get("top_n", 3)), 1)
        if normalized.get("gross_exposure") is not None:
            normalized["slot_weight"] = float(normalized["gross_exposure"]) / top_n
    row = pd.Series(normalized)
    rows = []
    for horizon in horizons:
        metrics = _fast_combo_metrics(prepared, row, kind, int(horizon), positive_only)
        if metrics:
            rows.append({"horizon": int(horizon), **params, **metrics})
    return pd.DataFrame(rows)


def evaluate_fixed_params(predictions: Path, params: dict, horizons: list[int], kind: str,
                          watchlist: set[str], positive_only: bool,
                          start_date: str | pd.Timestamp | None = None,
                          end_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Evaluate one frozen parameter set without tuning; used by the promotion holdout gate."""
    pred = _load_predictions(predictions, watchlist)
    if pred.empty:
        return pd.DataFrame()
    if start_date is not None:
        pred = pred[pred["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        pred = pred[pred["date"] < pd.Timestamp(end_date)].copy()
    if pred.empty:
        return pd.DataFrame()
    pred = _ensure_targets(pred, horizons)
    prepared = _prepare_fast_grid(pred, horizons)
    return evaluate_prepared_params(prepared, params, horizons, kind, positive_only)


def evaluate_prepared_returns(pred: pd.DataFrame, params: dict, horizons: list[int], kind: str,
                              positive_only: bool) -> dict[int, pd.DataFrame]:
    """Return dated portfolio returns from an already loaded holdout dataset."""
    if pred.empty:
        return {}
    normalized = dict(params)
    pred = _apply_model_blend(pred, _empty_to_none(normalized.get("lgbm_weight")))
    pred = _score_pred(
        pred,
        float(_empty_to_none(normalized.get("ic_weight")) or 0.0),
        float(_empty_to_none(normalized.get("naive_weight")) or 0.0),
        elastic_weight=float(_empty_to_none(normalized.get("elastic_weight")) or 0.0),
        catboost_weight=float(_empty_to_none(normalized.get("catboost_weight")) or 0.0),
        extra_trees_weight=float(_empty_to_none(normalized.get("extra_trees_weight")) or 0.0),
    )
    top_n = int(normalized.get("top_n", 3)) if kind == "short" else 3
    max_weight = _empty_to_none(normalized.get("slot_weight", normalized.get("max_weight")))
    if max_weight is None and normalized.get("gross_exposure") is not None:
        max_weight = float(normalized["gross_exposure"]) / max(top_n, 1)
    if max_weight is None:
        max_weight = 1.0 / max(top_n, 1)
    pred_quantile = _empty_to_none(normalized.get("pred_quantile"))
    ridge_quantile = _empty_to_none(normalized.get("ridge_quantile"))
    stride = _rebalance_stride(normalized)
    pred = _stride_predictions(pred, normalized)
    result: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        returns, _ = backtest.portfolio_from_predictions(
            pred,
            horizon=int(horizon),
            top_n=top_n,
            max_weight=float(max_weight),
            positive_only=positive_only,
            pred_quantile=float(pred_quantile) if pred_quantile is not None else None,
            ridge_quantile=float(ridge_quantile) if ridge_quantile is not None else None,
            filter_untradable=True,
            no_refill=True,
            require_tradability=True,
            hold_rank_buffer=_hold_rank_buffer(normalized),
        )
        if not returns.empty:
            returns = returns.copy()
            returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
            returns["rebalance_stride"] = stride
            returns["hold_rank_buffer"] = _hold_rank_buffer(normalized)
            result[int(horizon)] = returns.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return result


def evaluate_fixed_returns(predictions: Path, params: dict, horizons: list[int], kind: str,
                           watchlist: set[str], positive_only: bool,
                           start_date: str | pd.Timestamp | None = None,
                           end_date: str | pd.Timestamp | None = None) -> dict[int, pd.DataFrame]:
    """Return dated portfolio returns for non-overlap and monthly stability checks."""
    pred, _ = prepare_fixed_context(predictions, horizons, watchlist, start_date, end_date)
    return evaluate_prepared_returns(pred, params, horizons, kind, positive_only)


def stability_decision(candidate: dict[int, pd.DataFrame], baseline: dict[int, pd.DataFrame],
                       min_significant_sharpe_gain: float = 0.50,
                       max_other_sharpe_decline: float = 0.50,
                       min_monthly_win_rate: float = 2 / 3) -> dict:
    """Accept one clear non-overlap win when all other horizons are broadly stable."""
    details = []
    improved = significant = acceptable = 0
    monthly_win_rates = []
    for horizon in sorted(set(candidate) & set(baseline)):
        candidate_frame = candidate[horizon]
        stride_values = pd.to_numeric(
            candidate_frame.get("rebalance_stride", pd.Series([1])),
            errors="coerce",
        ).dropna()
        stride = max(int(stride_values.max()) if not stride_values.empty else 1, 1)
        merged = candidate_frame[["date", "ret"]].merge(
            baseline[horizon][["date", "ret"]], on="date", suffixes=("_candidate", "_baseline"))
        if merged.empty:
            continue
        sampling_step = max(int(np.ceil(int(horizon) / stride)), 1)
        sampled = merged.iloc[::sampling_step].copy()
        effective_stride = stride * sampling_step
        c_metrics = _evaluate_returns(
            sampled["ret_candidate"], int(horizon),
            rebalance_stride=effective_stride,
        )
        b_metrics = _evaluate_returns(
            sampled["ret_baseline"], int(horizon),
            rebalance_stride=effective_stride,
        )
        c_sharpe = c_metrics.get("sharpe")
        b_sharpe = b_metrics.get("sharpe")
        sharpe_gain = None
        is_significant = is_acceptable = False
        if c_sharpe is not None and b_sharpe is not None:
            sharpe_gain = float(c_sharpe - b_sharpe)
            if sharpe_gain > 0:
                improved += 1
            is_significant = sharpe_gain >= float(min_significant_sharpe_gain)
            is_acceptable = sharpe_gain >= -float(max_other_sharpe_decline)
            significant += int(is_significant)
            acceptable += int(is_acceptable)
        monthly = merged.set_index("date")[["ret_candidate", "ret_baseline"]].resample("ME").apply(
            lambda x: (1.0 + x).prod() - 1.0)
        valid_months = monthly.dropna()
        wins = int((valid_months["ret_candidate"] > valid_months["ret_baseline"]).sum())
        months = int(len(valid_months))
        monthly_win_rate = wins / months if months else 0.0
        monthly_win_rates.append(monthly_win_rate)
        details.append({"horizon": int(horizon), "nonoverlap_candidate_sharpe": c_sharpe,
                        "nonoverlap_baseline_sharpe": b_sharpe,
                        "nonoverlap_sharpe_gain": sharpe_gain,
                        "significant_improvement": is_significant,
                        "within_allowed_decline": is_acceptable,
                        "monthly_wins": wins, "months": int(len(valid_months))})
    common = len(details)
    monthly_win_rate = (
        float(np.mean(monthly_win_rates)) if monthly_win_rates else 0.0
    )
    passed = (
        significant >= 1
        and acceptable == common
        and common > 0
        and monthly_win_rates
        and all(rate >= float(min_monthly_win_rate) for rate in monthly_win_rates)
    )
    return {"passed": bool(passed), "nonoverlap_improved_horizons": improved,
            "significant_improved_horizons": significant,
            "acceptable_horizons": acceptable,
            "min_significant_sharpe_gain": float(min_significant_sharpe_gain),
            "max_other_sharpe_decline": float(max_other_sharpe_decline),
            "common_horizons": common, "monthly_win_rate": monthly_win_rate,
            "monthly_win_rates": monthly_win_rates,
            "monthly_wins": int(sum(int(d["monthly_wins"]) for d in details)),
            "months": int(sum(int(d["months"]) for d in details)), "details": details}


def promotion_decision(candidate: pd.DataFrame, baseline: pd.DataFrame,
                       min_sharpe_gain: float = 0.10, max_drawdown_worsening: float = 0.02,
                       min_improved_horizons: int = 2) -> dict:
    """Conservative candidate gate on an untouched holdout period."""
    merged = candidate.merge(baseline, on="horizon", suffixes=("_candidate", "_baseline"))
    if merged.empty or len(merged) < min_improved_horizons:
        return {"promote": False, "reason": "insufficient_common_horizons", "common_horizons": len(merged)}
    c_sharpe = pd.to_numeric(merged["sharpe_candidate"], errors="coerce")
    b_sharpe = pd.to_numeric(merged["sharpe_baseline"], errors="coerce")
    c_dd = pd.to_numeric(merged["max_drawdown_candidate"], errors="coerce")
    b_dd = pd.to_numeric(merged["max_drawdown_baseline"], errors="coerce")
    valid = c_sharpe.notna() & b_sharpe.notna() & c_dd.notna() & b_dd.notna()
    merged = merged[valid].copy()
    if len(merged) < min_improved_horizons:
        return {"promote": False, "reason": "insufficient_valid_metrics", "common_horizons": len(merged)}
    merged["sharpe_gain"] = c_sharpe[valid].to_numpy() - b_sharpe[valid].to_numpy()
    merged["drawdown_change"] = c_dd[valid].to_numpy() - b_dd[valid].to_numpy()
    avg_gain = float(merged["sharpe_gain"].mean())
    improved = int((merged["sharpe_gain"] > 0).sum())
    worst_dd_change = float(merged["drawdown_change"].min())
    promote = avg_gain >= float(min_sharpe_gain) and improved >= int(min_improved_horizons) and worst_dd_change >= -float(max_drawdown_worsening)
    reason = "passed" if promote else "gain_or_stability_threshold_not_met"
    return {
        "promote": bool(promote), "reason": reason,
        "avg_sharpe_gain": avg_gain, "improved_horizons": improved,
        "common_horizons": int(len(merged)), "worst_drawdown_change": worst_dd_change,
        "details": merged[["horizon", "sharpe_candidate", "sharpe_baseline", "sharpe_gain",
                           "max_drawdown_candidate", "max_drawdown_baseline", "drawdown_change"]].to_dict("records"),
    }


_GRID_PRED: pd.DataFrame | None = None
_GRID_PREPARED: dict | None = None
_GRID_HORIZONS: list[int] = []
_GRID_KIND = "short"
_GRID_POSITIVE_ONLY = True
_GRID_SOURCE = ""


def _evaluate_combo_process(item: tuple[int, pd.Series]) -> list[dict]:
    """Evaluate one combo in a worker using fork-inherited read-only grid data."""
    if _GRID_PRED is None or _GRID_PREPARED is None:
        raise RuntimeError("grid worker state is not initialized")
    param_id, combo = item
    combo_rows: list[dict] = []
    with threadpool_limits(limits=1, user_api="blas"):
        for horizon in _GRID_HORIZONS:
            if horizon not in _GRID_PREPARED["targets"]:
                continue
            metrics = _fast_combo_metrics(
                _GRID_PREPARED, combo, _GRID_KIND, horizon, _GRID_POSITIVE_ONLY
            )
            if not metrics:
                metrics = _run_combo(
                    _GRID_PRED, combo, _GRID_KIND, horizon, _GRID_POSITIVE_ONLY
                )
            if not metrics:
                continue
            row = combo.to_dict()
            row.update({
                "param_id": param_id,
                "source": _GRID_SOURCE,
                "horizon": int(horizon),
                **metrics,
            })
            combo_rows.append(row)
    return combo_rows


def run_grid(predictions: Path, template: Path, output: Path, best_output: Path, horizons: list[int],
             kind: str, watchlist: set[str], positive_only: bool,
             start_date: str | pd.Timestamp | None = None,
             end_date: str | pd.Timestamp | None = None,
             fixed_params: dict | None = None,
             catboost_weights: list[float] | None = None,
             neighborhood: dict | None = None,
             workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, template, output, best_output = (
        Path(predictions), Path(template), Path(output), Path(best_output))
    pred = _load_predictions(predictions, watchlist)
    if start_date is not None:
        pred = pred[pred["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        pred = pred[pred["date"] < pd.Timestamp(end_date)].copy()
    if pred.empty:
        raise RuntimeError(f"prediction file has no rows in requested range: {predictions}")
    pred = _ensure_targets(pred, horizons)
    combos = _template_combos(template, kind)
    for key, value in (fixed_params or {}).items():
        combos[key] = value
    if catboost_weights:
        weights = pd.DataFrame({"catboost_weight": [float(x) for x in catboost_weights]})
        combos = combos.assign(_k=1).merge(weights.assign(_k=1), on="_k").drop(columns="_k")
    if neighborhood:
        limits = {
            "ic_weight": 0.03,
            "top_n": 1.0,
            "ridge_quantile": 0.05,
            "pred_quantile": 0.05,
            "naive_weight": 0.10,
        }
        mask = pd.Series(True, index=combos.index)
        for key, limit in limits.items():
            anchor = _empty_to_none(neighborhood.get(key))
            if anchor is None or key not in combos.columns:
                continue
            values = pd.to_numeric(combos[key], errors="coerce")
            mask &= values.notna() & ((values - float(anchor)).abs() <= limit + 1e-12)
        if neighborhood.get("gross_exposure") is not None and "gross_exposure" in combos.columns:
            gross = pd.to_numeric(combos["gross_exposure"], errors="coerce")
            mask &= (gross - float(neighborhood["gross_exposure"])).abs() <= 1e-12
        combos = combos[mask].reset_index(drop=True)
        if combos.empty:
            raise RuntimeError("grid neighborhood contains no parameter combinations")
    prepared = _prepare_fast_grid(pred, horizons)
    rows: list[dict] = []
    source = predictions.name
    items = [(int(idx) + 1, combo) for idx, combo in combos.iterrows()]

    global _GRID_PRED, _GRID_PREPARED, _GRID_HORIZONS, _GRID_KIND
    global _GRID_POSITIVE_ONLY, _GRID_SOURCE
    _GRID_PRED = pred
    _GRID_PREPARED = prepared
    _GRID_HORIZONS = list(horizons)
    _GRID_KIND = kind
    _GRID_POSITIVE_ONLY = positive_only
    _GRID_SOURCE = source

    worker_count = max(int(workers or min(8, os.cpu_count() or 1)), 1)
    if worker_count > (os.cpu_count() or worker_count):
        raise ValueError(
            f"workers={worker_count} exceeds available CPUs={os.cpu_count()}"
        )
    use_processes = os.name == "posix" and "fork" in mp.get_all_start_methods()
    if use_processes:
        context = mp.get_context("fork")
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {"max_workers": worker_count, "mp_context": context}
    else:
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {"max_workers": worker_count}
    with executor_cls(**executor_kwargs) as executor:
        chunksize = max(1, min(16, len(items) // max(worker_count * 8, 1)))
        for position, combo_rows in enumerate(
            executor.map(_evaluate_combo_process, items, chunksize=chunksize), start=1
        ):
            param_id = items[position - 1][0]
            if param_id == 1 or param_id % 200 == 0:
                print(f"[grid] combo={param_id}/{len(combos)}", flush=True)
            rows.extend(combo_rows)
    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("grid produced no rows")
    first_cols = ["param_id", "source"]
    grid = grid[[c for c in first_cols if c in grid.columns] + [c for c in grid.columns if c not in first_cols]]
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output, index=False)
    best = _selection_summary(grid)
    if not best.empty:
        best.to_parquet(best_output, index=False)
    return grid, best


def main() -> None:
    ap = argparse.ArgumentParser(description="Run watchlist multi-horizon grid on prediction artifact")
    ap.add_argument("--predictions", required=True, help="prediction parquet file, absolute or relative to QUANT_DATA_DIR")
    ap.add_argument("--template", required=True, help="existing grid parquet used as parameter template")
    ap.add_argument("--output", required=True, help="output grid parquet")
    ap.add_argument("--best-output", default="", help="output best-summary parquet; default derives from --output")
    ap.add_argument("--horizons", required=True, help="comma-separated horizons, e.g. 1,2,3")
    ap.add_argument(
        "--rebalance-stride", type=int, default=1,
        help="evaluate every N authoritative exchange sessions; 1 keeps daily evaluation",
    )
    ap.add_argument(
        "--hold-rank-buffer", type=int, default=None,
        help="fix held-name rank buffer; omit to search the registered grid dimension",
    )
    ap.add_argument("--kind", choices=["short", "swing"], required=True)
    ap.add_argument("--watchlist", default=config.MAINBOARD_UNIVERSE_FILE)
    ap.add_argument("--start-date", default="", help="inclusive evaluation start date")
    ap.add_argument("--end-date", default="", help="exclusive evaluation end date")
    ap.add_argument("--no-positive-only", action="store_true")
    ap.add_argument(
        "--workers", type=int, default=8,
        help="parallel grid workers; capped by available CPUs (default: 8)",
    )
    args = ap.parse_args()

    predictions = _resolve(args.predictions)
    template = _resolve(args.template)
    output = _resolve(args.output)
    best_output = _resolve(args.best_output) if args.best_output else output.with_name(output.stem + "_best.parquet")
    horizons = _parse_horizons(args.horizons)
    watchlist = _read_watchlist(args.watchlist)
    grid, best = run_grid(
        predictions=predictions,
        template=template,
        output=output,
        best_output=best_output,
        horizons=horizons,
        kind=args.kind,
        watchlist=watchlist,
        positive_only=not args.no_positive_only,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        fixed_params={
            "rebalance_stride": max(int(args.rebalance_stride), 1),
            **(
                {"hold_rank_buffer": max(int(args.hold_rank_buffer), 0)}
                if args.hold_rank_buffer is not None else {}
            ),
        },
        workers=args.workers,
    )
    print(f"[grid] rows={len(grid)} horizons={sorted(grid['horizon'].dropna().astype(int).unique())} -> {output}", flush=True)
    if not best.empty:
        print("[best]", flush=True)
        print(best.head(10).to_string(index=False), flush=True)
        print(f"[best] -> {best_output}", flush=True)


if __name__ == "__main__":
    main()
