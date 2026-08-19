"""Conservative selection/holdout evaluation for optional prediction legs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, watchlist_grid


def _optional_leg_holdings(
    frame: pd.DataFrame,
    params: dict,
    horizons: list[int],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    scored = watchlist_grid._apply_model_blend(  # noqa: SLF001
        frame, watchlist_grid._empty_to_none(params.get("lgbm_weight"))  # noqa: SLF001
    )
    scored = watchlist_grid._score_pred(  # noqa: SLF001
        scored,
        float(watchlist_grid._empty_to_none(params.get("ic_weight")) or 0.0),  # noqa: SLF001
        float(watchlist_grid._empty_to_none(params.get("naive_weight")) or 0.0),  # noqa: SLF001
        elastic_weight=float(watchlist_grid._empty_to_none(params.get("elastic_weight")) or 0.0),  # noqa: SLF001
        catboost_weight=float(watchlist_grid._empty_to_none(params.get("catboost_weight")) or 0.0),  # noqa: SLF001
        extra_trees_weight=float(
            watchlist_grid._empty_to_none(params.get("extra_trees_weight")) or 0.0  # noqa: SLF001
        ),
    )
    top_n = int(params.get("top_n", 3))
    max_weight = watchlist_grid._empty_to_none(  # noqa: SLF001
        params.get("slot_weight", params.get("max_weight"))
    )
    if max_weight is None and params.get("gross_exposure") is not None:
        max_weight = float(params["gross_exposure"]) / max(top_n, 1)
    if max_weight is None:
        max_weight = 1.0 / max(top_n, 1)
    pred_quantile = watchlist_grid._empty_to_none(params.get("pred_quantile"))  # noqa: SLF001
    ridge_quantile = watchlist_grid._empty_to_none(params.get("ridge_quantile"))  # noqa: SLF001
    returns: dict[int, pd.DataFrame] = {}
    holdings: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        period_returns, period_holdings = backtest.portfolio_from_predictions(
            scored,
            horizon=int(horizon),
            top_n=top_n,
            max_weight=float(max_weight),
            positive_only=True,
            pred_quantile=float(pred_quantile) if pred_quantile is not None else None,
            ridge_quantile=float(ridge_quantile) if ridge_quantile is not None else None,
        )
        if not period_returns.empty:
            returns[int(horizon)] = period_returns
        if not period_holdings.empty:
            holdings[int(horizon)] = period_holdings
    return returns, holdings


def attribute_optional_leg(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    selected_weight: float,
    output: Path,
    industry_meta: Path | None = None,
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Explain a frozen optional leg without using attribution results for selection."""
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"optional leg is empty: {leg_column}")
    frame["optional_leg_z"] = frame.groupby("date")[leg_column].transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0)
            if values.std(ddof=0)
            else 0.0
        )
    ).fillna(0.0)
    frame["catboost_z"] = frame["optional_leg_z"]
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    horizon_values = [int(value) for value in horizons]
    frame = frame[frame["date"] >= holdout].copy()
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001

    def parameters(weight: float) -> dict:
        result = dict(champion_params)
        result["catboost_weight"] = float(weight)
        return result

    baseline_returns, baseline_holdings = _optional_leg_holdings(
        frame, parameters(0.0), horizon_values
    )
    candidate_returns, candidate_holdings = _optional_leg_holdings(
        frame, parameters(selected_weight), horizon_values
    )
    industry_by_code = pd.Series(dtype="string")
    industry_updated_at = None
    if industry_meta is not None:
        metadata = pd.read_parquet(industry_meta)
        if {"code", "a_industry"}.issubset(metadata.columns):
            metadata = metadata.copy()
            metadata["code"] = metadata["code"].astype(str).str.zfill(6)
            industry_by_code = metadata.drop_duplicates("code", keep="last").set_index("code")[
                "a_industry"
            ]
            if "meta_updated_at" in metadata:
                updated = pd.to_datetime(metadata["meta_updated_at"], errors="coerce").max()
                industry_updated_at = updated.isoformat() if pd.notna(updated) else None

    horizon_reports = []
    monthly_parts = []
    stock_parts = []
    industry_parts = []
    for horizon in horizon_values:
        base_return = baseline_returns.get(horizon, pd.DataFrame())
        candidate_return = candidate_returns.get(horizon, pd.DataFrame())
        base_hold = baseline_holdings.get(horizon, pd.DataFrame())
        candidate_hold = candidate_holdings.get(horizon, pd.DataFrame())
        if base_return.empty or candidate_return.empty:
            continue
        daily = base_return[["date", "ret", "turnover"]].merge(
            candidate_return[["date", "ret", "turnover"]],
            on="date",
            suffixes=("_baseline", "_candidate"),
        )
        daily["gain"] = daily["ret_candidate"] - daily["ret_baseline"]
        daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
        monthly = daily.groupby("month", as_index=False)["gain"].sum()
        monthly["horizon"] = horizon
        monthly_parts.append(monthly)

        base_sets = base_hold.groupby("date")["code"].agg(lambda values: set(values.astype(str)))
        candidate_sets = candidate_hold.groupby("date")["code"].agg(
            lambda values: set(values.astype(str))
        )
        common_dates = base_sets.index.intersection(candidate_sets.index)
        jaccard = [
            len(base_sets.loc[date] & candidate_sets.loc[date])
            / max(len(base_sets.loc[date] | candidate_sets.loc[date]), 1)
            for date in common_dates
        ]
        horizon_reports.append({
            "horizon": horizon,
            "days": int(len(daily)),
            "mean_daily_gain": float(daily["gain"].mean()),
            "positive_gain_day_rate": float((daily["gain"] > 0).mean()),
            "avg_pick_jaccard": float(np.mean(jaccard)) if jaccard else None,
            "avg_turnover_baseline": float(daily["turnover_baseline"].mean()),
            "avg_turnover_candidate": float(daily["turnover_candidate"].mean()),
        })

        target = f"target_ret_{horizon}d"
        for label, holdings_frame, sign in (
            ("baseline", base_hold, -1.0),
            ("candidate", candidate_hold, 1.0),
        ):
            if holdings_frame.empty or target not in holdings_frame:
                continue
            contribution = holdings_frame.copy()
            contribution["contribution"] = (
                pd.to_numeric(contribution[target], errors="coerce")
                * pd.to_numeric(contribution["weight"], errors="coerce")
                * sign
            )
            contribution["portfolio"] = label
            contribution["horizon"] = horizon
            stock_parts.append(
                contribution.groupby(["horizon", "code"], as_index=False)["contribution"].sum()
            )
            if not industry_by_code.empty:
                contribution["industry"] = contribution["code"].map(industry_by_code).fillna("UNKNOWN")
                industry_parts.append(
                    contribution.groupby(["horizon", "industry"], as_index=False)[
                        "contribution"
                    ].sum()
                )

    monthly_frame = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    stock_frame = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame()
    industry_frame = pd.concat(industry_parts, ignore_index=True) if industry_parts else pd.DataFrame()

    def concentration(values: pd.Series) -> dict:
        absolute = values.abs().sort_values(ascending=False)
        total = float(absolute.sum())
        return {
            "top1_absolute_share": float(absolute.head(1).sum() / total) if total else None,
            "top3_absolute_share": float(absolute.head(3).sum() / total) if total else None,
        }

    report = {
        "diagnostic_only": True,
        "publishable": False,
        "reason": "attribution uses a current industry snapshot and cannot establish PIT validity",
        "selected_weight": float(selected_weight),
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "industry_meta": str(industry_meta) if industry_meta else None,
        "industry_updated_at": industry_updated_at,
        "horizons": horizon_reports,
        "monthly_concentration": concentration(monthly_frame["gain"]) if not monthly_frame.empty else {},
        "top_months": (
            monthly_frame.sort_values("gain", ascending=False).head(10).to_dict("records")
            if not monthly_frame.empty else []
        ),
        "bottom_months": (
            monthly_frame.sort_values("gain").head(10).to_dict("records")
            if not monthly_frame.empty else []
        ),
        "stock_concentration": (
            concentration(stock_frame.groupby("code")["contribution"].sum())
            if not stock_frame.empty else {}
        ),
        "top_stock_contributors": (
            stock_frame.groupby("code", as_index=False)["contribution"].sum()
            .sort_values("contribution", ascending=False).head(20).to_dict("records")
            if not stock_frame.empty else []
        ),
        "top_industry_contributors": (
            industry_frame.groupby("industry", as_index=False)["contribution"].sum()
            .sort_values("contribution", ascending=False).head(20).to_dict("records")
            if not industry_frame.empty else []
        ),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_optional_leg(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    output: Path,
    weights: tuple[float, ...] = (0.0, 0.03, 0.05, 0.10),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame.columns or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"optional leg is empty: {leg_column}")
    frame["optional_leg_z"] = frame.groupby("date")[leg_column].transform(
        lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) else 0.0
    ).fillna(0.0)
    frame["catboost_z"] = frame["optional_leg_z"]
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    horizon_values = [int(value) for value in horizons]

    def prepare(period_start: pd.Timestamp, period_end: pd.Timestamp):
        subset = frame[(frame["date"] >= period_start) & (frame["date"] < period_end)].copy()
        subset = watchlist_grid._ensure_targets(subset, horizon_values)  # noqa: SLF001
        return subset, watchlist_grid._prepare_fast_grid(subset, horizon_values)  # noqa: SLF001

    selection_frame, selection_prepared = prepare(start, holdout)
    holdout_frame, holdout_prepared = prepare(holdout, end)

    def parameters(weight: float) -> dict:
        result = dict(champion_params)
        result["catboost_weight"] = float(weight)
        return result

    selection = []
    for weight in weights:
        metrics = watchlist_grid.evaluate_prepared_params(
            selection_prepared, parameters(weight), horizon_values, "short", True
        )
        if metrics.empty or "sharpe" not in metrics or "max_drawdown" not in metrics:
            selection.append({
                "weight": float(weight),
                "avg_sharpe": None,
                "worst_drawdown": None,
                "details": [],
            })
            continue
        selection.append({
            "weight": float(weight),
            "avg_sharpe": float(pd.to_numeric(metrics["sharpe"], errors="coerce").mean()),
            "worst_drawdown": float(pd.to_numeric(metrics["max_drawdown"], errors="coerce").min()),
            "details": metrics.to_dict("records"),
        })
    baseline_selection = next(row for row in selection if row["weight"] == 0.0)
    if baseline_selection["avg_sharpe"] is None:
        raise ValueError("optional leg baseline has no evaluable returns")
    eligible = [
        row for row in selection
        if row["weight"] > 0
        and row["avg_sharpe"] is not None
        and row["worst_drawdown"] is not None
        and row["avg_sharpe"] > baseline_selection["avg_sharpe"]
        and row["worst_drawdown"] >= baseline_selection["worst_drawdown"] - 0.02
    ]
    winner = max(eligible, key=lambda row: row["avg_sharpe"]) if eligible else {"weight": 0.0}
    selected_weight = float(winner["weight"])
    baseline_params = parameters(0.0)
    candidate_params = parameters(selected_weight)
    baseline_eval = watchlist_grid.evaluate_prepared_params(
        holdout_prepared, baseline_params, horizon_values, "short", True
    )
    candidate_eval = watchlist_grid.evaluate_prepared_params(
        holdout_prepared, candidate_params, horizon_values, "short", True
    )
    promotion = watchlist_grid.promotion_decision(
        candidate_eval,
        baseline_eval,
        min_sharpe_gain=0.10,
        max_drawdown_worsening=0.02,
        min_improved_horizons=1,
    )
    baseline_returns = watchlist_grid.evaluate_prepared_returns(
        holdout_frame, baseline_params, horizon_values, "short", True
    )
    candidate_returns = watchlist_grid.evaluate_prepared_returns(
        holdout_frame, candidate_params, horizon_values, "short", True
    )
    stability = watchlist_grid.stability_decision(
        candidate_returns, baseline_returns,
        search_family_size=sum(1 for row in selection if float(row["weight"]) > 0),
    )
    evaluated_horizons = sorted(
        int(value) for value in baseline_eval.get("horizon", pd.Series(dtype=int)).dropna().unique()
    )
    signal_by_horizon = []
    for horizon in evaluated_horizons:
        target = f"target_ret_{horizon}d"
        daily_ic = []
        for _, group in holdout_frame[["date", leg_column, target]].dropna().groupby("date"):
            if len(group) >= 20:
                daily_ic.append(group[leg_column].corr(group[target], method="spearman"))
        signal_by_horizon.append({
            "horizon": int(horizon),
            "daily_rank_ic_mean": float(np.nanmean(daily_ic)) if daily_ic else None,
            "positive_ic_day_rate": float(np.mean(np.asarray(daily_ic) > 0)) if daily_ic else None,
            "days": len(daily_ic),
        })
    existing = [
        column for column in ("ridge_pred", "lgbm_pred", "elastic_pred", "extra_trees_pred")
        if column in holdout_frame.columns
    ]
    ranks = pd.DataFrame({
        column: holdout_frame.groupby("date")[column].rank(pct=True)
        for column in existing + [leg_column]
    })
    report = {
        "leg_column": leg_column,
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "requested_horizons": horizon_values,
        "evaluated_horizons": evaluated_horizons,
        "backtest_settings": {
            "fill": "next_open" if backtest.bt_use_open_fill() else "close",
            "filter_untradable": backtest.bt_filter_untradable(),
            "cost_roundtrip": backtest.bt_cost_roundtrip(),
        },
        "selection": selection,
        "selected_weight": selected_weight,
        "independent_signal": signal_by_horizon,
        "rank_correlation": ranks.corr(method="spearman").to_dict(),
        "holdout_baseline": baseline_eval.to_dict("records"),
        "holdout_candidate": candidate_eval.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(selected_weight > 0 and promotion.get("promote") and stability.get("passed")),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
