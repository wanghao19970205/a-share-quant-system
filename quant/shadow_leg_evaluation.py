"""Conservative selection/holdout evaluation for optional prediction legs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import watchlist_grid


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
    frame["code"] = frame["code"].astype(str)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame.columns or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"optional leg is empty: {leg_column}")
    frame["catboost_z"] = frame.groupby("date")[leg_column].transform(
        lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) else 0.0
    ).fillna(0.0)
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
        selection.append({
            "weight": float(weight),
            "avg_sharpe": float(pd.to_numeric(metrics["sharpe"], errors="coerce").mean()),
            "worst_drawdown": float(pd.to_numeric(metrics["max_drawdown"], errors="coerce").min()),
            "details": metrics.to_dict("records"),
        })
    baseline_selection = next(row for row in selection if row["weight"] == 0.0)
    eligible = [
        row for row in selection
        if row["weight"] > 0
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
    stability = watchlist_grid.stability_decision(candidate_returns, baseline_returns)
    target = "target_ret_3d"
    daily_ic = []
    for _, group in holdout_frame[["date", leg_column, target]].dropna().groupby("date"):
        if len(group) >= 20:
            daily_ic.append(group[leg_column].corr(group[target], method="spearman"))
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
        "selection": selection,
        "selected_weight": selected_weight,
        "independent_signal": {
            "daily_rank_ic_mean": float(np.nanmean(daily_ic)),
            "positive_ic_day_rate": float(np.mean(np.asarray(daily_ic) > 0)),
            "days": len(daily_ic),
        },
        "rank_correlation": ranks.corr(method="spearman").to_dict(),
        "holdout_baseline": baseline_eval.to_dict("records"),
        "holdout_candidate": candidate_eval.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(selected_weight > 0 and promotion.get("promote") and stability.get("passed")),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
