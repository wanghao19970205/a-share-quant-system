"""Shadow-only helpers for neutral residual and market-regime model experiments.

The functions in this module never publish active artifacts. They provide point-in-time
label neutralization and deterministic regime construction for walk-forward research.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant import full_train_batched as batched
from quant import model


def neutralize_target_cross_section(
    frame: pd.DataFrame,
    target: str,
    industry: pd.Series | None = None,
    exposure_columns: tuple[str, ...] = ("log_mv_total", "volatility_20", "ret_20d"),
    min_rows: int = 30,
    industry_shrinkage: float = 0.0,
) -> pd.Series:
    """Return daily target residuals after industry and numeric exposure controls."""
    if float(industry_shrinkage) < 0:
        raise ValueError("industry_shrinkage must be non-negative")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    industry_values = industry.reindex(frame.index) if industry is not None else None
    for _, idx in frame.groupby("date", sort=False).groups.items():
        group = frame.loc[idx]
        y = pd.to_numeric(group[target], errors="coerce")
        parts: list[pd.DataFrame] = []
        numeric = [column for column in exposure_columns if column in group.columns]
        if numeric:
            parts.append(group[numeric].apply(pd.to_numeric, errors="coerce"))
        industry_columns: list[str] = []
        if industry_values is not None:
            categories = industry_values.loc[idx].astype("string").fillna("UNKNOWN")
            dummies = pd.get_dummies(
                categories, prefix="industry", drop_first=True, dtype=float
            )
            industry_columns = dummies.columns.tolist()
            parts.append(dummies)
        if not parts:
            centered = y - y.mean()
            result.loc[idx] = centered
            continue
        x = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
        valid_columns = x.notna().any(axis=0)
        x = x.loc[:, valid_columns].fillna(0.0)
        industry_columns = [column for column in industry_columns if column in x]
        x.insert(0, "const", 1.0)
        valid = y.notna()
        if int(valid.sum()) < max(int(min_rows), x.shape[1] + 3):
            result.loc[idx] = y - y.mean()
            continue
        matrix = x.to_numpy(dtype=float)
        y_values = y.loc[valid].to_numpy(dtype=float)
        train_matrix = matrix[valid]
        if float(industry_shrinkage) > 0 and industry_columns:
            penalty = np.zeros(x.shape[1], dtype=float)
            penalty[[x.columns.get_loc(column) for column in industry_columns]] = float(
                industry_shrinkage
            )
            beta = np.linalg.lstsq(
                train_matrix.T @ train_matrix + np.diag(penalty),
                train_matrix.T @ y_values,
                rcond=None,
            )[0]
        else:
            beta = np.linalg.lstsq(train_matrix, y_values, rcond=None)[0]
        result.loc[idx] = y.to_numpy(dtype=float) - matrix @ beta
    return result


def _load_industry_metadata(
    industry_meta: Path | None,
    industry_history: Path | None,
) -> tuple[str, bool, pd.Series | None, pd.DataFrame | None, str | None]:
    if industry_meta is not None and industry_history is not None:
        raise ValueError("use either industry_meta or industry_history, not both")
    if industry_history is not None:
        history = pd.read_parquet(industry_history)
        required = {"code", "industry", "valid_from", "available_from"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"industry history columns missing: {sorted(missing)}")
        history = history.copy()
        history["code"] = history["code"].astype(str).str.zfill(6)
        for column in ("valid_from", "valid_to", "available_from"):
            if column not in history:
                history[column] = pd.NaT
            history[column] = pd.to_datetime(history[column], errors="coerce").dt.normalize()
        history = history.dropna(subset=["code", "industry", "valid_from", "available_from"])
        if history.empty:
            raise ValueError("industry history is empty after validation")
        if (history["available_from"] < history["valid_from"]).any():
            raise ValueError("industry history available_from predates valid_from")
        if "source_updated_at" in history.columns:
            source_updated = pd.to_datetime(history["source_updated_at"], errors="coerce").dt.normalize()
            known_source_dates = source_updated.notna()
            if (source_updated[known_source_dates] > history.loc[known_source_dates, "available_from"]).any():
                raise ValueError("industry history was published after available_from")
        updated = history["available_from"].max()
        return "strict_pit_industry", True, None, history, updated.isoformat()
    if industry_meta is not None:
        metadata = pd.read_parquet(industry_meta)
        if not {"code", "a_industry"}.issubset(metadata.columns):
            raise ValueError("industry metadata requires code and a_industry")
        metadata = metadata.copy()
        metadata["code"] = metadata["code"].astype(str).str.zfill(6)
        mapping = metadata.drop_duplicates("code", keep="last").set_index("code")["a_industry"]
        updated_at = None
        if "meta_updated_at" in metadata.columns:
            updated = pd.to_datetime(metadata["meta_updated_at"], errors="coerce").max()
            updated_at = updated.isoformat() if pd.notna(updated) else None
        return "industry_research", False, mapping, None, updated_at
    return "strict_pit", True, None, None, None


def _pit_industry_for_frame(frame: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    left = frame[["code", "date"]].copy()
    left["_row_index"] = frame.index
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    candidates = left.merge(history, on="code", how="left")
    open_or_active = (
        candidates["valid_to"].isna()
        | (candidates["valid_to"] > candidates["date"])
    )
    visible = candidates[
        (candidates["valid_from"] <= candidates["date"])
        & open_or_active
        & (candidates["available_from"] <= candidates["date"])
    ].copy()
    visible = visible.sort_values(
        ["_row_index", "available_from", "valid_from"], ascending=[True, True, True]
    ).drop_duplicates("_row_index", keep="last")
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[visible["_row_index"].to_numpy()] = visible["industry"].astype("string").to_numpy()
    return result


def build_residual_shadow(
    source_predictions: Path,
    factor_audit: Path,
    prepared_dir: Path,
    output_dir: Path,
    threads: int = 8,
    n_estimators: int = 200,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 40,
    industry_meta: Path | None = None,
    industry_history: Path | None = None,
    industry_shrinkage: float = 0.0,
    rank_bins: int = 5,
    eval_at: tuple[int, ...] = (3,),
) -> dict:
    """Build a shadow residual leg with explicit PIT publication boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    window_dir = output_dir / "windows"
    window_dir.mkdir(exist_ok=True)
    exposure_columns = ("log_mv_total", "volatility_20", "ret_20d")
    mode, publishable, industry_by_code, industry_history_frame, industry_updated_at = (
        _load_industry_metadata(industry_meta, industry_history)
    )
    recipe = {
        "mode": mode,
        "source": str(source_predictions),
        "audit": str(factor_audit),
        "prepared_dir": str(prepared_dir),
        "model": "lightgbm_ranker",
        "target": "target_ret_3d",
        "exposure_columns": list(exposure_columns),
        "threads": int(threads),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "early_stopping_rounds": int(early_stopping_rounds),
        "industry_meta": str(industry_meta) if industry_meta is not None else None,
        "industry_history": str(industry_history) if industry_history is not None else None,
        "industry_shrinkage": float(industry_shrinkage),
        "rank_bins": int(rank_bins),
        "eval_at": [int(value) for value in eval_at],
        "publishable": publishable,
    }
    recipe_path = output_dir / "recipe.json"
    if recipe_path.exists():
        existing = json.loads(recipe_path.read_text(encoding="utf-8"))
        if existing != recipe:
            raise RuntimeError("residual shadow recipe changed; use a new output directory")
    else:
        recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")

    base = pd.read_parquet(source_predictions)
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base["code"] = base["code"].astype(str)
    audit = pd.read_parquet(factor_audit)
    for column in ("test_start", "train_start", "train_end"):
        audit[column] = pd.to_datetime(audit[column], errors="coerce").dt.normalize()
    starts = sorted(audit["test_start"].dropna().unique())
    parts: list[pd.DataFrame] = []
    windows: list[dict] = []
    target = "target_ret_3d"

    for index, raw_start in enumerate(starts):
        current = pd.Timestamp(raw_start)
        next_start = (
            pd.Timestamp(starts[index + 1])
            if index + 1 < len(starts)
            else base["date"].max() + pd.Timedelta(days=1)
        )
        window_audit = audit[audit["test_start"] == current]
        factors = window_audit.loc[
            window_audit["selected"].astype(bool), "factor"
        ].astype(str).tolist()
        train_start = pd.Timestamp(window_audit["train_start"].iloc[0])
        train_end = pd.Timestamp(window_audit["train_end"].iloc[0])
        valid_start = current - pd.DateOffset(months=1)
        cached_prediction = window_dir / f"{current:%Y-%m-%d}.parquet"
        cached_report = window_dir / f"{current:%Y-%m-%d}.json"
        if cached_prediction.exists() and cached_report.exists():
            prediction = pd.read_parquet(cached_prediction)
            parts.append(prediction)
            windows.append(json.loads(cached_report.read_text(encoding="utf-8")))
            print(
                f"[residual-shadow] window={index + 1}/{len(starts)} "
                f"test={current.date()} cache-hit rows={len(prediction)}",
                flush=True,
            )
            continue

        columns = list(dict.fromkeys(
            ["code", "date", target] + factors + list(exposure_columns)
        ))
        frame = batched._load_window(  # noqa: SLF001
            prepared_dir, train_start, next_start, columns=columns, cache=None
        )
        if frame.empty or "date" not in frame:
            raise ValueError(
                "prepared panel returned no rows for "
                f"{train_start.date()} to {next_start.date()}: {prepared_dir}"
            )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        valid_dates = frame.loc[
            (frame["date"] >= valid_start) & (frame["date"] < current), "date"
        ]
        valid_end = batched._purged_end(valid_dates, current, 3)  # noqa: SLF001
        if valid_end is None:
            raise RuntimeError(f"validation boundary is empty: {current.date()}")

        label_mask = frame["date"] < current
        label_frame = frame.loc[label_mask]
        if industry_history_frame is not None:
            industry = _pit_industry_for_frame(label_frame, industry_history_frame)
        elif industry_by_code is not None:
            industry = label_frame["code"].astype(str).map(industry_by_code)
        else:
            industry = None
        frame.loc[label_mask, target] = neutralize_target_cross_section(
            label_frame,
            target,
            industry=industry,
            exposure_columns=exposure_columns,
            industry_shrinkage=industry_shrinkage,
        )
        result = model.train_lightgbm_ranker(
            frame,
            factors,
            horizon=3,
            train_end=str(train_end.date()),
            valid_end=str(valid_end.date()),
            predict_start=str(current.date()),
            decay_half_life_days=60.0,
            min_weight=0.03,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=threads,
            rank_bins=rank_bins,
            eval_at=eval_at,
        )
        if not result.ok:
            raise RuntimeError(f"residual window {current.date()} failed: {result.message}")
        prediction = result.predictions[
            (result.predictions["date"] >= current)
            & (result.predictions["date"] < next_start)
        ][["code", "date", "pred"]].rename(columns={"pred": "residual_pred"})
        window = {
            "test_start": str(current.date()),
            "test_end": str((next_start - pd.Timedelta(days=1)).date()),
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "factors": len(factors),
            "rows": len(prediction),
            "metrics": result.metrics,
        }
        temporary_prediction = cached_prediction.with_suffix(".tmp.parquet")
        prediction.to_parquet(temporary_prediction, index=False)
        os.replace(temporary_prediction, cached_prediction)
        temporary_report = cached_report.with_suffix(".tmp.json")
        temporary_report.write_text(
            json.dumps(window, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_report, cached_report)
        parts.append(prediction)
        windows.append(window)
        print(
            f"[residual-shadow] window={index + 1}/{len(starts)} "
            f"test={current.date()} rows={len(prediction)}",
            flush=True,
        )

    leg = pd.concat(parts, ignore_index=True)
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce").dt.normalize()
    leg["code"] = leg["code"].astype(str)
    merged = base.merge(leg, on=["code", "date"], how="left", validate="one_to_one")
    temporary = output_dir / "predictions.tmp.parquet"
    merged.to_parquet(temporary, index=False)
    os.replace(temporary, output_dir / "predictions.parquet")
    report = {
        "mode": mode,
        "publishable": publishable,
        "industry_meta": str(industry_meta) if industry_meta is not None else None,
        "industry_history": str(industry_history) if industry_history is not None else None,
        "industry_updated_at": industry_updated_at,
        "source": source_predictions.name,
        "rows": len(merged),
        "date_min": str(merged["date"].min().date()),
        "date_max": str(merged["date"].max().date()),
        "coverage": float(merged["residual_pred"].notna().mean()),
        "windows": windows,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _champion_score(frame: pd.DataFrame, champion_params: dict) -> pd.DataFrame:
    """Rebuild the frozen production score without looking at realized returns."""
    from quant import watchlist_grid

    return watchlist_grid._score_pred(  # noqa: SLF001
        frame,
        float(watchlist_grid._empty_to_none(champion_params.get("ic_weight")) or 0.0),  # noqa: SLF001
        float(watchlist_grid._empty_to_none(champion_params.get("naive_weight")) or 0.0),  # noqa: SLF001
        lgbm_weight=watchlist_grid._empty_to_none(champion_params.get("lgbm_weight")),  # noqa: SLF001
        elastic_weight=float(
            watchlist_grid._empty_to_none(champion_params.get("elastic_weight")) or 0.0  # noqa: SLF001
        ),
        catboost_weight=float(
            watchlist_grid._empty_to_none(champion_params.get("catboost_weight")) or 0.0  # noqa: SLF001
        ),
        extra_trees_weight=float(
            watchlist_grid._empty_to_none(champion_params.get("extra_trees_weight")) or 0.0  # noqa: SLF001
        ),
    )


def confidence_from_model_agreement(
    frame: pd.DataFrame,
    champion_params: dict,
) -> pd.Series:
    """Measure same-day rank agreement between active model legs and the champion."""
    scored = _champion_score(frame, champion_params)
    champion_rank = scored.groupby("date")["pred"].rank(method="average", pct=True)
    lgbm_weight = champion_params.get("lgbm_weight")
    components: list[tuple[str, float]] = []
    if lgbm_weight is not None and {"ridge_pred", "lgbm_pred"}.issubset(scored.columns):
        weight = float(lgbm_weight)
        components.extend((("lgbm_pred", abs(weight)), ("ridge_pred", abs(1.0 - weight))))
    elif "base_pred" in scored:
        components.append(("base_pred", 1.0))
    for parameter, column in (
        ("elastic_weight", "elastic_pred"),
        ("catboost_weight", "catboost_pred"),
        ("extra_trees_weight", "extra_trees_pred"),
        ("ic_weight", "ic_pred"),
        ("naive_weight", "rule_score"),
    ):
        weight = abs(float(champion_params.get(parameter) or 0.0))
        if weight and column in scored:
            components.append((column, weight))
    if len(components) < 2:
        raise ValueError("confidence gate requires at least two active model legs")

    numerator = pd.Series(0.0, index=scored.index)
    denominator = pd.Series(0.0, index=scored.index)
    for column, weight in components:
        values = pd.to_numeric(scored[column], errors="coerce")
        rank = values.groupby(scored["date"]).rank(method="average", pct=True)
        valid = rank.notna() & champion_rank.notna()
        closeness = (1.0 - (rank - champion_rank).abs()).clip(lower=0.0, upper=1.0)
        numerator.loc[valid] += float(weight) * closeness.loc[valid]
        denominator.loc[valid] += float(weight)
    return (numerator / denominator.replace(0.0, np.nan)).astype(float)


def orthogonalize_increment_cross_section(
    frame: pd.DataFrame,
    candidate: str,
    controls: tuple[str, ...] = ("champion_score",),
    min_rows: int = 30,
) -> pd.Series:
    """Return daily rank residuals orthogonal to the supplied existing scores."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby("date", sort=False).groups.items():
        group = frame.loc[idx]
        y = pd.to_numeric(group[candidate], errors="coerce").rank(method="average", pct=True)
        x = pd.DataFrame({
            column: pd.to_numeric(group[column], errors="coerce").rank(method="average", pct=True)
            for column in controls
            if column in group
        }, index=group.index)
        valid = y.notna() & x.notna().all(axis=1)
        if int(valid.sum()) < max(int(min_rows), x.shape[1] + 3):
            continue
        y_centered = y.loc[valid] - y.loc[valid].mean()
        x_centered = x.loc[valid] - x.loc[valid].mean()
        matrix = x_centered.to_numpy(dtype=float)
        beta = np.linalg.lstsq(matrix, y_centered.to_numpy(dtype=float), rcond=None)[0]
        residual = y_centered.to_numpy(dtype=float) - matrix @ beta
        scale = float(np.std(residual, ddof=0))
        result.loc[y_centered.index] = residual / scale if scale > 0 else 0.0
    return result


def _portfolio_evaluation(
    frame: pd.DataFrame,
    champion_params: dict,
    horizons: list[int],
    confidence_quantile: float | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    from quant import backtest, watchlist_grid

    scored = _champion_score(frame, champion_params)
    if confidence_quantile is not None:
        threshold = scored.groupby("date")["confidence_score"].transform(
            lambda values: values.quantile(float(confidence_quantile))
        )
        scored.loc[scored["confidence_score"] < threshold, "pred"] = np.nan
    top_n = int(champion_params.get("top_n", 3))
    max_weight = watchlist_grid._empty_to_none(  # noqa: SLF001
        champion_params.get("slot_weight", champion_params.get("max_weight"))
    )
    if max_weight is None and champion_params.get("gross_exposure") is not None:
        max_weight = float(champion_params["gross_exposure"]) / max(top_n, 1)
    if max_weight is None:
        max_weight = 1.0 / max(top_n, 1)
    pred_quantile = watchlist_grid._empty_to_none(champion_params.get("pred_quantile"))  # noqa: SLF001
    ridge_quantile = watchlist_grid._empty_to_none(champion_params.get("ridge_quantile"))  # noqa: SLF001
    rows: list[dict] = []
    returns: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        period_returns, _ = backtest.portfolio_from_predictions(
            scored,
            horizon=int(horizon),
            top_n=top_n,
            max_weight=float(max_weight),
            positive_only=True,
            pred_quantile=float(pred_quantile) if pred_quantile is not None else None,
            ridge_quantile=float(ridge_quantile) if ridge_quantile is not None else None,
        )
        if period_returns.empty:
            continue
        returns[int(horizon)] = period_returns
        rows.append({
            "horizon": int(horizon),
            **watchlist_grid._evaluate_returns(period_returns["ret"], int(horizon)),  # noqa: SLF001
        })
    return pd.DataFrame(rows), returns


def evaluate_confidence_gate_shadow(
    predictions: Path,
    champion_params: dict,
    watchlist: set[str],
    output_dir: Path,
    gate_quantiles: tuple[float | None, ...] = (None, 0.50, 0.60, 0.70, 0.80),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Select a model-agreement gate, then evaluate it on an untouched holdout."""
    from quant import watchlist_grid

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    frame["confidence_score"] = confidence_from_model_agreement(frame, champion_params)
    horizon_values = [int(value) for value in horizons]
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    selection_frame = frame[(frame["date"] >= start) & (frame["date"] < holdout)].copy()
    holdout_frame = frame[(frame["date"] >= holdout) & (frame["date"] < end)].copy()

    selection: list[dict] = []
    for quantile in gate_quantiles:
        metrics, _ = _portfolio_evaluation(
            selection_frame, champion_params, horizon_values, quantile
        )
        if metrics.empty or "sharpe" not in metrics or "max_drawdown" not in metrics:
            selection.append({
                "gate_quantile": quantile,
                "avg_sharpe": None,
                "worst_drawdown": None,
                "details": [],
            })
            continue
        selection.append({
            "gate_quantile": quantile,
            "avg_sharpe": float(pd.to_numeric(metrics["sharpe"], errors="coerce").mean()),
            "worst_drawdown": float(
                pd.to_numeric(metrics["max_drawdown"], errors="coerce").min()
            ),
            "details": metrics.to_dict("records"),
        })
    baseline_selection = next(row for row in selection if row["gate_quantile"] is None)
    if baseline_selection["avg_sharpe"] is None:
        raise ValueError("confidence gate baseline has no evaluable returns")
    eligible = [
        row for row in selection
        if row["gate_quantile"] is not None
        and row["avg_sharpe"] is not None
        and row["worst_drawdown"] is not None
        and row["avg_sharpe"] > baseline_selection["avg_sharpe"]
        and row["worst_drawdown"] >= baseline_selection["worst_drawdown"] - 0.02
    ]
    winner = max(eligible, key=lambda row: row["avg_sharpe"]) if eligible else baseline_selection
    selected_quantile = winner["gate_quantile"]
    baseline_eval, baseline_returns = _portfolio_evaluation(
        holdout_frame, champion_params, horizon_values, None
    )
    candidate_eval, candidate_returns = _portfolio_evaluation(
        holdout_frame, champion_params, horizon_values, selected_quantile
    )
    promotion = watchlist_grid.promotion_decision(
        candidate_eval,
        baseline_eval,
        min_sharpe_gain=0.10,
        max_drawdown_worsening=0.02,
        min_improved_horizons=1,
    )
    stability = watchlist_grid.stability_decision(candidate_returns, baseline_returns)
    report = {
        "candidate": "confidence_gate",
        "publishable": False,
        "reason": "independent shadow candidate; production artifacts are never written",
        "source": str(predictions),
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "selection": selection,
        "selected_gate_quantile": selected_quantile,
        "holdout_baseline": baseline_eval.to_dict("records"),
        "holdout_candidate": candidate_eval.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(
            selected_quantile is not None
            and promotion.get("promote")
            and stability.get("passed")
        ),
    }
    temporary = output_dir / "confidence.tmp.parquet"
    frame[["code", "date", "confidence_score"]].to_parquet(temporary, index=False)
    os.replace(temporary, output_dir / "confidence.parquet")
    (output_dir / "shadow_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def build_orthogonal_increment_shadow(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    output_dir: Path,
    weights: tuple[float, ...] = (0.0, 0.03, 0.05, 0.10),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Orthogonalize an optional leg to the champion and evaluate it independently."""
    from quant import shadow_leg_evaluation

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"orthogonal source leg is empty: {leg_column}")
    frame["champion_score"] = _champion_score(frame, champion_params)["pred"]
    frame["orthogonal_increment_pred"] = orthogonalize_increment_cross_section(
        frame, leg_column, controls=("champion_score",)
    )
    if frame["orthogonal_increment_pred"].notna().sum() == 0:
        raise ValueError("orthogonal increment is empty after cross-sectional residualization")
    temporary = output_dir / "predictions.tmp.parquet"
    frame.to_parquet(temporary, index=False)
    output = output_dir / "predictions.parquet"
    os.replace(temporary, output)
    report = shadow_leg_evaluation.evaluate_optional_leg(
        output,
        "orthogonal_increment_pred",
        champion_params,
        watchlist,
        output_dir / "shadow_evaluation.json",
        weights=weights,
        holdout_months=holdout_months,
        horizons=horizons,
    )
    report["candidate"] = "orthogonal_increment"
    report["source_leg"] = leg_column
    report["orthogonal_controls"] = ["champion_score"]
    report["publishable"] = False
    report["reason"] = "independent shadow candidate; production artifacts are never written"
    (output_dir / "shadow_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recipe = {
        "source": str(predictions),
        "source_leg": leg_column,
        "orthogonal_controls": ["champion_score"],
        "weights": list(weights),
        "holdout_months": int(holdout_months),
        "horizons": list(horizons),
        "publishable": False,
    }
    (output_dir / "recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _topk_residual_rerank_scores(
    frame: pd.DataFrame,
    leg_column: str,
    champion_params: dict,
    pool_size: int,
    residual_weight: float | pd.Series,
) -> pd.DataFrame:
    """Screen with the frozen champion, then rerank only its daily TopK pool."""
    from quant import watchlist_grid

    if int(pool_size) < int(champion_params.get("top_n", 2)):
        raise ValueError("pool_size must be at least champion top_n")
    scored = _champion_score(frame, champion_params)
    scored["champion_score"] = pd.to_numeric(scored["pred"], errors="coerce")
    eligible = scored["champion_score"].notna()
    if bool(champion_params.get("positive_only", True)):
        eligible &= scored["champion_score"] > 0
    pred_quantile = watchlist_grid._empty_to_none(champion_params.get("pred_quantile"))  # noqa: SLF001
    if pred_quantile is not None:
        threshold = scored["champion_score"].where(eligible).groupby(scored["date"]).transform(
            lambda values: values.quantile(float(pred_quantile)) if values.notna().sum() >= 5 else -np.inf
        )
        eligible &= scored["champion_score"] >= threshold
    ridge_quantile = watchlist_grid._empty_to_none(champion_params.get("ridge_quantile"))  # noqa: SLF001
    if ridge_quantile is not None and "ridge_pred" in scored:
        ridge = pd.to_numeric(scored["ridge_pred"], errors="coerce")
        ridge_threshold = ridge.where(eligible).groupby(scored["date"]).transform(
            lambda values: values.quantile(float(ridge_quantile)) if values.notna().sum() >= 5 else -np.inf
        )
        eligible &= ridge >= ridge_threshold
    scored["champion_pool_rank"] = scored["champion_score"].where(eligible).groupby(
        scored["date"]
    ).rank(method="first", ascending=False)
    in_pool = eligible & scored["champion_pool_rank"].le(int(pool_size))
    residual = pd.to_numeric(scored[leg_column], errors="coerce").where(in_pool)
    residual_mean = residual.groupby(scored["date"]).transform("mean")
    residual_std = residual.groupby(scored["date"]).transform(lambda values: values.std(ddof=0))
    scored["residual_pool_z"] = ((residual - residual_mean) / residual_std.where(residual_std > 0)).fillna(0.0)
    if isinstance(residual_weight, pd.Series):
        if isinstance(residual_weight.index, pd.DatetimeIndex):
            daily_weight = scored["date"].map(residual_weight)
        else:
            daily_weight = residual_weight.reindex(scored.index)
        weight = pd.to_numeric(daily_weight, errors="coerce").fillna(0.0)
    else:
        weight = pd.Series(float(residual_weight), index=scored.index)
    scored["residual_rerank_weight"] = weight.astype(float)
    scored["pred"] = (
        scored["champion_score"]
        + scored["residual_rerank_weight"] * scored["residual_pool_z"]
    ).where(in_pool)
    scored["in_champion_pool"] = in_pool
    return scored


def _topk_rerank_evaluation(
    frame: pd.DataFrame,
    leg_column: str,
    champion_params: dict,
    pool_size: int,
    residual_weight: float | pd.Series,
    horizons: list[int],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    from quant import backtest, watchlist_grid

    candidate = _topk_residual_rerank_scores(
        frame, leg_column, champion_params, pool_size, residual_weight
    )
    top_n = int(champion_params.get("top_n", 2))
    max_weight = watchlist_grid._empty_to_none(  # noqa: SLF001
        champion_params.get("slot_weight", champion_params.get("max_weight"))
    )
    if max_weight is None and champion_params.get("gross_exposure") is not None:
        max_weight = float(champion_params["gross_exposure"]) / max(top_n, 1)
    if max_weight is None:
        max_weight = 1.0 / max(top_n, 1)
    rows: list[dict] = []
    returns: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        period_returns, holdings = backtest.portfolio_from_predictions(
            candidate,
            horizon=int(horizon),
            top_n=top_n,
            max_weight=float(max_weight),
            positive_only=True,
        )
        if period_returns.empty:
            continue
        metrics = watchlist_grid._evaluate_returns(period_returns["ret"], int(horizon))  # noqa: SLF001
        target = f"target_ret_{int(horizon)}d"
        metrics["direction_win_rate"] = (
            float((pd.to_numeric(holdings[target], errors="coerce") > 0).mean())
            if target in holdings else None
        )
        metrics["avg_turnover"] = float(period_returns["turnover"].mean())
        metrics["avg_holdings"] = float(period_returns["n_holdings"].mean())
        rows.append({"horizon": int(horizon), **metrics})
        period_returns = period_returns.copy()
        period_returns["date"] = pd.to_datetime(period_returns["date"], errors="coerce")
        returns[int(horizon)] = period_returns.dropna(subset=["date"])
    return pd.DataFrame(rows), returns


def evaluate_topk_residual_rerank_shadow(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    output_dir: Path,
    pool_size: int = 10,
    weights: tuple[float, ...] = (0.0, 0.03, 0.05, 0.10),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Freeze a TopK residual rerank recipe on selection data and test holdout."""
    from quant import watchlist_grid

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"TopK rerank source is empty: {leg_column}")
    horizon_values = [int(value) for value in horizons]
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    selection_frame = frame[(frame["date"] >= start) & (frame["date"] < holdout)].copy()
    holdout_frame = frame[(frame["date"] >= holdout) & (frame["date"] < end)].copy()

    selection: list[dict] = []
    for weight in weights:
        metrics, _ = _topk_rerank_evaluation(
            selection_frame,
            leg_column,
            champion_params,
            int(pool_size),
            float(weight),
            horizon_values,
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
    baseline_eval, baseline_returns = _topk_rerank_evaluation(
        holdout_frame, leg_column, champion_params, int(pool_size), 0.0, horizon_values
    )
    candidate_eval, candidate_returns = _topk_rerank_evaluation(
        holdout_frame,
        leg_column,
        champion_params,
        int(pool_size),
        selected_weight,
        horizon_values,
    )
    promotion = watchlist_grid.promotion_decision(
        candidate_eval,
        baseline_eval,
        min_sharpe_gain=0.10,
        max_drawdown_worsening=0.02,
        min_improved_horizons=1,
    )
    stability = watchlist_grid.stability_decision(candidate_returns, baseline_returns)
    holdout_scored = _topk_residual_rerank_scores(
        holdout_frame, leg_column, champion_params, int(pool_size), selected_weight
    )
    pool_ic = []
    target = "target_ret_3d"
    for _, group in holdout_scored.loc[
        holdout_scored["in_champion_pool"], ["date", leg_column, target]
    ].dropna().groupby("date"):
        if len(group) >= 5:
            pool_ic.append(group[leg_column].corr(group[target], method="spearman"))
    report = {
        "candidate": "strict_pit_residual_top10_rerank_top2",
        "publishable": False,
        "reason": "independent shadow candidate; production artifacts are never written",
        "source": str(predictions),
        "leg_column": leg_column,
        "pool_size": int(pool_size),
        "top_n": int(champion_params.get("top_n", 2)),
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "selection": selection,
        "selected_weight": selected_weight,
        "holdout_pool_rank_ic_mean": float(np.nanmean(pool_ic)) if pool_ic else None,
        "holdout_pool_positive_ic_day_rate": (
            float(np.mean(np.asarray(pool_ic) > 0)) if pool_ic else None
        ),
        "holdout_pool_ic_days": len(pool_ic),
        "holdout_baseline": baseline_eval.to_dict("records"),
        "holdout_candidate": candidate_eval.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(selected_weight > 0 and promotion.get("promote") and stability.get("passed")),
    }
    holdout_output = output_dir / "holdout_predictions.parquet"
    temporary = output_dir / "holdout_predictions.tmp.parquet"
    holdout_scored.to_parquet(temporary, index=False)
    os.replace(temporary, holdout_output)
    (output_dir / "recipe.json").write_text(
        json.dumps({
            "candidate": report["candidate"],
            "source": str(predictions),
            "leg_column": leg_column,
            "pool_size": int(pool_size),
            "weights": list(weights),
            "holdout_months": int(holdout_months),
            "horizons": horizon_values,
            "champion_params": champion_params,
            "publishable": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "shadow_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def topk_residual_state_features(
    frame: pd.DataFrame,
    leg_column: str,
    champion_params: dict,
    pool_size: int = 10,
) -> pd.DataFrame:
    """Build same-day TopK residual confidence features without realized returns."""
    scored = _topk_residual_rerank_scores(
        frame, leg_column, champion_params, int(pool_size), 0.0
    )
    pool = scored[scored["in_champion_pool"]].copy()
    pool[leg_column] = pd.to_numeric(pool[leg_column], errors="coerce")
    columns = [
        "date",
        "topk_residual_coverage",
        "topk_residual_dispersion",
        "topk_residual_top_gap_z",
    ]
    rows: list[dict] = []
    for date, group in pool.groupby("date", sort=True):
        values = group[leg_column].dropna()
        pool_count = int(len(group))
        count = int(len(values))
        if count == 0:
            rows.append({
                "date": date,
                "topk_residual_coverage": 0.0,
                "topk_residual_dispersion": np.nan,
                "topk_residual_top_gap_z": np.nan,
            })
            continue
        dispersion = float(values.std(ddof=0))
        top_gap = float(values.nlargest(min(2, count)).mean() - values.median())
        rows.append({
            "date": date,
            "topk_residual_coverage": float(count / max(pool_count, 1)),
            "topk_residual_dispersion": dispersion,
            "topk_residual_top_gap_z": top_gap / dispersion if dispersion > 0 else 0.0,
        })
    return pd.DataFrame(rows, columns=columns)


def evaluate_topk_residual_confidence_gate_shadow(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    output_dir: Path,
    pool_size: int = 10,
    rerank_weight: float = 0.05,
    state_quantiles: tuple[float, ...] = (0.60, 0.70, 0.80),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Select a TopK residual on/off gate, then evaluate an untouched holdout."""
    from quant import watchlist_grid

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if float(rerank_weight) <= 0:
        raise ValueError("rerank_weight must be positive")
    if not state_quantiles or any(not 0.0 < float(value) < 1.0 for value in state_quantiles):
        raise ValueError("state_quantiles must be strictly between zero and one")
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"TopK gate source is empty: {leg_column}")
    horizon_values = [int(value) for value in horizons]
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001
    states = topk_residual_state_features(
        frame, leg_column, champion_params, int(pool_size)
    )
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    selection_frame = frame[(frame["date"] >= start) & (frame["date"] < holdout)].copy()
    holdout_frame = frame[(frame["date"] >= holdout) & (frame["date"] < end)].copy()
    selection_states = states[states["date"] < holdout].copy()
    holdout_states = states[states["date"] >= holdout].copy()

    baseline_eval, baseline_returns = _topk_rerank_evaluation(
        selection_frame, leg_column, champion_params, int(pool_size), 0.0, horizon_values
    )
    if baseline_eval.empty:
        raise ValueError("TopK gate baseline has no evaluable returns")
    baseline_sharpe = float(pd.to_numeric(baseline_eval["sharpe"], errors="coerce").mean())
    baseline_drawdown = float(pd.to_numeric(baseline_eval["max_drawdown"], errors="coerce").min())
    recipes: list[dict] = []
    features = (
        "topk_residual_coverage",
        "topk_residual_dispersion",
        "topk_residual_top_gap_z",
    )
    for feature in features:
        valid = pd.to_numeric(selection_states[feature], errors="coerce").dropna()
        if valid.nunique() < 2:
            continue
        for quantile in state_quantiles:
            threshold = float(valid.quantile(float(quantile)))
            activation = selection_states.set_index("date")[feature].ge(threshold)
            daily_weight = activation.map({True: float(rerank_weight), False: 0.0})
            metrics, _ = _topk_rerank_evaluation(
                selection_frame,
                leg_column,
                champion_params,
                int(pool_size),
                daily_weight,
                horizon_values,
            )
            if metrics.empty:
                continue
            recipes.append({
                "feature": feature,
                "quantile": float(quantile),
                "threshold": threshold,
                "rerank_weight": float(rerank_weight),
                "selection_activation_rate": float(activation.mean()),
                "avg_sharpe": float(pd.to_numeric(metrics["sharpe"], errors="coerce").mean()),
                "worst_drawdown": float(
                    pd.to_numeric(metrics["max_drawdown"], errors="coerce").min()
                ),
                "details": metrics.to_dict("records"),
            })
    eligible = [
        row for row in recipes
        if row["avg_sharpe"] > baseline_sharpe
        and row["worst_drawdown"] >= baseline_drawdown - 0.02
    ]
    winner = max(eligible, key=lambda row: row["avg_sharpe"]) if eligible else None
    holdout_baseline, holdout_baseline_returns = _topk_rerank_evaluation(
        holdout_frame, leg_column, champion_params, int(pool_size), 0.0, horizon_values
    )
    if winner is None:
        holdout_activation = pd.Series(False, index=holdout_states["date"])
        holdout_activation.index = holdout_states["date"]
        holdout_candidate = holdout_baseline
        holdout_candidate_returns = holdout_baseline_returns
    else:
        holdout_activation = holdout_states.set_index("date")[winner["feature"]].ge(
            winner["threshold"]
        )
        holdout_daily_weight = holdout_activation.map(
            {True: float(rerank_weight), False: 0.0}
        )
        holdout_candidate, holdout_candidate_returns = _topk_rerank_evaluation(
            holdout_frame,
            leg_column,
            champion_params,
            int(pool_size),
            holdout_daily_weight,
            horizon_values,
        )
    promotion = watchlist_grid.promotion_decision(
        holdout_candidate,
        holdout_baseline,
        min_sharpe_gain=0.10,
        max_drawdown_worsening=0.02,
        min_improved_horizons=1,
    )
    stability = watchlist_grid.stability_decision(
        holdout_candidate_returns, holdout_baseline_returns
    )
    holdout_weights = holdout_activation.map({True: float(rerank_weight), False: 0.0})
    holdout_scored = _topk_residual_rerank_scores(
        holdout_frame,
        leg_column,
        champion_params,
        int(pool_size),
        holdout_weights,
    )
    report = {
        "candidate": "strict_pit_top10_residual_confidence_gate_top2",
        "publishable": False,
        "reason": "independent shadow candidate; production artifacts are never written",
        "source": str(predictions),
        "leg_column": leg_column,
        "pool_size": int(pool_size),
        "top_n": int(champion_params.get("top_n", 2)),
        "rerank_weight": float(rerank_weight),
        "requested_horizons": horizon_values,
        "evaluated_horizons": sorted(
            int(value) for value in holdout_baseline.get("horizon", pd.Series(dtype=int)).dropna().unique()
        ),
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "baseline_selection": baseline_eval.to_dict("records"),
        "recipes": recipes,
        "selected_recipe": winner,
        "holdout_activation_rate": float(holdout_activation.mean()),
        "holdout_baseline": holdout_baseline.to_dict("records"),
        "holdout_candidate": holdout_candidate.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(winner and promotion.get("promote") and stability.get("passed")),
    }
    states.to_parquet(output_dir / "state_features.parquet", index=False)
    temporary = output_dir / "holdout_predictions.tmp.parquet"
    holdout_scored.to_parquet(temporary, index=False)
    os.replace(temporary, output_dir / "holdout_predictions.parquet")
    (output_dir / "recipe.json").write_text(
        json.dumps({
            "candidate": report["candidate"],
            "source": str(predictions),
            "leg_column": leg_column,
            "pool_size": int(pool_size),
            "rerank_weight": float(rerank_weight),
            "state_quantiles": list(state_quantiles),
            "state_features": list(features),
            "holdout_months": int(holdout_months),
            "horizons": horizon_values,
            "champion_params": champion_params,
            "publishable": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "shadow_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def residual_state_features(frame: pd.DataFrame, leg_column: str) -> pd.DataFrame:
    """Build same-day residual state features without realized-return inputs."""
    if leg_column not in frame:
        raise ValueError(f"residual state source is missing: {leg_column}")
    values = pd.to_numeric(frame[leg_column], errors="coerce")
    work = pd.DataFrame({"date": frame["date"], "value": values})

    def summarize(values: pd.Series) -> pd.Series:
        valid = values.dropna()
        count = int(len(valid))
        if count == 0:
            return pd.Series({
                "residual_coverage": 0.0,
                "residual_dispersion": np.nan,
                "residual_top_gap_z": np.nan,
            })
        dispersion = float(valid.std(ddof=0))
        top_count = min(2, count)
        top_gap = float(valid.nlargest(top_count).mean() - valid.median())
        return pd.Series({
            "residual_coverage": float(count / max(len(values), 1)),
            "residual_dispersion": dispersion,
            "residual_top_gap_z": top_gap / dispersion if dispersion > 0 else 0.0,
        })

    result = work.groupby("date", sort=True)["value"].apply(summarize).unstack().reset_index()
    trailing_dispersion = result["residual_dispersion"].shift(1).rolling(
        60, min_periods=20
    ).median()
    result["residual_dispersion_ratio"] = (
        result["residual_dispersion"] / trailing_dispersion.replace(0.0, np.nan)
    )
    return result


def _evaluate_dynamic_residual(
    frame: pd.DataFrame,
    champion_params: dict,
    horizons: list[int],
    leg_column: str,
    daily_weight: pd.Series,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    from quant import watchlist_grid

    candidate = frame.copy()
    raw = pd.to_numeric(candidate[leg_column], errors="coerce")
    candidate["catboost_z"] = raw.groupby(candidate["date"]).transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0)
            if values.std(ddof=0)
            else 0.0
        )
    ).fillna(0.0)
    candidate["catboost_z"] *= candidate["date"].map(daily_weight).fillna(0.0)
    params = dict(champion_params)
    params["catboost_weight"] = 1.0
    prepared = watchlist_grid._prepare_fast_grid(candidate, horizons)  # noqa: SLF001
    metrics = watchlist_grid.evaluate_prepared_params(
        prepared, params, horizons, "short", True
    )
    returns = watchlist_grid.evaluate_prepared_returns(
        candidate, params, horizons, "short", True
    )
    return metrics, returns


def evaluate_residual_state_gate_shadow(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    output_dir: Path,
    high_weights: tuple[float, ...] = (0.05, 0.10),
    low_weights: tuple[float, ...] = (0.0, 0.03),
    state_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70),
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Freeze a residual-strength gate on selection data and test it on holdout."""
    from quant import watchlist_grid

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"residual state source is empty: {leg_column}")
    horizon_values = [int(value) for value in horizons]
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001
    states = residual_state_features(frame, leg_column)
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    selection_frame = frame[(frame["date"] >= start) & (frame["date"] < holdout)].copy()
    holdout_frame = frame[(frame["date"] >= holdout) & (frame["date"] < end)].copy()
    selection_states = states[states["date"] < holdout].copy()
    holdout_states = states[states["date"] >= holdout].copy()

    baseline_weights = pd.Series(0.0, index=states["date"])
    baseline_weights.index = states["date"]
    baseline_selection, _ = _evaluate_dynamic_residual(
        selection_frame, champion_params, horizon_values, leg_column, baseline_weights
    )
    if baseline_selection.empty:
        raise ValueError("residual state baseline has no evaluable returns")
    baseline_sharpe = float(pd.to_numeric(baseline_selection["sharpe"], errors="coerce").mean())
    baseline_drawdown = float(
        pd.to_numeric(baseline_selection["max_drawdown"], errors="coerce").min()
    )
    recipes: list[dict] = []
    for feature in ("residual_dispersion_ratio", "residual_top_gap_z"):
        for quantile in state_quantiles:
            threshold = float(selection_states[feature].quantile(float(quantile)))
            for high_weight in high_weights:
                for low_weight in low_weights:
                    selection_daily = selection_states.set_index("date")[feature].ge(threshold).map(
                        {True: float(high_weight), False: float(low_weight)}
                    )
                    metrics, _ = _evaluate_dynamic_residual(
                        selection_frame,
                        champion_params,
                        horizon_values,
                        leg_column,
                        selection_daily,
                    )
                    if metrics.empty:
                        continue
                    avg_sharpe = float(pd.to_numeric(metrics["sharpe"], errors="coerce").mean())
                    worst_drawdown = float(
                        pd.to_numeric(metrics["max_drawdown"], errors="coerce").min()
                    )
                    recipes.append({
                        "feature": feature,
                        "quantile": float(quantile),
                        "threshold": threshold,
                        "high_weight": float(high_weight),
                        "low_weight": float(low_weight),
                        "avg_sharpe": avg_sharpe,
                        "worst_drawdown": worst_drawdown,
                        "details": metrics.to_dict("records"),
                    })
    eligible = [
        row for row in recipes
        if row["avg_sharpe"] > baseline_sharpe
        and row["worst_drawdown"] >= baseline_drawdown - 0.02
    ]
    winner = max(eligible, key=lambda row: row["avg_sharpe"]) if eligible else None
    baseline_eval, baseline_returns = _evaluate_dynamic_residual(
        holdout_frame, champion_params, horizon_values, leg_column, baseline_weights
    )
    if winner is None:
        candidate_eval, candidate_returns = baseline_eval, baseline_returns
    else:
        holdout_daily = holdout_states.set_index("date")[winner["feature"]].ge(
            winner["threshold"]
        ).map({True: winner["high_weight"], False: winner["low_weight"]})
        candidate_eval, candidate_returns = _evaluate_dynamic_residual(
            holdout_frame,
            champion_params,
            horizon_values,
            leg_column,
            holdout_daily,
        )
    promotion = watchlist_grid.promotion_decision(
        candidate_eval,
        baseline_eval,
        min_sharpe_gain=0.10,
        max_drawdown_worsening=0.02,
        min_improved_horizons=1,
    )
    stability = watchlist_grid.stability_decision(candidate_returns, baseline_returns)
    report = {
        "candidate": "strict_pit_residual_state_gate",
        "publishable": False,
        "reason": "independent shadow candidate; production artifacts are never written",
        "source": str(predictions),
        "leg_column": leg_column,
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "baseline_selection": baseline_selection.to_dict("records"),
        "recipes": recipes,
        "selected_recipe": winner,
        "holdout_baseline": baseline_eval.to_dict("records"),
        "holdout_candidate": candidate_eval.to_dict("records"),
        "promotion_gate": promotion,
        "stability_gate": stability,
        "passed": bool(winner and promotion.get("promote") and stability.get("passed")),
    }
    states.to_parquet(output_dir / "state_features.parquet", index=False)
    (output_dir / "shadow_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def attribute_strict_pit_residual(
    predictions: Path,
    leg_column: str,
    champion_params: dict,
    watchlist: set[str],
    industry_history: Path,
    output_dir: Path,
    selected_weight: float = 0.10,
    holdout_months: int = 6,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Diagnose holdout residual behavior using only strict PIT industry labels."""
    from quant import backtest, watchlist_grid

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].isin(watchlist)].copy()
    if leg_column not in frame or frame[leg_column].notna().sum() == 0:
        raise ValueError(f"residual attribution source is empty: {leg_column}")
    _, publishable, _, history, updated = _load_industry_metadata(None, industry_history)
    if not publishable or history is None:
        raise ValueError("strict PIT industry history is required")
    end = frame["date"].max() + pd.Timedelta(days=1)
    holdout = end - pd.DateOffset(months=holdout_months)
    frame = frame[frame["date"] >= holdout].copy()
    horizon_values = [int(value) for value in horizons]
    frame = watchlist_grid._ensure_targets(frame, horizon_values)  # noqa: SLF001
    frame["pit_industry"] = _pit_industry_for_frame(frame, history).fillna("UNKNOWN")
    raw = pd.to_numeric(frame[leg_column], errors="coerce")
    frame["residual_z"] = raw.groupby(frame["date"]).transform(
        lambda values: (
            (values - values.mean()) / values.std(ddof=0)
            if values.std(ddof=0)
            else 0.0
        )
    ).fillna(0.0)
    frame["residual_decile"] = frame.groupby("date")[leg_column].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    scored = _champion_score(frame, champion_params)
    baseline = scored.copy()
    candidate = scored.copy()
    candidate["pred"] = candidate["pred"] + float(selected_weight) * frame["residual_z"]
    topn_reports: list[dict] = []
    monthly_parts: list[pd.DataFrame] = []
    industry_parts: list[pd.DataFrame] = []
    for top_n in (2, 5, 10):
        max_weight = float(champion_params.get("gross_exposure", 0.3)) / max(top_n, 1)
        for horizon in horizon_values:
            base_returns, base_holdings = backtest.portfolio_from_predictions(
                baseline, horizon, top_n, max_weight, positive_only=True
            )
            candidate_returns, candidate_holdings = backtest.portfolio_from_predictions(
                candidate, horizon, top_n, max_weight, positive_only=True
            )
            if base_returns.empty or candidate_returns.empty:
                continue
            merged = base_returns[["date", "ret"]].merge(
                candidate_returns[["date", "ret"]],
                on="date",
                suffixes=("_baseline", "_candidate"),
            )
            merged["gain"] = merged["ret_candidate"] - merged["ret_baseline"]
            merged["month"] = merged["date"].dt.to_period("M").astype(str)
            monthly = merged.groupby("month", as_index=False)["gain"].sum()
            monthly["top_n"] = top_n
            monthly["horizon"] = horizon
            monthly_parts.append(monthly)
            topn_reports.append({
                "top_n": top_n,
                "horizon": horizon,
                "baseline": watchlist_grid._evaluate_returns(  # noqa: SLF001
                    base_returns["ret"], horizon
                ),
                "candidate": watchlist_grid._evaluate_returns(  # noqa: SLF001
                    candidate_returns["ret"], horizon
                ),
                "positive_gain_day_rate": float((merged["gain"] > 0).mean()),
            })
            target = f"target_ret_{horizon}d"
            if target in candidate_holdings:
                contribution = candidate_holdings.copy()
                lookup = frame.set_index(["date", "code"])["pit_industry"]
                contribution["pit_industry"] = [
                    lookup.get((date, str(code).zfill(6)), "UNKNOWN")
                    for date, code in zip(contribution["date"], contribution["code"])
                ]
                contribution["contribution"] = (
                    pd.to_numeric(contribution[target], errors="coerce")
                    * pd.to_numeric(contribution["weight"], errors="coerce")
                )
                grouped = contribution.groupby("pit_industry", as_index=False)[
                    "contribution"
                ].sum()
                grouped["top_n"] = top_n
                grouped["horizon"] = horizon
                industry_parts.append(grouped)
    deciles = []
    for horizon in horizon_values:
        target = f"target_ret_{horizon}d"
        if target not in frame:
            continue
        grouped = frame.groupby("residual_decile")[target].agg(["mean", "count"]).reset_index()
        grouped["horizon"] = horizon
        deciles.extend(grouped.to_dict("records"))
    monthly_frame = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    industry_frame = pd.concat(industry_parts, ignore_index=True) if industry_parts else pd.DataFrame()
    report = {
        "candidate": "strict_pit_residual_attribution",
        "diagnostic_only": True,
        "publishable": False,
        "reason": "holdout attribution is diagnostic and cannot select or publish a model",
        "industry_history": str(industry_history),
        "industry_updated_at": updated,
        "selected_weight": float(selected_weight),
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "decile_returns": deciles,
        "topn_comparison": topn_reports,
        "monthly_gains": monthly_frame.to_dict("records"),
        "industry_contribution": (
            industry_frame.sort_values("contribution", ascending=False).to_dict("records")
            if not industry_frame.empty else []
        ),
    }
    (output_dir / "attribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def build_market_regimes(
    daily_market: pd.DataFrame,
    trend_window: int = 20,
    volatility_window: int = 20,
    history_window: int = 252,
) -> pd.DataFrame:
    """Build deterministic regimes from same-day and trailing market breadth data."""
    required = {"date", "median_return", "breadth"}
    missing = required - set(daily_market.columns)
    if missing:
        raise ValueError(f"market state columns missing: {sorted(missing)}")
    out = daily_market.copy().sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["median_return"] = pd.to_numeric(out["median_return"], errors="coerce")
    out["breadth"] = pd.to_numeric(out["breadth"], errors="coerce")
    out["market_trend"] = out["median_return"].rolling(trend_window, min_periods=trend_window).sum()
    out["market_volatility"] = out["median_return"].rolling(
        volatility_window, min_periods=volatility_window
    ).std()
    threshold = out["market_volatility"].rolling(
        history_window, min_periods=max(volatility_window, history_window // 4)
    ).median()
    trend = np.where(
        (out["market_trend"] > 0) & (out["breadth"] >= 0.5), "up", "down_or_weak"
    )
    volatility = np.where(out["market_volatility"] > threshold, "high_vol", "normal_vol")
    out["regime"] = pd.Series(trend, index=out.index) + "__" + pd.Series(volatility, index=out.index)
    unavailable = out[["market_trend", "market_volatility"]].isna().any(axis=1) | threshold.isna()
    out.loc[unavailable, "regime"] = "insufficient_history"
    return out


def _daily_market_from_prices(
    price_dir: Path,
    codes: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for code in sorted(codes):
        path = price_dir / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            price = pd.read_parquet(path, columns=["date", "close"])
        except Exception:  # noqa: BLE001 research input may contain an incomplete file
            continue
        price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
        price["close"] = pd.to_numeric(price["close"], errors="coerce")
        price = price.dropna().sort_values("date").drop_duplicates("date", keep="last")
        price["pct_change"] = price["close"].pct_change(fill_method=None)
        rows.append(price[(price["date"] >= start) & (price["date"] < end)][["date", "pct_change"]])
    if not rows:
        raise ValueError("no price rows available for market regimes")
    market_rows = pd.concat(rows, ignore_index=True).dropna(subset=["pct_change"])
    return market_rows.groupby("date")["pct_change"].agg(
        median_return="median",
        breadth=lambda values: float((values > 0).mean()),
        n_stocks="count",
    ).reset_index()


def evaluate_market_regime_weights(
    predictions: Path,
    champion_params: dict,
    watchlist: set[str],
    output: Path,
    price_dir: Path | None = None,
    candidate_weights: tuple[float, ...] = (0.70, 0.85, 1.00),
    holdout_months: int = 6,
) -> dict:
    """Select LightGBM weight per trailing market regime, then run frozen holdout."""
    from quant import watchlist_grid

    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str)
    all_codes = set(frame["code"])
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=1)
    price_dir = price_dir or predictions.parent / "price"
    daily_market = _daily_market_from_prices(price_dir, all_codes, start, end)
    regimes = build_market_regimes(daily_market)
    frame = frame[frame["code"].isin(watchlist)].copy()
    frame = frame.merge(regimes[["date", "regime"]], on="date", how="left", validate="many_to_one")
    holdout = end - pd.DateOffset(months=holdout_months)
    horizons = [1, 2, 3]

    def returns_for(subset: pd.DataFrame, weight: float) -> pd.DataFrame:
        params = dict(champion_params)
        params["lgbm_weight"] = float(weight)
        by_horizon = watchlist_grid.evaluate_prepared_returns(
            subset, params, horizons, "short", True
        )
        parts = []
        for horizon, values in by_horizon.items():
            part = values.copy()
            part["horizon"] = int(horizon)
            part["weight"] = float(weight)
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def prepare(period_start: pd.Timestamp, period_end: pd.Timestamp) -> pd.DataFrame:
        subset = frame[(frame["date"] >= period_start) & (frame["date"] < period_end)].copy()
        return watchlist_grid._ensure_targets(subset, horizons)  # noqa: SLF001

    selection_frame = prepare(start, holdout)
    selection_returns = []
    for weight in candidate_weights:
        values = returns_for(selection_frame, weight)
        values = values.merge(regimes[["date", "regime"]], on="date", how="left")
        selection_returns.append(values)
    selection_returns_frame = pd.concat(selection_returns, ignore_index=True)
    selected = select_regime_weights(
        selection_returns_frame, list(candidate_weights), min_observations=20
    )

    holdout_frame = prepare(holdout, end)
    incumbent_weight = float(champion_params.get("lgbm_weight", 0.85))
    baseline_returns = returns_for(holdout_frame, incumbent_weight)
    candidates = {weight: returns_for(holdout_frame, weight) for weight in candidate_weights}
    date_regime = regimes.set_index("date")["regime"]
    adaptive_parts = []
    for weight, values in candidates.items():
        regime = values["date"].map(date_regime)
        chosen = regime.map(selected).fillna(incumbent_weight)
        adaptive_parts.append(values[chosen == weight])
    adaptive_returns = pd.concat(adaptive_parts, ignore_index=True).sort_values(["horizon", "date"])
    baseline_dict = {
        int(horizon): values.drop(columns=["horizon", "weight"], errors="ignore")
        for horizon, values in baseline_returns.groupby("horizon")
    }
    adaptive_dict = {
        int(horizon): values.drop(columns=["horizon", "weight"], errors="ignore")
        for horizon, values in adaptive_returns.groupby("horizon")
    }
    stability = watchlist_grid.stability_decision(adaptive_dict, baseline_dict)

    def summarize(values: pd.DataFrame) -> list[dict]:
        rows = []
        for horizon, group in values.groupby("horizon"):
            metrics = watchlist_grid._evaluate_returns(group["ret"], int(horizon))  # noqa: SLF001
            rows.append({"horizon": int(horizon), **metrics})
        return rows

    baseline_summary = summarize(baseline_returns)
    adaptive_summary = summarize(adaptive_returns)
    baseline_sharpe = {row["horizon"]: row["sharpe"] for row in baseline_summary}
    adaptive_sharpe = {row["horizon"]: row["sharpe"] for row in adaptive_summary}
    gains = [
        adaptive_sharpe[horizon] - baseline_sharpe[horizon]
        for horizon in sorted(set(baseline_sharpe) & set(adaptive_sharpe))
        if baseline_sharpe[horizon] is not None and adaptive_sharpe[horizon] is not None
    ]
    report = {
        "selection_range": [str(start.date()), str((holdout - pd.Timedelta(days=1)).date())],
        "holdout_range": [str(holdout.date()), str((end - pd.Timedelta(days=1)).date())],
        "candidate_weights": list(candidate_weights),
        "selected_weights": selected,
        "regime_counts": frame.groupby("regime")["date"].nunique().to_dict(),
        "holdout_baseline": baseline_summary,
        "holdout_adaptive": adaptive_summary,
        "avg_sharpe_gain": float(np.mean(gains)) if gains else None,
        "improved_horizons": int(sum(value > 0 for value in gains)),
        "stability_gate": stability,
        "passed": bool(gains and np.mean(gains) >= 0.10 and any(value > 0 for value in gains) and stability.get("passed")),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def select_regime_weights(
    returns: pd.DataFrame,
    candidate_weights: list[float],
    min_observations: int = 20,
) -> dict[str, float]:
    """Select each regime's weight using only supplied selection-period returns."""
    required = {"date", "regime", "weight", "ret"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"regime return columns missing: {sorted(missing)}")
    selected: dict[str, float] = {}
    for regime, group in returns.groupby("regime", sort=True):
        rows = []
        for weight in candidate_weights:
            weighted = group[group["weight"] == weight]
            scores = []
            horizon_groups = (
                weighted.groupby("horizon", sort=True)
                if "horizon" in weighted.columns
                else [(1, weighted)]
            )
            for horizon, horizon_group in horizon_groups:
                values = pd.to_numeric(horizon_group["ret"], errors="coerce").dropna()
                if len(values) < min_observations:
                    continue
                values = values.clip(lower=-0.99)
                nav = (1.0 + values).cumprod()
                periods_per_year = 252 / max(int(horizon), 1)
                annual = nav.iloc[-1] ** (periods_per_year / len(values)) - 1 if nav.iloc[-1] > 0 else np.nan
                volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
                if volatility > 0 and np.isfinite(annual):
                    scores.append(float(annual / volatility))
            if scores:
                rows.append((float(np.mean(scores)), -abs(float(weight)), float(weight)))
        if rows:
            selected[str(regime)] = max(rows)[2]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict PIT residual shadow predictions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--industry-meta", default="")
    parser.add_argument("--industry-history", default="")
    parser.add_argument("--industry-shrinkage", type=float, default=0.0)
    parser.add_argument("--rank-bins", type=int, default=5)
    parser.add_argument("--eval-at", type=int, nargs="+", default=[3])
    args = parser.parse_args()
    report = build_residual_shadow(
        Path(args.source),
        Path(args.audit),
        Path(args.prepared_dir),
        Path(args.output_dir),
        threads=args.threads,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        industry_meta=Path(args.industry_meta) if args.industry_meta else None,
        industry_history=Path(args.industry_history) if args.industry_history else None,
        industry_shrinkage=args.industry_shrinkage,
        rank_bins=args.rank_bins,
        eval_at=tuple(args.eval_at),
    )
    print(
        json.dumps(
            {key: report[key] for key in ("rows", "date_min", "date_max", "coverage")},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
