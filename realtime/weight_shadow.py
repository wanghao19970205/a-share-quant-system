"""Realtime-only return-weight shadow evaluation.

Reads historical prediction rows with realized ``target_ret_1d`` values, performs
walk-forward non-negative stacking plus deterministic evolutionary refinement,
and writes versioned manifests under the realtime ledger. It never changes active
model artifacts, scheduler configuration, environment weights, or paper accounts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RealtimeConfig, load

MODEL_COLUMNS = ("ridge_pred", "elastic_pred", "extra_trees_pred")
BASELINE_WEIGHTS = np.array([0.30, 0.20, 0.50], dtype=float)
TARGET_COLUMN = "target_ret_1d"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _normalize(weights) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    return values / total if total > 0 else BASELINE_WEIGHTS.copy()


def _blend(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    available = np.isfinite(matrix)
    numerator = np.where(available, matrix, 0.0) @ weights
    denominator = available.astype(float) @ weights
    return np.divide(numerator, denominator, out=np.full(len(matrix), np.nan),
                     where=denominator > 0)


def _loss(matrix: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    prediction = _blend(matrix, weights)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return float("inf")
    errors = np.clip(prediction[valid] - target[valid], -0.20, 0.20)
    return float(np.mean(errors * errors))


def _projected_stacking(matrix: np.ndarray, target: np.ndarray,
                        initial: np.ndarray = BASELINE_WEIGHTS) -> np.ndarray:
    """Projected finite-difference descent on the non-negative unit simplex."""
    weights = _normalize(initial)
    step = 0.20
    epsilon = 1e-4
    best = _loss(matrix, target, weights)
    for _ in range(80):
        gradient = np.zeros(len(weights), dtype=float)
        for index in range(len(weights)):
            shifted = weights.copy()
            shifted[index] += epsilon
            gradient[index] = (_loss(matrix, target, _normalize(shifted)) - best) / epsilon
        candidate = _normalize(weights - step * gradient)
        candidate_loss = _loss(matrix, target, candidate)
        if candidate_loss + 1e-12 < best:
            weights, best = candidate, candidate_loss
            step = min(0.5, step * 1.05)
        else:
            step *= 0.5
        if step < 1e-5:
            break
    return weights


def _evolutionary_refine(matrix: np.ndarray, target: np.ndarray,
                         seed_weights: np.ndarray, seed: int = 20260805) -> np.ndarray:
    """Deterministic simplex evolution around the constrained stacking solution."""
    rng = np.random.default_rng(seed)
    population = [BASELINE_WEIGHTS.copy(), _normalize(seed_weights)]
    population.extend(rng.dirichlet(np.ones(3), size=22))
    population = np.asarray(population, dtype=float)
    for generation in range(24):
        losses = np.array([_loss(matrix, target, row) for row in population])
        elite = population[np.argsort(losses)[:6]]
        children = [row.copy() for row in elite]
        scale = max(0.01, 0.12 * (1.0 - generation / 24.0))
        while len(children) < 24:
            left, right = elite[rng.integers(0, len(elite), size=2)]
            child = _normalize((left + right) / 2.0 + rng.normal(0.0, scale, 3))
            children.append(child)
        population = np.asarray(children)
    losses = np.array([_loss(matrix, target, row) for row in population])
    return _normalize(population[int(np.argmin(losses))])


def optimize_weights(frame: pd.DataFrame, seed: int = 20260805) -> np.ndarray:
    matrix = frame.loc[:, MODEL_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(float)
    valid_target = np.isfinite(target) & np.isfinite(matrix).any(axis=1)
    matrix, target = matrix[valid_target], target[valid_target]
    if not len(target):
        return BASELINE_WEIGHTS.copy()
    # Bound runtime and prevent one very large cross-section from dominating evolution.
    if len(target) > 20000:
        indices = np.linspace(0, len(target) - 1, 20000, dtype=int)
        matrix, target = matrix[indices], target[indices]
    stacked = _projected_stacking(matrix, target)
    return _evolutionary_refine(matrix, target, stacked, seed=seed)


def _daily_returns(frame: pd.DataFrame, weights: np.ndarray, cost: float,
                   top_n: int = 10) -> pd.Series:
    scored = frame.copy()
    matrix = scored.loc[:, MODEL_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    scored["shadow_expected_return"] = _blend(matrix, weights)
    scored[TARGET_COLUMN] = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    scored = scored.dropna(subset=["date", "shadow_expected_return", TARGET_COLUMN])
    values = {}
    for date, group in scored.groupby("date", sort=True):
        selected = group.nlargest(min(top_n, len(group)), "shadow_expected_return")
        if not selected.empty:
            values[pd.Timestamp(date)] = float(selected[TARGET_COLUMN].mean()) - cost
    return pd.Series(values, dtype=float).sort_index()


def _metrics(returns: pd.Series) -> dict:
    if returns.empty:
        return {"days": 0, "mean_return": None, "hit_rate": None,
                "sharpe": None, "max_drawdown": None}
    curve = (1.0 + returns.clip(lower=-0.99)).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    return {
        "days": int(len(returns)),
        "mean_return": float(returns.mean()),
        "hit_rate": float((returns > 0).mean()),
        "sharpe": float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else None,
        "max_drawdown": float(drawdown.min()),
    }


def walk_forward(frame: pd.DataFrame, train_days: int = 60,
                 rebalance_days: int = 5, cost: float = 0.002) -> dict:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"]).sort_values(["date", "code"])
    dates = list(data["date"].drop_duplicates())
    candidate_parts = []
    baseline_parts = []
    weight_history = []
    current_weights: Optional[np.ndarray] = None
    for index in range(train_days, len(dates)):
        if current_weights is None or (index - train_days) % rebalance_days == 0:
            train_set = set(dates[max(0, index - train_days):index])
            current_weights = optimize_weights(
                data[data["date"].isin(train_set)], seed=20260805 + index)
            weight_history.append({
                "effective_date": str(pd.Timestamp(dates[index]).date()),
                "training_start": str(pd.Timestamp(dates[max(0, index - train_days)]).date()),
                "training_end": str(pd.Timestamp(dates[index - 1]).date()),
                "weights": dict(zip(MODEL_COLUMNS, map(float, current_weights))),
            })
        day = data[data["date"] == dates[index]]
        candidate_parts.append(_daily_returns(day, current_weights, cost))
        baseline_parts.append(_daily_returns(day, BASELINE_WEIGHTS, cost))
    candidate = pd.concat(candidate_parts).sort_index() if candidate_parts else pd.Series(dtype=float)
    baseline = pd.concat(baseline_parts).sort_index() if baseline_parts else pd.Series(dtype=float)
    final_weights = current_weights if current_weights is not None else BASELINE_WEIGHTS
    return {
        "weights": dict(zip(MODEL_COLUMNS, map(float, final_weights))),
        "weight_history": weight_history,
        "candidate": _metrics(candidate),
        "baseline": _metrics(baseline),
    }


def evaluate(predictions_file: Path, output_dir: Path, train_days: int = 60,
             min_oos_days: int = 20, rebalance_days: int = 5,
             cost: float = 0.002) -> dict:
    source = Path(predictions_file).resolve()
    output = Path(output_dir).resolve()
    frame = pd.read_parquet(source)
    required = {"code", "date", TARGET_COLUMN, *MODEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight shadow source missing columns: {missing}")
    result = walk_forward(frame, train_days=train_days,
                          rebalance_days=rebalance_days, cost=cost)
    candidate, baseline = result["candidate"], result["baseline"]
    days = int(candidate["days"] or 0)
    beats_baseline = bool(
        days >= min_oos_days and candidate["mean_return"] is not None and
        baseline["mean_return"] is not None and
        candidate["mean_return"] > baseline["mean_return"] and
        candidate["max_drawdown"] <= 0 and baseline["max_drawdown"] <= 0 and
        candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02)
    state = "shadow" if days < min_oos_days else ("eligible" if beats_baseline else "rejected")
    recipe = {
        "source": str(source), "source_mtime_ns": source.stat().st_mtime_ns,
        "train_days": int(train_days), "min_oos_days": int(min_oos_days),
        "rebalance_days": int(rebalance_days), "cost_roundtrip": float(cost),
        "models": list(MODEL_COLUMNS), "target": TARGET_COLUMN,
    }
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    version = f"rtw-{dt.datetime.now():%Y%m%dT%H%M%S}-{digest}"
    manifest = {
        "schema_version": 1, "version": version,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "realtime_paper_only", "state": state,
        "publishable": False, "auto_apply": False,
        "production_weights_unchanged": True,
        "recipe": recipe, "proposed_weights": result["weights"],
        "oos": {"candidate": candidate, "baseline": baseline,
                "minimum_shadow_days": int(min_oos_days)},
        "promotion": {
            "manual_review_required": True,
            "recommendation": ("review_for_manual_promotion" if state == "eligible"
                               else "continue_shadow" if state == "shadow"
                               else "do_not_promote"),
        },
        "weight_history": result["weight_history"],
    }
    _atomic_json(output / "versions" / f"{version}.json", manifest)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def evaluate_config(cfg: RealtimeConfig) -> dict:
    return evaluate(
        cfg.weight_shadow_predictions_file, cfg.weight_shadow_dir,
        train_days=cfg.weight_shadow_train_days,
        min_oos_days=cfg.weight_shadow_min_oos_days,
        rebalance_days=cfg.weight_shadow_rebalance_days,
        cost=cfg.paper_cost,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime-only return-weight shadow evaluation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load()
    if args.predictions is not None:
        cfg.weight_shadow_predictions_file = args.predictions
    if args.output_dir is not None:
        cfg.weight_shadow_dir = args.output_dir
    manifest = evaluate_config(cfg)
    print(json.dumps({
        "version": manifest["version"], "state": manifest["state"],
        "proposed_weights": manifest["proposed_weights"],
        "recommendation": manifest["promotion"]["recommendation"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
