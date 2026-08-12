"""Realtime-only return-weight shadow evaluation.

Reads historical prediction rows with realized ``target_ret_1d`` values, performs
walk-forward non-negative stacking plus deterministic evolutionary refinement,
and writes versioned manifests under the realtime ledger. Eligible candidates can
atomically update a realtime-only active manifest; model artifacts, environment
weights, scheduled training, and paper accounts remain untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RealtimeConfig, load
from .weight_manifest import (append_history, atomic_write_json,
                              load_active_manifest, normalize_weights)

MODEL_COLUMNS = ("ridge_pred", "elastic_pred", "extra_trees_pred")
BASELINE_WEIGHTS = np.array([0.30, 0.20, 0.50], dtype=float)
TARGET_COLUMN = "target_ret_1d"
TARGET_CANDIDATES = (TARGET_COLUMN, "tradable_ret_1d", "open_ret_1d")


def _normalize_target_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for column in TARGET_CANDIDATES:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        normalized = frame.copy()
        normalized[TARGET_COLUMN] = values
        return normalized, column
    raise ValueError(
        "weight shadow source has no realized target values in "
        f"{list(TARGET_CANDIDATES)}"
    )


def _atomic_json(path: Path, value: dict) -> None:
    atomic_write_json(path, value)


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


def _daily_return_value(frame: pd.DataFrame, weights: np.ndarray, cost: float,
                        top_n: int = 10) -> Optional[float]:
    """计算单日 Top-N 收益；输入列已数值化时不再复制 DataFrame。"""
    if frame.empty:
        return None
    matrix = frame.loc[:, MODEL_COLUMNS].to_numpy(dtype=float, copy=False)
    target = frame[TARGET_COLUMN].to_numpy(dtype=float, copy=False)
    prediction = _blend(matrix, weights)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return None
    valid_prediction = prediction[valid]
    valid_target = target[valid]
    count = min(max(1, int(top_n)), len(valid_prediction))
    selected = np.argsort(-valid_prediction, kind="stable")[:count]
    return float(valid_target[selected].mean()) - cost


def _daily_returns(frame: pd.DataFrame, weights: np.ndarray, cost: float,
                   top_n: int = 10) -> pd.Series:
    scored = frame.copy()
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce").dt.normalize()
    for column in (*MODEL_COLUMNS, TARGET_COLUMN):
        scored[column] = pd.to_numeric(scored[column], errors="coerce")
    values = {}
    for date, group in scored.dropna(subset=["date"]).groupby("date", sort=True):
        value = _daily_return_value(group, weights, cost, top_n=top_n)
        if value is not None:
            values[pd.Timestamp(date)] = value
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
                 rebalance_days: int = 5, cost: float = 0.002,
                 baseline_weights: np.ndarray = BASELINE_WEIGHTS,
                 workers: int = 1) -> dict:
    data = frame.copy()
    baseline_weights = _normalize(baseline_weights)
    workers = max(1, min(4, int(workers)))
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in (*MODEL_COLUMNS, TARGET_COLUMN):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    model_matrix = data.loc[:, MODEL_COLUMNS].to_numpy(dtype=float, copy=False)
    realized = np.isfinite(data[TARGET_COLUMN].to_numpy(dtype=float, copy=False))
    usable = realized & np.isfinite(model_matrix).any(axis=1) & data["date"].notna().to_numpy()
    data = data.loc[usable].sort_values(["date", "code"])
    dates = list(data["date"].drop_duplicates())
    day_frames = {date: day for date, day in data.groupby("date", sort=True)}
    rebalance_indices = list(range(train_days, len(dates), rebalance_days))

    def _fit(index: int) -> tuple[int, np.ndarray]:
        train_set = set(dates[max(0, index - train_days):index])
        weights = optimize_weights(
            data[data["date"].isin(train_set)], seed=20260805 + index)
        return index, weights

    if workers > 1 and len(rebalance_indices) > 1:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="rt-weight") as executor:
            fitted = dict(executor.map(_fit, rebalance_indices))
    else:
        fitted = dict(_fit(index) for index in rebalance_indices)

    candidate_parts = []
    baseline_parts = []
    weight_history = []
    current_weights: Optional[np.ndarray] = None
    for index in range(train_days, len(dates)):
        if index in fitted:
            current_weights = fitted[index]
            weight_history.append({
                "effective_date": str(pd.Timestamp(dates[index]).date()),
                "training_start": str(pd.Timestamp(dates[max(0, index - train_days)]).date()),
                "training_end": str(pd.Timestamp(dates[index - 1]).date()),
                "weights": dict(zip(MODEL_COLUMNS, map(float, current_weights))),
            })
        day = day_frames[dates[index]]
        candidate_value = _daily_return_value(day, current_weights, cost)
        baseline_value = _daily_return_value(day, baseline_weights, cost)
        if candidate_value is not None:
            candidate_parts.append((pd.Timestamp(dates[index]), candidate_value))
        if baseline_value is not None:
            baseline_parts.append((pd.Timestamp(dates[index]), baseline_value))
    candidate = pd.Series(dict(candidate_parts), dtype=float).sort_index()
    baseline = pd.Series(dict(baseline_parts), dtype=float).sort_index()
    final_weights = current_weights if current_weights is not None else BASELINE_WEIGHTS
    return {
        "weights": dict(zip(MODEL_COLUMNS, map(float, final_weights))),
        "weight_history": weight_history,
        "candidate": _metrics(candidate),
        "baseline": _metrics(baseline),
        "evaluation_start": str(candidate.index[0].date()) if not candidate.empty else None,
        "evaluation_end": str(candidate.index[-1].date()) if not candidate.empty else None,
        "realized_dates": len(dates),
        "valid_rows": int(len(data)),
    }


def evaluate(predictions_file: Path, output_dir: Path, train_days: int = 60,
             min_oos_days: int = 40, rebalance_days: int = 5,
             cost: float = 0.002,
             baseline_weights: np.ndarray = BASELINE_WEIGHTS,
             workers: int = 1) -> dict:
    source = Path(predictions_file).resolve()
    output = Path(output_dir).resolve()
    frame = pd.read_parquet(source)
    required = {"code", "date", *MODEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight shadow source missing columns: {missing}")
    frame, target_column = _normalize_target_column(frame)
    baseline_weights = _normalize(baseline_weights)
    result = walk_forward(frame, train_days=train_days,
                          rebalance_days=rebalance_days, cost=cost,
                          baseline_weights=baseline_weights, workers=workers)
    candidate, baseline = result["candidate"], result["baseline"]
    days = int(candidate["days"] or 0)
    beats_baseline = bool(
        days >= min_oos_days and candidate["mean_return"] is not None and
        baseline["mean_return"] is not None and
        candidate["mean_return"] > baseline["mean_return"] and
        candidate["max_drawdown"] <= 0 and baseline["max_drawdown"] <= 0 and
        candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.02)
    state = "shadow" if days < min_oos_days else ("eligible" if beats_baseline else "rejected")
    evaluation_start = result["evaluation_start"]
    evaluation_end = result["evaluation_end"]
    proposed_vector = np.array([result["weights"][column] for column in MODEL_COLUMNS])
    max_component_step = float(np.max(np.abs(proposed_vector - baseline_weights)))
    recipe = {
        "source": str(source), "source_mtime_ns": source.stat().st_mtime_ns,
        "train_days": int(train_days), "min_oos_days": int(min_oos_days),
        "rebalance_days": int(rebalance_days), "workers": int(workers),
        "cost_roundtrip": float(cost),
        "models": list(MODEL_COLUMNS), "target": target_column,
        "baseline_weights": dict(zip(MODEL_COLUMNS, map(float, baseline_weights))),
        "optimizer_objective": "clipped_cross_section_mse",
        "evaluation_strategy": "top10_equal_weight_daily",
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
        "evaluation_period": {"start": evaluation_start, "end": evaluation_end},
        "data_quality": {
            "realized_dates": result["realized_dates"],
            "valid_rows": result["valid_rows"],
            "unrealized_dates_excluded": True,
        },
        "weight_diagnostics": {
            "max_component_step_from_baseline": max_component_step,
            "l1_distance_from_baseline": float(np.abs(proposed_vector - baseline_weights).sum()),
            "effective_model_count": float(1.0 / np.square(proposed_vector).sum()),
            "objective_alignment": "optimizer_mse_vs_promotion_top10_return",
        },
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


def _configured_weights(cfg: RealtimeConfig) -> dict[str, float]:
    if not getattr(cfg, "ensemble_return_enabled", True):
        return {"ridge_pred": 1.0, "elastic_pred": 0.0, "extra_trees_pred": 0.0}
    values = normalize_weights({
        "ridge_pred": getattr(cfg, "ensemble_ridge_weight", 0.30),
        "elastic_pred": getattr(cfg, "ensemble_elastic_weight", 0.20),
        "extra_trees_pred": getattr(cfg, "ensemble_extra_trees_weight", 0.50),
    })
    return values or dict(zip(MODEL_COLUMNS, map(float, BASELINE_WEIGHTS)))


def _current_weights(cfg: RealtimeConfig) -> tuple[dict[str, float], Optional[dict]]:
    active = load_active_manifest(cfg.weight_active_manifest_file)
    if active is not None and getattr(cfg, "weight_auto_promote_enabled", True):
        return dict(active["weights"]), active
    return _configured_weights(cfg), None


def evaluate_config(cfg: RealtimeConfig) -> dict:
    current, _ = _current_weights(cfg)
    return evaluate(
        cfg.weight_shadow_predictions_file, cfg.weight_shadow_dir,
        train_days=cfg.weight_shadow_train_days,
        min_oos_days=cfg.weight_shadow_min_oos_days,
        rebalance_days=cfg.weight_shadow_rebalance_days,
        cost=cfg.paper_cost,
        baseline_weights=np.array([current[column] for column in MODEL_COLUMNS]),
        workers=getattr(cfg, "weight_shadow_workers", 4),
    )


def _promotion_reasons(manifest: dict, cfg: RealtimeConfig,
                       current_weights: dict[str, float], active: Optional[dict],
                       observed_dates: list[pd.Timestamp]) -> list[str]:
    reasons: list[str] = []
    if manifest.get("state") != "eligible":
        reasons.append(f"state={manifest.get('state')}")
    candidate = manifest.get("oos", {}).get("candidate", {})
    baseline = manifest.get("oos", {}).get("baseline", {})
    if int(candidate.get("days") or 0) < cfg.weight_shadow_min_oos_days:
        reasons.append("insufficient_oos_days")
    if candidate.get("mean_return") is None or baseline.get("mean_return") is None or \
            candidate.get("mean_return", 0.0) <= baseline.get("mean_return", 0.0):
        reasons.append("mean_return_not_improved")
    candidate_sharpe = candidate.get("sharpe")
    baseline_sharpe = baseline.get("sharpe")
    if (baseline_sharpe is not None and
            (candidate_sharpe is None or candidate_sharpe <
             baseline_sharpe + cfg.weight_min_sharpe_improvement)):
        reasons.append("sharpe_not_improved")
    if (candidate.get("hit_rate") is None or baseline.get("hit_rate") is None or
            candidate["hit_rate"] < baseline["hit_rate"] + cfg.weight_min_hit_rate_delta):
        reasons.append("hit_rate_gate")
    if (candidate.get("max_drawdown") is None or baseline.get("max_drawdown") is None or
            candidate["max_drawdown"] < baseline["max_drawdown"] -
            cfg.weight_max_drawdown_worsening):
        reasons.append("drawdown_gate")
    proposed = normalize_weights(manifest.get("proposed_weights"))
    if proposed is None:
        reasons.append("invalid_proposed_weights")
    elif max(abs(proposed[column] - current_weights[column])
             for column in MODEL_COLUMNS) > cfg.weight_max_step:
        reasons.append("weight_step_gate")
    if active is not None and cfg.weight_promotion_cooldown_days > 0:
        cutoff = pd.to_datetime(active.get("evaluation_end"), errors="coerce")
        new_dates = [value for value in observed_dates if pd.notna(cutoff) and value > cutoff]
        if len(new_dates) < cfg.weight_promotion_cooldown_days:
            reasons.append("promotion_cooldown")
    return reasons


def _forward_returns(frame: pd.DataFrame, weights: dict[str, float],
                     after_date: str, cost: float) -> pd.Series:
    cutoff = pd.to_datetime(after_date, errors="coerce")
    if pd.isna(cutoff):
        return pd.Series(dtype=float)
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data[data["date"] > cutoff]
    vector = np.array([weights[column] for column in MODEL_COLUMNS], dtype=float)
    return _daily_returns(data, vector, cost)


def _maybe_rollback(cfg: RealtimeConfig, frame: pd.DataFrame,
                    active: Optional[dict]) -> Optional[dict]:
    if active is None or not isinstance(active.get("previous"), dict):
        return None
    previous = active["previous"]
    previous_weights = normalize_weights(previous.get("weights"))
    if previous_weights is None:
        return None
    current_returns = _forward_returns(
        frame, active["weights"], str(active.get("evaluation_end") or ""), cfg.paper_cost)
    previous_returns = _forward_returns(
        frame, previous_weights, str(active.get("evaluation_end") or ""), cfg.paper_cost)
    common = current_returns.index.intersection(previous_returns.index)
    if len(common) < cfg.weight_rollback_days:
        return None
    common = common[-cfg.weight_rollback_days:]
    current_window = current_returns.loc[common]
    previous_window = previous_returns.loc[common]
    current_total = float((1.0 + current_window).prod() - 1.0)
    previous_total = float((1.0 + previous_window).prod() - 1.0)
    if not (current_window.mean() < previous_window.mean() and
            current_total < previous_total):
        return None
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rollback = {
        "schema_version": 1, "scope": "realtime_return_weights", "state": "active",
        "version": str(previous.get("version") or "configured-default"),
        "promoted_at": now, "evaluation_end": str(common[-1].date()),
        "weights": previous_weights, "action": "automatic_rollback",
        "rollback_from": active.get("version"),
        "previous": {
            "version": active.get("version"), "weights": active["weights"],
            "promoted_at": active.get("promoted_at"),
            "evaluation_end": active.get("evaluation_end"),
        },
        "forward_check": {
            "days": len(common), "current_return": current_total,
            "previous_return": previous_total,
        },
    }
    atomic_write_json(cfg.weight_active_manifest_file, rollback)
    append_history(Path(cfg.weight_shadow_dir) / "promotion_history.jsonl", rollback)
    return rollback


def run_routine(cfg: RealtimeConfig) -> dict:
    source = Path(cfg.weight_shadow_predictions_file)
    frame = pd.read_parquet(source)
    required = {"code", "date", *MODEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weight shadow source missing columns: {missing}")
    frame, _ = _normalize_target_column(frame)
    current_weights, active = _current_weights(cfg)
    rollback = _maybe_rollback(cfg, frame, active)
    if rollback is not None:
        return {"action": "rolled_back", "active": rollback}

    manifest = evaluate(
        source, cfg.weight_shadow_dir,
        train_days=cfg.weight_shadow_train_days,
        min_oos_days=cfg.weight_shadow_min_oos_days,
        rebalance_days=cfg.weight_shadow_rebalance_days,
        cost=cfg.paper_cost,
        baseline_weights=np.array([current_weights[column] for column in MODEL_COLUMNS]),
        workers=getattr(cfg, "weight_shadow_workers", 4),
    )
    realized_mask = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").notna()
    dates = sorted(pd.to_datetime(
        frame.loc[realized_mask, "date"], errors="coerce").dropna().dt.normalize().unique())
    observed_dates = [pd.Timestamp(value) for value in dates]
    reasons = _promotion_reasons(manifest, cfg, current_weights, active, observed_dates)
    decision = {
        "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "version": manifest["version"], "action": "kept_champion",
        "state": manifest["state"], "reasons": reasons,
        "current_weights": current_weights,
        "proposed_weights": manifest["proposed_weights"],
    }
    if getattr(cfg, "weight_auto_promote_enabled", True) and not reasons:
        proposed = normalize_weights(manifest["proposed_weights"])
        previous = {
            "version": active.get("version") if active else "configured-default",
            "weights": current_weights,
            "promoted_at": active.get("promoted_at") if active else None,
            "evaluation_end": active.get("evaluation_end") if active else None,
        }
        promoted = {
            "schema_version": 1, "scope": "realtime_return_weights", "state": "active",
            "version": manifest["version"],
            "promoted_at": decision["time"],
            "evaluation_end": manifest["evaluation_period"]["end"],
            "weights": proposed, "action": "automatic_promotion",
            "source_manifest": f"versions/{manifest['version']}.json",
            "oos": manifest["oos"], "previous": previous,
        }
        atomic_write_json(cfg.weight_active_manifest_file, promoted)
        decision.update({"action": "promoted", "active": promoted})
    append_history(Path(cfg.weight_shadow_dir) / "promotion_history.jsonl", decision)
    manifest["promotion"].update({
        "manual_review_required": False,
        "automatic_decision": decision["action"],
        "automatic_reasons": reasons,
    })
    manifest["auto_apply"] = decision["action"] == "promoted"
    manifest["production_weights_unchanged"] = decision["action"] != "promoted"
    _atomic_json(Path(cfg.weight_shadow_dir) / "versions" / f"{manifest['version']}.json", manifest)
    _atomic_json(Path(cfg.weight_shadow_dir) / "manifest.json", manifest)
    return {"action": decision["action"], "manifest": manifest,
            "active": decision.get("active")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realtime-only return-weight shadow evaluation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    cfg = load()
    if args.predictions is not None:
        cfg.weight_shadow_predictions_file = args.predictions
    if args.output_dir is not None:
        cfg.weight_shadow_dir = args.output_dir
    if args.evaluate_only:
        manifest = evaluate_config(cfg)
        result = {
            "action": "evaluated_only", "version": manifest["version"],
            "state": manifest["state"], "proposed_weights": manifest["proposed_weights"],
        }
    else:
        routine = run_routine(cfg)
        manifest = routine.get("manifest") or {}
        active = routine.get("active") or {}
        result = {
            "action": routine["action"],
            "version": manifest.get("version") or active.get("version"),
            "state": manifest.get("state") or active.get("state"),
            "active_weights": active.get("weights"),
            "proposed_weights": manifest.get("proposed_weights"),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
