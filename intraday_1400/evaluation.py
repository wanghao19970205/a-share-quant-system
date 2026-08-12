from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.storage import atomic_json


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min()) if len(curve) else float("nan")


def _daily_turnover(selected: pd.DataFrame) -> float:
    sets = [set(part["code"].astype(str)) for _, part in selected.groupby("date", sort=True)]
    if len(sets) < 2:
        return float("nan")
    values = [1.0 - len(left & right) / max(len(left | right), 1) for left, right in zip(sets, sets[1:])]
    return float(np.mean(values))


def _bootstrap_ci(values: pd.Series, samples: int = 1000, seed: int = 42) -> list[float] | None:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 10:
        return None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(clean, size=(samples, len(clean)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _circular_block_bootstrap_means(
    values: np.ndarray,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean) or samples <= 0:
        return np.asarray([], dtype=float)
    length = len(clean)
    block = min(max(int(block_length), 1), length)
    block_count = int(np.ceil(length / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, length, size=(int(samples), block_count))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % length
    draws = clean[indices.reshape(int(samples), -1)[:, :length]]
    return draws.mean(axis=1)


def _paired_block_bootstrap(
    model: pd.Series,
    baseline: pd.Series,
    samples: int = 2000,
    block_length: int = 5,
    seed: int = 42,
) -> dict:
    paired = pd.concat(
        [model.rename("model"), baseline.rename("baseline")], axis=1, join="inner",
    ).dropna()
    gain = paired["model"] - paired["baseline"]
    if gain.empty:
        return {"days": 0, "available": False}
    values = gain.to_numpy(dtype=float)
    observed = float(values.mean())
    boot = _circular_block_bootstrap_means(values, samples, block_length, seed)
    centered_boot = _circular_block_bootstrap_means(
        values - observed, samples, block_length, seed + 1,
    )
    result = {
        "available": bool(len(values) >= 10 and len(boot)),
        "days": int(len(values)),
        "paired_mean_gain": observed,
        "model_mean_return": float(paired["model"].mean()),
        "baseline_mean_return": float(paired["baseline"].mean()),
        "block_method": "circular_moving_block",
        "block_length": min(max(int(block_length), 1), len(values)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }
    if result["available"]:
        result["ci95"] = [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ]
        result["p_value_one_sided"] = float(
            (np.count_nonzero(centered_boot >= observed) + 1) / (len(centered_boot) + 1)
        )
    else:
        result["ci95"] = None
        result["p_value_one_sided"] = None
    return result


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _daily_returns(
    selected: pd.DataFrame,
    gross_column: str,
    cost: float,
    rebalance_stride: int = 1,
) -> tuple[pd.Series, pd.DataFrame, int, int, float]:
    stride = max(int(rebalance_stride), 1)
    evaluated = selected.copy()
    rebalance_dates = pd.Index(evaluated["date"].dropna().sort_values().unique())[::stride]
    evaluated = evaluated[evaluated["date"].isin(rebalance_dates)].copy()
    observed_column = next(
        (column for column in ("target_outcome_observed_t1", "outcome_observed") if column in evaluated),
        None,
    )
    observed = (
        evaluated[observed_column].fillna(False).astype(bool)
        if observed_column else pd.Series(True, index=evaluated.index)
    )
    buyable = (
        evaluated["entry_buyable"].fillna(False).astype(bool)
        if "entry_buyable" in evaluated else pd.Series(True, index=evaluated.index)
    )
    immature = buyable & ~observed
    immature_targets = int(immature.sum())
    evaluated = evaluated[~immature].copy()
    buyable = buyable[~immature]
    missing_targets = int((buyable & evaluated[gross_column].isna()).sum())
    unsellable_return = float(
        os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
    )
    evaluated.loc[~buyable, gross_column] = float(cost)
    unsellable = buyable & evaluated[gross_column].isna()
    evaluated.loc[unsellable, gross_column] = unsellable_return + float(cost)
    daily = (evaluated.groupby("date")[gross_column].mean() - float(cost)).dropna()
    return daily, evaluated, missing_targets, immature_targets, unsellable_return


def _metrics(selected: pd.DataFrame, gross_column: str, cost: float, rebalance_stride: int = 1) -> dict:
    daily, evaluated, missing_targets, immature_targets, unsellable_return = _daily_returns(
        selected, gross_column, cost, rebalance_stride,
    )
    if daily.empty:
        return {"days": 0}
    std = float(daily.std())
    periods_per_year = 252.0 / max(int(rebalance_stride), 1)
    return {
        "days": int(len(daily)),
        "mean_return": float(daily.mean()),
        "median_return": float(daily.median()),
        "win_rate": float((daily > 0).mean()),
        "sharpe": float(daily.mean() / std * np.sqrt(periods_per_year)) if std > 0 else None,
        "max_drawdown": _max_drawdown(daily),
        "compound_return": float((1.0 + daily).prod() - 1.0),
        "mean_return_ci95": _bootstrap_ci(daily),
        "turnover": _daily_turnover(evaluated),
        "mean_names": float(evaluated.groupby("date")["code"].nunique().mean()),
        "missing_targets": missing_targets,
        "immature_targets": immature_targets,
        "unsellable_return": unsellable_return,
    }


def _mean_return_only(
    selected: pd.DataFrame,
    gross_column: str,
    cost: float,
    rebalance_stride: int = 1,
) -> float:
    """Compute one control sample mean without bootstrap or auxiliary metrics."""
    daily, _, _, _, _ = _daily_returns(selected, gross_column, cost, rebalance_stride)
    return float(daily.mean()) if not daily.empty else float("nan")


_CONTROL_FRAME: pd.DataFrame | None = None
_CONTROL_GROSS_COLUMNS: dict[str, str] = {}
_CONTROL_COSTS: tuple[float, ...] = ()
_CONTROL_REBALANCE_STRIDE = 1


def _init_control_worker(
    frame: pd.DataFrame,
    gross_columns: dict[str, str],
    costs: tuple[float, ...],
    rebalance_stride: int,
) -> None:
    global _CONTROL_FRAME, _CONTROL_GROSS_COLUMNS, _CONTROL_COSTS, _CONTROL_REBALANCE_STRIDE
    _CONTROL_FRAME = frame
    _CONTROL_GROSS_COLUMNS = gross_columns
    _CONTROL_COSTS = costs
    _CONTROL_REBALANCE_STRIDE = max(int(rebalance_stride), 1)


def _control_seed_worker(
    args: tuple[int, int],
) -> dict[tuple[int, str, float, str], tuple[float, pd.Series]]:
    top_n, seed = args
    if _CONTROL_FRAME is None:
        raise RuntimeError("control worker was not initialized")
    controlled = _CONTROL_FRAME.copy()
    rng = np.random.default_rng(int(seed))
    controlled["random_score"] = rng.random(len(controlled))
    controlled["shuffled_score"] = controlled.groupby("date")["score"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    random_rank = controlled.groupby("date")["random_score"].rank(method="first", ascending=False)
    shuffled_rank = controlled.groupby("date")["shuffled_score"].rank(method="first", ascending=False)
    selected = {
        "random_topn": controlled[random_rank <= int(top_n)],
        "score_shuffle": controlled[shuffled_rank <= int(top_n)],
    }
    output: dict[tuple[int, str, float, str], tuple[float, pd.Series]] = {}
    for target, gross_column in _CONTROL_GROSS_COLUMNS.items():
        for cost in _CONTROL_COSTS:
            for kind, frame in selected.items():
                daily, _, _, _, _ = _daily_returns(
                    frame, gross_column, cost, _CONTROL_REBALANCE_STRIDE,
                )
                output[(int(top_n), target, cost, kind)] = (
                    float(daily.mean()) if not daily.empty else float("nan"),
                    daily,
                )
    return output


def _load_labels() -> tuple[pd.DataFrame, dict]:
    paths = sorted(config.PREPARED_DIR.glob("????-??.parquet"))
    if not paths:
        raise RuntimeError("no prepared labels")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        target_columns = [column for column in frame.columns if column.startswith("target_")]
        frames.append(frame[["code", "date", "entry_buyable"] + target_columns])
    labels = pd.concat(frames, ignore_index=True)
    labels["code"] = labels["code"].astype(str).str[:6]
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    embedded_cost = float(os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002") or 0.002)
    target_columns = [
        column for column in labels.columns
        if (column == "target_net_ret_t1" or column.endswith("_proxy_net"))
        and pd.api.types.is_numeric_dtype(labels[column])
    ]
    gross_columns: dict[str, str] = {}
    for target in target_columns:
        gross_column = f"gross__{target}"
        labels[gross_column] = labels[target] + embedded_cost
        gross_columns[target] = gross_column
    labels = labels.drop_duplicates(["code", "date"], keep="last")
    return labels, {"embedded_cost": embedded_cost, "gross_columns": gross_columns}


def _load_prediction(path: Path, score_candidates: tuple[str, ...] = ("pred", "ensemble_pred", "score")) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    score = next((column for column in score_candidates if column in frame.columns), None)
    if score is None:
        raise ValueError(f"no score column in {path}")
    frame = frame[["code", "date", score]].rename(columns={score: "score"})
    frame["code"] = frame["code"].astype(str).str[:6]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.dropna(subset=["code", "date", "score"]).drop_duplicates(["code", "date"], keep="last")


def _evaluate_source(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    causal: bool,
    top_values: tuple[int, ...],
    costs: tuple[float, ...],
    metadata: dict,
) -> dict:
    merged = labels.merge(predictions, on=["code", "date"], how="inner", validate="one_to_one")
    merged["daily_rank"] = merged.groupby("date")["score"].rank(method="first", ascending=False)
    source_report = {
        "available": True,
        "path": str(path),
        "causal_at_1400": causal,
        "overlap_rows": int(len(merged)),
        "overlap_days": int(merged["date"].nunique()),
        "top": {},
    }
    for top_n in top_values:
        selected = merged[merged["daily_rank"] <= int(top_n)].copy()
        source_report["top"][str(top_n)] = {
            target: {
                f"cost_{int(round(cost * 10000))}bp": _metrics(selected, gross_column, cost)
                for cost in costs
            }
            for target, gross_column in metadata["gross_columns"].items()
        }
    return source_report


def _distribution_summary(values: list[float]) -> dict:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(clean):
        return {"samples": 0}
    return {
        "samples": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "p025": float(np.quantile(clean, 0.025)),
        "p975": float(np.quantile(clean, 0.975)),
    }


def _mean_control_daily(
    results: list[dict],
    key: tuple[int, str, float, str],
) -> pd.Series:
    series = [result[key][1].rename(index) for index, result in enumerate(results) if key in result]
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).mean(axis=1).sort_index()


def _evaluate_negative_controls(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    top_values: tuple[int, ...],
    costs: tuple[float, ...],
    metadata: dict,
    samples: int = 200,
    seed: int = 42,
    workers: int = 1,
    rebalance_stride: int = 1,
    bootstrap_samples: int = 2000,
    block_length: int = 5,
) -> dict:
    merged = labels.merge(predictions, on=["code", "date"], how="inner", validate="one_to_one")
    report = {
        "samples": int(samples),
        "seed": int(seed),
        "rebalance_stride": max(int(rebalance_stride), 1),
        "paired_bootstrap": {
            "method": "circular_moving_block",
            "samples": int(bootstrap_samples),
            "block_length": int(block_length),
            "seed": int(seed),
            "multiplicity": "holm_fwer",
        },
        "overlap_rows": int(len(merged)),
        "overlap_days": int(merged["date"].nunique()),
        "market_equal_weight": {},
        "top": {},
    }
    for target, gross_column in metadata["gross_columns"].items():
        report["market_equal_weight"][target] = {
            f"cost_{int(round(cost * 10000))}bp": _metrics(
                merged, gross_column, cost, rebalance_stride,
            )
            for cost in costs
        }
    worker_count = max(int(workers), 1)
    report["workers"] = worker_count
    jobs = [
        (int(top_n), int(seed) + index)
        for top_n in top_values
        for index in range(max(int(samples), 0))
    ]
    if worker_count > 1 and jobs:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_control_worker,
            initargs=(merged, metadata["gross_columns"], costs, rebalance_stride),
        ) as pool:
            results = list(pool.map(_control_seed_worker, jobs, chunksize=1))
    else:
        _init_control_worker(merged, metadata["gross_columns"], costs, rebalance_stride)
        results = [_control_seed_worker(job) for job in jobs]
    merged["model_rank"] = merged.groupby("date")["score"].rank(method="first", ascending=False)
    inference_entries: list[dict] = []
    for top_n in top_values:
        top_report: dict = {}
        model_selected = merged[merged["model_rank"] <= int(top_n)]
        for target, gross_column in metadata["gross_columns"].items():
            target_report: dict = {}
            for cost in costs:
                cost_key = f"cost_{int(round(cost * 10000))}bp"
                random_key = (int(top_n), target, cost, "random_topn")
                shuffle_key = (int(top_n), target, cost, "score_shuffle")
                random_values = [result[random_key][0] for result in results if random_key in result]
                shuffle_values = [result[shuffle_key][0] for result in results if shuffle_key in result]
                model_daily, _, _, _, _ = _daily_returns(
                    model_selected, gross_column, cost, rebalance_stride,
                )
                market_daily, _, _, _, _ = _daily_returns(
                    merged, gross_column, cost, rebalance_stride,
                )
                baselines = {
                    "market_equal_weight": market_daily,
                    "random_topn": _mean_control_daily(results, random_key),
                    "score_shuffle": _mean_control_daily(results, shuffle_key),
                }
                paired = {}
                for baseline_index, (baseline_name, baseline_daily) in enumerate(baselines.items()):
                    comparison = _paired_block_bootstrap(
                        model_daily,
                        baseline_daily,
                        samples=bootstrap_samples,
                        block_length=block_length,
                        seed=int(seed) + baseline_index,
                    )
                    comparison["baseline_definition"] = (
                        "equal_weight_market_daily_return"
                        if baseline_name == "market_equal_weight"
                        else "mean_daily_return_across_control_seeds"
                    )
                    if baseline_name != "market_equal_weight":
                        comparison["control_seed_samples"] = int(samples)
                    paired[baseline_name] = comparison
                    if comparison.get("p_value_one_sided") is not None:
                        inference_entries.append(comparison)
                target_report[cost_key] = {
                    "random_topn": _distribution_summary(random_values),
                    "score_shuffle": _distribution_summary(shuffle_values),
                    "paired_comparisons": paired,
                }
            top_report[target] = target_report
        report["top"][str(top_n)] = top_report
    adjusted = _holm_adjust([entry["p_value_one_sided"] for entry in inference_entries])
    for entry, value in zip(inference_entries, adjusted):
        entry["p_value_holm"] = float(value)
        entry["holm_family_size"] = len(inference_entries)
        entry["reject_holm_0_05"] = bool(value <= 0.05)
    return report


def _common_labels(labels: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = labels[["code", "date"]].drop_duplicates()
    for frame in predictions.values():
        keys = keys.merge(frame[["code", "date"]], on=["code", "date"], how="inner")
    return labels.merge(keys.drop_duplicates(), on=["code", "date"], how="inner", validate="one_to_one")


def evaluate(
    top_values: tuple[int, ...] = (5, 10, 20, 30),
    costs: tuple[float, ...] = (0.0, 0.001, 0.002, 0.003),
    control_samples: int = 200,
    control_seed: int = 42,
    control_workers: int = 1,
    bootstrap_samples: int = 2000,
    block_length: int = 5,
) -> dict:
    labels, metadata = _load_labels()
    manifest_path = config.MODEL_DIR / "intraday_1400_shadow_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    train_end = pd.Timestamp(manifest.get("train_end", labels["date"].min()))
    valid_end = pd.Timestamp(manifest.get("valid_end", labels["date"].max()))
    labels = labels[(labels["date"] > train_end) & (labels["date"] <= valid_end)].copy()

    sources: dict[str, tuple[Path, bool]] = {
        "asof_base": (config.MODEL_DIR / "asof_base_predictions.parquet", True),
        "asof_plus_intraday": (config.MODEL_DIR / "asof_plus_intraday_predictions.parquet", True),
    }
    quant_dir = Path(os.environ.get("QUANT_DATA_DIR", "quant_data"))
    sources["close_active_reference"] = (
        quant_dir / "active_quant_short_predictions.parquet",
        False,
    )
    report = {
        "schema_version": config.SCHEMA_VERSION,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
        "label_rows": int(len(labels)),
        "label_days": int(labels["date"].nunique()),
        "costs": list(costs),
        "top_values": list(top_values),
        "sources": {},
        "common_universe": {},
        "negative_controls": {},
        **metadata,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, (path, causal) in sources.items():
        if not path.exists():
            report["sources"][name] = {"available": False, "path": str(path)}
            continue
        predictions = _load_prediction(path)
        loaded[name] = predictions
        report["sources"][name] = _evaluate_source(
            labels, predictions, path, causal, top_values, costs, metadata,
        )

    common_groups = {
        "causal_common": ["asof_base", "asof_plus_intraday"],
        "all_sources_common": list(sources),
    }
    for group_name, names in common_groups.items():
        if not all(name in loaded for name in names):
            report["common_universe"][group_name] = {"available": False, "sources": names}
            continue
        group_predictions = {name: loaded[name] for name in names}
        common = _common_labels(labels, group_predictions)
        group_report = {
            "available": True,
            "sources": names,
            "common_rows": int(len(common)),
            "common_days": int(common["date"].nunique()),
            "results": {},
        }
        for name in names:
            path, causal = sources[name]
            group_report["results"][name] = _evaluate_source(
                common, loaded[name], path, causal, top_values, costs, metadata,
            )
        report["common_universe"][group_name] = group_report
    causal_names = ["asof_base", "asof_plus_intraday"]
    if all(name in loaded for name in causal_names):
        causal_common = _common_labels(labels, {name: loaded[name] for name in causal_names})
        report["negative_controls"] = _evaluate_negative_controls(
            causal_common,
            loaded["asof_plus_intraday"],
            top_values,
            costs,
            metadata,
            samples=control_samples,
            seed=control_seed,
            workers=control_workers,
            bootstrap_samples=bootstrap_samples,
            block_length=block_length,
        )
    else:
        report["negative_controls"] = {"available": False, "sources": causal_names}
    atomic_json(report, config.REPORT_DIR / "fair_daily_topn_evaluation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair daily TopN evaluation for intraday 14:00 models")
    parser.add_argument("--top", default="5,10,20,30")
    parser.add_argument("--costs-bp", default="0,10,20,30")
    parser.add_argument("--control-samples", type=int, default=200)
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument("--control-workers", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=5)
    args = parser.parse_args()
    top_values = tuple(int(value) for value in args.top.split(",") if int(value) > 0)
    costs = tuple(float(value) / 10000.0 for value in args.costs_bp.split(","))
    evaluate(
        top_values,
        costs,
        args.control_samples,
        args.control_seed,
        args.control_workers,
        args.bootstrap_samples,
        args.block_length,
    )


if __name__ == "__main__":
    main()
