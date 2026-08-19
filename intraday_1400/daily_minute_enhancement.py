from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_1400 import config, pipeline
from intraday_1400.direct_return_experiment import (
    EXECUTION_CONFIG,
    _execution_labels,
    _selected_features,
    _simulate_adaptive_label_race,
)
from intraday_1400.fair_race_pipeline import (
    DEFAULT_TRAINED_VARIANTS,
    _causal_eligible_predictions,
    default_daily_prepared_dir,
    load_joined_prepared,
    panel_for_variant,
    screen_window_features,
    _cap_training_panel,
)
from intraday_1400.offline_race import ExecutionConfig, compare_execution_records
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records
from intraday_1400.target_redesign_backfill import (
    FOLD_POSITIONS,
    _cycle_lock,
    HISTORICAL_END,
    HISTORICAL_START,
    TOP_N,
    combine_historical_labels,
    registered_folds,
    validate_historical_dates,
)
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from quant import model


PROTOCOL = "intraday_1400_daily_minute_enhancement_v2"
BASELINE = "daily_asof_baseline"
ALL_MINUTE = "daily_plus_all_minute"
MINUTE_FAMILIES = ("speed", "path", "volume_vwap", "risk", "dependence", "context")
MODEL_RECIPE = {
    # v2: train on the executable label the race is scored on. The raw daily
    # 1d return rewards names that close limit-up, which are exactly the names
    # that cannot be bought at 14:50, so a model trained on it ranks unbuyable
    # names first. The adaptive t+3 label is net of cost and undefined for
    # unbuyable entries, and _xy drops those rows from train/valid while
    # predictions still cover the full eligible universe.
    "target": "adaptive_realized_net_ret_t3",
    "selection_pool": "order_time_buyable",
    "ridge": {"weight": 0.15, "alpha": 10.0},
    "lightgbm_ranker": {
        "weight": 0.85,
        "n_estimators": 200,
        "learning_rate": 0.015,
        "early_stopping_rounds": 0,
        "rank_bins": 5,
        "eval_at": [10],
    },
    "decay_half_life_days": 60.0,
    "min_weight": 0.03,
    "max_train_rows": 100_000,
    "top_n": TOP_N,
    "roundtrip_cost": 0.002,
    "schema_version": config.SCHEMA_VERSION,
    "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
    "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
    "label_recipe_version": config.LABEL_RECIPE_VERSION,
    "train_recipe_version": config.TRAIN_RECIPE_VERSION,
    "cutoff_time": "13:55",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "historical_start": str(HISTORICAL_START.date()),
        "historical_end": str(HISTORICAL_END.date()),
        "fold_positions": FOLD_POSITIONS,
        "baseline": BASELINE,
        "candidate_grid": [ALL_MINUTE, *[f"daily_plus_{name}" for name in MINUTE_FAMILIES]],
        "model_recipe": MODEL_RECIPE,
        "feature_screening": {
            "method": "recompute_from_verified_1355_prepared",
            "train_end": "2025-08-06",
            "total_feature_budget": 80,
        },
        "selection_gate": {
            "minimum_fill_rate": 0.60,
            "minimum_fold_wins_vs_baseline": 3,
            "mean_return_must_exceed_baseline": True,
            "mean_and_compound_return_must_be_positive": True,
            "max_drawdown_tolerance": 0.05,
        },
        "historical_results_are_untouched": False,
        "production_publication": False,
        "human_approval_required": True,
    }


def minute_features_by_family(minute_features: list[str]) -> dict[str, list[str]]:
    grouped = {name: [] for name in MINUTE_FAMILIES}
    for feature in minute_features:
        family = pipeline._minute_family(feature.removeprefix("minute__"))
        if family not in grouped:
            raise ValueError(f"unknown minute feature family: {family}")
        grouped[family].append(feature)
    if any(not grouped[name] for name in MINUTE_FAMILIES):
        empty = [name for name in MINUTE_FAMILIES if not grouped[name]]
        raise ValueError(f"registered minute feature families are empty: {empty}")
    return grouped


def candidate_features(
    base_features: list[str],
    minute_features: list[str],
) -> dict[str, list[str]]:
    grouped = minute_features_by_family(minute_features)
    result = {BASELINE: list(base_features), ALL_MINUTE: [*base_features, *minute_features]}
    for family in MINUTE_FAMILIES:
        result[f"daily_plus_{family}"] = [*base_features, *grouped[family]]
    if any(len(features) != len(set(features)) for features in result.values()):
        raise ValueError("daily-minute candidate features must be unique")
    return result


def _causal_screened_features(
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
    prepared_provenance: dict,
    max_rows: int | None = None,
) -> tuple[dict, list[str], list[str], dict]:
    source_window, _, _, _ = _selected_features(screening_report_path)
    train_end = pd.Timestamp(source_window["train_end"])
    manifest_path = Path(prepared_provenance["feature_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_features = set(manifest.get("features", []))
    minute_sources = [
        feature for feature in manifest.get("features", [])
        if str(feature).startswith("m5_")
    ]
    if not minute_sources:
        raise RuntimeError("verified feature manifest has no minute feature sources")
    screening_panel, groups = load_joined_prepared(
        daily_dir,
        intraday_dir,
        pd.Timestamp("2018-01-01"),
        train_end,
        minute_features=minute_sources,
        max_rows=max_rows,
    )
    # The causal gate only consumes the matched base and minute selections.
    # Avoid screening unrelated diagnostic variants that cannot affect output.
    screening_variants = tuple(
        variant for variant in DEFAULT_TRAINED_VARIANTS
        if variant.name in {"daily_close_control", "daily_close_plus_minute_control"}
    )
    selected = screen_window_features(
        screening_panel,
        groups,
        train_end,
        variants=screening_variants,
        align_controls=False,
    )
    legacy_control = selected.get("daily_asof_plus_minute_control", {})
    close_control = selected.get("daily_close_control")
    if close_control is None:
        # Keep compatibility with callers that provide the already aligned control.
        base_features = list(legacy_control["asof_matched"])
    else:
        base_features = [
            f"asof__{feature.removeprefix('daily__')}"
            for feature in close_control["daily_matched"]
        ]
    minute_control = selected.get("daily_close_plus_minute_control", legacy_control)
    if groups.get("minute"):
        causal_variant = next(
            variant for variant in DEFAULT_TRAINED_VARIANTS
            if variant.name == "daily_asof_plus_minute_control"
        )
        selected_minute, selected_by_family = pipeline._select_minute_features_grouped(
            panel_for_variant(screening_panel, causal_variant),
            groups["minute"],
            str(train_end.date()),
            top_n=40,
            quota=5,
            label_col="target_excess_ret_t1",
        )
        minute_features = list(selected_minute)
    else:
        selected_by_family = {}
        minute_features = list(minute_control["minute"])
    if not base_features or not minute_features:
        raise RuntimeError("causal daily-minute screening selected an empty feature group")
    selected_sources = {
        feature.removeprefix("asof__") for feature in base_features
    } | {
        feature.removeprefix("minute__") for feature in minute_features
    }
    missing_manifest = sorted(selected_sources - manifest_features)
    if missing_manifest:
        raise RuntimeError(
            f"causal selected features are missing from the prepared manifest: {missing_manifest}"
        )
    evidence = {
        "method": "recomputed_from_verified_1355_prepared",
        "train_end": str(train_end.date()),
        "screening_rows": int(len(screening_panel)),
        "screening_dates": int(screening_panel["date"].nunique()),
        "selected_hash": _canonical_hash({"base": base_features, "minute": minute_features}),
        "minute_family_counts": {
            family: len(selected_by_family.get(family, []))
            for family in MINUTE_FAMILIES
        },
        "minute_selection_source": "verified_feature_manifest_recomputed_by_family",
        "source_report_window_metadata_hash": _canonical_hash({
            key: source_window.get(key)
            for key in ("window", "train_end", "purge_date", "valid_start", "valid_end")
        }),
    }
    return source_window, base_features, minute_features, evidence


def _causal_screened_features_cached(
    cache_path: Path,
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
    inputs: dict,
    max_rows: int | None = None,
) -> tuple[dict, list[str], list[str], dict]:
    """Return the causal screening selection, reusing a verified cache.

    Screening is a deterministic function of the frozen inputs, yet a
    fold/candidate subset run repeats it for every pair. The cache key carries
    the full input hash set, which already covers the prepared provenance, the
    screening report and every code dependency, so any change to data or code
    forces a recomputation.
    """
    cache_key = {
        "protocol": PROTOCOL,
        "input_hashes": inputs,
        "max_rows": int(max_rows) if max_rows is not None else None,
    }
    cache_path = Path(cache_path)
    if cache_path.is_file():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved.get("cache_key") == cache_key:
            print("[daily-minute] reuse screening cache", flush=True)
            return (
                saved["feature_window"],
                list(saved["base_features"]),
                list(saved["minute_features"]),
                saved["screening_evidence"],
            )
    feature_window, base_features, minute_features, evidence = _causal_screened_features(
        screening_report_path,
        daily_dir,
        intraday_dir,
        inputs["prepared_provenance"],
        max_rows=max_rows,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        {
            "cache_key": cache_key,
            "feature_window": feature_window,
            "base_features": base_features,
            "minute_features": minute_features,
            "screening_evidence": evidence,
        },
        cache_path,
    )
    return feature_window, base_features, minute_features, evidence


def _intraday_eligible_keys(intraday_dir: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    start_month = pd.Timestamp(dates[0]).strftime("%Y-%m")
    end_month = pd.Timestamp(dates[-1]).strftime("%Y-%m")
    paths = [
        path for path in sorted(Path(intraday_dir).glob("????-??.parquet"))
        if start_month <= path.stem <= end_month
    ]
    if not paths:
        raise RuntimeError("intraday prepared eligibility artifacts are unavailable")
    parts = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["date", "code", "signal_eligible"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["code"] = frame["code"].astype(str).str[:6]
        parts.append(frame[
            frame["date"].isin(dates)
            & frame["signal_eligible"].fillna(False).astype(bool)
        ][["date", "code"]])
    keys = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"])
    if keys.duplicated(["date", "code"]).any():
        raise RuntimeError("intraday prepared eligibility contains duplicate keys")
    return keys.sort_values(["date", "code"]).reset_index(drop=True)


def _build_panel(
    labels: pd.DataFrame,
    daily_dir: Path,
    intraday_dir: Path,
    base_features: list[str],
    minute_features: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    dates = pd.DatetimeIndex(labels["date"].unique()).sort_values()
    panel, _ = load_joined_prepared(
        daily_dir,
        intraday_dir,
        dates[0],
        dates[-1],
        daily_features=[name.removeprefix("asof__") for name in base_features],
        asof_features=[name.removeprefix("asof__") for name in base_features],
        minute_features=[name.removeprefix("minute__") for name in minute_features],
        key_filter=labels[["code", "date"]],
    )
    if "daily_target_ret_1d" not in panel:
        raise ValueError("matched daily h1 target is unavailable")
    selected_features = [*base_features, *minute_features]
    missing_features = sorted(set(selected_features) - set(panel.columns))
    if missing_features:
        raise ValueError(f"matched panel is missing selected features: {missing_features}")
    eligible = panel[panel["signal_eligible"].fillna(False).astype(bool)].copy()
    joined_prepared_keys = eligible[["date", "code"]].drop_duplicates(["date", "code"])
    intraday_eligible_keys = _intraday_eligible_keys(intraday_dir, dates)
    label_keys = labels[["date", "code"]].drop_duplicates(["date", "code"])
    eligibility_check = intraday_eligible_keys.merge(
        label_keys, on=["date", "code"], how="outer", indicator=True
    )
    if (eligibility_check["_merge"] == "right_only").any():
        raise ValueError("historical adaptive labels contain keys outside intraday signal eligibility")
    joined_check = joined_prepared_keys.merge(
        label_keys, on=["date", "code"], how="outer", indicator=True
    )
    panel = eligible.merge(labels, on=["date", "code"], how="inner", validate="one_to_one")
    if MODEL_RECIPE["target"] not in panel:
        raise ValueError(f"model training target {MODEL_RECIPE['target']} is unavailable")
    final_keys = panel[["date", "code"]].drop_duplicates(["date", "code"])

    def hashes_by_date(keys: pd.DataFrame) -> dict[str, str]:
        return {
            str(pd.Timestamp(date).date()): _canonical_hash(sorted(group["code"].astype(str)))
            for date, group in keys.sort_values(["date", "code"]).groupby("date")
        }

    hashes = {
        "intraday_signal_eligible": hashes_by_date(intraday_eligible_keys),
        "joined_prepared_signal_eligible": hashes_by_date(joined_prepared_keys),
        "adaptive_labels": hashes_by_date(label_keys),
        "final_matched": hashes_by_date(final_keys),
        "final_all_keys": _canonical_hash(
            final_keys.sort_values(["date", "code"]).astype(str).to_dict("records")
        ),
    }
    panel.attrs["prepared_vs_label_key_counts"] = {
        "intraday_eligibility_vs_labels": {
            str(name): int(count)
            for name, count in eligibility_check["_merge"].value_counts().items()
        },
        "joined_prepared_vs_labels": {
            str(name): int(count)
            for name, count in joined_check["_merge"].value_counts().items()
        },
        "final_matched_keys": int(len(final_keys)),
    }
    return panel, hashes


def _project_model_panel(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    columns = ["code", "date", str(MODEL_RECIPE["target"]), *features]
    missing = sorted(set(columns) - set(panel.columns))
    if missing:
        raise ValueError(f"daily-minute model panel is missing columns: {missing}")
    return panel.loc[:, list(dict.fromkeys(columns))].copy()


def _model_worker(config_path: Path) -> None:
    worker = json.loads(Path(config_path).read_text(encoding="utf-8"))
    panel = pd.read_parquet(worker["panel_path"])
    common = {
        "panel": panel,
        "features": worker["features"],
        "horizon": 1,
        "train_end": worker["train_end"],
        "valid_end": worker["valid_end"],
        "predict_start": worker["predict_start"],
        "predict_end": worker["predict_end"],
        "decay_half_life_days": float(MODEL_RECIPE["decay_half_life_days"]),
        "min_weight": float(MODEL_RECIPE["min_weight"]),
        "label_col": str(MODEL_RECIPE["target"]),
        "train_mask_col": None,
    }
    if worker["model"] == "ridge":
        recipe = MODEL_RECIPE["ridge"]
        result = model.train_ridge(**common, alpha=float(recipe["alpha"]))
    elif worker["model"] == "lightgbm_ranker":
        recipe = MODEL_RECIPE["lightgbm_ranker"]
        result = model.train_lightgbm_ranker(
            **common,
            n_estimators=int(recipe["n_estimators"]),
            learning_rate=float(recipe["learning_rate"]),
            early_stopping_rounds=int(recipe["early_stopping_rounds"]),
            n_jobs=int(worker["threads"]),
            rank_bins=int(recipe["rank_bins"]),
            eval_at=tuple(recipe["eval_at"]),
        )
    else:
        raise ValueError(f"unknown daily-minute model worker: {worker['model']}")
    if not result.ok:
        raise RuntimeError(result.message)
    predictions_path = Path(worker["predictions_path"])
    metrics_path = Path(worker["metrics_path"])
    atomic_parquet(result.predictions, predictions_path)
    atomic_json({
        "model": worker["model"],
        "rows": len(result.predictions),
        "predictions_sha256": artifact_hash(predictions_path),
        "metrics": result.metrics,
    }, metrics_path)


def _fit_daily_head(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    candidate: str,
    fold: str,
    artifact_dir: Path | None = None,
    max_train_rows: int | None = None,
) -> pd.DataFrame:
    panel = _cap_training_panel(
        panel,
        train_end,
        max_train_rows=(
            int(max_train_rows)
            if max_train_rows is not None
            else int(MODEL_RECIPE["max_train_rows"])
        ),
    )
    cleanup_dir = artifact_dir is None
    worker_dir = Path(artifact_dir or tempfile.mkdtemp(prefix="daily_minute_workers_"))
    worker_dir.mkdir(parents=True, exist_ok=True)
    panel_path = worker_dir / "model_panel.parquet"
    atomic_parquet(panel, panel_path)
    del panel
    gc.collect()
    predictions = {}
    try:
        for model_name in ("ridge", "lightgbm_ranker"):
            predictions_path = worker_dir / f"{model_name}_predictions.parquet"
            metrics_path = worker_dir / f"{model_name}_metrics.json"
            config_path = worker_dir / f"{model_name}_worker.json"
            atomic_json({
                "model": model_name,
                "panel_path": str(panel_path),
                "features": features,
                "train_end": str(train_end.date()),
                "valid_end": str(oos_end.date()),
                "predict_start": str(oos_start.date()),
                "predict_end": str(oos_end.date()),
                "threads": max(int(threads), 1),
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
            }, config_path)
            worker_code = (
                "from pathlib import Path; "
                "from intraday_1400.daily_minute_enhancement import _model_worker; "
                "_model_worker(Path(__import__('sys').argv[1]))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", worker_code, str(config_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "worker exited without output")[-2000:]
                raise RuntimeError(f"{candidate} {fold} {model_name} worker failed: {detail}")
            if not predictions_path.is_file() or not metrics_path.is_file():
                raise RuntimeError(f"{candidate} {fold} {model_name} worker produced no artifact")
            metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metadata.get("predictions_sha256") != artifact_hash(predictions_path):
                raise RuntimeError(f"{candidate} {fold} {model_name} prediction artifact hash mismatch")
            predictions[model_name] = pd.read_parquet(predictions_path)
    finally:
        panel_path.unlink(missing_ok=True)
        if cleanup_dir:
            shutil.rmtree(worker_dir, ignore_errors=True)
    ridge_recipe = MODEL_RECIPE["ridge"]
    ranker_recipe = MODEL_RECIPE["lightgbm_ranker"]
    merged = predictions["ridge"][['code', 'date', 'pred']].rename(
        columns={"pred": "ridge_pred"}
    ).merge(
        predictions["lightgbm_ranker"][["code", "date", "pred"]].rename(columns={"pred": "ranker_pred"}),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["ridge_z"] = pipeline._prediction_zscore(merged, "ridge_pred")
    merged["ranker_z"] = pipeline._prediction_zscore(merged, "ranker_pred")
    merged["score"] = (
        float(ridge_recipe["weight"]) * merged["ridge_z"]
        + float(ranker_recipe["weight"]) * merged["ranker_z"]
    )
    return merged[["code", "date", "score"]]


def _evaluate_candidate(
    name: str,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    filtered = _causal_eligible_predictions({name: predictions}, panel, dates[0], dates[-1])[name]
    records = _simulate_adaptive_label_race(
        {name: filtered},
        _execution_labels(panel, dates),
        ExecutionConfig(top_n=TOP_N, select_from_buyable_only=True, **EXECUTION_CONFIG),
    )
    account = cash_normalized_execution_records(records, dates, top_n=TOP_N, models=[name])
    comparison = compare_execution_records(account)
    return records, comparison["models"][name]


def select_enhancement(
    aggregate: dict[str, dict],
    fold_metrics: list[dict],
) -> dict:
    if BASELINE not in aggregate:
        raise ValueError("daily-minute selection requires the matched daily baseline")
    expected = {BASELINE, ALL_MINUTE, *[f"daily_plus_{name}" for name in MINUTE_FAMILIES]}
    if set(aggregate) != expected:
        raise ValueError("daily-minute selection requires the exact registered candidate grid")
    registered_fold_names = [item["name"] for item in FOLD_POSITIONS]
    fold_names = [item.get("name") for item in fold_metrics]
    if fold_names != registered_fold_names:
        raise ValueError("daily-minute selection requires the ordered registered four folds")
    for fold in fold_metrics:
        if set(fold.get("models", {})) != expected:
            raise ValueError(f"daily-minute fold {fold.get('name')} has an incomplete candidate grid")
        for name, metrics in fold["models"].items():
            value = metrics.get("mean_return")
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"daily-minute fold {fold['name']} candidate {name} is non-finite")
    baseline = aggregate[BASELINE]
    eligible = []
    for name in sorted(expected - {BASELINE}):
        metrics = aggregate[name]
        required = [metrics.get(key) for key in ("mean_return", "compound_return", "max_drawdown", "mean_filled_names")]
        if not np.isfinite(np.asarray(required, dtype=float)).all():
            raise ValueError(f"candidate {name} has non-finite metrics")
        fold_wins = sum(
            item["models"][name]["mean_return"] > item["models"][BASELINE]["mean_return"]
            for item in fold_metrics
        )
        fill_rate = float(metrics["mean_filled_names"]) / TOP_N
        if (
            fill_rate >= 0.60
            # next_branch=forward_shadow moves a candidate toward production, so
            # beating the baseline is necessary but not sufficient: a candidate
            # that loses money out of sample in every fold must not be promoted
            # just because the baseline loses more.
            and float(metrics["mean_return"]) > 0.0
            and float(metrics["compound_return"]) > 0.0
            and float(metrics["mean_return"]) > float(baseline["mean_return"])
            and float(metrics["max_drawdown"]) >= float(baseline["max_drawdown"]) - 0.05
            and fold_wins >= 3
        ):
            eligible.append((name, fold_wins))
    if not eligible:
        return {
            "status": "no_enhancement_passed",
            "selected": BASELINE,
            "next_branch": "minute_feature_residualization",
        }
    selected = max(
        eligible,
        key=lambda item: (
            float(aggregate[item[0]]["mean_return"]),
            float(aggregate[item[0]]["compound_return"]),
            item[1],
            item[0],
        ),
    )[0]
    return {
        "status": "enhancement_selected",
        "selected": selected,
        "next_branch": "forward_shadow",
    }


def _validate_research_paths(output_dir: Path, state_dir: Path, intraday_dir: Path) -> None:
    research_root = intraday_dir.resolve().parent
    for name, path in (("output", output_dir.resolve()), ("state", state_dir.resolve())):
        if research_root != path and research_root not in path.parents:
            raise RuntimeError(f"daily-minute {name} path must stay under the isolated intraday research root")
    forbidden = ("active_quant", "factor_panel_mainboard_active", "scheduler")
    for path in (output_dir, state_dir):
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError("daily-minute research cannot write production or scheduler artifacts")


def _prepared_provenance(intraday_dir: Path) -> dict:
    if config.CUTOFF_TIME != "13:55":
        raise RuntimeError(f"daily-minute research requires cutoff 13:55, found {config.CUTOFF_TIME}")
    data_root = intraday_dir.resolve().parent
    prepare_state = data_root / "checkpoints" / "prepare_state.json"
    feature_manifest = data_root / "models" / "feature_manifest.json"
    if not prepare_state.is_file() or not feature_manifest.is_file():
        raise RuntimeError("minute prepared provenance artifacts are missing")
    prepared_months = sorted(path.stem for path in intraday_dir.glob("????-??.parquet"))
    if not prepared_months:
        raise RuntimeError("minute prepared provenance has no monthly artifacts")
    state = json.loads(prepare_state.read_text(encoding="utf-8"))
    prepare_recipe = {
        "schema_version": config.SCHEMA_VERSION,
        "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "cutoff_time": config.CUTOFF_TIME,
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
    }
    verified_signatures = {}
    for month in prepared_months:
        month_dir = data_root / "features" / month
        part_paths = sorted(month_dir.glob("*.parquet"))
        if not part_paths:
            raise RuntimeError(f"minute prepare provenance source is missing for {month}")
        industry_history = Path(
            os.environ.get("SNAPSHOT_DIR", "snapshots")
        ) / "sw_industry_history_pit.parquet"
        signature_paths = (
            part_paths
            + ([industry_history] if industry_history.exists() else [])
            + pipeline._nonprice_sources(month)
        )
        signature_payload = (
            f"{pipeline._signature(signature_paths)}:"
            f"{json.dumps(prepare_recipe, sort_keys=True)}"
        )
        expected = hashlib.sha1(signature_payload.encode("utf-8")).hexdigest()
        if state.get(month) != expected:
            raise RuntimeError(f"minute prepared signature does not attest the frozen recipe for {month}")
        verified_signatures[month] = expected
    return {
        "cutoff_time": config.CUTOFF_TIME,
        "prepare_recipe": prepare_recipe,
        "label_recipe_version": config.LABEL_RECIPE_VERSION,
        "train_recipe_version": config.TRAIN_RECIPE_VERSION,
        "prepare_state": {
            "path": str(prepare_state.resolve()), "sha256": artifact_hash(prepare_state)
        },
        "feature_manifest": {
            "path": str(feature_manifest.resolve()), "sha256": artifact_hash(feature_manifest)
        },
        "prepared_months": prepared_months,
        "verified_monthly_signatures": verified_signatures,
    }


def _input_hashes(
    label_paths: list[Path],
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
) -> dict:
    module_dir = Path(__file__).parent
    dependencies = {
        "controller": Path(__file__),
        "model": module_dir.parent / "quant" / "model.py",
        "pipeline": module_dir / "pipeline.py",
        "fair_race": module_dir / "fair_race_pipeline.py",
        "direct_return": module_dir / "direct_return_experiment.py",
        "offline_race": module_dir / "offline_race.py",
        "cash_normalization": module_dir / "structural_combo_holdout.py",
        "storage": module_dir / "storage.py",
    }
    return {
        "labels": [
            {"path": str(Path(path).resolve()), "sha256": artifact_hash(path)}
            for path in label_paths
        ],
        "screening_report": {
            "path": str(Path(screening_report_path).resolve()),
            "sha256": artifact_hash(screening_report_path),
        },
        "daily_prepared": {"path": str(daily_dir.resolve()), "sha256": artifact_hash(daily_dir)},
        "intraday_prepared": {
            "path": str(intraday_dir.resolve()), "sha256": artifact_hash(intraday_dir)
        },
        "prepared_provenance": _prepared_provenance(intraday_dir),
        "dependencies": {
            name: {"path": str(path.resolve()), "sha256": artifact_hash(path)}
            for name, path in dependencies.items()
        },
    }


def _validate_saved_inputs(inputs: dict) -> None:
    for evidence in inputs.get("labels", []):
        if artifact_hash(Path(evidence.get("path", ""))) != evidence.get("sha256"):
            raise RuntimeError("daily-minute historical label artifact changed")
    for name in ("screening_report", "daily_prepared", "intraday_prepared"):
        evidence = inputs.get(name, {})
        if artifact_hash(Path(evidence.get("path", ""))) != evidence.get("sha256"):
            raise RuntimeError(f"daily-minute {name} input changed")
    provenance = inputs.get("prepared_provenance", {})
    for name in ("prepare_state", "feature_manifest"):
        evidence = provenance.get(name, {})
        if artifact_hash(Path(evidence.get("path", ""))) != evidence.get("sha256"):
            raise RuntimeError(f"daily-minute prepared {name} changed")
    for name, evidence in inputs.get("dependencies", {}).items():
        if artifact_hash(Path(evidence.get("path", ""))) != evidence.get("sha256"):
            raise RuntimeError(f"daily-minute dependency {name} changed")


def validate_state(state: dict, verify_inputs: bool = True) -> None:
    content = dict(state)
    state_hash = content.pop("state_hash", None)
    if state_hash != _canonical_hash(content):
        raise RuntimeError("daily-minute state was modified")
    if state.get("protocol") != PROTOCOL or state.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("daily-minute protocol changed")
    if state.get("production_publication") is not False or state.get("eligible_for_production") is not False:
        raise RuntimeError("daily-minute research isolation cannot be disabled")
    if state.get("untouched_holdout") is not False or state.get("human_approval_required") is not True:
        raise RuntimeError("daily-minute historical evidence status changed")
    for name in ("report", "execution_records", "daily_returns"):
        evidence = state.get(name, {})
        path = Path(evidence.get("path", ""))
        if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"daily-minute {name} artifact changed")
    if verify_inputs:
        _validate_saved_inputs(state.get("input_hashes", {}))


def _fold_candidate_paths(checkpoint_dir: Path, fold_name: str, candidate: str) -> tuple[Path, Path]:
    key = f"{fold_name}__{candidate}"
    return checkpoint_dir / f"{key}.parquet", checkpoint_dir / f"{key}.json"


def _run_enhancement_race_unlocked(
    label_paths: list[Path],
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    only_fold: str | None = None,
    only_candidate: str | None = None,
    gate_only: bool = False,
    max_train_rows: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    state_dir = Path(state_dir)
    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    _validate_research_paths(output_dir, state_dir, intraday_dir)
    state_path = state_dir / "manifest.json"
    inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_state(state, verify_inputs=False)
        if state.get("input_hashes") != inputs:
            raise RuntimeError("daily-minute invocation does not match the frozen inputs")
        return state
    labels = combine_historical_labels(label_paths)
    dates = validate_historical_dates(labels, load_trading_calendar(intraday_dir))
    folds = registered_folds(dates)
    if only_fold is not None:
        folds = [fold for fold in folds if fold["name"] == only_fold]
        if not folds:
            raise ValueError(f"unknown daily-minute fold: {only_fold}")
    feature_window, base_features, minute_features, screening_evidence = _causal_screened_features_cached(
        output_dir / "screening_cache.json",
        screening_report_path,
        daily_dir,
        intraday_dir,
        inputs,
        max_rows=(int(max_train_rows) if gate_only and max_train_rows is not None else None),
    )
    grid = candidate_features(base_features, minute_features)
    if only_candidate is not None:
        if only_candidate not in grid:
            raise ValueError(f"unknown daily-minute candidate: {only_candidate}")
        grid = {only_candidate: grid[only_candidate]}
    if gate_only and (only_fold is None or only_candidate is None):
        raise ValueError("gate-only mode requires --only-fold and --only-candidate")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_record_paths = []
    fold_reports = []
    universe_hashes_by_fold = {}
    prepared_vs_label_key_counts_by_fold = {}
    for fold in folds:
        train_end = pd.Timestamp(fold["train"][-1])
        oos_start = pd.Timestamp(fold["oos"][0])
        oos_end = pd.Timestamp(fold["oos"][-1])
        # Keep only execution labels/state resident for the fold. Candidate
        # feature panels are loaded one at a time below, preventing the full
        # feature panel and a model matrix from overlapping in memory.
        execution_labels = labels[labels["date"].isin(fold["oos"])].copy()
        execution_panel, universe_hashes = _build_panel(
            execution_labels,
            daily_dir,
            intraday_dir,
            base_features,
            minute_features,
        )
        universe_hashes_by_fold[fold["name"]] = universe_hashes
        prepared_vs_label_key_counts_by_fold[fold["name"]] = execution_panel.attrs.get(
            "prepared_vs_label_key_counts", {}
        )
        fold_panel = execution_panel[
            ~execution_panel["date"].isin(fold["purge"])
        ].copy()
        del execution_labels, execution_panel
        gc.collect()
        models = {}
        for name, features in grid.items():
            records_path, metrics_path = _fold_candidate_paths(
                checkpoint_dir, fold["name"], name
            )
            checkpoint_key = {
                "protocol": PROTOCOL,
                "input_hashes": inputs,
                "fold": fold["name"],
                "candidate": name,
                "features": features,
            }
            if records_path.is_file() and metrics_path.is_file():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                if saved.get("checkpoint_key") != checkpoint_key:
                    raise RuntimeError(
                        f"daily-minute checkpoint does not match frozen inputs: {records_path}"
                    )
                records = pd.read_parquet(records_path)
                if artifact_hash(records_path) != saved.get("records_sha256"):
                    raise RuntimeError(f"daily-minute checkpoint artifact changed: {records_path}")
                metrics = saved["metrics"]
                print(
                    f"[daily-minute] reuse fold={fold['name']} candidate={name}",
                    flush=True,
                )
            else:
                print(
                    f"[daily-minute] fold={fold['name']} candidate={name} "
                    f"features={len(features)} train_end={train_end.date()} "
                    f"oos={oos_start.date()}..{oos_end.date()}",
                    flush=True,
                )
                # Load the same frozen selection for every candidate so the
                # prepared join, universe and schema stay identical, then keep
                # only the candidate's own features in the model panel.
                candidate_source_labels = labels[labels["date"] <= oos_end].copy()
                candidate_labels = _cap_training_panel(
                    candidate_source_labels,
                    train_end,
                    max_train_rows=int(MODEL_RECIPE["max_train_rows"]),
                )
                del candidate_source_labels
                candidate_panel, _ = _build_panel(
                    candidate_labels,
                    daily_dir,
                    intraday_dir,
                    base_features,
                    minute_features,
                )
                del candidate_labels
                candidate_panel = candidate_panel[
                    ~candidate_panel["date"].isin(fold["purge"])
                ]
                model_panel = _project_model_panel(candidate_panel, features)
                del candidate_panel
                predictions = _fit_daily_head(
                    model_panel, features, train_end, oos_start, oos_end,
                    model_threads, name, fold["name"],
                    checkpoint_dir / "model_workers" / fold["name"] / name,
                    max_train_rows,
                )
                del model_panel
                gc.collect()
                records, metrics = _evaluate_candidate(
                    name, predictions, fold_panel, fold["oos"]
                )
                records["fold"] = fold["name"]
                atomic_parquet(records, records_path)
                atomic_json(
                    {
                        "checkpoint_key": checkpoint_key,
                        "records_sha256": artifact_hash(records_path),
                        "metrics": metrics,
                    },
                    metrics_path,
                )
                del predictions
            checkpoint_record_paths.append(records_path)
            models[name] = metrics
            del records, metrics
            gc.collect()
        fold_reports.append({
            "name": fold["name"],
            "train_end": str(train_end.date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge"]],
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "oos_days": len(fold["oos"]),
            "models": models,
        })
        del fold_panel, models
        gc.collect()
    if gate_only:
        gate_report = {
            "protocol": PROTOCOL,
            "gate_only": True,
            "folds": [fold["name"] for fold in folds],
            "candidates": list(grid),
            "input_hashes": inputs,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "checkpoint_count": len(checkpoint_record_paths),
            "max_train_rows": int(max_train_rows) if max_train_rows is not None else int(MODEL_RECIPE["max_train_rows"]),
            "production_candidate": False,
            "human_approval_required": True,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(gate_report, output_dir / "gate_report.json")
        return gate_report
    records = pd.concat(
        [pd.read_parquet(path) for path in checkpoint_record_paths],
        ignore_index=True,
    )
    del checkpoint_record_paths
    gc.collect()
    oos_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(value) for fold in folds for value in fold["oos"]})
    )
    account = cash_normalized_execution_records(
        records, oos_dates, top_n=TOP_N, models=list(grid)
    )
    comparison = compare_execution_records(account)
    daily_returns = comparison["daily_returns"]
    if len(folds) != len(FOLD_POSITIONS) or len(oos_dates) != sum(
        end - start + 1 for start, end in (item["oos"] for item in FOLD_POSITIONS)
    ):
        raise RuntimeError("daily-minute walk-forward coverage differs from the registered folds")
    actual_oos_dates = pd.DatetimeIndex(daily_returns["signal_date"].unique()).sort_values()
    if not actual_oos_dates.equals(oos_dates):
        raise RuntimeError("daily-minute comparison does not cover the exact OOS date index")
    for name in grid:
        if name not in daily_returns or daily_returns[name].isna().any():
            raise RuntimeError(f"daily-minute candidate {name} has missing OOS cohort returns")
    decision = select_enhancement(comparison["models"], fold_reports)
    final_inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    if final_inputs != inputs:
        raise RuntimeError("daily-minute inputs changed during evaluation")
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "protocol_payload": protocol_payload(),
        "historical_backfill": True,
        "untouched_holdout": False,
        "production_publication": False,
        "input_hashes": inputs,
        "prepared_provenance": inputs["prepared_provenance"],
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "causal_feature_screening": screening_evidence,
        "feature_grid": grid,
        "eligible_universe_hashes_by_fold": universe_hashes_by_fold,
        "prepared_vs_label_key_counts_by_fold": prepared_vs_label_key_counts_by_fold,
        "universe_policy": (
            "adaptive labels must be a subset of raw intraday signal eligibility; keys without "
            "a matched daily prepared row are excluded uniformly; every candidate uses the same "
            "final daily-and-intraday labeled intersection without refill"
        ),
        "historical_days": len(dates),
        "walk_forward_oos_days": len(oos_dates),
        "folds": fold_reports,
        "account_comparison": {
            "models": comparison["models"],
            "pairwise": comparison["pairwise"],
        },
        "decision": decision,
        "production_candidate": False,
        "human_approval_required": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "execution_records.parquet"
    daily_path = output_dir / "daily_returns.parquet"
    report_path = output_dir / "report.json"
    atomic_parquet(records, records_path)
    atomic_parquet(daily_returns, daily_path)
    atomic_json(report, report_path)
    state = {
        "protocol": PROTOCOL,
        "protocol_hash": report["protocol_hash"],
        "status": decision["status"],
        "selected": decision["selected"],
        "next_branch": decision["next_branch"],
        "selection_basis": "historical_reused_walk_forward",
        "untouched_holdout": False,
        "eligible_for_production": False,
        "input_hashes": inputs,
        "prepared_provenance": inputs["prepared_provenance"],
        "report": {"path": str(report_path.resolve()), "sha256": artifact_hash(report_path)},
        "execution_records": {
            "path": str(records_path.resolve()), "sha256": artifact_hash(records_path)
        },
        "daily_returns": {"path": str(daily_path.resolve()), "sha256": artifact_hash(daily_path)},
        "human_approval_required": True,
        "production_publication": False,
    }
    state["state_hash"] = _canonical_hash(state)
    atomic_json(state, state_path)
    return state


def run_enhancement_race(
    label_paths: list[Path],
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    only_fold: str | None = None,
    only_candidate: str | None = None,
    gate_only: bool = False,
    max_train_rows: int | None = None,
) -> dict:
    state_dir = Path(state_dir)
    with _cycle_lock(state_dir):
        return _run_enhancement_race_unlocked(
            label_paths,
            screening_report_path,
            output_dir,
            state_dir,
            daily_dir,
            intraday_dir,
            model_threads,
            only_fold,
            only_candidate,
            gate_only,
            max_train_rows,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated daily h1 plus minute-feature walk-forward race")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--screening-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path)
    parser.add_argument("--intraday-dir", type=Path, default=config.PREPARED_DIR)
    parser.add_argument("--model-threads", type=int, default=8)
    parser.add_argument("--only-fold", choices=[item["name"] for item in FOLD_POSITIONS])
    parser.add_argument("--only-candidate")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    args = parser.parse_args()
    result = run_enhancement_race(
        args.labels, args.screening_report, args.output_dir, args.state_dir,
        args.daily_dir, args.intraday_dir, args.model_threads,
        args.only_fold, args.only_candidate, args.gate_only,
        args.max_train_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
