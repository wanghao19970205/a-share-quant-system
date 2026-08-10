from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    _causal_eligible_predictions,
    default_daily_prepared_dir,
    load_joined_prepared,
    screen_window_features,
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


PROTOCOL = "intraday_1400_daily_minute_enhancement_v1"
BASELINE = "daily_asof_baseline"
ALL_MINUTE = "daily_plus_all_minute"
MINUTE_FAMILIES = ("speed", "path", "volume_vwap", "risk", "dependence", "context")
MODEL_RECIPE = {
    "target": "daily_target_ret_1d",
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
) -> tuple[dict, list[str], list[str], dict]:
    source_window, _, _, _ = _selected_features(screening_report_path)
    train_end = pd.Timestamp(source_window["train_end"])
    screening_panel, groups = load_joined_prepared(
        daily_dir,
        intraday_dir,
        pd.Timestamp("2018-01-01"),
        train_end,
    )
    selected = screen_window_features(screening_panel, groups, train_end)
    control = selected["daily_asof_plus_minute_control"]
    base_features = list(control["asof_matched"])
    minute_features = list(control["minute"])
    if not base_features or not minute_features:
        raise RuntimeError("causal daily-minute screening selected an empty feature group")
    manifest_path = Path(prepared_provenance["feature_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_features = set(manifest.get("features", []))
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
        "source_report_window_metadata_hash": _canonical_hash({
            key: source_window.get(key)
            for key in ("window", "train_end", "purge_date", "valid_start", "valid_end")
        }),
    }
    return source_window, base_features, minute_features, evidence


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


def _fit_daily_head(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    candidate: str,
    fold: str,
) -> pd.DataFrame:
    common = {
        "panel": panel,
        "features": features,
        "horizon": 1,
        "train_end": str(train_end.date()),
        "valid_end": str(oos_end.date()),
        "predict_start": str(oos_start.date()),
        "predict_end": str(oos_end.date()),
        "decay_half_life_days": float(MODEL_RECIPE["decay_half_life_days"]),
        "min_weight": float(MODEL_RECIPE["min_weight"]),
        "label_col": "daily_target_ret_1d",
        "train_mask_col": None,
    }
    ridge_recipe = MODEL_RECIPE["ridge"]
    ranker_recipe = MODEL_RECIPE["lightgbm_ranker"]
    ridge = model.train_ridge(**common, alpha=float(ridge_recipe["alpha"]))
    ranker = model.train_lightgbm_ranker(
        **common,
        n_estimators=int(ranker_recipe["n_estimators"]),
        learning_rate=float(ranker_recipe["learning_rate"]),
        early_stopping_rounds=int(ranker_recipe["early_stopping_rounds"]),
        n_jobs=threads,
        rank_bins=int(ranker_recipe["rank_bins"]),
        eval_at=tuple(ranker_recipe["eval_at"]),
    )
    for name, result in (("ridge", ridge), ("lightgbm_ranker", ranker)):
        if not result.ok:
            raise RuntimeError(f"{candidate} {fold} {name} failed: {result.message}")
    merged = ridge.predictions[["code", "date", "pred"]].rename(
        columns={"pred": "ridge_pred"}
    ).merge(
        ranker.predictions[["code", "date", "pred"]].rename(columns={"pred": "ranker_pred"}),
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
        ExecutionConfig(top_n=TOP_N, **EXECUTION_CONFIG),
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


def _run_enhancement_race_unlocked(
    label_paths: list[Path],
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
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
    feature_window, base_features, minute_features, screening_evidence = _causal_screened_features(
        screening_report_path,
        daily_dir,
        intraday_dir,
        inputs["prepared_provenance"],
    )
    grid = candidate_features(base_features, minute_features)
    panel, universe_hashes = _build_panel(
        labels, daily_dir, intraday_dir, base_features, minute_features
    )
    all_records = []
    fold_reports = []
    for fold in folds:
        fold_panel = panel[~panel["date"].isin(fold["purge"])].copy()
        train_end = pd.Timestamp(fold["train"][-1])
        oos_start = pd.Timestamp(fold["oos"][0])
        oos_end = pd.Timestamp(fold["oos"][-1])
        models = {}
        for name, features in grid.items():
            print(
                f"[daily-minute] fold={fold['name']} candidate={name} "
                f"features={len(features)} train_end={train_end.date()} "
                f"oos={oos_start.date()}..{oos_end.date()}",
                flush=True,
            )
            predictions = _fit_daily_head(
                fold_panel, features, train_end, oos_start, oos_end,
                model_threads, name, fold["name"],
            )
            records, metrics = _evaluate_candidate(name, predictions, fold_panel, fold["oos"])
            records["fold"] = fold["name"]
            all_records.append(records)
            models[name] = metrics
        fold_reports.append({
            "name": fold["name"],
            "train_end": str(train_end.date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge"]],
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "oos_days": len(fold["oos"]),
            "models": models,
        })
    records = pd.concat(all_records, ignore_index=True)
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
        "historical_backfill": True,
        "untouched_holdout": False,
        "production_publication": False,
        "input_hashes": inputs,
        "prepared_provenance": inputs["prepared_provenance"],
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "causal_feature_screening": screening_evidence,
        "feature_grid": grid,
        "eligible_universe_hashes": universe_hashes,
        "prepared_vs_label_key_counts": panel.attrs.get("prepared_vs_label_key_counts", {}),
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
    args = parser.parse_args()
    result = run_enhancement_race(
        args.labels, args.screening_report, args.output_dir, args.state_dir,
        args.daily_dir, args.intraday_dir, args.model_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
