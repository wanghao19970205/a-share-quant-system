from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_1400.daily_minute_enhancement import (
    BASELINE,
    MODEL_RECIPE,
    _build_panel,
    _canonical_hash,
    _causal_screened_features,
    _cycle_lock,
    _evaluate_candidate,
    _fit_daily_head,
    _input_hashes as parent_input_hashes,
    _prepared_provenance,
    _validate_research_paths,
    validate_state as validate_parent_state,
)
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.offline_race import compare_execution_records
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records
from intraday_1400.target_redesign_backfill import (
    FOLD_POSITIONS,
    TOP_N,
    combine_historical_labels,
    registered_folds,
    validate_historical_dates,
)
from intraday_1400.fair_race_pipeline import default_daily_prepared_dir


PROTOCOL = "intraday_1400_minute_feature_residualization_v1"
PARENT_PROTOCOL = "intraday_1400_daily_minute_enhancement_v1"
RESIDUAL_CANDIDATES = (
    "daily_plus_resid_all_ols",
    "daily_plus_resid_all_ridge",
)
RESIDUALIZER_RECIPE = {
    "controls": "same_day_asof_daily_features",
    "fit_scope": "registered_fold_train_dates_only",
    "oos_transform": "frozen_train_parameters",
    "candidate_universe": "same_final_labeled_intersection_as_parent",
    "maximum_estimated_peak_bytes": 12 * 1024 ** 3,
    "representative_dry_run": {
        "panel_rows": 798743,
        "fold_rows": 789730,
        "residual_features": 40,
        "prediction_rows": 549469,
        "peak_rss_bytes": 10119732 * 1024,
        "measured_on": "remote_scheduler_container",
        "candidates_run_sequentially": True,
    },
    "candidates": {
        "daily_plus_resid_all_ols": {"alpha": 0.0},
        "daily_plus_resid_all_ridge": {"alpha": 10.0},
    },
}


def protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "fold_positions": FOLD_POSITIONS,
        "baseline": BASELINE,
        "candidate_grid": list(RESIDUAL_CANDIDATES),
        "residualizer_recipe": RESIDUALIZER_RECIPE,
        "model_recipe": MODEL_RECIPE,
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


def _fit_transform_residuals(
    panel: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    controls: list[str],
    minute_features: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    source_columns = [*controls, *minute_features]
    panel_bytes = int(panel.memory_usage(index=True, deep=True).sum())
    estimated_peak_bytes = int(
        panel_bytes * 2
        + len(panel) * len(source_columns) * np.dtype(np.float32).itemsize
        + len(panel) * (len(controls) + 1) * np.dtype(np.float64).itemsize
    )
    if estimated_peak_bytes > int(RESIDUALIZER_RECIPE["maximum_estimated_peak_bytes"]):
        raise RuntimeError(
            f"residualizer estimated peak memory {estimated_peak_bytes} exceeds protocol budget"
        )
    source = panel[source_columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    train_mask = panel["date"].isin(train_dates).to_numpy()
    x_train_raw = source.loc[train_mask, controls].to_numpy(dtype=np.float64)
    control_mean = np.nanmean(x_train_raw, axis=0)
    control_mean = np.where(np.isfinite(control_mean), control_mean, 0.0)
    control_scale = np.nanstd(x_train_raw, axis=0)
    control_scale = np.where(np.isfinite(control_scale) & (control_scale > 1e-8), control_scale, 1.0)

    def normalized_controls(values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, control_mean)
        return (filled - control_mean) / control_scale

    x_train = normalized_controls(x_train_raw)
    x_all = normalized_controls(source[controls].to_numpy(dtype=np.float64))
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_all = np.column_stack([np.ones(len(x_all)), x_all])
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    xtx = x_train.T @ x_train + penalty
    inverse = np.linalg.pinv(xtx, rcond=1e-10)
    output = panel.copy()
    residual_features = []
    coefficient_hashes = {}
    training_rows = {}
    for feature in minute_features:
        y_train = source.loc[train_mask, feature].to_numpy(dtype=np.float64)
        finite = np.isfinite(y_train)
        if int(finite.sum()) < x_train.shape[1] + 20:
            raise RuntimeError(f"insufficient residualizer training rows for {feature}")
        if finite.all():
            beta = inverse @ (x_train.T @ y_train)
        else:
            local_x = x_train[finite]
            local_penalty = np.eye(local_x.shape[1], dtype=np.float64) * float(alpha)
            local_penalty[0, 0] = 0.0
            beta = np.linalg.pinv(local_x.T @ local_x + local_penalty, rcond=1e-10) @ (
                local_x.T @ y_train[finite]
            )
        y_all = source[feature].to_numpy(dtype=np.float64)
        residual = y_all - x_all @ beta
        residual[~np.isfinite(y_all)] = np.nan
        name = f"minute_resid__{feature.removeprefix('minute__')}"
        output[name] = residual.astype(np.float32)
        residual_features.append(name)
        coefficient_hashes[feature] = hashlib.sha256(beta.tobytes()).hexdigest()
        training_rows[feature] = int(finite.sum())
    evidence = {
        "alpha": float(alpha),
        "estimated_peak_bytes": estimated_peak_bytes,
        "panel_rows": int(len(panel)),
        "controls": controls,
        "minute_features": minute_features,
        "coefficient_hashes": coefficient_hashes,
        "training_rows": training_rows,
        "control_mean_hash": hashlib.sha256(control_mean.tobytes()).hexdigest(),
        "control_scale_hash": hashlib.sha256(control_scale.tobytes()).hexdigest(),
    }
    return output, residual_features, evidence


def select_residual_enhancement(aggregate: dict[str, dict], folds: list[dict]) -> dict:
    expected = {BASELINE, *RESIDUAL_CANDIDATES}
    if set(aggregate) != expected:
        raise ValueError("residualization selection requires the exact registered candidate grid")
    if [item.get("name") for item in folds] != [item["name"] for item in FOLD_POSITIONS]:
        raise ValueError("residualization selection requires the ordered registered four folds")
    for fold in folds:
        if set(fold.get("models", {})) != expected:
            raise ValueError(f"residualization fold {fold.get('name')} has an incomplete candidate grid")
        for name, metrics in fold["models"].items():
            value = metrics.get("mean_return")
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"residualization fold {fold['name']} candidate {name} is non-finite")
    baseline = aggregate[BASELINE]
    eligible = []
    for name in RESIDUAL_CANDIDATES:
        metrics = aggregate[name]
        required = [metrics.get(key) for key in (
            "mean_return", "compound_return", "max_drawdown", "mean_filled_names"
        )]
        if not np.isfinite(np.asarray(required, dtype=float)).all():
            raise ValueError(f"residual candidate {name} has non-finite metrics")
        wins = sum(
            fold["models"][name]["mean_return"] > fold["models"][BASELINE]["mean_return"]
            for fold in folds
        )
        if (
            float(metrics["mean_filled_names"]) / TOP_N >= 0.60
            and float(metrics["mean_return"]) > float(baseline["mean_return"])
            and float(metrics["max_drawdown"]) >= float(baseline["max_drawdown"]) - 0.05
            and wins >= 3
        ):
            eligible.append((name, wins))
    if not eligible:
        return {
            "status": "no_residual_enhancement_passed",
            "selected": BASELINE,
            "next_branch": "daily_baseline_retained",
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
        "status": "residual_enhancement_selected",
        "selected": selected,
        "next_branch": "forward_shadow",
    }


def _input_hashes(
    label_paths: list[Path],
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
    parent_state_path: Path,
) -> dict:
    result = parent_input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    result["residualization_controller"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": artifact_hash(Path(__file__)),
    }
    result["parent_state"] = {
        "path": str(parent_state_path.resolve()),
        "sha256": artifact_hash(parent_state_path),
    }
    return result


def _parent_binding(parent_state: dict) -> dict:
    parent_report = json.loads(
        Path(parent_state["report"]["path"]).read_text(encoding="utf-8")
    )
    parent_universe = parent_report.get("eligible_universe_hashes", {})
    if not parent_universe:
        raise RuntimeError("parent daily-minute report has no frozen universe hashes")
    return {
        "protocol": parent_state["protocol"],
        "protocol_hash": parent_state["protocol_hash"],
        "state_hash": parent_state["state_hash"],
        "status": parent_state["status"],
        "selected": parent_state["selected"],
        "next_branch": parent_state["next_branch"],
        "input_hashes_hash": _canonical_hash(parent_state["input_hashes"]),
        "eligible_universe_hash": _canonical_hash(parent_universe),
        "final_universe_hash": parent_universe.get("final_all_keys"),
        "artifacts": {
            name: parent_state[name]
            for name in ("report", "execution_records", "daily_returns")
        },
    }


def _validate_state(state: dict) -> None:
    content = dict(state)
    state_hash = content.pop("state_hash", None)
    if state_hash != _canonical_hash(content):
        raise RuntimeError("residualization state was modified")
    if state.get("protocol") != PROTOCOL or state.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("residualization protocol changed")
    if state.get("production_publication") is not False or state.get("eligible_for_production") is not False:
        raise RuntimeError("residualization production isolation changed")
    if state.get("untouched_holdout") is not False or state.get("human_approval_required") is not True:
        raise RuntimeError("residualization historical evidence status changed")
    for name in ("report", "execution_records", "daily_returns"):
        evidence = state.get(name, {})
        path = Path(evidence.get("path", ""))
        if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"residualization {name} artifact changed")
    report = json.loads(Path(state["report"]["path"]).read_text(encoding="utf-8"))
    if (
        report.get("protocol") != PROTOCOL
        or report.get("protocol_hash") != state.get("protocol_hash")
        or report.get("input_hashes") != state.get("input_hashes")
        or report.get("parent_binding") != state.get("parent_binding")
        or report.get("decision", {}).get("status") != state.get("status")
        or report.get("decision", {}).get("selected") != state.get("selected")
        or report.get("decision", {}).get("next_branch") != state.get("next_branch")
        or report.get("untouched_holdout") is not False
        or report.get("human_approval_required") is not True
    ):
        raise RuntimeError("residualization report and state are inconsistent")


def _run_unlocked(
    label_paths: list[Path],
    screening_report_path: Path,
    parent_state_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path,
    intraday_dir: Path,
    model_threads: int,
) -> dict:
    _validate_research_paths(output_dir, state_dir, intraday_dir)
    parent_state = json.loads(parent_state_path.read_text(encoding="utf-8"))
    validate_parent_state(parent_state)
    if (
        parent_state.get("status") != "no_enhancement_passed"
        or parent_state.get("next_branch") != "minute_feature_residualization"
    ):
        raise RuntimeError("parent daily-minute state did not route to residualization")
    parent_binding = _parent_binding(parent_state)
    state_path = state_dir / "manifest.json"
    inputs = _input_hashes(
        label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path
    )
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_state(state)
        if state.get("input_hashes") != inputs:
            raise RuntimeError("residualization invocation differs from frozen inputs")
        return state
    labels = combine_historical_labels(label_paths)
    dates = validate_historical_dates(labels, load_trading_calendar(intraday_dir))
    folds = registered_folds(dates)
    provenance = _prepared_provenance(intraday_dir)
    feature_window, controls, minute_features, screening_evidence = _causal_screened_features(
        screening_report_path, daily_dir, intraday_dir, provenance
    )
    panel, universe_hashes = _build_panel(
        labels, daily_dir, intraday_dir, controls, minute_features
    )
    if (
        _canonical_hash(universe_hashes) != parent_binding["eligible_universe_hash"]
        or universe_hashes.get("final_all_keys") != parent_binding["final_universe_hash"]
    ):
        raise RuntimeError("residualization universe differs from the frozen parent race")
    all_records = []
    fold_reports = []
    for fold in folds:
        fold_panel = panel[~panel["date"].isin(fold["purge"])].copy()
        train_end = pd.Timestamp(fold["train"][-1])
        oos_start = pd.Timestamp(fold["oos"][0])
        oos_end = pd.Timestamp(fold["oos"][-1])
        models = {}
        residualizers = {}
        baseline_predictions = _fit_daily_head(
            fold_panel, controls, train_end, oos_start, oos_end,
            model_threads, BASELINE, fold["name"],
        )
        baseline_records, baseline_metrics = _evaluate_candidate(
            BASELINE, baseline_predictions, fold_panel, fold["oos"]
        )
        baseline_records["fold"] = fold["name"]
        all_records.append(baseline_records)
        models[BASELINE] = baseline_metrics
        for name in RESIDUAL_CANDIDATES:
            recipe = RESIDUALIZER_RECIPE["candidates"][name]
            transformed, residual_features, evidence = _fit_transform_residuals(
                fold_panel,
                fold["train"],
                controls,
                minute_features,
                float(recipe["alpha"]),
            )
            predictions = _fit_daily_head(
                transformed,
                [*controls, *residual_features],
                train_end,
                oos_start,
                oos_end,
                model_threads,
                name,
                fold["name"],
            )
            records, metrics = _evaluate_candidate(name, predictions, transformed, fold["oos"])
            records["fold"] = fold["name"]
            all_records.append(records)
            models[name] = metrics
            residualizers[name] = evidence
            del transformed
        fold_reports.append({
            "name": fold["name"],
            "train_end": str(train_end.date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge"]],
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "oos_days": int(len(fold["oos"])),
            "models": models,
            "residualizers": residualizers,
        })
    records = pd.concat(all_records, ignore_index=True)
    oos_dates = pd.DatetimeIndex(sorted({pd.Timestamp(value) for fold in folds for value in fold["oos"]}))
    account = cash_normalized_execution_records(
        records, oos_dates, top_n=TOP_N, models=[BASELINE, *RESIDUAL_CANDIDATES]
    )
    comparison = compare_execution_records(account)
    daily_returns = comparison["daily_returns"]
    if not pd.DatetimeIndex(daily_returns["signal_date"].unique()).sort_values().equals(oos_dates):
        raise RuntimeError("residualization comparison does not cover exact OOS dates")
    if len(folds) != len(FOLD_POSITIONS) or len(oos_dates) != sum(
        end - start + 1 for start, end in (item["oos"] for item in FOLD_POSITIONS)
    ):
        raise RuntimeError("residualization coverage differs from registered folds")
    for name in (BASELINE, *RESIDUAL_CANDIDATES):
        if name not in daily_returns or daily_returns[name].isna().any():
            raise RuntimeError(f"residual candidate {name} has missing OOS cohort returns")
    decision = select_residual_enhancement(comparison["models"], fold_reports)
    final_inputs = _input_hashes(
        label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path
    )
    if final_inputs != inputs:
        raise RuntimeError("residualization inputs changed during evaluation")
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "parent_state": inputs["parent_state"],
        "parent_binding": parent_binding,
        "historical_backfill": True,
        "untouched_holdout": False,
        "production_publication": False,
        "input_hashes": inputs,
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "causal_feature_screening": screening_evidence,
        "residualizer_recipe": RESIDUALIZER_RECIPE,
        "eligible_universe_hashes": universe_hashes,
        "prepared_vs_label_key_counts": panel.attrs.get("prepared_vs_label_key_counts", {}),
        "walk_forward_oos_days": int(len(oos_dates)),
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
        "parent_binding": parent_binding,
        "report": {"path": str(report_path.resolve()), "sha256": artifact_hash(report_path)},
        "execution_records": {"path": str(records_path.resolve()), "sha256": artifact_hash(records_path)},
        "daily_returns": {"path": str(daily_path.resolve()), "sha256": artifact_hash(daily_path)},
        "human_approval_required": True,
        "production_publication": False,
    }
    state["state_hash"] = _canonical_hash(state)
    atomic_json(state, state_path)
    return state


def run_residualization_race(
    label_paths: list[Path],
    screening_report_path: Path,
    parent_state_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
) -> dict:
    from intraday_1400 import config

    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    state_dir = Path(state_dir)
    with _cycle_lock(state_dir):
        return _run_unlocked(
            label_paths,
            Path(screening_report_path),
            Path(parent_state_path),
            Path(output_dir),
            state_dir,
            daily_dir,
            intraday_dir,
            model_threads,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-fold-only minute feature residualization race")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--screening-report", type=Path, required=True)
    parser.add_argument("--parent-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path)
    parser.add_argument("--intraday-dir", type=Path)
    parser.add_argument("--model-threads", type=int, default=8)
    args = parser.parse_args()
    result = run_residualization_race(
        args.labels,
        args.screening_report,
        args.parent_state,
        args.output_dir,
        args.state_dir,
        args.daily_dir,
        args.intraday_dir,
        args.model_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
