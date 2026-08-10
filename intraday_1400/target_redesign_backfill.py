from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.direct_return_experiment import (
    EXECUTION_CONFIG,
    FEATURE_SCREENING_TRAIN_END,
    _execution_labels,
    _selected_features,
    _simulate_adaptive_label_race,
)
from intraday_1400.fair_race_pipeline import (
    _causal_eligible_predictions,
    default_daily_prepared_dir,
)
from intraday_1400.offline_race import ExecutionConfig, compare_execution_records
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records
from intraday_1400.target_redesign import build_target_columns, conditional_payoff_scores
from quant import model


PROTOCOL = "intraday_1400_target_redesign_backfill_v2"
HISTORICAL_START = pd.Timestamp("2025-07-01")
HISTORICAL_END = pd.Timestamp("2026-08-03")
FIRST_PROSPECTIVE_SIGNAL_DATE = pd.Timestamp("2026-08-10")
TOTAL_DATES = 266
TOP_N = 10
FAMILY_ORDER = ("downside_quantile", "cross_sectional_rank", "conditional_payoff")
FOLD_POSITIONS = (
    {"name": "wf1", "train": [0, 79], "purge": [80, 82], "oos": [83, 122]},
    {"name": "wf2", "train": [0, 122], "purge": [123, 125], "oos": [126, 165]},
    {"name": "wf3", "train": [0, 165], "purge": [166, 168], "oos": [169, 215]},
    {"name": "wf4", "train": [0, 215], "purge": [216, 218], "oos": [219, 265]},
)
MODEL_RECIPE = {
    "downside_quantile": {
        "learner": "lightgbm_quantile",
        "target": "target_downside_source",
        "alpha": 0.20,
        "n_estimators": 200,
        "learning_rate": 0.015,
        "early_stopping_rounds": 0,
    },
    "cross_sectional_rank": {
        "learner": "lightgbm_lambdarank",
        "target": "target_cross_sectional_rank",
        "rank_bins": 5,
        "eval_at": [10],
        "n_estimators": 200,
        "learning_rate": 0.015,
        "early_stopping_rounds": 0,
    },
    "conditional_payoff": {
        "learners": {
            "entry": "lightgbm_classifier",
            "exit_given_entry": "lightgbm_classifier_or_empirical_constant",
            "conditional_return": "lightgbm_regression",
        },
        "stress_return": -0.10,
        "conditional_return_clip": [-0.10, 0.10],
        "n_estimators": 200,
        "learning_rate": 0.015,
        "minority_weight": 10.0,
        "max_train_rows": 150_000,
    },
    "common": {
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
        "model_threads": "runtime",
        "top_n": TOP_N,
        "roundtrip_cost": 0.002,
        "feature_screening_train_end": str(FEATURE_SCREENING_TRAIN_END.date()),
    },
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "historical_start": str(HISTORICAL_START.date()),
        "historical_end": str(HISTORICAL_END.date()),
        "first_prospective_signal_date": str(FIRST_PROSPECTIVE_SIGNAL_DATE.date()),
        "total_dates": TOTAL_DATES,
        "fold_positions": FOLD_POSITIONS,
        "families": FAMILY_ORDER,
        "model_recipe": MODEL_RECIPE,
        "historical_intervals_are_untouched": False,
        "historical_results_may_select_family": True,
        "historical_results_may_promote_production": False,
        "production_requires_append_only_forward": True,
        "production_publication": False,
        "human_approval_required": True,
    }


def _normalize_label_artifact(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    required = {
        "code", "date", "adaptive_entry_buyable", "adaptive_liquidated_by_t3",
        "adaptive_realized_net_ret_t3", "adaptive_stress_net_ret_t3",
        "adaptive_horizon_observed_t3",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"historical labels missing {sorted(missing)} in {path}")
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str[:6]
    data = data.dropna(subset=["date", "code"])
    if data.duplicated(["date", "code"]).any():
        raise ValueError(f"duplicate date+code keys inside {path}")
    if not data["adaptive_horizon_observed_t3"].fillna(False).astype(bool).all():
        raise ValueError(f"historical labels are not fully mature in {path}")
    if data["adaptive_stress_net_ret_t3"].isna().any():
        raise ValueError(f"historical stress labels are incomplete in {path}")
    data["source_artifact"] = str(Path(path).resolve())
    return data


def combine_historical_labels(paths: list[Path]) -> pd.DataFrame:
    if len(paths) < 2:
        raise ValueError("backfill requires multiple independent historical label artifacts")
    parts = [_normalize_label_artifact(Path(path)) for path in paths]
    for order, part in enumerate(parts):
        part["source_order"] = int(order)
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["date", "code", "source_order"]
    )
    compare_columns = [
        "adaptive_entry_buyable", "adaptive_liquidated_by_t3",
        "adaptive_realized_net_ret_t3", "adaptive_stress_net_ret_t3",
        "adaptive_horizon_observed_t3",
    ]
    duplicated = combined[combined.duplicated(["date", "code"], keep=False)]
    conflict_keys = set()
    for column in compare_columns:
        mismatch = duplicated.groupby(["date", "code"])[column].nunique(dropna=False) > 1
        conflict_keys.update(mismatch[mismatch].index.tolist())
    combined = combined.drop_duplicates(["date", "code"], keep="last")
    combined = combined[combined["date"] <= HISTORICAL_END]
    dates = pd.DatetimeIndex(combined["date"].unique()).sort_values()
    if len(dates) != TOTAL_DATES:
        raise ValueError(f"backfill requires exactly {TOTAL_DATES} unique historical dates, found {len(dates)}")
    result = combined.drop(columns=["source_artifact", "source_order"]).reset_index(drop=True)
    result.attrs["overlap_conflict_count"] = int(len(conflict_keys))
    result.attrs["source_precedence"] = [str(Path(path).resolve()) for path in paths]
    return result


def validate_historical_dates(
    labels: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(labels["date"].unique()).normalize().sort_values()
    calendar = pd.DatetimeIndex(trading_calendar).normalize().drop_duplicates().sort_values()
    expected = calendar[(calendar >= HISTORICAL_START) & (calendar <= HISTORICAL_END)]
    if len(expected) != TOTAL_DATES:
        raise ValueError(
            f"prepared calendar has {len(expected)} sessions for the frozen historical interval, "
            f"expected {TOTAL_DATES}"
        )
    if not dates.equals(expected):
        raise ValueError("historical labels do not match the exact frozen prepared trading calendar")
    if dates[0] != HISTORICAL_START or dates[-1] != HISTORICAL_END:
        raise ValueError("historical label boundaries changed")
    return dates


def registered_folds(dates: pd.DatetimeIndex) -> list[dict]:
    ordered = pd.DatetimeIndex(dates).normalize().drop_duplicates().sort_values()
    if len(ordered) != TOTAL_DATES:
        raise ValueError(f"registered folds require exactly {TOTAL_DATES} dates")
    folds = []
    for recipe in FOLD_POSITIONS:
        train = ordered[recipe["train"][0]:recipe["train"][1] + 1]
        purge = ordered[recipe["purge"][0]:recipe["purge"][1] + 1]
        oos = ordered[recipe["oos"][0]:recipe["oos"][1] + 1]
        if len(purge) != 3 or train[-1] >= purge[0] or purge[-1] >= oos[0]:
            raise AssertionError("invalid purged walk-forward fold")
        folds.append({"name": recipe["name"], "train": train, "purge": purge, "oos": oos})
    return folds


def _merge_prepared_panel(
    labels: pd.DataFrame,
    daily_dir: Path,
    intraday_dir: Path,
    base_features: list[str],
    minute_features: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    from intraday_1400.fair_race_pipeline import load_joined_prepared

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
    eligible_panel = panel[panel["signal_eligible"].fillna(False).astype(bool)].copy()
    prepared_keys = eligible_panel[["date", "code"]].drop_duplicates(["date", "code"])
    label_keys = labels[["date", "code"]].drop_duplicates(["date", "code"])
    key_check = prepared_keys.merge(label_keys, on=["date", "code"], how="outer", indicator=True)
    counts = key_check["_merge"].value_counts().to_dict()
    if (key_check["_merge"] == "left_only").any():
        raise ValueError(
            "historical labels contain keys outside the prepared eligible universe: "
            f"{counts}"
        )
    labels = labels.merge(label_keys, on=["code", "date"], how="inner", validate="one_to_one")
    targets = build_target_columns(labels)
    panel = eligible_panel.merge(targets, on=["code", "date"], how="inner", validate="one_to_one")
    universe_hashes = {
        str(pd.Timestamp(date).date()): _canonical_hash(sorted(group["code"].astype(str)))
        for date, group in label_keys.sort_values(["date", "code"]).groupby("date")
    }
    panel.attrs["prepared_vs_label_key_counts"] = counts
    return panel, [*base_features, *minute_features], universe_hashes


def _require_ok(result: model.TrainResult, family: str, fold: str) -> pd.DataFrame:
    if not result.ok:
        raise RuntimeError(f"{family} failed in {fold}: {result.message}")
    return result.predictions[["code", "date", "pred"]].copy()


def _fit_downside(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    fold: str,
) -> pd.DataFrame:
    result = model.train_lightgbm(
        panel, features, horizon=1,
        train_end=str(train_end.date()), valid_end=str(oos_end.date()),
        predict_start=str(oos_start.date()), decay_half_life_days=60.0, min_weight=0.03,
        n_estimators=200, learning_rate=0.015, early_stopping_rounds=0,
        n_jobs=threads, label_col="target_downside_source", objective="quantile", alpha=0.20,
    )
    predictions = _require_ok(result, "downside_quantile", fold)
    predictions["score"] = predictions["pred"]
    return predictions[["code", "date", "score"]]


def _fit_rank(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    fold: str,
) -> pd.DataFrame:
    result = model.train_lightgbm_ranker(
        panel, features, horizon=1,
        train_end=str(train_end.date()), valid_end=str(oos_end.date()),
        predict_start=str(oos_start.date()), decay_half_life_days=60.0, min_weight=0.03,
        n_estimators=200, learning_rate=0.015, early_stopping_rounds=0,
        n_jobs=threads, rank_bins=5, eval_at=(10,), label_col="target_cross_sectional_rank",
    )
    predictions = _require_ok(result, "cross_sectional_rank", fold)
    predictions["score"] = predictions["pred"]
    return predictions[["code", "date", "score"]]


def _fit_probability(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    name: str,
    fold: str,
) -> pd.DataFrame:
    train_target = pd.to_numeric(
        panel.loc[panel["date"] <= train_end, target], errors="coerce"
    ).dropna()
    keys = panel.loc[
        (panel["date"] >= oos_start) & (panel["date"] <= oos_end), ["code", "date"]
    ].drop_duplicates(["code", "date"])
    if train_target.nunique() < 2:
        prior = float(train_target.mean()) if len(train_target) else 0.0
        result = keys.copy()
        result[name] = prior
        return result
    fitted = model.train_binary_classifier(
        panel, features, target, "lightgbm",
        train_end=str(train_end.date()), valid_end=str(oos_end.date()),
        predict_start=str(oos_start.date()), decay_half_life_days=60.0, min_weight=0.03,
        minority_weight=10.0, n_estimators=200, learning_rate=0.015,
        max_train_rows=150_000, n_jobs=threads,
    )
    predictions = _require_ok(fitted, f"conditional_payoff:{name}", fold)
    return predictions.rename(columns={"pred": name})


def _fit_conditional(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    threads: int,
    fold: str,
) -> pd.DataFrame:
    entry = _fit_probability(
        panel, features, "target_entry_buyable", train_end, oos_start, oos_end,
        threads, "entry_probability", fold,
    )
    exit_probability = _fit_probability(
        panel, features, "target_exit_t3_given_entry", train_end, oos_start, oos_end,
        threads, "exit_probability", fold,
    )
    return_result = model.train_lightgbm(
        panel, features, horizon=1,
        train_end=str(train_end.date()), valid_end=str(oos_end.date()),
        predict_start=str(oos_start.date()), decay_half_life_days=60.0, min_weight=0.03,
        n_estimators=200, learning_rate=0.015, early_stopping_rounds=0,
        n_jobs=threads, label_col="target_conditional_return",
    )
    conditional_return = _require_ok(return_result, "conditional_payoff:return", fold).rename(
        columns={"pred": "conditional_return"}
    )
    merged = entry.merge(exit_probability, on=["code", "date"], validate="one_to_one")
    merged = merged.merge(conditional_return, on=["code", "date"], validate="one_to_one")
    merged["score"] = conditional_payoff_scores(
        merged["entry_probability"], merged["exit_probability"], merged["conditional_return"]
    )
    return merged[["code", "date", "score"]]


def _evaluate(
    family: str,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    filtered = _causal_eligible_predictions(
        {family: predictions}, panel, dates[0], dates[-1]
    )[family]
    records = _simulate_adaptive_label_race(
        {family: filtered},
        _execution_labels(panel, dates),
        ExecutionConfig(top_n=TOP_N, **EXECUTION_CONFIG),
    )
    account = cash_normalized_execution_records(records, dates, top_n=TOP_N, models=[family])
    comparison = compare_execution_records(account)
    return records, comparison["models"][family]


def select_family(comparison: dict) -> str:
    if set(comparison) != set(FAMILY_ORDER):
        raise ValueError("selection requires all three registered target families")
    if any(int(comparison[name].get("days", 0)) != 174 for name in FAMILY_ORDER):
        raise ValueError("each target family requires exactly 174 walk-forward OOS days")
    required = ("mean_return", "compound_return", "max_drawdown", "mean_filled_names")
    eligible = []
    for name in FAMILY_ORDER:
        values = [float(comparison[name].get(metric, np.nan)) for metric in required]
        if not np.isfinite(values).all():
            raise ValueError(f"target family {name} has non-finite selection metrics")
        if values[3] / float(TOP_N) >= 0.60:
            eligible.append(name)
    if not eligible:
        raise ValueError("no target family meets the fixed 60% fill-rate gate")

    def rank(name: str) -> tuple:
        metrics = comparison[name]
        return (
            float(metrics["mean_return"]),
            float(metrics["compound_return"]),
            float(metrics["max_drawdown"]),
            -FAMILY_ORDER.index(name),
        )

    return max(eligible, key=rank)


def _environment_versions() -> dict:
    packages = {}
    for name in ("pandas", "numpy", "lightgbm", "pyarrow", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {"python": platform.python_version(), "packages": packages}


def _input_hashes(
    label_paths: list[Path],
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
) -> dict:
    module_dir = Path(__file__).parent
    dependency_paths = {
        "controller_code": Path(__file__),
        "model_code": module_dir.parent / "quant" / "model.py",
        "target_code": module_dir / "target_redesign.py",
        "direct_return_code": module_dir / "direct_return_experiment.py",
        "fair_race_code": module_dir / "fair_race_pipeline.py",
        "offline_race_code": module_dir / "offline_race.py",
        "cash_normalization_code": module_dir / "structural_combo_holdout.py",
        "storage_code": module_dir / "storage.py",
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
        "intraday_prepared": {"path": str(intraday_dir.resolve()), "sha256": artifact_hash(intraday_dir)},
        "dependencies": {
            name: {"path": str(path.resolve()), "sha256": artifact_hash(path)}
            for name, path in dependency_paths.items()
        },
        "environment": _environment_versions(),
    }


@contextmanager
def _cycle_lock(state_dir: Path) -> Iterator[None]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".cycle.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another target-redesign backfill cycle is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_frozen_family(state: dict) -> None:
    content = dict(state)
    freeze_hash = content.pop("freeze_hash", None)
    if freeze_hash != _canonical_hash(content):
        raise RuntimeError("target-redesign backfill frozen state was modified")
    if state.get("protocol") != PROTOCOL or state.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("target-redesign backfill protocol changed")
    if state.get("selected_family") not in FAMILY_ORDER:
        raise RuntimeError("frozen target family is outside the registered grid")
    if state.get("selection_basis") != "historical_reused_walk_forward":
        raise RuntimeError("frozen target family selection basis changed")
    if state.get("untouched_holdout") is not False or state.get("eligible_for_production") is not False:
        raise RuntimeError("historical backfill cannot be treated as production evidence")
    if state.get("production_publication") is not False or state.get("human_approval_required") is not True:
        raise RuntimeError("production isolation cannot be disabled")
    artifacts = state.get("artifacts", {})
    if set(artifacts) != {"report", "execution_records", "cohort_daily_returns"}:
        raise RuntimeError("frozen target-redesign artifact ledger is incomplete")
    for name, evidence in artifacts.items():
        path = Path(evidence.get("path", ""))
        if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"frozen target-redesign {name} artifact changed")
    report = json.loads(Path(artifacts["report"]["path"]).read_text(encoding="utf-8"))
    if (
        report.get("historical_backfill") is not True
        or report.get("untouched_holdout") is not False
        or report.get("historical_results_may_promote_production") is not False
        or report.get("selected_family") != state.get("selected_family")
    ):
        raise RuntimeError("frozen target-redesign report status is invalid")
    immutable_inputs = state.get("input_hashes", {})
    for evidence in immutable_inputs.get("labels", []):
        if artifact_hash(Path(evidence["path"])) != evidence["sha256"]:
            raise RuntimeError("frozen historical label artifact changed")
    screening = immutable_inputs.get("screening_report", {})
    if artifact_hash(Path(screening.get("path", ""))) != screening.get("sha256"):
        raise RuntimeError("frozen screening report changed")
    for name, evidence in immutable_inputs.get("dependencies", {}).items():
        if artifact_hash(Path(evidence["path"])) != evidence["sha256"]:
            raise RuntimeError(f"frozen dependency {name} changed")


def _run_backfill_unlocked(
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
    state_path = state_dir / "frozen_family.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        validate_frozen_family(frozen)
        return frozen
    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    labels = combine_historical_labels(label_paths)
    trading_calendar = load_trading_calendar(intraday_dir)
    dates = validate_historical_dates(labels, trading_calendar)
    folds = registered_folds(dates)
    feature_window, base_features, minute_features, features = _selected_features(screening_report_path)
    panel, features, eligible_universe_hashes = _merge_prepared_panel(
        labels, daily_dir, intraday_dir, base_features, minute_features
    )
    prepared_vs_label_key_counts = panel.attrs.get("prepared_vs_label_key_counts", {})
    all_records = []
    fold_reports = []
    for fold in folds:
        fold_panel = panel[~panel["date"].isin(fold["purge"])].copy()
        train_end = pd.Timestamp(fold["train"][-1])
        oos_start = pd.Timestamp(fold["oos"][0])
        oos_end = pd.Timestamp(fold["oos"][-1])
        fitters = {
            "downside_quantile": _fit_downside,
            "cross_sectional_rank": _fit_rank,
            "conditional_payoff": _fit_conditional,
        }
        metrics = {}
        for family in FAMILY_ORDER:
            print(
                f"[target-redesign-backfill] fold={fold['name']} family={family} "
                f"train_end={train_end.date()} oos={oos_start.date()}..{oos_end.date()}",
                flush=True,
            )
            predictions = fitters[family](
                fold_panel, features, train_end, oos_start, oos_end, model_threads, fold["name"]
            )
            records, family_metrics = _evaluate(family, predictions, fold_panel, fold["oos"])
            records["fold"] = fold["name"]
            all_records.append(records)
            metrics[family] = family_metrics
        fold_reports.append({
            "name": fold["name"],
            "train_start": str(pd.Timestamp(fold["train"][0]).date()),
            "train_end": str(train_end.date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge"]],
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "oos_days": int(len(fold["oos"])),
            "status": "historical_walk_forward_not_untouched",
            "models": metrics,
        })
    execution_records = pd.concat(all_records, ignore_index=True)
    oos_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(value) for fold in folds for value in fold["oos"]})
    )
    account = cash_normalized_execution_records(
        execution_records, oos_dates, top_n=TOP_N, models=list(FAMILY_ORDER)
    )
    comparison = compare_execution_records(account)
    daily_returns = comparison["daily_returns"]
    family_dates = pd.DatetimeIndex(daily_returns["signal_date"].unique()).sort_values()
    if not family_dates.equals(oos_dates):
        raise RuntimeError("walk-forward comparison does not cover the exact OOS date index")
    for family in FAMILY_ORDER:
        if family not in daily_returns or daily_returns[family].isna().any():
            raise RuntimeError(f"target family {family} has missing OOS cohort returns")
    selected = select_family(comparison["models"])
    final_inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    if final_inputs != inputs:
        raise RuntimeError("target-redesign backfill inputs changed during evaluation")
    report = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "historical_backfill": True,
        "untouched_holdout": False,
        "historical_results_may_promote_production": False,
        "input_hashes": inputs,
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "feature_hash": _canonical_hash(features),
        "features": features,
        "historical_start": str(dates[0].date()),
        "historical_end": str(dates[-1].date()),
        "historical_days": int(len(dates)),
        "overlap_conflict_count": int(labels.attrs.get("overlap_conflict_count", 0)),
        "source_precedence": labels.attrs.get("source_precedence", []),
        "walk_forward_oos_days": int(len(oos_dates)),
        "eligible_universe_hashes": eligible_universe_hashes,
        "prepared_vs_label_key_counts": prepared_vs_label_key_counts,
        "universe_policy": "labels_must_be_subset_of_prepared_signal_eligible; unlabelled_prepared_keys_are_excluded",
        "folds": fold_reports,
        "evaluation_unit": "signal_cohort_normalized_top10",
        "overlapping_position_capital_accounting": False,
        "cohort_metrics_are_not_portfolio_capital_returns": True,
        "cohort_comparison": {
            "models": comparison["models"],
            "pairwise": comparison["pairwise"],
        },
        "selected_family": selected,
        "selected_recipe": MODEL_RECIPE[selected],
        "first_prospective_signal_date": str(FIRST_PROSPECTIVE_SIGNAL_DATE.date()),
        "forward_shadow_required": True,
        "production_candidate": False,
        "production_publication": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backfill_report.json"
    records_path = output_dir / "backfill_execution_records.parquet"
    daily_path = output_dir / "backfill_cohort_daily_returns.parquet"
    atomic_parquet(execution_records, records_path)
    atomic_parquet(comparison["daily_returns"], daily_path)
    atomic_json(report, report_path)
    frozen = {
        "protocol": PROTOCOL,
        "protocol_hash": report["protocol_hash"],
        "status": "family_frozen_awaiting_forward",
        "selection_basis": "historical_reused_walk_forward",
        "historical_backfill": True,
        "untouched_holdout": False,
        "eligible_for_production": False,
        "selected_family": selected,
        "selected_recipe": MODEL_RECIPE[selected],
        "feature_hash": report["feature_hash"],
        "features": features,
        "input_hashes": inputs,
        "backfill_report": {
            "path": str(report_path.resolve()), "sha256": artifact_hash(report_path)
        },
        "artifacts": {
            "report": {
                "path": str(report_path.resolve()), "sha256": artifact_hash(report_path)
            },
            "execution_records": {
                "path": str(records_path.resolve()), "sha256": artifact_hash(records_path)
            },
            "cohort_daily_returns": {
                "path": str(daily_path.resolve()), "sha256": artifact_hash(daily_path)
            },
        },
        "first_prospective_signal_date": str(FIRST_PROSPECTIVE_SIGNAL_DATE.date()),
        "human_approval_required": True,
        "production_publication": False,
    }
    frozen["freeze_hash"] = _canonical_hash(frozen)
    atomic_json(frozen, state_path)
    return frozen


def reselect_existing_report(
    report_path: Path,
    output_dir: Path,
    state_dir: Path,
) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("protocol") != PROTOCOL or report.get("historical_backfill") is not True:
        raise RuntimeError("only a historical backfill report can be reselected")
    selected = select_family(report["cohort_comparison"]["models"])
    corrected = dict(report)
    original_report_hash = artifact_hash(report_path)
    corrected["historical_metrics_source_report_hash"] = original_report_hash
    corrected["selected_family_before_fill_gate"] = report.get("selected_family")
    corrected["selected_family"] = selected
    corrected["selection_rule"] = "highest historical cohort mean among families with mean_filled_names/top_n >= 0.60"
    corrected["fill_rate_gate"] = {"top_n": TOP_N, "minimum": 0.60}
    corrected["production_candidate"] = False
    corrected["untouched_holdout"] = False
    original_inputs = report["input_hashes"]
    label_paths = [Path(item["path"]) for item in original_inputs["labels"]]
    current_inputs = _input_hashes(
        label_paths,
        Path(original_inputs["screening_report"]["path"]),
        Path(original_inputs["daily_prepared"]["path"]),
        Path(original_inputs["intraday_prepared"]["path"]),
    )
    corrected["input_hashes"] = current_inputs
    output_dir = Path(output_dir)
    state_dir = Path(state_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / "backfill_report_fill_gate.json"
    atomic_json(corrected, corrected_path)
    # Reuse the original immutable execution artifacts; the correction only changes selection metadata.
    original_state = Path(report_path).parent.parent / "target_redesign_backfill_v2_state" / "frozen_family.json"
    if original_state.exists():
        old = json.loads(original_state.read_text(encoding="utf-8"))
        old_artifacts = old["artifacts"]
    else:
        raise RuntimeError("original frozen backfill state is required for artifact reuse")
    frozen = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "status": "family_frozen_awaiting_forward",
        "selection_basis": "historical_reused_walk_forward_with_fill_gate",
        "historical_backfill": True,
        "untouched_holdout": False,
        "eligible_for_production": False,
        "selected_family": selected,
        "selected_recipe": MODEL_RECIPE[selected],
        "feature_hash": corrected["feature_hash"],
        "features": corrected["features"],
        "input_hashes": corrected["input_hashes"],
        "backfill_report": {
            "path": str(corrected_path.resolve()), "sha256": artifact_hash(corrected_path)
        },
        "artifacts": {
            "report": {"path": str(corrected_path.resolve()), "sha256": artifact_hash(corrected_path)},
            "execution_records": old_artifacts["execution_records"],
            "cohort_daily_returns": old_artifacts["cohort_daily_returns"],
        },
        "first_prospective_signal_date": str(FIRST_PROSPECTIVE_SIGNAL_DATE.date()),
        "human_approval_required": True,
        "production_publication": False,
    }
    frozen["freeze_hash"] = _canonical_hash(frozen)
    atomic_json(frozen, state_dir / "frozen_family.json")
    return frozen


def run_backfill(
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
        return _run_backfill_unlocked(
            label_paths, screening_report_path, output_dir, state_dir,
            daily_dir, intraday_dir, model_threads,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical purged target-redesign walk-forward race")
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--screening-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path)
    parser.add_argument("--intraday-dir", type=Path, default=config.PREPARED_DIR)
    parser.add_argument("--model-threads", type=int, default=8)
    args = parser.parse_args()
    result = run_backfill(
        args.labels,
        args.screening_report,
        args.output_dir,
        args.state_dir,
        args.daily_dir,
        args.intraday_dir,
        args.model_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
