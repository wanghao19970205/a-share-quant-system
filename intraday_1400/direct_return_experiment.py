from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from intraday_1400 import config
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.fair_race_pipeline import (
    _FOUR_MODEL_WEIGHTS,
    _causal_eligible_predictions,
    _fit_four_model_head,
    _json_report,
    default_daily_prepared_dir,
    load_joined_prepared,
)
from intraday_1400.offline_race import ExecutionConfig, compare_execution_records, simulate_fixed_exit_race
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records


PROTOCOL = "intraday_1400_direct_return_v1"
DEVELOPMENT_START = pd.Timestamp("2026-01-28")
DEVELOPMENT_END = pd.Timestamp("2026-04-13")
CALIBRATION_START = pd.Timestamp("2026-04-17")
CALIBRATION_END = pd.Timestamp("2026-04-30")
HOLDOUT_START = pd.Timestamp("2026-05-11")
HOLDOUT_END = pd.Timestamp("2026-08-03")
DEVELOPMENT_DAYS = 47
CALIBRATION_DAYS = 10
HOLDOUT_DAYS = 60
CONSUMED_INTERVAL_START = pd.Timestamp("2025-11-03")
CONSUMED_INTERVAL_END = pd.Timestamp("2026-01-27")
FEATURE_SCREENING_TRAIN_END = pd.Timestamp("2025-08-06")
TARGETS = (
    "adaptive_stress_net_ret_t3",
    "adaptive_realized_net_ret_t3",
)
TOP_NS = (5, 10, 15)
TRAINING_RECIPE = {
    "models": ["ridge", "lightgbm", "elastic_net", "extra_trees"],
    "weights": _FOUR_MODEL_WEIGHTS,
    "ridge": {"alpha": 10.0},
    "lightgbm": {"n_estimators": 200, "learning_rate": 0.015, "early_stopping_rounds": 0},
    "elastic_net": {"alpha": 0.001, "l1_ratio": 0.5},
    "extra_trees": {"n_estimators": 80, "max_train_rows": 150_000},
    "decay_half_life_days": 60.0,
    "min_weight": 0.03,
    "train_recipe_version": config.TRAIN_RECIPE_VERSION,
    "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
    "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
    "label_recipe_version": config.LABEL_RECIPE_VERSION,
}
EXECUTION_CONFIG = {
    "entry_time": "14:50",
    "time_exit_signal": "14:45",
    "roundtrip_cost": 0.002,
    "unsellable_return": -0.10,
    "stop_loss": 0.05,
    "take_profit": 0.09,
    "trailing_arm": 0.03,
    "trailing_drawdown": 0.02,
    "max_exit_sessions": 3,
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "development": [str(DEVELOPMENT_START.date()), str(DEVELOPMENT_END.date()), DEVELOPMENT_DAYS],
        "calibration": [str(CALIBRATION_START.date()), str(CALIBRATION_END.date()), CALIBRATION_DAYS],
        "holdout": [str(HOLDOUT_START.date()), str(HOLDOUT_END.date()), HOLDOUT_DAYS],
        "consumed_interval_excluded_from_selection": [
            str(CONSUMED_INTERVAL_START.date()), str(CONSUMED_INTERVAL_END.date())
        ],
        "feature_screening_train_end": str(FEATURE_SCREENING_TRAIN_END.date()),
        "targets": list(TARGETS),
        "top_n": list(TOP_NS),
        "training_recipe": TRAINING_RECIPE,
        "execution": EXECUTION_CONFIG,
        "selection_rule": ["mean_return", "compound_return", "max_drawdown", "target_order", "smaller_top_n"],
        "production_publication": False,
        "human_approval_required": True,
    }


@contextmanager
def _cycle_lock(state_dir: Path) -> Iterator[None]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".cycle.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another direct-return cycle is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(path)
    required = {
        "code", "date", "adaptive_entry_buyable", "adaptive_horizon_observed_t3",
        "adaptive_realized_net_ret_t3", "adaptive_stress_net_ret_t3",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"adaptive labels missing {sorted(missing)}")
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["code"] = labels["code"].astype(str).str[:6]
    labels = labels.dropna(subset=["date", "code"])
    if labels.duplicated(["code", "date"]).any():
        raise ValueError("adaptive labels require unique date+code keys")
    if labels["adaptive_stress_net_ret_t3"].isna().any():
        raise ValueError("adaptive stress target must be complete")
    return labels.sort_values(["date", "code"]).reset_index(drop=True)


def _dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame["date"].dropna().unique()).sort_values()


def _registered_dates(
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_count: int,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    registered = dates[(dates >= start) & (dates <= end)]
    if len(registered) != int(expected_count):
        raise ValueError(
            f"prepared trading calendar has {len(registered)} dates for {start.date()}..{end.date()}, "
            f"expected {int(expected_count)}"
        )
    return registered


def validate_selection_labels(
    base_labels: pd.DataFrame,
    selection_labels: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> dict:
    base_dates = _dates(base_labels)
    selection_dates = _dates(selection_labels)
    if base_dates.empty or selection_dates.empty:
        raise ValueError("base and selection labels must both contain dates")
    forbidden = base_dates[(base_dates >= CONSUMED_INTERVAL_START) & (base_dates <= CONSUMED_INTERVAL_END)]
    if len(forbidden):
        raise ValueError("previously consumed holdout cannot enter direct-return recipe selection")
    if base_dates[-1] >= DEVELOPMENT_START:
        raise ValueError("base labels must end before direct-return development")
    development = selection_dates[
        (selection_dates >= DEVELOPMENT_START) & (selection_dates <= DEVELOPMENT_END)
    ]
    calibration = selection_dates[
        (selection_dates >= CALIBRATION_START) & (selection_dates <= CALIBRATION_END)
    ]
    allowed = development.union(calibration)
    if len(development) != DEVELOPMENT_DAYS or len(calibration) != CALIBRATION_DAYS:
        raise ValueError("selection labels require exactly 47 development and 10 calibration dates")
    if not selection_dates.equals(allowed):
        raise ValueError("selection labels may contain only registered development and calibration dates")
    expected_development = _registered_dates(
        trading_calendar, DEVELOPMENT_START, DEVELOPMENT_END, DEVELOPMENT_DAYS
    )
    expected_calibration = _registered_dates(
        trading_calendar, CALIBRATION_START, CALIBRATION_END, CALIBRATION_DAYS
    )
    if not development.equals(expected_development) or not calibration.equals(expected_calibration):
        raise ValueError("selection dates differ from the prepared A-share trading calendar")
    return {
        "development_dates": development,
        "calibration_dates": calibration,
        "training_start": pd.Timestamp(base_dates[0]),
    }


def validate_holdout_labels(
    labels: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    dates = _dates(labels)
    if len(dates) != HOLDOUT_DAYS:
        raise ValueError(f"holdout requires exactly {HOLDOUT_DAYS} trading dates")
    if dates[0] != HOLDOUT_START or dates[-1] != HOLDOUT_END:
        raise ValueError("holdout boundaries do not match the frozen protocol")
    expected = _registered_dates(trading_calendar, HOLDOUT_START, HOLDOUT_END, HOLDOUT_DAYS)
    if not dates.equals(expected):
        raise ValueError("holdout dates differ from the prepared A-share trading calendar")
    return dates


def split_registered_labels(
    combined_labels_path: Path,
    output_dir: Path,
    prepared_dir: Path | None = None,
) -> dict:
    labels = _normalize_labels(combined_labels_path)
    selection = labels[
        ((labels["date"] >= DEVELOPMENT_START) & (labels["date"] <= DEVELOPMENT_END))
        | ((labels["date"] >= CALIBRATION_START) & (labels["date"] <= CALIBRATION_END))
    ].copy()
    holdout = labels[
        (labels["date"] >= HOLDOUT_START) & (labels["date"] <= HOLDOUT_END)
    ].copy()
    # Validate registered date counts without permitting purge rows into either artifact.
    selection_dates = _dates(selection)
    development = selection_dates[
        (selection_dates >= DEVELOPMENT_START) & (selection_dates <= DEVELOPMENT_END)
    ]
    calibration = selection_dates[
        (selection_dates >= CALIBRATION_START) & (selection_dates <= CALIBRATION_END)
    ]
    if len(development) != DEVELOPMENT_DAYS or len(calibration) != CALIBRATION_DAYS:
        raise ValueError("combined labels do not contain the complete registered selection dates")
    calendar = load_trading_calendar(Path(prepared_dir or config.PREPARED_DIR))
    if not development.equals(_registered_dates(
        calendar, DEVELOPMENT_START, DEVELOPMENT_END, DEVELOPMENT_DAYS
    )) or not calibration.equals(_registered_dates(
        calendar, CALIBRATION_START, CALIBRATION_END, CALIBRATION_DAYS
    )):
        raise ValueError("combined selection dates differ from the prepared A-share trading calendar")
    validate_holdout_labels(holdout, calendar)
    all_registered = development.union(calibration).union(_dates(holdout))
    extra_dates = _dates(labels).difference(all_registered)
    allowed_purge = extra_dates[
        ((extra_dates >= pd.Timestamp("2026-04-14")) & (extra_dates <= pd.Timestamp("2026-04-16")))
        | ((extra_dates >= pd.Timestamp("2026-05-06")) & (extra_dates <= pd.Timestamp("2026-05-08")))
    ]
    if not extra_dates.equals(allowed_purge):
        raise ValueError("combined labels contain dates outside registered splits and purge intervals")
    output_dir = Path(output_dir)
    selection_path = output_dir / "selection_labels.parquet"
    holdout_path = output_dir / "sealed_holdout_labels.parquet"
    atomic_parquet(selection, selection_path)
    atomic_parquet(holdout, holdout_path)
    manifest = {
        "protocol": PROTOCOL,
        "source": {"path": str(Path(combined_labels_path).resolve()), "sha256": artifact_hash(combined_labels_path)},
        "selection": {"path": str(selection_path.resolve()), "sha256": artifact_hash(selection_path)},
        "holdout": {"path": str(holdout_path.resolve()), "sha256": artifact_hash(holdout_path)},
        "selection_days": int(len(selection_dates)),
        "holdout_days": int(len(_dates(holdout))),
        "purge_rows_excluded": int(len(labels) - len(selection) - len(holdout)),
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    atomic_json(manifest, output_dir / "label_split_manifest.json")
    return manifest


def validate_split_manifest(
    manifest_path: Path,
    selection_labels_path: Path,
    holdout_labels_path: Path | None = None,
) -> dict:
    manifest = _read_json(manifest_path)
    content = dict(manifest)
    manifest_hash = content.pop("manifest_hash", None)
    if manifest_hash != _canonical_hash(content):
        raise RuntimeError("direct-return label split manifest was modified")
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("wrong direct-return label split protocol")
    expected_selection = manifest.get("selection", {})
    selection_path = Path(selection_labels_path).resolve()
    if (
        expected_selection.get("path") != str(selection_path)
        or expected_selection.get("sha256") != artifact_hash(selection_path)
    ):
        raise RuntimeError("selection labels do not match the sealed split manifest")
    if holdout_labels_path is not None:
        expected_holdout = manifest.get("holdout", {})
        holdout_path = Path(holdout_labels_path).resolve()
        if (
            expected_holdout.get("path") != str(holdout_path)
            or expected_holdout.get("sha256") != artifact_hash(holdout_path)
        ):
            raise RuntimeError("holdout labels do not match the sealed split manifest")
    return manifest


def _selected_features(screening_report_path: Path) -> tuple[dict, list[str], list[str], list[str]]:
    report = _read_json(screening_report_path)
    windows = [
        item for item in report.get("windows", [])
        if pd.Timestamp(item["train_end"]) == FEATURE_SCREENING_TRAIN_END
    ]
    if len(windows) != 1:
        raise ValueError(
            f"feature screening requires exactly one frozen train_end="
            f"{FEATURE_SCREENING_TRAIN_END.date()} window"
        )
    window = windows[0]
    selected = window["selected"]["daily_asof_plus_minute_control"]
    base = list(selected["asof_matched"])
    minute = list(selected["minute"])
    features = [*base, *minute]
    if not features or len(features) != len(set(features)):
        raise ValueError("frozen direct-return features must be non-empty and unique")
    return window, base, minute, features


def _input_hashes(
    base_labels_path: Path,
    selection_labels_path: Path,
    split_manifest_path: Path,
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
) -> dict:
    paths = {
        "base_labels": Path(base_labels_path),
        "selection_labels": Path(selection_labels_path),
        "split_manifest": Path(split_manifest_path),
        "screening_report": Path(screening_report_path),
        "daily_prepared": Path(daily_dir),
        "intraday_prepared": Path(intraday_dir),
        "controller_code": Path(__file__),
        "model_head_code": Path(__file__).with_name("fair_race_pipeline.py"),
        "quant_model_code": Path(__file__).parent.parent / "quant" / "model.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": artifact_hash(path)}
        for name, path in paths.items()
    }


def _execution_labels(panel: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return panel[panel["date"].isin(dates)][[
        "code", "date", "adaptive_entry_buyable", "adaptive_realized_net_ret_t3",
        "adaptive_horizon_observed_t3",
    ]].rename(columns={
        "adaptive_entry_buyable": "entry_buyable",
        "adaptive_realized_net_ret_t3": "target_net_ret_t1",
        "adaptive_horizon_observed_t3": "target_outcome_observed_t1",
    })


def _simulate_adaptive_label_race(
    predictions: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    records, _ = simulate_fixed_exit_race(predictions, labels, execution_config)
    entered = records["entry_buyable"].fillna(False).astype(bool)
    sellable = records["exit_sellable"].fillna(False).astype(bool)
    records.loc[entered & sellable, "exit_reason"] = "adaptive_t3_exit"
    records.loc[entered & ~sellable, "exit_reason"] = "adaptive_t3_unsellable"
    return records


def _evaluate_recipe(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target: str,
    top_n: int,
) -> tuple[pd.DataFrame, dict]:
    name = f"{target}__top{int(top_n)}"
    records = _simulate_adaptive_label_race(
        {name: predictions},
        _execution_labels(panel, dates),
        ExecutionConfig(top_n=int(top_n), **EXECUTION_CONFIG),
    )
    account = cash_normalized_execution_records(records, dates, top_n=int(top_n), models=[name])
    report = compare_execution_records(account)
    metrics = report["models"].get(name, {"days": 0})
    return records, {"name": name, "target": target, "top_n": int(top_n), "metrics": metrics}


def select_recipe(candidates: list[dict]) -> dict:
    expected = {(target, top_n) for target in TARGETS for top_n in TOP_NS}
    actual = {(item.get("target"), item.get("top_n")) for item in candidates}
    if actual != expected or len(candidates) != len(expected):
        raise ValueError("recipe selection requires the exact six registered candidates")
    if any(int(item.get("metrics", {}).get("days", 0)) != CALIBRATION_DAYS for item in candidates):
        raise ValueError("every recipe requires exactly ten fixed-capital calibration days")

    def rank(item: dict) -> tuple:
        metrics = item["metrics"]
        target_priority = -TARGETS.index(item["target"])
        return (
            float(metrics.get("mean_return", float("-inf"))),
            float(metrics.get("compound_return", float("-inf"))),
            float(metrics.get("max_drawdown", float("-inf"))),
            target_priority,
            -int(item["top_n"]),
        )

    return max(candidates, key=rank)


def _build_panel(
    labels: pd.DataFrame,
    base_features: list[str],
    minute_features: list[str],
    daily_dir: Path,
    intraday_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    panel, _ = load_joined_prepared(
        daily_dir,
        intraday_dir,
        start,
        end,
        daily_features=[name.removeprefix("asof__") for name in base_features],
        asof_features=[name.removeprefix("asof__") for name in base_features],
        minute_features=[name.removeprefix("minute__") for name in minute_features],
    )
    panel = panel.merge(labels, on=["code", "date"], how="inner", validate="one_to_one")
    return panel, [*base_features, *minute_features]


def develop_and_freeze(
    base_labels_path: Path,
    selection_labels_path: Path,
    split_manifest_path: Path,
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
) -> dict:
    output_dir = Path(output_dir)
    state_dir = Path(state_dir)
    freeze_path = state_dir / "frozen_recipe.json"
    with _cycle_lock(state_dir):
        if freeze_path.exists():
            manifest = _read_json(freeze_path)
            validate_frozen_recipe(manifest)
            return manifest
        daily_dir = Path(daily_dir or default_daily_prepared_dir())
        intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
        validate_split_manifest(split_manifest_path, selection_labels_path)
        before = _input_hashes(
            base_labels_path, selection_labels_path, split_manifest_path,
            screening_report_path, daily_dir, intraday_dir
        )
        base_labels = _normalize_labels(base_labels_path)
        selection_labels = _normalize_labels(selection_labels_path)
        trading_calendar = load_trading_calendar(intraday_dir)
        split = validate_selection_labels(base_labels, selection_labels, trading_calendar)
        feature_window, base_features, minute_features, features = _selected_features(screening_report_path)
        training_labels = pd.concat([
            base_labels,
            selection_labels[selection_labels["date"].isin(split["development_dates"])],
        ], ignore_index=True).drop_duplicates(["code", "date"], keep="last")
        panel_labels = pd.concat([
            training_labels,
            selection_labels[selection_labels["date"].isin(split["calibration_dates"])],
        ], ignore_index=True)
        panel, features = _build_panel(
            panel_labels, base_features, minute_features, daily_dir, intraday_dir,
            split["training_start"], CALIBRATION_END,
        )
        all_records = []
        candidates = []
        model_metrics = {}
        for target in TARGETS:
            print(f"[direct-return:development] target={target}", flush=True)
            head = _fit_four_model_head(
                panel, features, target, DEVELOPMENT_END,
                CALIBRATION_START, CALIBRATION_END, model_threads,
            )
            model_metrics[target] = head["metrics"]
            eligible = _causal_eligible_predictions(
                {target: head["predictions"]}, panel, CALIBRATION_START, CALIBRATION_END
            )[target]
            for top_n in TOP_NS:
                records, candidate = _evaluate_recipe(
                    eligible, panel, split["calibration_dates"], target, top_n
                )
                all_records.append(records)
                candidates.append(candidate)
        selected = select_recipe(candidates)
        after = _input_hashes(
            base_labels_path, selection_labels_path, split_manifest_path,
            screening_report_path, daily_dir, intraday_dir
        )
        if after != before:
            raise RuntimeError("direct-return selection inputs changed during evaluation")
        report = {
            "experiment": PROTOCOL,
            "phase": "development_calibration",
            "protocol_hash": _canonical_hash(_protocol_payload()),
            "input_hashes": before,
            "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
            "features": features,
            "feature_hash": _canonical_hash(features),
            "training_recipe": TRAINING_RECIPE,
            "training_recipe_hash": _canonical_hash(TRAINING_RECIPE),
            "execution_recipe": {"source": "adaptive_t3_replay_labels", **EXECUTION_CONFIG},
            "development_start": str(DEVELOPMENT_START.date()),
            "development_end": str(DEVELOPMENT_END.date()),
            "development_days": DEVELOPMENT_DAYS,
            "calibration_start": str(CALIBRATION_START.date()),
            "calibration_end": str(CALIBRATION_END.date()),
            "calibration_days": CALIBRATION_DAYS,
            "candidates": candidates,
            "selected_recipe": {
                "target": selected["target"], "top_n": selected["top_n"], "name": selected["name"]
            },
            "model_metrics": model_metrics,
            "holdout_read": False,
            "production_publication": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        execution_records = pd.concat(all_records, ignore_index=True)
        atomic_parquet(execution_records, output_dir / "calibration_execution_records.parquet")
        atomic_json(report, output_dir / "development_report.json")
        manifest = {
            "protocol": PROTOCOL,
            "protocol_hash": _canonical_hash(_protocol_payload()),
            "status": "recipe_frozen",
            "selected_recipe": report["selected_recipe"],
            "features": features,
            "feature_hash": report["feature_hash"],
            "feature_screening_train_end": report["feature_screening_train_end"],
            "training_recipe": TRAINING_RECIPE,
            "training_recipe_hash": _canonical_hash(TRAINING_RECIPE),
            "execution_recipe": report["execution_recipe"],
            "selection_input_hashes": before,
            "development_report": str((output_dir / "development_report.json").resolve()),
            "development_report_hash": artifact_hash(output_dir / "development_report.json"),
            "holdout_claim": None,
            "production_isolated": True,
            "human_approval_required": True,
            "production_publication": False,
        }
        manifest["freeze_hash"] = _canonical_hash(manifest)
        atomic_json(manifest, freeze_path)
        return manifest


def validate_frozen_recipe(manifest: dict) -> None:
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("wrong direct-return protocol")
    if manifest.get("protocol_hash") != _canonical_hash(_protocol_payload()):
        raise RuntimeError("direct-return protocol changed; create a new state directory")
    content = dict(manifest)
    freeze_hash = content.pop("freeze_hash", None)
    if freeze_hash != _canonical_hash(content):
        raise RuntimeError("frozen recipe manifest was modified")
    if manifest.get("selected_recipe", {}).get("target") not in TARGETS:
        raise RuntimeError("frozen target is outside the registered grid")
    if manifest.get("selected_recipe", {}).get("top_n") not in TOP_NS:
        raise RuntimeError("frozen TopN is outside the registered grid")
    if manifest.get("feature_hash") != _canonical_hash(manifest.get("features", [])):
        raise RuntimeError("frozen feature list was modified")
    if manifest.get("training_recipe") != TRAINING_RECIPE:
        raise RuntimeError("frozen training recipe differs from the current implementation")
    if manifest.get("training_recipe_hash") != _canonical_hash(TRAINING_RECIPE):
        raise RuntimeError("frozen training recipe hash is invalid")
    report_path = Path(manifest.get("development_report", ""))
    if not report_path.is_file() or artifact_hash(report_path) != manifest.get("development_report_hash"):
        raise RuntimeError("frozen development report is missing or changed")
    frozen_inputs = manifest.get("selection_input_hashes", {})
    required_inputs = {
        "base_labels", "selection_labels", "split_manifest", "screening_report",
        "daily_prepared", "intraday_prepared", "controller_code", "model_head_code",
        "quant_model_code",
    }
    if set(frozen_inputs) != required_inputs:
        raise RuntimeError("frozen selection input ledger is incomplete")
    for name in sorted(required_inputs):
        evidence = frozen_inputs.get(name, {})
        path = Path(evidence.get("path", ""))
        if not path.exists() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"frozen {name} is missing or changed")
    if manifest.get("production_isolated") is not True or manifest.get("human_approval_required") is not True:
        raise RuntimeError("direct-return research isolation cannot be disabled")
    if manifest.get("production_publication") is not False:
        raise RuntimeError("automatic production publication is forbidden")


def _twenty_day_blocks(records: pd.DataFrame, dates: pd.DatetimeIndex, top_n: int, model: str) -> list[dict]:
    blocks = []
    for index in range(3):
        block_dates = dates[index * 20:(index + 1) * 20]
        selected = records[records["signal_date"].isin(block_dates)]
        account = cash_normalized_execution_records(
            selected, block_dates, top_n=top_n, models=[model]
        )
        blocks.append({
            "block": index + 1,
            "start": str(pd.Timestamp(block_dates[0]).date()),
            "end": str(pd.Timestamp(block_dates[-1]).date()),
            "account_comparison": _json_report(compare_execution_records(account)),
        })
    return blocks


def _holdout_input_hashes(
    frozen_recipe_path: Path,
    base_labels_path: Path,
    selection_labels_path: Path,
    split_manifest_path: Path,
    holdout_labels_path: Path,
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
) -> dict:
    paths = {
        "frozen_recipe": Path(frozen_recipe_path),
        "base_labels": Path(base_labels_path),
        "selection_labels": Path(selection_labels_path),
        "split_manifest": Path(split_manifest_path),
        "holdout_labels": Path(holdout_labels_path),
        "screening_report": Path(screening_report_path),
        "daily_prepared": Path(daily_dir),
        "intraday_prepared": Path(intraday_dir),
        "controller_code": Path(__file__),
        "model_head_code": Path(__file__).with_name("fair_race_pipeline.py"),
        "quant_model_code": Path(__file__).parent.parent / "quant" / "model.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": artifact_hash(path)}
        for name, path in paths.items()
    }


def _seal_holdout_state(state: dict) -> dict:
    sealed = dict(state)
    sealed.pop("state_hash", None)
    sealed["state_hash"] = _canonical_hash(sealed)
    return sealed


def _validate_holdout_state(state: dict) -> None:
    content = dict(state)
    state_hash = content.pop("state_hash", None)
    if state_hash != _canonical_hash(content):
        raise RuntimeError("direct-return holdout state was modified")
    if state.get("protocol") != PROTOCOL:
        raise RuntimeError("wrong direct-return holdout-state protocol")
    if state.get("start") != str(HOLDOUT_START.date()) or state.get("end") != str(HOLDOUT_END.date()):
        raise RuntimeError("direct-return holdout-state interval changed")
    if state.get("claim_hash") != _canonical_hash(state.get("input_hashes", {})):
        raise RuntimeError("direct-return holdout claim hash is invalid")
    if state.get("production_publication") is not False:
        raise RuntimeError("automatic production publication is forbidden")


def run_holdout_cycle(
    base_labels_path: Path,
    selection_labels_path: Path,
    split_manifest_path: Path,
    holdout_labels_path: Path,
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
) -> dict:
    output_dir = Path(output_dir)
    state_dir = Path(state_dir)
    frozen_recipe_path = state_dir / "frozen_recipe.json"
    state_path = state_dir / "holdout_state.json"
    report_path = output_dir / "holdout_report.json"
    with _cycle_lock(state_dir):
        manifest = _read_json(frozen_recipe_path)
        validate_frozen_recipe(manifest)
        if state_path.exists():
            state = _read_json(state_path)
            _validate_holdout_state(state)
            if state.get("freeze_hash") != manifest.get("freeze_hash"):
                raise RuntimeError("holdout claim is bound to a different frozen recipe")
            if state.get("status") == "consumed":
                artifacts = state.get("artifacts", {})
                for name, evidence in artifacts.items():
                    path = Path(evidence.get("path", ""))
                    if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
                        raise RuntimeError(f"consumed direct-return {name} changed")
                if set(artifacts) != {"report", "execution_records", "account_daily_returns"}:
                    raise RuntimeError("consumed direct-return artifact ledger is incomplete")
                return _read_json(report_path)
            if state.get("status") == "claimed":
                state["status"] = "abandoned"
                state["failure"] = "recovered stale claim from interrupted cycle"
                atomic_json(_seal_holdout_state(state), state_path)
        daily_dir = Path(daily_dir or default_daily_prepared_dir())
        intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
        validate_split_manifest(
            split_manifest_path, selection_labels_path, holdout_labels_path
        )
        before = _holdout_input_hashes(
            frozen_recipe_path, base_labels_path, selection_labels_path, split_manifest_path,
            holdout_labels_path, screening_report_path, daily_dir, intraday_dir,
        )
        for name, frozen_evidence in manifest["selection_input_hashes"].items():
            if before.get(name) != frozen_evidence:
                raise RuntimeError(f"holdout {name} differs from the frozen selection input")
        holdout_labels = _normalize_labels(holdout_labels_path)
        trading_calendar = load_trading_calendar(intraday_dir)
        holdout_dates = validate_holdout_labels(holdout_labels, trading_calendar)
        state = {
            "protocol": PROTOCOL,
            "status": "claimed",
            "start": str(holdout_dates[0].date()),
            "end": str(holdout_dates[-1].date()),
            "input_hashes": before,
            "claim_hash": _canonical_hash(before),
            "freeze_hash": manifest["freeze_hash"],
            "production_publication": False,
        }
        state = _seal_holdout_state(state)
        atomic_json(state, state_path)
        try:
            base_labels = _normalize_labels(base_labels_path)
            selection_labels = _normalize_labels(selection_labels_path)
            validate_selection_labels(base_labels, selection_labels, trading_calendar)
            training_labels = pd.concat([base_labels, selection_labels], ignore_index=True).drop_duplicates(
                ["code", "date"], keep="last"
            )
            labels = pd.concat([training_labels, holdout_labels], ignore_index=True)
            features = list(manifest["features"])
            base_features = [name for name in features if name.startswith("asof__")]
            minute_features = [name for name in features if name.startswith("minute__")]
            panel, _ = _build_panel(
                labels, base_features, minute_features, daily_dir, intraday_dir,
                pd.Timestamp(training_labels["date"].min()), HOLDOUT_END,
            )
            recipe = manifest["selected_recipe"]
            target = recipe["target"]
            top_n = int(recipe["top_n"])
            head = _fit_four_model_head(
                panel, features, target, CALIBRATION_END, HOLDOUT_START, HOLDOUT_END, model_threads
            )
            predictions = _causal_eligible_predictions(
                {recipe["name"]: head["predictions"]}, panel, HOLDOUT_START, HOLDOUT_END
            )[recipe["name"]]
            records = _simulate_adaptive_label_race(
                {recipe["name"]: predictions},
                _execution_labels(panel, holdout_dates),
                ExecutionConfig(top_n=top_n, **EXECUTION_CONFIG),
            )
            account = cash_normalized_execution_records(
                records, holdout_dates, top_n=top_n, models=[recipe["name"]]
            )
            comparison = compare_execution_records(account)
            metrics = comparison["models"][recipe["name"]]
            blocks = _twenty_day_blocks(records, holdout_dates, top_n, recipe["name"])
            positive_blocks = sum(
                block["account_comparison"]["models"][recipe["name"]]["mean_return"] > 0
                for block in blocks
            )
            fill_rate = float(metrics["mean_filled_names"]) / float(top_n)
            forward_shadow = (
                float(metrics["mean_return"]) > 0
                and float(metrics["compound_return"]) > 0
                and float(metrics["max_drawdown"]) >= -0.20
                and fill_rate >= 0.60
                and positive_blocks >= 2
            )
            after = _holdout_input_hashes(
                frozen_recipe_path, base_labels_path, selection_labels_path, split_manifest_path,
                holdout_labels_path, screening_report_path, daily_dir, intraday_dir,
            )
            if after != before:
                raise RuntimeError("direct-return holdout inputs changed during evaluation")
            report = {
                "experiment": PROTOCOL,
                "phase": "untouched_holdout",
                "untouched_holdout": True,
                "input_hashes": before,
                "freeze_hash": manifest["freeze_hash"],
                "selected_recipe": recipe,
                "training_recipe": TRAINING_RECIPE,
                "execution_recipe": {"source": "adaptive_t3_replay_labels", **EXECUTION_CONFIG},
                "training_cutoff": str(CALIBRATION_END.date()),
                "holdout_start": str(HOLDOUT_START.date()),
                "holdout_end": str(HOLDOUT_END.date()),
                "holdout_days": HOLDOUT_DAYS,
                "account_comparison": _json_report(comparison),
                "twenty_day_blocks": blocks,
                "fill_rate": fill_rate,
                "positive_twenty_day_blocks": positive_blocks,
                "forward_shadow_eligible": forward_shadow,
                "next_branch": "forward_shadow" if forward_shadow else "target_redesign",
                "production_candidate": False,
                "production_publication": False,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            records_path = output_dir / "holdout_execution_records.parquet"
            daily_returns_path = output_dir / "holdout_account_daily_returns.parquet"
            atomic_parquet(records, records_path)
            atomic_parquet(comparison["daily_returns"], daily_returns_path)
            atomic_json(report, report_path)
            artifacts = {
                "report": {"path": str(report_path.resolve()), "sha256": artifact_hash(report_path)},
                "execution_records": {
                    "path": str(records_path.resolve()), "sha256": artifact_hash(records_path)
                },
                "account_daily_returns": {
                    "path": str(daily_returns_path.resolve()), "sha256": artifact_hash(daily_returns_path)
                },
            }
            state.update({
                "status": "consumed",
                "artifacts": artifacts,
                "forward_shadow_eligible": forward_shadow,
                "production_publication": False,
            })
            state = _seal_holdout_state(state)
            atomic_json(state, state_path)
            return report
        except Exception as error:
            state["status"] = "abandoned"
            state["failure"] = f"{type(error).__name__}: {error}"
            atomic_json(_seal_holdout_state(state), state_path)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe direct adaptive-return experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split-labels")
    split.add_argument("--combined-labels", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--prepared-dir", type=Path, default=config.PREPARED_DIR)
    for command in ("develop-and-freeze", "run-holdout-cycle"):
        child = subparsers.add_parser(command)
        child.add_argument("--base-labels", type=Path, required=True)
        child.add_argument("--selection-labels", type=Path, required=True)
        child.add_argument("--split-manifest", type=Path, required=True)
        child.add_argument("--screening-report", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--state-dir", type=Path, required=True)
        child.add_argument("--model-threads", type=int, default=8)
        if command == "run-holdout-cycle":
            child.add_argument("--holdout-labels", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "split-labels":
        result = split_registered_labels(args.combined_labels, args.output_dir, args.prepared_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    common = {
        "base_labels_path": args.base_labels,
        "selection_labels_path": args.selection_labels,
        "split_manifest_path": args.split_manifest,
        "screening_report_path": args.screening_report,
        "output_dir": args.output_dir,
        "state_dir": args.state_dir,
        "model_threads": args.model_threads,
    }
    if args.command == "develop-and-freeze":
        result = develop_and_freeze(**common)
    else:
        result = run_holdout_cycle(holdout_labels_path=args.holdout_labels, **common)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
