from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from intraday_1400 import config
from intraday_1400.fair_race_pipeline import (
    _FOUR_MODEL_WEIGHTS,
    _causal_eligible_predictions,
    _cap_training_panel,
    _cash_complete_targets,
    _exit_expected_value_frame,
    _fit_four_model_head,
    _json_report,
    _score_frame,
    default_daily_prepared_dir,
    load_joined_prepared,
)
from intraday_1400.offline_race import ExecutionConfig, compare_execution_records, simulate_fixed_exit_race
from intraday_1400.storage import atomic_json, atomic_parquet


FORWARD_PROTOCOL = "e3_minus_10_causal_rolling_v1"
FORWARD_PENALTY = -0.10
FORWARD_TOP_N = 10
FORWARD_ROUNDTRIP_COST = 0.002
FORWARD_UNSELLABLE_RETURN = -0.10
FORWARD_MAX_TRAIN_ROWS = 600_000


def _month_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("????-??.parquet"))


def _available_dates(directory: Path) -> pd.DatetimeIndex:
    dates = []
    for path in _month_files(directory):
        frame = pd.read_parquet(path, columns=["date"])
        dates.extend(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize().unique())
    return pd.DatetimeIndex(dates).drop_duplicates().sort_values()


def available_common_dates(daily_dir: Path, intraday_dir: Path) -> pd.DatetimeIndex:
    daily = _available_dates(Path(daily_dir))
    intraday = _available_dates(Path(intraday_dir))
    common = daily.intersection(intraday).sort_values()
    if common.empty:
        raise RuntimeError("daily and intraday prepared data have no common dates")
    return common


def _screening_recipe(screening_report_path: Path) -> dict:
    report_bytes = screening_report_path.read_bytes()
    screening = json.loads(report_bytes.decode("utf-8"))
    windows = screening.get("windows", [])
    if not windows:
        raise ValueError("screening report has no windows")
    latest = max(windows, key=lambda item: pd.Timestamp(item["train_end"]))
    selected = latest["selected"]["daily_asof_plus_minute_control"]
    base_features = list(selected["asof_matched"])
    minute_features = list(selected["minute"])
    if not base_features or not minute_features:
        raise ValueError("forward recipe requires frozen as-of and minute features")
    return {
        "screening_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "screening_window": int(latest["window"]),
        "screening_train_end": str(pd.Timestamp(latest["train_end"]).date()),
        "base_features": base_features,
        "minute_features": minute_features,
    }


def _immutable_payload(cutoff_date: pd.Timestamp, recipe: dict) -> dict:
    return {
        "protocol": FORWARD_PROTOCOL,
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "forward_rule": "strictly after cutoff_date",
        "training_rule": "for signal D train through D-2 and purge D-1",
        "models": list(_FOUR_MODEL_WEIGHTS),
        "model_weights": _FOUR_MODEL_WEIGHTS,
        "penalty": FORWARD_PENALTY,
        "top_n": FORWARD_TOP_N,
        "roundtrip_cost": FORWARD_ROUNDTRIP_COST,
        "unsellable_return": FORWARD_UNSELLABLE_RETURN,
        "max_train_rows": FORWARD_MAX_TRAIN_ROWS,
        "schema_version": config.SCHEMA_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
        "label_recipe_version": config.LABEL_RECIPE_VERSION,
        "train_recipe_version": config.TRAIN_RECIPE_VERSION,
        "cutoff_time": config.CUTOFF_TIME,
        **recipe,
    }


def _payload_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initialize_forward_evaluation(
    screening_report_path: Path,
    state_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
) -> dict:
    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    dates = available_common_dates(daily_dir, intraday_dir)
    cutoff_date = pd.Timestamp(dates[-1])
    payload = {
        **_immutable_payload(cutoff_date, _screening_recipe(Path(screening_report_path))),
        "daily_prepared_dir": str(daily_dir.resolve()),
        "intraday_prepared_dir": str(intraday_dir.resolve()),
    }
    manifest = {
        **payload,
        "immutable_sha256": _payload_hash(payload),
        "processed_dates": [],
        "status": "initialized_waiting_for_new_mature_dates",
    }
    state_dir = Path(state_dir)
    path = state_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_forward_manifest(existing, screening_report_path)
        return existing
    atomic_json(manifest, path)
    return manifest


def validate_forward_manifest(manifest: dict, screening_report_path: Path) -> None:
    mutable = {"immutable_sha256", "processed_dates", "status"}
    payload = {key: value for key, value in manifest.items() if key not in mutable}
    if manifest.get("immutable_sha256") != _payload_hash(payload):
        raise RuntimeError("forward manifest immutable configuration was modified")
    current_recipe = _screening_recipe(Path(screening_report_path))
    for key in ("screening_sha256", "base_features", "minute_features"):
        if manifest.get(key) != current_recipe.get(key):
            raise RuntimeError(f"forward recipe drift detected: {key}")
    version_checks = {
        "schema_version": config.SCHEMA_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
        "label_recipe_version": config.LABEL_RECIPE_VERSION,
        "train_recipe_version": config.TRAIN_RECIPE_VERSION,
        "cutoff_time": config.CUTOFF_TIME,
    }
    for key, value in version_checks.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"forward recipe version drift detected: {key}")


def _mature_forward_dates(
    manifest: dict,
    daily_dir: Path,
    intraday_dir: Path,
) -> list[pd.Timestamp]:
    common = available_common_dates(daily_dir, intraday_dir)
    cutoff = pd.Timestamp(manifest["cutoff_date"])
    processed = {pd.Timestamp(value) for value in manifest.get("processed_dates", [])}
    candidates = [pd.Timestamp(value) for value in common if pd.Timestamp(value) > cutoff and pd.Timestamp(value) not in processed]
    mature = []
    for date in candidates:
        month_path = intraday_dir / f"{date:%Y-%m}.parquet"
        if not month_path.exists():
            continue
        labels = pd.read_parquet(month_path, columns=["date", "target_outcome_observed_t1"])
        day = labels[pd.to_datetime(labels["date"], errors="coerce").dt.normalize() == date]
        if not day.empty and day["target_outcome_observed_t1"].fillna(False).astype(bool).any():
            mature.append(date)
    return mature


def _append_records(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh.copy()
    combined = combined.drop_duplicates(["model", "signal_date", "code"], keep="last")
    return combined.sort_values(["signal_date", "model", "rank", "code"]).reset_index(drop=True)


def run_forward_evaluation_once(
    screening_report_path: Path,
    state_dir: Path,
    model_threads: int = 8,
) -> dict:
    state_dir = Path(state_dir)
    manifest_path = state_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("forward evaluation is not initialized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_forward_manifest(manifest, screening_report_path)
    daily_dir = Path(manifest["daily_prepared_dir"])
    intraday_dir = Path(manifest["intraday_prepared_dir"])
    mature_dates = _mature_forward_dates(manifest, daily_dir, intraday_dir)
    if not mature_dates:
        manifest["status"] = "waiting_for_new_mature_dates"
        atomic_json(manifest, manifest_path)
        return {"status": manifest["status"], "processed_dates": manifest.get("processed_dates", [])}
    common_dates = available_common_dates(daily_dir, intraday_dir)
    base_features = list(manifest["base_features"])
    minute_features = list(manifest["minute_features"])
    features = [*base_features, *minute_features]
    source_base = [name.removeprefix("asof__") for name in base_features]
    source_minute = [name.removeprefix("minute__") for name in minute_features]
    all_fresh_records = []
    for signal_date in mature_dates:
        prior = common_dates[common_dates < signal_date]
        if len(prior) < 2:
            raise RuntimeError(f"insufficient prior dates for {signal_date.date()}")
        train_end = pd.Timestamp(prior[-2])
        purge_date = pd.Timestamp(prior[-1])
        train_start = max(pd.Timestamp("2023-01-01"), signal_date - pd.DateOffset(months=48))
        panel, _ = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            signal_date,
            daily_features=source_base,
            asof_features=source_base,
            minute_features=source_minute,
            exclude_dates=[purge_date],
        )
        full_panel = panel
        sampled = _cap_training_panel(panel, train_end, max_train_rows=FORWARD_MAX_TRAIN_ROWS)
        data = _cash_complete_targets(sampled)
        baseline = _fit_four_model_head(
            data, features, "target_penalty_net_ret_t1",
            train_end, signal_date, signal_date, model_threads,
        )
        sellable = _fit_four_model_head(
            data, features, "target_exit_sellable_t1",
            train_end, signal_date, signal_date, model_threads,
        )
        conditional_return = _fit_four_model_head(
            data, features, "target_net_ret_t1",
            train_end, signal_date, signal_date, model_threads,
        )
        predictions = {
            "forward_e0": _score_frame(baseline["predictions"], "forward_e0"),
            "forward_e3_minus_10": _exit_expected_value_frame(
                sellable["predictions"], conditional_return["predictions"],
                FORWARD_PENALTY, "forward_e3_minus_10",
            ),
        }
        predictions = _causal_eligible_predictions(predictions, full_panel, signal_date, signal_date)
        prediction_dir = state_dir / "predictions"
        for name, frame in predictions.items():
            atomic_parquet(
                frame,
                prediction_dir / f"{signal_date:%Y-%m-%d}_{name}.parquet",
            )
        labels = full_panel.loc[
            full_panel["date"] == signal_date,
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        records, _ = simulate_fixed_exit_race(
            predictions,
            labels,
            ExecutionConfig(
                top_n=FORWARD_TOP_N,
                roundtrip_cost=FORWARD_ROUNDTRIP_COST,
                unsellable_return=FORWARD_UNSELLABLE_RETURN,
            ),
        )
        records["train_end"] = train_end
        records["purge_date"] = purge_date
        all_fresh_records.append(records)
    fresh_records = pd.concat(all_fresh_records, ignore_index=True)
    records_path = state_dir / "execution_records.parquet"
    records = _append_records(records_path, fresh_records)
    comparison = compare_execution_records(records)
    completed_dates = sorted(set([
        *manifest.get("processed_dates", []),
        *(str(value.date()) for value in mature_dates),
    ]))
    report = {
        "protocol": FORWARD_PROTOCOL,
        "cutoff_date": manifest["cutoff_date"],
        "processed_dates": completed_dates,
        "development_only": False,
        "frozen_configuration_sha256": manifest["immutable_sha256"],
        "comparison": _json_report(comparison),
    }
    atomic_parquet(records, records_path)
    atomic_parquet(comparison["daily_returns"], state_dir / "daily_returns.parquet")
    atomic_json(report, state_dir / "forward_report.json")
    manifest["processed_dates"] = completed_dates
    manifest["status"] = "active"
    atomic_json(manifest, manifest_path)
    return report


def forward_status(state_dir: Path) -> dict:
    path = Path(state_dir) / "manifest.json"
    if not path.exists():
        return {"status": "not_initialized"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    report_path = Path(state_dir) / "forward_report.json"
    return {
        "status": manifest.get("status"),
        "cutoff_date": manifest.get("cutoff_date"),
        "processed_dates": manifest.get("processed_dates", []),
        "immutable_sha256": manifest.get("immutable_sha256"),
        "report_exists": report_path.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen forward evaluation for intraday E3 minus 10%")
    parser.add_argument("command", choices=("init", "run", "status"))
    parser.add_argument("--screening-report", type=Path, default=config.REPORT_DIR / "fair_race_feature_screening.json")
    parser.add_argument("--state-dir", type=Path, default=config.DATA_ROOT / "forward_e3_minus_10")
    parser.add_argument("--daily-dir", type=Path)
    parser.add_argument("--intraday-dir", type=Path)
    parser.add_argument("--model-threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize_forward_evaluation(
            args.screening_report, args.state_dir, args.daily_dir, args.intraday_dir
        )
    elif args.command == "run":
        result = run_forward_evaluation_once(args.screening_report, args.state_dir, args.model_threads)
    else:
        result = forward_status(args.state_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
