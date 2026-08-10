from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from intraday_1400 import config
from intraday_1400.fair_race_pipeline import _json_report
from intraday_1400.offline_race import (
    ExecutionConfig,
    common_prediction_universe,
    compare_execution_records,
    normalize_predictions,
    simulate_fixed_exit_race,
)
from intraday_1400.storage import atomic_json, atomic_parquet
from quant import config as quant_config


DUAL_PROTOCOL = "daily_actual_vs_intraday_e3_forward_v1"
ENTRY_CUTOFF = "14:50"
TOP_N = 10
ROUNDTRIP_COST = 0.002
UNSELLABLE_RETURN = -0.10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_daily_publication(
    active_predictions_path: Path,
    active_manifest_path: Path,
    ledger_dir: Path,
    captured_at: pd.Timestamp | None = None,
) -> dict:
    active_predictions_path = Path(active_predictions_path)
    active_manifest_path = Path(active_manifest_path)
    if not active_predictions_path.exists() or not active_manifest_path.exists():
        raise FileNotFoundError("active daily predictions and manifest are required")
    active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))
    published_at = pd.Timestamp(active_manifest.get("published_at"))
    if pd.isna(published_at):
        raise ValueError("active manifest published_at is required")
    predictions = normalize_predictions(pd.read_parquet(active_predictions_path))
    if predictions.empty:
        raise RuntimeError("active daily predictions are empty")
    signal_date = pd.Timestamp(predictions["date"].max()).normalize()
    publication_deadline = signal_date + pd.Timedelta(hours=14, minutes=50)
    if published_at > publication_deadline:
        raise RuntimeError(
            f"daily publication for {signal_date.date()} occurred after {ENTRY_CUTOFF}"
        )
    day = predictions[predictions["date"] == signal_date].reset_index(drop=True)
    if day.empty:
        raise RuntimeError("latest daily prediction date has no rows")
    captured_at = pd.Timestamp(captured_at or pd.Timestamp.now())
    ledger_dir = Path(ledger_dir)
    snapshot_path = ledger_dir / "predictions" / f"{signal_date:%Y-%m-%d}.parquet"
    metadata_path = ledger_dir / "metadata" / f"{signal_date:%Y-%m-%d}.json"
    metadata = {
        "signal_date": str(signal_date.date()),
        "published_at": published_at.isoformat(),
        "captured_at": captured_at.isoformat(),
        "publication_deadline": publication_deadline.isoformat(),
        "active_predictions_sha256": _sha256_file(active_predictions_path),
        "active_manifest_sha256": _sha256_file(active_manifest_path),
        "rows": int(len(day)),
        "source": str(active_predictions_path.resolve()),
    }
    if snapshot_path.exists() or metadata_path.exists():
        if not snapshot_path.exists() or not metadata_path.exists():
            raise RuntimeError("daily publication ledger is partially written")
        existing = normalize_predictions(pd.read_parquet(snapshot_path))
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not existing.equals(day.reset_index(drop=True)):
            raise RuntimeError(f"daily publication snapshot is immutable for {signal_date.date()}")
        return existing_metadata
    atomic_parquet(day, snapshot_path)
    atomic_json(metadata, metadata_path)
    return metadata


def initialize_dual_paper_evaluation(
    minute_state_dir: Path,
    state_dir: Path,
    active_predictions_path: Path | None = None,
    active_manifest_path: Path | None = None,
) -> dict:
    minute_state_dir = Path(minute_state_dir)
    minute_manifest_path = minute_state_dir / "manifest.json"
    if not minute_manifest_path.exists():
        raise RuntimeError("minute forward evaluation must be initialized first")
    minute_manifest = json.loads(minute_manifest_path.read_text(encoding="utf-8"))
    active_predictions_path = Path(
        active_predictions_path
        or Path(quant_config.QUANT_DIR) / "active_quant_short_predictions.parquet"
    )
    active_manifest_path = Path(
        active_manifest_path or Path(quant_config.QUANT_DIR) / "active_quant_model.json"
    )
    payload = {
        "protocol": DUAL_PROTOCOL,
        "cutoff_date": minute_manifest["cutoff_date"],
        "minute_immutable_sha256": minute_manifest["immutable_sha256"],
        "minute_state_dir": str(minute_state_dir.resolve()),
        "active_predictions_path": str(active_predictions_path.resolve()),
        "active_manifest_path": str(active_manifest_path.resolve()),
        "entry_time": ENTRY_CUTOFF,
        "top_n": TOP_N,
        "roundtrip_cost": ROUNDTRIP_COST,
        "unsellable_return": UNSELLABLE_RETURN,
        "main_race": "independent candidate universes; unbuyable selections remain cash; no backfill",
        "diagnostic_race": "common date-code universe only",
    }
    manifest = {
        **payload,
        "immutable_sha256": _canonical_hash(payload),
        "processed_dates": [],
        "status": "initialized_waiting_for_daily_and_minute_snapshots",
    }
    state_dir = Path(state_dir)
    path = state_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_dual_manifest(existing)
        return existing
    atomic_json(manifest, path)
    return manifest


def validate_dual_manifest(manifest: dict) -> None:
    mutable = {"immutable_sha256", "processed_dates", "status"}
    payload = {key: value for key, value in manifest.items() if key not in mutable}
    if manifest.get("immutable_sha256") != _canonical_hash(payload):
        raise RuntimeError("dual paper manifest immutable configuration was modified")
    minute_manifest_path = Path(manifest["minute_state_dir"]) / "manifest.json"
    if not minute_manifest_path.exists():
        raise RuntimeError("minute forward manifest is missing")
    minute_manifest = json.loads(minute_manifest_path.read_text(encoding="utf-8"))
    if minute_manifest.get("immutable_sha256") != manifest.get("minute_immutable_sha256"):
        raise RuntimeError("minute forward protocol drift detected")


def _labels_for_date(intraday_dir: Path, signal_date: pd.Timestamp) -> pd.DataFrame:
    path = Path(intraday_dir) / f"{signal_date:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"prepared labels missing for {signal_date.date()}")
    columns = [
        "code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[frame["date"] == signal_date].copy()


def evaluate_prediction_pair(
    daily_predictions: pd.DataFrame,
    minute_predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    execution = ExecutionConfig(
        top_n=TOP_N,
        entry_time=ENTRY_CUTOFF,
        roundtrip_cost=ROUNDTRIP_COST,
        unsellable_return=UNSELLABLE_RETURN,
    )
    main_records = []
    for name, frame in (
        ("daily_actual", daily_predictions),
        ("minute_e3_minus_10", minute_predictions),
    ):
        records, _ = simulate_fixed_exit_race({name: frame}, labels, execution)
        main_records.append(records)
    common = common_prediction_universe({
        "common_daily_actual": daily_predictions,
        "common_minute_e3_minus_10": minute_predictions,
    })
    diagnostic_records, _ = simulate_fixed_exit_race(common, labels, execution)
    return pd.concat(main_records, ignore_index=True), diagnostic_records


def _append_records(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh.copy()
    return (
        combined.drop_duplicates(["model", "signal_date", "code"], keep="last")
        .sort_values(["signal_date", "model", "rank", "code"])
        .reset_index(drop=True)
    )


def run_dual_paper_evaluation_once(state_dir: Path) -> dict:
    state_dir = Path(state_dir)
    manifest_path = state_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("dual paper evaluation is not initialized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_dual_manifest(manifest)
    minute_state_dir = Path(manifest["minute_state_dir"])
    minute_manifest = json.loads((minute_state_dir / "manifest.json").read_text(encoding="utf-8"))
    daily_ledger = state_dir / "daily_publications" / "predictions"
    minute_predictions_dir = minute_state_dir / "predictions"
    daily_dates = {path.stem for path in daily_ledger.glob("????-??-??.parquet")}
    minute_dates = {
        path.name.split("_forward_e3_minus_10.parquet")[0]
        for path in minute_predictions_dir.glob("????-??-??_forward_e3_minus_10.parquet")
    }
    processed = set(manifest.get("processed_dates", []))
    dates = sorted((daily_dates & minute_dates) - processed)
    if not dates:
        manifest["status"] = "waiting_for_paired_mature_dates"
        atomic_json(manifest, manifest_path)
        return {"status": manifest["status"], "processed_dates": sorted(processed)}
    fresh_main = []
    fresh_diagnostic = []
    for value in dates:
        signal_date = pd.Timestamp(value)
        daily = pd.read_parquet(daily_ledger / f"{value}.parquet")
        minute = pd.read_parquet(
            minute_predictions_dir / f"{value}_forward_e3_minus_10.parquet"
        )
        labels = _labels_for_date(Path(minute_manifest["intraday_prepared_dir"]), signal_date)
        main_records, diagnostic_records = evaluate_prediction_pair(daily, minute, labels)
        fresh_main.append(main_records)
        fresh_diagnostic.append(diagnostic_records)
    main_path = state_dir / "main_execution_records.parquet"
    diagnostic_path = state_dir / "common_execution_records.parquet"
    main = _append_records(main_path, pd.concat(fresh_main, ignore_index=True))
    diagnostic = _append_records(
        diagnostic_path, pd.concat(fresh_diagnostic, ignore_index=True)
    )
    main_comparison = compare_execution_records(main)
    diagnostic_comparison = compare_execution_records(diagnostic)
    completed_dates = sorted(processed | set(dates))
    report = {
        "protocol": DUAL_PROTOCOL,
        "cutoff_date": manifest["cutoff_date"],
        "processed_dates": completed_dates,
        "main_comparison": {
            key: value for key, value in main_comparison.items() if key != "daily_returns"
        },
        "common_universe_comparison": {
            key: value for key, value in diagnostic_comparison.items() if key != "daily_returns"
        },
    }
    atomic_parquet(main, main_path)
    atomic_parquet(diagnostic, diagnostic_path)
    atomic_parquet(main_comparison["daily_returns"], state_dir / "main_daily_returns.parquet")
    atomic_parquet(
        diagnostic_comparison["daily_returns"], state_dir / "common_daily_returns.parquet"
    )
    atomic_json(report, state_dir / "dual_paper_report.json")
    manifest["processed_dates"] = completed_dates
    manifest["status"] = "active"
    atomic_json(manifest, manifest_path)
    return report


def run_retrospective_bootstrap(
    active_predictions_path: Path,
    e3_experiment_dir: Path,
    intraday_prepared_dir: Path,
    output_dir: Path,
) -> dict:
    daily_all = normalize_predictions(pd.read_parquet(active_predictions_path))
    main_records = []
    diagnostic_records = []
    windows = []
    for prediction_path in sorted(
        Path(e3_experiment_dir).glob("window_*_exec_e3_exit_risk_10_predictions.parquet")
    ):
        window_index = int(prediction_path.name.split("_")[1])
        minute = normalize_predictions(pd.read_parquet(prediction_path))
        dates = pd.DatetimeIndex(minute["date"].drop_duplicates()).sort_values()
        daily = daily_all[daily_all["date"].isin(dates)].copy()
        label_parts = []
        for month in sorted(set(dates.strftime("%Y-%m"))):
            path = Path(intraday_prepared_dir) / f"{month}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"prepared labels missing: {path}")
            label_parts.append(pd.read_parquet(path, columns=[
                "code", "date", "entry_buyable", "target_net_ret_t1",
                "target_outcome_observed_t1",
            ]))
        labels = pd.concat(label_parts, ignore_index=True)
        labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
        labels = labels[labels["date"].isin(dates)].copy()
        if daily["date"].nunique() != len(dates):
            missing = dates.difference(pd.DatetimeIndex(daily["date"].unique()))
            raise RuntimeError(f"daily active history missing dates: {missing.tolist()}")
        main, diagnostic = evaluate_prediction_pair(daily, minute, labels)
        main["window"] = window_index
        diagnostic["window"] = window_index
        main_records.append(main)
        diagnostic_records.append(diagnostic)
        windows.append({
            "window": window_index,
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "days": int(len(dates)),
            "daily_rows": int(len(daily)),
            "minute_rows": int(len(minute)),
        })
    if not main_records:
        raise RuntimeError("no E3 prediction windows found")
    main = pd.concat(main_records, ignore_index=True)
    diagnostic = pd.concat(diagnostic_records, ignore_index=True)
    main_comparison = compare_execution_records(main)
    diagnostic_comparison = compare_execution_records(diagnostic)
    report = {
        "protocol": "retrospective_daily_active_vs_e3_bootstrap_v1",
        "causal_limit": (
            "daily active history has no row-level publication timestamps and may have been rewritten; "
            "this bootstrap cannot replace the frozen forward paper race"
        ),
        "windows": windows,
        "main_comparison": _json_report(main_comparison),
        "common_universe_comparison": _json_report(diagnostic_comparison),
    }
    output_dir = Path(output_dir)
    atomic_parquet(main, output_dir / "main_execution_records.parquet")
    atomic_parquet(diagnostic, output_dir / "common_execution_records.parquet")
    atomic_parquet(main_comparison["daily_returns"], output_dir / "main_daily_returns.parquet")
    atomic_parquet(
        diagnostic_comparison["daily_returns"], output_dir / "common_daily_returns.parquet"
    )
    atomic_json(report, output_dir / "retrospective_report.json")
    return report


def dual_status(state_dir: Path) -> dict:
    state_dir = Path(state_dir)
    path = state_dir / "manifest.json"
    if not path.exists():
        return {"status": "not_initialized"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ledger = state_dir / "daily_publications" / "predictions"
    return {
        "status": manifest.get("status"),
        "cutoff_date": manifest.get("cutoff_date"),
        "processed_dates": manifest.get("processed_dates", []),
        "daily_snapshot_dates": sorted(path.stem for path in ledger.glob("????-??-??.parquet")),
        "immutable_sha256": manifest.get("immutable_sha256"),
        "report_exists": (state_dir / "dual_paper_report.json").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward paper race: actual daily output versus intraday E3")
    parser.add_argument("command", choices=("init", "capture", "run", "status", "retrospective"))
    parser.add_argument("--minute-state-dir", type=Path, default=config.DATA_ROOT / "forward_e3_minus_10")
    parser.add_argument("--state-dir", type=Path, default=config.DATA_ROOT / "dual_paper_daily_vs_e3")
    parser.add_argument(
        "--active-predictions", type=Path,
        default=Path(quant_config.QUANT_DIR) / "active_quant_short_predictions.parquet",
    )
    parser.add_argument(
        "--active-manifest", type=Path,
        default=Path(quant_config.QUANT_DIR) / "active_quant_model.json",
    )
    parser.add_argument(
        "--e3-experiment-dir", type=Path,
        default=config.DATA_ROOT / "exit_risk_unfiltered",
    )
    parser.add_argument(
        "--intraday-prepared-dir", type=Path,
        default=config.PREPARED_DIR,
    )
    parser.add_argument(
        "--retrospective-output", type=Path,
        default=config.DATA_ROOT / "retrospective_daily_vs_e3",
    )
    args = parser.parse_args()
    if args.command == "init":
        result = initialize_dual_paper_evaluation(
            args.minute_state_dir,
            args.state_dir,
            args.active_predictions,
            args.active_manifest,
        )
    elif args.command == "capture":
        result = capture_daily_publication(
            args.active_predictions,
            args.active_manifest,
            args.state_dir / "daily_publications",
        )
    elif args.command == "run":
        result = run_dual_paper_evaluation_once(args.state_dir)
    elif args.command == "retrospective":
        result = run_retrospective_bootstrap(
            args.active_predictions,
            args.e3_experiment_dir,
            args.intraday_prepared_dir,
            args.retrospective_output,
        )
    else:
        result = dual_status(args.state_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
