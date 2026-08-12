"""自动纯日更优化流水线。

每轮只运行一个注册分支；评估失败后按固定顺序进入下一分支。所有分支
使用 next-open、Top2、无递补和固定现金仓位，研究结果不会发布到 active。
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from quant import config
from intraday_1400.evaluation import _paired_block_bootstrap
from intraday_1400.storage import artifact_hash, atomic_json

PROTOCOL = "daily_optimization_pipeline_v2"
TOP_N = 2
COST = 0.002
MIN_FILLED = 1.0
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SEED = 42

BRANCHES = {
    "open_buyin_ridge": {
        "target_mode": "open-buyin-mask",
        "lgbm_weight": 0.0,
        "ic_weight": 0.0,
        "positive_only": False,
        "description": "Ridge-only open target with next-open buyability training mask",
    },
    "open_ridge": {
        "target_mode": "open-label",
        "lgbm_weight": 0.0,
        "ic_weight": 0.0,
        "positive_only": False,
        "description": "Ridge-only next-open target without buyability mask",
    },
    "open_regularized_lgbm": {
        "target_mode": "open-buyin-mask",
        "lgbm_weight": 0.20,
        "ic_weight": 0.0,
        "positive_only": False,
        "description": "Ridge 80% plus constrained LightGBM 20%",
    },
    "open_extratrees": {
        "target_mode": "open-buyin-mask",
        "lgbm_weight": 0.0,
        "ic_weight": 0.0,
        "positive_only": False,
        "extra_trees": True,
        "extra_trees_weight": 0.20,
        "description": "Ridge baseline with a low-weight ExtraTrees nonlinear leg",
    },
    "open_random_forest": {
        "target_mode": "open-buyin-mask",
        "lgbm_weight": 0.0,
        "ic_weight": 0.0,
        "positive_only": False,
        "random_forest": True,
        "random_forest_weight": 0.20,
        "description": "Replace the LightGBM allocation with RandomForest 20%",
    },
}
BRANCH_ORDER = tuple(BRANCHES)


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "top_n": TOP_N,
        "cost_roundtrip": COST,
        "minimum_filled_names": MIN_FILLED,
        "branches": BRANCHES,
        "selection": {
            "paired_baseline_required": True,
            "paired_ci95_lower_gt_zero": True,
            "bootstrap_method": "circular_moving_block",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_filled_names": MIN_FILLED,
            "max_drawdown_floor": -0.60,
            "minimum_positive_months": 3,
        },
        "production_publication": False,
        "human_approval_required": True,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _file_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": artifact_hash(path),
    }


def _directory_signature(path: Path, pattern: str = "*.parquet") -> dict[str, Any]:
    path = Path(path).resolve()
    files = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    entries = [
        (candidate.relative_to(path).as_posix(), int(candidate.stat().st_size), int(candidate.stat().st_mtime_ns))
        for candidate in files
    ]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "files": len(entries),
        "signature": _canonical_hash(entries),
    }


def _input_snapshot() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    quant_root = Path(config.QUANT_DIR).resolve()
    return {
        "controller": _file_identity(Path(__file__)),
        "full_train_batched": _file_identity(project_root / "quant" / "full_train_batched.py"),
        "model": _file_identity(project_root / "quant" / "model.py"),
        "engineering": _file_identity(project_root / "quant" / "factors" / "engineering.py"),
        "universe": _file_identity(Path(config.MAINBOARD_UNIVERSE_FILE)),
        "selection": _file_identity(quant_root / "factor_selection_lh1000_cont.parquet"),
        "prepared_monthly": _directory_signature(
            quant_root / "factor_panel_mainboard_active_h1_parts" / "prepared_monthly"
        ),
        "price": _directory_signature(Path(config.PRICE_DIR)),
    }


def _execution_environment() -> dict[str, str]:
    return {
        "SCHEDULER_DISABLED": "1",
        "QUANT_BT_FILL": "next_open",
        "QUANT_BT_FILTER_UNTRADABLE": "1",
        "QUANT_BT_COST_ROUNDTRIP": str(COST),
    }


def _with_state_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("state_hash", None)
    result["state_hash"] = _canonical_hash(result)
    return result


def _persist_attempt(value: dict[str, Any], path: Path) -> dict[str, Any]:
    persisted = _with_state_hash(value)
    atomic_json(persisted, path)
    return persisted


def validate_attempt_manifest(
    manifest: dict[str, Any],
    *,
    verify_current_inputs: bool = False,
    verify_artifact: bool = True,
) -> None:
    stored_hash = manifest.get("state_hash")
    unsigned = dict(manifest)
    unsigned.pop("state_hash", None)
    if stored_hash != _canonical_hash(unsigned):
        raise RuntimeError("daily optimization attempt state hash mismatch")
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("daily optimization protocol mismatch")
    if manifest.get("production_publication") is not False or manifest.get("human_approval_required") is not True:
        raise RuntimeError("daily optimization production isolation changed")
    if manifest.get("controller_sha256") != artifact_hash(Path(__file__)):
        raise RuntimeError("daily optimization controller source changed")
    if manifest.get("command_hash") != _canonical_hash(manifest.get("command")):
        raise RuntimeError("daily optimization command hash mismatch")
    if manifest.get("execution_environment") != _execution_environment():
        raise RuntimeError("daily optimization execution environment mismatch")
    if verify_current_inputs and manifest.get("input_snapshot") != _input_snapshot():
        raise RuntimeError("daily optimization inputs changed")
    metrics = manifest.get("metrics")
    if verify_artifact and metrics:
        prediction_path = Path(metrics["prediction_path"])
        if not prediction_path.is_file() or artifact_hash(prediction_path) != metrics.get("prediction_sha256"):
            raise RuntimeError("daily optimization prediction artifact changed")
        if manifest.get("decision") != choose_next_branch(
            str(manifest["branch"]),
            {"metrics": metrics},
            manifest.get("baseline_metrics"),
        ):
            raise RuntimeError("daily optimization decision does not match persisted metrics")


@contextmanager
def _cycle_lock(research_root: Path) -> Iterator[None]:
    state_dir = Path(research_root) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".cycle.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another daily optimization pipeline is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _metrics(prediction_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(prediction_path)
    required = {"date", "code", "pred", "open_ret_1d", "buyable_next"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction artifact missing open execution columns: {missing}")
    if frame.empty:
        raise ValueError("prediction artifact is empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["pred"] = pd.to_numeric(frame["pred"], errors="coerce")
    frame["open_ret_1d"] = pd.to_numeric(frame["open_ret_1d"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("prediction artifact contains invalid dates")
    if not frame["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("prediction artifact contains invalid stock codes")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("prediction artifact contains duplicate date/code rows")
    if not np.isfinite(frame["pred"].to_numpy(dtype=float)).all():
        raise ValueError("prediction artifact contains non-finite predictions")
    observed_returns = frame["open_ret_1d"].dropna().to_numpy(dtype=float)
    if not np.isfinite(observed_returns).all():
        raise ValueError("prediction artifact contains non-finite open returns")
    if not pd.api.types.is_bool_dtype(frame["buyable_next"].dtype):
        raise ValueError("prediction artifact buyable_next must be boolean")
    candidates_per_day = frame.groupby("date").size()
    if (candidates_per_day < TOP_N).any():
        raise ValueError(f"prediction artifact has fewer than Top{TOP_N} candidates on a date")
    selected = (
        frame.sort_values(["date", "pred", "code"], ascending=[True, False, True])
        .groupby("date", group_keys=False)
        .head(TOP_N)
        .copy()
    )
    selected["filled"] = selected["buyable_next"].fillna(False).astype(bool)
    selected["slot_ret"] = np.where(
        selected["filled"], selected["open_ret_1d"].fillna(-0.10) - COST, 0.0
    )
    daily = selected.groupby("date")["slot_ret"].sum().mul(1.0 / TOP_N)
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    monthly = (1.0 + daily).groupby(daily.index.to_period("M")).prod() - 1.0
    return {
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": artifact_hash(prediction_path),
        "days": int(len(daily)),
        "mean_return": float(daily.mean()),
        "compound_return": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "mean_filled_names": float(selected.groupby("date")["filled"].sum().mean()),
        "fill_rate": float(selected["filled"].mean()),
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
        "daily_returns": [
            {"date": str(pd.Timestamp(date).date()), "return": float(value)}
            for date, value in daily.items()
        ],
    }


def _daily_return_series(metrics: dict[str, Any]) -> pd.Series:
    rows = metrics.get("daily_returns")
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics missing daily returns for paired baseline comparison")
    frame = pd.DataFrame(rows)
    if set(frame.columns) != {"date", "return"}:
        raise ValueError("daily returns must contain only date and return")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    if (
        frame["date"].isna().any()
        or frame["date"].duplicated().any()
        or not np.isfinite(frame["return"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("daily returns contain invalid or duplicate observations")
    return frame.set_index("date")["return"].sort_index()


def choose_next_branch(
    current: str,
    result: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = result.get("metrics", result)
    if baseline_metrics is None:
        raise ValueError("paired incumbent baseline metrics are required")
    comparison = _paired_block_bootstrap(
        _daily_return_series(metrics),
        _daily_return_series(baseline_metrics),
        samples=BOOTSTRAP_SAMPLES,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )
    ci95 = comparison.get("ci95")
    passed = (
        bool(comparison.get("available"))
        and isinstance(ci95, list)
        and len(ci95) == 2
        and float(ci95[0]) > 0.0
        and float(metrics.get("mean_filled_names", 0.0)) >= MIN_FILLED
        and float(metrics.get("max_drawdown", -1.0)) >= -0.60
        and int(metrics.get("positive_months", 0)) >= 3
    )
    if passed:
        return {
            "status": "candidate_requires_independent_reproduction",
            "selected": current,
            "next_branch": "independent_reproduction",
            "reason": "candidate passed paired incumbent and development gates",
            "paired_incumbent_comparison": comparison,
        }
    index = BRANCH_ORDER.index(current)
    if index + 1 < len(BRANCH_ORDER):
        return {
            "status": "branch_failed",
            "selected": "daily_baseline_retained",
            "next_branch": BRANCH_ORDER[index + 1],
            "reason": "candidate failed paired incumbent or development gates",
            "paired_incumbent_comparison": comparison,
        }
    return {
        "status": "all_registered_daily_branches_failed",
        "selected": "daily_baseline_retained",
        "next_branch": "human_review_required",
        "reason": "registered pure-daily branch order exhausted",
        "paired_incumbent_comparison": comparison,
    }


def _train_command(branch: str, output_prefix: str, cache_dir: Path, recent_windows: int) -> list[str]:
    recipe = BRANCHES[branch]
    command = [
        sys.executable,
        "-m",
        "quant.full_train_batched",
        "--name",
        "factor_panel_mainboard_active_h1",
        "--selection",
        "factor_selection_lh1000_cont",
        "--output-prefix",
        output_prefix,
        "--horizon",
        "1",
        "--refresh-months",
        "0",
        "--universe-file",
        config.MAINBOARD_UNIVERSE_FILE,
        "--top-n",
        str(TOP_N),
        "--n-estimators",
        "200",
        "--learning-rate",
        "0.015",
        "--early-stopping-rounds",
        "40",
        "--model-threads",
        "4",
        "--ridge-quantile",
        "0.7",
        "--lgbm-weight",
        str(recipe["lgbm_weight"]),
        "--ic-weight",
        str(recipe["ic_weight"]),
        "--rank-vote-weight",
        "0",
        "--decay-half-life-days",
        "60",
        "--min-weight",
        "0.03",
        "--train-months",
        "36",
        "--recent-windows",
        str(recent_windows),
        "--rolling-factor-select",
        "--rolling-top-factors",
        "30",
        "--max-factor-ic-corr",
        "0.85",
        "--purge-horizon",
        "--train-target-mode",
        recipe["target_mode"],
        "--window-cache-dir",
        str(cache_dir),
    ]
    if recipe.get("positive_only"):
        command.append("--positive-only")
    if recipe.get("extra_trees"):
        command.extend([
            "--extra-trees",
            "--extra-trees-estimators", "120",
            "--extra-trees-max-train-rows", "300000",
            "--extra-trees-weight", str(recipe.get("extra_trees_weight", 0.0)),
        ])
    if recipe.get("random_forest"):
        command.extend([
            "--random-forest",
            "--random-forest-estimators", "120",
            "--random-forest-max-train-rows", "300000",
            "--random-forest-weight", str(recipe.get("random_forest_weight", 0.0)),
        ])
    return command


def _new_run_id(branch: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{branch}_{uuid.uuid4().hex[:8]}"


def _prediction_path(output_prefix: str) -> Path:
    return Path(config.QUANT_DIR) / f"{output_prefix}_bt_ridge_lightgbm_ranker_ensemble_predictions.parquet"


def _write_log(path: Path, value: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value or "", encoding="utf-8")


def _retryable_failure(failure: dict[str, Any]) -> bool:
    if failure.get("kind") in {"timeout", "os_error"}:
        return True
    return int(failure.get("returncode", 0)) in {-15, -9, 137, 143}


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    on_start: Any = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if on_start is not None:
        on_start(int(process.pid))
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout if stdout is not None else error.stdout,
            stderr=stderr if stderr is not None else error.stderr,
        ) from error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_once(
    branch: str,
    research_root: Path,
    recent_windows: int = 12,
    execute: bool = True,
    timeout_seconds: int = 8 * 60 * 60,
    max_retries: int = 1,
    baseline_prediction_path: Path | None = None,
) -> dict[str, Any]:
    if branch not in BRANCHES:
        raise ValueError(f"unregistered daily branch: {branch}")
    if recent_windows <= 0 or timeout_seconds <= 0 or max_retries < 0:
        raise ValueError("recent_windows and timeout_seconds must be positive; max_retries must be non-negative")

    research_root = Path(research_root).resolve()
    run_id = _new_run_id(branch)
    output_prefix = f"daily_auto_{branch}_{run_id.split('_', 1)[0]}_{run_id.rsplit('_', 1)[-1]}"
    output_dir = research_root / "attempts" / run_id
    cache_dir = research_root / "window_cache" / run_id
    manifest_path = output_dir / "attempt.json"
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=False)

    prediction_path = _prediction_path(output_prefix)
    if prediction_path.exists():
        raise RuntimeError(f"unique prediction artifact already exists: {prediction_path}")
    command = _train_command(branch, output_prefix, cache_dir, recent_windows)
    execution_environment = _execution_environment()
    input_snapshot = _input_snapshot()
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "run_id": run_id,
        "branch": branch,
        "output_prefix": output_prefix,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "prediction_path": str(prediction_path.resolve()),
        "baseline_prediction": (
            _file_identity(Path(baseline_prediction_path))
            if baseline_prediction_path is not None
            else None
        ),
        "command": command,
        "command_hash": _canonical_hash(command),
        "controller_sha256": artifact_hash(Path(__file__)),
        "input_snapshot": input_snapshot,
        "execution_environment": execution_environment,
        "started_at": _utc_now(),
        "status": "running" if execute else "planned",
        "execution_attempts": [],
        "production_publication": False,
        "human_approval_required": True,
    }
    result = _persist_attempt(result, manifest_path)
    if not execute:
        return result

    started_ns = time.time_ns()
    project_root = Path(__file__).resolve().parent.parent
    failure: dict[str, Any] | None = None
    for attempt_number in range(1, max_retries + 2):
        attempt_started = _utc_now()
        try:
            def record_process(pid: int) -> None:
                result["pid"] = pid
                result["process_group_id"] = pid
                _persist_attempt(result, manifest_path)

            completed = _run_subprocess(
                command,
                cwd=project_root,
                env={**os.environ, **execution_environment},
                timeout=timeout_seconds,
                on_start=record_process,
            )
            stdout_path = output_dir / f"stdout.{attempt_number}.log"
            stderr_path = output_dir / f"stderr.{attempt_number}.log"
            _write_log(stdout_path, completed.stdout)
            _write_log(stderr_path, completed.stderr)
            execution_record = {
                "attempt": attempt_number,
                "started_at": attempt_started,
                "finished_at": _utc_now(),
                "returncode": int(completed.returncode),
                "stdout": _file_identity(stdout_path),
                "stderr": _file_identity(stderr_path),
            }
            result["execution_attempts"].append(execution_record)
            if completed.returncode == 0:
                failure = None
                result.pop("last_failure", None)
                break
            failure = {
                "kind": "returncode",
                "returncode": int(completed.returncode),
                "message": f"training command exited with {completed.returncode}",
            }
        except subprocess.TimeoutExpired as error:
            stdout_path = output_dir / f"stdout.{attempt_number}.log"
            stderr_path = output_dir / f"stderr.{attempt_number}.log"
            _write_log(stdout_path, error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout)
            _write_log(stderr_path, error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr)
            failure = {
                "kind": "timeout",
                "message": f"training command exceeded {timeout_seconds} seconds",
            }
            result["execution_attempts"].append({
                "attempt": attempt_number,
                "started_at": attempt_started,
                "finished_at": _utc_now(),
                **failure,
                "stdout": _file_identity(stdout_path),
                "stderr": _file_identity(stderr_path),
            })
        except OSError as error:
            failure = {"kind": "os_error", "message": str(error)}
            result["execution_attempts"].append({
                "attempt": attempt_number,
                "started_at": attempt_started,
                "finished_at": _utc_now(),
                **failure,
            })

        result["last_failure"] = failure
        result = _persist_attempt(result, manifest_path)
        if attempt_number > max_retries or not _retryable_failure(failure):
            break

    if failure is not None:
        result["status"] = "branch_execution_failed"
        result["finished_at"] = _utc_now()
        result["last_failure"] = failure
        return _persist_attempt(result, manifest_path)

    after_snapshot = _input_snapshot()
    if after_snapshot != input_snapshot:
        result["status"] = "input_drift_detected"
        result["finished_at"] = _utc_now()
        result["after_input_snapshot"] = after_snapshot
        return _persist_attempt(result, manifest_path)
    if not prediction_path.is_file():
        result["status"] = "branch_artifact_missing"
        result["finished_at"] = _utc_now()
        return _persist_attempt(result, manifest_path)
    if prediction_path.stat().st_mtime_ns < started_ns:
        result["status"] = "stale_prediction_artifact"
        result["finished_at"] = _utc_now()
        return _persist_attempt(result, manifest_path)

    try:
        if baseline_prediction_path is None:
            raise ValueError("paired incumbent baseline prediction artifact is required")
        baseline_path = Path(baseline_prediction_path).resolve()
        if not baseline_path.is_file():
            raise ValueError(
                f"incumbent baseline prediction artifact unavailable: {baseline_path}"
            )
        if _file_identity(baseline_path) != result["baseline_prediction"]:
            raise RuntimeError("incumbent baseline prediction artifact changed")
        result["metrics"] = _metrics(prediction_path)
        result["baseline_metrics"] = _metrics(baseline_path)
        result["decision"] = choose_next_branch(
            branch, result, result["baseline_metrics"],
        )
        result["status"] = "evaluated"
    except Exception as error:  # Persist malformed artifacts instead of losing the attempt.
        result["status"] = "artifact_validation_failed"
        result["artifact_error"] = {"type": type(error).__name__, "message": str(error)}
    result["finished_at"] = _utc_now()
    return _persist_attempt(result, manifest_path)


def _recover_interrupted_attempts(research_root: Path) -> list[str]:
    recovered: list[str] = []
    for path in sorted((Path(research_root) / "attempts").glob("*/attempt.json")):
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_attempt_manifest(manifest, verify_artifact=False)
        if manifest.get("status") == "running":
            process_group_id = manifest.get("process_group_id")
            if process_group_id:
                try:
                    os.killpg(int(process_group_id), 0)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    raise RuntimeError("cannot verify whether interrupted training process ended") from error
                else:
                    raise RuntimeError(
                        f"previous daily training process group is still alive: {process_group_id}"
                    )
            manifest["status"] = "interrupted"
            manifest["finished_at"] = _utc_now()
            manifest["last_failure"] = {
                "kind": "stale_running_state",
                "message": "previous controller stopped before completing this attempt",
            }
            _persist_attempt(manifest, path)
            recovered.append(str(path.resolve()))
    return recovered


def _load_pipeline_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    unsigned = dict(state)
    stored_hash = unsigned.pop("state_hash", None)
    if stored_hash != _canonical_hash(unsigned):
        raise RuntimeError("daily optimization pipeline state hash mismatch")
    if state.get("protocol") != PROTOCOL or state.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("daily optimization pipeline protocol mismatch")
    if state.get("production_publication") is not False or state.get("human_approval_required") is not True:
        raise RuntimeError("daily optimization pipeline production isolation changed")
    for entry in state.get("attempts", []):
        attempt_path = Path(entry["manifest_path"])
        if not attempt_path.is_file() or artifact_hash(attempt_path) != entry.get("manifest_sha256"):
            raise RuntimeError("daily optimization pipeline attempt manifest changed")
        with attempt_path.open("r", encoding="utf-8") as handle:
            validate_attempt_manifest(json.load(handle), verify_artifact=False)
    return state


def run_pipeline(
    initial_branch: str,
    research_root: Path,
    recent_windows: int = 12,
    execute: bool = True,
    timeout_seconds: int = 8 * 60 * 60,
    max_retries: int = 1,
    baseline_prediction_path: Path | None = None,
) -> dict[str, Any]:
    if initial_branch not in BRANCHES:
        raise ValueError(f"unregistered daily branch: {initial_branch}")
    research_root = Path(research_root).resolve()
    research_root.mkdir(parents=True, exist_ok=True)
    pipeline_path = research_root / "state" / "pipeline.json"
    with _cycle_lock(research_root):
        existing = _load_pipeline_state(pipeline_path)
        if existing and existing.get("status") in {
            "candidate_requires_independent_reproduction",
            "all_registered_daily_branches_failed",
            "human_review_required",
            "pipeline_blocked",
        }:
            return existing
        recovered = _recover_interrupted_attempts(research_root)
        if existing and existing.get("status") == "running" and recovered:
            existing["status"] = "pipeline_blocked"
            existing["reason"] = "an interrupted subprocess requires explicit operator recovery"
            for entry in existing.get("attempts", []):
                attempt_path = Path(entry["manifest_path"])
                if attempt_path.is_file():
                    entry["manifest_sha256"] = artifact_hash(attempt_path)
            existing["recovered_attempts"] = recovered
            existing["finished_at"] = _utc_now()
            return _persist_attempt(existing, pipeline_path)
        if existing and existing.get("status") == "running":
            attempts = existing.get("attempts", [])
            if not attempts:
                raise RuntimeError("running pipeline state has no persisted attempt")
            last_manifest_path = Path(attempts[-1]["manifest_path"])
            with last_manifest_path.open("r", encoding="utf-8") as handle:
                last_result = json.load(handle)
            if last_result.get("status") != "evaluated":
                raise RuntimeError("running pipeline state does not have a completed evaluable attempt")
            last_decision = last_result.get("decision", {})
            if last_decision.get("status") != "branch_failed":
                return existing
            branch = str(last_decision["next_branch"])
            pipeline_state = existing
            pipeline_state["resumed_at"] = _utc_now()
        else:
            branch = initial_branch
            pipeline_state = {
                "protocol": PROTOCOL,
                "protocol_hash": _canonical_hash(protocol_payload()),
                "started_at": _utc_now(),
                "initial_branch": initial_branch,
                "attempts": [],
                "recovered_attempts": recovered,
                "production_publication": False,
                "human_approval_required": True,
            }
        pipeline_state["status"] = "running" if execute else "planned"
        pipeline_state = _persist_attempt(pipeline_state, pipeline_path)
        while branch in BRANCHES:
            result = run_once(
                branch,
                research_root,
                recent_windows=recent_windows,
                execute=execute,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                baseline_prediction_path=baseline_prediction_path,
            )
            attempt_path = Path(result["output_dir"]) / "attempt.json"
            pipeline_state["attempts"].append({
                "branch": branch,
                "run_id": result["run_id"],
                "status": result["status"],
                "manifest_path": str(attempt_path.resolve()),
                "manifest_sha256": artifact_hash(attempt_path),
            })
            if not execute:
                pipeline_state["status"] = "planned"
                break
            decision = result.get("decision")
            if result.get("status") != "evaluated" or not isinstance(decision, dict):
                pipeline_state["status"] = "pipeline_blocked"
                pipeline_state["reason"] = result.get("status")
                pipeline_state = _persist_attempt(pipeline_state, pipeline_path)
                break
            if decision.get("status") != "branch_failed":
                pipeline_state["status"] = str(decision["status"])
                pipeline_state["selected"] = decision.get("selected")
                pipeline_state["next_branch"] = decision.get("next_branch")
                pipeline_state = _persist_attempt(pipeline_state, pipeline_path)
                break
            branch = str(decision["next_branch"])
            pipeline_state["next_branch"] = branch
            pipeline_state = _persist_attempt(pipeline_state, pipeline_path)
        else:
            pipeline_state["status"] = "pipeline_blocked"
            pipeline_state["reason"] = "controller routed outside the registered branch set"
        pipeline_state["finished_at"] = _utc_now()
        return _persist_attempt(pipeline_state, pipeline_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic pure-daily optimization pipeline")
    parser.add_argument("--branch", choices=BRANCH_ORDER, default=BRANCH_ORDER[0])
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-predictions", type=Path, required=True,
        help="frozen incumbent prediction artifact for paired common-date comparison",
    )
    parser.add_argument("--recent-windows", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=8 * 60 * 60)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        args.branch,
        args.research_root,
        recent_windows=args.recent_windows,
        execute=not args.dry_run,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        baseline_prediction_path=args.baseline_predictions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("status") == "pipeline_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
