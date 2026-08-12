"""Validation and atomic persistence for realtime-only active return weights."""
from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
from pathlib import Path
from typing import Optional

MODEL_COLUMNS = ("ridge_pred", "elastic_pred", "extra_trees_pred")


def normalize_weights(value) -> Optional[dict[str, float]]:
    if not isinstance(value, dict) or set(value) != set(MODEL_COLUMNS):
        return None
    cleaned: dict[str, float] = {}
    for column in MODEL_COLUMNS:
        try:
            number = float(value[column])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        cleaned[column] = number
    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {column: cleaned[column] / total for column in MODEL_COLUMNS}


def load_active_manifest(path: Path) -> Optional[dict]:
    source = Path(path)
    if not source.exists():
        return None
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            manifest.get("scope") != "realtime_return_weights" or
            manifest.get("state") != "active"):
        return None
    weights = normalize_weights(manifest.get("weights"))
    if weights is None or not manifest.get("version"):
        return None
    result = dict(manifest)
    result["weights"] = weights
    previous = manifest.get("previous")
    if isinstance(previous, dict):
        previous_weights = normalize_weights(previous.get("weights"))
        if previous_weights is not None:
            result["previous"] = {**previous, "weights": previous_weights}
        else:
            result.pop("previous", None)
    return result


def atomic_write_json(path: Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        temporary = target.with_suffix(
            target.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def append_history(path: Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
