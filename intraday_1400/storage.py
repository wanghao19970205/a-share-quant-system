from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False, row_group_size=4096)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_parquet_if_changed(frame: pd.DataFrame, path: Path) -> bool:
    """Write a Parquet file only when its logical tabular content changed."""
    if path.exists():
        old = pd.read_parquet(path)
        if list(old.columns) == list(frame.columns) and old.reset_index(drop=True).equals(
            frame.reset_index(drop=True)
        ):
            return False
    atomic_parquet(frame, path)
    return True


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def artifact_hash(path: Path) -> str:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    for candidate in files:
        relative = candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def upsert_month(frame: pd.DataFrame, directory: Path, key_columns: list[str]) -> list[Path]:
    if frame is None or frame.empty:
        return []
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    data = data.dropna(subset=["date"])
    written: list[Path] = []
    for month, part in data.groupby(data["date"].dt.strftime("%Y-%m"), sort=True):
        path = directory / f"{month}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        merged = pd.concat([old, part], ignore_index=True) if not old.empty else part
        merged = (
            merged.drop_duplicates(subset=key_columns, keep="last")
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        atomic_parquet(merged, path)
        written.append(path)
    return written


def write_partition_part(frame: pd.DataFrame, directory: Path, month: str, part_name: str) -> Path:
    path = directory / month / f"{part_name}.parquet"
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").astype("datetime64[ns]")
    data = data.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
    atomic_parquet(data, path)
    return path


def upsert_partition_part(
    frame: pd.DataFrame,
    directory: Path,
    month: str,
    part_name: str,
    key_columns: list[str],
) -> Path:
    path = directory / month / f"{part_name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([old, frame], ignore_index=True) if not old.empty else frame.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").astype("datetime64[ns]")
    merged = (
        merged.dropna(subset=["date"])
        .drop_duplicates(subset=key_columns, keep="last")
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    atomic_parquet_if_changed(merged, path)
    return path


def read_parts(directory: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    paths = sorted(directory.glob("????-??/*.parquet"))
    if not paths:
        paths = sorted(directory.glob("????-??.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype("datetime64[ns]")
    if start:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)
