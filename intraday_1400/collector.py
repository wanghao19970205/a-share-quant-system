from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.features import aggregate_many
from intraday_1400.storage import atomic_json, upsert_partition_part
from stock_analyzer import amazingdata_source as source


@dataclass
class FetchStats:
    requests: int = 0
    symbols_requested: int = 0
    symbols_returned: int = 0
    bars: int = 0
    rows: int = 0
    query_seconds: float = 0.0
    aggregate_seconds: float = 0.0
    write_seconds: float = 0.0
    seconds: float = 0.0


def _chunks(items: list[str], size: int):
    for offset in range(0, len(items), size):
        yield offset // size, items[offset:offset + size]


def _month_ranges(start: str, end: str) -> list[tuple[str, str, str]]:
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    ranges: list[tuple[str, str, str]] = []
    current = first.replace(day=1)
    while current <= last:
        month_end = current + pd.offsets.MonthEnd(0)
        part_start = max(first, current)
        part_end = min(last, month_end)
        ranges.append((current.strftime("%Y-%m"), part_start.strftime("%Y%m%d"), part_end.strftime("%Y%m%d")))
        current = current + pd.offsets.MonthBegin(1)
    return ranges


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "completed": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "completed": {}}


def _batch_key(codes: list[str]) -> str:
    digest = hashlib.sha1("\n".join(codes).encode("ascii")).hexdigest()[:12]
    return f"{codes[0]}-{codes[-1]}-{len(codes)}-{digest}"


def _factor_fingerprint(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if values.empty or not np.isfinite(values.iloc[-1]) or values.iloc[-1] == 0:
        return ""
    normalized = (values / values.iloc[-1]).round(12)
    change_points = normalized[normalized.ne(normalized.shift(1))]
    payload = "\n".join(
        f"{pd.Timestamp(index).date()}:{value:.12g}"
        for index, value in change_points.items()
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _query_min5(broker_codes: list[str], start: str, end: str) -> dict:
    import AmazingData as ad

    return source.sdk_call(
        source._market.query_kline,  # noqa: SLF001 - the adapter owns the only SDK session
        broker_codes,
        begin_date=int(start),
        end_date=int(end),
        period=ad.constant.Period.min5.value,
        timeout=config.MINUTE_TIMEOUT,
    )


def collect(
    codes: list[str],
    start: str,
    end: str,
    batch_size: int = config.MINUTE_BATCH_SIZE,
    partition_size: int = config.PARTITION_SIZE,
    feature_workers: int = config.FEATURE_WORKERS,
    resume: bool = True,
    codes_file_manifest: dict | None = None,
) -> FetchStats:
    """Collect in one SDK session; CPU workers only aggregate local DataFrames."""
    config.ensure_dirs()
    if not codes:
        return FetchStats()
    if int(batch_size) % int(partition_size) != 0:
        raise ValueError(
            f"batch_size={batch_size} must be a multiple of stable partition_size={partition_size}"
        )
    source.set_credentials()
    if not source._ensure_login():  # noqa: SLF001 - one explicit login for the whole workflow
        raise RuntimeError(f"AmazingData login failed: {source._last_error}")  # noqa: SLF001
    checkpoint_path = config.CHECKPOINT_DIR / "collect_global.json"
    checkpoint = _load_checkpoint(checkpoint_path) if resume else {"schema_version": 1, "completed": {}}
    if resume:
        # Merge earlier range-specific checkpoints without overwriting newer global entries.
        for previous_path in sorted(config.CHECKPOINT_DIR.glob("collect_*.json")):
            if previous_path == checkpoint_path:
                continue
            previous = _load_checkpoint(previous_path)
            for key, value in previous.get("completed", {}).items():
                checkpoint.setdefault("completed", {}).setdefault(key, value)
        if checkpoint.get("completed"):
            atomic_json(checkpoint, checkpoint_path)
    stats = FetchStats()
    factor_state_path = config.CHECKPOINT_DIR / "factor_fingerprints.json"
    # Factor continuity is independent of data checkpoint resume and must never be reset.
    factor_state = _load_checkpoint(factor_state_path)
    factor_state.pop("completed", None)
    factor_changes: set[str] = set()
    workflow_started = time.perf_counter()
    month_ranges = _month_ranges(start, end)
    batches = list(_chunks(sorted(set(str(code)[:6] for code in codes)), max(int(batch_size), 1)))
    for batch_index, batch_codes in batches:
        batch_key = _batch_key(batch_codes)
        broker_mapping = {code: source._to_broker_code(code) for code in batch_codes}  # noqa: SLF001
        broker_codes = list(broker_mapping.values())
        factor_started = time.perf_counter()
        factor_frame = source._get_factor_frame(tuple(broker_codes))  # noqa: SLF001
        if factor_frame is None or factor_frame.empty:
            raise RuntimeError(f"factor fetch failed for batch {batch_key}; refuse mixed raw/qfq output")
        pending_fingerprints: dict[str, str] = {}
        known_fingerprints = factor_state.setdefault("codes", {})
        for code, broker_code in broker_mapping.items():
            factor = source._factor_series(factor_frame, broker_code)  # noqa: SLF001
            fingerprint = _factor_fingerprint(factor) if factor is not None else ""
            if not fingerprint:
                raise RuntimeError(f"empty factor fingerprint for {broker_code}")
            previous_fingerprint = known_fingerprints.get(code)
            if previous_fingerprint and previous_fingerprint != fingerprint:
                factor_changes.add(code)
            pending_fingerprints[code] = fingerprint
        print(
            f"[intraday1400:factor] batch={batch_index + 1}/{len(batches)} "
            f"codes={len(batch_codes)} seconds={time.perf_counter() - factor_started:.1f}",
            flush=True,
        )
        for month_index, (month, part_start, part_end) in enumerate(month_ranges):
            checkpoint_key = f"{batch_key}:{month}"
            if resume and checkpoint.get("completed", {}).get(checkpoint_key):
                print(f"[intraday1400:resume] skip={checkpoint_key}", flush=True)
                continue
            request_started = time.perf_counter()
            raw = _query_min5(broker_codes, part_start, part_end)
            elapsed = time.perf_counter() - request_started
            if not isinstance(raw, dict):
                raise TypeError(f"minute query returned {type(raw).__name__}, expected dict")
            items: list[tuple[str, pd.DataFrame]] = []
            bars = 0
            for code, broker_code in broker_mapping.items():
                frame = source._normalize_kline(raw.get(broker_code))  # noqa: SLF001
                if frame is None or frame.empty:
                    continue
                factor = source._factor_series(factor_frame, broker_code)  # noqa: SLF001
                if factor is None or factor.empty:
                    raise RuntimeError(f"missing factor for {broker_code}; refuse mixed raw/qfq output")
                adjusted = source._apply_adjust(frame, factor, "qfq")  # noqa: SLF001
                if adjusted is None or adjusted.empty:
                    continue
                raw_close = pd.to_numeric(frame["close"], errors="coerce")
                adjusted_close = pd.to_numeric(adjusted["close"], errors="coerce")
                scale = adjusted_close / raw_close.replace(0, np.nan)
                raw_vwap = (
                    pd.to_numeric(frame.get("amount"), errors="coerce")
                    / pd.to_numeric(frame.get("volume"), errors="coerce").replace(0, np.nan)
                )
                adjusted["bar_vwap_qfq"] = raw_vwap * scale
                adjusted["factor_status"] = "ok"
                adjusted["factor_version"] = pending_fingerprints[code]
                items.append((code, adjusted))
                bars += len(adjusted)
            aggregate_started = time.perf_counter()
            aggregated = aggregate_many(items, workers=feature_workers, cutoff_time=config.CUTOFF_TIME)
            aggregate_elapsed = time.perf_counter() - aggregate_started
            write_started = time.perf_counter()
            if not aggregated.empty:
                aggregated["factor_status"] = "ok"
                label_columns = [
                    column for column in aggregated.columns
                    if column in {"code", "date", "schema_version", "factor_version"}
                    or column.startswith("label_")
                ]
                for _, partition_codes in _chunks(batch_codes, max(int(partition_size), 1)):
                    part_key = _batch_key(partition_codes)
                    part = aggregated[aggregated["code"].isin(partition_codes)].copy()
                    if part.empty:
                        continue
                    upsert_partition_part(
                        part[label_columns],
                        config.LABEL_DIR,
                        month,
                        part_key,
                        ["code", "date", "schema_version"],
                    )
                    feature_frame = part.drop(columns=[
                        column for column in part.columns if column.startswith("label_")
                    ])
                    upsert_partition_part(
                        feature_frame,
                        config.ASOF_PRICE_DIR,
                        month,
                        part_key,
                        ["code", "date", "asof_time", "schema_version"],
                    )
            write_elapsed = time.perf_counter() - write_started
            stats.requests += 1
            stats.symbols_requested += len(batch_codes)
            stats.symbols_returned += len(items)
            stats.bars += bars
            stats.rows += len(aggregated)
            stats.query_seconds += elapsed
            stats.aggregate_seconds += aggregate_elapsed
            stats.write_seconds += write_elapsed
            checkpoint.setdefault("completed", {})[checkpoint_key] = {
                "codes": len(batch_codes),
                "symbols_returned": len(items),
                "bars": bars,
                "rows": len(aggregated),
                "seconds": round(elapsed, 3),
                "completed_at": pd.Timestamp.now().isoformat(),
            }
            checkpoint["updated_at"] = pd.Timestamp.now().isoformat()
            atomic_json(checkpoint, checkpoint_path)
            print(
                f"[intraday1400:kline] batch={batch_index + 1}/{len(batches)} "
                f"month={month_index + 1}/{len(month_ranges)} period={month} "
                f"codes={len(items)}/{len(batch_codes)} bars={bars} rows={len(aggregated)} "
                f"query={elapsed:.1f}s aggregate={aggregate_elapsed:.1f}s write={write_elapsed:.1f}s",
                flush=True,
            )
        known_fingerprints.update(pending_fingerprints)
        factor_state["updated_at"] = pd.Timestamp.now().isoformat()
        atomic_json(factor_state, factor_state_path)
    if factor_changes:
        rebuild_path = config.CHECKPOINT_DIR / "factor_rebuild_required.json"
        previous_rebuild = _load_checkpoint(rebuild_path)
        codes_to_rebuild = sorted(set(previous_rebuild.get("codes", [])) | factor_changes)
        atomic_json({
            "codes": codes_to_rebuild,
            "detected_at": pd.Timestamp.now().isoformat(),
            "reason": "qfq change points changed; rebuild each code's complete 14:00 history before publish",
        }, rebuild_path)
        print(f"[intraday1400:factor] rebuild_required={len(codes_to_rebuild)}", flush=True)
    stats.seconds = time.perf_counter() - workflow_started
    import resource
    report = {
        **stats.__dict__,
        "peak_rss_native": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "schema_version": config.SCHEMA_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "start": start,
        "end": end,
        "batch_size": batch_size,
        "partition_size": partition_size,
        "feature_workers": feature_workers,
        "single_sdk_login": True,
        "factor_changes": sorted(factor_changes),
        "codes_file": codes_file_manifest,
        "finished_at": pd.Timestamp.now().isoformat(),
    }
    atomic_json(report, config.REPORT_DIR / f"collect_{start}_{end}.json")
    return stats


def _read_codes(path: str, limit: int = 0) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    import re

    codes = list(dict.fromkeys(re.findall(r"(?<!\d)(\d{6})(?!\d)", text)))
    return codes[:limit] if limit else codes


def _codes_file_manifest(path: str, codes: list[str], limit: int = 0) -> dict:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"codes file unavailable: {source_path}")
    raw = source_path.read_bytes()
    all_codes = _read_codes(str(source_path))
    if not all_codes or not codes:
        raise RuntimeError(f"codes file contains no valid six-digit codes: {source_path}")
    return {
        "path": str(source_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "source_code_count": len(all_codes),
        "effective_code_count": len(codes),
        "limit": int(limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-login AmazingData min5 collector for 14:00 model")
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=config.MINUTE_BATCH_SIZE)
    parser.add_argument("--partition-size", type=int, default=config.PARTITION_SIZE)
    parser.add_argument("--feature-workers", type=int, default=config.FEATURE_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    lock_handle = config.LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("intraday_1400 workflow already running") from exc
    codes = _read_codes(args.codes_file, args.limit)
    codes_manifest = _codes_file_manifest(args.codes_file, codes, args.limit)
    stats = collect(
        codes,
        args.start,
        args.end,
        batch_size=args.batch_size,
        partition_size=args.partition_size,
        feature_workers=args.feature_workers,
        resume=not args.no_resume,
        codes_file_manifest=codes_manifest,
    )
    print(f"[intraday1400:done] {stats}", flush=True)
    # Avoid native SDK teardown crashes after all atomic writes and log flushes complete.
    os._exit(0)


if __name__ == "__main__":
    main()
