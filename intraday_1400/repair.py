from __future__ import annotations

import argparse
import fcntl
import json
import os

from intraday_1400 import config
from intraday_1400.collector import _chunks, _read_codes, collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild complete 14:00 qfq history for changed factors")
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", required=True)
    parser.add_argument("--batch-size", type=int, default=800)
    args = parser.parse_args()
    config.ensure_dirs()
    rebuild_path = config.CHECKPOINT_DIR / "factor_rebuild_required.json"
    if not rebuild_path.exists():
        print("[intraday1400:repair] no factor rebuild required", flush=True)
        return
    payload = json.loads(rebuild_path.read_text(encoding="utf-8"))
    changed_codes = sorted(set(str(code)[:6] for code in payload.get("codes", [])))
    if not changed_codes:
        rebuild_path.unlink(missing_ok=True)
        print("[intraday1400:repair] empty rebuild list cleared", flush=True)
        return
    changed = set(changed_codes)
    universe = sorted(set(_read_codes(args.codes_file)))
    codes = [
        code
        for _, partition in _chunks(universe, config.PARTITION_SIZE)
        if changed.intersection(partition)
        for code in partition
    ]
    lock_handle = config.LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("intraday_1400 workflow already running") from exc
    stats = collect(
        codes,
        args.start,
        args.end,
        batch_size=args.batch_size,
        partition_size=config.PARTITION_SIZE,
        feature_workers=config.FEATURE_WORKERS,
        resume=False,
    )
    rebuild_path.unlink(missing_ok=True)
    print(
        f"[intraday1400:repair] changed={len(changed_codes)} "
        f"partition_codes={len(codes)} rows={stats.rows}",
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
