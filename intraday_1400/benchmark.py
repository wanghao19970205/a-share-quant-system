from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from intraday_1400 import config
from intraday_1400.collector import _query_min5, _read_codes
from intraday_1400.storage import atomic_json
from stock_analyzer import amazingdata_source as source


def run(codes: list[str], trade_date: str, sizes: list[int]) -> dict:
    """Benchmark sequentially in one process and one SDK login."""
    config.ensure_dirs()
    source.set_credentials()
    if not source._ensure_login():  # noqa: SLF001
        raise RuntimeError(f"AmazingData login failed: {source._last_error}")  # noqa: SLF001
    rows: list[dict] = []
    for size in sizes:
        selected = codes[:size]
        broker_codes = [source._to_broker_code(code) for code in selected]  # noqa: SLF001
        factor_started = time.perf_counter()
        factors = source._get_factor_frame(tuple(broker_codes))  # noqa: SLF001
        factor_seconds = time.perf_counter() - factor_started
        kline_started = time.perf_counter()
        raw = _query_min5(broker_codes, trade_date, trade_date)
        kline_seconds = time.perf_counter() - kline_started
        returned = sum(1 for code in broker_codes if isinstance(raw, dict) and raw.get(code) is not None)
        bars = sum(len(raw.get(code)) for code in broker_codes if isinstance(raw, dict) and raw.get(code) is not None)
        row = {
            "size": size,
            "factor_seconds": round(factor_seconds, 3),
            "kline_seconds": round(kline_seconds, 3),
            "total_seconds": round(factor_seconds + kline_seconds, 3),
            "returned": returned,
            "coverage": returned / max(size, 1),
            "bars": bars,
            "factor_ok": factors is not None and not factors.empty,
        }
        rows.append(row)
        print(f"[intraday1400:benchmark] {row}", flush=True)
    report = {
        "trade_date": trade_date,
        "sizes": rows,
        "single_login": True,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    atomic_json(report, config.REPORT_DIR / f"benchmark_{trade_date}.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-login min5 batch benchmark")
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--sizes", default="20,50,100,200")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if int(value) > 0]
    report = run(_read_codes(args.codes_file), args.trade_date, sizes)
    print(report, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
