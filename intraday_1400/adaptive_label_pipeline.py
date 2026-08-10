from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.adaptive_exit_replay import (
    REPLAY_CONFIG,
    build_fetch_plan,
    fetch_execution_bars,
    load_trading_calendar,
    replay_selected_trades,
)
from intraday_1400.storage import atomic_json, atomic_parquet


ADAPTIVE_LABEL_RECIPE_VERSION = 1
ADAPTIVE_LABEL_PROTOCOL = "adaptive_t3_liquidation_labels_v1"


def deterministic_code_sample(codes, size: int) -> list[str]:
    unique = sorted(set(str(code)[:6] for code in codes))
    ranked = sorted(
        unique,
        key=lambda code: (hashlib.sha256(code.encode("ascii")).hexdigest(), code),
    )
    return ranked[:max(int(size), 0)]


def load_pilot_universe(
    prepared_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sample_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    months = pd.period_range(start=start, end=end, freq="M").astype(str)
    parts = []
    columns = ["code", "date", "signal_eligible", "entry_buyable", "target_exit_sellable_t1"]
    for month in months:
        path = Path(prepared_dir) / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"prepared month missing: {path}")
        parts.append(pd.read_parquet(path, columns=columns))
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["code"] = panel["code"].astype(str).str[:6]
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    selected_codes = deterministic_code_sample(panel["code"].unique(), sample_size)
    eligible = panel[
        panel["code"].isin(selected_codes)
        & panel["signal_eligible"].fillna(False).astype(bool)
    ].copy()
    return eligible.sort_values(["date", "code"]).reset_index(drop=True), selected_codes


def build_label_trades(eligible: pd.DataFrame) -> pd.DataFrame:
    trades = eligible[["code", "date"]].rename(columns={"date": "signal_date"}).copy()
    trades["model"] = "adaptive_label"
    trades["rank"] = 1
    trades["score"] = 0.0
    trades["entry_buyable"] = eligible.get(
        "entry_buyable", pd.Series(False, index=eligible.index)
    ).fillna(False).astype(bool).to_numpy()
    trades["exit_sellable"] = eligible.get(
        "target_exit_sellable_t1", pd.Series(False, index=eligible.index)
    ).fillna(False).astype(bool).to_numpy()
    return trades


def adaptive_records_to_labels(records: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    labels = records.rename(columns={"signal_date": "date"}).copy()
    calendar_positions = {
        pd.Timestamp(value): index
        for index, value in enumerate(pd.DatetimeIndex(calendar).normalize())
    }
    exit_session = pd.to_datetime(labels["exit_timestamp"], errors="coerce").dt.normalize()
    delays = []
    for date, exit_date in zip(pd.to_datetime(labels["date"]), exit_session):
        if pd.isna(exit_date):
            delays.append(np.nan)
        else:
            delays.append(calendar_positions.get(pd.Timestamp(exit_date), np.nan) - calendar_positions.get(pd.Timestamp(date), np.nan))
    entered = labels["entry_buyable"].fillna(False).astype(bool)
    sold = labels["exit_sellable"].fillna(False).astype(bool)
    output = pd.DataFrame({
        "code": labels["code"].astype(str).str[:6],
        "date": pd.to_datetime(labels["date"]).dt.normalize(),
        "adaptive_label_recipe_version": ADAPTIVE_LABEL_RECIPE_VERSION,
        "adaptive_horizon_observed_t3": True,
        "adaptive_entry_buyable": entered.astype(float),
        "adaptive_liquidated_by_t3": sold.where(entered, np.nan).astype(float),
        "adaptive_open_t3": (entered & ~sold).where(entered, np.nan).astype(float),
        "adaptive_realized_net_ret_t3": pd.to_numeric(labels["net_return"], errors="coerce").where(entered & sold),
        "adaptive_stress_net_ret_t3": pd.to_numeric(labels["penalty_net_return"], errors="coerce"),
        "adaptive_entry_reason": labels["entry_reason"].astype(str),
        "adaptive_exit_reason": labels["exit_reason"].astype(str),
        "adaptive_exit_timestamp": pd.to_datetime(labels["exit_timestamp"], errors="coerce"),
        "adaptive_exit_delay_sessions": delays,
        "adaptive_entry_price": pd.to_numeric(labels["entry_price"], errors="coerce"),
        "adaptive_exit_price": pd.to_numeric(labels["exit_price"], errors="coerce"),
    })
    return output.sort_values(["date", "code"]).reset_index(drop=True)


def label_diagnostics(labels: pd.DataFrame) -> dict:
    entered = labels["adaptive_entry_buyable"].fillna(0).astype(bool)
    liquidated = labels["adaptive_liquidated_by_t3"].fillna(0).astype(bool)
    realized = pd.to_numeric(labels["adaptive_realized_net_ret_t3"], errors="coerce")
    return {
        "rows": int(len(labels)),
        "dates": int(labels["date"].nunique()),
        "codes": int(labels["code"].nunique()),
        "entry_rate": float(entered.mean()),
        "entered": int(entered.sum()),
        "liquidated": int((entered & liquidated).sum()),
        "open_t3": int((entered & ~liquidated).sum()),
        "liquidation_rate_given_entry": float(liquidated[entered].mean()) if entered.any() else None,
        "realized_mean": float(realized.mean()) if realized.notna().any() else None,
        "realized_median": float(realized.median()) if realized.notna().any() else None,
        "stress_mean": float(pd.to_numeric(labels["adaptive_stress_net_ret_t3"], errors="coerce").mean()),
        "exit_reasons": {
            str(key): int(value)
            for key, value in labels.loc[entered, "adaptive_exit_reason"].value_counts().items()
        },
    }


def run_pilot(
    prepared_dir: Path,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sample_size: int = 400,
    batch_size: int = 200,
    fetch: bool = True,
) -> dict:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    calendar = load_trading_calendar(prepared_dir)
    eligible, selected_codes = load_pilot_universe(prepared_dir, start, end, sample_size)
    trades = build_label_trades(eligible)
    plan, incomplete_dates = build_fetch_plan(trades, calendar, REPLAY_CONFIG.max_exit_sessions)
    complete = trades[~pd.to_datetime(trades["signal_date"]).dt.normalize().isin(incomplete_dates)].copy()
    output_dir = Path(output_dir)
    bars_path = output_dir / "pilot_execution_bars.parquet"
    fetch_report_path = output_dir / "pilot_fetch_report.json"
    if fetch:
        bars, fetch_report = fetch_execution_bars(plan, batch_size=batch_size)
        atomic_parquet(bars, bars_path)
        atomic_json(fetch_report, fetch_report_path)
    else:
        bars = pd.read_parquet(bars_path)
        fetch_report = json.loads(fetch_report_path.read_text(encoding="utf-8"))
    records = replay_selected_trades(complete, bars, calendar, REPLAY_CONFIG)
    labels = adaptive_records_to_labels(records, calendar)
    diagnostics = label_diagnostics(labels)
    report = {
        "protocol": ADAPTIVE_LABEL_PROTOCOL,
        "recipe_version": ADAPTIVE_LABEL_RECIPE_VERSION,
        "start": str(start.date()),
        "end": str(end.date()),
        "sample_size": int(sample_size),
        "selected_codes": selected_codes,
        "eligible_rows": int(len(eligible)),
        "complete_rows": int(len(complete)),
        "incomplete_dates": [str(value.date()) for value in incomplete_dates],
        "fetch": fetch_report,
        "diagnostics": diagnostics,
        "execution_config": REPLAY_CONFIG.__dict__,
    }
    atomic_parquet(labels, output_dir / "pilot_adaptive_labels.parquet")
    atomic_parquet(records, output_dir / "pilot_adaptive_records.parquet")
    atomic_json(report, output_dir / "pilot_adaptive_label_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot all-candidate adaptive T+3 label construction")
    parser.add_argument("--prepared-dir", type=Path, default=config.PREPARED_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=config.DATA_ROOT / "adaptive_label_pilot_2025q3",
    )
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2025-10-31")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--reuse-bars", action="store_true")
    args = parser.parse_args()
    result = run_pilot(
        args.prepared_dir,
        args.output_dir,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end),
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        fetch=not args.reuse_bars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
