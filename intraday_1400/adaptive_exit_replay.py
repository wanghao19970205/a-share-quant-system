from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.collector import _query_min5
from intraday_1400.offline_race import (
    ExecutionConfig,
    _adaptive_exit,
    _bar_price,
    _locked_bar,
    _normalize_bars,
    _previous_close,
    compare_execution_records,
)
from intraday_1400.storage import atomic_json, atomic_parquet
from stock_analyzer import amazingdata_source as source


REPLAY_PROTOCOL = "daily_vs_e3_adaptive_exit_v1"
REPLAY_CONFIG = ExecutionConfig(
    top_n=10,
    entry_time="14:50",
    time_exit_signal="14:45",
    roundtrip_cost=0.002,
    unsellable_return=-0.10,
    stop_loss=0.05,
    take_profit=0.09,
    trailing_arm=0.03,
    trailing_drawdown=0.02,
    max_exit_sessions=3,
)


def load_trading_calendar(prepared_dir: Path) -> pd.DatetimeIndex:
    dates = []
    for path in sorted(Path(prepared_dir).glob("????-??.parquet")):
        frame = pd.read_parquet(path, columns=["date"])
        dates.extend(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize().unique())
    calendar = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    if calendar.empty:
        raise RuntimeError("prepared trading calendar is empty")
    return calendar


def build_fetch_plan(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    max_exit_sessions: int = 3,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    required = {"code", "signal_date"}
    if not required.issubset(trades.columns):
        raise ValueError(f"trades require {sorted(required)}")
    dates = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    positions = {pd.Timestamp(value): index for index, value in enumerate(dates)}
    rows = []
    incomplete = set()
    for row in trades[["code", "signal_date"]].drop_duplicates().itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date).normalize()
        index = positions.get(signal_date)
        if index is None or index < 1 or index + int(max_exit_sessions) >= len(dates):
            incomplete.add(signal_date)
            continue
        for session in dates[index - 1:index + int(max_exit_sessions) + 1]:
            rows.append({"code": str(row.code)[:6], "signal_date": signal_date, "session": session})
    plan = pd.DataFrame(rows)
    if not plan.empty:
        plan = plan.drop_duplicates(["code", "session"]).sort_values(["session", "code"]).reset_index(drop=True)
    return plan, sorted(incomplete)


def _chunks(items: list[str], size: int):
    for offset in range(0, len(items), max(int(size), 1)):
        yield items[offset:offset + max(int(size), 1)]


def fetch_execution_bars(
    plan: pd.DataFrame,
    batch_size: int = 200,
) -> tuple[pd.DataFrame, dict]:
    if plan.empty:
        raise ValueError("execution fetch plan is empty")
    source.set_credentials()
    if not source._ensure_login():  # noqa: SLF001
        raise RuntimeError(f"AmazingData login failed: {source._last_error}")  # noqa: SLF001
    outputs = []
    requests = 0
    requested_codes = 0
    returned_pairs = set()
    plan = plan.copy()
    plan["month"] = pd.to_datetime(plan["session"]).dt.strftime("%Y-%m")
    for month, month_plan in plan.groupby("month", sort=True):
        codes = sorted(month_plan["code"].unique())
        start = pd.Timestamp(month_plan["session"].min()).strftime("%Y%m%d")
        end = pd.Timestamp(month_plan["session"].max()).strftime("%Y%m%d")
        requested_pairs = set(zip(month_plan["code"], pd.to_datetime(month_plan["session"])))
        for batch_index, batch in enumerate(_chunks(codes, batch_size), start=1):
            broker_mapping = {code: source._to_broker_code(code) for code in batch}  # noqa: SLF001
            broker_codes = list(broker_mapping.values())
            factors = source._get_factor_frame(tuple(broker_codes))  # noqa: SLF001
            if factors is None or factors.empty:
                raise RuntimeError(f"factor fetch failed for {month} batch {batch_index}")
            raw = _query_min5(broker_codes, start, end)
            if not isinstance(raw, dict):
                raise TypeError(f"minute query returned {type(raw).__name__}")
            requests += 1
            requested_codes += len(batch)
            for code, broker_code in broker_mapping.items():
                frame = source._normalize_kline(raw.get(broker_code))  # noqa: SLF001
                if frame is None or frame.empty:
                    continue
                factor = source._factor_series(factors, broker_code)  # noqa: SLF001
                if factor is None or factor.empty:
                    raise RuntimeError(f"missing qfq factor for {broker_code}")
                raw_frame = frame.copy()
                adjusted = source._apply_adjust(frame, factor, "qfq")  # noqa: SLF001
                if adjusted is None or adjusted.empty:
                    continue
                adjusted["code"] = code
                adjusted["timestamp"] = pd.to_datetime(adjusted["date"], errors="coerce")
                adjusted["session"] = adjusted["timestamp"].dt.normalize()
                keep = [
                    (code, pd.Timestamp(session)) in requested_pairs
                    for session in adjusted["session"]
                ]
                adjusted = adjusted.loc[keep].copy()
                raw_frame = raw_frame.loc[keep].copy()
                if adjusted.empty:
                    continue
                for column in ("open", "high", "low", "close"):
                    adjusted[f"raw_{column}"] = pd.to_numeric(raw_frame[column], errors="coerce").to_numpy()
                raw_volume = pd.to_numeric(raw_frame.get("volume"), errors="coerce")
                raw_amount = pd.to_numeric(raw_frame.get("amount"), errors="coerce")
                adjusted["volume"] = raw_volume.to_numpy()
                adjusted["amount"] = raw_amount.to_numpy()
                raw_vwap = raw_amount / raw_volume.replace(0, np.nan)
                scale = pd.to_numeric(adjusted["close"], errors="coerce") / pd.to_numeric(
                    raw_frame["close"], errors="coerce"
                ).replace(0, np.nan)
                adjusted["bar_vwap_qfq"] = raw_vwap.to_numpy() * scale.to_numpy()
                adjusted["factor_status"] = "ok"
                outputs.append(adjusted[[
                    "code", "timestamp", "session", "open", "high", "low", "close",
                    "volume", "amount", "bar_vwap_qfq", "raw_open", "raw_high",
                    "raw_low", "raw_close", "factor_status",
                ]])
                returned_pairs.update(zip(adjusted["code"], adjusted["session"]))
            print(
                f"[adaptive-fetch] month={month} batch={batch_index} codes={len(batch)}",
                flush=True,
            )
    if not outputs:
        raise RuntimeError("execution bar query returned no rows")
    bars = pd.concat(outputs, ignore_index=True)
    bars = _normalize_bars(bars)
    planned_pairs = set(zip(plan["code"], pd.to_datetime(plan["session"])))
    report = {
        "requests": int(requests),
        "requested_codes_across_batches": int(requested_codes),
        "planned_code_sessions": int(len(planned_pairs)),
        "returned_code_sessions": int(len(returned_pairs)),
        "coverage": float(len(returned_pairs) / max(len(planned_pairs), 1)),
        "bars": int(len(bars)),
        "start": str(pd.Timestamp(plan["session"].min()).date()),
        "end": str(pd.Timestamp(plan["session"].max()).date()),
    }
    return bars, report


def _replay_one(
    row,
    code_bars: pd.DataFrame,
    sessions: list[pd.Timestamp],
    config_value: ExecutionConfig,
) -> dict:
    signal_date = pd.Timestamp(row.signal_date).normalize()
    entry_clock = pd.Timestamp(f"2000-01-01 {config_value.entry_time}").time()
    record = {
        "model": str(row.model),
        "signal_date": signal_date,
        "code": str(row.code)[:6],
        "rank": int(row.rank),
        "score": float(row.score),
        "entry_timestamp": pd.NaT,
        "entry_price": np.nan,
        "entry_buyable": False,
        "entry_reason": "missing_entry_bar",
        "exit_timestamp": pd.NaT,
        "exit_price": np.nan,
        "exit_sellable": False,
        "exit_reason": "not_entered",
        "outcome_observed": True,
        "gross_return": 0.0,
        "cost": 0.0,
        "net_return": 0.0,
        "penalty_net_return": 0.0,
        "fixed_entry_buyable": bool(row.entry_buyable),
        "fixed_exit_sellable": bool(row.exit_sellable),
    }
    entry = code_bars[
        (code_bars["session"] == signal_date)
        & (code_bars["timestamp"].dt.time == entry_clock)
    ]
    if entry.empty:
        return record
    entry_row = entry.iloc[0]
    previous_close = _previous_close(code_bars, signal_date)
    if _locked_bar(entry_row, previous_close, "buy", config_value):
        record["entry_reason"] = "locked_or_no_volume_1450"
        return record
    entry_price = _bar_price(entry_row)
    record.update({
        "entry_timestamp": entry_row["timestamp"],
        "entry_price": entry_price,
        "entry_buyable": True,
        "entry_reason": "filled_1450",
        "cost": float(config_value.roundtrip_cost),
    })
    outcome = _adaptive_exit(code_bars, entry_price, signal_date, sessions, config_value)
    record.update(outcome)
    if record.get("exit_sellable"):
        record["penalty_net_return"] = float(record["net_return"])
    else:
        record["net_return"] = np.nan
        record["gross_return"] = np.nan
        record["penalty_net_return"] = float(config_value.unsellable_return)
    return record


def replay_selected_trades(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config_value: ExecutionConfig = REPLAY_CONFIG,
) -> pd.DataFrame:
    market = _normalize_bars(bars)
    sessions = [pd.Timestamp(value) for value in pd.DatetimeIndex(calendar).normalize().sort_values()]
    by_code = {
        code: frame.sort_values("timestamp").reset_index(drop=True)
        for code, frame in market.groupby("code", sort=False)
    }
    records = []
    for row in trades.sort_values(["signal_date", "model", "rank"]).itertuples(index=False):
        code_bars = by_code.get(str(row.code)[:6], market.iloc[0:0])
        records.append(_replay_one(row, code_bars, sessions, config_value))
    return pd.DataFrame(records)


def summarize_replay(records: pd.DataFrame) -> dict:
    penalty_records = records.copy()
    penalty_records["net_return"] = penalty_records["penalty_net_return"]
    comparison = compare_execution_records(penalty_records)
    breakdown = {}
    for model, frame in records.groupby("model"):
        entered = frame["entry_buyable"].fillna(False).astype(bool)
        sold = entered & frame["exit_sellable"].fillna(False).astype(bool)
        fixed_entry = frame["fixed_entry_buyable"].fillna(False).astype(bool)
        fixed_blocked = entered & fixed_entry & ~frame["fixed_exit_sellable"].fillna(False).astype(bool)
        recovered = fixed_blocked & frame["exit_sellable"].fillna(False).astype(bool)
        breakdown[model] = {
            "selected": int(len(frame)),
            "entered": int(entered.sum()),
            "sold": int(sold.sum()),
            "still_open_t3": int((entered & ~frame["exit_sellable"].fillna(False).astype(bool)).sum()),
            "fixed_unbuyable_but_raw_filled": int((entered & ~fixed_entry).sum()),
            "fixed_t1_blocked": int(fixed_blocked.sum()),
            "fixed_t1_blocked_recovered": int(recovered.sum()),
            "realized_mean": float(frame.loc[sold, "net_return"].mean()) if sold.any() else None,
            "penalty_mean_per_selection": float(frame["penalty_net_return"].mean()),
            "exit_reasons": {
                str(key): int(value)
                for key, value in frame.loc[entered, "exit_reason"].value_counts().items()
            },
        }
    return {
        "comparison": {
            key: value for key, value in comparison.items() if key != "daily_returns"
        },
        "daily_returns": comparison["daily_returns"],
        "breakdown": breakdown,
    }


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].sort_values(columns).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def run_replay(
    trades_path: Path,
    prepared_dir: Path,
    output_dir: Path,
    batch_size: int = 200,
    fetch: bool = True,
) -> dict:
    trades = pd.read_parquet(trades_path)
    calendar = load_trading_calendar(prepared_dir)
    plan, incomplete_dates = build_fetch_plan(trades, calendar, REPLAY_CONFIG.max_exit_sessions)
    complete_trades = trades[~pd.to_datetime(trades["signal_date"]).dt.normalize().isin(incomplete_dates)].copy()
    output_dir = Path(output_dir)
    bars_manifest_path = output_dir / "execution_bars_manifest.json"
    if fetch:
        bars, fetch_report = fetch_execution_bars(plan, batch_size=batch_size)
        content_hash = _frame_hash(bars, ["code", "timestamp", "open", "high", "low", "close"])
        bars_path = output_dir / f"execution_bars_{content_hash[:16]}.parquet"
        atomic_parquet(bars, bars_path)
        bars_manifest = {
            "protocol": REPLAY_PROTOCOL,
            "bars_path": str(bars_path),
            "content_sha256": content_hash,
            "trade_input_sha256": _frame_hash(
                complete_trades, ["model", "signal_date", "code", "rank", "score"]
            ),
            "incomplete_signal_dates": [str(value.date()) for value in incomplete_dates],
            "fetch": fetch_report,
        }
        atomic_json(bars_manifest, bars_manifest_path)
    else:
        bars_manifest = json.loads(bars_manifest_path.read_text(encoding="utf-8"))
        bars_path = Path(bars_manifest["bars_path"])
        bars = pd.read_parquet(bars_path)
    records = replay_selected_trades(complete_trades, bars, calendar, REPLAY_CONFIG)
    summary = summarize_replay(records)
    report = {
        "protocol": REPLAY_PROTOCOL,
        "config": REPLAY_CONFIG.__dict__,
        "complete_signal_dates": int(pd.to_datetime(complete_trades["signal_date"]).nunique()),
        "incomplete_signal_dates": [str(value.date()) for value in incomplete_dates],
        "bars_manifest": bars_manifest,
        "comparison": summary["comparison"],
        "breakdown": summary["breakdown"],
    }
    atomic_parquet(records, output_dir / "adaptive_execution_records.parquet")
    atomic_parquet(summary["daily_returns"], output_dir / "adaptive_daily_returns.parquet")
    atomic_json(report, output_dir / "adaptive_replay_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted adaptive exit replay for daily versus E3")
    parser.add_argument(
        "--trades", type=Path,
        default=config.DATA_ROOT / "retrospective_daily_vs_e3" / "main_execution_records.parquet",
    )
    parser.add_argument("--prepared-dir", type=Path, default=config.PREPARED_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=config.DATA_ROOT / "retrospective_daily_vs_e3_adaptive_replay",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--reuse-bars", action="store_true")
    args = parser.parse_args()
    report = run_replay(
        args.trades,
        args.prepared_dir,
        args.output_dir,
        batch_size=args.batch_size,
        fetch=not args.reuse_bars,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
