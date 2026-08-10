from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from intraday_1400 import config
from intraday_1400.compat import publish_realtime_shadow
from intraday_1400.evaluation import evaluate as evaluate_daily_topn
from intraday_1400.features import add_asof_base_features, feature_columns
from intraday_1400.storage import atomic_json, atomic_parquet, atomic_parquet_if_changed
from quant import model
from quant.factors import engineering


def _partition_files(directory: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("????-??/*.parquet")):
        grouped.setdefault(path.name, []).append(path)
    return grouped


def _signature(paths: list[Path]) -> str:
    payload = "\n".join(
        f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in sorted(paths)
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _global_label_calendar() -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Return all market dates and dates with at least one completed 14:50 label bar."""
    paths = sorted(config.LABEL_DIR.glob("????-??/*.parquet"))
    workers = max(int(os.environ.get("INTRADAY_1400_PIPELINE_WORKERS", "8") or 8), 1)

    def read_dates(path: Path) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        frame = pd.read_parquet(path, columns=["date", "label_entry_bar_present"])
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        observed = dates[frame["label_entry_bar_present"].fillna(False).astype(bool)]
        return dates.dropna().tolist(), observed.dropna().tolist()

    dates: set[pd.Timestamp] = set()
    observed_dates: set[pd.Timestamp] = set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="intraday-calendar") as executor:
        for values, observed in executor.map(read_dates, paths):
            dates.update(values)
            observed_dates.update(observed)
    return sorted(dates), sorted(observed_dates)


def _global_label_trade_dates() -> list[pd.Timestamp]:
    """Return the market-session union across all stable label partitions."""
    return _global_label_calendar()[0]


def _read_matching_labels(
    part_name: str,
    trade_dates: list[pd.Timestamp] | None = None,
    observed_trade_dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    paths = sorted(config.LABEL_DIR.glob(f"????-??/{part_name}"))
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    if "schema_version" in labels:
        labels = labels[labels["schema_version"] == config.SCHEMA_VERSION].copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").astype("datetime64[ns]")
    labels = labels.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    if trade_dates is None or observed_trade_dates is None:
        global_dates, global_observed_dates = _global_label_calendar()
        calendar_dates = trade_dates if trade_dates is not None else global_dates
        completed_dates = (
            observed_trade_dates
            if observed_trade_dates is not None
            else global_observed_dates
        )
    else:
        calendar_dates = trade_dates
        completed_dates = observed_trade_dates
    next_date = {
        pd.Timestamp(calendar_dates[index]): pd.Timestamp(calendar_dates[index + 1])
        for index in range(len(calendar_dates) - 1)
    }
    labels["previous_close"] = labels.groupby("code")["label_close_1455"].shift(1)
    labels["signal_previous_close"] = labels["previous_close"]
    labels["entry_return"] = labels["label_entry_price_1450"] / labels["previous_close"] - 1.0
    entry_locked = (
        (labels["label_entry_high_1450"] - labels["label_entry_low_1450"]).abs() <= 0.005
    ) & (labels["entry_return"] >= 0.045)
    labels["entry_buyable"] = (
        labels["label_entry_bar_present"].fillna(False).astype(bool)
        & (labels["label_entry_volume_1450"].fillna(0.0) > 0)
        & ~entry_locked
        & (labels["entry_return"] < 0.098)
    )
    labels["next_trade_date"] = labels["date"].map(next_date)
    completed_date_set = {pd.Timestamp(value) for value in completed_dates}
    labels["target_outcome_observed_t1"] = labels["next_trade_date"].isin(completed_date_set)
    for optional_column in ("label_high_to_1450", "label_low_to_1450"):
        if optional_column not in labels:
            labels[optional_column] = np.nan
    exits = labels[[
        "code", "date", "label_entry_price_1450", "label_entry_high_1450",
        "label_entry_low_1450", "label_entry_volume_1450", "label_entry_bar_present",
        "label_high_to_1450", "label_low_to_1450", "entry_return",
    ]].copy()
    exits["exit_day_row_present_t1"] = True
    exits = exits.rename(columns={
        "date": "next_trade_date",
        "label_entry_price_1450": "label_exit_price_t1",
        "label_entry_high_1450": "label_exit_high_t1",
        "label_entry_low_1450": "label_exit_low_t1",
        "label_entry_volume_1450": "label_exit_volume_t1",
        "label_entry_bar_present": "label_exit_bar_present_t1",
        "label_high_to_1450": "label_exit_path_high_t1",
        "label_low_to_1450": "label_exit_path_low_t1",
        "entry_return": "label_exit_day_return_t1",
    })
    labels = labels.merge(exits, on=["code", "next_trade_date"], how="left", validate="many_to_one")
    exit_locked_down = (
        (labels["label_exit_high_t1"] - labels["label_exit_low_t1"]).abs() <= 0.005
    ) & (labels["label_exit_day_return_t1"] <= -0.045)
    exit_day_present = labels["exit_day_row_present_t1"].astype("boolean").fillna(False).astype(bool)
    exit_bar_present = labels["label_exit_bar_present_t1"].astype("boolean").fillna(False).astype(bool)
    exit_positive_volume = labels["label_exit_volume_t1"].fillna(0.0) > 0
    exit_valid_price = labels["label_exit_price_t1"].notna() & (labels["label_exit_price_t1"] > 0)
    labels["exit_sellable_t1"] = (
        exit_day_present & exit_bar_present & exit_positive_volume
        & exit_valid_price & ~exit_locked_down
    )
    mature_position = (
        labels["entry_buyable"]
        & labels["target_outcome_observed_t1"]
    )
    cause_masks = {
        "target_exit_missing_day_t1": mature_position & ~exit_day_present,
        "target_exit_missing_bar_t1": mature_position & exit_day_present & ~exit_bar_present,
        "target_exit_zero_volume_t1": (
            mature_position & exit_day_present & exit_bar_present & ~exit_positive_volume
        ),
        "target_exit_flat_limit_down_t1": (
            mature_position & exit_day_present & exit_bar_present
            & exit_positive_volume & exit_locked_down
        ),
    }
    known_cause = pd.Series(False, index=labels.index)
    for column, mask in cause_masks.items():
        labels[column] = np.nan
        labels.loc[mature_position, column] = mask[mature_position].astype(float)
        known_cause |= mask
    other_unsellable = mature_position & ~known_cause & ~labels["exit_sellable_t1"]
    labels["target_exit_other_unsellable_t1"] = np.nan
    labels.loc[mature_position, "target_exit_other_unsellable_t1"] = (
        other_unsellable[mature_position].astype(float)
    )
    labels["target_exit_sellable_t1"] = np.nan
    labels.loc[mature_position, "target_exit_sellable_t1"] = (
        labels.loc[mature_position, "exit_sellable_t1"].astype(float)
    )
    cost = float(os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002") or 0.002)
    labels["target_net_ret_t1"] = (
        labels["label_exit_price_t1"] / labels["label_entry_price_1450"] - 1.0 - cost
    ).where(labels["entry_buyable"] & labels["exit_sellable_t1"])
    unsellable_return = float(
        os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
    )
    penalty_eligible = (
        labels["entry_buyable"]
        & labels["target_outcome_observed_t1"]
    )
    labels["target_penalty_net_ret_t1"] = labels["target_net_ret_t1"].where(
        labels["target_net_ret_t1"].notna(), unsellable_return
    ).where(penalty_eligible)
    labels["target_cash_net_ret_t1"] = labels["target_penalty_net_ret_t1"]
    labels.loc[~labels["entry_buyable"], "target_cash_net_ret_t1"] = 0.0
    labels["target_entry_fill"] = labels["entry_buyable"].astype(float)
    labels["target_mfe_t1"] = (
        labels["label_exit_path_high_t1"] / labels["label_entry_price_1450"] - 1.0
    ).where(labels["entry_buyable"])
    labels["target_mae_t1"] = (
        labels["label_exit_path_low_t1"] / labels["label_entry_price_1450"] - 1.0
    ).where(labels["entry_buyable"])
    stop_loss = float(os.environ.get("INTRADAY_1400_STOP_LOSS", "0.03") or 0.03)
    labels["target_stop_hit"] = labels["target_mae_t1"] <= -abs(stop_loss)
    return labels[[
        "code", "date", "label_entry_price_1450", "target_net_ret_t1",
        "target_penalty_net_ret_t1", "target_cash_net_ret_t1",
        "target_entry_fill", "target_outcome_observed_t1", "signal_previous_close",
        "target_exit_sellable_t1", "target_exit_missing_day_t1",
        "target_exit_missing_bar_t1", "target_exit_zero_volume_t1",
        "target_exit_flat_limit_down_t1", "target_exit_other_unsellable_t1",
        "target_mfe_t1", "target_mae_t1", "target_stop_hit",
        "entry_buyable", "exit_sellable_t1",
    ]]


def _add_variant_proxy_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add auditable V1-V5 OHLCV proxy labels; these are not exact L1 replays."""
    data = frame.sort_values(["code", "date"]).copy()
    grouped = data.groupby("code", sort=False)
    entry = pd.to_numeric(data["label_entry_price_1450"], errors="coerce")
    next_close_1400 = grouped["close"].shift(-1)
    next_vwap_gap = grouped["m5_price_vwap_gap"].shift(-1)
    atr = pd.to_numeric(data.get("atr14"), errors="coerce")
    cost = float(os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002") or 0.002)
    generic_gross = data["target_net_ret_t1"] + cost
    mae_to_1450 = pd.to_numeric(data.get("target_mae_t1"), errors="coerce")
    mfe_to_1450 = pd.to_numeric(data.get("target_mfe_t1"), errors="coerce")

    v1_gross = generic_gross.copy()
    v1_stop = mae_to_1450 <= -0.05
    v1_take = (~v1_stop) & (mfe_to_1450 >= 0.09)
    v1_vwap_break = (~v1_stop) & (~v1_take) & (next_vwap_gap < -0.02)
    v1_gross = v1_gross.mask(v1_stop, -0.05)
    v1_gross = v1_gross.mask(v1_take, 0.09)
    v1_gross = v1_gross.mask(v1_vwap_break, next_close_1400 / entry - 1.0)

    # V2 has the same T+1 fixed thresholds in the first holding day. Breakeven/limit-open
    # ordering cannot be recovered from daily aggregates and remains an explicit proxy gap.
    v2_gross = v1_gross.copy()

    risk_return = 2.0 * atr / entry
    v3_gross = generic_gross.copy()
    v3_stop = mae_to_1450 <= -risk_return
    v3_take = (~v3_stop) & (mfe_to_1450 >= 2.0 * risk_return)
    v3_gross = v3_gross.mask(v3_stop, -risk_return)
    v3_gross = v3_gross.mask(v3_take, 2.0 * risk_return)

    valid = data["entry_buyable"].fillna(False) & data["target_net_ret_t1"].notna()
    data["target_v1_proxy_net"] = (v1_gross - cost).where(valid)
    data["target_v2_proxy_net"] = (v2_gross - cost).where(valid)
    data["target_v3_proxy_net"] = (v3_gross - cost).where(valid)
    data["target_v4_proxy_net"] = data["target_v3_proxy_net"]
    data["target_v5_proxy_net"] = data["target_v3_proxy_net"]
    data["target_proxy_mfe_to_1450"] = mfe_to_1450.where(valid)
    data["target_proxy_mae_to_1450"] = mae_to_1450.where(valid)
    data["target_proxy_intrabar_ambiguous"] = (
        (v1_stop & (mfe_to_1450 >= 0.09))
        | (v3_stop & (mfe_to_1450 >= 2.0 * risk_return))
    ).where(valid)
    return data


def build_feature_parts() -> dict:
    """Build rolling features one stock-batch at a time to bound memory."""
    config.ensure_dirs()
    grouped = _partition_files(config.ASOF_PRICE_DIR)
    trade_dates, observed_trade_dates = _global_label_calendar()
    state_path = config.CHECKPOINT_DIR / "feature_build_state.json"
    state = _read_state(state_path)
    report = {
        "parts": 0,
        "cache_hits": 0,
        "rows": 0,
        "changed_feature_files": 0,
        "seconds": 0.0,
    }
    started = time.perf_counter()
    workers = max(int(os.environ.get("INTRADAY_1400_PIPELINE_WORKERS", "8") or 8), 1)
    feature_recipe = {
        "schema_version": config.SCHEMA_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "label_recipe_version": config.LABEL_RECIPE_VERSION,
        "cutoff_time": config.CUTOFF_TIME,
        "roundtrip_cost": os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002"),
    }
    pending: list[tuple[int, str, list[Path], str]] = []
    for index, (part_name, paths) in enumerate(sorted(grouped.items())):
        label_paths = sorted(config.LABEL_DIR.glob(f"????-??/{part_name}"))
        signature = hashlib.sha1(
            f"{_signature(paths + label_paths)}:{json.dumps(feature_recipe, sort_keys=True)}".encode("utf-8")
        ).hexdigest()
        if state.get(part_name) == signature:
            report["cache_hits"] += 1
            print(f"[intraday1400:features] part={index + 1}/{len(grouped)} cache-hit", flush=True)
        else:
            pending.append((index, part_name, paths, signature))

    def build_one(item: tuple[int, str, list[Path], str]) -> dict:
        index, part_name, paths, signature = item
        raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if "schema_version" in raw:
            raw = raw[raw["schema_version"] == config.SCHEMA_VERSION].copy()
        if "is_complete" in raw:
            raw = raw[raw["is_complete"].fillna(False)].copy()
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").astype("datetime64[ns]")
        raw = raw.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
        enriched = add_asof_base_features(raw)
        labels = _read_matching_labels(
            part_name,
            trade_dates=trade_dates,
            observed_trade_dates=observed_trade_dates,
        )
        if not labels.empty:
            enriched = enriched.merge(labels, on=["code", "date"], how="left", validate="one_to_one")
            previous_close = pd.to_numeric(enriched.get("signal_previous_close"), errors="coerce")
            enriched["signal_eligible"] = (
                enriched.get("is_complete", False).fillna(False).astype(bool)
                & (pd.to_numeric(enriched.get("volume"), errors="coerce").fillna(0.0) > 0)
                & previous_close.notna()
            )
            enriched = _add_variant_proxy_labels(enriched)
        changed_files = 0
        for month, month_frame in enriched.groupby(enriched["date"].dt.strftime("%Y-%m"), sort=True):
            changed_files += int(atomic_parquet_if_changed(
                month_frame.sort_values(["date", "code"]).reset_index(drop=True),
                config.FEATURE_DIR / month / part_name,
            ))
        return {
            "index": index,
            "part_name": part_name,
            "signature": signature,
            "rows": len(enriched),
            "codes": int(enriched["code"].nunique()),
            "changed_files": changed_files,
        }

    results = Parallel(
        n_jobs=workers,
        backend="loky",
        return_as="generator_unordered",
    )(delayed(build_one)(item) for item in pending)
    for result in results:
        report["parts"] += 1
        report["rows"] += result["rows"]
        report["changed_feature_files"] += result["changed_files"]
        state[result["part_name"]] = result["signature"]
        atomic_json(state, state_path)
        print(
            f"[intraday1400:features] part={result['index'] + 1}/{len(grouped)} "
            f"codes={result['codes']} rows={result['rows']} workers={workers} backend=loky",
            flush=True,
        )
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    report["seconds"] = round(time.perf_counter() - started, 3)
    atomic_json(report, config.REPORT_DIR / "feature_build.json")
    return report


def _existing_nonprice_source(month: str) -> Path | None:
    quant_dir = Path(os.environ.get("QUANT_DATA_DIR", "quant_data"))
    candidates = sorted(
        quant_dir.glob(f"*_parts/prepared_monthly/{month}.parquet"),
        key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _nonprice_sources(month: str) -> list[Path]:
    previous = (pd.Timestamp(f"{month}-01") - pd.DateOffset(months=1)).strftime("%Y-%m")
    return [
        path for path in (_existing_nonprice_source(previous), _existing_nonprice_source(month))
        if path is not None
    ]


def _merge_lagged_nonprice(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    source_paths = _nonprice_sources(month)
    if not source_paths:
        return frame
    source = pd.concat([pd.read_parquet(path) for path in source_paths], ignore_index=True)
    prefixes = (
        "yjbb_", "income_", "cashflow_", "balance_", "forecast_",
        "block_trade_", "lhb_", "margin_",
    )
    columns = [
        column for column in source.columns
        if column.startswith(prefixes) and pd.api.types.is_numeric_dtype(source[column])
    ]
    if not columns or not {"code", "date"}.issubset(source.columns):
        return frame
    left = frame.copy()
    left["code"] = left["code"].astype(str).str[:6]
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    source = source[["code", "date"] + columns].copy()
    source["code"] = source["code"].astype(str).str[:6]
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    source = source.dropna(subset=["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    left = left.sort_values(["date", "code"]).reset_index(drop=True)
    source = source.sort_values(["date", "code"]).reset_index(drop=True)
    return pd.merge_asof(
        left,
        source,
        on="date",
        by="code",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values(["date", "code"]).reset_index(drop=True)


def _add_market_industry_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-time market and strict-PIT industry relative features."""
    data = frame.copy()
    context_columns = [
        column for column in (
            "m5_ret_15m", "m5_ret_30m", "m5_ret_60m", "m5_ret_open",
            "m5_slope_30m", "m5_slope_60m", "m5_realized_vol",
            "m5_trend_efficiency", "m5_price_vwap_gap", "m5_signed_volume_ratio",
        ) if column in data
    ]
    for column in context_columns:
        market_median = data.groupby("date")[column].transform("median")
        data[f"{column}_market_excess"] = data[column] - market_median
        data[f"{column}_market_rank"] = data.groupby("date")[column].rank(pct=True)

    history_path = Path(os.environ.get("SNAPSHOT_DIR", "snapshots")) / "sw_industry_history_pit.parquet"
    if history_path.exists():
        try:
            from quant.model_expansion_experiment import (
                _load_industry_metadata,
                _pit_industry_for_frame,
            )
            _, publishable, _, history, _ = _load_industry_metadata(None, history_path)
            if publishable and history is not None:
                data["_pit_industry"] = _pit_industry_for_frame(data, history).fillna("UNKNOWN")
                for column in context_columns:
                    grouped = data.groupby(["date", "_pit_industry"])[column]
                    data[f"{column}_industry_excess"] = data[column] - grouped.transform("median")
                    data[f"{column}_industry_rank"] = grouped.rank(pct=True)
        except (OSError, ValueError, KeyError) as error:
            print(f"[intraday1400:prepare] PIT industry unavailable: {type(error).__name__}", flush=True)
    return data


def prepare_months() -> dict:
    """Perform full-universe same-day winsorization and z-scoring one month at a time."""
    config.ensure_dirs()
    month_dirs = sorted(path for path in config.FEATURE_DIR.glob("????-??") if path.is_dir())
    state_path = config.CHECKPOINT_DIR / "prepare_state.json"
    state = _read_state(state_path)
    report = {"months": 0, "cache_hits": 0, "rows": 0, "features": 0, "seconds": 0.0}
    started = time.perf_counter()
    feature_manifest: list[str] | None = None
    workers = max(int(os.environ.get("INTRADAY_1400_PIPELINE_WORKERS", "8") or 8), 1)
    prepare_recipe = {
        "schema_version": config.SCHEMA_VERSION,
        "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "cutoff_time": config.CUTOFF_TIME,
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
    }
    pending: list[tuple[int, Path, list[Path], str, Path]] = []
    for index, month_dir in enumerate(month_dirs):
        part_paths = sorted(month_dir.glob("*.parquet"))
        if not part_paths:
            continue
        industry_history_path = Path(os.environ.get("SNAPSHOT_DIR", "snapshots")) / "sw_industry_history_pit.parquet"
        nonprice_sources = _nonprice_sources(month_dir.name)
        signature_paths = (
            part_paths
            + ([industry_history_path] if industry_history_path.exists() else [])
            + nonprice_sources
        )
        signature = hashlib.sha1(
            f"{_signature(signature_paths)}:{json.dumps(prepare_recipe, sort_keys=True)}".encode("utf-8")
        ).hexdigest()
        prepared_path = config.PREPARED_DIR / f"{month_dir.name}.parquet"
        if state.get(month_dir.name) == signature and prepared_path.exists():
            cached_columns = pd.read_parquet(prepared_path).columns.tolist()
            cached_features = [
                column for column in cached_columns
                if column not in {"code", "date", "entry_buyable", "signal_eligible"}
                and not column.startswith("target_")
            ]
            if feature_manifest is None:
                feature_manifest = cached_features
            elif feature_manifest != cached_features:
                raise RuntimeError(f"cached feature schema drift in {month_dir.name}")
            report["cache_hits"] += 1
            report["features"] = len(cached_features)
            print(f"[intraday1400:prepare] month={month_dir.name} cache-hit", flush=True)
        else:
            pending.append((index, month_dir, part_paths, signature, prepared_path))

    def prepare_one(item: tuple[int, Path, list[Path], str, Path]) -> dict:
        index, month_dir, part_paths, signature, prepared_path = item
        frame = pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype("datetime64[ns]")
        frame = frame.sort_values(["date", "code"]).drop_duplicates(["code", "date"], keep="last")
        frame = _merge_lagged_nonprice(frame, month_dir.name)
        frame = _add_market_industry_context(frame)
        columns = feature_columns(frame)
        normalized = engineering.winsorize_cross_section(frame, columns)
        normalized = engineering.zscore_cross_section(normalized, columns)
        target_columns = [column for column in normalized.columns if column.startswith("target_")]
        keep = ["code", "date", "entry_buyable", "signal_eligible"] + target_columns + columns
        keep = [column for column in keep if column in normalized.columns]
        prepared = normalized[keep].reset_index(drop=True)
        atomic_parquet(prepared, prepared_path)
        return {
            "index": index,
            "month": month_dir.name,
            "signature": signature,
            "rows": len(prepared),
            "columns": columns,
        }

    results = Parallel(
        n_jobs=workers,
        backend="loky",
        return_as="generator_unordered",
    )(delayed(prepare_one)(item) for item in pending)
    for result in results:
        columns = result["columns"]
        if feature_manifest is None:
            feature_manifest = columns
        elif feature_manifest != columns:
            raise RuntimeError(f"feature schema drift in {result['month']}")
        report["months"] += 1
        report["rows"] += result["rows"]
        report["features"] = len(columns)
        state[result["month"]] = result["signature"]
        atomic_json(state, state_path)
        print(
            f"[intraday1400:prepare] month={result['index'] + 1}/{len(month_dirs)} "
            f"name={result['month']} rows={result['rows']} features={len(columns)} "
            f"workers={workers} backend=loky",
            flush=True,
        )
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    report["seconds"] = round(time.perf_counter() - started, 3)
    atomic_json({"features": feature_manifest or []}, config.MODEL_DIR / "feature_manifest.json")
    atomic_json(report, config.REPORT_DIR / "prepare.json")
    return report


def _add_cross_sectional_training_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove each day's market return from each available regression target."""
    target_pairs = (
        ("target_net_ret_t1", "target_excess_ret_t1"),
        ("target_penalty_net_ret_t1", "target_penalty_excess_ret_t1"),
        ("target_cash_net_ret_t1", "target_cash_excess_ret_t1"),
    )
    for source, destination in target_pairs:
        if source not in frame:
            continue
        target = pd.to_numeric(frame[source], errors="coerce")
        daily_mean = target.groupby(frame["date"]).transform("mean")
        frame[destination] = target - daily_mean
    return frame


def _training_target_columns(mode: str) -> tuple[str, str]:
    if mode == "legacy":
        return "target_excess_ret_t1", "target_net_ret_t1"
    if mode == "penalty_aware":
        return "target_penalty_excess_ret_t1", "target_penalty_net_ret_t1"
    raise ValueError(f"unknown training target mode: {mode}")


def _load_prepared(
    max_months: int | None = None,
    columns: list[str] | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_rows: int | None = None,
    exclude_dates: list[str | pd.Timestamp] | None = None,
) -> pd.DataFrame:
    paths = sorted(config.PREPARED_DIR.glob("????-??.parquet"))
    cutoff = pd.Timestamp(end_date) if end_date is not None else None
    if cutoff is not None:
        cutoff_month = cutoff.strftime("%Y-%m")
        paths = [path for path in paths if path.stem <= cutoff_month]
    if max_months and len(paths) > max_months:
        paths = paths[-max_months:]
    selected_columns = None
    if columns is not None:
        selected_columns = list(dict.fromkeys([
            "code", "date", "target_net_ret_t1", *columns,
        ]))
    excluded = {
        pd.Timestamp(value).normalize() for value in (exclude_dates or [])
    }
    rows_per_part = None
    if max_rows is not None and paths:
        rows_per_part = max(int(max_rows) // len(paths), 1)
    parts: list[pd.DataFrame] = []
    for path in paths:
        part = pd.read_parquet(path, columns=selected_columns)
        if cutoff is not None:
            part = part[part["date"] <= cutoff]
        if excluded:
            part = part[~part["date"].isin(excluded)]
        if rows_per_part is not None and len(part) > rows_per_part:
            step = max(len(part) // rows_per_part, 1)
            part = part.iloc[::step].head(rows_per_part)
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    if max_rows is not None and len(panel) > int(max_rows):
        panel = panel.iloc[:int(max_rows)].copy()
    panel.sort_values(["date", "code"], inplace=True, ignore_index=True)
    return _add_cross_sectional_training_target(panel)


def _prediction_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("date")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((frame[column] - mean) / std).fillna(0.0)


def _evaluate_close_baseline(
    panel: pd.DataFrame,
    train_end: str,
    valid_end: str,
    top_n: int = 10,
) -> dict:
    quant_dir = Path(os.environ.get("QUANT_DATA_DIR", "quant_data"))
    path = quant_dir / "active_quant_short_predictions.parquet"
    if not path.exists():
        return {"ok": False, "reason": f"missing {path}"}
    predictions = pd.read_parquet(path)
    pred_column = next(
        (column for column in ("pred", "ensemble_pred", "score") if column in predictions.columns),
        None,
    )
    if pred_column is None or not {"code", "date"}.issubset(predictions.columns):
        return {"ok": False, "reason": "baseline prediction schema mismatch"}
    predictions = predictions[["code", "date", pred_column]].copy()
    predictions["code"] = predictions["code"].astype(str).str[:6]
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").astype("datetime64[ns]")
    labels = panel[["code", "date", "target_net_ret_t1", "entry_buyable"]].copy()
    labels = labels[
        (labels["date"] > pd.Timestamp(train_end))
        & (labels["date"] <= pd.Timestamp(valid_end))
    ]
    merged = labels.merge(predictions, on=["code", "date"], how="inner")
    merged = merged[merged["entry_buyable"].fillna(False)]
    if merged.empty:
        return {"ok": False, "reason": "no overlapping baseline labels"}
    merged["rank"] = merged.groupby("date")[pred_column].rank(method="first", ascending=False)
    selected = merged[merged["rank"] <= int(top_n)].copy()
    missing_targets = int(selected["target_net_ret_t1"].isna().sum())
    unsellable_return = float(
        os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
    )
    selected["evaluation_return"] = selected["target_net_ret_t1"].fillna(unsellable_return)
    daily = selected.groupby("date")["evaluation_return"].mean()
    if daily.empty:
        return {"ok": False, "reason": "no top baseline rows"}
    std = float(daily.std())
    return {
        "ok": True,
        "days": int(len(daily)),
        "mean_net_return": float(daily.mean()),
        "win_rate": float((daily > 0).mean()),
        "sharpe": float(daily.mean() / std * np.sqrt(252)) if std > 0 else None,
        "max_drawdown": float(((1.0 + daily).cumprod() / (1.0 + daily).cumprod().cummax() - 1.0).min()),
        "mean_names": float(selected.groupby("date")["code"].nunique().mean()),
        "missing_targets": missing_targets,
        "unsellable_return": unsellable_return,
    }


def _select_features(
    panel: pd.DataFrame,
    features: list[str],
    train_end: str,
    top_n: int,
    label_col: str = "target_excess_ret_t1",
    max_rows: int = 400_000,
) -> list[str]:
    train = panel[(panel["date"] <= pd.Timestamp(train_end)) & panel[label_col].notna()]
    if len(train) > max_rows:
        step = max(len(train) // max_rows, 1)
        train = train.iloc[::step].head(max_rows)
    target = train.groupby("date")[label_col].rank(pct=True) - 0.5
    scores: list[tuple[float, str]] = []
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if valid.sum() < 100 or values[valid].std() == 0:
            continue
        correlation = values[valid].corr(target[valid])
        if pd.notna(correlation):
            scores.append((abs(float(correlation)), feature))
    selected = [feature for _, feature in sorted(scores, reverse=True)[:max(int(top_n), 1)]]
    if not selected:
        raise RuntimeError("feature screening selected no usable columns")
    return selected


def _minute_family(feature: str) -> str:
    if "_market_" in feature or "_industry_" in feature:
        return "context"
    if any(token in feature for token in ("corr", "autocorr", "persistence", "lead_")):
        return "dependence"
    if any(token in feature for token in ("volume", "vwap", "amihud", "amount", "signed_")):
        return "volume_vwap"
    if any(token in feature for token in ("vol", "semivar", "downside", "drawdown", "kurt", "skew", "jump", "max_bar")):
        return "risk"
    if any(token in feature for token in ("ret_", "slope", "accel", "speed")):
        return "speed"
    return "path"


def _select_minute_features_grouped(
    panel: pd.DataFrame,
    features: list[str],
    train_end: str,
    top_n: int,
    quota: int = 5,
    label_col: str = "target_excess_ret_t1",
    max_rows: int = 400_000,
) -> tuple[list[str], dict[str, list[str]]]:
    train = panel[(panel["date"] <= pd.Timestamp(train_end)) & panel[label_col].notna()]
    if len(train) > max_rows:
        step = max(len(train) // max_rows, 1)
        train = train.iloc[::step].head(max_rows)
    target = train.groupby("date")[label_col].rank(pct=True) - 0.5
    scores: dict[str, float] = {}
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if valid.sum() < 100 or values[valid].std() == 0:
            continue
        correlation = values[valid].corr(target[valid])
        if pd.notna(correlation):
            scores[feature] = abs(float(correlation))
    grouped: dict[str, list[str]] = {}
    for feature in sorted(scores, key=scores.get, reverse=True):
        grouped.setdefault(_minute_family(feature), []).append(feature)
    selected: list[str] = []
    selected_by_family: dict[str, list[str]] = {}
    for family in ("speed", "path", "volume_vwap", "risk", "dependence", "context"):
        chosen = grouped.get(family, [])[:max(int(quota), 0)]
        selected.extend(chosen)
        selected_by_family[family] = chosen
    for feature in sorted(scores, key=scores.get, reverse=True):
        if len(selected) >= top_n:
            break
        if feature not in selected:
            selected.append(feature)
            selected_by_family.setdefault(_minute_family(feature), []).append(feature)
    if not selected:
        raise RuntimeError("grouped minute screening selected no usable columns")
    return selected[:top_n], selected_by_family


def _realized_leg_metrics(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    train_end: str,
    valid_end: str,
    top_n: int = 10,
) -> dict:
    """Evaluate a model leg by daily selection on actual net returns."""
    labels = panel[["code", "date", "target_net_ret_t1", "entry_buyable"]]
    labels = labels[
        (labels["date"] > pd.Timestamp(train_end))
        & (labels["date"] <= pd.Timestamp(valid_end))
        & labels["entry_buyable"].fillna(False)
    ]
    scores = predictions[["code", "date", "pred"]]
    merged = labels.merge(scores, on=["code", "date"], how="inner", validate="one_to_one")
    if merged.empty:
        return {"realized_days": 0}
    daily_ic = merged.groupby("date", sort=False)[["pred", "target_net_ret_t1"]].apply(
        lambda day: day["pred"].corr(day["target_net_ret_t1"], method="spearman")
    ).dropna()
    merged["daily_rank"] = merged.groupby("date")["pred"].rank(method="first", ascending=False)
    selected = merged[merged["daily_rank"] <= int(top_n)].copy()
    missing_targets = int(selected["target_net_ret_t1"].isna().sum())
    unsellable_return = float(
        os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
    )
    selected["evaluation_return"] = selected["target_net_ret_t1"].fillna(unsellable_return)
    daily_top = selected.groupby("date")["evaluation_return"].mean()
    return {
        "realized_days": int(len(daily_top)),
        "realized_daily_rank_ic": float(daily_ic.mean()) if not daily_ic.empty else None,
        f"realized_top{int(top_n)}_mean_net_return": (
            float(daily_top.mean()) if not daily_top.empty else None
        ),
        f"realized_top{int(top_n)}_win_rate": (
            float((daily_top > 0).mean()) if not daily_top.empty else None
        ),
        "realized_mean_names": float(selected.groupby("date")["code"].nunique().mean()),
        "realized_missing_targets": missing_targets,
        "realized_unsellable_return": unsellable_return,
    }


def _project_model_panel(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    target_columns = [column for column in panel if column.startswith("target_")]
    columns = ["code", "date", "entry_buyable", *target_columns, *features]
    return panel.loc[:, list(dict.fromkeys(columns))]


def _fit_variant(
    name: str,
    panel: pd.DataFrame,
    features: list[str],
    train_end: str,
    valid_end: str,
    predict_start: str,
    model_threads: int,
    target_mode: str = "legacy",
) -> dict:
    started = time.perf_counter()
    regression_target, ranking_target = _training_target_columns(target_mode)
    model_panel = _project_model_panel(panel, features)
    common = {
        "panel": model_panel,
        "features": features,
        "horizon": 1,
        "train_end": train_end,
        "valid_end": valid_end,
        "predict_start": predict_start,
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
        # Buyability at 14:50 is not known when the 14:00 signal is formed.
        "train_mask_col": None,
    }
    regression = {**common, "label_col": regression_target}
    ranking = {**common, "label_col": ranking_target}
    results = [
        model.train_ridge(alpha=10.0, **regression),
        model.train_lightgbm_ranker(
            n_estimators=200,
            learning_rate=0.015,
            early_stopping_rounds=40,
            n_jobs=model_threads,
            **ranking,
        ),
        model.train_elastic_net(alpha=0.001, l1_ratio=0.5, **regression),
        model.train_extra_trees(n_estimators=120, max_train_rows=300_000, **regression),
    ]
    for result in results:
        if result.ok and not result.predictions.empty:
            result.metrics.update(_realized_leg_metrics(
                panel, result.predictions, train_end, valid_end,
            ))
    successful = [result for result in results if result.ok and not result.predictions.empty]
    if not successful:
        raise RuntimeError(f"all {name} model legs failed")
    merged: pd.DataFrame | None = None
    weights = {"ridge": 0.15, "lightgbm_ranker": 0.55, "elastic_net": 0.10, "extra_trees": 0.20}
    for result in successful:
        leg = result.predictions[["code", "date", "pred"]].rename(
            columns={"pred": f"{result.model}_pred"}
        )
        merged = leg if merged is None else merged.merge(leg, on=["code", "date"], how="inner")
    assert merged is not None
    merged["pred"] = 0.0
    used_weight = 0.0
    for result in successful:
        column = f"{result.model}_pred"
        weight = weights.get(result.model, 0.0)
        merged[f"{result.model}_z"] = _prediction_zscore(merged, column)
        merged["pred"] += weight * merged[f"{result.model}_z"]
        used_weight += weight
    if used_weight > 0:
        merged["pred"] /= used_weight
    merged["rank"] = merged.groupby("date")["pred"].rank(method="first", ascending=False).astype(int)
    merged = merged.sort_values(["date", "rank", "code"]).reset_index(drop=True)
    atomic_parquet(merged, config.MODEL_DIR / f"{name}_predictions.parquet")
    return {
        "name": name,
        "features": features,
        "weights": weights,
        "metrics": {result.model: result.metrics for result in results},
        "failed": {result.model: result.message for result in results if not result.ok},
        "positive_legs": [
            result.model for result in successful
            if (result.metrics.get("realized_daily_rank_ic") or 0.0) > 0.0
            and (result.metrics.get("realized_top10_mean_net_return") or 0.0) > 0.0
        ],
        "predictions": merged,
        "seconds": round(time.perf_counter() - started, 3),
    }


def train_shadow(model_threads: int = 12) -> dict:
    """Train isolated close/as-of ablations and publish only shadow artifacts."""
    config.ensure_dirs()
    train_months = max(int(os.environ.get("INTRADAY_1400_TRAIN_MONTHS", "36") or 36), 12)
    prepared_paths = sorted(config.PREPARED_DIR.glob("????-??.parquet"))[-(train_months + 6):]
    target_mode = os.environ.get(
        "INTRADAY_1400_TRAIN_TARGET_MODE", "penalty_aware"
    ).strip().lower()
    regression_target, ranking_target = _training_target_columns(target_mode)
    recipe = {
        "schema_version": config.SCHEMA_VERSION,
        "feature_recipe_version": config.FEATURE_RECIPE_VERSION,
        "prepare_recipe_version": config.PREPARE_RECIPE_VERSION,
        "label_recipe_version": config.LABEL_RECIPE_VERSION,
        "train_recipe_version": config.TRAIN_RECIPE_VERSION,
        "training_target_mode": target_mode,
        "regression_target": regression_target,
        "ranking_target": ranking_target,
        "roundtrip_cost": os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002"),
        "cutoff_time": config.CUTOFF_TIME,
        "train_months": train_months,
        "top_base": os.environ.get("INTRADAY_1400_TOP_BASE_FEATURES", "40"),
        "top_minute": os.environ.get("INTRADAY_1400_TOP_MINUTE_FEATURES", "40"),
        "minute_family_quota": os.environ.get("INTRADAY_1400_MINUTE_FAMILY_QUOTA", "5"),
        "run_ablation": os.environ.get("INTRADAY_1400_RUN_ABLATION", "1"),
        "run_family_ablation": os.environ.get("INTRADAY_1400_RUN_FAMILY_ABLATION", "1"),
        "model_threads": int(model_threads),
    }
    input_signature = hashlib.sha1(
        f"{_signature(prepared_paths)}:{json.dumps(recipe, sort_keys=True)}".encode("utf-8")
    ).hexdigest()
    manifest_path = config.MODEL_DIR / "intraday_1400_shadow_manifest.json"
    prediction_path = config.MODEL_DIR / "intraday_1400_shadow_predictions.parquet"
    previous = _read_state(manifest_path)
    if previous.get("input_signature") == input_signature and prediction_path.exists():
        previous["cache_hit"] = True
        print("[intraday1400:train] cache-hit unchanged prepared content", flush=True)
        return previous
    panel = _load_prepared(max_months=train_months + 6)
    if panel.empty:
        raise RuntimeError("no prepared intraday_1400 panel")
    all_features = json.loads((config.MODEL_DIR / "feature_manifest.json").read_text())["features"]
    labeled_dates = sorted(panel.loc[panel["target_net_ret_t1"].notna(), "date"].dropna().unique())
    if len(labeled_dates) < 120:
        raise RuntimeError(f"insufficient labeled dates: {len(labeled_dates)}")
    valid_days = min(60, max(20, len(labeled_dates) // 5))
    train_end = pd.Timestamp(labeled_dates[-valid_days - 1]).strftime("%Y-%m-%d")
    valid_end = pd.Timestamp(labeled_dates[-1]).strftime("%Y-%m-%d")
    # Include the validation segment in prediction output so every model leg can report
    # labeled out-of-sample metrics; the latest unlabeled date remains available for shadow use.
    predict_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    started = time.perf_counter()
    top_base = max(int(os.environ.get("INTRADAY_1400_TOP_BASE_FEATURES", "40") or 40), 10)
    top_minute = max(int(os.environ.get("INTRADAY_1400_TOP_MINUTE_FEATURES", "40") or 40), 10)
    base_candidates = [feature for feature in all_features if not feature.startswith("m5_")]
    minute_candidates = [feature for feature in all_features if feature.startswith("m5_")]
    base_features = _select_features(
        panel, base_candidates, train_end, top_base, label_col=regression_target,
    )
    minute_features, minute_features_by_family = _select_minute_features_grouped(
        panel,
        minute_candidates,
        train_end,
        top_minute,
        quota=int(os.environ.get("INTRADAY_1400_MINUTE_FAMILY_QUOTA", "5") or 5),
        label_col=regression_target,
    )
    features = base_features + minute_features
    atomic_json({
        "train_end": train_end,
        "base_features": base_features,
        "minute_features": minute_features,
        "minute_features_by_family": minute_features_by_family,
    }, config.MODEL_DIR / "selected_features.json")
    run_ablation = os.environ.get("INTRADAY_1400_RUN_ABLATION", "1").lower() not in {"0", "false", "no"}
    base_variant = (
        _fit_variant(
            "asof_base", panel, base_features, train_end, valid_end,
            predict_start, model_threads, target_mode=target_mode,
        )
        if run_ablation else None
    )
    plus_variant = _fit_variant(
        "asof_plus_intraday", panel, features, train_end, valid_end,
        predict_start, model_threads, target_mode=target_mode,
    )
    family_ablation: dict[str, dict] = {}
    if os.environ.get("INTRADAY_1400_RUN_FAMILY_ABLATION", "1").lower() not in {"0", "false", "no"}:
        full_lgb_metrics = plus_variant["metrics"].get("lightgbm_ranker", {})
        for family, family_features in minute_features_by_family.items():
            reduced_features = [feature for feature in features if feature not in set(family_features)]
            result = model.train_lightgbm_ranker(
                panel=_project_model_panel(panel, reduced_features),
                features=reduced_features,
                horizon=1,
                train_end=train_end,
                valid_end=valid_end,
                predict_start=predict_start,
                n_estimators=160,
                learning_rate=0.015,
                early_stopping_rounds=30,
                decay_half_life_days=60.0,
                min_weight=0.03,
                n_jobs=model_threads,
                label_col=ranking_target,
                train_mask_col="entry_buyable",
            )
            if result.ok and not result.predictions.empty:
                result.metrics.update(_realized_leg_metrics(
                    panel, result.predictions, train_end, valid_end,
                ))
            family_ablation[family] = {
                "removed_features": family_features,
                "metrics_without_family": result.metrics,
                "delta_daily_rank_ic": (
                    (full_lgb_metrics.get("realized_daily_rank_ic") or 0.0)
                    - (result.metrics.get("realized_daily_rank_ic") or 0.0)
                ),
                "delta_top10_mean_net_return": (
                    (full_lgb_metrics.get("realized_top10_mean_net_return") or 0.0)
                    - (result.metrics.get("realized_top10_mean_net_return") or 0.0)
                ),
            }
    merged = plus_variant["predictions"]
    atomic_parquet(merged, prediction_path)
    successful_metrics = plus_variant["metrics"]
    positive_legs = plus_variant["positive_legs"]
    weights = plus_variant["weights"]
    def _variant_summary(variant: dict | None) -> dict | None:
        if variant is None:
            return None
        metrics = variant["metrics"]
        rank_ics = [
            value.get("realized_daily_rank_ic") for value in metrics.values()
            if value.get("realized_daily_rank_ic") is not None
        ]
        top_returns = [
            value.get("realized_top10_mean_net_return") for value in metrics.values()
            if value.get("realized_top10_mean_net_return") is not None
        ]
        return {
            "feature_count": len(variant["features"]),
            "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else None,
            "mean_top_net_return": float(np.mean(top_returns)) if top_returns else None,
            "positive_legs": variant["positive_legs"],
            "seconds": variant["seconds"],
        }

    base_summary = _variant_summary(base_variant)
    plus_summary = _variant_summary(plus_variant)
    intraday_incremental = bool(
        base_summary and plus_summary
        and plus_summary["mean_rank_ic"] is not None
        and base_summary["mean_rank_ic"] is not None
        and plus_summary["mean_rank_ic"] > base_summary["mean_rank_ic"]
        and plus_summary["mean_top_net_return"] is not None
        and base_summary["mean_top_net_return"] is not None
        and plus_summary["mean_top_net_return"] > base_summary["mean_top_net_return"]
    )
    close_baseline = _evaluate_close_baseline(panel, train_end, valid_end)
    shadow_gate = {
        "passed": len(positive_legs) >= 2 and (intraday_incremental or not run_ablation),
        "positive_legs": positive_legs,
        "required_positive_legs": 2,
        "intraday_features_incremental": intraday_incremental,
        "promotion_allowed": False,
        "reason": "shadow-only until 20-40 forward trading days complete",
    }
    manifest = {
        "schema_version": config.SCHEMA_VERSION,
        "input_signature": input_signature,
        "cache_hit": False,
        "recipe": recipe,
        "mode": "shadow",
        "published_at": pd.Timestamp.now().isoformat(),
        "latest_date": str(pd.Timestamp(merged["date"].max()).date()),
        "train_end": train_end,
        "valid_end": valid_end,
        "predict_start": predict_start,
        "features": features,
        "weights": weights,
        "metrics": successful_metrics,
        "ablations": {
            "close_baseline": close_baseline,
            "asof_base": base_summary,
            "asof_plus_intraday": plus_summary,
            "minute_feature_families": family_ablation,
        },
        "shadow_gate": shadow_gate,
        "failed": plus_variant["failed"],
        "rows": len(merged),
        "seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(manifest, manifest_path)
    print(
        f"[intraday1400:train] rows={len(merged)} latest={manifest['latest_date']} "
        f"seconds={manifest['seconds']}",
        flush=True,
    )
    return manifest


def validate_minute_families(model_threads: int = 12) -> dict:
    """Measure each minute family on top of the same causal as-of base features."""
    panel = _load_prepared(max_months=42)
    manifest = _read_state(config.MODEL_DIR / "intraday_1400_shadow_manifest.json")
    selected = _read_state(config.MODEL_DIR / "selected_features.json")
    if panel.empty or not manifest or not selected:
        raise RuntimeError("missing prepared panel or trained feature manifests")
    train_end = str(manifest["train_end"])
    valid_end = str(manifest["valid_end"])
    predict_start = str(manifest["predict_start"])
    target_mode = manifest.get("recipe", {}).get("training_target_mode", "legacy")
    _, ranking_target = _training_target_columns(target_mode)
    base_features = list(selected["base_features"])
    families = selected.get("minute_features_by_family", {})
    report = {
        "train_end": train_end,
        "valid_end": valid_end,
        "base_lightgbm": manifest.get("metrics", {}).get("lightgbm_ranker", {}),
        "families": {},
    }
    for family in ("speed", "path", "volume_vwap", "risk", "dependence", "context"):
        minute_features = list(families.get(family, []))
        features = base_features + minute_features
        result = model.train_lightgbm_ranker(
            panel=_project_model_panel(panel, features),
            features=features,
            horizon=1,
            train_end=train_end,
            valid_end=valid_end,
            predict_start=predict_start,
            n_estimators=200,
            learning_rate=0.015,
            early_stopping_rounds=40,
            decay_half_life_days=60.0,
            min_weight=0.03,
            n_jobs=model_threads,
            label_col=ranking_target,
            train_mask_col="entry_buyable",
        )
        if result.ok and not result.predictions.empty:
            result.metrics.update(_realized_leg_metrics(
                panel, result.predictions, train_end, valid_end,
            ))
        report["families"][family] = {
            "features": minute_features,
            "metrics": result.metrics,
            "ok": result.ok,
            "message": result.message,
        }
        print(
            f"[intraday1400:family] name={family} features={len(minute_features)} "
            f"rank_ic={result.metrics.get('realized_daily_rank_ic')} "
            f"top10={result.metrics.get('realized_top10_mean_net_return')}",
            flush=True,
        )
    atomic_json(report, config.REPORT_DIR / "minute_family_validation.json")
    return report


def rolling_validate(
    model_threads: int = 12,
    windows: int = 4,
    valid_days: int = 60,
    top_n: int = 10,
) -> dict:
    if windows <= 0 or valid_days <= 0 or top_n <= 0:
        raise ValueError("windows, valid_days, and top_n must be positive")
    all_features = json.loads((config.MODEL_DIR / "feature_manifest.json").read_text())["features"]
    calendar = _load_prepared(max_months=48, columns=[])
    if calendar.empty:
        raise RuntimeError("no prepared panel for rolling validation")
    labeled_dates = sorted(calendar.loc[
        calendar["target_net_ret_t1"].notna(), "date"
    ].dropna().unique())
    del calendar
    required = windows * valid_days + 120
    if len(labeled_dates) < required:
        raise RuntimeError(f"insufficient rolling dates: {len(labeled_dates)} < {required}")

    window_specs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for window_index in range(windows):
        end_offset = window_index * valid_days
        valid_end_index = len(labeled_dates) - 1 - end_offset
        valid_start_index = valid_end_index - valid_days + 1
        purge_index = valid_start_index - 1
        train_end_index = purge_index - 1
        window_specs.append((
            pd.Timestamp(labeled_dates[train_end_index]),
            pd.Timestamp(labeled_dates[purge_index]),
            pd.Timestamp(labeled_dates[valid_start_index]),
            pd.Timestamp(labeled_dates[valid_end_index]),
        ))

    screening = _load_prepared(
        max_months=48,
        columns=["target_penalty_net_ret_t1", *all_features],
        end_date=max(spec[3] for spec in window_specs),
        max_rows=400_000,
    )
    base_candidates = [feature for feature in all_features if not feature.startswith("m5_")]
    minute_candidates = [feature for feature in all_features if feature.startswith("m5_")]
    selections: list[dict[str, tuple[list[str], list[str], dict[str, list[str]]]]] = []
    for train_end, _, _, _ in window_specs:
        window_selections = {}
        for target_mode in ("legacy", "penalty_aware"):
            regression_target, _ = _training_target_columns(target_mode)
            base_features = _select_features(
                screening, base_candidates, str(train_end.date()), 40,
                label_col=regression_target, max_rows=400_000,
            )
            minute_features, grouped = _select_minute_features_grouped(
                screening, minute_candidates, str(train_end.date()), 40, quota=5,
                label_col=regression_target, max_rows=400_000,
            )
            window_selections[target_mode] = (base_features, minute_features, grouped)
        selections.append(window_selections)
    del screening
    gc.collect()

    reports: list[dict] = []
    for window_index, ((train_end, purge_date, valid_start, valid_end), selection) in enumerate(
        zip(window_specs, selections), start=1,
    ):
        all_selected_features = list(dict.fromkeys(
            feature
            for base_features, minute_features, _ in selection.values()
            for feature in base_features + minute_features
        ))
        window_panel = _load_prepared(
            max_months=48,
            columns=["entry_buyable", "target_penalty_net_ret_t1", *all_selected_features],
            end_date=valid_end,
            exclude_dates=[purge_date],
        )
        suffix = f"rolling_{window_index}"
        variants = {}
        for target_mode, (base_features, minute_features, grouped) in selection.items():
            features = base_features + minute_features
            base = _fit_variant(
                f"{suffix}_{target_mode}_base", window_panel, base_features,
                str(train_end.date()), str(valid_end.date()), str(valid_start.date()),
                model_threads, target_mode=target_mode,
            )
            plus = _fit_variant(
                f"{suffix}_{target_mode}_plus", window_panel, features,
                str(train_end.date()), str(valid_end.date()), str(valid_start.date()),
                model_threads, target_mode=target_mode,
            )
            variants[target_mode] = {
                "base": base,
                "plus": plus,
                "minute_features_by_family": grouped,
            }

        labels = window_panel[["code", "date", "target_net_ret_t1", "entry_buyable"]]
        labels = labels[
            (labels["date"] >= valid_start)
            & (labels["date"] <= valid_end)
            & labels["entry_buyable"].fillna(False)
        ]

        def daily_top(variant: dict) -> dict:
            predictions = variant["predictions"][["code", "date", "pred"]]
            merged = labels.merge(predictions, on=["code", "date"], how="inner")
            merged["rank"] = merged.groupby("date")["pred"].rank(method="first", ascending=False)
            selected = merged[merged["rank"] <= top_n].copy()
            missing_targets = int(selected["target_net_ret_t1"].isna().sum())
            unsellable_return = float(
                os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
            )
            selected["evaluation_return"] = selected["target_net_ret_t1"].fillna(unsellable_return)
            daily = selected.groupby("date")["evaluation_return"].mean()
            std = float(daily.std())
            return {
                "days": int(len(daily)),
                "mean_names": float(selected.groupby("date")["evaluation_return"].count().mean()),
                "missing_targets": missing_targets,
                "unsellable_return": unsellable_return,
                "mean_net_return": float(daily.mean()),
                "win_rate": float((daily > 0).mean()),
                "sharpe": float(daily.mean() / std * np.sqrt(252)) if std > 0 else None,
                "max_drawdown": float(((1.0 + daily).cumprod() / (1.0 + daily).cumprod().cummax() - 1.0).min()),
            }

        comparison = {}
        for target_mode, fitted in variants.items():
            comparison[target_mode] = {
                "base": {
                    "metrics": fitted["base"]["metrics"],
                    "daily_top": daily_top(fitted["base"]),
                },
                "plus": {
                    "metrics": fitted["plus"]["metrics"],
                    "daily_top": daily_top(fitted["plus"]),
                },
                "minute_features_by_family": fitted["minute_features_by_family"],
            }
        legacy = comparison["legacy"]
        penalty_aware = comparison["penalty_aware"]
        reports.append({
            "window": window_index,
            "train_end": str(train_end.date()),
            "purge_date": str(purge_date.date()),
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "base": legacy["base"],
            "plus": legacy["plus"],
            "minute_features_by_family": legacy["minute_features_by_family"],
            "target_comparison": comparison,
        })
        print(
            f"[intraday1400:rolling] window={window_index}/{windows} "
            f"valid={valid_start.date()}..{valid_end.date()} "
            f"legacy_plus={legacy['plus']['daily_top']['mean_net_return']:.6f} "
            f"penalty_plus={penalty_aware['plus']['daily_top']['mean_net_return']:.6f}",
            flush=True,
        )
        del window_panel, variants, labels
        gc.collect()

    def target_summary(target_mode: str) -> dict:
        comparisons = [item["target_comparison"][target_mode] for item in reports]
        return {
            "base_mean_net_return": float(np.mean([
                item["base"]["daily_top"]["mean_net_return"] for item in comparisons
            ])),
            "plus_mean_net_return": float(np.mean([
                item["plus"]["daily_top"]["mean_net_return"] for item in comparisons
            ])),
            "plus_wins": int(sum(
                item["plus"]["daily_top"]["mean_net_return"]
                > item["base"]["daily_top"]["mean_net_return"]
                for item in comparisons
            )),
        }

    legacy_summary = target_summary("legacy")
    penalty_summary = target_summary("penalty_aware")
    summary = {
        "windows": reports,
        **legacy_summary,
        "target_comparison": {
            "legacy": legacy_summary,
            "penalty_aware": penalty_summary,
            "penalty_plus_improvement": (
                penalty_summary["plus_mean_net_return"]
                - legacy_summary["plus_mean_net_return"]
            ),
        },
        "top_n": top_n,
        "valid_days": valid_days,
    }
    atomic_json(summary, config.REPORT_DIR / "rolling_validation.json")
    return summary


def run_pipeline(model_threads: int = 12) -> dict:
    started = time.perf_counter()
    feature_report = build_feature_parts()
    prepare_report = prepare_months()
    train_report = train_shadow(model_threads=model_threads)
    compatibility_report = publish_realtime_shadow()
    evaluation_report = evaluate_daily_topn()
    report = {
        "features": feature_report,
        "prepare": prepare_report,
        "train": train_report,
        "realtime_compatibility": compatibility_report,
        "fair_daily_topn_evaluation": evaluation_report,
        "seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(report, config.REPORT_DIR / "pipeline.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and train isolated intraday 14:00 shadow model")
    parser.add_argument("command", choices=["features", "prepare", "train", "family", "rolling", "all"])
    parser.add_argument("--model-threads", type=int, default=12)
    parser.add_argument("--rolling-windows", type=int, default=4)
    parser.add_argument("--rolling-valid-days", type=int, default=60)
    args = parser.parse_args()
    if args.command == "features":
        print(build_feature_parts())
    elif args.command == "prepare":
        print(prepare_months())
    elif args.command == "train":
        print(train_shadow(model_threads=args.model_threads))
    elif args.command == "family":
        print(validate_minute_families(model_threads=args.model_threads))
    elif args.command == "rolling":
        print(rolling_validate(
            model_threads=args.model_threads,
            windows=args.rolling_windows,
            valid_days=args.rolling_valid_days,
        ))
    else:
        print(run_pipeline(model_threads=args.model_threads))


if __name__ == "__main__":
    main()
