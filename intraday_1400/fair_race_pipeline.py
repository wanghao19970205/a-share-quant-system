from __future__ import annotations

import ctypes
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from intraday_1400 import pipeline
from intraday_1400.features import feature_columns as intraday_feature_columns
from intraday_1400.offline_race import (
    ExecutionConfig,
    compare_execution_records,
    simulate_fixed_exit_race,
)
from intraday_1400.storage import atomic_json, atomic_parquet
from quant import config as quant_config
from quant import model


@dataclass(frozen=True)
class RaceVariant:
    name: str
    feature_groups: tuple[str, ...]
    training_target: str
    causal_at_1400: bool
    role: str


DEFAULT_TRAINED_VARIANTS = (
    RaceVariant(
        "daily_current_retrained",
        ("daily",),
        "daily_h1",
        False,
        "system_retrained_reference",
    ),
    RaceVariant(
        "daily_plus_minute_current_target",
        ("daily", "minute"),
        "daily_h1",
        False,
        "system_diagnostic",
    ),
    RaceVariant(
        "daily_close_control",
        ("daily_matched",),
        "execution",
        False,
        "lookahead_diagnostic",
    ),
    RaceVariant(
        "daily_close_plus_minute_control",
        ("daily_matched", "minute"),
        "execution",
        False,
        "lookahead_diagnostic",
    ),
    RaceVariant(
        "daily_asof_control",
        ("asof_matched",),
        "execution",
        True,
        "causal_control",
    ),
    RaceVariant(
        "daily_asof_plus_minute_control",
        ("asof_matched", "minute"),
        "execution",
        True,
        "causal_control",
    ),
    RaceVariant(
        "minute_legacy",
        ("asof", "minute"),
        "execution_legacy",
        True,
        "tradable_candidate",
    ),
    RaceVariant(
        "minute_penalty",
        ("asof", "minute"),
        "execution",
        True,
        "tradable_candidate",
    ),
)


_KEY_COLUMNS = {"code", "date"}
_INTRADAY_STATE_COLUMNS = {"entry_buyable", "signal_eligible"}
_EXECUTION_TARGET_COLUMNS = {
    "target_net_ret_t1",
    "target_penalty_net_ret_t1",
    "target_cash_net_ret_t1",
    "target_entry_fill",
    "target_outcome_observed_t1",
    "target_excess_ret_t1",
    "target_penalty_excess_ret_t1",
    "target_cash_excess_ret_t1",
    "target_exit_sellable_t1",
    "target_exit_missing_day_t1",
    "target_exit_missing_bar_t1",
    "target_exit_zero_volume_t1",
    "target_exit_flat_limit_down_t1",
    "target_exit_other_unsellable_t1",
}


def default_daily_prepared_dir() -> Path:
    return (
        Path(quant_config.QUANT_DIR)
        / "factor_panel_mainboard_active_h1_parts"
        / "prepared_monthly"
    )


def _month_paths(directory: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Path]:
    start_month = pd.Timestamp(start).strftime("%Y-%m")
    end_month = pd.Timestamp(end).strftime("%Y-%m")
    return {
        path.stem: path
        for path in sorted(directory.glob("????-??.parquet"))
        if start_month <= path.stem <= end_month
    }


def _normalize_keys(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if not _KEY_COLUMNS.issubset(frame.columns):
        raise ValueError(f"{source} frame requires code and date")
    data = frame.copy()
    data["code"] = data["code"].astype(str).str[:6]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["code", "date"])
    if data.duplicated(["code", "date"]).any():
        raise ValueError(f"{source} frame contains duplicate code/date keys")
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def _daily_feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in _KEY_COLUMNS
        and column != target_column
        and not column.startswith("target_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def merge_prepared_frames(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    daily_target_column: str = "target_ret_1d",
    daily_features: list[str] | None = None,
    asof_features: list[str] | None = None,
    minute_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    daily_data = _normalize_keys(daily, "daily")
    intraday_data = _normalize_keys(intraday, "intraday")
    if daily_target_column not in daily_data:
        raise ValueError(f"daily frame missing {daily_target_column}")
    required_targets = {
        "target_net_ret_t1",
        "target_penalty_net_ret_t1",
        "target_outcome_observed_t1",
    }
    missing_targets = required_targets - set(intraday_data.columns)
    if missing_targets:
        raise ValueError(f"intraday frame missing {sorted(missing_targets)}")

    available_daily = _daily_feature_columns(daily_data, daily_target_column)
    selected_daily = available_daily if daily_features is None else [
        column for column in daily_features if column in available_daily
    ]
    available_intraday = intraday_feature_columns(intraday_data)
    available_asof = [column for column in available_intraday if not column.startswith("m5_")]
    available_minute = [column for column in available_intraday if column.startswith("m5_")]
    selected_asof = available_asof if asof_features is None else [
        column for column in asof_features if column in available_asof
    ]
    selected_minute = available_minute if minute_features is None else [
        column for column in minute_features if column in available_minute
    ]
    if not selected_daily or not selected_asof or not selected_minute:
        raise ValueError("daily, asof, and minute feature groups must all be nonempty")

    daily_rename = {column: f"daily__{column}" for column in selected_daily}
    daily_part = daily_data[
        ["code", "date", daily_target_column, *selected_daily]
    ].rename(columns={daily_target_column: "daily_target_ret_1d", **daily_rename})

    target_columns = [
        column for column in intraday_data.columns
        if column in _EXECUTION_TARGET_COLUMNS
    ]
    state_columns = [
        column for column in _INTRADAY_STATE_COLUMNS
        if column in intraday_data.columns
    ]
    intraday_rename = {
        **{column: f"asof__{column}" for column in selected_asof},
        **{column: f"minute__{column}" for column in selected_minute},
    }
    intraday_part = intraday_data[
        [
            "code",
            "date",
            *state_columns,
            *target_columns,
            *selected_asof,
            *selected_minute,
        ]
    ].rename(columns=intraday_rename)
    panel = daily_part.merge(
        intraday_part,
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    panel = pipeline._add_cross_sectional_training_target(panel)
    daily_by_name = {column: daily_rename[column] for column in selected_daily}
    asof_by_name = {column: intraday_rename[column] for column in selected_asof}
    matched_names = sorted(set(daily_by_name) & set(asof_by_name))
    if not matched_names:
        raise ValueError("daily and asof feature groups have no matching feature names")
    groups = {
        "daily": list(daily_by_name.values()),
        "asof": list(asof_by_name.values()),
        "minute": [intraday_rename[column] for column in selected_minute],
        "daily_matched": [daily_by_name[column] for column in matched_names],
        "asof_matched": [asof_by_name[column] for column in matched_names],
    }
    return panel.sort_values(["date", "code"]).reset_index(drop=True), groups


def load_joined_prepared(
    daily_dir: Path,
    intraday_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    daily_features: list[str] | None = None,
    asof_features: list[str] | None = None,
    minute_features: list[str] | None = None,
    max_rows: int | None = None,
    exclude_dates: list[pd.Timestamp] | None = None,
    key_filter: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    daily_paths = _month_paths(daily_dir, start, end)
    intraday_paths = _month_paths(intraday_dir, start, end)
    months = sorted(set(daily_paths) & set(intraday_paths))
    if not months:
        raise RuntimeError("daily and intraday prepared data have no overlapping months")
    excluded = {
        pd.Timestamp(value).normalize() for value in (exclude_dates or [])
    }
    filter_keys = None
    if key_filter is not None:
        filter_keys = _normalize_keys(key_filter[["code", "date"]], "key_filter")
    rows_per_month = None
    if max_rows is not None:
        rows_per_month = max(int(max_rows) // len(months), 1)
    parts = []
    groups: dict[str, list[str]] | None = None
    for month in months:
        daily_columns = None if daily_features is None else [
            "code", "date", "target_ret_1d", *daily_features,
        ]
        intraday_columns = None
        if asof_features is not None and minute_features is not None:
            candidates = [
                "code",
                "date",
                "entry_buyable",
                "signal_eligible",
                "target_net_ret_t1",
                "target_penalty_net_ret_t1",
                "target_cash_net_ret_t1",
                "target_entry_fill",
                "target_outcome_observed_t1",
                "target_exit_sellable_t1",
                "target_exit_missing_day_t1",
                "target_exit_missing_bar_t1",
                "target_exit_zero_volume_t1",
                "target_exit_flat_limit_down_t1",
                "target_exit_other_unsellable_t1",
                *asof_features,
                *minute_features,
            ]
            available = set(pq.ParquetFile(intraday_paths[month]).schema.names)
            intraday_columns = [column for column in candidates if column in available]
        daily = pd.read_parquet(daily_paths[month], columns=daily_columns)
        intraday = pd.read_parquet(intraday_paths[month], columns=intraday_columns)
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        intraday["date"] = pd.to_datetime(intraday["date"], errors="coerce").dt.normalize()
        daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        intraday = intraday[(intraday["date"] >= start) & (intraday["date"] <= end)]
        if excluded:
            daily = daily[~daily["date"].isin(excluded)]
            intraday = intraday[~intraday["date"].isin(excluded)]
        if filter_keys is not None:
            month_keys = filter_keys[
                (filter_keys["date"] >= start)
                & (filter_keys["date"] <= end)
                & (filter_keys["date"] >= pd.Timestamp(month + "-01"))
                & (filter_keys["date"] < pd.Timestamp(month + "-01") + pd.offsets.MonthBegin(1))
            ]
            daily = daily.merge(month_keys, on=["code", "date"], how="inner", validate="one_to_one")
            intraday = intraday.merge(month_keys, on=["code", "date"], how="inner", validate="one_to_one")
        if daily.empty or intraday.empty:
            continue
        merged, month_groups = merge_prepared_frames(
            daily,
            intraday,
            daily_features=daily_features,
            asof_features=asof_features,
            minute_features=minute_features,
        )
        if groups is None:
            groups = month_groups
        elif groups != month_groups:
            raise RuntimeError(f"prepared feature schema drift in {month}")
        if rows_per_month is not None and len(merged) > rows_per_month:
            step = max(len(merged) // rows_per_month, 1)
            merged = merged.iloc[::step].head(rows_per_month)
        parts.append(merged)
    if not parts or groups is None:
        raise RuntimeError("joined prepared panel is empty")
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "code"])
    if max_rows is not None and len(panel) > int(max_rows):
        step = max(len(panel) // int(max_rows), 1)
        panel = panel.iloc[::step].head(int(max_rows))
    return panel.reset_index(drop=True), groups


def variant_features(groups: dict[str, list[str]], variant: RaceVariant) -> list[str]:
    missing = [group for group in variant.feature_groups if not groups.get(group)]
    if missing:
        raise ValueError(f"variant {variant.name} missing feature groups {missing}")
    return list(dict.fromkeys(
        feature
        for group in variant.feature_groups
        for feature in groups[group]
    ))


def _daily_excess_target(panel: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(panel["daily_target_ret_1d"], errors="coerce")
    return target - target.groupby(panel["date"]).transform("mean")


def panel_for_variant(panel: pd.DataFrame, variant: RaceVariant) -> pd.DataFrame:
    if variant.training_target in {"execution", "execution_legacy"}:
        data = panel
    else:
        data = panel.copy(deep=True)
    if variant.training_target == "execution":
        required = {"target_penalty_net_ret_t1", "target_penalty_excess_ret_t1"}
        if not required.issubset(data.columns):
            raise ValueError(f"execution target missing {sorted(required - set(data.columns))}")
    elif variant.training_target == "execution_legacy":
        required = {"target_net_ret_t1", "target_excess_ret_t1"}
        if not required.issubset(data.columns):
            raise ValueError(f"legacy execution target missing {sorted(required - set(data.columns))}")
    elif variant.training_target == "daily_h1":
        if "daily_target_ret_1d" not in data:
            raise ValueError("daily h1 target is unavailable")
        data["target_penalty_net_ret_t1"] = pd.to_numeric(
            data["daily_target_ret_1d"], errors="coerce"
        )
        data["target_penalty_excess_ret_t1"] = _daily_excess_target(data)
    else:
        raise ValueError(f"unknown training target {variant.training_target}")
    return data


def select_variant_features(
    panel: pd.DataFrame,
    candidates: list[str],
    train_end: pd.Timestamp,
    top_n: int,
    label_column: str = "target_penalty_excess_ret_t1",
) -> list[str]:
    return pipeline._select_features(
        panel,
        candidates,
        str(pd.Timestamp(train_end).date()),
        top_n,
        label_col=label_column,
    )


def _fit_ridge_lgbm_variant(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
    target_mode: str,
    lgbm_weight: float = 0.75,
) -> dict:
    regression_target, ranking_target = pipeline._training_target_columns(target_mode)
    model_panel = pipeline._project_model_panel(panel, features)
    common = {
        "panel": model_panel,
        "features": features,
        "horizon": 1,
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "predict_start": str(pd.Timestamp(valid_start).date()),
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
        # Entry tradability is known only after the 14:00 signal and must not
        # determine which historical rows enter the training sample.
        "train_mask_col": None,
    }
    ridge = model.train_ridge(
        alpha=10.0,
        label_col=regression_target,
        **common,
    )
    ranker = model.train_lightgbm_ranker(
        n_estimators=200,
        learning_rate=0.015,
        early_stopping_rounds=0,
        n_jobs=model_threads,
        label_col=ranking_target,
        **common,
    )
    if not ridge.ok or not ranker.ok:
        raise RuntimeError(
            f"fair race model failed: ridge={ridge.message}; ranker={ranker.message}"
        )
    left = ridge.predictions[["code", "date", "pred"]].rename(
        columns={"pred": "ridge_pred"}
    )
    right = ranker.predictions[["code", "date", "pred"]].rename(
        columns={"pred": "lgbm_pred"}
    )
    merged = left.merge(right, on=["code", "date"], how="inner", validate="one_to_one")
    merged["ridge_z"] = pipeline._prediction_zscore(merged, "ridge_pred")
    merged["lgbm_z"] = pipeline._prediction_zscore(merged, "lgbm_pred")
    weight = min(max(float(lgbm_weight), 0.0), 1.0)
    merged["pred"] = weight * merged["lgbm_z"] + (1.0 - weight) * merged["ridge_z"]
    metrics = {}
    for result in (ridge, ranker):
        result.metrics.update(pipeline._realized_leg_metrics(
            panel,
            result.predictions,
            str(pd.Timestamp(train_end).date()),
            str(pd.Timestamp(valid_end).date()),
        ))
        metrics[result.model] = result.metrics
    return {
        "predictions": merged,
        "metrics": metrics,
        "weights": {"ridge": 1.0 - weight, "lightgbm_ranker": weight},
    }


def fit_window_variants(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
    variants: tuple[RaceVariant, ...] = DEFAULT_TRAINED_VARIANTS,
    total_feature_budget: int = 80,
    selected_features_by_variant: dict[str, dict[str, list[str]]] | None = None,
    fit_profile: str = "ridge_lgbm",
    lgbm_weight: float = 0.75,
) -> tuple[dict[str, pd.DataFrame], dict]:
    predictions = {}
    manifest = {
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_start": str(pd.Timestamp(valid_start).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "fit_profile": fit_profile,
        "lgbm_weight": float(lgbm_weight),
        "total_feature_budget": int(total_feature_budget),
        "variants": {},
    }
    for variant in variants:
        variant_panel = panel_for_variant(panel, variant)
        target_mode = "legacy" if variant.training_target == "execution_legacy" else "penalty_aware"
        selection_label = (
            "target_excess_ret_t1"
            if target_mode == "legacy"
            else "target_penalty_excess_ret_t1"
        )
        selected = []
        selected_by_group = {}
        preset = (selected_features_by_variant or {}).get(variant.name)
        group_budget = max(int(total_feature_budget) // len(variant.feature_groups), 1)
        for group in variant.feature_groups:
            if preset is not None:
                chosen = [feature for feature in preset.get(group, []) if feature in groups[group]]
                if not chosen:
                    raise ValueError(f"variant {variant.name} has no preset features for {group}")
            else:
                chosen = select_variant_features(
                    variant_panel,
                    groups[group],
                    train_end,
                    group_budget,
                    label_column=selection_label,
                )
            selected.extend(chosen)
            selected_by_group[group] = chosen
        selected = list(dict.fromkeys(selected))
        if fit_profile == "ridge_lgbm":
            fitted = _fit_ridge_lgbm_variant(
                variant_panel,
                selected,
                train_end,
                valid_start,
                valid_end,
                model_threads,
                target_mode,
                lgbm_weight=lgbm_weight,
            )
        elif fit_profile == "full":
            fitted = pipeline._fit_variant(
                variant.name,
                variant_panel,
                selected,
                str(pd.Timestamp(train_end).date()),
                str(pd.Timestamp(valid_end).date()),
                str(pd.Timestamp(valid_start).date()),
                model_threads,
                target_mode=target_mode,
            )
        else:
            raise ValueError(f"unknown fit profile: {fit_profile}")
        prediction = fitted["predictions"][["code", "date", "pred"]].copy()
        prediction = prediction[
            (prediction["date"] >= pd.Timestamp(valid_start))
            & (prediction["date"] <= pd.Timestamp(valid_end))
        ]
        predictions[variant.name] = prediction.rename(columns={"pred": "score"})
        manifest["variants"][variant.name] = {
            **asdict(variant),
            "selected_features": selected_by_group,
            "feature_count": len(selected),
            "metrics": fitted["metrics"],
        }
    return predictions, manifest


def rolling_window_specs(
    calendar: pd.DataFrame,
    windows: int = 4,
    valid_days: int = 60,
    min_training_days: int = 120,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if windows <= 0 or valid_days <= 0 or min_training_days <= 0:
        raise ValueError("rolling window parameters must be positive")
    required = {"date", "target_net_ret_t1"}
    if not required.issubset(calendar.columns):
        raise ValueError(f"calendar missing {sorted(required - set(calendar.columns))}")
    dates = sorted(pd.Timestamp(value) for value in calendar.loc[
        calendar["target_net_ret_t1"].notna(), "date"
    ].dropna().unique())
    minimum = windows * valid_days + min_training_days
    if len(dates) < minimum:
        raise RuntimeError(f"insufficient rolling dates: {len(dates)} < {minimum}")
    specs = []
    for window_index in range(windows):
        valid_end_index = len(dates) - 1 - window_index * valid_days
        valid_start_index = valid_end_index - valid_days + 1
        purge_index = valid_start_index - 1
        train_end_index = purge_index - 1
        specs.append((
            dates[train_end_index],
            dates[purge_index],
            dates[valid_start_index],
            dates[valid_end_index],
        ))
    return specs


def screen_window_features(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    train_end: pd.Timestamp,
    variants: tuple[RaceVariant, ...] = DEFAULT_TRAINED_VARIANTS,
    total_feature_budget: int = 80,
    align_controls: bool = True,
) -> dict[str, dict[str, list[str]]]:
    selected = {}
    for variant in variants:
        variant_panel = panel_for_variant(panel, variant)
        target_mode = "legacy" if variant.training_target == "execution_legacy" else "penalty_aware"
        selection_label = (
            "target_excess_ret_t1"
            if target_mode == "legacy"
            else "target_penalty_excess_ret_t1"
        )
        group_budget = max(int(total_feature_budget) // len(variant.feature_groups), 1)
        selected[variant.name] = {
            group: select_variant_features(
                variant_panel,
                groups[group],
                train_end,
                group_budget,
                label_column=selection_label,
            )
            for group in variant.feature_groups
        }
    return align_control_feature_selections(selected, groups) if align_controls else selected


def align_control_feature_selections(
    selected: dict[str, dict[str, list[str]]],
    groups: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    close_base = selected.get("daily_close_control", {}).get("daily_matched", [])
    close_plus = selected.get("daily_close_plus_minute_control", {})
    if close_base:
        asof_available = set(groups.get("asof_matched", []))
        mapped_asof = [
            f"asof__{feature.removeprefix('daily__')}"
            for feature in close_base
            if f"asof__{feature.removeprefix('daily__')}" in asof_available
        ]
        if len(mapped_asof) != len(close_base):
            raise ValueError("matched daily/asof control features are not one-to-one")
        selected["daily_close_plus_minute_control"]["daily_matched"] = close_base.copy()
        selected["daily_asof_control"]["asof_matched"] = mapped_asof.copy()
        selected["daily_asof_plus_minute_control"]["asof_matched"] = mapped_asof.copy()
    minute_control = close_plus.get("minute", [])
    if minute_control:
        selected["daily_asof_plus_minute_control"]["minute"] = minute_control.copy()

    # Keep the daily baseline identical when measuring the incremental value
    # of adding minute features.
    daily_base = selected.get("daily_current_retrained", {}).get("daily", [])
    if daily_base and "daily_plus_minute_current_target" in selected:
        selected["daily_plus_minute_current_target"]["daily"] = daily_base.copy()
    return selected


def source_features_for_selection(
    selected: dict[str, dict[str, list[str]]],
) -> tuple[list[str], list[str], list[str]]:
    daily = set()
    asof = set()
    minute = set()
    for groups in selected.values():
        for name, features in groups.items():
            if name.startswith("daily"):
                daily.update(feature.removeprefix("daily__") for feature in features)
            elif name.startswith("asof"):
                asof.update(feature.removeprefix("asof__") for feature in features)
            elif name == "minute":
                minute.update(feature.removeprefix("minute__") for feature in features)
    return sorted(daily), sorted(asof), sorted(minute)


def write_window_artifacts(
    predictions: dict[str, pd.DataFrame],
    manifest: dict,
    output_dir: Path,
    window: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in predictions.items():
        path = output_dir / f"window_{int(window)}_{name}_predictions.parquet"
        atomic_parquet(frame, path)
        paths[name] = path
    atomic_json(manifest, output_dir / f"window_{int(window)}_manifest.json")
    return paths


def load_prediction_history(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    score_column = next(
        (column for column in ("score", "pred", "ensemble_pred") if column in frame),
        None,
    )
    if score_column is None or not _KEY_COLUMNS.issubset(frame.columns):
        raise ValueError(f"prediction history schema mismatch: {path}")
    return frame[["code", "date", score_column]].rename(columns={score_column: "score"})


def _json_report(report: dict) -> dict:
    return {
        key: value
        for key, value in report.items()
        if key != "daily_returns"
    }


def run_screened_rolling_race(
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    active_daily_predictions: Path | None = None,
    model_threads: int = 12,
    fit_profile: str = "ridge_lgbm",
    lgbm_weight: float = 0.75,
    top_n: int = 10,
) -> dict:
    screening = json.loads(screening_report_path.read_text(encoding="utf-8"))
    daily_dir = daily_dir or default_daily_prepared_dir()
    intraday_dir = intraday_dir or pipeline.config.PREPARED_DIR
    active_history = (
        load_prediction_history(active_daily_predictions)
        if active_daily_predictions is not None and active_daily_predictions.exists()
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    window_reports = []
    for window in screening.get("windows", []):
        window_index = int(window["window"])
        train_end = pd.Timestamp(window["train_end"])
        purge_date = pd.Timestamp(window["purge_date"])
        valid_start = pd.Timestamp(window["valid_start"])
        valid_end = pd.Timestamp(window["valid_end"])
        selected = window["selected"]
        daily_features, asof_features, minute_features = source_features_for_selection(selected)
        train_start = max(
            pd.Timestamp("2023-01-01"),
            valid_end - pd.DateOffset(months=48),
        )
        panel, groups = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            valid_end,
            daily_features=daily_features,
            asof_features=asof_features,
            minute_features=minute_features,
            exclude_dates=[purge_date],
        )
        predictions, manifest = fit_window_variants(
            panel,
            groups,
            train_end,
            valid_start,
            valid_end,
            model_threads,
            selected_features_by_variant=selected,
            fit_profile=fit_profile,
            lgbm_weight=lgbm_weight,
        )
        reference = None
        if active_history is not None:
            reference = active_history[
                (active_history["date"] >= valid_start)
                & (active_history["date"] <= valid_end)
            ].copy()
            if reference.empty:
                reference = None
            else:
                predictions["daily_current_reference"] = reference
                manifest["variants"]["daily_current_reference"] = {
                    "causal_at_1400": None,
                    "role": "external_reference",
                    "reason": "history has no row-level publication timestamp",
                    "excluded_from_primary_common_universe": True,
                }
        predictions = _causal_eligible_predictions(
            predictions, panel, valid_start, valid_end
        )
        if reference is not None:
            reference = predictions["daily_current_reference"]
        labels = panel[
            [
                "code",
                "date",
                "entry_buyable",
                "target_net_ret_t1",
                "target_outcome_observed_t1",
            ]
        ]
        labels = labels[
            (labels["date"] >= valid_start)
            & (labels["date"] <= valid_end)
        ]
        primary_predictions = {
            name: frame for name, frame in predictions.items()
            if name != "daily_current_reference"
        }
        records, comparison = simulate_fixed_exit_race(
            primary_predictions,
            labels,
            ExecutionConfig(top_n=top_n),
        )
        reference_comparison = None
        if reference is not None:
            reference_pair = {
                "daily_current_retrained": predictions["daily_current_retrained"],
                "daily_current_reference": reference,
            }
            _, reference_comparison = simulate_fixed_exit_race(
                reference_pair,
                labels,
                ExecutionConfig(top_n=top_n),
            )
        records["window"] = window_index
        all_records.append(records)
        manifest["comparison"] = _json_report(comparison)
        manifest["reference_comparison"] = (
            _json_report(reference_comparison) if reference_comparison is not None else None
        )
        manifest["daily_current_reference_included"] = reference is not None
        write_window_artifacts(predictions, manifest, output_dir, window_index)
        atomic_parquet(records, output_dir / f"window_{window_index}_execution_records.parquet")
        daily_returns = comparison["daily_returns"].copy()
        daily_returns["window"] = window_index
        atomic_parquet(daily_returns, output_dir / f"window_{window_index}_daily_returns.parquet")
        window_reports.append({
            "window": window_index,
            "train_end": str(train_end.date()),
            "purge_date": str(purge_date.date()),
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "comparison": _json_report(comparison),
        })
    if not all_records:
        raise RuntimeError("fair race produced no execution records")
    combined_records = pd.concat(all_records, ignore_index=True)
    combined_report = compare_execution_records(combined_records)
    atomic_parquet(combined_records, output_dir / "execution_records.parquet")
    atomic_parquet(combined_report["daily_returns"], output_dir / "daily_returns.parquet")
    report = {
        "fit_profile": fit_profile,
        "lgbm_weight": float(lgbm_weight),
        "top_n": int(top_n),
        "windows": window_reports,
        "combined": _json_report(combined_report),
    }
    atomic_json(report, output_dir / "fair_race_report.json")
    return report


_FOUR_MODEL_WEIGHTS = {
    "ridge": 0.15,
    "lightgbm": 0.55,
    "elastic_net": 0.10,
    "extra_trees": 0.20,
}


def _release_native_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _fit_four_model_head(
    panel: pd.DataFrame,
    features: list[str],
    target_column: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
) -> dict:
    required = {"code", "date", target_column, *features}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"four-model head missing {sorted(missing)}")
    common = {
        "panel": panel,
        "features": features,
        "horizon": 1,
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "predict_start": str(pd.Timestamp(valid_start).date()),
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
        "label_col": target_column,
        "train_mask_col": None,
    }
    trainers = (
        ("ridge", lambda: model.train_ridge(alpha=10.0, **common)),
        ("lightgbm", lambda: model.train_lightgbm(
            n_estimators=200,
            learning_rate=0.015,
            early_stopping_rounds=0,
            n_jobs=model_threads,
            **common,
        )),
        ("elastic_net", lambda: model.train_elastic_net(
            alpha=0.001, l1_ratio=0.5, **common
        )),
        ("extra_trees", lambda: model.train_extra_trees(
            n_estimators=80,
            max_train_rows=150_000,
            **common,
        )),
    )
    results = []
    for model_name, train in trainers:
        result = train()
        results.append(result)
        print(f"[target-experiment:model] model={model_name} ok={result.ok}", flush=True)
        _release_native_memory()
    failed = {result.model: result.message for result in results if not result.ok}
    if failed:
        raise RuntimeError(f"four-model head failed: {failed}")
    merged: pd.DataFrame | None = None
    for result in results:
        leg = result.predictions[["code", "date", "pred"]].rename(
            columns={"pred": f"{result.model}_pred"}
        )
        merged = leg if merged is None else merged.merge(
            leg, on=["code", "date"], how="inner", validate="one_to_one"
        )
    assert merged is not None
    merged["raw_pred"] = 0.0
    merged["score"] = 0.0
    for result in results:
        column = f"{result.model}_pred"
        weight = _FOUR_MODEL_WEIGHTS[result.model]
        merged["raw_pred"] += weight * merged[column]
        merged["score"] += weight * pipeline._prediction_zscore(merged, column)
    return {
        "predictions": merged,
        "metrics": {result.model: result.metrics for result in results},
        "weights": _FOUR_MODEL_WEIGHTS.copy(),
        "target": target_column,
    }


_CLASSIFIER_WEIGHTS = {
    "ridge_classifier": 0.15,
    "lightgbm_classifier": 0.55,
    "elastic_logistic": 0.10,
    "extra_trees_classifier": 0.20,
}


def _fit_four_classifier_head(
    panel: pd.DataFrame,
    features: list[str],
    target_column: str,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
    minority_weight: float = 20.0,
) -> dict:
    results = []
    for classifier in ("ridge", "lightgbm", "elastic", "extra_trees"):
        result = model.train_binary_classifier(
            panel,
            features,
            target_column,
            classifier,
            train_end=str(pd.Timestamp(train_end).date()),
            valid_end=str(pd.Timestamp(valid_end).date()),
            predict_start=str(pd.Timestamp(valid_start).date()),
            decay_half_life_days=60.0,
            min_weight=0.03,
            minority_weight=minority_weight,
            n_estimators=120,
            max_train_rows=150_000,
            n_jobs=model_threads,
        )
        results.append(result)
        print(f"[exit-risk:classifier] model={result.model} ok={result.ok}", flush=True)
        _release_native_memory()
    failed = {result.model: result.message for result in results if not result.ok}
    if failed:
        raise RuntimeError(f"four-classifier head failed: {failed}")
    merged = None
    for result in results:
        leg = result.predictions[["code", "date", "pred"]].rename(
            columns={"pred": f"{result.model}_pred"}
        )
        merged = leg if merged is None else merged.merge(
            leg, on=["code", "date"], how="inner", validate="one_to_one"
        )
    assert merged is not None
    merged["raw_pred"] = 0.0
    for result in results:
        merged["raw_pred"] += (
            _CLASSIFIER_WEIGHTS[result.model] * merged[f"{result.model}_pred"]
        )
    merged["raw_pred"] = merged["raw_pred"].clip(0.0, 1.0)
    return {
        "predictions": merged,
        "metrics": {result.model: result.metrics for result in results},
        "weights": _CLASSIFIER_WEIGHTS.copy(),
        "target": target_column,
    }


def _score_frame(frame: pd.DataFrame, name: str, score: pd.Series | None = None) -> pd.DataFrame:
    result = frame[["code", "date"]].copy()
    result["score"] = pd.to_numeric(frame["score"] if score is None else score, errors="coerce")
    result["model_variant"] = name
    return result


def _cash_complete_targets(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    if "target_cash_net_ret_t1" not in data:
        buyable = data["entry_buyable"].fillna(False).astype(bool)
        data["target_cash_net_ret_t1"] = pd.to_numeric(
            data["target_penalty_net_ret_t1"], errors="coerce"
        )
        data.loc[~buyable, "target_cash_net_ret_t1"] = 0.0
    if "target_entry_fill" not in data:
        data["target_entry_fill"] = data["entry_buyable"].fillna(False).astype(float)
    if "target_exit_sellable_t1" not in data:
        mature_position = (
            data["entry_buyable"].fillna(False).astype(bool)
            & data["target_outcome_observed_t1"].fillna(False).astype(bool)
        )
        data["target_exit_sellable_t1"] = np.nan
        data.loc[mature_position, "target_exit_sellable_t1"] = (
            data.loc[mature_position, "target_net_ret_t1"].notna().astype(float)
        )
    return pipeline._add_cross_sectional_training_target(data)


def _causal_eligible_predictions(
    predictions: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if "signal_eligible" not in panel:
        raise ValueError("signal_eligible is required for causal pre-ranking eligibility")
    keys = panel.loc[
        panel["signal_eligible"].fillna(False).astype(bool)
        & (panel["date"] >= pd.Timestamp(valid_start))
        & (panel["date"] <= pd.Timestamp(valid_end)),
        ["code", "date"],
    ].drop_duplicates(["code", "date"])
    if keys.empty:
        raise RuntimeError("causal signal-eligible universe is empty")
    return {
        name: normalize.merge(keys, on=["code", "date"], how="inner", validate="one_to_one")
        for name, normalize in predictions.items()
    }


def _cap_training_panel(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    max_train_rows: int = 600_000,
    exclude_dates: tuple[pd.Timestamp, ...] = (),
) -> pd.DataFrame:
    cutoff = pd.Timestamp(train_end)
    excluded = pd.DatetimeIndex(pd.to_datetime(list(exclude_dates))).normalize()
    keep = ~panel["date"].isin(excluded) if len(excluded) else pd.Series(True, index=panel.index)
    train = panel[(panel["date"] <= cutoff) & keep]
    future = panel[(panel["date"] > cutoff) & keep]
    if max_train_rows <= 0 or len(train) <= int(max_train_rows):
        return pd.concat([train, future], ignore_index=True)
    hashes = pd.util.hash_pandas_object(
        train[["date", "code"]], index=False
    ).to_numpy(dtype=np.uint64)
    keep = np.argpartition(hashes, int(max_train_rows) - 1)[:int(max_train_rows)]
    sampled = train.iloc[np.sort(keep)]
    return pd.concat([sampled, future], ignore_index=True).sort_values(
        ["date", "code"]
    ).reset_index(drop=True)


def fit_target_experiment_window(
    panel: pd.DataFrame,
    features: list[str],
    base_features: list[str],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int = 8,
    h1_weights: tuple[float, ...] = (0.25, 0.50, 0.75),
    max_train_rows: int = 600_000,
) -> tuple[dict[str, pd.DataFrame], dict]:
    original_train_rows = int((panel["date"] <= pd.Timestamp(train_end)).sum())
    panel = _cap_training_panel(panel, train_end, max_train_rows=max_train_rows)
    sampled_train_rows = int((panel["date"] <= pd.Timestamp(train_end)).sum())
    data = _cash_complete_targets(panel)
    heads = {}
    specs = (
        ("exec_e0_asof", base_features, "target_penalty_net_ret_t1"),
        ("exec_e0_asof_minute", features, "target_penalty_net_ret_t1"),
        ("exec_e1_cash", features, "target_cash_net_ret_t1"),
        ("entry_fill", features, "target_entry_fill"),
        ("h1_causal", features, "daily_target_ret_1d"),
    )
    for name, head_features, target in specs:
        print(f"[target-experiment:fit] head={name} features={len(head_features)}", flush=True)
        heads[name] = _fit_four_model_head(
            data,
            head_features,
            target,
            train_end,
            valid_start,
            valid_end,
            model_threads,
        )
    predictions = {
        name: _score_frame(head["predictions"], name)
        for name, head in heads.items()
        if name != "entry_fill"
    }
    e0 = heads["exec_e0_asof_minute"]["predictions"]
    fill = heads["entry_fill"]["predictions"]
    e2 = e0[["code", "date", "raw_pred"]].rename(
        columns={"raw_pred": "conditional_return"}
    ).merge(
        fill[["code", "date", "raw_pred"]].rename(columns={"raw_pred": "fill_probability"}),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    e2["fill_probability"] = e2["fill_probability"].clip(0.0, 1.0)
    e2["expected_value"] = e2["fill_probability"] * e2["conditional_return"]
    e2["score"] = pipeline._prediction_zscore(e2, "expected_value")
    predictions["exec_e2_two_head"] = _score_frame(e2, "exec_e2_two_head")

    h1 = predictions["h1_causal"].rename(columns={"score": "h1_score"})
    execution = predictions["exec_e1_cash"].rename(columns={"score": "execution_score"})
    fusion = execution[["code", "date", "execution_score"]].merge(
        h1[["code", "date", "h1_score"]],
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    for weight in h1_weights:
        bounded = min(max(float(weight), 0.0), 1.0)
        name = f"fusion_h1_{int(round(100 * bounded)):02d}"
        combined = bounded * fusion["h1_score"] + (1.0 - bounded) * fusion["execution_score"]
        predictions[name] = _score_frame(fusion.assign(score=combined), name)
    predictions = _causal_eligible_predictions(predictions, data, valid_start, valid_end)
    manifest = {
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_start": str(pd.Timestamp(valid_start).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "models": list(_FOUR_MODEL_WEIGHTS),
        "model_weights": _FOUR_MODEL_WEIGHTS,
        "features": features,
        "base_features": base_features,
        "h1_weights": list(h1_weights),
        "original_train_rows": original_train_rows,
        "sampled_train_rows": sampled_train_rows,
        "max_train_rows": int(max_train_rows),
        "training_sample": "deterministic date-code hash, outer validation retained in full",
        "signal_eligibility": "complete through 13:55, positive cumulative volume, prior close available; no return threshold",
        "heads": {
            name: {
                "target": head["target"],
                "metrics": head["metrics"],
            }
            for name, head in heads.items()
        },
    }
    return predictions, manifest


def run_three_window_target_experiment(
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    top_n: int = 10,
) -> dict:
    screening = json.loads(screening_report_path.read_text(encoding="utf-8"))
    windows = screening.get("windows", [])[:3]
    if len(windows) != 3:
        raise ValueError("target experiment requires exactly three screening windows")
    daily_dir = daily_dir or default_daily_prepared_dir()
    intraday_dir = intraday_dir or pipeline.config.PREPARED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    window_reports = []
    for window in windows:
        window_index = int(window["window"])
        train_end = pd.Timestamp(window["train_end"])
        purge_date = pd.Timestamp(window["purge_date"])
        valid_start = pd.Timestamp(window["valid_start"])
        valid_end = pd.Timestamp(window["valid_end"])
        selected = window["selected"]["daily_asof_plus_minute_control"]
        base_features = selected["asof_matched"]
        minute_features = selected["minute"]
        daily_source = [feature.removeprefix("asof__") for feature in base_features]
        asof_source = [feature.removeprefix("asof__") for feature in base_features]
        minute_source = [feature.removeprefix("minute__") for feature in minute_features]
        train_start = max(pd.Timestamp("2023-01-01"), valid_end - pd.DateOffset(months=48))
        panel, _ = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            valid_end,
            daily_features=daily_source,
            asof_features=asof_source,
            minute_features=minute_source,
            exclude_dates=[purge_date],
        )
        features = [*base_features, *minute_features]
        predictions, manifest = fit_target_experiment_window(
            panel,
            features,
            base_features,
            train_end,
            valid_start,
            valid_end,
            model_threads=model_threads,
        )
        labels = panel.loc[
            (panel["date"] >= valid_start) & (panel["date"] <= valid_end),
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        records, comparison = simulate_fixed_exit_race(
            predictions,
            labels,
            ExecutionConfig(top_n=top_n),
        )
        records["window"] = window_index
        all_records.append(records)
        manifest["comparison"] = _json_report(comparison)
        write_window_artifacts(predictions, manifest, output_dir, window_index)
        atomic_parquet(records, output_dir / f"window_{window_index}_execution_records.parquet")
        daily_returns = comparison["daily_returns"].copy()
        daily_returns["window"] = window_index
        atomic_parquet(daily_returns, output_dir / f"window_{window_index}_daily_returns.parquet")
        window_reports.append({
            "window": window_index,
            "train_end": str(train_end.date()),
            "purge_date": str(purge_date.date()),
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "comparison": _json_report(comparison),
        })
        print(f"[target-experiment:window] completed={window_index}/3", flush=True)
    records = pd.concat(all_records, ignore_index=True)
    combined = compare_execution_records(records)
    atomic_parquet(records, output_dir / "execution_records.parquet")
    atomic_parquet(combined["daily_returns"], output_dir / "daily_returns.parquet")
    report = {
        "fit_profile": "four_model_regression",
        "models": list(_FOUR_MODEL_WEIGHTS),
        "top_n": int(top_n),
        "development_only": True,
        "reason": "these historical windows were already observed during experiment design",
        "windows": window_reports,
        "combined": _json_report(combined),
    }
    atomic_json(report, output_dir / "target_experiment_report.json")
    return report


def _exit_expected_value_frame(
    sellable_head: pd.DataFrame,
    return_head: pd.DataFrame,
    penalty: float,
    name: str,
) -> pd.DataFrame:
    merged = sellable_head[["code", "date", "raw_pred"]].rename(
        columns={"raw_pred": "sell_probability"}
    ).merge(
        return_head[["code", "date", "raw_pred"]].rename(
            columns={"raw_pred": "conditional_return"}
        ),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["sell_probability"] = merged["sell_probability"].clip(0.0, 1.0)
    merged["conditional_return"] = merged["conditional_return"].clip(-0.10, 0.10)
    merged["expected_value"] = (
        merged["sell_probability"] * merged["conditional_return"]
        + (1.0 - merged["sell_probability"]) * float(penalty)
    )
    merged["score"] = pipeline._prediction_zscore(merged, "expected_value")
    return _score_frame(merged, name)


def _top50_risk_constrained_frame(
    sellable_head: pd.DataFrame,
    return_head: pd.DataFrame,
    name: str,
    candidate_n: int = 50,
    risk_exclusions: int = 10,
) -> pd.DataFrame:
    returns = return_head[["code", "date", "raw_pred"]].copy()
    returns["return_score"] = pipeline._prediction_zscore(returns, "raw_pred")
    merged = returns[["code", "date", "return_score"]].merge(
        sellable_head[["code", "date", "raw_pred"]].rename(
            columns={"raw_pred": "sell_probability"}
        ),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["score"] = -1e12
    for _, indices in merged.groupby("date", sort=False).groups.items():
        day = merged.loc[indices]
        candidates = day.nlargest(min(int(candidate_n), len(day)), "return_score")
        exclude_n = min(max(int(risk_exclusions), 0), max(len(candidates) - 10, 0))
        if exclude_n:
            excluded = candidates.nsmallest(exclude_n, "sell_probability").index
            candidates = candidates.drop(index=excluded)
        merged.loc[candidates.index, "score"] = candidates["return_score"]
    return _score_frame(merged, name)


def run_three_window_exit_risk_experiment(
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    top_n: int = 10,
    penalties: tuple[float, ...] = (-0.10,),
    max_train_rows: int = 600_000,
) -> dict:
    screening = json.loads(screening_report_path.read_text(encoding="utf-8"))
    windows = screening.get("windows", [])[:3]
    if len(windows) != 3:
        raise ValueError("exit-risk experiment requires exactly three screening windows")
    daily_dir = daily_dir or default_daily_prepared_dir()
    intraday_dir = intraday_dir or pipeline.config.PREPARED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    window_reports = []
    for window in windows:
        window_index = int(window["window"])
        train_end = pd.Timestamp(window["train_end"])
        purge_date = pd.Timestamp(window["purge_date"])
        valid_start = pd.Timestamp(window["valid_start"])
        valid_end = pd.Timestamp(window["valid_end"])
        selected = window["selected"]["daily_asof_plus_minute_control"]
        base_features = selected["asof_matched"]
        minute_features = selected["minute"]
        source_base = [feature.removeprefix("asof__") for feature in base_features]
        source_minute = [feature.removeprefix("minute__") for feature in minute_features]
        train_start = max(pd.Timestamp("2023-01-01"), valid_end - pd.DateOffset(months=48))
        panel, _ = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            valid_end,
            daily_features=source_base,
            asof_features=source_base,
            minute_features=source_minute,
            exclude_dates=[purge_date],
        )
        full_panel = panel
        panel = _cap_training_panel(panel, train_end, max_train_rows=max_train_rows)
        data = _cash_complete_targets(panel)
        features = [*base_features, *minute_features]
        print(f"[exit-risk:fit] window={window_index} head=e0_baseline", flush=True)
        baseline_head = _fit_four_model_head(
            data, features, "target_penalty_net_ret_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        print(f"[exit-risk:fit] window={window_index} head=sellable", flush=True)
        sellable_head = _fit_four_model_head(
            data, features, "target_exit_sellable_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        print(f"[exit-risk:fit] window={window_index} head=conditional_return", flush=True)
        return_head = _fit_four_model_head(
            data, features, "target_net_ret_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        predictions = {
            "exec_e0_asof_minute": _score_frame(
                baseline_head["predictions"], "exec_e0_asof_minute"
            )
        }
        for penalty in penalties:
            name = f"exec_e3_exit_risk_{int(round(abs(penalty) * 100)):02d}"
            predictions[name] = _exit_expected_value_frame(
                sellable_head["predictions"], return_head["predictions"], penalty, name
            )
        predictions = _causal_eligible_predictions(
            predictions, full_panel, valid_start, valid_end
        )
        labels = full_panel.loc[
            (full_panel["date"] >= valid_start) & (full_panel["date"] <= valid_end),
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        records, comparison = simulate_fixed_exit_race(
            predictions, labels, ExecutionConfig(top_n=top_n)
        )
        records["window"] = window_index
        all_records.append(records)
        manifest = {
            "window": window_index,
            "train_end": str(train_end.date()),
            "purge_date": str(purge_date.date()),
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "features": features,
            "models": list(_FOUR_MODEL_WEIGHTS),
            "penalties": list(penalties),
            "max_train_rows": int(max_train_rows),
            "heads": {
                "e0_baseline": baseline_head["metrics"],
                "sellable": sellable_head["metrics"],
                "conditional_return": return_head["metrics"],
            },
            "comparison": _json_report(comparison),
        }
        for name, frame in predictions.items():
            atomic_parquet(frame, output_dir / f"window_{window_index}_{name}_predictions.parquet")
        atomic_json(manifest, output_dir / f"window_{window_index}_manifest.json")
        atomic_parquet(records, output_dir / f"window_{window_index}_execution_records.parquet")
        daily_returns = comparison["daily_returns"].copy()
        daily_returns["window"] = window_index
        atomic_parquet(daily_returns, output_dir / f"window_{window_index}_daily_returns.parquet")
        window_reports.append({
            "window": window_index,
            "comparison": _json_report(comparison),
        })
        print(f"[exit-risk:window] completed={window_index}/3", flush=True)
    records = pd.concat(all_records, ignore_index=True)
    combined = compare_execution_records(records)
    atomic_parquet(records, output_dir / "execution_records.parquet")
    atomic_parquet(combined["daily_returns"], output_dir / "daily_returns.parquet")
    report = {
        "fit_profile": "four_model_exit_risk",
        "top_n": int(top_n),
        "development_only": True,
        "windows": window_reports,
        "combined": _json_report(combined),
    }
    atomic_json(report, output_dir / "exit_risk_report.json")
    return report


_EXIT_RISK_ASOF_SOURCES = (
    "risk_trading_gap_days",
    "risk_gap_event_count_20",
    "risk_near_limit_up_count_20",
    "risk_near_limit_down_count_20",
    "risk_flat_intraday_count_20",
    "risk_volume_vs_median_20",
    "risk_amount_vs_median_20",
    "volume_ratio_5d",
    "volume_ratio_10d",
    "volume_ratio_20d",
    "volatility_20",
    "drawdown_20",
    "range_pos_20",
    "intraday_range",
    "amount_chg_5",
)
_EXIT_RISK_MINUTE_SOURCES = (
    "m5_volume_vs_20d_median",
    "m5_amount_vs_20d_median",
    "m5_volume_last_30_share",
    "m5_volume_last_60_share",
    "m5_volume_hhi",
    "m5_signed_volume_ratio",
    "m5_amihud",
    "m5_realized_vol",
    "m5_downside_semivar",
)


def run_three_window_exit_classifier_experiment(
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    top_n: int = 10,
    max_train_rows: int = 600_000,
) -> dict:
    screening = json.loads(screening_report_path.read_text(encoding="utf-8"))
    windows = screening.get("windows", [])[:3]
    if len(windows) != 3:
        raise ValueError("exit-classifier experiment requires exactly three windows")
    daily_dir = daily_dir or default_daily_prepared_dir()
    intraday_dir = intraday_dir or pipeline.config.PREPARED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    window_reports = []
    for window in windows:
        window_index = int(window["window"])
        train_end = pd.Timestamp(window["train_end"])
        purge_date = pd.Timestamp(window["purge_date"])
        valid_start = pd.Timestamp(window["valid_start"])
        valid_end = pd.Timestamp(window["valid_end"])
        selected = window["selected"]["daily_asof_plus_minute_control"]
        selected_base = selected["asof_matched"]
        selected_minute = selected["minute"]
        base_sources = [feature.removeprefix("asof__") for feature in selected_base]
        minute_sources = [feature.removeprefix("minute__") for feature in selected_minute]
        all_asof_sources = list(dict.fromkeys([*base_sources, *_EXIT_RISK_ASOF_SOURCES]))
        all_minute_sources = list(dict.fromkeys([*minute_sources, *_EXIT_RISK_MINUTE_SOURCES]))
        train_start = max(pd.Timestamp("2023-01-01"), valid_end - pd.DateOffset(months=48))
        panel, _ = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            valid_end,
            daily_features=base_sources,
            asof_features=all_asof_sources,
            minute_features=all_minute_sources,
            exclude_dates=[purge_date],
        )
        full_panel = panel
        panel = _cap_training_panel(panel, train_end, max_train_rows=max_train_rows)
        data = _cash_complete_targets(panel)
        base_features = [*selected_base, *selected_minute]
        risk_features = list(dict.fromkeys([
            *base_features,
            *(f"asof__{name}" for name in _EXIT_RISK_ASOF_SOURCES),
            *(f"minute__{name}" for name in _EXIT_RISK_MINUTE_SOURCES),
        ]))
        risk_features = [feature for feature in risk_features if feature in data]
        print(f"[exit-classifier:fit] window={window_index} head=regression", flush=True)
        regression_head = _fit_four_model_head(
            data, base_features, "target_exit_sellable_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        print(f"[exit-classifier:fit] window={window_index} head=return", flush=True)
        return_head = _fit_four_model_head(
            data, base_features, "target_net_ret_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        print(f"[exit-classifier:fit] window={window_index} head=classifier_base", flush=True)
        classifier_base = _fit_four_classifier_head(
            data, base_features, "target_exit_sellable_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        print(f"[exit-classifier:fit] window={window_index} head=classifier_risk", flush=True)
        classifier_risk = _fit_four_classifier_head(
            data, risk_features, "target_exit_sellable_t1",
            train_end, valid_start, valid_end, model_threads,
        )
        predictions = {
            "exit_a_regression_ev": _exit_expected_value_frame(
                regression_head["predictions"], return_head["predictions"],
                -0.10, "exit_a_regression_ev",
            ),
            "exit_b_classifier_ev": _exit_expected_value_frame(
                classifier_base["predictions"], return_head["predictions"],
                -0.10, "exit_b_classifier_ev",
            ),
            "exit_c_top50_risk": _top50_risk_constrained_frame(
                classifier_base["predictions"], return_head["predictions"],
                "exit_c_top50_risk",
            ),
            "exit_d_risk_features": _top50_risk_constrained_frame(
                classifier_risk["predictions"], return_head["predictions"],
                "exit_d_risk_features",
            ),
        }
        predictions = _causal_eligible_predictions(
            predictions, full_panel, valid_start, valid_end
        )
        labels = full_panel.loc[
            (full_panel["date"] >= valid_start) & (full_panel["date"] <= valid_end),
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        records, comparison = simulate_fixed_exit_race(
            predictions, labels, ExecutionConfig(top_n=top_n)
        )
        records["window"] = window_index
        all_records.append(records)
        manifest = {
            "window": window_index,
            "train_end": str(train_end.date()),
            "purge_date": str(purge_date.date()),
            "valid_start": str(valid_start.date()),
            "valid_end": str(valid_end.date()),
            "base_features": base_features,
            "risk_features": risk_features,
            "minority_weight": 20.0,
            "candidate_n": 50,
            "risk_exclusions": 10,
            "max_train_rows": int(max_train_rows),
            "cause_counts_train": {
                column: int(pd.to_numeric(data.loc[data["date"] <= train_end, column], errors="coerce").fillna(0).sum())
                for column in (
                    "target_exit_missing_day_t1", "target_exit_missing_bar_t1",
                    "target_exit_zero_volume_t1", "target_exit_flat_limit_down_t1",
                    "target_exit_other_unsellable_t1",
                )
            },
            "heads": {
                "regression": regression_head["metrics"],
                "return": return_head["metrics"],
                "classifier_base": classifier_base["metrics"],
                "classifier_risk": classifier_risk["metrics"],
            },
            "comparison": _json_report(comparison),
        }
        for name, frame in predictions.items():
            atomic_parquet(frame, output_dir / f"window_{window_index}_{name}_predictions.parquet")
        atomic_json(manifest, output_dir / f"window_{window_index}_manifest.json")
        atomic_parquet(records, output_dir / f"window_{window_index}_execution_records.parquet")
        daily_returns = comparison["daily_returns"].copy()
        daily_returns["window"] = window_index
        atomic_parquet(daily_returns, output_dir / f"window_{window_index}_daily_returns.parquet")
        window_reports.append({"window": window_index, "comparison": _json_report(comparison)})
        print(f"[exit-classifier:window] completed={window_index}/3", flush=True)
    records = pd.concat(all_records, ignore_index=True)
    combined = compare_execution_records(records)
    atomic_parquet(records, output_dir / "execution_records.parquet")
    atomic_parquet(combined["daily_returns"], output_dir / "daily_returns.parquet")
    report = {
        "fit_profile": "four_classifier_exit_risk",
        "top_n": int(top_n),
        "development_only": True,
        "windows": window_reports,
        "combined": _json_report(combined),
    }
    atomic_json(report, output_dir / "exit_classifier_report.json")
    return report


_NESTED_MIN_FILL_RATIO = 0.99
_NESTED_RECIPE_GRID = tuple(
    {
        "name": f"R{h_index}{risk_index}_{h_label}{risk_label}",
        "h1_weight": h1_weight,
        "risk_remove_n": risk_remove_n,
    }
    for h_index, (h1_weight, h_label) in enumerate(
        ((0.0, "e0"), (0.25, "h125"), (0.50, "h150"))
    )
    for risk_index, (risk_remove_n, risk_label) in enumerate(
        ((0, ""), (5, "_safe05"), (10, "_safe10"))
    )
)


def _inner_window_spec(
    panel: pd.DataFrame,
    outer_train_end: pd.Timestamp,
    valid_days: int = 60,
) -> dict:
    cutoff = pd.Timestamp(outer_train_end).normalize()
    observed = panel.loc[
        (panel["date"] <= cutoff)
        & panel["target_outcome_observed_t1"].fillna(False).astype(bool),
        "date",
    ]
    dates = pd.DatetimeIndex(pd.to_datetime(observed, errors="coerce").dropna().unique()).sort_values()
    required = int(valid_days) + 2
    if len(dates) < required:
        raise ValueError(f"inner window requires at least {required} mature trading dates")
    valid_dates = dates[-int(valid_days):]
    return {
        "train_end": pd.Timestamp(dates[-int(valid_days) - 2]),
        "purge_date": pd.Timestamp(dates[-int(valid_days) - 1]),
        "valid_start": pd.Timestamp(valid_dates[0]),
        "valid_end": pd.Timestamp(valid_dates[-1]),
        "valid_days": int(valid_days),
    }


def _build_return_fusion_frame(
    e0_head: pd.DataFrame,
    h1_head: pd.DataFrame,
    h1_weight: float,
    name: str,
) -> pd.DataFrame:
    weight = float(h1_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("h1_weight must be between zero and one")
    merged = e0_head[["code", "date", "score"]].rename(
        columns={"score": "e0_score"}
    ).merge(
        h1_head[["code", "date", "score"]].rename(columns={"score": "h1_score"}),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["score"] = (1.0 - weight) * merged["e0_score"] + weight * merged["h1_score"]
    merged["model_variant"] = name
    return merged


def _build_safety_constrained_frame(
    sellable_head: pd.DataFrame,
    return_frame: pd.DataFrame,
    name: str,
    candidate_n: int = 50,
    risk_remove_n: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    candidate_n = int(candidate_n)
    risk_remove_n = int(risk_remove_n)
    top_n = int(top_n)
    if top_n <= 0 or candidate_n < top_n or risk_remove_n < 0:
        raise ValueError("safety constraint requires candidate_n >= top_n > 0 and nonnegative removal")
    if candidate_n - risk_remove_n < top_n:
        raise ValueError("safety constraint must retain at least top_n candidates")
    merged = return_frame[["code", "date", "score"]].rename(
        columns={"score": "return_score"}
    ).merge(
        sellable_head[["code", "date", "raw_pred"]].rename(
            columns={"raw_pred": "sell_probability"}
        ),
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["code"] = merged["code"].astype(str).str[:6]
    merged["candidate_rank"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged["risk_rank"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged["safety_eligible"] = False
    retained = []
    for _, indices in merged.groupby("date", sort=True).groups.items():
        day = merged.loc[indices].sort_values(
            ["return_score", "code"], ascending=[False, True], kind="mergesort"
        )
        candidates = day.head(min(candidate_n, len(day))).copy()
        if len(candidates) < top_n:
            continue
        candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
        risk_order = candidates.sort_values(
            ["sell_probability", "code"], ascending=[True, True], kind="mergesort"
        )
        risk_order["risk_rank"] = np.arange(1, len(risk_order) + 1)
        candidates = candidates.join(risk_order[["risk_rank"]], rsuffix="_ordered")
        candidates["risk_rank"] = candidates["risk_rank_ordered"]
        candidates = candidates.drop(columns=["risk_rank_ordered"])
        survivors = candidates[candidates["risk_rank"] > risk_remove_n].copy()
        if len(survivors) < top_n:
            continue
        survivors["safety_eligible"] = True
        retained.append(survivors)
    if not retained:
        return pd.DataFrame(columns=[
            "code", "date", "score", "model_variant", "return_score",
            "sell_probability", "candidate_rank", "risk_rank", "safety_eligible",
        ])
    result = pd.concat(retained, ignore_index=True)
    result["score"] = result["return_score"]
    result["model_variant"] = name
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def _build_nested_recipe_frames(
    e0_head: pd.DataFrame,
    h1_head: pd.DataFrame,
    sellable_head: pd.DataFrame,
    top_n: int,
    candidate_n: int = 50,
) -> dict[str, pd.DataFrame]:
    frames = {}
    fused = {}
    for recipe in _NESTED_RECIPE_GRID:
        weight = float(recipe["h1_weight"])
        if weight not in fused:
            fused[weight] = _build_return_fusion_frame(
                e0_head, h1_head, weight, f"fusion_h1_{int(weight * 100):02d}"
            )
        name = str(recipe["name"])
        if int(recipe["risk_remove_n"]) == 0:
            frame = fused[weight].copy()
            frame["model_variant"] = name
            frames[name] = frame
        else:
            frames[name] = _build_safety_constrained_frame(
                sellable_head,
                fused[weight],
                name,
                candidate_n=candidate_n,
                risk_remove_n=int(recipe["risk_remove_n"]),
                top_n=top_n,
            )
    return frames


def _evaluate_recipe_frames(
    predictions: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    records, comparison = simulate_fixed_exit_race(
        predictions, labels, ExecutionConfig(top_n=top_n)
    )
    metrics = {}
    for name in predictions:
        model_records = records[records["model"] == name]
        summary = comparison["models"][name]
        normal = (
            model_records["entry_buyable"].fillna(False).astype(bool)
            & model_records["exit_sellable"].fillna(False).astype(bool)
            & model_records["outcome_observed"].fillna(False).astype(bool)
        )
        metrics[name] = {
            **summary,
            "selected": int(len(model_records)),
            "normal_sellable_mean": (
                float(model_records.loc[normal, "net_return"].mean()) if normal.any() else None
            ),
            "unsellable_rate": float(summary["unsellable"] / max(len(model_records), 1)),
        }
    return records, metrics


def _select_inner_recipe(metrics: dict[str, dict]) -> dict:
    if "R00_e0" not in metrics:
        raise ValueError("inner recipe metrics require R00_e0 baseline")
    baseline = metrics["R00_e0"]
    baseline_normal = baseline.get("normal_sellable_mean")
    baseline_filled = float(baseline.get("mean_filled_names") or 0.0)
    eligible = []
    for recipe in _NESTED_RECIPE_GRID:
        name = str(recipe["name"])
        if name not in metrics:
            continue
        current = metrics[name]
        current_filled = float(current.get("mean_filled_names") or 0.0)
        if current_filled < _NESTED_MIN_FILL_RATIO * baseline_filled:
            continue
        risk_remove_n = int(recipe["risk_remove_n"])
        qualifies = risk_remove_n == 0
        if risk_remove_n > 0 and baseline_normal is not None:
            risk_improved = (
                int(current["unsellable"]) <= int(baseline["unsellable"]) - 1
                or float(current["unsellable_rate"]) <= 0.8 * float(baseline["unsellable_rate"])
            )
            current_normal = current.get("normal_sellable_mean")
            qualifies = (
                risk_improved
                and current_normal is not None
                and float(current_normal) >= float(baseline_normal) - 0.0001
            )
        if qualifies:
            eligible.append((recipe, current))
    if not eligible:
        recipe = next(item for item in _NESTED_RECIPE_GRID if item["name"] == "R00_e0")
        return {**recipe, "selection_reason": "fallback_no_eligible_recipe"}
    eligible.sort(key=lambda item: (
        -float(item[1]["mean_return"]),
        float(item[1]["unsellable_rate"]),
        -float(
            item[1]["normal_sellable_mean"]
            if item[1].get("normal_sellable_mean") is not None
            else -1e9
        ),
        float(item[0]["h1_weight"]),
        int(item[0]["risk_remove_n"]),
        str(item[0]["name"]),
    ))
    return {**eligible[0][0], "selection_reason": "best_inner_mean_return"}


def _fit_nested_heads(
    panel: pd.DataFrame,
    features: list[str],
    risk_features: list[str],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
    max_train_rows: int,
    exclude_dates: tuple[pd.Timestamp, ...] = (),
) -> dict:
    sampled = _cap_training_panel(
        panel,
        train_end,
        max_train_rows=max_train_rows,
        exclude_dates=exclude_dates,
    )
    data = _cash_complete_targets(sampled)
    print(f"[nested-safety:fit] cutoff={train_end.date()} head=e0", flush=True)
    e0 = _fit_four_model_head(
        data, features, "target_penalty_net_ret_t1",
        train_end, valid_start, valid_end, model_threads,
    )
    print(f"[nested-safety:fit] cutoff={train_end.date()} head=h1", flush=True)
    h1 = _fit_four_model_head(
        data, features, "daily_target_ret_1d",
        train_end, valid_start, valid_end, model_threads,
    )
    print(f"[nested-safety:fit] cutoff={train_end.date()} head=safety", flush=True)
    safety = _fit_four_classifier_head(
        data, risk_features, "target_exit_sellable_t1",
        train_end, valid_start, valid_end, model_threads,
    )
    return {
        "e0": e0,
        "h1": h1,
        "safety": safety,
        "sampled_train_rows": int((sampled["date"] <= train_end).sum()),
    }


def run_nested_exit_safety_h1_fusion_experiment(
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    model_threads: int = 8,
    top_n: int = 10,
    candidate_n: int = 50,
    inner_valid_days: int = 60,
    max_train_rows: int = 600_000,
) -> dict:
    screening = json.loads(screening_report_path.read_text(encoding="utf-8"))
    windows = screening.get("windows", [])[:3]
    if len(windows) != 3:
        raise ValueError("nested safety experiment requires exactly three outer windows")
    daily_dir = daily_dir or default_daily_prepared_dir()
    intraday_dir = intraday_dir or pipeline.config.PREPARED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_outer_records = []
    window_reports = []
    for window in windows:
        window_index = int(window["window"])
        outer_train_end = pd.Timestamp(window["train_end"])
        outer_purge_date = pd.Timestamp(window["purge_date"])
        outer_valid_start = pd.Timestamp(window["valid_start"])
        outer_valid_end = pd.Timestamp(window["valid_end"])
        selected = window["selected"]["daily_asof_plus_minute_control"]
        selected_base = selected["asof_matched"]
        selected_minute = selected["minute"]
        base_sources = [feature.removeprefix("asof__") for feature in selected_base]
        minute_sources = [feature.removeprefix("minute__") for feature in selected_minute]
        all_asof_sources = list(dict.fromkeys([*base_sources, *_EXIT_RISK_ASOF_SOURCES]))
        all_minute_sources = list(dict.fromkeys([*minute_sources, *_EXIT_RISK_MINUTE_SOURCES]))
        train_start = max(pd.Timestamp("2023-01-01"), outer_valid_end - pd.DateOffset(months=48))
        panel, _ = load_joined_prepared(
            daily_dir,
            intraday_dir,
            train_start,
            outer_valid_end,
            daily_features=base_sources,
            asof_features=all_asof_sources,
            minute_features=all_minute_sources,
            exclude_dates=[outer_purge_date],
        )
        features = [*selected_base, *selected_minute]
        risk_features = list(dict.fromkeys([
            *features,
            *(f"asof__{name}" for name in _EXIT_RISK_ASOF_SOURCES),
            *(f"minute__{name}" for name in _EXIT_RISK_MINUTE_SOURCES),
        ]))
        risk_features = [feature for feature in risk_features if feature in panel]
        inner = _inner_window_spec(panel, outer_train_end, valid_days=inner_valid_days)
        inner_heads = _fit_nested_heads(
            panel,
            features,
            risk_features,
            inner["train_end"],
            inner["valid_start"],
            inner["valid_end"],
            model_threads,
            max_train_rows,
            exclude_dates=(inner["purge_date"],),
        )
        inner_sources = {
            "e0": _score_frame(inner_heads["e0"]["predictions"], "e0"),
            "h1": _score_frame(inner_heads["h1"]["predictions"], "h1"),
            "safety": inner_heads["safety"]["predictions"],
        }
        inner_sources = _causal_eligible_predictions(
            inner_sources, panel, inner["valid_start"], inner["valid_end"]
        )
        inner_recipes = _build_nested_recipe_frames(
            inner_sources["e0"], inner_sources["h1"], inner_sources["safety"],
            top_n=top_n, candidate_n=candidate_n,
        )
        inner_labels = panel.loc[
            (panel["date"] >= inner["valid_start"]) & (panel["date"] <= inner["valid_end"]),
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        inner_records, inner_metrics = _evaluate_recipe_frames(
            inner_recipes, inner_labels, top_n=top_n
        )
        chosen = _select_inner_recipe(inner_metrics)
        chosen_name = str(chosen["name"])
        inner_head_metrics = {
            name: head["metrics"]
            for name, head in inner_heads.items()
            if isinstance(head, dict) and "metrics" in head
        }
        print(
            f"[nested-safety:inner] window={window_index} selected={chosen_name}",
            flush=True,
        )
        del inner_heads, inner_sources, inner_recipes
        _release_native_memory()
        outer_heads = _fit_nested_heads(
            panel,
            features,
            risk_features,
            outer_train_end,
            outer_valid_start,
            outer_valid_end,
            model_threads,
            max_train_rows,
        )
        outer_sources = {
            "e0": _score_frame(outer_heads["e0"]["predictions"], "e0"),
            "h1": _score_frame(outer_heads["h1"]["predictions"], "h1"),
            "safety": outer_heads["safety"]["predictions"],
        }
        outer_sources = _causal_eligible_predictions(
            outer_sources, panel, outer_valid_start, outer_valid_end
        )
        outer_recipes = _build_nested_recipe_frames(
            outer_sources["e0"], outer_sources["h1"], outer_sources["safety"],
            top_n=top_n, candidate_n=candidate_n,
        )
        selected_frame = outer_recipes[chosen_name].copy()
        selected_frame["model_variant"] = "nested_selected"
        baseline_frame = outer_recipes["R00_e0"].copy()
        baseline_frame["model_variant"] = "outer_e0_baseline"
        outer_labels = panel.loc[
            (panel["date"] >= outer_valid_start) & (panel["date"] <= outer_valid_end),
            ["code", "date", "entry_buyable", "target_net_ret_t1", "target_outcome_observed_t1"],
        ]
        outer_records, outer_metrics = _evaluate_recipe_frames(
            {"nested_selected": selected_frame, "outer_e0_baseline": baseline_frame},
            outer_labels,
            top_n=top_n,
        )
        outer_records["window"] = window_index
        all_outer_records.append(outer_records)
        outer_head_metrics = {
            name: head["metrics"]
            for name, head in outer_heads.items()
            if isinstance(head, dict) and "metrics" in head
        }
        manifest = {
            "window": window_index,
            "outer": {
                "train_end": str(outer_train_end.date()),
                "purge_date": str(outer_purge_date.date()),
                "valid_start": str(outer_valid_start.date()),
                "valid_end": str(outer_valid_end.date()),
            },
            "inner": {key: str(value.date()) if isinstance(value, pd.Timestamp) else value for key, value in inner.items()},
            "recipe_grid": list(_NESTED_RECIPE_GRID),
            "selected_recipe": chosen,
            "inner_metrics": inner_metrics,
            "outer_metrics": outer_metrics,
            "features": features,
            "risk_features": risk_features,
            "inner_heads": inner_head_metrics,
            "outer_heads": outer_head_metrics,
            "max_train_rows": int(max_train_rows),
            "development_only": True,
        }
        atomic_json(manifest, output_dir / f"window_{window_index}_manifest.json")
        atomic_parquet(inner_records, output_dir / f"window_{window_index}_inner_execution_records.parquet")
        atomic_parquet(outer_records, output_dir / f"window_{window_index}_outer_execution_records.parquet")
        atomic_parquet(selected_frame, output_dir / f"window_{window_index}_selected_predictions.parquet")
        window_reports.append({
            "window": window_index,
            "selected_recipe": chosen,
            "inner_metrics": inner_metrics,
            "outer_metrics": outer_metrics,
        })
        del (
            panel, outer_heads, outer_sources, outer_recipes, selected_frame,
            baseline_frame, inner_records, outer_records,
        )
        _release_native_memory()
        print(f"[nested-safety:window] completed={window_index}/3", flush=True)
    records = pd.concat(all_outer_records, ignore_index=True)
    combined = compare_execution_records(records)
    atomic_parquet(records, output_dir / "outer_execution_records.parquet")
    atomic_parquet(combined["daily_returns"], output_dir / "outer_daily_returns.parquet")
    report = {
        "experiment": "nested_exit_safety_h1_fusion_v2_fill_gate",
        "development_only": True,
        "min_inner_fill_ratio": _NESTED_MIN_FILL_RATIO,
        "top_n": int(top_n),
        "candidate_n": int(candidate_n),
        "inner_valid_days": int(inner_valid_days),
        "recipe_grid": list(_NESTED_RECIPE_GRID),
        "windows": window_reports,
        "combined": _json_report(combined),
    }
    atomic_json(report, output_dir / "nested_safety_report.json")
    return report


def model_dependencies() -> dict:
    return {
        "ensemble_legs": [
            model.train_ridge.__name__,
            model.train_lightgbm_ranker.__name__,
            model.train_elastic_net.__name__,
            model.train_extra_trees.__name__,
        ],
        "trained_variants": [asdict(variant) for variant in DEFAULT_TRAINED_VARIANTS],
        "external_prediction_source": "active_quant_short_predictions.parquet",
    }
