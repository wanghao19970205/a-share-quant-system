from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_1400 import config, direct_return_experiment
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.storage import artifact_hash, atomic_json
from quant import model


PROTOCOL = "intraday_1400_target_redesign_v1"
CONSUMED_HOLDOUT_END = pd.Timestamp("2026-08-03")
FEATURE_SCREENING_TRAIN_END = pd.Timestamp("2025-08-06")
TARGET_FAMILIES = (
    "downside_quantile",
    "cross_sectional_rank",
    "conditional_payoff",
)
DEVELOPMENT_DAYS = 47
PURGE_DAYS = 3
CALIBRATION_DAYS = 10
HOLDOUT_DAYS = 60
TOTAL_DATES = 123
TOP_N = 10
DOWNSIDE_ALPHA = 0.20
STRESS_RETURN = -0.10
SPLIT_POSITIONS = {
    "development": [0, 46],
    "purge_1": [47, 49],
    "calibration": [50, 59],
    "purge_2": [60, 62],
    "holdout": [63, 122],
}

MODEL_RECIPES = {
    "downside_quantile": {
        "model": "lightgbm_quantile",
        "source_target": "adaptive_stress_net_ret_t3",
        "objective": "quantile",
        "alpha": DOWNSIDE_ALPHA,
        "n_estimators": 200,
        "learning_rate": 0.015,
        "early_stopping_rounds": 0,
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
    },
    "cross_sectional_rank": {
        "target": "target_cross_sectional_rank",
        "source_target": "adaptive_stress_net_ret_t3",
        "models": ["ridge", "lightgbm_ranker", "elastic_net", "extra_trees"],
        "weights": {
            "ridge": 0.15,
            "lightgbm_ranker": 0.55,
            "elastic_net": 0.10,
            "extra_trees": 0.20,
        },
        "rank_bins": 5,
        "n_estimators": {"lightgbm_ranker": 200, "extra_trees": 80},
        "learning_rate": 0.015,
        "early_stopping_rounds": 0,
        "max_train_rows": 150_000,
    },
    "conditional_payoff": {
        "heads": ["entry_probability", "exit_probability_given_entry", "conditional_return"],
        "classifier_weights": {
            "ridge_classifier": 0.15,
            "lightgbm_classifier": 0.55,
            "elastic_logistic": 0.10,
            "extra_trees_classifier": 0.20,
        },
        "regressor_weights": {
            "ridge": 0.15,
            "lightgbm": 0.55,
            "elastic_net": 0.10,
            "extra_trees": 0.20,
        },
        "stress_return": STRESS_RETURN,
        "conditional_return_clip": [-0.10, 0.10],
        "probability_calibration": "platt_on_calibration_only",
    },
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "parent_consumed_through": str(CONSUMED_HOLDOUT_END.date()),
        "feature_screening_train_end": str(FEATURE_SCREENING_TRAIN_END.date()),
        "target_families": list(TARGET_FAMILIES),
        "model_recipes": MODEL_RECIPES,
        "split": {
            "development_days": DEVELOPMENT_DAYS,
            "purge_1_days": PURGE_DAYS,
            "calibration_days": CALIBRATION_DAYS,
            "purge_2_days": PURGE_DAYS,
            "holdout_days": HOLDOUT_DAYS,
            "total_dates": TOTAL_DATES,
        },
        "execution": {
            "top_n": TOP_N,
            "entry_time": "14:50",
            "roundtrip_cost": 0.002,
            "unsellable_return": STRESS_RETURN,
            "max_exit_sessions": 3,
            "fixed_capital_cash_slots": True,
        },
        "production_publication": False,
        "human_approval_required": True,
    }


def _normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "code", "date", "adaptive_entry_buyable", "adaptive_liquidated_by_t3",
        "adaptive_realized_net_ret_t3", "adaptive_stress_net_ret_t3",
        "adaptive_horizon_observed_t3",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"target-redesign labels missing {sorted(missing)}")
    data = labels.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str[:6]
    data = data.dropna(subset=["date", "code"])
    if data.duplicated(["date", "code"]).any():
        raise ValueError("target-redesign labels require unique date+code keys")
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def _expected_label_keys(prepared_dir: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    wanted = pd.DatetimeIndex(dates).normalize().drop_duplicates().sort_values()
    parts = []
    for month in wanted.strftime("%Y-%m").unique():
        path = Path(prepared_dir) / f"{month}.parquet"
        if not path.exists():
            continue
        available = pd.read_parquet(path, columns=["code", "date", "signal_eligible"])
        available["date"] = pd.to_datetime(available["date"], errors="coerce").dt.normalize()
        available["code"] = available["code"].astype(str).str[:6]
        available = available[
            available["date"].isin(wanted)
            & available["signal_eligible"].fillna(False).astype(bool)
        ]
        parts.append(available[["date", "code"]])
    if not parts:
        return pd.DataFrame(columns=["date", "code"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"])


def _eligible_universe_hashes(
    expected_keys: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[str, str]:
    wanted = pd.DatetimeIndex(dates).normalize().drop_duplicates().sort_values()
    frame = expected_keys.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str[:6]
    hashes = {}
    for date in wanted:
        codes = sorted(frame.loc[frame["date"] == date, "code"].drop_duplicates())
        hashes[str(pd.Timestamp(date).date())] = _canonical_hash(codes)
    return hashes


def mature_dates_after_consumed(
    labels: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    expected_keys: pd.DataFrame,
) -> pd.DatetimeIndex:
    data = _normalize_labels(labels)
    calendar = pd.DatetimeIndex(trading_calendar).normalize().drop_duplicates().sort_values()
    candidate_dates = calendar[calendar > CONSUMED_HOLDOUT_END]
    expected = expected_keys.copy()
    expected["date"] = pd.to_datetime(expected["date"], errors="coerce").dt.normalize()
    expected["code"] = expected["code"].astype(str).str[:6]
    expected = expected[expected["date"].isin(candidate_dates)].drop_duplicates(["date", "code"])
    extra = data[data["date"] > CONSUMED_HOLDOUT_END][["date", "code"]].merge(
        expected, on=["date", "code"], how="left", indicator=True
    )
    if (extra["_merge"] == "left_only").any():
        raise ValueError("target-redesign labels contain keys outside the prepared eligible universe")
    mature = []
    for date, expected_day in expected.groupby("date", sort=True):
        day = data[data["date"] == pd.Timestamp(date)]
        if set(day["code"]) != set(expected_day["code"]):
            continue
        observed = day["adaptive_horizon_observed_t3"].fillna(False).astype(bool)
        stress_complete = pd.to_numeric(
            day["adaptive_stress_net_ret_t3"], errors="coerce"
        ).notna()
        if observed.all() and stress_complete.all():
            mature.append(pd.Timestamp(date))
    return pd.DatetimeIndex(mature).sort_values()


def registered_split(mature_dates: pd.DatetimeIndex) -> dict:
    dates = pd.DatetimeIndex(mature_dates).normalize().drop_duplicates().sort_values()
    dates = dates[dates > CONSUMED_HOLDOUT_END]
    if len(dates) < TOTAL_DATES:
        raise ValueError(
            f"target redesign requires {TOTAL_DATES} mature dates after "
            f"{CONSUMED_HOLDOUT_END.date()}, found {len(dates)}"
        )
    dates = dates[:TOTAL_DATES]
    development = dates[:DEVELOPMENT_DAYS]
    purge_1 = dates[DEVELOPMENT_DAYS:DEVELOPMENT_DAYS + PURGE_DAYS]
    calibration_start = DEVELOPMENT_DAYS + PURGE_DAYS
    calibration = dates[calibration_start:calibration_start + CALIBRATION_DAYS]
    purge_2_start = calibration_start + CALIBRATION_DAYS
    purge_2 = dates[purge_2_start:purge_2_start + PURGE_DAYS]
    holdout = dates[purge_2_start + PURGE_DAYS:]
    result = {
        "development": development,
        "purge_1": purge_1,
        "calibration": calibration,
        "purge_2": purge_2,
        "holdout": holdout,
    }
    expected = {
        "development": DEVELOPMENT_DAYS,
        "purge_1": PURGE_DAYS,
        "calibration": CALIBRATION_DAYS,
        "purge_2": PURGE_DAYS,
        "holdout": HOLDOUT_DAYS,
    }
    if any(len(result[name]) != count for name, count in expected.items()):
        raise AssertionError("target-redesign positional split is incomplete")
    return result


def split_manifest(split: dict, trading_calendar: pd.DatetimeIndex) -> dict:
    date_segments = {
        name: [str(pd.Timestamp(value).date()) for value in values]
        for name, values in split.items()
    }
    ordered_dates = [date for name in SPLIT_POSITIONS for date in date_segments[name]]
    calendar = pd.DatetimeIndex(trading_calendar).normalize().drop_duplicates().sort_values()
    calendar_strings = [str(pd.Timestamp(value).date()) for value in calendar]
    expected = [
        value for value in calendar_strings
        if pd.Timestamp(value) > CONSUMED_HOLDOUT_END
    ][:TOTAL_DATES]
    if ordered_dates != expected:
        raise ValueError("target-redesign split must equal the first 123 prepared trading sessions")
    manifest = {
        "protocol": PROTOCOL,
        "source_start_exclusive": str(CONSUMED_HOLDOUT_END.date()),
        "total_dates": TOTAL_DATES,
        "positions": SPLIT_POSITIONS,
        "dates": date_segments,
        "trading_calendar_evidence": {
            "dates": calendar_strings,
            "sha256": _canonical_hash(calendar_strings),
        },
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    return manifest


def build_target_columns(labels: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_labels(labels)
    observed = data["adaptive_horizon_observed_t3"].fillna(False).astype(bool)
    entered = data["adaptive_entry_buyable"].fillna(False).astype(bool)
    liquidated = data["adaptive_liquidated_by_t3"].fillna(False).astype(bool)
    stress = pd.to_numeric(data["adaptive_stress_net_ret_t3"], errors="coerce").where(observed)
    data["target_downside_source"] = stress
    data["target_cross_sectional_rank"] = stress.groupby(data["date"]).rank(
        method="average", pct=True
    ) - 0.5
    data["target_entry_buyable"] = entered.astype(float).where(observed)
    data["target_exit_t3_given_entry"] = liquidated.astype(float).where(observed & entered)
    data["target_conditional_return"] = pd.to_numeric(
        data["adaptive_realized_net_ret_t3"], errors="coerce"
    ).where(observed & entered & liquidated)
    return data


def conditional_payoff_scores(
    entry_probability: pd.Series,
    exit_probability_given_entry: pd.Series,
    conditional_return: pd.Series,
) -> pd.Series:
    p_entry = pd.to_numeric(entry_probability, errors="coerce").clip(0.0, 1.0)
    p_exit = pd.to_numeric(exit_probability_given_entry, errors="coerce").clip(0.0, 1.0)
    payoff = pd.to_numeric(conditional_return, errors="coerce").clip(-0.10, 0.10)
    return p_entry * (p_exit * payoff + (1.0 - p_exit) * STRESS_RETURN)


def fit_downside_quantile_head(
    panel: pd.DataFrame,
    features: list[str],
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    model_threads: int,
):
    recipe = MODEL_RECIPES["downside_quantile"]
    return model.train_lightgbm(
        panel=panel,
        features=features,
        horizon=1,
        train_end=str(pd.Timestamp(train_end).date()),
        valid_end=str(pd.Timestamp(valid_end).date()),
        predict_start=str(pd.Timestamp(valid_start).date()),
        decay_half_life_days=recipe["decay_half_life_days"],
        min_weight=recipe["min_weight"],
        n_estimators=recipe["n_estimators"],
        learning_rate=recipe["learning_rate"],
        early_stopping_rounds=recipe["early_stopping_rounds"],
        n_jobs=model_threads,
        label_col="target_downside_source",
        train_mask_col=None,
        objective=recipe["objective"],
        alpha=recipe["alpha"],
    )


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _seal_state(state: dict) -> dict:
    result = dict(state)
    result.pop("state_hash", None)
    result["state_hash"] = _canonical_hash(result)
    return result


def _validate_parent_evidence(
    parent_state: dict,
    parent_holdout_report_path: Path,
) -> None:
    direct_return_experiment._validate_holdout_state(parent_state)
    if parent_state.get("protocol") != "intraday_1400_direct_return_v1":
        raise RuntimeError("target redesign requires the direct-return parent protocol")
    if parent_state.get("status") != "consumed":
        raise RuntimeError("target redesign requires a consumed parent holdout")
    artifacts = parent_state.get("artifacts", {})
    if set(artifacts) != {"report", "execution_records", "account_daily_returns"}:
        raise RuntimeError("parent consumed artifact ledger is incomplete")
    for name, evidence in artifacts.items():
        path = Path(evidence.get("path", ""))
        if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"parent consumed {name} artifact is missing or changed")
    report_evidence = artifacts["report"]
    report_path = Path(parent_holdout_report_path).resolve()
    if (
        report_evidence.get("path") != str(report_path)
        or report_evidence.get("sha256") != artifact_hash(report_path)
    ):
        raise RuntimeError("parent holdout report does not match its consumed artifact ledger")


def _validate_ready_split(state: dict) -> None:
    evidence = state.get("split_manifest", {})
    path = Path(evidence.get("path", ""))
    if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
        raise RuntimeError("target-redesign split manifest is missing or changed")
    manifest = _read_json(path)
    content = dict(manifest)
    manifest_hash = content.pop("manifest_hash", None)
    if manifest_hash != _canonical_hash(content):
        raise RuntimeError("target-redesign split manifest was modified")
    if manifest.get("protocol") != PROTOCOL or manifest.get("total_dates") != TOTAL_DATES:
        raise RuntimeError("target-redesign split manifest protocol changed")
    if manifest.get("source_start_exclusive") != str(CONSUMED_HOLDOUT_END.date()):
        raise RuntimeError("target-redesign split source boundary changed")
    if manifest.get("positions") != SPLIT_POSITIONS:
        raise RuntimeError("target-redesign split positions changed")
    all_dates = []
    for name, expected_count in (
        ("development", DEVELOPMENT_DAYS),
        ("purge_1", PURGE_DAYS),
        ("calibration", CALIBRATION_DAYS),
        ("purge_2", PURGE_DAYS),
        ("holdout", HOLDOUT_DAYS),
    ):
        values = manifest.get("dates", {}).get(name, [])
        if len(values) != expected_count:
            raise RuntimeError(f"target-redesign split {name} is incomplete")
        all_dates.extend(pd.to_datetime(values))
    ordered = pd.DatetimeIndex(all_dates)
    if len(ordered) != TOTAL_DATES or not ordered.equals(ordered.sort_values().drop_duplicates()):
        raise RuntimeError("target-redesign split dates are not strictly ordered and unique")
    if ordered[0] <= CONSUMED_HOLDOUT_END:
        raise RuntimeError("target-redesign split overlaps the consumed parent holdout")
    ordered_strings = [str(pd.Timestamp(value).date()) for value in ordered]
    calendar_evidence = manifest.get("trading_calendar_evidence", {})
    calendar_strings = list(calendar_evidence.get("dates", []))
    if calendar_evidence.get("sha256") != _canonical_hash(calendar_strings):
        raise RuntimeError("target-redesign frozen trading calendar evidence changed")
    expected = [
        value for value in calendar_strings
        if pd.Timestamp(value) > CONSUMED_HOLDOUT_END
    ][:TOTAL_DATES]
    if ordered_strings != expected:
        raise RuntimeError("target-redesign split dates do not match the frozen trading calendar")


def initialize_or_refresh(
    state_dir: Path,
    parent_holdout_state_path: Path,
    parent_holdout_report_path: Path,
    labels_path: Path | None = None,
    prepared_dir: Path | None = None,
) -> dict:
    parent_state = _read_json(parent_holdout_state_path)
    parent_report = _read_json(parent_holdout_report_path)
    _validate_parent_evidence(parent_state, parent_holdout_report_path)
    if parent_report.get("freeze_hash") != parent_state.get("freeze_hash"):
        raise RuntimeError("parent holdout report freeze hash does not match its consumed state")
    if parent_report.get("input_hashes") != parent_state.get("input_hashes"):
        raise RuntimeError("parent holdout report inputs do not match its consumed state")
    if parent_report.get("next_branch") != "target_redesign":
        raise RuntimeError("parent holdout did not select target_redesign")
    if parent_report.get("holdout_end") != str(CONSUMED_HOLDOUT_END.date()):
        raise RuntimeError("target-redesign parent holdout boundary changed")
    state_dir = Path(state_dir)
    state_path = state_dir / "manifest.json"
    existing = _read_json(state_path) if state_path.exists() else None
    parent_evidence = {
        "state": {
            "path": str(Path(parent_holdout_state_path).resolve()),
            "sha256": artifact_hash(parent_holdout_state_path),
        },
        "report": {
            "path": str(Path(parent_holdout_report_path).resolve()),
            "sha256": artifact_hash(parent_holdout_report_path),
        },
    }
    if existing:
        content = dict(existing)
        state_hash = content.pop("state_hash", None)
        if state_hash != _canonical_hash(content):
            raise RuntimeError("target-redesign state was modified")
        if existing.get("protocol_hash") != _canonical_hash(protocol_payload()):
            raise RuntimeError("target-redesign protocol changed; create a new state directory")
        if existing.get("parent_evidence") != parent_evidence:
            raise RuntimeError("target-redesign parent evidence changed")
        if existing.get("status") == "development_ready":
            _validate_ready_split(existing)
            return existing
        if labels_path is None:
            return existing
    available_dates = pd.DatetimeIndex([])
    label_snapshot = None
    if labels_path is not None:
        labels = pd.read_parquet(labels_path)
        prepared_path = Path(prepared_dir or config.PREPARED_DIR)
        calendar = load_trading_calendar(prepared_path)
        expected_keys = _expected_label_keys(prepared_path, calendar[calendar > CONSUMED_HOLDOUT_END])
        available_dates = mature_dates_after_consumed(labels, calendar, expected_keys)
        mature_date_strings = [str(pd.Timestamp(value).date()) for value in available_dates]
        universe_hashes = _eligible_universe_hashes(expected_keys, available_dates)
        if existing:
            previous_snapshot = existing.get("latest_label_snapshot") or {}
            previous_dates = list(previous_snapshot.get("mature_date_list", []))
            if mature_date_strings[:len(previous_dates)] != previous_dates:
                raise RuntimeError("target-redesign mature dates must extend the previous exact prefix")
            previous_hashes = previous_snapshot.get("eligible_universe_hashes", {})
            if any(universe_hashes.get(date) != value for date, value in previous_hashes.items()):
                raise RuntimeError("target-redesign historical eligible universe changed")
        label_snapshot = {
            "path": str(Path(labels_path).resolve()),
            "sha256": artifact_hash(labels_path),
            "mature_dates": int(len(available_dates)),
            "mature_date_list": mature_date_strings,
            "eligible_universe_hashes": universe_hashes,
            "latest_mature_date": (
                str(pd.Timestamp(available_dates[-1]).date()) if len(available_dates) else None
            ),
        }
    state = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(protocol_payload()),
        "status": "awaiting_123_mature_dates",
        "required_mature_dates": TOTAL_DATES,
        "available_mature_dates": int(len(available_dates)),
        "source_start_exclusive": str(CONSUMED_HOLDOUT_END.date()),
        "parent_evidence": parent_evidence,
        "latest_label_snapshot": label_snapshot,
        "target_families": list(TARGET_FAMILIES),
        "production_isolated": True,
        "human_approval_required": True,
        "production_publication": False,
    }
    if len(available_dates) >= TOTAL_DATES:
        split = registered_split(available_dates)
        manifest = split_manifest(split, calendar)
        atomic_json(manifest, state_dir / "split_manifest.json")
        state["status"] = "development_ready"
        state["split_manifest"] = {
            "path": str((state_dir / "split_manifest.json").resolve()),
            "sha256": artifact_hash(state_dir / "split_manifest.json"),
        }
    state = _seal_state(state)
    atomic_json(state, state_path)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-redesign readiness controller")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--parent-holdout-state", type=Path, required=True)
    parser.add_argument("--parent-holdout-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--prepared-dir", type=Path, default=config.PREPARED_DIR)
    args = parser.parse_args()
    result = initialize_or_refresh(
        args.state_dir,
        args.parent_holdout_state,
        args.parent_holdout_report,
        args.labels,
        args.prepared_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
