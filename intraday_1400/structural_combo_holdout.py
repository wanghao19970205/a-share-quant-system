from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intraday_1400 import config
from intraday_1400.fair_race_pipeline import (
    _fit_four_classifier_head,
    _fit_four_model_head,
    _json_report,
    default_daily_prepared_dir,
    load_joined_prepared,
)
from intraday_1400.offline_race import ExecutionConfig, compare_execution_records, simulate_fixed_exit_race
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo import (
    apply_probability_calibrator,
    build_daily_execution_filter_scores,
    build_e0_e4_staged_scores,
    build_structural_combo_scores,
    fit_probability_calibrator,
    shift_daily_prior_to_signal,
)
from intraday_1400.structural_combo_experiment import (
    CALIBRATION_END,
    CALIBRATION_START,
    TRAIN_END,
    _fit_liquidation_head,
    _selected_feature_window,
    _selected_features,
)
from quant import config as quant_config


HOLDOUT_DAYS = 60


def holdout_input_hashes(
    training_labels_path: Path,
    holdout_labels_path: Path,
    screening_report_path: Path,
    daily_dir: Path,
    intraday_dir: Path,
    active_predictions_path: Path,
) -> dict:
    paths = {
        "training_labels": Path(training_labels_path),
        "holdout_labels": Path(holdout_labels_path),
        "screening_report": Path(screening_report_path),
        "daily_prepared": Path(daily_dir),
        "intraday_prepared": Path(intraday_dir),
        "daily_predictions": Path(active_predictions_path),
    }
    return {
        name: {"path": str(path.resolve()), "sha256": artifact_hash(path)}
        for name, path in paths.items()
    }


def first_holdout_dates(labels: pd.DataFrame, count: int = HOLDOUT_DAYS) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(labels["date"], errors="coerce").dropna().unique()).sort_values()
    if len(dates) < int(count):
        raise ValueError(f"holdout requires {int(count)} dates, found {len(dates)}")
    return dates[:int(count)]


def validated_holdout_dates(
    training_labels: pd.DataFrame,
    holdout_labels: pd.DataFrame,
    count: int = HOLDOUT_DAYS,
) -> pd.DatetimeIndex:
    dates = first_holdout_dates(holdout_labels, count)
    training_end = pd.Timestamp(pd.to_datetime(training_labels["date"], errors="coerce").max())
    if pd.Timestamp(dates[0]) <= max(training_end, CALIBRATION_END):
        raise ValueError("holdout dates must begin after all training labels and calibration")
    return dates


def _calibrated_evaluation(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target: str,
    evaluation_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    calibration = predictions[
        (predictions["date"] >= CALIBRATION_START)
        & (predictions["date"] <= CALIBRATION_END)
    ][["code", "date", "raw_pred"]].merge(
        panel[["code", "date", target]],
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    calibrator = fit_probability_calibrator(calibration["raw_pred"], calibration[target])
    evaluation = predictions[predictions["date"].isin(evaluation_dates)].copy()
    evaluation["raw_pred"] = apply_probability_calibrator(
        evaluation["raw_pred"], calibrator
    ).to_numpy()
    return evaluation, calibrator


def _evaluation_frame(predictions: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return predictions[predictions["date"].isin(dates)].copy()


def cash_normalized_execution_records(
    records: pd.DataFrame,
    dates: pd.DatetimeIndex,
    top_n: int = 10,
    models: list[str] | None = None,
) -> pd.DataFrame:
    normalized = [records.copy()]
    represented = sorted(records["model"].dropna().astype(str).unique())
    models = represented if models is None else sorted(str(model) for model in models)
    counts = records.groupby(["model", "signal_date"], sort=True).size()
    cash_rows = []
    for model in models:
        for date in dates:
            count = int(counts.get((model, pd.Timestamp(date)), 0))
            if count > int(top_n):
                raise ValueError(f"{model} selected {count} names above top_n={int(top_n)}")
            for slot in range(count, int(top_n)):
                cash_rows.append({
                    "model": model,
                    "signal_date": pd.Timestamp(date),
                    "code": f"cash_{slot:02d}",
                    "entry_buyable": False,
                    "exit_sellable": True,
                    "outcome_observed": True,
                    "net_return": 0.0,
                })
    if cash_rows:
        normalized.append(pd.DataFrame(cash_rows))
    return pd.concat(normalized, ignore_index=True, sort=False)


def _block_reports(
    records: pd.DataFrame,
    dates: pd.DatetimeIndex,
    models: list[str],
) -> list[dict]:
    reports = []
    for index in range(3):
        block = dates[index * 20:(index + 1) * 20]
        selected = records[records["signal_date"].isin(block)].copy()
        account = cash_normalized_execution_records(selected, block, models=models)
        reports.append({
            "block": index + 1,
            "start": str(pd.Timestamp(block[0]).date()),
            "end": str(pd.Timestamp(block[-1]).date()),
            "comparison": _json_report(compare_execution_records(selected)),
            "account_comparison": _json_report(compare_execution_records(account)),
        })
    return reports


def run_frozen_holdout(
    training_labels_path: Path,
    holdout_labels_path: Path,
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    active_predictions_path: Path | None = None,
    model_threads: int = 8,
    expected_input_hashes: dict | None = None,
) -> dict:
    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    active_predictions_path = Path(
        active_predictions_path
        or Path(quant_config.QUANT_DIR) / "active_quant_short_predictions.parquet"
    )
    input_hashes = holdout_input_hashes(
        training_labels_path,
        holdout_labels_path,
        screening_report_path,
        daily_dir,
        intraday_dir,
        active_predictions_path,
    )
    if expected_input_hashes is not None and input_hashes != expected_input_hashes:
        raise RuntimeError("holdout inputs changed after claim")
    training_labels = pd.read_parquet(training_labels_path)
    holdout_labels = pd.read_parquet(holdout_labels_path)
    for frame in (training_labels, holdout_labels):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["code"] = frame["code"].astype(str).str[:6]
    evaluation_dates = validated_holdout_dates(
        training_labels, holdout_labels, HOLDOUT_DAYS
    )
    training_label_end = pd.Timestamp(training_labels["date"].max())
    evaluation_end = pd.Timestamp(evaluation_dates[-1])
    labels = pd.concat([
        training_labels,
        holdout_labels[holdout_labels["date"].isin(evaluation_dates)],
    ], ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    feature_window = _selected_feature_window(screening_report_path)
    base, minute, features = _selected_features(screening_report_path)
    panel, _ = load_joined_prepared(
        daily_dir,
        intraday_dir,
        pd.Timestamp("2025-07-01"),
        evaluation_end,
        daily_features=[name.removeprefix("asof__") for name in base],
        asof_features=[name.removeprefix("asof__") for name in base],
        minute_features=[name.removeprefix("minute__") for name in minute],
    )
    panel = panel.merge(labels, on=["code", "date"], how="inner", validate="one_to_one")
    panel["target_e0_direct"] = panel["adaptive_stress_net_ret_t3"]
    panel["target_e1_buy"] = panel["adaptive_entry_buyable"]
    panel["target_e2_liquidate"] = panel["adaptive_liquidated_by_t3"]
    panel["target_e3_return"] = panel["adaptive_realized_net_ret_t3"]
    head_args = (TRAIN_END, CALIBRATION_START, evaluation_end, model_threads)
    print("[structural-holdout] fit=E0 direct", flush=True)
    e0 = _fit_four_model_head(panel, features, "target_e0_direct", *head_args)
    print("[structural-holdout] fit=E1 buy", flush=True)
    e1 = _fit_four_classifier_head(
        panel, features, "target_e1_buy", *head_args, minority_weight=10.0
    )
    print("[structural-holdout] fit=E2 liquidate", flush=True)
    e2 = _fit_liquidation_head(
        panel,
        features,
        "target_e2_liquidate",
        model_threads,
        prediction_end=evaluation_end,
    )
    print("[structural-holdout] fit=E3 conditional return", flush=True)
    e3 = _fit_four_model_head(panel, features, "target_e3_return", *head_args)
    e1_eval, e1_calibrator = _calibrated_evaluation(
        e1["predictions"], panel, "target_e1_buy", evaluation_dates
    )
    e2_eval, e2_calibrator = _calibrated_evaluation(
        e2["predictions"], panel, "target_e2_liquidate", evaluation_dates
    )
    minute_scores = build_structural_combo_scores(
        _evaluation_frame(e0["predictions"], evaluation_dates),
        e1_eval,
        e2_eval,
        _evaluation_frame(e3["predictions"], evaluation_dates),
    )
    daily = pd.read_parquet(active_predictions_path, columns=["code", "date", "pred"])
    calendar = pd.DatetimeIndex(pd.to_datetime(panel["date"].unique())).sort_values()
    daily_prior = shift_daily_prior_to_signal(daily, calendar)
    daily_prior = daily_prior[
        daily_prior["date"].isin(evaluation_dates)
        & daily_prior["code"].isin(panel["code"].unique())
    ].copy()
    daily_prior["score"] = daily_prior["e4_daily_prior"]
    daily_prior["model_variant"] = "e4_daily_top10"
    staged = build_e0_e4_staged_scores(
        daily_prior,
        minute_scores["c2_fixed_50_50"],
        candidate_n=100,
    )
    structural_staged = build_e0_e4_staged_scores(
        daily_prior,
        minute_scores["c1_structural"],
        candidate_n=100,
    )
    buy_filtered_daily = build_daily_execution_filter_scores(
        daily_prior,
        e1_eval,
        candidate_n=100,
        minimum_buy_probability=0.50,
    )
    predictions = {
        "e4_daily_top10": daily_prior,
        "e0_direct": minute_scores["c0_direct"],
        "e1_e3_structural": minute_scores["c1_structural"],
        "e0_e3_minute_block": minute_scores["c2_fixed_50_50"],
        "e0_e4_staged_combo": staged["e0_e4_staged_combo"],
        "h1_daily_top100_structural_rerank": structural_staged["e4_top100_minute_block"],
        "h2_daily_top100_buy_filter": buy_filtered_daily,
    }
    execution_labels = panel[panel["date"].isin(evaluation_dates)][[
        "code", "date", "adaptive_entry_buyable", "adaptive_realized_net_ret_t3",
        "adaptive_horizon_observed_t3",
    ]].rename(columns={
        "adaptive_entry_buyable": "entry_buyable",
        "adaptive_realized_net_ret_t3": "target_net_ret_t1",
        "adaptive_horizon_observed_t3": "target_outcome_observed_t1",
    })
    records = []
    for name, frame in predictions.items():
        model_records, _ = simulate_fixed_exit_race(
            {name: frame},
            execution_labels,
            ExecutionConfig(top_n=10, roundtrip_cost=0.002, unsellable_return=-0.10),
        )
        records.append(model_records)
    execution_records = pd.concat(records, ignore_index=True)
    final_input_hashes = holdout_input_hashes(
        training_labels_path,
        holdout_labels_path,
        screening_report_path,
        daily_dir,
        intraday_dir,
        active_predictions_path,
    )
    if final_input_hashes != input_hashes:
        raise RuntimeError("holdout inputs changed during evaluation")
    comparison = compare_execution_records(execution_records)
    account_records = cash_normalized_execution_records(
        execution_records, evaluation_dates, models=list(predictions)
    )
    account_comparison = compare_execution_records(account_records)
    report = {
        "experiment": "frozen_e0_e4_structural_holdout_60d_v1",
        "untouched_holdout": expected_input_hashes is not None,
        "input_hashes": input_hashes,
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "training_cutoff": str(TRAIN_END.date()),
        "training_labels_end": str(training_label_end.date()),
        "calibration_start": str(CALIBRATION_START.date()),
        "calibration_end": str(CALIBRATION_END.date()),
        "holdout_start": str(pd.Timestamp(evaluation_dates[0]).date()),
        "holdout_end": str(evaluation_end.date()),
        "holdout_days": int(len(evaluation_dates)),
        "variants": list(predictions),
        "calibration": {"e1_buy": e1_calibrator, "e2_liquidate": e2_calibrator},
        "comparison": _json_report(comparison),
        "account_comparison": _json_report(account_comparison),
        "account_return_definition": "sum selected net returns divided by fixed top_n=10; missing slots are cash at zero return",
        "twenty_day_blocks": _block_reports(
            execution_records, evaluation_dates, list(predictions)
        ),
        "daily_history_causal": False,
        "daily_history_caveat": "active daily history has no row-level publication timestamps; daily-dependent variants are diagnostic only",
    }
    output_dir = Path(output_dir)
    atomic_parquet(execution_records, output_dir / "execution_records.parquet")
    atomic_parquet(comparison["daily_returns"], output_dir / "daily_returns.parquet")
    atomic_parquet(
        account_comparison["daily_returns"], output_dir / "account_daily_returns.parquet"
    )
    atomic_json(report, output_dir / "holdout_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen 60-day E0-E4 structural holdout")
    parser.add_argument(
        "--training-labels", type=Path,
        default=config.DATA_ROOT / "adaptive_label_full_2025q3" / "pilot_adaptive_labels.parquet",
    )
    parser.add_argument(
        "--holdout-labels", type=Path,
        default=config.DATA_ROOT / "adaptive_label_holdout_2025q4" / "pilot_adaptive_labels.parquet",
    )
    parser.add_argument(
        "--screening-report", type=Path,
        default=config.REPORT_DIR / "fair_race_feature_screening.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=config.DATA_ROOT / "structural_combo_holdout_60d",
    )
    parser.add_argument("--model-threads", type=int, default=8)
    args = parser.parse_args()
    result = run_frozen_holdout(
        args.training_labels,
        args.holdout_labels,
        args.screening_report,
        args.output_dir,
        model_threads=args.model_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
