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
from intraday_1400.storage import atomic_json, atomic_parquet
from intraday_1400.structural_combo import (
    apply_probability_calibrator,
    build_e0_e4_staged_scores,
    build_structural_combo_scores,
    fit_probability_calibrator,
    shift_daily_prior_to_signal,
)
from quant import config as quant_config


TRAIN_END = pd.Timestamp("2025-08-15")
CALIBRATION_START = pd.Timestamp("2025-08-21")
CALIBRATION_END = pd.Timestamp("2025-09-12")
VALID_START = pd.Timestamp("2025-09-18")
VALID_END = pd.Timestamp("2025-10-31")


def _selected_feature_window(screening_report_path: Path) -> dict:
    report = json.loads(Path(screening_report_path).read_text(encoding="utf-8"))
    eligible = [
        item
        for item in report["windows"]
        if pd.Timestamp(item["train_end"]) <= TRAIN_END
    ]
    if not eligible:
        raise ValueError(f"no screening window is causal for train_end={TRAIN_END.date()}")
    return max(eligible, key=lambda item: pd.Timestamp(item["train_end"]))


def _selected_features(screening_report_path: Path) -> tuple[list[str], list[str], list[str]]:
    selected = _selected_feature_window(screening_report_path)["selected"][
        "daily_asof_plus_minute_control"
    ]
    base = list(selected["asof_matched"])
    minute = list(selected["minute"])
    return base, minute, [*base, *minute]


def _calibrate_classifier(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, dict]:
    calibration = predictions[
        (predictions["date"] >= CALIBRATION_START)
        & (predictions["date"] <= CALIBRATION_END)
    ][["code", "date", "raw_pred"]].merge(
        labels[["code", "date", target]],
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    calibrator = fit_probability_calibrator(calibration["raw_pred"], calibration[target])
    evaluation = predictions[
        (predictions["date"] >= VALID_START)
        & (predictions["date"] <= VALID_END)
    ].copy()
    evaluation["raw_pred"] = apply_probability_calibrator(
        evaluation["raw_pred"], calibrator
    ).to_numpy()
    return evaluation, calibrator


def _evaluation_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[
        (predictions["date"] >= VALID_START)
        & (predictions["date"] <= VALID_END)
    ].copy()


def _fit_liquidation_head(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    model_threads: int,
    min_open_samples: int = 20,
    prediction_end: pd.Timestamp = VALID_END,
) -> dict:
    train = panel[(panel["date"] <= TRAIN_END) & panel[target].notna()]
    open_samples = int((pd.to_numeric(train[target], errors="coerce") < 0.5).sum())
    if open_samples >= int(min_open_samples):
        return _fit_four_classifier_head(
            panel,
            features,
            target,
            TRAIN_END,
            CALIBRATION_START,
            prediction_end,
            model_threads,
            minority_weight=20.0,
        )
    probability = float(pd.to_numeric(train[target], errors="coerce").mean())
    predictions = panel[
        (panel["date"] >= CALIBRATION_START) & (panel["date"] <= prediction_end)
    ][["code", "date"]].copy()
    predictions["raw_pred"] = probability
    return {
        "predictions": predictions,
        "metrics": {
            "constant_empirical": {
                "train_rows": int(len(train)),
                "open_samples": open_samples,
                "probability": probability,
                "reason": f"fewer than {int(min_open_samples)} open-T3 training samples",
            }
        },
        "weights": {"constant_empirical": 1.0},
        "target": target,
    }


def run_structural_combo_experiment(
    labels_path: Path,
    screening_report_path: Path,
    output_dir: Path,
    daily_dir: Path | None = None,
    intraday_dir: Path | None = None,
    active_predictions_path: Path | None = None,
    model_threads: int = 8,
) -> dict:
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["code"] = labels["code"].astype(str).str[:6]
    feature_window = _selected_feature_window(screening_report_path)
    base, minute, features = _selected_features(screening_report_path)
    daily_dir = Path(daily_dir or default_daily_prepared_dir())
    intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    panel, _ = load_joined_prepared(
        daily_dir,
        intraday_dir,
        pd.Timestamp("2025-07-01"),
        VALID_END,
        daily_features=[name.removeprefix("asof__") for name in base],
        asof_features=[name.removeprefix("asof__") for name in base],
        minute_features=[name.removeprefix("minute__") for name in minute],
    )
    panel = panel.merge(labels, on=["code", "date"], how="inner", validate="one_to_one")
    panel["target_e0_direct"] = panel["adaptive_stress_net_ret_t3"]
    panel["target_e1_buy"] = panel["adaptive_entry_buyable"]
    panel["target_e2_liquidate"] = panel["adaptive_liquidated_by_t3"]
    panel["target_e3_return"] = panel["adaptive_realized_net_ret_t3"]
    head_args = (TRAIN_END, CALIBRATION_START, VALID_END, model_threads)
    print("[structural-combo] fit=E0 direct", flush=True)
    e0 = _fit_four_model_head(panel, features, "target_e0_direct", *head_args)
    print("[structural-combo] fit=E1 buy", flush=True)
    e1 = _fit_four_classifier_head(
        panel, features, "target_e1_buy", *head_args, minority_weight=10.0
    )
    print("[structural-combo] fit=E2 liquidate", flush=True)
    e2 = _fit_liquidation_head(
        panel, features, "target_e2_liquidate", model_threads
    )
    print("[structural-combo] fit=E3 conditional return", flush=True)
    e3 = _fit_four_model_head(panel, features, "target_e3_return", *head_args)
    e1_eval, e1_calibrator = _calibrate_classifier(
        e1["predictions"], panel, "target_e1_buy"
    )
    e2_eval, e2_calibrator = _calibrate_classifier(
        e2["predictions"], panel, "target_e2_liquidate"
    )
    minute_scores = build_structural_combo_scores(
        _evaluation_frame(e0["predictions"]),
        e1_eval,
        e2_eval,
        _evaluation_frame(e3["predictions"]),
    )
    active_predictions_path = Path(
        active_predictions_path
        or Path(quant_config.QUANT_DIR) / "active_quant_short_predictions.parquet"
    )
    daily = pd.read_parquet(active_predictions_path, columns=["code", "date", "pred"])
    calendar = pd.DatetimeIndex(pd.to_datetime(panel["date"].unique())).sort_values()
    daily_prior = shift_daily_prior_to_signal(daily, calendar)
    daily_prior = daily_prior[
        (daily_prior["date"] >= VALID_START)
        & (daily_prior["date"] <= VALID_END)
        & daily_prior["code"].isin(panel["code"].unique())
    ].copy()
    daily_prior["score"] = daily_prior["e4_daily_prior"]
    daily_prior["model_variant"] = "e4_daily_top10"
    staged = build_e0_e4_staged_scores(
        daily_prior,
        minute_scores["c2_fixed_50_50"],
        candidate_n=100,
    )
    predictions = {
        "e4_daily_top10": daily_prior,
        "e0_direct": minute_scores["c0_direct"],
        "e1_e3_structural": minute_scores["c1_structural"],
        "e0_e3_minute_block": minute_scores["c2_fixed_50_50"],
        "e0_e4_staged_combo": staged["e0_e4_staged_combo"],
    }
    execution_labels = panel[
        (panel["date"] >= VALID_START) & (panel["date"] <= VALID_END)
    ][["code", "date", "adaptive_entry_buyable", "adaptive_realized_net_ret_t3", "adaptive_horizon_observed_t3"]].rename(columns={
        "adaptive_entry_buyable": "entry_buyable",
        "adaptive_realized_net_ret_t3": "target_net_ret_t1",
        "adaptive_horizon_observed_t3": "target_outcome_observed_t1",
    })
    records = []
    per_model = {}
    for name, frame in predictions.items():
        model_records, comparison = simulate_fixed_exit_race(
            {name: frame},
            execution_labels,
            ExecutionConfig(top_n=10, roundtrip_cost=0.002, unsellable_return=-0.10),
        )
        records.append(model_records)
        per_model[name] = comparison["models"][name]
    execution_records = pd.concat(records, ignore_index=True)
    comparison = compare_execution_records(execution_records)
    report = {
        "experiment": "e0_e4_staged_structural_combo_v1",
        "development_only": True,
        "feature_screening_train_end": str(pd.Timestamp(feature_window["train_end"]).date()),
        "split": {
            "train_end": str(TRAIN_END.date()),
            "calibration_start": str(CALIBRATION_START.date()),
            "calibration_end": str(CALIBRATION_END.date()),
            "valid_start": str(VALID_START.date()),
            "valid_end": str(VALID_END.date()),
            "purge_sessions": 3,
        },
        "variants": list(predictions),
        "calibration": {"e1_buy": e1_calibrator, "e2_liquidate": e2_calibrator},
        "heads": {
            "e0": e0["metrics"],
            "e1": e1["metrics"],
            "e2": e2["metrics"],
            "e3": e3["metrics"],
        },
        "comparison": _json_report(comparison),
    }
    output_dir = Path(output_dir)
    atomic_parquet(execution_records, output_dir / "execution_records.parquet")
    atomic_parquet(comparison["daily_returns"], output_dir / "daily_returns.parquet")
    for name, frame in predictions.items():
        atomic_parquet(frame, output_dir / f"{name}_predictions.parquet")
    atomic_json(report, output_dir / "structural_combo_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot E0-E4 staged structural combination experiment")
    parser.add_argument(
        "--labels", type=Path,
        default=config.DATA_ROOT / "adaptive_label_pilot_2025q3" / "pilot_adaptive_labels.parquet",
    )
    parser.add_argument(
        "--screening-report", type=Path,
        default=config.REPORT_DIR / "fair_race_feature_screening.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=config.DATA_ROOT / "structural_combo_pilot_2025q3",
    )
    parser.add_argument("--model-threads", type=int, default=8)
    args = parser.parse_args()
    result = run_structural_combo_experiment(
        args.labels,
        args.screening_report,
        args.output_dir,
        model_threads=args.model_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
