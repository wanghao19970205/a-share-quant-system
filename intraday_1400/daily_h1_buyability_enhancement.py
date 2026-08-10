from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_1400 import config, pipeline
from intraday_1400.daily_minute_enhancement import (
    BASELINE as DAILY_BASELINE,
    _build_panel,
    _canonical_hash,
    _causal_screened_features,
    _cycle_lock,
    _evaluate_candidate,
    _fit_daily_head,
    _input_hashes as parent_input_hashes,
    _prepared_provenance,
    _validate_research_paths,
)
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.fair_race_pipeline import default_daily_prepared_dir
from intraday_1400.offline_race import compare_execution_records
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records
from intraday_1400.target_redesign_backfill import (
    FOLD_POSITIONS,
    TOP_N,
    combine_historical_labels,
    registered_folds,
    validate_historical_dates,
)
from quant import model

PROTOCOL = "intraday_1400_daily_h1_buyability_enhancement_v1"
BASELINE = "baseline"
CANDIDATES = (BASELINE, "h1_buy_zblend_50", "h1_buy_zblend_75", "h1_buy_constrained_top50")
BUY_LABEL = "adaptive_entry_buyable"
MODEL_RECIPE = {
    "daily_head": "daily_minute_enhancement._fit_daily_head",
    "buyability_head": {
        "classifier": "lightgbm",
        "features": "verified_1355_asof_plus_minute",
        "minority_weight": 1.0,
        "n_estimators": 160,
        "learning_rate": 0.02,
        "max_train_rows": 400000,
        "enforce_max_train_rows": True,
        "predict_scope": "current_fold_oos_only",
        "decay_half_life_days": 60.0,
        "min_weight": 0.03,
        "oos_calibration": False,
        "oos_threshold_selection": False,
    },
    "top_n": TOP_N,
    "cutoff_time": "13:55",
    "execution": "adaptive_t3_exact_top10_no_refill_fixed_capital",
}


def protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "parent_protocol": "intraday_1400_minute_feature_residualization_v1",
        "baseline": BASELINE,
        "candidate_grid": list(CANDIDATES),
        "fold_positions": FOLD_POSITIONS,
        "model_recipe": MODEL_RECIPE,
        "buyability_label": "adaptive_entry_buyable_train_fold_only",
        "selection_gate": {"minimum_fill_rate": 0.60, "minimum_fold_wins_vs_baseline": 3,
                           "mean_return_must_exceed_baseline": True, "max_drawdown_tolerance": 0.05},
        "four_point_five_percent_filter": False,
        "production_publication": False,
        "human_approval_required": True,
    }


def _zscore_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame["date"]).transform("mean")
    std = values.groupby(frame["date"]).transform("std")
    return ((values - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clipped_logit(probability: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    p = pd.to_numeric(probability, errors="coerce").clip(epsilon, 1.0 - epsilon)
    return np.log(p / (1.0 - p))


def build_buyability_scores(daily_h1: pd.DataFrame, p_buy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct scores without using any outcome, fill, threshold, or future data."""
    h1 = daily_h1[["code", "date", "score"]].rename(columns={"score": "daily_h1"})
    buy = p_buy[["code", "date", "pred"]].rename(columns={"pred": "p_buy"})
    merged = h1.merge(buy, on=["code", "date"], how="inner", validate="one_to_one")
    merged["h1_z"] = _zscore_by_date(merged, "daily_h1")
    merged["buy_logit"] = clipped_logit(merged["p_buy"])
    merged["buy_logit_z"] = _zscore_by_date(merged, "buy_logit")
    baseline = h1.assign(model_variant=BASELINE, score=h1["daily_h1"])
    z50 = merged.assign(model_variant="h1_buy_zblend_50", score=0.50 * merged["h1_z"] + 0.50 * merged["buy_logit_z"])
    z75 = merged.assign(model_variant="h1_buy_zblend_75", score=0.75 * merged["h1_z"] + 0.25 * merged["buy_logit_z"])
    ranked = merged.sort_values(["date", "daily_h1", "p_buy", "code"], ascending=[True, False, False, True])
    constrained = ranked.groupby("date", group_keys=False).head(50).copy()
    constrained = constrained.sort_values(["date", "p_buy", "daily_h1", "code"], ascending=[True, False, False, True])
    constrained["score"] = constrained["p_buy"]
    constrained["model_variant"] = "h1_buy_constrained_top50"
    return {name: value[["code", "date", "score", "model_variant"]].reset_index(drop=True)
            for name, value in ((BASELINE, baseline), ("h1_buy_zblend_50", z50),
                                ("h1_buy_zblend_75", z75), ("h1_buy_constrained_top50", constrained))}


def select_enhancement(aggregate: dict[str, dict], fold_metrics: list[dict]) -> dict:
    if set(aggregate) != set(CANDIDATES):
        raise ValueError("buyability selection requires the exact registered candidate grid")
    if [item.get("name") for item in fold_metrics] != [item["name"] for item in FOLD_POSITIONS]:
        raise ValueError("buyability selection requires the ordered registered four folds")
    for fold in fold_metrics:
        if set(fold.get("models", {})) != set(CANDIDATES):
            raise ValueError(f"buyability fold {fold.get('name')} has an incomplete candidate grid")
        for name, metrics in fold["models"].items():
            value = metrics.get("mean_return")
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"buyability fold {fold['name']} candidate {name} is non-finite")
    for name, metrics in aggregate.items():
        required = [metrics.get(k) for k in (
            "mean_return", "compound_return", "max_drawdown", "mean_filled_names"
        )]
        if not np.isfinite(np.asarray(required, dtype=float)).all():
            raise ValueError(f"candidate {name} has non-finite metrics")
    baseline = aggregate[BASELINE]
    eligible = []
    for name in CANDIDATES[1:]:
        metrics = aggregate[name]
        wins = sum(f["models"][name]["mean_return"] > f["models"][BASELINE]["mean_return"] for f in fold_metrics)
        if (float(metrics["mean_filled_names"]) / TOP_N >= .60 and float(metrics["mean_return"]) > float(baseline["mean_return"])
                and float(metrics["max_drawdown"]) >= float(baseline["max_drawdown"]) - .05 and wins >= 3):
            eligible.append((name, wins))
    if not eligible:
        return {"status": "no_enhancement_passed", "selected": BASELINE, "next_branch": "daily_baseline_retained"}
    selected = max(eligible, key=lambda x: (float(aggregate[x[0]]["mean_return"]), float(aggregate[x[0]]["compound_return"]), x[1], x[0]))[0]
    return {"status": "enhancement_selected", "selected": selected, "next_branch": "forward_shadow"}


def _fold_evidence(folds: list[dict]) -> list[dict]:
    return [
        {
            "name": fold["name"],
            "train_end": str(pd.Timestamp(fold["train_end"]).date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge_dates"]],
            "oos_start": str(pd.Timestamp(fold["oos_start"]).date()),
            "oos_end": str(pd.Timestamp(fold["oos_end"]).date()),
            "oos_days": int(fold["oos_days"]),
        }
        for fold in folds
    ]


def _parent_binding(parent_state: dict) -> dict:
    report = json.loads(Path(parent_state["report"]["path"]).read_text(encoding="utf-8"))
    universe = report.get("eligible_universe_hashes", {})
    if not universe:
        raise RuntimeError("parent state has no frozen universe hashes")
    return {"protocol": parent_state["protocol"], "protocol_hash": parent_state["protocol_hash"],
            "state_hash": parent_state["state_hash"], "status": parent_state["status"],
            "selected": parent_state["selected"], "next_branch": parent_state["next_branch"],
            "input_hashes_hash": _canonical_hash(parent_state["input_hashes"]),
            "eligible_universe_hash": _canonical_hash(universe),
            "final_universe_hash": universe.get("final_all_keys"),
            "fold_evidence_hash": _canonical_hash(_fold_evidence(report.get("folds", []))),
    }


def _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path):
    result = parent_input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    result["controller"] = {"path": str(Path(__file__).resolve()), "sha256": artifact_hash(Path(__file__))}
    result["parent_state"] = {"path": str(parent_state_path.resolve()), "sha256": artifact_hash(parent_state_path)}
    return result


def validate_state(state: dict) -> None:
    content = dict(state)
    state_hash = content.pop("state_hash", None)
    if state_hash != _canonical_hash(content):
        raise RuntimeError("buyability state was modified")
    if state.get("protocol") != PROTOCOL or state.get("protocol_hash") != _canonical_hash(protocol_payload()):
        raise RuntimeError("buyability protocol changed")
    if state.get("eligible_for_production") is not False or state.get("production_publication") is not False:
        raise RuntimeError("buyability production isolation changed")
    if state.get("untouched_holdout") is not False or state.get("human_approval_required") is not True:
        raise RuntimeError("buyability historical evidence status changed")
    for name in ("report", "execution_records", "daily_returns"):
        evidence = state.get(name, {})
        path = Path(evidence.get("path", ""))
        if not path.is_file() or artifact_hash(path) != evidence.get("sha256"):
            raise RuntimeError(f"buyability {name} artifact changed")
    report = json.loads(Path(state["report"]["path"]).read_text(encoding="utf-8"))
    decision = report.get("decision", {})
    if (
        report.get("protocol_hash") != state.get("protocol_hash")
        or report.get("input_hashes") != state.get("input_hashes")
        or report.get("parent_binding") != state.get("parent_binding")
        or decision.get("status") != state.get("status")
        or decision.get("selected") != state.get("selected")
        or decision.get("next_branch") != state.get("next_branch")
    ):
        raise RuntimeError("buyability report and state are inconsistent")


def _run_enhancement_race_unlocked(label_paths, screening_report_path, parent_state_path, output_dir, state_dir,
                                   daily_dir=None, intraday_dir=None, model_threads=8) -> dict:
    daily_dir = Path(daily_dir or default_daily_prepared_dir()); intraday_dir = Path(intraday_dir or config.PREPARED_DIR)
    output_dir = Path(output_dir); state_dir = Path(state_dir)
    _validate_research_paths(output_dir, state_dir, intraday_dir)
    parent_state_path = Path(parent_state_path)
    parent = json.loads(parent_state_path.read_text(encoding="utf-8"))
    if parent.get("protocol") != "intraday_1400_minute_feature_residualization_v1":
        raise RuntimeError("buyability enhancement requires the completed residualization parent")
    from intraday_1400.minute_feature_residualization import _validate_state as validate_residual_state
    validate_residual_state(parent)
    valid_route = (
        parent.get("status") == "no_residual_enhancement_passed"
        and parent.get("next_branch") == "daily_baseline_retained"
    )
    if not valid_route:
        raise RuntimeError("residualization parent did not route to daily baseline retention")
    binding = _parent_binding(parent)
    state_path = state_dir / "manifest.json"
    inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_state(state)
        if state.get("input_hashes") != inputs:
            raise RuntimeError("buyability invocation differs from frozen inputs")
        return state
    labels = combine_historical_labels(label_paths)
    dates = validate_historical_dates(labels, load_trading_calendar(intraday_dir))
    folds = registered_folds(dates)
    actual_fold_evidence = _fold_evidence([
        {
            "name": fold["name"],
            "train_end": fold["train"][-1],
            "purge_dates": fold["purge"],
            "oos_start": fold["oos"][0],
            "oos_end": fold["oos"][-1],
            "oos_days": len(fold["oos"]),
        }
        for fold in folds
    ])
    if _canonical_hash(actual_fold_evidence) != binding["fold_evidence_hash"]:
        raise RuntimeError("buyability folds differ from the residualization parent")
    provenance = _prepared_provenance(intraday_dir)
    window, base_features, minute_features, screening = _causal_screened_features(screening_report_path, daily_dir, intraday_dir, provenance)
    panel, universe = _build_panel(labels, daily_dir, intraday_dir, base_features, minute_features)
    if _canonical_hash(universe) != binding["eligible_universe_hash"]: raise RuntimeError("buyability universe differs from parent")
    panel[BUY_LABEL] = pd.to_numeric(panel[BUY_LABEL], errors="coerce")
    all_records=[]; fold_reports=[]
    for fold in folds:
        fp = panel[~panel["date"].isin(fold["purge"])].copy(); train_end=pd.Timestamp(fold["train"][-1]); oos_start=pd.Timestamp(fold["oos"][0]); oos_end=pd.Timestamp(fold["oos"][-1])
        h1 = _fit_daily_head(fp, base_features, train_end, oos_start, oos_end, model_threads, BASELINE, fold["name"])
        result = model.train_binary_classifier(
            fp,
            [*base_features, *minute_features],
            BUY_LABEL,
            "lightgbm",
            train_end=str(train_end.date()),
            valid_end=str(oos_end.date()),
            predict_start=str(oos_start.date()),
            predict_end=str(oos_end.date()),
            decay_half_life_days=60.0,
            min_weight=.03,
            minority_weight=1.0,
            n_estimators=160,
            learning_rate=.02,
            max_train_rows=400000,
            enforce_max_train_rows=True,
            n_jobs=model_threads,
        )
        if not result.ok: raise RuntimeError(f"{fold['name']} buyability classifier failed: {result.message}")
        scores = build_buyability_scores(h1, result.predictions)
        models={}
        for name in CANDIDATES:
            records, metrics = _evaluate_candidate(name, scores[name], fp, fold["oos"]); records["fold"]=fold["name"]; all_records.append(records); models[name]=metrics
        fold_reports.append({
            "name": fold["name"],
            "train_end": str(train_end.date()),
            "purge_dates": [str(pd.Timestamp(value).date()) for value in fold["purge"]],
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "oos_days": int(len(fold["oos"])),
            "buyability_classifier_metrics": result.metrics,
            "buyability_feature_count": int(len(base_features) + len(minute_features)),
            "models": models,
        })
    records = pd.concat(all_records, ignore_index=True)
    oos_dates = pd.DatetimeIndex(sorted({pd.Timestamp(x) for f in folds for x in f["oos"]}))
    comparison = compare_execution_records(
        cash_normalized_execution_records(records, oos_dates, top_n=TOP_N, models=list(CANDIDATES))
    )
    daily_returns = comparison["daily_returns"]
    if not pd.DatetimeIndex(daily_returns["signal_date"].unique()).sort_values().equals(oos_dates):
        raise RuntimeError("buyability comparison does not cover exact OOS dates")
    for name in CANDIDATES:
        if name not in daily_returns or daily_returns[name].isna().any():
            raise RuntimeError(f"buyability candidate {name} has missing OOS returns")
    decision = select_enhancement(comparison["models"], fold_reports)
    final_inputs = _input_hashes(
        label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path
    )
    if final_inputs != inputs:
        raise RuntimeError("buyability inputs changed during evaluation")
    report={"protocol":PROTOCOL,"protocol_hash":_canonical_hash(protocol_payload()),"parent_binding":binding,"input_hashes":inputs,"prepared_provenance":provenance,"causal_feature_screening":screening,"feature_screening_train_end":str(pd.Timestamp(window["train_end"]).date()),"candidate_grid":list(CANDIDATES),"eligible_universe_hashes":universe,"folds":fold_reports,"account_comparison":{"models":comparison["models"],"pairwise":comparison["pairwise"]},"decision":decision,"production_publication":False,"production_candidate":False,"human_approval_required":True,"untouched_holdout":False}
    output_dir.mkdir(parents=True, exist_ok=True); state_dir.mkdir(parents=True, exist_ok=True)
    records_path=output_dir/"execution_records.parquet"; daily_path=output_dir/"daily_returns.parquet"; report_path=output_dir/"report.json"
    atomic_parquet(records, records_path); atomic_parquet(comparison["daily_returns"], daily_path); atomic_json(report, report_path)
    state={"protocol":PROTOCOL,"protocol_hash":report["protocol_hash"],"status":decision["status"],"selected":decision["selected"],"next_branch":decision["next_branch"],"input_hashes":inputs,"parent_binding":binding,"report":{"path":str(report_path.resolve()),"sha256":artifact_hash(report_path)},"execution_records":{"path":str(records_path.resolve()),"sha256":artifact_hash(records_path)},"daily_returns":{"path":str(daily_path.resolve()),"sha256":artifact_hash(daily_path)},"eligible_for_production":False,"production_publication":False,"human_approval_required":True,"untouched_holdout":False}
    state["state_hash"]=_canonical_hash(state); atomic_json(state,state_path); return state


def run_enhancement_race(label_paths, screening_report_path, parent_state_path, output_dir, state_dir,
                         daily_dir=None, intraday_dir=None, model_threads=8) -> dict:
    state_dir = Path(state_dir)
    with _cycle_lock(state_dir):
        return _run_enhancement_race_unlocked(
            label_paths, screening_report_path, parent_state_path, output_dir, state_dir,
            daily_dir, intraday_dir, model_threads,
        )


def main() -> None:
    p=argparse.ArgumentParser(description="Isolated daily H1 plus 13:55 buyability enhancement")
    p.add_argument("--labels", type=Path, action="append", required=True); p.add_argument("--screening-report", type=Path, required=True); p.add_argument("--parent-state", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--state-dir", type=Path, required=True); p.add_argument("--daily-dir", type=Path); p.add_argument("--intraday-dir", type=Path); p.add_argument("--model-threads", type=int, default=8)
    a=p.parse_args(); print(json.dumps(run_enhancement_race(a.labels,a.screening_report,a.parent_state,a.output_dir,a.state_dir,a.daily_dir,a.intraday_dir,a.model_threads), ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
