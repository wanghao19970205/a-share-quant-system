from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_1400 import config, daily_h1_buyability_enhancement as buyability_module, pipeline
from intraday_1400.adaptive_exit_replay import load_trading_calendar
from intraday_1400.daily_h1_buyability_enhancement import (
    BASELINE as PARENT_BASELINE,
    _build_panel,
    _canonical_hash,
    _causal_screened_features,
    _cycle_lock,
    _evaluate_candidate,
    _fit_daily_head,
    _prepared_provenance,
    _validate_research_paths,
)
from intraday_1400.daily_minute_enhancement import _input_hashes as parent_input_hashes
from intraday_1400.fair_race_pipeline import default_daily_prepared_dir
from intraday_1400.offline_race import compare_execution_records
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet
from intraday_1400.structural_combo_holdout import cash_normalized_execution_records
from intraday_1400.target_redesign_backfill import FOLD_POSITIONS, TOP_N, combine_historical_labels, registered_folds, validate_historical_dates
from quant import model

PROTOCOL = "intraday_1400_daily_h1_payoff_rerank_v2"
PARENT_PROTOCOL = "intraday_1400_daily_h1_executable_payoff_v1"
BASELINE = "baseline"
CANDIDATES = (BASELINE, "h1_top50_payoff_rerank", "payoff_top50_buy_rerank")
BUY_LABEL = "adaptive_entry_buyable"
LIQUIDATED_LABEL = "adaptive_liquidated_by_t3"
RETURN_LABEL = "adaptive_realized_net_ret_t3"
MODEL_RECIPE = {
    "daily_h1_head": "daily_h1_buyability_enhancement._fit_daily_head",
    "features": "verified_1355_asof_plus_minute",
    "buyability_head": {"classifier": "lightgbm", "minority_weight": 1.0, "n_estimators": 160, "learning_rate": 0.02, "max_train_rows": 400000, "predict_scope": "current_fold_oos_only", "oos_calibration": False},
    "liquidation_head": {"target": LIQUIDATED_LABEL, "train_rows": "adaptive_entry_buyable_only", "classifier": "lightgbm", "minority_weight": 1.0, "n_estimators": 160, "learning_rate": 0.02, "max_train_rows": 400000, "predict_scope": "current_fold_oos_only", "oos_calibration": False},
    "return_head": {"target": RETURN_LABEL, "train_rows": "adaptive_entry_buyable_and_liquidated_only", "model": "lightgbm_regression", "n_estimators": 200, "learning_rate": 0.015, "max_train_rows": 400000, "enforce_max_train_rows": True, "early_stopping": False},
    "scoring": {
        "h1_top50_payoff_rerank": "top50_by_h1_then_rank_executable_value",
        "payoff_top50_buy_rerank": "top50_by_executable_value_then_rank_p_buy",
    },
    "cutoff_time": "13:55", "execution": "adaptive_t3_exact_top10_no_refill_fixed_capital", "fill_rate": 0.60,
}


def protocol_payload() -> dict:
    return {"protocol": PROTOCOL, "parent_protocol": PARENT_PROTOCOL, "parent_controller_replayable": False, "evidence_policy": "development_only_requires_independent_reproduction_before_shadow", "candidate_grid": list(CANDIDATES), "baseline": BASELINE, "fold_positions": FOLD_POSITIONS, "model_recipe": MODEL_RECIPE, "selection_gate": {"minimum_fill_rate": .60, "minimum_fold_wins_vs_baseline": 3, "mean_return_must_exceed_baseline": True, "max_drawdown_tolerance": .05}, "production_publication": False, "human_approval_required": True}


def _zscore_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame["date"]).transform("mean")
    std = values.groupby(frame["date"]).transform("std").replace(0.0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_payoff_scores(daily_h1: pd.DataFrame, p_buy: pd.DataFrame, p_liquidated: pd.DataFrame, pred_return: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Score strictly from 13:55 features and OOS predictions, never realized outcomes."""
    key_sets = [
        set(frame[["code", "date"]].itertuples(index=False, name=None))
        for frame in (daily_h1, p_buy, p_liquidated, pred_return)
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        labels = ("daily_h1", "buyability", "liquidation", "return")
        details = {
            name: {"rows": len(frame), "keys": len(keys), "missing_vs_daily": len(key_sets[0] - keys), "extra_vs_daily": len(keys - key_sets[0])}
            for name, frame, keys in zip(labels, (daily_h1, p_buy, p_liquidated, pred_return), key_sets)
        }
        raise ValueError(f"payoff heads do not share exact OOS prediction universe: {details}")
    h1 = daily_h1[["code", "date", "score"]].rename(columns={"score": "h1"})
    out = h1.merge(p_buy[["code", "date", "pred"]].rename(columns={"pred": "p_buy"}), on=["code", "date"], validate="one_to_one")
    out = out.merge(p_liquidated[["code", "date", "pred"]].rename(columns={"pred": "p_liquidated"}), on=["code", "date"], validate="one_to_one")
    out = out.merge(pred_return[["code", "date", "pred"]].rename(columns={"pred": "pred_return"}), on=["code", "date"], validate="one_to_one")
    out["executable_value"] = out["p_buy"] * (out["p_liquidated"] * out["pred_return"].clip(-.10, .10) + (1.0 - out["p_liquidated"]) * -.10)
    h1_pool = out.sort_values(
        ["date", "h1", "code"], ascending=[True, False, True]
    ).groupby("date", group_keys=False).head(50)
    h1_payoff = h1_pool.sort_values(
        ["date", "executable_value", "h1", "code"],
        ascending=[True, False, False, True],
    ).assign(model_variant="h1_top50_payoff_rerank", score=lambda frame: frame["executable_value"])
    payoff_pool = out.sort_values(
        ["date", "executable_value", "code"], ascending=[True, False, True]
    ).groupby("date", group_keys=False).head(50)
    payoff_buy = payoff_pool.sort_values(
        ["date", "p_buy", "executable_value", "code"],
        ascending=[True, False, False, True],
    ).assign(model_variant="payoff_top50_buy_rerank", score=lambda frame: frame["p_buy"])
    result = {
        BASELINE: h1.assign(model_variant=BASELINE, score=h1["h1"]),
        "h1_top50_payoff_rerank": h1_payoff,
        "payoff_top50_buy_rerank": payoff_buy,
    }
    return {
        name: frame[["code", "date", "score", "model_variant"]].reset_index(drop=True)
        for name, frame in result.items()
    }


def select_enhancement(aggregate: dict[str, dict], fold_metrics: list[dict]) -> dict:
    if set(aggregate) != set(CANDIDATES) or [f.get("name") for f in fold_metrics] != [f["name"] for f in FOLD_POSITIONS]:
        raise ValueError("payoff selection requires the exact registered candidate grid and ordered four folds")
    for metrics in aggregate.values():
        if not np.isfinite(np.asarray([metrics.get(k) for k in ("mean_return", "compound_return", "max_drawdown", "mean_filled_names")], dtype=float)).all():
            raise ValueError("payoff candidate has non-finite metrics")
    baseline = aggregate[BASELINE]; eligible = []
    for name in CANDIDATES[1:]:
        metrics = aggregate[name]
        wins = sum(f["models"][name]["mean_return"] > f["models"][BASELINE]["mean_return"] for f in fold_metrics)
        if float(metrics["mean_filled_names"]) / TOP_N >= .60 and float(metrics["mean_return"]) > float(baseline["mean_return"]) and float(metrics["max_drawdown"]) >= float(baseline["max_drawdown"]) - .05 and wins >= 3:
            eligible.append((name, wins))
    if not eligible:
        return {"status": "no_payoff_enhancement_passed", "selected": BASELINE, "next_branch": "daily_baseline_retained"}
    selected = max(eligible, key=lambda x: (aggregate[x[0]]["mean_return"], aggregate[x[0]]["compound_return"], x[1], x[0]))[0]
    return {"status": "development_candidate_selected", "selected": selected, "next_branch": "independent_reproduction"}


def _fold_evidence(folds: list[dict]) -> list[dict]:
    return [{"name": f["name"], "train_end": str(pd.Timestamp(f["train_end"]).date()), "purge_dates": [str(pd.Timestamp(x).date()) for x in f["purge_dates"]], "oos_start": str(pd.Timestamp(f["oos_start"]).date()), "oos_end": str(pd.Timestamp(f["oos_end"]).date()), "oos_days": int(f["oos_days"])} for f in folds]


def _parent_binding(parent: dict) -> dict:
    report = json.loads(Path(parent["report"]["path"]).read_text(encoding="utf-8"))
    universe = report.get("eligible_universe_hashes", {})
    if not universe: raise RuntimeError("parent state has no frozen universe hashes")
    return {"protocol": parent["protocol"], "protocol_hash": parent["protocol_hash"], "state_hash": parent["state_hash"], "status": parent["status"], "selected": parent["selected"], "next_branch": parent["next_branch"], "input_hashes_hash": _canonical_hash(parent["input_hashes"]), "eligible_universe_hash": _canonical_hash(universe), "final_universe_hash": universe.get("final_all_keys"), "fold_evidence_hash": _canonical_hash(_fold_evidence(report.get("folds", [])))}


def _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path):
    result = parent_input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir)
    result["controller"] = {"path": str(Path(__file__).resolve()), "sha256": artifact_hash(Path(__file__))}
    buyability_path = Path(buyability_module.__file__).resolve()
    result["buyability_dependency"] = {"path": str(buyability_path), "sha256": artifact_hash(buyability_path)}
    result["parent_state"] = {"path": str(Path(parent_state_path).resolve()), "sha256": artifact_hash(parent_state_path)}
    return result


def validate_state(state: dict) -> None:
    content = dict(state); state_hash = content.pop("state_hash", None)
    if state_hash != _canonical_hash(content): raise RuntimeError("payoff state was modified")
    if state.get("protocol_hash") != _canonical_hash(protocol_payload()) or state.get("protocol") != PROTOCOL: raise RuntimeError("payoff protocol changed")
    if state.get("eligible_for_production") is not False or state.get("production_publication") is not False: raise RuntimeError("payoff production isolation changed")
    for name in ("report", "execution_records", "daily_returns"):
        item = state.get(name, {}); path = Path(item.get("path", ""))
        if not path.is_file() or artifact_hash(path) != item.get("sha256"): raise RuntimeError(f"payoff {name} artifact changed")
    report = json.loads(Path(state["report"]["path"]).read_text(encoding="utf-8"))
    if report.get("protocol_hash") != state.get("protocol_hash") or report.get("decision") != {k: state[k] for k in ("status", "selected", "next_branch")} or report.get("input_hashes") != state.get("input_hashes") or report.get("parent_binding") != state.get("parent_binding"):
        raise RuntimeError("payoff report and state are inconsistent")
    hashes = state.get("before_after_hashes", {})
    if hashes.get("before") != _canonical_hash(state["input_hashes"]) or hashes.get("after") != hashes.get("before"):
        raise RuntimeError("payoff before/after input hashes are inconsistent")


def _run_unlocked(label_paths, screening_report_path, parent_state_path, output_dir, state_dir, daily_dir=None, intraday_dir=None, model_threads=8) -> dict:
    if int(model_threads) <= 0:
        raise ValueError("payoff model_threads must be positive")
    daily_dir, intraday_dir, output_dir, state_dir = map(Path, (daily_dir or default_daily_prepared_dir(), intraday_dir or config.PREPARED_DIR, output_dir, state_dir))
    _validate_research_paths(output_dir, state_dir, intraday_dir)
    parent_state_path = Path(parent_state_path)
    parent = json.loads(parent_state_path.read_text(encoding="utf-8"))
    parent_content = dict(parent)
    parent_hash = parent_content.pop("state_hash", None)
    if parent_hash != _canonical_hash(parent_content):
        raise RuntimeError("payoff v1 parent state was modified")
    if parent.get("protocol") != PARENT_PROTOCOL:
        raise RuntimeError("payoff rerank requires the completed payoff v1 parent")
    for artifact_name in ("report", "execution_records", "daily_returns"):
        artifact = parent.get(artifact_name, {})
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.is_file() or artifact_hash(artifact_path) != artifact.get("sha256"):
            raise RuntimeError(f"payoff v1 parent {artifact_name} changed")
    parent_report = json.loads(Path(parent["report"]["path"]).read_text(encoding="utf-8"))
    parent_decision = parent_report.get("decision", {})
    if (
        parent_report.get("protocol_hash") != parent.get("protocol_hash")
        or parent_report.get("input_hashes") != parent.get("input_hashes")
        or parent_decision.get("status") != parent.get("status")
        or parent_decision.get("selected") != parent.get("selected")
        or parent_decision.get("next_branch") != parent.get("next_branch")
    ):
        raise RuntimeError("payoff v1 parent report and state are inconsistent")
    if not (
        parent.get("status") == "no_payoff_enhancement_passed"
        and parent.get("selected") == PARENT_BASELINE
        and parent.get("next_branch") == "daily_baseline_retained"
    ):
        raise RuntimeError("payoff v1 parent did not route to daily baseline retention")
    binding = _parent_binding(parent); state_path = state_dir / "manifest.json"; inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8")); validate_state(state)
        if state.get("input_hashes") != inputs: raise RuntimeError("payoff invocation differs from frozen inputs")
        return state
    labels = combine_historical_labels(label_paths); dates = validate_historical_dates(labels, load_trading_calendar(intraday_dir)); folds = registered_folds(dates)
    evidence = _fold_evidence([{**f, "train_end": f["train"][-1], "purge_dates": f["purge"], "oos_start": f["oos"][0], "oos_end": f["oos"][-1], "oos_days": len(f["oos"])} for f in folds])
    if _canonical_hash(evidence) != binding["fold_evidence_hash"]: raise RuntimeError("payoff folds differ from parent")
    provenance = _prepared_provenance(intraday_dir); window, base_features, minute_features, screening = _causal_screened_features(screening_report_path, daily_dir, intraday_dir, provenance); panel, universe = _build_panel(labels, daily_dir, intraday_dir, base_features, minute_features)
    if _canonical_hash(universe) != binding["eligible_universe_hash"]: raise RuntimeError("payoff universe differs from parent")
    panel[BUY_LABEL] = pd.to_numeric(panel[BUY_LABEL], errors="coerce"); panel[LIQUIDATED_LABEL] = pd.to_numeric(panel[LIQUIDATED_LABEL], errors="coerce"); panel[RETURN_LABEL] = pd.to_numeric(panel[RETURN_LABEL], errors="coerce")
    all_records=[]; fold_reports=[]
    features = [*base_features, *minute_features]
    for fold in folds:
        fp = panel[~panel["date"].isin(fold["purge"])].copy(); train_end, oos_start, oos_end = pd.Timestamp(fold["train"][-1]), pd.Timestamp(fold["oos"][0]), pd.Timestamp(fold["oos"][-1])
        entered = fp[BUY_LABEL].fillna(0).astype(float) > .5
        liquidated = fp[LIQUIDATED_LABEL].fillna(0).astype(float) > .5
        # Make conditional populations explicit. The model splitter still applies these masks to train rows only.
        fp[LIQUIDATED_LABEL] = fp[LIQUIDATED_LABEL].where(entered, np.nan)
        fp["_payoff_return_train_row"] = entered & liquidated
        h1 = _fit_daily_head(fp, base_features, train_end, oos_start, oos_end, model_threads, BASELINE, fold["name"])
        common = dict(panel=fp, features=features, classifier="lightgbm", train_end=str(train_end.date()), valid_end=str(oos_end.date()), predict_start=str(oos_start.date()), predict_end=str(oos_end.date()), decay_half_life_days=60.0, min_weight=.03, minority_weight=1.0, n_estimators=160, learning_rate=.02, max_train_rows=400000, enforce_max_train_rows=True, n_jobs=model_threads)
        buy = model.train_binary_classifier(label_col=BUY_LABEL, **common)
        liq = model.train_binary_classifier(label_col=LIQUIDATED_LABEL, **common)
        ret = model.train_lightgbm(
            fp,
            features,
            train_end=str(train_end.date()),
            valid_end=str(oos_end.date()),
            predict_start=str(oos_start.date()),
            predict_end=str(oos_end.date()),
            decay_half_life_days=60.0,
            min_weight=.03,
            n_estimators=200,
            learning_rate=.015,
            early_stopping_rounds=0,
            n_jobs=model_threads,
            label_col=RETURN_LABEL,
            train_mask_col="_payoff_return_train_row",
            max_train_rows=400000,
            enforce_max_train_rows=True,
        )
        for name, result in (("buyability", buy), ("liquidation", liq), ("return", ret)):
            if not result.ok: raise RuntimeError(f"{fold['name']} {name} head failed: {result.message}")
        # Conditional targets are NaN outside their training populations; explicitly assert this invariant.
        train_rows = fp[fp["date"] <= train_end]
        entered_train_count = int((train_rows[BUY_LABEL].fillna(0).astype(bool)).sum())
        liquidated_train_count = int((train_rows[BUY_LABEL].fillna(0).astype(bool) & train_rows[LIQUIDATED_LABEL].fillna(0).astype(bool)).sum())
        if train_rows.loc[train_rows[BUY_LABEL].fillna(0).astype(bool), LIQUIDATED_LABEL].notna().sum() == 0: raise RuntimeError(f"{fold['name']} has no entered liquidation labels")
        if train_rows.loc[(train_rows[BUY_LABEL].fillna(0).astype(bool)) & (train_rows[LIQUIDATED_LABEL].fillna(0).astype(bool)), RETURN_LABEL].notna().sum() == 0: raise RuntimeError(f"{fold['name']} has no realized return labels")
        scores = build_payoff_scores(h1, buy.predictions, liq.predictions, ret.predictions); models={}
        for name in CANDIDATES:
            records, metrics = _evaluate_candidate(name, scores[name], fp, fold["oos"]); records["fold"] = fold["name"]; all_records.append(records); models[name] = metrics
        fold_reports.append({"name": fold["name"], "train_end": str(train_end.date()), "purge_dates": [str(pd.Timestamp(x).date()) for x in fold["purge"]], "oos_start": str(oos_start.date()), "oos_end": str(oos_end.date()), "oos_days": len(fold["oos"]), "buyability_classifier_metrics": buy.metrics, "liquidation_classifier_metrics": liq.metrics, "return_regression_metrics": ret.metrics, "entered_train_rows": entered_train_count, "liquidated_train_rows": liquidated_train_count, "models": models})
    records = pd.concat(all_records, ignore_index=True); oos_dates = pd.DatetimeIndex(sorted({pd.Timestamp(x) for f in folds for x in f["oos"]})); comparison = compare_execution_records(cash_normalized_execution_records(records, oos_dates, top_n=TOP_N, models=list(CANDIDATES))); decision = select_enhancement(comparison["models"], fold_reports)
    final_inputs = _input_hashes(label_paths, screening_report_path, daily_dir, intraday_dir, parent_state_path)
    if final_inputs != inputs: raise RuntimeError("payoff inputs changed during evaluation")
    report = {"protocol": PROTOCOL, "protocol_hash": _canonical_hash(protocol_payload()), "parent_binding": binding, "input_hashes": inputs, "before_after_hashes": {"before": _canonical_hash(inputs), "after": _canonical_hash(final_inputs)}, "prepared_provenance": provenance, "causal_feature_screening": screening, "feature_screening_train_end": str(pd.Timestamp(window["train_end"]).date()), "eligible_universe_hashes": universe, "folds": fold_reports, "account_comparison": {"models": comparison["models"], "pairwise": comparison["pairwise"]}, "decision": decision, "parent_controller_replayable": False, "evidence_status": "development_only", "production_publication": False, "production_candidate": False, "human_approval_required": True, "untouched_holdout": False}
    output_dir.mkdir(parents=True, exist_ok=True); state_dir.mkdir(parents=True, exist_ok=True); records_path=output_dir/"execution_records.parquet"; daily_path=output_dir/"daily_returns.parquet"; report_path=output_dir/"report.json"; atomic_parquet(records, records_path); atomic_parquet(comparison["daily_returns"], daily_path); atomic_json(report, report_path)
    state = {"protocol": PROTOCOL, "protocol_hash": report["protocol_hash"], "status": decision["status"], "selected": decision["selected"], "next_branch": decision["next_branch"], "input_hashes": inputs, "before_after_hashes": report["before_after_hashes"], "parent_binding": binding, "report": {"path": str(report_path.resolve()), "sha256": artifact_hash(report_path)}, "execution_records": {"path": str(records_path.resolve()), "sha256": artifact_hash(records_path)}, "daily_returns": {"path": str(daily_path.resolve()), "sha256": artifact_hash(daily_path)}, "eligible_for_production": False, "production_publication": False, "human_approval_required": True, "untouched_holdout": False}; state["state_hash"] = _canonical_hash(state); atomic_json(state, state_path); return state


def run_enhancement_race(label_paths, screening_report_path, parent_state_path, output_dir, state_dir, daily_dir=None, intraday_dir=None, model_threads=8) -> dict:
    state_dir = Path(state_dir)
    with _cycle_lock(state_dir):
        return _run_unlocked(label_paths, screening_report_path, parent_state_path, output_dir, state_dir, daily_dir, intraday_dir, model_threads)


def run_research(*args, **kwargs) -> dict:
    return run_enhancement_race(*args, **kwargs)


def main() -> None:
    p=argparse.ArgumentParser(description="Isolated daily H1 executable-payoff research"); p.add_argument("--labels", type=Path, action="append", required=True); p.add_argument("--screening-report", type=Path, required=True); p.add_argument("--parent-state", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--state-dir", type=Path, required=True); p.add_argument("--daily-dir", type=Path); p.add_argument("--intraday-dir", type=Path); p.add_argument("--model-threads", type=int, default=8); a=p.parse_args(); print(json.dumps(run_research(a.labels, a.screening_report, a.parent_state, a.output_dir, a.state_dir, a.daily_dir, a.intraday_dir, a.model_threads), ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
