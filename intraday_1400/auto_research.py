from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from intraday_1400 import config
from intraday_1400.storage import artifact_hash, atomic_json


PROTOCOL = "intraday_1400_auto_research_v2"
BASELINE_MODEL = "e4_daily_top10"
TOP_N = 10

BRANCH_RECIPES = {
    "execution_filter": {
        "hypothesis": "Minute data adds value mainly by rejecting low-probability entries.",
        "parent_variants": ["h2_daily_top100_buy_filter"],
        "registered_grid": {
            "daily_candidate_n": [50, 100, 200],
            "minimum_buy_probability": [0.45, 0.50, 0.55],
        },
    },
    "candidate_rerank": {
        "hypothesis": "Minute data adds value by reranking a causal T-1 daily candidate set.",
        "parent_variants": ["h1_daily_top100_structural_rerank", "e0_e4_staged_combo"],
        "registered_grid": {
            "daily_candidate_n": [50, 100, 200],
            "minute_score": ["e1_e3_structural", "e0_direct", "e0_e3_fixed_50_50"],
        },
    },
    "independent_structural": {
        "hypothesis": "Independent minute structural ranking is stronger than daily-conditioned ranking.",
        "parent_variants": ["e1_e3_structural", "e0_e3_minute_block"],
        "registered_grid": {
            "score": ["e1_e3_structural", "e0_e3_fixed_50_50"],
            "top_n": [5, 10, 15],
        },
    },
    "direct_return": {
        "hypothesis": "A direct adaptive stress-return target is the strongest minute signal.",
        "parent_variants": ["e0_direct"],
        "registered_grid": {
            "target": ["adaptive_stress_net_ret_t3", "adaptive_realized_net_ret_t3"],
            "top_n": [5, 10, 15],
        },
    },
    "target_redesign": {
        "hypothesis": "Existing return heads do not generalize and require a new development interval.",
        "parent_variants": [BASELINE_MODEL],
        "registered_grid": {
            "target_family": ["downside_quantile", "cross_sectional_rank", "conditional_payoff"],
        },
    },
}

DAILY_HISTORY_DEPENDENT_VARIANTS = {
    "e4_daily_top10",
    "e0_e4_staged_combo",
    "h1_daily_top100_structural_rerank",
    "h2_daily_top100_buy_filter",
}

VARIANT_BRANCH = {
    "h2_daily_top100_buy_filter": "execution_filter",
    "h1_daily_top100_structural_rerank": "candidate_rerank",
    "e0_e4_staged_combo": "candidate_rerank",
    "e1_e3_structural": "independent_structural",
    "e0_e3_minute_block": "independent_structural",
    "e0_direct": "direct_return",
    BASELINE_MODEL: "target_redesign",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return artifact_hash(path)


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _protocol_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "baseline_model": BASELINE_MODEL,
        "top_n": TOP_N,
        "branch_recipes": BRANCH_RECIPES,
        "variant_branch": VARIANT_BRANCH,
        "daily_history_dependent_variants": sorted(DAILY_HISTORY_DEPENDENT_VARIANTS),
        "production_publication": False,
        "holdout_claim_required": True,
    }


@contextmanager
def _cycle_lock(state_dir: Path) -> Iterator[None]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".cycle.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another automatic research cycle is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _state_lock(state_dir: Path) -> Iterator[None]:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_or_initialize_unlocked(state_dir: Path) -> dict:
    manifest_path = Path(state_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        validate_research_state(manifest)
        return manifest
    payload = _protocol_payload()
    manifest = {
        "protocol": PROTOCOL,
        "protocol_hash": _canonical_hash(payload),
        "production_isolated": True,
        "human_approval_required": True,
        "holdout_claims": [],
        "consumed_holdouts": [],
        "experiments": [],
        "next_experiment": None,
    }
    atomic_json(manifest, manifest_path)
    return manifest


def initialize_research_state(state_dir: Path) -> dict:
    with _state_lock(state_dir):
        return _load_or_initialize_unlocked(state_dir)


def validate_research_state(manifest: dict) -> None:
    expected = _canonical_hash(_protocol_payload())
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_hash") != expected:
        raise RuntimeError("automatic research protocol changed; create a new state directory")
    if manifest.get("production_isolated") is not True:
        raise RuntimeError("automatic research must remain isolated from production")
    if manifest.get("human_approval_required") is not True:
        raise RuntimeError("human production approval cannot be disabled")


def _metric_number(metrics: dict, name: str, default: float = 0.0) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else float(default)


def evaluate_holdout_report(report: dict) -> dict:
    if report.get("untouched_holdout") is not True:
        raise ValueError("only an untouched holdout report can drive automatic branching")
    if int(report.get("holdout_days", 0)) < 60:
        raise ValueError("automatic branching requires at least 60 holdout days")
    comparison = report.get("account_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("holdout report missing fixed-capital account comparison")
    models = comparison.get("models", {})
    if BASELINE_MODEL not in models:
        raise ValueError(f"holdout report missing baseline {BASELINE_MODEL}")
    eligible = {
        name: metrics
        for name, metrics in models.items()
        if int(metrics.get("days", 0)) == 60
    }
    if set(eligible) != set(models) or not eligible:
        raise ValueError("every holdout model must have exactly 60 fixed-capital days")
    diagnostic_overall_winner = max(
        eligible,
        key=lambda name: (_metric_number(eligible[name], "mean_return", -1e9), name),
    )
    daily_history_causal = report.get("daily_history_causal") is True
    branch_models = (
        eligible
        if daily_history_causal
        else {
            name: metrics
            for name, metrics in eligible.items()
            if name not in DAILY_HISTORY_DEPENDENT_VARIANTS
        }
    )
    if not branch_models:
        raise ValueError("no causally eligible model can select the next branch")
    winner = max(
        branch_models,
        key=lambda name: (_metric_number(branch_models[name], "mean_return", -1e9), name),
    )
    baseline = eligible[BASELINE_MODEL]
    winner_metrics = eligible[winner]
    baseline_mean = _metric_number(baseline, "mean_return")
    winner_mean = _metric_number(winner_metrics, "mean_return")
    blocks = report.get("twenty_day_blocks", [])
    if len(blocks) != 3 or [block.get("block") for block in blocks] != [1, 2, 3]:
        raise ValueError("holdout report requires exactly three ordered 20-day blocks")
    previous_end = None
    block_deltas = []
    for block in blocks:
        start = pd.Timestamp(block.get("start"))
        end = pd.Timestamp(block.get("end"))
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("invalid 20-day block interval")
        if previous_end is not None and start <= previous_end:
            raise ValueError("20-day blocks must be non-overlapping and ordered")
        previous_end = end
        block_models = block.get("account_comparison", {}).get("models", {})
        if set(block_models) != set(models):
            raise ValueError("every 20-day block must contain every holdout model")
        if any(int(metrics.get("days", 0)) != 20 for metrics in block_models.values()):
            raise ValueError("every block model must have exactly 20 fixed-capital days")
        block_deltas.append(
            _metric_number(block_models[winner], "mean_return")
            - _metric_number(block_models[BASELINE_MODEL], "mean_return")
        )
    if (
        str(pd.Timestamp(blocks[0]["start"]).date()) != str(report.get("holdout_start"))
        or str(pd.Timestamp(blocks[-1]["end"]).date()) != str(report.get("holdout_end"))
    ):
        raise ValueError("20-day blocks must cover the declared holdout boundaries")
    positive_blocks = sum(delta > 0 for delta in block_deltas)
    mean_names = _metric_number(winner_metrics, "mean_names")
    mean_filled = _metric_number(winner_metrics, "mean_filled_names")
    fill_rate = mean_filled / mean_names if mean_names > 0 else 0.0
    relative_improvement = winner_mean - baseline_mean
    relative_winner = (
        daily_history_causal
        and winner != BASELINE_MODEL
        and relative_improvement > 0
    )
    research_winner = relative_winner and positive_blocks >= 2
    causal_provenance = winner not in DAILY_HISTORY_DEPENDENT_VARIANTS or daily_history_causal
    forward_shadow = (
        research_winner
        and causal_provenance
        and winner_mean > 0
        and _metric_number(winner_metrics, "compound_return") > 0
        and _metric_number(winner_metrics, "max_drawdown", -1.0) >= -0.20
        and fill_rate >= 0.60
    )
    branch = (
        VARIANT_BRANCH.get(winner, "target_redesign")
        if (not daily_history_causal or relative_winner)
        else "target_redesign"
    )
    return {
        "winner": winner,
        "diagnostic_overall_winner": diagnostic_overall_winner,
        "baseline": BASELINE_MODEL,
        "winner_mean_return": winner_mean,
        "baseline_mean_return": baseline_mean,
        "relative_improvement": relative_improvement,
        "positive_twenty_day_blocks": int(positive_blocks),
        "evaluated_twenty_day_blocks": int(len(block_deltas)),
        "fill_rate": fill_rate,
        "daily_history_causal": daily_history_causal,
        "causal_provenance": causal_provenance,
        "research_winner": research_winner,
        "forward_shadow_eligible": forward_shadow,
        "production_candidate": False,
        "next_branch": branch,
        "reason": (
            "freeze winner for append-only forward shadow evaluation"
            if forward_shadow
            else "continue on a new development interval with the registered branch"
        ),
    }


def _intervals_overlap(left: dict, right: dict) -> bool:
    return not (left["end"] < right["start"] or right["end"] < left["start"])


def claim_holdout(
    state_dir: Path,
    claim_id: str,
    start: str,
    end: str,
    labels_path: Path,
    input_hashes: dict | None = None,
) -> dict:
    state_dir = Path(state_dir)
    labels_path = Path(labels_path).resolve()
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    interval = {"start": str(start), "end": str(end)}
    if interval["end"] < interval["start"]:
        raise ValueError("holdout end must not precede start")
    labels_hash = _file_hash(labels_path)
    if input_hashes is None:
        input_hashes = {
            "holdout_labels": {"path": str(labels_path), "sha256": labels_hash}
        }
    if input_hashes.get("holdout_labels", {}).get("sha256") != labels_hash:
        raise RuntimeError("claimed holdout label hash does not match input hashes")
    claim = {
        **interval,
        "claim_id": str(claim_id),
        "labels_path": str(labels_path),
        "labels_hash": labels_hash,
        "input_hashes": input_hashes,
        "status": "claimed",
    }
    with _state_lock(state_dir):
        manifest = _load_or_initialize_unlocked(state_dir)
        for existing in manifest["holdout_claims"]:
            if existing["claim_id"] == claim["claim_id"]:
                identity_keys = (
                    "start", "end", "claim_id", "labels_path", "labels_hash", "input_hashes",
                )
                if any(existing.get(key) != claim.get(key) for key in identity_keys):
                    raise RuntimeError(f"holdout claim {claim_id} is immutable")
                if existing["status"] == "abandoned":
                    existing["status"] = "claimed"
                    existing.pop("failure", None)
                    atomic_json(manifest, state_dir / "manifest.json")
                return existing
            if existing["status"] != "abandoned" and _intervals_overlap(interval, existing):
                raise RuntimeError(
                    f"holdout interval {start}..{end} overlaps claim "
                    f"{existing['start']}..{existing['end']}"
                )
        manifest["holdout_claims"].append(claim)
        atomic_json(manifest, state_dir / "manifest.json")
    return claim


def run_frozen_holdout_cycle(
    training_labels_path: Path,
    holdout_labels_path: Path,
    screening_report_path: Path,
    output_dir: Path,
    state_dir: Path,
    model_threads: int = 8,
) -> dict:
    from intraday_1400.fair_race_pipeline import default_daily_prepared_dir
    from intraday_1400.structural_combo_holdout import (
        holdout_input_hashes,
        run_frozen_holdout,
        validated_holdout_dates,
    )
    from quant import config as quant_config

    def consume_generated_report(report_path: Path, generated_report: dict) -> dict:
        persisted_report = _read_json(report_path)
        if persisted_report != generated_report:
            raise RuntimeError("persisted holdout report differs from generated result")
        report_hash = _file_hash(report_path)
        interval = {
            "start": str(persisted_report["holdout_start"]),
            "end": str(persisted_report["holdout_end"]),
        }
        with _state_lock(state_dir):
            manifest = _load_or_initialize_unlocked(state_dir)
            matching_claims = [
                item
                for item in manifest["holdout_claims"]
                if (
                    item["status"] == "claimed"
                    and item["start"] == interval["start"]
                    and item["end"] == interval["end"]
                )
            ]
            if len(matching_claims) != 1:
                raise RuntimeError("holdout interval must be claimed exactly once before evaluation")
            claimed = matching_claims[0]
            if persisted_report.get("input_hashes") != claimed["input_hashes"]:
                raise RuntimeError("holdout report inputs do not match the claimed artifacts")
            decision = evaluate_holdout_report(persisted_report)
            experiment_id = (
                f"{persisted_report.get('experiment', 'holdout')}-{report_hash[:12]}"
            )
            consumed = {
                **interval,
                "claim_id": claimed["claim_id"],
                "experiment_id": experiment_id,
                "labels_hash": claimed["labels_hash"],
                "input_hashes": claimed["input_hashes"],
                "report_hash": report_hash,
                "consumed_for": "branch_selection_only",
            }
            next_experiment = {
                "parent_id": experiment_id,
                "branch": decision["next_branch"],
                "status": (
                    "awaiting_forward_dates"
                    if decision["forward_shadow_eligible"]
                    else "awaiting_new_development_split"
                ),
                "recipe": BRANCH_RECIPES[decision["next_branch"]],
                "holdout_reuse_forbidden": True,
                "production_publication": False,
            }
            next_experiment["config_hash"] = _canonical_hash(next_experiment)
            claimed["status"] = "consumed"
            claimed["experiment_id"] = experiment_id
            claimed["report_hash"] = report_hash
            manifest["consumed_holdouts"].append(consumed)
            manifest["experiments"].append({
                "experiment_id": experiment_id,
                "status": "evaluated",
                "report_path": str(report_path),
                "report_hash": report_hash,
                "holdout": interval,
                "decision": decision,
            })
            manifest["next_experiment"] = next_experiment
            atomic_json(decision, state_dir / "decisions" / f"{experiment_id}.json")
            atomic_json(manifest, state_dir / "manifest.json")
        return decision

    with _cycle_lock(state_dir):
        training_labels_path = Path(training_labels_path).resolve()
        holdout_labels_path = Path(holdout_labels_path).resolve()
        screening_report_path = Path(screening_report_path).resolve()
        output_dir = Path(output_dir).resolve()
        daily_dir = Path(default_daily_prepared_dir())
        intraday_dir = Path(config.PREPARED_DIR)
        active_predictions_path = (
            Path(quant_config.QUANT_DIR) / "active_quant_short_predictions.parquet"
        )
        training_dates = pd.read_parquet(training_labels_path, columns=["date"])
        labels = pd.read_parquet(holdout_labels_path, columns=["date"])
        dates = validated_holdout_dates(training_dates, labels)
        start = str(pd.Timestamp(dates[0]).date())
        end = str(pd.Timestamp(dates[-1]).date())
        input_hashes = holdout_input_hashes(
            training_labels_path,
            holdout_labels_path,
            screening_report_path,
            daily_dir,
            intraday_dir,
            active_predictions_path,
        )
        labels_hash = input_hashes["holdout_labels"]["sha256"]
        input_fingerprint = _canonical_hash(input_hashes)
        claim_id = f"structural-holdout-60d-{input_fingerprint[:12]}"
        with _state_lock(state_dir):
            manifest = _load_or_initialize_unlocked(state_dir)
            recovered = False
            for existing in manifest["holdout_claims"]:
                if (
                    existing["status"] == "claimed"
                    and existing["claim_id"] != claim_id
                    and _intervals_overlap({"start": start, "end": end}, existing)
                ):
                    existing["status"] = "abandoned"
                    existing["failure"] = "stale_claim_recovered"
                    recovered = True
            if recovered:
                atomic_json(manifest, Path(state_dir) / "manifest.json")
        claim = claim_holdout(
            state_dir,
            claim_id,
            start,
            end,
            holdout_labels_path,
            input_hashes=input_hashes,
        )
        report_path = output_dir / "holdout_report.json"
        if claim["status"] == "consumed":
            manifest = initialize_research_state(state_dir)
            experiment = next(
                item
                for item in manifest["experiments"]
                if item["experiment_id"] == claim["experiment_id"]
            )
            if not report_path.exists() or _file_hash(report_path) != experiment["report_hash"]:
                raise RuntimeError("consumed holdout report is missing or changed")
            return {"report": _read_json(report_path), "decision": experiment["decision"]}
        try:
            report = run_frozen_holdout(
                training_labels_path,
                holdout_labels_path,
                screening_report_path,
                output_dir,
                daily_dir=daily_dir,
                intraday_dir=intraday_dir,
                active_predictions_path=active_predictions_path,
                model_threads=model_threads,
                expected_input_hashes=input_hashes,
            )
            decision = consume_generated_report(report_path, report)
            return {"report": report, "decision": decision}
        except Exception as error:
            with _state_lock(state_dir):
                manifest = _load_or_initialize_unlocked(state_dir)
                stored_claim = next(
                    item
                    for item in manifest["holdout_claims"]
                    if item["claim_id"] == claim["claim_id"]
                )
                if stored_claim["status"] == "claimed":
                    stored_claim["status"] = "abandoned"
                    stored_claim["failure"] = type(error).__name__
                    atomic_json(manifest, Path(state_dir) / "manifest.json")
            raise


def research_status(state_dir: Path) -> dict:
    manifest = initialize_research_state(state_dir)
    return {
        "protocol": manifest["protocol"],
        "protocol_hash": manifest["protocol_hash"],
        "production_isolated": manifest["production_isolated"],
        "human_approval_required": manifest["human_approval_required"],
        "experiments": len(manifest["experiments"]),
        "holdout_claims": manifest["holdout_claims"],
        "consumed_holdouts": manifest["consumed_holdouts"],
        "next_experiment": manifest["next_experiment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe automatic intraday research controller")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=config.DATA_ROOT / "auto_research_state_v2",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    cycle = subparsers.add_parser("run-holdout-cycle")
    cycle.add_argument("--training-labels", type=Path, required=True)
    cycle.add_argument("--holdout-labels", type=Path, required=True)
    cycle.add_argument("--screening-report", type=Path, required=True)
    cycle.add_argument("--output-dir", type=Path, required=True)
    cycle.add_argument("--model-threads", type=int, default=8)
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "init":
        result = initialize_research_state(args.state_dir)
    elif args.command == "run-holdout-cycle":
        result = run_frozen_holdout_cycle(
            args.training_labels,
            args.holdout_labels,
            args.screening_report,
            args.output_dir,
            args.state_dir,
            model_threads=args.model_threads,
        )
    else:
        result = research_status(args.state_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
