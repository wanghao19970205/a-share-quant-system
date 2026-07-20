"""白名单舆情特征分组消融与后段确认。

分组只在较早的 selection 区间选择，入选组合才进入后段 confirmation。
全部结果写入全新目录，不修改现有模型、预测或发布产物。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant import model
from quant import sentiment_feature_experiment as base


def sentiment_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "lexicon_direction": [
            c for c in columns if "_local_decay_" in c or c.endswith("positive_ratio_7d")
            or c.endswith("negative_ratio_7d")
        ],
        "llm_structured": [c for c in columns if "_structured_decay_" in c],
        "hybrid": [c for c in columns if "_hybrid_decay_" in c],
        "activity_category": [
            c for c in columns if "article_count_" in c or c.endswith("announcement_count_7d")
            or c.endswith("research_count_7d") or c.endswith("stock_news_count_7d")
        ],
        "llm_quality": [
            c for c in columns if c.endswith("llm_coverage_7d") or c.endswith("relevance_7d")
            or c.endswith("certainty_7d") or c.endswith("novelty_7d")
        ],
    }
    return {name: values for name, values in groups.items() if values}


def _train_variant(panel: pd.DataFrame, output_dir: Path, label: str, model_name: str,
                   features: list[str], horizon: int, train_end: pd.Timestamp,
                   valid_end: pd.Timestamp, predict_start: pd.Timestamp, top_n: int) -> tuple[dict, pd.DataFrame]:
    results = model.train_all(
        panel, horizon=horizon, models=[model_name], factors=features,
        train_end=str(train_end.date()), valid_end=str(valid_end.date()),
        predict_start=str(predict_start.date()), decay_half_life_days=90.0,
        min_weight=0.05, n_estimators=300, early_stopping_rounds=40,
    )
    result = results[0]
    if not result.ok:
        return {"ok": False, "message": result.message}, pd.DataFrame()
    predictions = result.predictions
    predictions.to_parquet(output_dir / f"{model_name}_{label}_predictions.parquet", index=False)
    portfolio, returns = base._portfolio_metrics(predictions, horizon, top_n)
    returns.to_parquet(output_dir / f"{model_name}_{label}_returns.parquet", index=False)
    return {
        "ok": True,
        "training_metrics": result.metrics,
        "test_metrics": base._prediction_metrics(predictions, horizon),
        "portfolio": portfolio,
    }, returns


def _comparison(candidate: dict, baseline: dict, candidate_returns: pd.DataFrame,
                baseline_returns: pd.DataFrame) -> dict:
    if not candidate.get("ok") or not baseline.get("ok"):
        return {
            "rank_ic_gain": None, "sharpe_gain": None, "monthly_win_rate": None,
            "drawdown_change": None,
        }
    rank_gain = ((candidate.get("test_metrics", {}).get("rank_ic") or 0.0)
                 - (baseline.get("test_metrics", {}).get("rank_ic") or 0.0))
    sharpe_gain = ((candidate.get("portfolio", {}).get("sharpe") or 0.0)
                   - (baseline.get("portfolio", {}).get("sharpe") or 0.0))
    candidate_dd = candidate.get("portfolio", {}).get("max_drawdown")
    baseline_dd = baseline.get("portfolio", {}).get("max_drawdown")
    drawdown_change = (float(candidate_dd - baseline_dd)
                       if candidate_dd is not None and baseline_dd is not None else None)
    return {
        "rank_ic_gain": rank_gain,
        "sharpe_gain": sharpe_gain,
        "monthly_win_rate": base._monthly_win_rate(candidate_returns, baseline_returns),
        "drawdown_change": drawdown_change,
    }


def _selection_pass(metrics: dict) -> bool:
    return bool(
        (metrics.get("rank_ic_gain") or 0.0) > 0
        and (metrics.get("sharpe_gain") or 0.0) >= 0
        and (metrics.get("monthly_win_rate") or 0.0) >= 0.50
        and metrics.get("drawdown_change") is not None
        and metrics["drawdown_change"] >= -0.03
    )


def _confirmation_pass(metrics: dict) -> bool:
    return bool(
        (metrics.get("rank_ic_gain") or 0.0) > 0
        and (metrics.get("sharpe_gain") or 0.0) >= 0.10
        and (metrics.get("monthly_win_rate") or 0.0) >= 0.55
        and metrics.get("drawdown_change") is not None
        and metrics["drawdown_change"] >= -0.02
    )


def _selection_score(metrics: dict) -> float:
    if not _selection_pass(metrics):
        return float("-inf")
    return float(metrics["rank_ic_gain"] + 0.01 * metrics["sharpe_gain"])


def run_ablation(panel: pd.DataFrame, output_dir: Path, horizon: int,
                 selection_train_end: pd.Timestamp, selection_valid_end: pd.Timestamp,
                 selection_predict_start: pd.Timestamp, selection_end: pd.Timestamp,
                 confirmation_train_end: pd.Timestamp, confirmation_valid_end: pd.Timestamp,
                 confirmation_predict_start: pd.Timestamp, top_n: int = 3,
                 models: tuple[str, ...] = ("ridge", "lightgbm_ranker"), max_groups: int = 3) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"消融输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    panel = base._standardize_sentiment(panel)
    feature_columns = [c for c in panel.columns if c.startswith(base._SENTIMENT_PREFIX)]
    groups = sentiment_feature_groups(feature_columns)
    base_features = base._base_features(panel, horizon)
    if not groups:
        raise RuntimeError("没有可用的舆情特征组")

    report: dict = {
        "horizon": horizon,
        "feature_groups": groups,
        "selection_window": {
            "train_end": str(selection_train_end.date()),
            "valid_end": str(selection_valid_end.date()),
            "predict_start": str(selection_predict_start.date()),
            "end": str(selection_end.date()),
        },
        "confirmation_window": {
            "train_end": str(confirmation_train_end.date()),
            "valid_end": str(confirmation_valid_end.date()),
            "predict_start": str(confirmation_predict_start.date()),
        },
        "models": {},
    }
    selection_panel = panel[panel["date"] <= selection_end].copy()
    selection_panel = base._purge_boundary_labels(
        selection_panel, horizon, [selection_train_end, selection_valid_end])
    confirmation_panel = base._purge_boundary_labels(
        panel.copy(), horizon, [confirmation_train_end, confirmation_valid_end])

    for model_name in models:
        model_dir = output_dir / model_name
        model_dir.mkdir()
        baseline, baseline_returns = _train_variant(
            selection_panel, model_dir, "selection_baseline", model_name, base_features,
            horizon, selection_train_end, selection_valid_end, selection_predict_start, top_n)
        selected: list[str] = []
        selected_features: list[str] = []
        steps: list[dict] = []
        remaining = list(groups)
        for step_number in range(1, max(int(max_groups), 0) + 1):
            candidates: dict[str, dict] = {}
            candidate_returns: dict[str, pd.DataFrame] = {}
            for group_name in remaining:
                features = base_features + selected_features + groups[group_name]
                label = f"selection_step{step_number}_{group_name}"
                result, returns = _train_variant(
                    selection_panel, model_dir, label, model_name, features, horizon,
                    selection_train_end, selection_valid_end, selection_predict_start, top_n)
                metrics = _comparison(result, baseline, returns, baseline_returns)
                candidates[group_name] = {"result": result, "comparison": metrics}
                candidate_returns[group_name] = returns
            ranked = sorted(candidates, key=lambda name: _selection_score(candidates[name]["comparison"]),
                            reverse=True)
            winner = ranked[0] if ranked and _selection_pass(candidates[ranked[0]]["comparison"]) else None
            steps.append({"step": step_number, "candidates": candidates, "winner": winner})
            if winner is None:
                break
            selected.append(winner)
            selected_features.extend(groups[winner])
            baseline = candidates[winner]["result"]
            baseline_returns = candidate_returns[winner]
            remaining.remove(winner)

        confirmation_baseline, confirmation_baseline_returns = _train_variant(
            confirmation_panel, model_dir, "confirmation_baseline", model_name, base_features,
            horizon, confirmation_train_end, confirmation_valid_end,
            confirmation_predict_start, top_n)
        if selected_features:
            confirmation_augmented, confirmation_augmented_returns = _train_variant(
                confirmation_panel, model_dir, "confirmation_selected", model_name,
                base_features + selected_features, horizon, confirmation_train_end,
                confirmation_valid_end, confirmation_predict_start, top_n)
            confirmation_comparison = _comparison(
                confirmation_augmented, confirmation_baseline,
                confirmation_augmented_returns, confirmation_baseline_returns)
        else:
            confirmation_augmented = {"ok": False, "message": "选择期没有入选特征组"}
            confirmation_comparison = _comparison(
                confirmation_augmented, confirmation_baseline,
                pd.DataFrame(), confirmation_baseline_returns)
        confirmed = bool(selected and _confirmation_pass(confirmation_comparison))
        report["models"][model_name] = {
            "selection_steps": steps,
            "selected_groups": selected,
            "selected_features": selected_features,
            "confirmation_baseline": confirmation_baseline,
            "confirmation_selected": confirmation_augmented,
            "confirmation_comparison": confirmation_comparison,
            "confirmed": confirmed,
        }

    comparable = list(report["models"].values())
    report["promote_to_full_a_backfill"] = bool(
        comparable and all(item.get("confirmed", False) for item in comparable))
    (output_dir / "ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行隔离的白名单舆情特征分组消融")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--news-dir", required=True)
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--selection-train-end", required=True)
    parser.add_argument("--selection-valid-end", required=True)
    parser.add_argument("--selection-predict-start", required=True)
    parser.add_argument("--selection-end", required=True)
    parser.add_argument("--confirmation-train-end", required=True)
    parser.add_argument("--confirmation-valid-end", required=True)
    parser.add_argument("--confirmation-predict-start", required=True)
    parser.add_argument("--models", default="ridge,lightgbm_ranker")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--cutoff-hour", type=int, default=15)
    args = parser.parse_args()

    codes = base.read_watchlist(Path(args.watchlist))
    panel = base.load_monthly_panel(
        Path(args.prepared_dir), codes, pd.Timestamp(args.start), pd.Timestamp(args.end))
    panel = base.build_feature_panel(
        panel, Path(args.news_dir), codes, cutoff_hour=args.cutoff_hour)
    report = run_ablation(
        panel, Path(args.output_dir).resolve(), args.horizon,
        pd.Timestamp(args.selection_train_end), pd.Timestamp(args.selection_valid_end),
        pd.Timestamp(args.selection_predict_start), pd.Timestamp(args.selection_end),
        pd.Timestamp(args.confirmation_train_end), pd.Timestamp(args.confirmation_valid_end),
        pd.Timestamp(args.confirmation_predict_start), top_n=args.top_n,
        models=tuple(item.strip() for item in args.models.split(",") if item.strip()),
        max_groups=args.max_groups,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
