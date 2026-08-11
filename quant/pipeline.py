"""量化选股 · 单数据仓训练评估流水线。

用法示例：
    QUANT_DATA_DIR=quant_data/hs300 python -m quant.pipeline --horizon 5
    QUANT_DATA_DIR=quant_data/full_a_sample python -m quant.pipeline --horizon 5 --top-n 20
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from quant import backtest, model, warehouse
from quant.factors import engineering
import quant.select as selector


def _expand_selected_factors(selected: list[str], all_features: list[str]) -> list[str]:
    """Keep selected raw factors and their cross-sectional bucket dummies together."""
    all_set = set(all_features)
    expanded: list[str] = []
    for f in selected:
        if f in all_set and f not in expanded:
            expanded.append(f)
        prefix = f"{f}_q_"
        for qf in all_features:
            if qf.startswith(prefix) and qf not in expanded:
                expanded.append(qf)
    for f in all_features:
        if f.startswith("cat_") and f not in expanded:
            expanded.append(f)
    return expanded


def _validate_explicit_holdout_boundaries(
    train_end: str | None,
    valid_end: str | None,
    predict_start: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    if not all((train_end, valid_end, predict_start)):
        raise ValueError(
            "pipeline requires explicit train_end, valid_end and predict_start "
            "to keep early-stopping validation separate from final evaluation"
        )
    train_cut = pd.Timestamp(train_end)
    valid_cut = pd.Timestamp(valid_end)
    prediction_cut = pd.Timestamp(predict_start)
    if not train_cut < valid_cut < prediction_cut:
        raise ValueError(
            "pipeline boundaries must satisfy train_end < valid_end < predict_start"
        )
    return train_cut, valid_cut, prediction_cut


def _selection_manifest(
    selection_name: str,
    label_col: str,
    candidates: list[str],
    selected: list[str],
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    predict_start: pd.Timestamp,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "selection_name": selection_name,
        "label_col": label_col,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
        "predict_start": str(predict_start.date()),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "candidate_pool_sha256": hashlib.sha256(
            "\n".join(sorted(map(str, candidates))).encode("utf-8")
        ).hexdigest(),
        "selected_sha256": hashlib.sha256(
            "\n".join(map(str, selected)).encode("utf-8")
        ).hexdigest(),
        "generator_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }])


def _continuous_selected(selected: list[str], all_features: list[str], max_base_features: int) -> list[str]:
    feature_summary = engineering.summarize_features(all_features)
    continuous = set(feature_summary[feature_summary["kind"].str.endswith("continuous")]["feature"].astype(str))
    out: list[str] = []
    for f in selected:
        if f in continuous and f in all_features and f not in out:
            out.append(f)
        if len(out) >= max_base_features:
            break
    return out


def _add_feature_crosses(panel: pd.DataFrame, selected: list[str], all_features: list[str],
                         max_base_features: int = 10, max_cross_features: int = 30) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Create low-order crosses from selected continuous factors only."""
    if max_base_features <= 1 or max_cross_features <= 0:
        return panel, all_features, pd.DataFrame()
    bases = _continuous_selected(selected, all_features, max_base_features)
    if len(bases) <= 1:
        return panel, all_features, pd.DataFrame()
    out = panel.copy()
    cross_features: list[str] = []
    rows = []
    for i, left in enumerate(bases):
        for right in bases[i + 1:]:
            if len(cross_features) >= max_cross_features:
                break
            name = f"cross__{left}__x__{right}"
            out[name] = pd.to_numeric(out[left], errors="coerce") * pd.to_numeric(out[right], errors="coerce")
            out[name] = out[name].replace([float("inf"), float("-inf")], pd.NA)
            cross_features.append(name)
            rows.append({"feature": name, "left": left, "right": right})
        if len(cross_features) >= max_cross_features:
            break
    return out, all_features + cross_features, pd.DataFrame(rows)


def run(horizon: int = 5, panel_name: str = "factor_panel", selection_name: str = "factor_selection",
        top_factors: int = 50, models: list[str] | None = None, top_n: int = 3,
        train_months: int = 24, validation_months: int = 1, test_months: int = 1,
        train_end: str | None = None, valid_end: str | None = None,
        predict_start: str | None = None,
        decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
        ridge_alpha: float = 10.0, limit: int = 0, min_price_rows: int = 0,
        add_discrete: bool = True, add_crosses: bool = False,
        cross_base_features: int = 10, max_cross_features: int = 30,
        n_estimators: int = 200, learning_rate: float | None = None,
        early_stopping_rounds: int = 40, output_prefix: str = "",
        max_weight: float | None = None, positive_only: bool = False,
        pred_quantile: float | None = None, volatility_quantile: float | None = None,
        turnover_quantile: float | None = None, rule_weight: float = 0.0,
        min_rule_score: float | None = None, ridge_quantile: float | None = None,
        lgbm_weight: float = 1.0) -> pd.DataFrame:
    train_cut, valid_cut, prediction_cut = _validate_explicit_holdout_boundaries(
        train_end, valid_end, predict_start
    )
    raw = engineering.build_panel(horizon=horizon, limit=limit, min_price_rows=min_price_rows)
    if raw.empty:
        raise RuntimeError("数据仓里没有可用价格数据，无法生成因子面板")
    panel, feats = engineering.prepare_features(raw, horizon=horizon, add_discrete=add_discrete)

    label_col = f"target_ret_{horizon}d"
    picked = selector.select_factors(
        panel, horizon=horizon, top_n=top_factors, label_col=label_col
    )
    selected = picked["factor"].tolist() if not picked.empty else feats
    cross_features: list[str] = []
    if add_crosses:
        panel, feats, cross_summary = _add_feature_crosses(panel, selected, feats,
                                                           max_base_features=cross_base_features,
                                                           max_cross_features=max_cross_features)
        cross_features = cross_summary["feature"].astype(str).tolist() if not cross_summary.empty else []
        cross_output = cross_summary
    else:
        cross_output = pd.DataFrame()

    name_prefix = f"{output_prefix}_" if output_prefix else ""
    warehouse.save(f"{name_prefix}feature_cross_summary", cross_output)

    warehouse.save(panel_name, panel)
    feature_summary = engineering.summarize_features(feats)
    warehouse.save(f"{name_prefix}feature_summary", feature_summary)
    warehouse.save(selection_name, picked)
    selection_manifest = _selection_manifest(
        selection_name,
        label_col,
        feats,
        selected,
        train_cut,
        valid_cut,
        prediction_cut,
    )
    warehouse.save(f"{selection_name}_manifest", selection_manifest)
    factors = _expand_selected_factors(selected, feats) + [f for f in cross_features if f not in selected]
    model_feature_summary = engineering.summarize_features(factors)
    warehouse.save(f"{name_prefix}model_feature_summary", model_feature_summary)
    warehouse.save(f"{name_prefix}model_feature_summary_counts", model_feature_summary.groupby("kind").size().rename("n_features").reset_index())

    if decay_half_life_days:
        w = model.time_decay_weights(panel, decay_half_life_days, min_weight=min_weight)
        weight_summary = pd.DataFrame([{
            "decay_half_life_days": decay_half_life_days,
            "min_weight": min_weight,
            "n_samples": len(w),
            "weight_min": float(w.min()) if len(w) else None,
            "weight_p25": float(pd.Series(w).quantile(0.25)) if len(w) else None,
            "weight_median": float(pd.Series(w).median()) if len(w) else None,
            "weight_p75": float(pd.Series(w).quantile(0.75)) if len(w) else None,
            "weight_max": float(w.max()) if len(w) else None,
        }])
        warehouse.save(f"{name_prefix}time_decay_weight_summary", weight_summary)

    models = models or ["ridge", "ic_weighted", "lightgbm_ranker", "ridge_lightgbm_ranker_ensemble"]
    run_meta = {
        "n_model_features": len(factors),
        "n_cross_features": len(cross_features),
        "add_crosses": bool(add_crosses),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "early_stopping_rounds": early_stopping_rounds,
        "max_weight": max_weight,
        "positive_only": positive_only,
        "pred_quantile": pred_quantile,
        "volatility_quantile": volatility_quantile,
        "turnover_quantile": turnover_quantile,
        "rule_weight": rule_weight,
        "min_rule_score": min_rule_score,
        "ridge_quantile": ridge_quantile,
        "lgbm_weight": lgbm_weight,
        "train_end": str(train_cut.date()),
        "valid_end": str(valid_cut.date()),
        "predict_start": str(prediction_cut.date()),
        "holdout_boundary_policy": "explicit_nonoverlapping",
    }
    rows = []
    holdout_models = [m for m in models if m != "ridge_lightgbm_ranker_ensemble"]
    for res in model.train_all(panel, horizon=horizon, models=holdout_models, factors=factors,
                               train_end=str(train_cut.date()), valid_end=str(valid_cut.date()),
                               predict_start=str(prediction_cut.date()),
                               decay_half_life_days=decay_half_life_days, min_weight=min_weight,
                               ridge_alpha=ridge_alpha, n_estimators=n_estimators,
                               learning_rate=learning_rate, early_stopping_rounds=early_stopping_rounds):
        rows.append({"stage": "holdout", "model": res.model, "ok": res.ok, "message": res.message, **run_meta, **res.metrics})
        if res.ok and not res.predictions.empty:
            warehouse.save(f"{name_prefix}pred_{res.model}", res.predictions)

    # Walk-forward uses the same strict train/validation/test split for every requested model.
    for model_name in models:
        bt = backtest.walk_forward(panel, factors, model_name=model_name, horizon=horizon, top_n=top_n,
                                   train_months=train_months, validation_months=validation_months,
                                   test_months=test_months,
                                   max_weight=max_weight,
                                   decay_half_life_days=decay_half_life_days, min_weight=min_weight,
                                   ridge_alpha=ridge_alpha,
                                   positive_only=positive_only, pred_quantile=pred_quantile,
                                   volatility_quantile=volatility_quantile, turnover_quantile=turnover_quantile,
                                   rule_weight=rule_weight, min_rule_score=min_rule_score,
                                   ridge_quantile=ridge_quantile, lgbm_weight=lgbm_weight,
                                   n_estimators=n_estimators,
                                   learning_rate=learning_rate, early_stopping_rounds=early_stopping_rounds)
        warehouse.save(f"{name_prefix}bt_{model_name}_returns", bt["returns"])
        warehouse.save(f"{name_prefix}bt_{model_name}_holdings", bt["holdings"])
        warehouse.save(f"{name_prefix}bt_{model_name}_predictions", bt["predictions"])
        rows.append({"stage": "walk_forward", "model": model_name, "ok": bool(bt["summary"]), "message": "ok", **run_meta, **bt["summary"]})

    summary = pd.DataFrame(rows)
    warehouse.save(f"{name_prefix}pipeline_summary", summary)
    counts = feature_summary.groupby("kind").size().rename("n_features").reset_index()
    warehouse.save(f"{name_prefix}feature_summary_counts", counts)
    return summary


def main():
    ap = argparse.ArgumentParser(description="运行量化选股训练评估流水线")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-factors", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="仅使用前 N 只股票，抽样或降级验证用")
    ap.add_argument("--min-price-rows", type=int, default=0, help="仅使用日线行数不少于该值的股票，长周期训练可过滤新股/短历史")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--validation-months", type=int, default=1, help="训练窗口末尾用于早停/调参的验证月份，不参与最终测试回测")
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--train-end", required=True, help="显式训练截止日期，不能省略")
    ap.add_argument("--valid-end", required=True, help="显式验证截止日期，不能省略")
    ap.add_argument("--predict-start", required=True, help="显式最终评估起始日期，必须晚于 valid-end")
    ap.add_argument("--decay-half-life-days", type=float, default=90.0, help="时间衰减半衰期；<=0 表示关闭")
    ap.add_argument("--min-weight", type=float, default=0.05, help="旧样本最小权重")
    ap.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge L2 正则强度")
    ap.add_argument("--models", default="ridge,ic_weighted,lightgbm_ranker,ridge_lightgbm_ranker_ensemble")
    ap.add_argument("--no-discrete", action="store_true", help="关闭连续因子分位 one-hot，适合全 A 长周期降内存训练")
    ap.add_argument("--add-crosses", action="store_true", help="对入选连续因子生成低阶交叉项")
    ap.add_argument("--cross-base-features", type=int, default=10, help="参与交叉的入选连续因子数上限")
    ap.add_argument("--max-cross-features", type=int, default=30, help="生成交叉特征数上限")
    ap.add_argument("--n-estimators", type=int, default=200, help="树模型最大训练轮次，配合验证集早停")
    ap.add_argument("--learning-rate", type=float, default=None, help="树模型学习率；默认按模型内部设置")
    ap.add_argument("--early-stopping-rounds", type=int, default=40, help="验证集早停轮数；<=0 关闭")
    ap.add_argument("--output-prefix", default="", help="实验输出前缀，避免覆盖默认 bt/pipeline_summary 产物")
    ap.add_argument("--max-weight", type=float, default=None, help="单票最大权重；默认 1/top_n")
    ap.add_argument("--positive-only", action="store_true", help="只买预测收益为正的股票")
    ap.add_argument("--pred-quantile", type=float, default=None, help="只在当日预测分数分位数以上选股")
    ap.add_argument("--volatility-quantile", type=float, default=None, help="过滤掉当日波动率分位数以上股票")
    ap.add_argument("--turnover-quantile", type=float, default=None, help="过滤掉当日换手率分位数以上股票")
    ap.add_argument("--rule-weight", type=float, default=0.0, help="规则分在融合排序中的权重；0 表示只用模型预测")
    ap.add_argument("--min-rule-score", type=float, default=None, help="只保留规则分不低于该阈值的候选")
    ap.add_argument("--ridge-quantile", type=float, default=None, help="融合模型中只保留 Ridge 分数分位数以上候选，如 0.5")
    ap.add_argument("--lgbm-weight", type=float, default=1.0, help="Ridge+LightGBM 融合排序中 LightGBM 权重")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    decay = args.decay_half_life_days if args.decay_half_life_days > 0 else None
    early_stopping_rounds = args.early_stopping_rounds if args.early_stopping_rounds > 0 else 0
    summary = run(horizon=args.horizon, top_factors=args.top_factors, models=models, top_n=args.top_n,
                  train_months=args.train_months, validation_months=args.validation_months,
                  test_months=args.test_months, train_end=args.train_end,
                  valid_end=args.valid_end, predict_start=args.predict_start,
                  decay_half_life_days=decay, min_weight=args.min_weight,
                  ridge_alpha=args.ridge_alpha, limit=args.limit, min_price_rows=args.min_price_rows,
                  add_discrete=not args.no_discrete, add_crosses=args.add_crosses,
                  cross_base_features=args.cross_base_features, max_cross_features=args.max_cross_features,
                  n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                  early_stopping_rounds=early_stopping_rounds,
                  output_prefix=args.output_prefix,
                  max_weight=args.max_weight, positive_only=args.positive_only,
                  pred_quantile=args.pred_quantile, volatility_quantile=args.volatility_quantile,
                  turnover_quantile=args.turnover_quantile,
                  rule_weight=args.rule_weight, min_rule_score=args.min_rule_score,
                  ridge_quantile=args.ridge_quantile, lgbm_weight=args.lgbm_weight)
    print(summary.to_string(index=False))
    counts_name = f"{args.output_prefix}_feature_summary_counts" if args.output_prefix else "feature_summary_counts"
    counts = warehouse.load(counts_name)
    if not counts.empty:
        print("\n特征类型汇总：")
        print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
