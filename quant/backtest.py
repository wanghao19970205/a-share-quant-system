"""量化选股 · walk-forward 回测与组合构建。"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from quant import warehouse
from quant.factors import engineering
from quant import model as qmodel


# ------------------------- 回测成交口径（可用环境变量切换） -------------------------
# 默认口径：按当日收盘价成交、计入 20bp 双边成本、含可交易性摩擦（涨停封板当日买不进、
# 跌停封板顺延到下一可卖日收盘卖出）。这是「尾盘 T 买、T+1 卖」的严格研究口径。
# 如需复现旧的乐观口径（不理会涨跌停或忽略成本），必须显式传入参数或环境变量。
# 其它开关：
#   QUANT_BT_FILL=next_open          # 次日开盘成交（此时用 open_ret + buyable_next）
#   QUANT_BT_COST_ROUNDTRIP=0.003    # 双边综合成本(佣金+印花税+滑点)
#   QUANT_BT_SELL_ROLL_MAX_DAYS=3    # 跌停顺延卖出的上限交易日数
def bt_use_open_fill() -> bool:
    return os.environ.get("QUANT_BT_FILL", "close").strip().lower() == "next_open"


def bt_filter_untradable() -> bool:
    return os.environ.get("QUANT_BT_FILTER_UNTRADABLE", "1").strip().lower() in ("1", "true", "yes", "on")


def bt_sell_roll_max_days() -> int:
    try:
        return max(int(os.environ.get("QUANT_BT_SELL_ROLL_MAX_DAYS", "3") or 3), 1)
    except Exception:  # noqa: BLE001
        return 3


def bt_cost_roundtrip() -> float:
    try:
        return float(os.environ.get("QUANT_BT_COST_ROUNDTRIP", "0.002") or 0.002)
    except Exception:  # noqa: BLE001
        return 0.002


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def evaluate_returns(ret: pd.Series, periods_per_year: int = 252) -> dict:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty:
        return {}
    ret = ret.clip(lower=-0.99)
    nav = (1 + ret).cumprod()
    annual_return = nav.iloc[-1] ** (periods_per_year / max(len(ret), 1)) - 1 if nav.iloc[-1] > 0 else np.nan
    annual_vol = ret.std() * np.sqrt(periods_per_year)
    # Sharpe uses arithmetic mean excess return, not compounded annual return.
    sharpe = ret.mean() / ret.std() * np.sqrt(periods_per_year) if ret.std() > 0 else None
    return {
        "periods": int(len(ret)),
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": float(annual_return) if np.isfinite(annual_return) else None,
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe) if sharpe is not None and np.isfinite(sharpe) else None,
        "max_drawdown": _max_drawdown(nav),
        "win_rate": float((ret > 0).mean()),
    }


def _daily_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if not std or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _weight_turnover(
    previous: np.ndarray,
    current: np.ndarray,
) -> float | np.ndarray:
    """Return one-way traded NAV from stock weights plus residual cash."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape or previous.ndim not in (1, 2):
        raise ValueError("weight arrays must have the same one- or two-dimensional shape")
    if (
        not np.isfinite(previous).all()
        or not np.isfinite(current).all()
        or (previous < 0).any()
        or (current < 0).any()
    ):
        raise ValueError("portfolio weights must be finite and non-negative")
    axis = previous.ndim - 1
    previous_exposure = previous.sum(axis=axis)
    current_exposure = current.sum(axis=axis)
    if (previous_exposure > 1.0 + 1e-12).any() or (
        current_exposure > 1.0 + 1e-12
    ).any():
        raise ValueError("portfolio stock weights must not exceed total NAV")
    previous_cash = np.maximum(1.0 - previous_exposure, 0.0)
    current_cash = np.maximum(1.0 - current_exposure, 0.0)
    stock_turnover = np.abs(current - previous).sum(axis=axis)
    turnover = 0.5 * (stock_turnover + np.abs(current_cash - previous_cash))
    if previous.ndim == 1:
        return float(turnover)
    return turnover


def _mapping_turnover(
    previous: dict[str, float],
    current: dict[str, float],
) -> float:
    codes = sorted(set(previous) | set(current))
    previous_weights = np.asarray([previous.get(code, 0.0) for code in codes])
    current_weights = np.asarray([current.get(code, 0.0) for code in codes])
    return float(_weight_turnover(previous_weights, current_weights))


def portfolio_from_predictions(pred: pd.DataFrame, horizon: int = 5, top_n: int = 20,
                               max_weight: float = 0.1, positive_only: bool = False,
                               pred_quantile: float | None = None,
                               volatility_quantile: float | None = None,
                               turnover_quantile: float | None = None,
                               rule_weight: float = 0.0, min_rule_score: float | None = None,
                               ridge_quantile: float | None = None,
                               use_open_fill: bool | None = None, filter_untradable: bool | None = None,
                               cost_roundtrip: float | None = None,
                               require_tradability: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由每日预测构建组合并计算收益。

    成交口径由参数或环境变量决定（默认：当日收盘成交、20bp 双边成本、过滤涨停/停牌）：
    - use_open_fill=True 且存在 ``open_ret_{h}d`` 列时，用「T+1 开盘买入、T+1+h 开盘卖出」的收益；
      收盘口径下若 filter_untradable=True 且存在 ``tradable_ret_{h}d`` 列，用跌停顺延后的实现收益，
      否则用 ``target_ret_{h}d``（信号日收盘→h日后收盘，乐观口径）。
    - filter_untradable=True 时剔除买不进的候选：next_open 口径看 ``buyable_next``（次日一字涨停），
      收盘口径看 ``buyable_close``（当日涨停封板/一字涨停）；缺列直接报错。
    - require_tradability=True 时，严格要求买入资格列存在；研究诊断不得把缺列结果标成严格口径。
    - cost_roundtrip：单次调仓双边综合成本，按当期换手比例计提；默认 20bp。
    """
    if use_open_fill is None:
        use_open_fill = bt_use_open_fill()
    if filter_untradable is None:
        filter_untradable = bt_filter_untradable()
    if cost_roundtrip is None:
        cost_roundtrip = bt_cost_roundtrip()
    target = f"target_ret_{horizon}d"
    open_col = f"open_ret_{horizon}d"
    tradable_col = f"tradable_ret_{horizon}d"
    if use_open_fill and open_col in pred.columns:
        ret_col = open_col
    elif filter_untradable and tradable_col in pred.columns:
        ret_col = tradable_col
    else:
        ret_col = target
    buy_col = "buyable_next" if use_open_fill else "buyable_close"
    if require_tradability and filter_untradable and buy_col not in pred.columns:
        raise ValueError(
            f"严格可交易回测缺少 {buy_col}；请先从价格面板补齐可交易列，"
            "不能静默降级到乐观口径"
        )
    holdings = []
    returns = []
    last_weights: dict[str, float] = {}
    if pred.empty or "pred" not in pred.columns or ret_col not in pred.columns:
        return pd.DataFrame(), pd.DataFrame()
    for date, g in pred.dropna(subset=["pred", ret_col]).groupby("date"):
        pool = g.copy()
        if filter_untradable and buy_col in pool.columns:
            pool = pool[pool[buy_col].fillna(False).astype(bool)]
        if positive_only:
            pool = pool[pool["pred"] > 0]
        if pred_quantile is not None and len(pool) >= 5:
            pool = pool[pool["pred"] >= pool["pred"].quantile(pred_quantile)]
        if volatility_quantile is not None and "volatility_10" in pool.columns and pool["volatility_10"].notna().sum() >= 5:
            pool = pool[pool["volatility_10"] <= pool["volatility_10"].quantile(volatility_quantile)]
        if turnover_quantile is not None and "turnover" in pool.columns and pool["turnover"].notna().sum() >= 5:
            pool = pool[pool["turnover"] <= pool["turnover"].quantile(turnover_quantile)]
        if min_rule_score is not None and "rule_score" in pool.columns:
            pool = pool[pd.to_numeric(pool["rule_score"], errors="coerce") >= min_rule_score]
        if ridge_quantile is not None and "ridge_pred" in pool.columns and pool["ridge_pred"].notna().sum() >= 5:
            pool = pool[pd.to_numeric(pool["ridge_pred"], errors="coerce") >= pd.to_numeric(pool["ridge_pred"], errors="coerce").quantile(ridge_quantile)]
        rank_col = "pred"
        if rule_weight and "rule_score" in pool.columns and pool["rule_score"].notna().sum() >= 5:
            pool = pool.copy()
            pool["ensemble_pred"] = _daily_zscore(pool["pred"]) + float(rule_weight) * _daily_zscore(pool["rule_score"])
            rank_col = "ensemble_pred"
        pick = pool.sort_values(rank_col, ascending=False).head(top_n).copy()
        if pick.empty:
            continue
        w = min(1.0 / max(len(pick), 1), max_weight)
        if w * len(pick) < 1:
            cash = 1 - w * len(pick)
        else:
            cash = 0.0
        pick["weight"] = w
        if rank_col != "pred":
            pick["pred"] = pick[rank_col]
        current_weights = dict(zip(
            pick["code"].astype(str),
            pick["weight"].astype(float),
        ))
        turnover = _mapping_turnover(last_weights, current_weights)
        last_weights = current_weights
        gross_ret = float((pick[ret_col] * pick["weight"]).sum())
        cost = float(turnover) * float(cost_roundtrip)
        period_ret = gross_ret - cost
        keep_cols = ["date", "code", "pred", ret_col, "weight"]
        if target in pick.columns and target not in keep_cols:
            keep_cols.append(target)
        holdings.append(pick[keep_cols])
        returns.append({"date": date, "ret": period_ret, "gross_ret": gross_ret, "cost": cost,
                        "cash": cash, "turnover": turnover, "n_holdings": len(pick)})
    h = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    r = pd.DataFrame(returns)
    if not r.empty:
        r["nav"] = (1 + r["ret"]).cumprod()
    return r, h


def walk_forward(panel: pd.DataFrame, factors: list[str], model_name: str = "ridge", horizon: int = 5,
                 train_months: int = 24, validation_months: int = 1, test_months: int = 1, top_n: int = 3,
                 max_weight: float | None = None,
                 decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                 ridge_alpha: float = 10.0,
                 positive_only: bool = False, pred_quantile: float | None = None,
                 volatility_quantile: float | None = None,
                 turnover_quantile: float | None = None,
                 rule_weight: float = 0.0, min_rule_score: float | None = None,
                 ridge_quantile: float | None = None, lgbm_weight: float = 1.0,
                 n_estimators: int = 200, learning_rate: float | None = None,
                 early_stopping_rounds: int = 40,
                 use_open_fill: bool | None = None, filter_untradable: bool | None = None,
                 cost_roundtrip: float | None = None) -> dict:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates = pd.Series(sorted(df["date"].dropna().unique()))
    if len(dates) < 30:
        return {"summary": {}, "returns": pd.DataFrame(), "holdings": pd.DataFrame(), "predictions": pd.DataFrame()}

    preds = []
    train_delta = pd.DateOffset(months=train_months)
    valid_delta = pd.DateOffset(months=validation_months)
    test_delta = pd.DateOffset(months=test_months)
    current = dates.min() + train_delta + valid_delta
    end = dates.max()
    if max_weight is None:
        max_weight = 1.0 / max(int(top_n), 1)
    registry = {
        "ridge": qmodel.train_ridge,
        "ic_weighted": qmodel.train_ic_weighted,
        "lightgbm": qmodel.train_lightgbm,
        "xgboost": qmodel.train_xgboost,
        "lightgbm_ranker": qmodel.train_lightgbm_ranker,
        "xgboost_ranker": qmodel.train_xgboost_ranker,
        "lstm": qmodel.train_lstm,
    }
    ensemble_model = model_name == "ridge_lightgbm_ranker_ensemble"
    fn = None if ensemble_model else registry.get(model_name)
    if not fn and not ensemble_model:
        raise ValueError(f"未知模型 {model_name}")

    while current < end:
        train_start = current - valid_delta - train_delta
        valid_start = current - valid_delta
        test_end = current + test_delta
        window = df[(df["date"] >= train_start) & (df["date"] < test_end)].copy()
        train_end = (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        valid_end = (current - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        kwargs = {
            "horizon": horizon,
            "train_end": train_end,
            "decay_half_life_days": decay_half_life_days,
            "min_weight": min_weight,
        }
        if ensemble_model:
            ridge_res = qmodel.train_ridge(window, factors, horizon=horizon, train_end=train_end,
                                           decay_half_life_days=decay_half_life_days, min_weight=min_weight,
                                           alpha=ridge_alpha)
            lgbm_res = qmodel.train_lightgbm_ranker(window, factors, horizon=horizon, train_end=train_end,
                                                    valid_end=valid_end, predict_start=current.strftime("%Y-%m-%d"),
                                                    decay_half_life_days=decay_half_life_days, min_weight=min_weight,
                                                    n_estimators=n_estimators, learning_rate=learning_rate,
                                                    early_stopping_rounds=early_stopping_rounds)
            if ridge_res.ok and lgbm_res.ok and not ridge_res.predictions.empty and not lgbm_res.predictions.empty:
                rp = ridge_res.predictions[(ridge_res.predictions["date"] >= current) & (ridge_res.predictions["date"] < test_end)].copy()
                lp = lgbm_res.predictions[(lgbm_res.predictions["date"] >= current) & (lgbm_res.predictions["date"] < test_end)].copy()
                p = lp.rename(columns={"pred": "lgbm_pred"}).merge(
                    rp[["code", "date", "pred"]].rename(columns={"pred": "ridge_pred"}),
                    on=["code", "date"], how="inner",
                )
                if not p.empty:
                    p["lgbm_z"] = p.groupby("date")["lgbm_pred"].transform(_daily_zscore)
                    p["ridge_z"] = p.groupby("date")["ridge_pred"].transform(_daily_zscore)
                    p["pred"] = float(lgbm_weight) * p["lgbm_z"] + (1 - float(lgbm_weight)) * p["ridge_z"]
                    p["model"] = model_name
                    preds.append(p[["code", "date", f"target_ret_{horizon}d", "pred", "model", "ridge_pred", "lgbm_pred"]])
        else:
            if model_name == "ridge":
                kwargs["alpha"] = ridge_alpha
            elif model_name in {"lightgbm", "xgboost", "lightgbm_ranker", "xgboost_ranker"}:
                kwargs["valid_end"] = valid_end
                kwargs["predict_start"] = current.strftime("%Y-%m-%d")
                kwargs["n_estimators"] = n_estimators
                kwargs["learning_rate"] = learning_rate
                kwargs["early_stopping_rounds"] = early_stopping_rounds
            res = fn(window, factors, **kwargs)
            if res.ok and not res.predictions.empty:
                p = res.predictions[(res.predictions["date"] >= current) & (res.predictions["date"] < test_end)].copy()
                p["model"] = model_name
                preds.append(p)
        current = test_end

    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    side_cols = [c for c in ("volatility_10", "turnover", "rule_score",
                             f"open_ret_{horizon}d", f"tradable_ret_{horizon}d",
                             "buyable_next", "buyable_close") if c in df.columns]
    if not pred.empty and side_cols:
        pred = pred.merge(df[["code", "date"] + side_cols].drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
    returns, holdings = portfolio_from_predictions(pred, horizon=horizon, top_n=top_n, max_weight=max_weight,
                                                   positive_only=positive_only, pred_quantile=pred_quantile,
                                                   volatility_quantile=volatility_quantile,
                                                   turnover_quantile=turnover_quantile,
                                                   rule_weight=rule_weight, min_rule_score=min_rule_score,
                                                   ridge_quantile=ridge_quantile,
                                                   use_open_fill=use_open_fill, filter_untradable=filter_untradable,
                                                   cost_roundtrip=cost_roundtrip)
    summary = evaluate_returns(
        returns["ret"] if not returns.empty else pd.Series(dtype=float),
        periods_per_year=max(1, 252 // horizon),
    )
    if not returns.empty:
        summary["avg_turnover"] = float(returns["turnover"].mean())
        summary["avg_holdings"] = float(returns["n_holdings"].mean())
        if "gross_ret" in returns.columns:
            summary["avg_cost"] = float(returns["cost"].mean())
            summary["gross_total_return"] = float((1 + returns["gross_ret"]).prod() - 1)
        _use_open = use_open_fill if use_open_fill is not None else bt_use_open_fill()
        summary["fill"] = "next_open" if (_use_open and f"open_ret_{horizon}d" in pred.columns) else "close"
    if ensemble_model:
        summary["ridge_quantile"] = ridge_quantile
        summary["lgbm_weight"] = lgbm_weight
    return {"summary": summary, "returns": returns, "holdings": holdings, "predictions": pred}


def main():
    ap = argparse.ArgumentParser(description="walk-forward 回测")
    ap.add_argument("--panel", default="factor_panel")
    ap.add_argument("--selection", default="factor_selection")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--validation-months", type=int, default=1, help="训练窗口末尾用于早停/调参的验证月份，不参与最终测试回测")
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--max-weight", type=float, default=None, help="单票最大权重；默认 1/top_n，即 topN 满仓等权")
    ap.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge L2 正则强度")
    ap.add_argument("--decay-half-life-days", type=float, default=90.0, help="时间衰减半衰期；<=0 表示关闭")
    ap.add_argument("--min-weight", type=float, default=0.05, help="旧样本最小权重")
    ap.add_argument("--positive-only", action="store_true", help="只买预测收益为正的股票")
    ap.add_argument("--pred-quantile", type=float, default=None, help="只在当日预测分数分位数以上选股，如 0.7")
    ap.add_argument("--volatility-quantile", type=float, default=None, help="过滤掉当日波动率分位数以上股票，如 0.8")
    ap.add_argument("--turnover-quantile", type=float, default=None, help="过滤掉当日换手率分位数以上股票，如 0.8")
    ap.add_argument("--rule-weight", type=float, default=0.0, help="规则分在融合排序中的权重；0 表示只用模型预测")
    ap.add_argument("--min-rule-score", type=float, default=None, help="只保留规则分不低于该阈值的候选")
    ap.add_argument("--ridge-quantile", type=float, default=None, help="融合模型中只保留 Ridge 分数分位数以上候选，如 0.5")
    ap.add_argument("--lgbm-weight", type=float, default=1.0, help="Ridge+LightGBM 融合排序中 LightGBM 权重")
    ap.add_argument("--n-estimators", type=int, default=200, help="树模型最大训练轮次，配合验证集早停")
    ap.add_argument("--learning-rate", type=float, default=None, help="树模型学习率；默认按模型内部设置")
    ap.add_argument("--early-stopping-rounds", type=int, default=40, help="验证集早停轮数；<=0 关闭")
    ap.add_argument("--fill", choices=["next_open", "close"], default=("next_open" if bt_use_open_fill() else "close"),
                    help="回测成交口径：close=尾盘T买(默认,含可交易性摩擦)；next_open=T+1开盘买/卖。也可用环境变量 QUANT_BT_FILL")
    ap.add_argument("--no-tradable-filter", action="store_true", help="关闭可交易性口径(还原乐观：不理会涨停买不进/跌停顺延卖出)")
    ap.add_argument("--cost-roundtrip", type=float, default=bt_cost_roundtrip(),
                    help="单次调仓双边综合成本(佣金+印花税+滑点)，按换手计提；默认0.002。也可用环境变量 QUANT_BT_COST_ROUNDTRIP")
    ap.add_argument("--output-prefix", default="", help="实验输出前缀，避免覆盖默认 bt_* 产物")
    args = ap.parse_args()

    panel = warehouse.load(args.panel)
    if panel.empty:
        raise SystemExit(f"面板不存在或为空：{args.panel}")
    sel = warehouse.load(args.selection)
    factors = sel["factor"].tolist() if not sel.empty and "factor" in sel.columns else engineering.feature_columns(panel, args.horizon)
    feature_summary = warehouse.load("feature_summary")
    if factors and not feature_summary.empty and "feature" in feature_summary.columns:
        from quant.pipeline import _expand_selected_factors
        factors = _expand_selected_factors(factors, feature_summary["feature"].astype(str).tolist())
    if not factors:
        factors = engineering.feature_columns(panel, args.horizon)
    decay = args.decay_half_life_days if args.decay_half_life_days > 0 else None
    early_stopping_rounds = args.early_stopping_rounds if args.early_stopping_rounds > 0 else 0
    res = walk_forward(panel, factors, model_name=args.model, horizon=args.horizon,
                       train_months=args.train_months, validation_months=args.validation_months,
                       test_months=args.test_months, top_n=args.top_n,
                       max_weight=args.max_weight,
                       decay_half_life_days=decay, min_weight=args.min_weight,
                       ridge_alpha=args.ridge_alpha,
                       positive_only=args.positive_only, pred_quantile=args.pred_quantile,
                       volatility_quantile=args.volatility_quantile, turnover_quantile=args.turnover_quantile,
                       rule_weight=args.rule_weight, min_rule_score=args.min_rule_score,
                       ridge_quantile=args.ridge_quantile, lgbm_weight=args.lgbm_weight,
                       n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                        early_stopping_rounds=early_stopping_rounds,
                        use_open_fill=(args.fill == "next_open"),
                        filter_untradable=(False if args.no_tradable_filter else bt_filter_untradable()),
                        cost_roundtrip=args.cost_roundtrip)
    name_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    warehouse.save(f"{name_prefix}bt_{args.model}_returns", res["returns"])
    warehouse.save(f"{name_prefix}bt_{args.model}_holdings", res["holdings"])
    warehouse.save(f"{name_prefix}bt_{args.model}_predictions", res["predictions"])
    summary = pd.DataFrame([{ "model": args.model, **res["summary"] }])
    warehouse.save(f"{name_prefix}bt_{args.model}_summary", summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
