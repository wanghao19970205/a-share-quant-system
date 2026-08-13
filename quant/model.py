"""量化选股 · 模型训练。

支持 LightGBM、XGBoost、LSTM；依赖未安装时会跳过对应模型并给出提示。
为了让流程在轻量环境里也能跑通，内置一个纯 numpy ridge baseline。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant import config, warehouse
from quant.factors import engineering


@dataclass
class TrainResult:
    model: str
    ok: bool
    message: str
    metrics: dict
    predictions: pd.DataFrame


def _split_time(panel: pd.DataFrame, train_end: str | None = None, valid_start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pd.api.types.is_datetime64_any_dtype(panel["date"]):
        df = panel
    else:
        df = panel.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates = sorted(df["date"].dropna().unique())
    if not dates:
        return df.iloc[0:0], df.iloc[0:0]
    if train_end:
        te = pd.Timestamp(train_end)
        train = df[df["date"] <= te]
        valid = df[df["date"] > te]
    elif valid_start:
        vs = pd.Timestamp(valid_start)
        train = df[df["date"] < vs]
        valid = df[df["date"] >= vs]
    else:
        cut = dates[int(len(dates) * 0.7)]
        train = df[df["date"] <= cut]
        valid = df[df["date"] > cut]
    return train, valid


def _split_train_valid_predict(panel: pd.DataFrame, train_end: str | None = None,
                               valid_end: str | None = None,
                               predict_start: str | None = None,
                               predict_end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time so validation can tune rounds without touching test predictions."""
    train, valid = _split_time(panel, train_end=train_end)
    if valid_end is not None:
        ve = pd.Timestamp(valid_end)
        valid = valid[valid["date"] <= ve]
    if pd.api.types.is_datetime64_any_dtype(panel["date"]):
        predict = panel
    else:
        predict = panel.copy()
        predict["date"] = pd.to_datetime(predict["date"], errors="coerce")
    if predict_start is not None:
        ps = pd.Timestamp(predict_start)
        predict = predict[predict["date"] >= ps]
    else:
        predict = valid
    if predict_end is not None:
        pe = pd.Timestamp(predict_end)
        predict = predict[predict["date"] <= pe]
    return train, valid, predict


def _apply_train_mask(train: pd.DataFrame, mask_col: str | None) -> pd.DataFrame:
    """A/B 变体A：仅对训练段剔除买入日封涨停(buyable_close==False)的行。

    mask_col=None（默认）时原样返回 → baseline 路径零行为变更。mask 只作用于**训练样本**，
    valid/predict 段不受影响（预测仍覆盖全票）。掩码只看**买入日 T 当天**是否可买入，
    与 T+h 无关，因此不会误伤「T 买、T+1 涨停」这类正样本。"""
    if not mask_col or mask_col not in train.columns:
        return train
    keep = train[mask_col].fillna(True) != False  # noqa: E712 — 显式区分 False 与 NaN
    return train[keep]



def _validate_feature_safety(features: list[str]) -> None:
    unsafe = sorted({feature for feature in features if engineering.is_forbidden_feature(feature)})
    if unsafe:
        raise ValueError(f"unsafe future-label model features: {unsafe}")


def _xy(df: pd.DataFrame, features: list[str], target: str):
    _validate_feature_safety(features)
    sub = df[["code", "date", target] + features].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    x = sub[features].fillna(0.0).to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    return sub, x, y


def _predict_x(df: pd.DataFrame, features: list[str], target: str):
    _validate_feature_safety(features)
    cols = ["code", "date"] + ([target] if target in df.columns else []) + features
    sub = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    if target not in sub.columns:
        sub[target] = np.nan
    sub = sub.dropna(subset=["code", "date"])
    x = sub[features].fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
    return sub, x, y


def _rank_xy(df: pd.DataFrame, features: list[str], target: str, n_bins: int = 5):
    _validate_feature_safety(features)
    sub = df[["code", "date", target] + features].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if sub.empty:
        return sub, np.empty((0, len(features))), np.array([]), np.array([], dtype=int), np.array([], dtype=int), np.array([])
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
    keep_idx = []
    labels = []
    group = []
    qid = []
    for qi, (_, g) in enumerate(sub.groupby("date", sort=False)):
        if len(g) < 2:
            continue
        rank_pct = g[target].rank(method="first", pct=True)
        rel = np.floor(rank_pct.to_numpy(dtype=float) * n_bins).astype(int)
        rel = np.clip(rel, 0, n_bins - 1)
        keep_idx.extend(g.index.tolist())
        labels.extend(rel.tolist())
        group.append(len(g))
        qid.extend([qi] * len(g))
    if not keep_idx:
        return sub.iloc[0:0], np.empty((0, len(features))), np.array([]), np.array([], dtype=int), np.array([], dtype=int), np.array([])
    sub = sub.loc[keep_idx].reset_index(drop=True)
    x = sub[features].fillna(0.0).to_numpy(dtype=float)
    y_rank = np.asarray(labels, dtype=int)
    y_raw = sub[target].to_numpy(dtype=float)
    return sub, x, y_rank, np.asarray(group, dtype=int), np.asarray(qid, dtype=int), y_raw


def _rank_predict_x(df: pd.DataFrame, features: list[str], target: str):
    cols = ["code", "date"] + ([target] if target in df.columns else []) + features
    sub = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    if target not in sub.columns:
        sub[target] = np.nan
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["code", "date"]).sort_values(["date", "code"]).reset_index(drop=True)
    if sub.empty:
        return sub, np.empty((0, len(features))), np.array([])
    x = sub[features].fillna(0.0).to_numpy(dtype=float)
    y_raw = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
    return sub, x, y_raw


def time_decay_weights(df: pd.DataFrame, half_life_days: float = 90.0,
                       min_weight: float = 0.05,
                       asof_date: str | pd.Timestamp | None = None,
                       normalize: bool = True) -> np.ndarray:
    """按样本日期生成时间衰减权重，供 loss/sample_weight 使用。

    half_life_days=90 表示约 3 个月以前的样本权重减半；更旧样本继续指数衰减。
    min_weight 防止旧数据完全退场，保留跨行情状态的泛化能力。
    """
    if df.empty or "date" not in df.columns:
        return np.ones(len(df), dtype=float)
    dates = pd.to_datetime(df["date"], errors="coerce")
    anchor = pd.Timestamp(asof_date) if asof_date is not None else dates.max()
    age_days = (anchor - dates).dt.days.clip(lower=0).fillna(0).to_numpy(dtype=float)
    half_life_days = max(float(half_life_days), 1.0)
    w = np.power(0.5, age_days / half_life_days)
    w = np.clip(w, float(min_weight), 1.0)
    if normalize and len(w) and np.isfinite(w).any():
        mean = np.nanmean(w)
        if mean > 0:
            w = w / mean
    return w.astype(float)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() == 0:
        return {"n": 0}
    y = y[ok]
    pred = pred[ok]
    corr = pd.Series(pred).corr(pd.Series(y), method="spearman") if len(y) > 2 else np.nan
    top = pred >= np.nanquantile(pred, 0.8) if len(pred) >= 5 else np.ones_like(pred, dtype=bool)
    return {
        "n": int(len(y)),
        "rank_ic": float(corr) if np.isfinite(corr) else None,
        "mse": float(np.mean((pred - y) ** 2)),
        "top_mean_return": float(np.mean(y[top])) if top.any() else None,
        "all_mean_return": float(np.mean(y)),
    }


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(probability)
    if ok.sum() == 0:
        return {"n": 0}
    truth = y[ok].astype(int)
    score = np.clip(probability[ok].astype(float), 0.0, 1.0)
    result = {
        "n": int(len(truth)),
        "positive_rate": float(truth.mean()),
        "brier_score": float(np.mean((score - truth) ** 2)),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        risk_truth = 1 - truth
        risk_score = 1.0 - score
        result["risk_pr_auc"] = float(average_precision_score(risk_truth, risk_score))
        result["roc_auc"] = float(roc_auc_score(truth, score)) if len(np.unique(truth)) > 1 else None
    except Exception:  # noqa: BLE001
        result["risk_pr_auc"] = None
        result["roc_auc"] = None
    return result


def train_binary_classifier(
    panel: pd.DataFrame,
    features: list[str],
    label_col: str,
    classifier: str,
    train_end: str | None = None,
    valid_end: str | None = None,
    predict_start: str | None = None,
    decay_half_life_days: float | None = 90.0,
    min_weight: float = 0.05,
    minority_weight: float = 20.0,
    n_estimators: int = 160,
    learning_rate: float = 0.02,
    max_train_rows: int = 150_000,
    n_jobs: int | None = None,
    predict_end: str | None = None,
    enforce_max_train_rows: bool = False,
) -> TrainResult:
    train, valid, predict_df = _split_train_valid_predict(
        panel,
        train_end=train_end,
        valid_end=valid_end,
        predict_start=predict_start,
        predict_end=predict_end,
    )
    train_sub, x, y = _xy(train, features, label_col)
    valid_sub, xv, yv = _xy(valid, features, label_col)
    pred_sub, xp, _ = _predict_x(predict_df, features, label_col)
    model_name = {
        "ridge": "ridge_classifier",
        "lightgbm": "lightgbm_classifier",
        "elastic": "elastic_logistic",
        "extra_trees": "extra_trees_classifier",
    }.get(classifier, classifier)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0 or len(np.unique(y)) < 2:
        return TrainResult(model_name, False, "分类训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = (
        time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        if decay_half_life_days else np.ones(len(train_sub), dtype=float)
    )
    sample_weight = sample_weight * np.where(y < 0.5, float(minority_weight), 1.0)
    n_train_rows_before_cap = int(len(y))
    should_cap = classifier == "extra_trees" or bool(enforce_max_train_rows)
    if should_cap and max_train_rows > 0 and len(y) > max_train_rows:
        rng = np.random.default_rng(42)
        probability = sample_weight / sample_weight.sum()
        indices = np.sort(rng.choice(len(y), size=max_train_rows, replace=False, p=probability))
        x = x[indices]
        y = y[indices]
        sample_weight = sample_weight[indices]
    try:
        if classifier == "ridge":
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=42
            )
        elif classifier == "elastic":
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(
                penalty="elasticnet", C=1.0, l1_ratio=0.5, solver="saga",
                max_iter=1000, tol=1e-4, random_state=42, n_jobs=n_jobs,
            )
        elif classifier == "lightgbm":
            import lightgbm as lgb
            estimator = lgb.LGBMClassifier(
                objective="binary", n_estimators=n_estimators, learning_rate=float(learning_rate),
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                verbose=-1, n_jobs=n_jobs,
            )
        elif classifier == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            estimator = ExtraTreesClassifier(
                n_estimators=n_estimators, max_depth=12, min_samples_leaf=20,
                max_features=0.7, bootstrap=False, n_jobs=n_jobs, random_state=42,
            )
        else:
            return TrainResult(model_name, False, f"未知分类器：{classifier}", {}, pd.DataFrame())
        estimator.fit(x, y.astype(int), sample_weight=sample_weight)
        valid_probability = estimator.predict_proba(xv)[:, 1]
        probability = estimator.predict_proba(xp)[:, 1]
    except Exception as exc:  # noqa: BLE001
        return TrainResult(model_name, False, f"分类训练失败：{exc}", {}, pd.DataFrame())
    output = pred_sub[["code", "date", label_col]].copy()
    output["pred"] = probability
    metrics = _classification_metrics(yv, valid_probability)
    metrics["minority_weight"] = float(minority_weight)
    metrics["n_estimators"] = int(n_estimators)
    metrics["learning_rate"] = float(learning_rate)
    metrics["max_train_rows"] = int(max_train_rows)
    metrics["enforce_max_train_rows"] = bool(enforce_max_train_rows)
    metrics["n_train_rows_before_cap"] = n_train_rows_before_cap
    metrics["n_train_rows"] = int(len(y))
    return TrainResult(model_name, True, "ok", metrics, output)


def _daily_feature_zscore(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        mean = s.groupby(df["date"]).transform("mean")
        std = s.groupby(df["date"]).transform("std").replace(0, np.nan)
        out[f] = ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _ic_weights(train: pd.DataFrame, features: list[str], target: str, min_obs: int = 20,
                decay_half_life_days: float | None = 90.0, min_weight: float = 0.05) -> pd.Series:
    cols = ["date", target] + features
    sub = train[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if sub.empty:
        return pd.Series(0.0, index=features, dtype=float)
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    counts = sub.groupby("date")[target].transform("count")
    sub = sub[counts >= min_obs]
    if sub.empty:
        return pd.Series(0.0, index=features, dtype=float)

    y_rank = sub.groupby("date")[target].rank(method="average", pct=True)
    y_mean = y_rank.groupby(sub["date"]).transform("mean")
    y_centered = y_rank - y_mean
    y_var = (y_centered * y_centered).groupby(sub["date"]).transform("sum")

    x_rank = sub.groupby("date")[features].rank(method="average", pct=True)
    x_mean = x_rank.groupby(sub["date"]).transform("mean")
    x_centered = x_rank - x_mean
    x_var = (x_centered * x_centered).groupby(sub["date"]).transform("sum")
    cov = x_centered.multiply(y_centered, axis=0).groupby(sub["date"]).sum()
    denom = np.sqrt(x_var.groupby(sub["date"]).first().multiply(y_var.groupby(sub["date"]).first(), axis=0))
    ic_df = cov.divide(denom).replace([np.inf, -np.inf], np.nan)

    if decay_half_life_days:
        day_weight = time_decay_weights(pd.DataFrame({"date": ic_df.index}), decay_half_life_days, min_weight=min_weight, normalize=False)
        w = pd.Series(day_weight, index=ic_df.index, dtype=float)
        valid_w = ic_df.notna().multiply(w, axis=0)
        mean_ic = ic_df.fillna(0.0).multiply(w, axis=0).sum() / valid_w.sum().replace(0, np.nan)
    else:
        mean_ic = ic_df.mean(skipna=True)
    mean_ic = mean_ic.fillna(0.0)
    std_ic = ic_df.std(skipna=True).replace(0, np.nan)
    weight = (mean_ic / std_ic).replace([np.inf, -np.inf], np.nan).fillna(mean_ic)
    norm = weight.abs().sum()
    if norm > 0:
        weight = weight / norm
    return weight.reindex(features).fillna(0.0)


def train_ridge(panel: pd.DataFrame, features: list[str], horizon: int = 5, alpha: float = 10.0,
                train_end: str | None = None, valid_end: str | None = None,
                predict_start: str | None = None, decay_half_life_days: float | None = 90.0,
                min_weight: float = 0.05,
                label_col: str | None = None, train_mask_col: str | None = None,
                predict_end: str | None = None) -> TrainResult:
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end,
        predict_start=predict_start, predict_end=predict_end,
    )
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y = _xy(train, features, target)
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, _ = _predict_x(predict_df, features, target)
    if len(y) < 20 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("ridge", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    x1 = np.c_[np.ones(len(x)), x]
    xv1 = np.c_[np.ones(len(xv)), xv]
    xp1 = np.c_[np.ones(len(xp)), xp]
    reg = np.eye(x1.shape[1]) * alpha
    reg[0, 0] = 0.0
    if decay_half_life_days:
        w = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        xw = x1 * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        beta = np.linalg.lstsq(xw.T @ xw + reg, xw.T @ yw, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(x1.T @ x1 + reg, x1.T @ y, rcond=None)[0]
    valid_pred = xv1 @ beta
    pred = xp1 @ beta
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yv, valid_pred)
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("ridge", True, "ok", metrics, out)


def train_elastic_net(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                      alpha: float = 0.001, l1_ratio: float = 0.5,
                      train_end: str | None = None, valid_end: str | None = None,
                      predict_start: str | None = None, decay_half_life_days: float | None = 90.0,
                      min_weight: float = 0.05,
                      label_col: str | None = None, train_mask_col: str | None = None) -> TrainResult:
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y = _xy(train, features, target)
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, _ = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("elastic_net", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    try:
        from sklearn.linear_model import ElasticNet
        sample_weight = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight) if decay_half_life_days else None
        model_value = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), fit_intercept=True,
                                 max_iter=2000, tol=1e-4, selection="cyclic", random_state=42)
        model_value.fit(x, y, sample_weight=sample_weight)
        valid_pred = model_value.predict(xv)
        pred = model_value.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("elastic_net", False, f"ElasticNet 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yv, valid_pred)
    metrics["alpha"] = float(alpha)
    metrics["l1_ratio"] = float(l1_ratio)
    metrics["n_nonzero_factors"] = int(np.count_nonzero(np.abs(model_value.coef_) > 1e-12))
    return TrainResult("elastic_net", True, "ok", metrics, out)


def train_ic_weighted(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                      train_end: str | None = None, valid_end: str | None = None,
                      predict_start: str | None = None, decay_half_life_days: float | None = 90.0,
                      min_weight: float = 0.05,
                      label_col: str | None = None, train_mask_col: str | None = None) -> TrainResult:
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, _, y = _xy(train, features, target)
    valid_sub, _, yv = _xy(valid, features, target)
    pred_sub, _, _ = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or pred_sub.empty:
        return TrainResult("ic_weighted", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    weight = _ic_weights(train_sub, features, target, decay_half_life_days=decay_half_life_days, min_weight=min_weight)
    if not np.isfinite(weight.to_numpy(dtype=float)).any() or weight.abs().sum() == 0:
        return TrainResult("ic_weighted", False, "因子 IC 权重为空", {}, pd.DataFrame())
    valid_xz = _daily_feature_zscore(valid_sub, features)
    valid_pred = valid_xz[features].to_numpy(dtype=float) @ weight.to_numpy(dtype=float)
    pred_xz = _daily_feature_zscore(pred_sub, features)
    pred = pred_xz[features].to_numpy(dtype=float) @ weight.to_numpy(dtype=float)
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yv, valid_pred)
    metrics["n_nonzero_factors"] = int((weight.abs() > 0).sum())
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("ic_weighted", True, "ok", metrics, out)


def train_rank_vote(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                    train_end: str | None = None, valid_end: str | None = None,
                    predict_start: str | None = None, decay_half_life_days: float | None = 90.0,
                    min_weight: float = 0.05,
                    label_col: str | None = None, train_mask_col: str | None = None) -> TrainResult:
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, _, y = _xy(train, features, target)
    valid_sub, _, yv = _xy(valid, features, target)
    pred_sub, _, _ = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or pred_sub.empty:
        return TrainResult("rank_vote", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    weight = _ic_weights(train_sub, features, target, decay_half_life_days=decay_half_life_days, min_weight=min_weight)
    if not np.isfinite(weight.to_numpy(dtype=float)).any() or weight.abs().sum() == 0:
        return TrainResult("rank_vote", False, "因子 IC 权重为空", {}, pd.DataFrame())
    valid_ranks = valid_sub.groupby("date")[features].rank(method="average", pct=True)
    valid_centered = (valid_ranks - 0.5).fillna(0.0)
    valid_pred = valid_centered[features].to_numpy(dtype=float) @ weight.to_numpy(dtype=float)
    ranks = pred_sub.groupby("date")[features].rank(method="average", pct=True)
    centered = (ranks - 0.5).fillna(0.0)
    pred = centered[features].to_numpy(dtype=float) @ weight.to_numpy(dtype=float)
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yv, valid_pred)
    metrics["n_nonzero_factors"] = int((weight.abs() > 0).sum())
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("rank_vote", True, "ok", metrics, out)


def _lightgbm_objective_params(objective: str, alpha: float | None) -> dict:
    objective = str(objective).strip().lower()
    if objective == "regression":
        if alpha is not None:
            raise ValueError("alpha is only valid for the quantile objective")
        return {"objective": "regression", "eval_metric": "l2"}
    if objective == "quantile":
        value = float(alpha) if alpha is not None else 0.20
        if not 0.0 < value < 1.0:
            raise ValueError("quantile alpha must be strictly between zero and one")
        return {"objective": "quantile", "alpha": value, "eval_metric": "quantile"}
    raise ValueError(f"unsupported LightGBM objective: {objective}")


def train_lightgbm(panel: pd.DataFrame, features: list[str], horizon: int = 5, train_end: str | None = None,
                   valid_end: str | None = None, predict_start: str | None = None,
                   decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                   n_estimators: int = 800, learning_rate: float | None = None,
                   early_stopping_rounds: int = 50, n_jobs: int | None = None,
                   label_col: str | None = None,
                   train_mask_col: str | None = None,
                   objective: str = "regression", alpha: float | None = None,
                   predict_end: str | None = None,
                   max_train_rows: int | None = None,
                   enforce_max_train_rows: bool = False) -> TrainResult:
    try:
        import lightgbm as lgb
    except Exception as e:  # noqa: BLE001
        return TrainResult("lightgbm", False, f"缺少 lightgbm：{e}", {}, pd.DataFrame())
    objective_params = _lightgbm_objective_params(objective, alpha)
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel,
        train_end=train_end,
        valid_end=valid_end,
        predict_start=predict_start,
        predict_end=predict_end,
    )
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y = _xy(train, features, target)
    n_train_rows_before_cap = int(len(y))
    if enforce_max_train_rows and max_train_rows and len(y) > int(max_train_rows):
        rng = np.random.default_rng(42)
        cap_weights = time_decay_weights(
            train_sub, decay_half_life_days, min_weight=min_weight
        ) if decay_half_life_days else np.ones(len(y), dtype=float)
        indices = np.sort(rng.choice(
            len(y), size=int(max_train_rows), replace=False,
            p=cap_weights / cap_weights.sum(),
        ))
        train_sub, x, y = train_sub.iloc[indices], x[indices], y[indices]
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, yp = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("lightgbm", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight) if decay_half_life_days else None
    try:
        model = lgb.LGBMRegressor(
            objective=objective_params["objective"],
            **({"alpha": objective_params["alpha"]} if "alpha" in objective_params else {}),
            n_estimators=n_estimators,
            learning_rate=learning_rate or 0.02,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=n_jobs,
        )
        fit_kwargs = {
            "sample_weight": sample_weight,
            "eval_set": [(xv, yv)],
            "eval_metric": objective_params["eval_metric"],
        }
        try:
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds else []
            model.fit(x, y, callbacks=callbacks, **fit_kwargs)
        except TypeError:
            model.fit(x, y, **fit_kwargs)
        pred = model.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("lightgbm", False, f"lightgbm 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp, pred)
    metrics["n_train_rows_before_cap"] = n_train_rows_before_cap
    metrics["n_train_rows"] = int(len(y))
    metrics["max_train_rows"] = int(max_train_rows) if max_train_rows is not None else None
    metrics["enforce_max_train_rows"] = bool(enforce_max_train_rows)
    if hasattr(model, "best_iteration_"):
        metrics["best_iteration"] = int(model.best_iteration_ or 0)
    metrics["objective"] = objective_params["objective"]
    if "alpha" in objective_params:
        metrics["alpha"] = float(objective_params["alpha"])
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("lightgbm", True, "ok", metrics, out)


def train_xgboost(panel: pd.DataFrame, features: list[str], horizon: int = 5, train_end: str | None = None,
                  valid_end: str | None = None, predict_start: str | None = None,
                  decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                  n_estimators: int = 800, learning_rate: float | None = None,
                  early_stopping_rounds: int = 50) -> TrainResult:
    try:
        import xgboost as xgb
    except Exception as e:  # noqa: BLE001
        return TrainResult("xgboost", False, f"缺少 xgboost：{e}", {}, pd.DataFrame())
    target = f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train_sub, x, y = _xy(train, features, target)
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, yp = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("xgboost", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight) if decay_half_life_days else None
    try:
        model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate or 0.02, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0, objective="reg:squarederror", random_state=42, n_jobs=1, verbosity=0)
        try:
            fit_kwargs = {"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}
            model.fit(x, y, sample_weight=sample_weight, eval_set=[(xv, yv)], verbose=False, **fit_kwargs)
        except TypeError:
            model.fit(x, y, sample_weight=sample_weight, eval_set=[(xv, yv)], verbose=False)
        pred = model.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("xgboost", False, f"xgboost 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp, pred)
    if hasattr(model, "best_iteration"):
        best_iteration = getattr(model, "best_iteration", None)
        metrics["best_iteration"] = int(best_iteration) if best_iteration is not None else 0
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("xgboost", True, "ok", metrics, out)


def train_lightgbm_ranker(panel: pd.DataFrame, features: list[str], horizon: int = 5, train_end: str | None = None,
                          valid_end: str | None = None, predict_start: str | None = None,
                          decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                          n_estimators: int = 800, learning_rate: float | None = None,
                          early_stopping_rounds: int = 50, n_jobs: int | None = None,
                          model_threads: int | None = None,
                          rank_bins: int = 5, eval_at: tuple[int, ...] = (3,),
                          label_col: str | None = None, train_mask_col: str | None = None,
                          predict_end: str | None = None) -> TrainResult:
    try:
        import lightgbm as lgb
    except Exception as e:  # noqa: BLE001
        return TrainResult("lightgbm_ranker", False, f"缺少 lightgbm：{e}", {}, pd.DataFrame())
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end,
        predict_start=predict_start, predict_end=predict_end,
    )
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y, group, _, _ = _rank_xy(train, features, target, n_bins=rank_bins)
    valid_sub, xv, yv, valid_group, _, yv_raw = _rank_xy(
        valid, features, target, n_bins=rank_bins
    )
    pred_sub, xp, yp_raw = _rank_predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("lightgbm_ranker", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight) if decay_half_life_days else None
    try:
        x_df = pd.DataFrame(x, columns=features)
        xv_df = pd.DataFrame(xv, columns=features)
        xp_df = pd.DataFrame(xp, columns=features)
        ranker_threads = model_threads if model_threads is not None else n_jobs
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=n_estimators, learning_rate=learning_rate or 0.02, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=ranker_threads)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds else []
        model.fit(x_df, y, group=group.tolist(), sample_weight=sample_weight,
                  eval_set=[(xv_df, yv)], eval_group=[valid_group.tolist()],
                  eval_at=list(eval_at), callbacks=callbacks)
        pred = model.predict(xp_df)
    except Exception as e:  # noqa: BLE001
        return TrainResult("lightgbm_ranker", False, f"lightgbm_ranker 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp_raw, pred)
    if hasattr(model, "best_iteration_"):
        metrics["best_iteration"] = int(model.best_iteration_ or 0)
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    metrics["rank_bins"] = int(rank_bins)
    metrics["eval_at"] = [int(value) for value in eval_at]
    return TrainResult("lightgbm_ranker", True, "ok", metrics, out)


def train_extra_trees(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                      train_end: str | None = None, valid_end: str | None = None,
                      predict_start: str | None = None,
                      decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                      n_estimators: int = 120, max_train_rows: int = 300_000,
                      label_col: str | None = None, train_mask_col: str | None = None) -> TrainResult:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except Exception as e:  # noqa: BLE001
        return TrainResult("extra_trees", False, f"缺少 scikit-learn：{e}", {}, pd.DataFrame())
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y = _xy(train, features, target)
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, yp = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("extra_trees", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = (
        time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        if decay_half_life_days else np.ones(len(train_sub), dtype=float)
    )
    if max_train_rows > 0 and len(train_sub) > max_train_rows:
        rng = np.random.default_rng(42)
        probability = sample_weight / sample_weight.sum()
        idx = np.sort(rng.choice(len(train_sub), size=max_train_rows, replace=False, p=probability))
        x = x[idx]
        y = y[idx]
        sample_weight = sample_weight[idx]
    try:
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=20,
            max_features=0.7,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(x, y, sample_weight=sample_weight)
        pred = model.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("extra_trees", False, f"extra_trees 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp, pred)
    metrics["n_estimators"] = int(n_estimators)
    metrics["n_train_rows"] = int(len(y))
    metrics["max_train_rows"] = int(max_train_rows)
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("extra_trees", True, "ok", metrics, out)


def train_random_forest(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                        train_end: str | None = None, valid_end: str | None = None,
                        predict_start: str | None = None,
                        decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                        n_estimators: int = 120, max_train_rows: int = 300_000,
                        label_col: str | None = None, train_mask_col: str | None = None) -> TrainResult:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as error:  # noqa: BLE001
        return TrainResult("random_forest", False, f"缺少 scikit-learn：{error}", {}, pd.DataFrame())
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y = _xy(train, features, target)
    valid_sub, xv, yv = _xy(valid, features, target)
    pred_sub, xp, yp = _predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("random_forest", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = (
        time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        if decay_half_life_days else np.ones(len(train_sub), dtype=float)
    )
    if max_train_rows > 0 and len(train_sub) > max_train_rows:
        rng = np.random.default_rng(42)
        probability = sample_weight / sample_weight.sum()
        idx = np.sort(rng.choice(len(train_sub), size=max_train_rows, replace=False, p=probability))
        x, y, sample_weight = x[idx], y[idx], sample_weight[idx]
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=12, min_samples_leaf=20,
            max_features=0.7, bootstrap=True, n_jobs=-1, random_state=42,
        )
        model.fit(x, y, sample_weight=sample_weight)
        pred = model.predict(xp)
    except Exception as error:  # noqa: BLE001
        return TrainResult("random_forest", False, f"random_forest 训练失败：{error}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp, pred)
    metrics.update({"n_estimators": int(n_estimators), "n_train_rows": int(len(y)),
                    "max_train_rows": int(max_train_rows)})
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("random_forest", True, "ok", metrics, out)


def train_catboost_ranker(panel: pd.DataFrame, features: list[str], horizon: int = 5,
                          train_end: str | None = None, valid_end: str | None = None,
                          predict_start: str | None = None,
                          decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                          n_estimators: int = 200, learning_rate: float | None = None,
                          early_stopping_rounds: int = 50, n_jobs: int | None = None,
                          max_train_rows: int = 0, label_col: str | None = None,
                          train_mask_col: str | None = None) -> TrainResult:
    try:
        from catboost import CatBoostRanker, Pool
    except Exception as e:  # noqa: BLE001
        return TrainResult("catboost_ranker", False, f"缺少 catboost：{e}", {}, pd.DataFrame())
    target = label_col or f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(
        panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train = _apply_train_mask(train, train_mask_col)
    train_sub, x, y, _, qid, _ = _rank_xy(train, features, target)
    valid_sub, xv, yv, _, valid_qid, yv_raw = _rank_xy(valid, features, target)
    pred_sub, xp, yp_raw = _rank_predict_x(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(xp) == 0:
        return TrainResult("catboost_ranker", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = (
        time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        if decay_half_life_days else np.ones(len(train_sub), dtype=float)
    )
    if max_train_rows > 0 and len(train_sub) > max_train_rows:
        # Ranker groups must stay intact. Keep the newest complete trading-day groups
        # until the row budget is reached; this also preserves the intended time decay.
        group_sizes = train_sub.groupby("date", sort=False).size()
        keep_dates: list[pd.Timestamp] = []
        kept_rows = 0
        for date, size in reversed(list(group_sizes.items())):
            if keep_dates and kept_rows + int(size) > max_train_rows:
                break
            keep_dates.append(date)
            kept_rows += int(size)
        keep = train_sub["date"].isin(keep_dates).to_numpy()
        train_sub = train_sub.loc[keep].reset_index(drop=True)
        x = x[keep]
        y = y[keep]
        sample_weight = sample_weight[keep]
        grouped = train_sub.groupby("date", sort=False).size().to_numpy(dtype=int)
        qid = np.repeat(np.arange(len(grouped), dtype=int), grouped)
    available_cpus = os.cpu_count() or 1
    requested_threads = int(n_jobs) if n_jobs and int(n_jobs) > 0 else available_cpus
    thread_count = max(1, min(requested_threads, 8, available_cpus))
    try:
        train_pool = Pool(x, label=y, group_id=qid, group_weight=sample_weight)
        valid_pool = Pool(xv, label=yv, group_id=valid_qid)
        model = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric="NDCG:top=3",
            iterations=n_estimators,
            learning_rate=learning_rate or 0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=42,
            thread_count=thread_count,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds or None,
            verbose=False,
        )
        pred = model.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("catboost_ranker", False, f"catboost_ranker 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp_raw, pred)
    best_iteration = model.get_best_iteration()
    metrics["best_iteration"] = int(best_iteration) if best_iteration is not None and best_iteration >= 0 else 0
    metrics["n_train_rows"] = int(len(train_sub))
    metrics["max_train_rows"] = int(max_train_rows)
    metrics["n_jobs"] = int(thread_count)
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("catboost_ranker", True, "ok", metrics, out)


def train_xgboost_ranker(panel: pd.DataFrame, features: list[str], horizon: int = 5, train_end: str | None = None,
                         valid_end: str | None = None, predict_start: str | None = None,
                         decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
                         n_estimators: int = 200, learning_rate: float | None = None,
                         early_stopping_rounds: int = 50) -> TrainResult:
    try:
        import xgboost as xgb
    except Exception as e:  # noqa: BLE001
        return TrainResult("xgboost_ranker", False, f"缺少 xgboost：{e}", {}, pd.DataFrame())
    target = f"target_ret_{horizon}d"
    train, valid, predict_df = _split_train_valid_predict(panel, train_end=train_end, valid_end=valid_end, predict_start=predict_start)
    train_sub, x, y, _, qid, _ = _rank_xy(train, features, target)
    valid_sub, xv, yv, _, valid_qid, yv_raw = _rank_xy(valid, features, target)
    pred_sub, xp, _, _, _, yp_raw = _rank_xy(predict_df, features, target)
    if len(y) < 50 or len(yv) == 0 or len(yp_raw) == 0:
        return TrainResult("xgboost_ranker", False, "训练、验证或预测样本不足", {}, pd.DataFrame())
    sample_weight = None
    if decay_half_life_days:
        row_weight = time_decay_weights(train_sub, decay_half_life_days, min_weight=min_weight)
        sample_weight = pd.DataFrame({"date": train_sub["date"].to_numpy(), "w": row_weight}).groupby("date", sort=False)["w"].mean().to_numpy(dtype=float)
    try:
        model = xgb.XGBRanker(n_estimators=n_estimators, learning_rate=learning_rate or 0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0, objective="rank:ndcg", eval_metric="ndcg@3", random_state=42, n_jobs=1, verbosity=0)
        try:
            fit_kwargs = {"early_stopping_rounds": early_stopping_rounds} if early_stopping_rounds else {}
            model.fit(x, y, qid=qid, sample_weight=sample_weight, eval_set=[(xv, yv)], eval_qid=[valid_qid], verbose=False, **fit_kwargs)
        except TypeError:
            model.fit(x, y, qid=qid, sample_weight=sample_weight, eval_set=[(xv, yv)], eval_qid=[valid_qid], verbose=False)
        pred = model.predict(xp)
    except Exception as e:  # noqa: BLE001
        return TrainResult("xgboost_ranker", False, f"xgboost_ranker 训练失败：{e}", {}, pd.DataFrame())
    out = pred_sub[["code", "date", target]].copy()
    out["pred"] = pred
    metrics = _metrics(yp_raw, pred)
    if hasattr(model, "best_iteration"):
        best_iteration = getattr(model, "best_iteration", None)
        metrics["best_iteration"] = int(best_iteration) if best_iteration is not None else 0
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("xgboost_ranker", True, "ok", metrics, out)


def train_lstm(panel: pd.DataFrame, features: list[str], horizon: int = 5, train_end: str | None = None,
               lookback: int = 20, decay_half_life_days: float | None = 90.0,
               min_weight: float = 0.05) -> TrainResult:
    try:
        import torch
        from torch import nn
    except Exception as e:  # noqa: BLE001
        return TrainResult("lstm", False, f"缺少 torch：{e}", {}, pd.DataFrame())

    target = f"target_ret_{horizon}d"
    train, valid = _split_time(panel, train_end=train_end)

    def seqs(df: pd.DataFrame):
        xs, ys, meta = [], [], []
        for code, g in df.sort_values("date").groupby("code"):
            vals = g[features].fillna(0.0).to_numpy(dtype=np.float32)
            y = g[target].to_numpy(dtype=np.float32)
            dates = g["date"].to_numpy()
            for i in range(lookback, len(g)):
                if not np.isfinite(y[i]):
                    continue
                xs.append(vals[i - lookback:i])
                ys.append(y[i])
                meta.append((code, dates[i]))
        if not xs:
            return None, None, [], None
        meta_df = pd.DataFrame(meta, columns=["code", "date"])
        weights = time_decay_weights(meta_df, decay_half_life_days, min_weight=min_weight) if decay_half_life_days else np.ones(len(meta), dtype=float)
        return torch.tensor(np.stack(xs)), torch.tensor(np.array(ys)).view(-1, 1), meta, torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    x, y, _, w = seqs(train)
    xv, yv, meta, _ = seqs(valid)
    if x is None or xv is None or len(y) < 50:
        return TrainResult("lstm", False, "训练或验证序列不足", {}, pd.DataFrame())

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(n, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
        def forward(self, z):
            o, _ = self.lstm(z)
            return self.fc(o[:, -1, :])

    net = Net(len(features))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(8):
        opt.zero_grad()
        err = (net(x) - y) ** 2
        loss = (err * w).mean() if w is not None else err.mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(xv).view(-1).numpy()
    yv_np = yv.view(-1).numpy()
    out = pd.DataFrame(meta, columns=["code", "date"])
    out[target] = yv_np
    out["pred"] = pred
    metrics = _metrics(yv_np, pred)
    if decay_half_life_days:
        metrics["decay_half_life_days"] = float(decay_half_life_days)
        metrics["min_weight"] = float(min_weight)
    return TrainResult("lstm", True, "ok", metrics, out)


def train_all(panel: pd.DataFrame, horizon: int = 5, models: list[str] | None = None,
              factors: list[str] | None = None, train_end: str | None = None,
              valid_end: str | None = None, predict_start: str | None = None,
              decay_half_life_days: float | None = 90.0, min_weight: float = 0.05,
              ridge_alpha: float = 10.0, n_estimators: int = 800,
              learning_rate: float | None = None, early_stopping_rounds: int = 50) -> list[TrainResult]:
    factors = factors or engineering.feature_columns(panel, horizon)
    models = models or ["ridge", "lightgbm", "xgboost", "lightgbm_ranker", "xgboost_ranker", "lstm"]
    registry = {
        "ridge": train_ridge,
        "ic_weighted": train_ic_weighted,
        "lightgbm": train_lightgbm,
        "xgboost": train_xgboost,
        "lightgbm_ranker": train_lightgbm_ranker,
        "xgboost_ranker": train_xgboost_ranker,
        "lstm": train_lstm,
    }
    results = []
    for m in models:
        fn = registry.get(m)
        if not fn:
            results.append(TrainResult(m, False, "未知模型", {}, pd.DataFrame()))
            continue
        kwargs = {
            "horizon": horizon,
            "train_end": train_end,
            "decay_half_life_days": decay_half_life_days,
            "min_weight": min_weight,
        }
        if m == "ridge":
            kwargs["alpha"] = ridge_alpha
            kwargs["valid_end"] = valid_end
            kwargs["predict_start"] = predict_start
        elif m in {"lightgbm", "xgboost", "lightgbm_ranker", "xgboost_ranker"}:
            kwargs["valid_end"] = valid_end
            kwargs["predict_start"] = predict_start
            kwargs["n_estimators"] = n_estimators
            kwargs["learning_rate"] = learning_rate
            kwargs["early_stopping_rounds"] = early_stopping_rounds
        results.append(fn(panel, factors, **kwargs))
    return results


def main():
    ap = argparse.ArgumentParser(description="训练量化选股模型")
    ap.add_argument("--panel", default="factor_panel")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--models", default="ridge,lightgbm,xgboost,lstm")
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--selection", default="factor_selection")
    ap.add_argument("--decay-half-life-days", type=float, default=90.0, help="时间衰减半衰期；<=0 表示关闭")
    ap.add_argument("--min-weight", type=float, default=0.05, help="旧样本最小权重")
    ap.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge L2 正则强度")
    args = ap.parse_args()

    panel = warehouse.load(args.panel)
    if panel.empty:
        raise SystemExit(f"面板不存在或为空：{args.panel}")
    sel = warehouse.load(args.selection)
    factors = sel["factor"].tolist() if not sel.empty and "factor" in sel.columns else engineering.feature_columns(panel, args.horizon)
    if not factors:
        factors = engineering.feature_columns(panel, args.horizon)
    decay = args.decay_half_life_days if args.decay_half_life_days > 0 else None
    results = train_all(panel, horizon=args.horizon, models=[m.strip() for m in args.models.split(",") if m.strip()],
                        factors=factors, train_end=args.train_end,
                        decay_half_life_days=decay, min_weight=args.min_weight,
                        ridge_alpha=args.ridge_alpha)

    rows = []
    for res in results:
        rows.append({"model": res.model, "ok": res.ok, "message": res.message, **res.metrics})
        if res.ok and not res.predictions.empty:
            warehouse.save(f"pred_{res.model}", res.predictions)
    summary = pd.DataFrame(rows)
    warehouse.save("model_summary", summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
