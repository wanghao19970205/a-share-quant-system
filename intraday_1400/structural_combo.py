from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


STRUCTURAL_STRESS_RETURN = -0.10
STRUCTURAL_BLEND_WEIGHT = 0.50


def fit_probability_calibrator(raw_score, truth) -> dict:
    frame = pd.DataFrame({
        "score": pd.to_numeric(pd.Series(raw_score), errors="coerce"),
        "truth": pd.to_numeric(pd.Series(truth), errors="coerce"),
    }).dropna()
    frame = frame[frame["truth"].isin([0.0, 1.0])]
    if frame.empty:
        raise ValueError("probability calibration requires observed binary labels")
    empirical = float(frame["truth"].mean())
    if frame["truth"].nunique() < 2 or frame["score"].nunique() < 2:
        return {
            "kind": "constant",
            "probability": empirical,
            "rows": int(len(frame)),
            "positive_rate": empirical,
        }
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(frame[["score"]], frame["truth"].astype(int))
    probability = model.predict_proba(frame[["score"]])[:, 1]
    metrics = {
        "brier_score": float(brier_score_loss(frame["truth"], probability)),
        "roc_auc": float(roc_auc_score(frame["truth"], probability)),
    }
    return {
        "kind": "platt",
        "intercept": float(model.intercept_[0]),
        "coefficient": float(model.coef_[0, 0]),
        "rows": int(len(frame)),
        "positive_rate": empirical,
        "metrics": metrics,
    }


def apply_probability_calibrator(raw_score, calibrator: dict) -> pd.Series:
    score = pd.to_numeric(pd.Series(raw_score), errors="coerce")
    if calibrator["kind"] == "constant":
        result = pd.Series(float(calibrator["probability"]), index=score.index)
    elif calibrator["kind"] == "platt":
        linear = float(calibrator["intercept"]) + float(calibrator["coefficient"]) * score
        result = 1.0 / (1.0 + np.exp(-linear.clip(-35.0, 35.0)))
    else:
        raise ValueError(f"unknown calibrator kind: {calibrator.get('kind')}")
    return result.clip(1e-4, 1.0 - 1e-4)


def _raw_head(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    value_column = next(
        (
            column
            for column in (name, "raw_pred", "probability", "score", "pred", "ensemble_pred")
            if column in frame
        ),
        None,
    )
    if value_column is None:
        raise ValueError(f"{name} head has no prediction value")
    result = frame[["code", "date", value_column]].copy()
    result = result.rename(columns={value_column: name})
    result["code"] = result["code"].astype(str).str[:6]
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result[name] = pd.to_numeric(result[name], errors="coerce")
    return result.dropna().drop_duplicates(["code", "date"], keep="last")


def _cross_section_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("date")[column]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda values: values.std(ddof=0)).replace(0.0, np.nan)
    return ((frame[column] - mean) / std).fillna(0.0)


def build_structural_combo_scores(
    direct_head: pd.DataFrame,
    buy_head: pd.DataFrame,
    liquidation_head: pd.DataFrame,
    conditional_return_head: pd.DataFrame,
    stress_return: float = STRUCTURAL_STRESS_RETURN,
    blend_weight: float = STRUCTURAL_BLEND_WEIGHT,
) -> dict[str, pd.DataFrame]:
    weight = float(blend_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("blend_weight must be between zero and one")
    merged = _raw_head(direct_head, "c0_direct")
    for frame, name in (
        (buy_head, "p_buy"),
        (liquidation_head, "p_liquidate"),
        (conditional_return_head, "conditional_return"),
    ):
        merged = merged.merge(
            _raw_head(frame, name),
            on=["code", "date"],
            how="inner",
            validate="one_to_one",
        )
    merged["p_buy"] = merged["p_buy"].clip(0.0, 1.0)
    merged["p_liquidate"] = merged["p_liquidate"].clip(0.0, 1.0)
    merged["conditional_return"] = merged["conditional_return"].clip(-0.20, 0.20)
    merged["c1_structural"] = merged["p_buy"] * (
        merged["p_liquidate"] * merged["conditional_return"]
        + (1.0 - merged["p_liquidate"]) * float(stress_return)
    )
    merged["c0_z"] = _cross_section_zscore(merged, "c0_direct")
    merged["c1_z"] = _cross_section_zscore(merged, "c1_structural")
    merged["c2_blend"] = weight * merged["c0_z"] + (1.0 - weight) * merged["c1_z"]
    outputs = {}
    for model, column in (
        ("c0_direct", "c0_direct"),
        ("c1_structural", "c1_structural"),
        ("c2_fixed_50_50", "c2_blend"),
    ):
        frame = merged.copy()
        frame["score"] = frame[column]
        frame["model_variant"] = model
        outputs[model] = frame
    return outputs


def shift_daily_prior_to_signal(
    daily_predictions: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    prior = _raw_head(daily_predictions, "e4_daily_prior")
    dates = pd.DatetimeIndex(trading_calendar).normalize().drop_duplicates().sort_values()
    next_date = {
        pd.Timestamp(dates[index]): pd.Timestamp(dates[index + 1])
        for index in range(len(dates) - 1)
    }
    prior["source_date"] = prior["date"]
    prior["date"] = prior["source_date"].map(next_date)
    return prior.dropna(subset=["date"]).reset_index(drop=True)


def build_daily_execution_filter_scores(
    daily_prior: pd.DataFrame,
    buy_probability: pd.DataFrame,
    candidate_n: int = 100,
    minimum_buy_probability: float = 0.50,
) -> pd.DataFrame:
    candidate_n = int(candidate_n)
    threshold = float(minimum_buy_probability)
    if candidate_n < 10:
        raise ValueError("candidate_n must retain at least ten names")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_buy_probability must be between zero and one")
    daily = _raw_head(daily_prior, "e4_daily_prior")
    buy = _raw_head(buy_probability, "p_buy")
    daily = daily.sort_values(
        ["date", "e4_daily_prior", "code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    daily["e4_candidate_rank"] = daily.groupby("date").cumcount() + 1
    candidates = daily[daily["e4_candidate_rank"] <= candidate_n]
    merged = candidates.merge(
        buy,
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged[merged["p_buy"] >= threshold].copy()
    merged["score"] = merged["e4_daily_prior"]
    merged["model_variant"] = "e4_top100_buy_filter_daily_rank"
    return merged.sort_values(["date", "code"]).reset_index(drop=True)


def e4_coverage_diagnostics(
    daily_prior: pd.DataFrame,
    minute_combo: pd.DataFrame,
    expected_dates: pd.DatetimeIndex,
    candidate_n: int = 100,
) -> dict:
    expected = pd.DatetimeIndex(expected_dates).normalize().drop_duplicates().sort_values()
    daily = _raw_head(daily_prior, "e4_daily_prior")
    minute = _raw_head(minute_combo, "minute_combo")
    available = pd.DatetimeIndex(daily["date"].unique()).sort_values()
    missing = expected.difference(available)
    if len(missing):
        formatted = ",".join(str(pd.Timestamp(value).date()) for value in missing)
        raise ValueError(f"E4 daily prior missing expected dates: {formatted}")
    daily = daily[daily["date"].isin(expected)].sort_values(
        ["date", "e4_daily_prior", "code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    daily["e4_candidate_rank"] = daily.groupby("date").cumcount() + 1
    candidates = daily[daily["e4_candidate_rank"] <= int(candidate_n)]
    merged = candidates.merge(
        minute[["code", "date"]],
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    candidate_rows = int(len(candidates))
    merged_rows = int(len(merged))
    calendar_payload = "\n".join(
        str(pd.Timestamp(value).date()) for value in expected
    ).encode("ascii")
    return {
        "calendar_sha256": hashlib.sha256(calendar_payload).hexdigest(),
        "expected_dates": int(len(expected)),
        "available_dates": int(len(available.intersection(expected))),
        "missing_dates": [],
        "daily_prior_rows": int(len(daily)),
        "candidate_rows": candidate_rows,
        "minute_matched_candidate_rows": merged_rows,
        "top_candidate_retention": (
            float(merged_rows / candidate_rows) if candidate_rows else None
        ),
    }


def build_e0_e4_staged_scores(
    daily_prior: pd.DataFrame,
    minute_combo: pd.DataFrame,
    candidate_n: int = 100,
) -> dict[str, pd.DataFrame]:
    candidate_n = int(candidate_n)
    if candidate_n < 10:
        raise ValueError("candidate_n must retain at least ten names")
    daily = _raw_head(daily_prior, "e4_daily_prior")
    minute = _raw_head(minute_combo, "minute_combo")
    daily = daily.sort_values(
        ["date", "e4_daily_prior", "code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    daily["e4_candidate_rank"] = daily.groupby("date").cumcount() + 1
    candidates = daily[daily["e4_candidate_rank"] <= candidate_n].copy()
    merged = candidates.merge(
        minute,
        on=["code", "date"],
        how="inner",
        validate="one_to_one",
    )
    merged["e4_z_top100"] = _cross_section_zscore(merged, "e4_daily_prior")
    merged["minute_z_top100"] = _cross_section_zscore(merged, "minute_combo")
    merged["e0_e4_score"] = (
        STRUCTURAL_BLEND_WEIGHT * merged["e4_z_top100"]
        + (1.0 - STRUCTURAL_BLEND_WEIGHT) * merged["minute_z_top100"]
    )
    minute_ranked = merged.copy()
    minute_ranked["score"] = minute_ranked["minute_combo"]
    minute_ranked["model_variant"] = "e4_top100_minute_block"
    full = merged.copy()
    full["score"] = full["e0_e4_score"]
    full["model_variant"] = "e0_e4_staged_combo"
    return {
        "e4_top100_minute_block": minute_ranked,
        "e0_e4_staged_combo": full,
    }
