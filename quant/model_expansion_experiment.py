"""Shadow-only helpers for neutral residual and market-regime model experiments.

The functions in this module never publish active artifacts. They provide point-in-time
label neutralization and deterministic regime construction for walk-forward research.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_target_cross_section(
    frame: pd.DataFrame,
    target: str,
    industry: pd.Series | None = None,
    exposure_columns: tuple[str, ...] = ("log_mv_total", "volatility_20", "ret_20d"),
    min_rows: int = 30,
) -> pd.Series:
    """Return daily target residuals after industry and numeric exposure controls."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    industry_values = industry.reindex(frame.index) if industry is not None else None
    for _, idx in frame.groupby("date", sort=False).groups.items():
        group = frame.loc[idx]
        y = pd.to_numeric(group[target], errors="coerce")
        parts: list[pd.DataFrame] = []
        numeric = [column for column in exposure_columns if column in group.columns]
        if numeric:
            parts.append(group[numeric].apply(pd.to_numeric, errors="coerce"))
        if industry_values is not None:
            categories = industry_values.loc[idx].astype("string").fillna("UNKNOWN")
            parts.append(pd.get_dummies(categories, prefix="industry", drop_first=True, dtype=float))
        if not parts:
            centered = y - y.mean()
            result.loc[idx] = centered
            continue
        x = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
        valid_columns = x.notna().any(axis=0)
        x = x.loc[:, valid_columns].fillna(0.0)
        x.insert(0, "const", 1.0)
        valid = y.notna()
        if int(valid.sum()) < max(int(min_rows), x.shape[1] + 3):
            result.loc[idx] = y - y.mean()
            continue
        matrix = x.to_numpy(dtype=float)
        beta = np.linalg.lstsq(matrix[valid], y.loc[valid].to_numpy(dtype=float), rcond=None)[0]
        result.loc[idx] = y.to_numpy(dtype=float) - matrix @ beta
    return result


def build_market_regimes(
    daily_market: pd.DataFrame,
    trend_window: int = 20,
    volatility_window: int = 20,
    history_window: int = 252,
) -> pd.DataFrame:
    """Build deterministic regimes from same-day and trailing market breadth data."""
    required = {"date", "median_return", "breadth"}
    missing = required - set(daily_market.columns)
    if missing:
        raise ValueError(f"market state columns missing: {sorted(missing)}")
    out = daily_market.copy().sort_values("date").drop_duplicates("date", keep="last")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["median_return"] = pd.to_numeric(out["median_return"], errors="coerce")
    out["breadth"] = pd.to_numeric(out["breadth"], errors="coerce")
    out["market_trend"] = out["median_return"].rolling(trend_window, min_periods=trend_window).sum()
    out["market_volatility"] = out["median_return"].rolling(
        volatility_window, min_periods=volatility_window
    ).std()
    threshold = out["market_volatility"].rolling(
        history_window, min_periods=max(volatility_window, history_window // 4)
    ).median()
    trend = np.where(
        (out["market_trend"] > 0) & (out["breadth"] >= 0.5), "up", "down_or_weak"
    )
    volatility = np.where(out["market_volatility"] > threshold, "high_vol", "normal_vol")
    out["regime"] = pd.Series(trend, index=out.index) + "__" + pd.Series(volatility, index=out.index)
    unavailable = out[["market_trend", "market_volatility"]].isna().any(axis=1) | threshold.isna()
    out.loc[unavailable, "regime"] = "insufficient_history"
    return out


def select_regime_weights(
    returns: pd.DataFrame,
    candidate_weights: list[float],
    min_observations: int = 20,
) -> dict[str, float]:
    """Select each regime's weight using only supplied selection-period returns."""
    required = {"date", "regime", "weight", "ret"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"regime return columns missing: {sorted(missing)}")
    selected: dict[str, float] = {}
    for regime, group in returns.groupby("regime", sort=True):
        rows = []
        for weight in candidate_weights:
            values = pd.to_numeric(group.loc[group["weight"] == weight, "ret"], errors="coerce").dropna()
            if len(values) < min_observations:
                continue
            std = float(values.std(ddof=1))
            sharpe = float(values.mean() / std * np.sqrt(252)) if std > 0 else float("-inf")
            rows.append((sharpe, -abs(float(weight)), float(weight)))
        if rows:
            selected[str(regime)] = max(rows)[2]
    return selected
