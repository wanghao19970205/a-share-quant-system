from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from intraday_1400 import config


_PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "amount", "vwap"}
_META_COLUMNS = {
    "code", "date", "asof_time", "cutoff_bar_time", "bar_count", "is_complete",
    "source", "period", "schema_version", "factor_status", "factor_version",
    "signal_last_bar_high", "signal_last_bar_low", "signal_last_bar_volume",
    "signal_return_1400", "signal_locked_up_1400", "signal_eligible",
}


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _return(values: np.ndarray, bars: int) -> float:
    if len(values) <= bars or values[-bars - 1] <= 0 or values[-1] <= 0:
        return float("nan")
    return float(math.log(values[-1] / values[-bars - 1]))


def _slope(values: np.ndarray, bars: int | None = None) -> float:
    sample = values[-bars:] if bars else values
    sample = sample[np.isfinite(sample) & (sample > 0)]
    if len(sample) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(sample), dtype=float), np.log(sample), 1)[0])


def _longest_streak(signs: np.ndarray, direction: int) -> int:
    best = current = 0
    for value in signs:
        if value == direction:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_drawdown(values: np.ndarray) -> float:
    if not len(values):
        return float("nan")
    peaks = np.maximum.accumulate(values)
    drawdowns = values / np.where(peaks == 0, np.nan, peaks) - 1.0
    return float(np.nanmin(drawdowns)) if np.isfinite(drawdowns).any() else float("nan")


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 3:
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def aggregate_symbol(raw: pd.DataFrame, code: str, cutoff_time: str = "13:55") -> pd.DataFrame:
    """Aggregate one symbol's 5-minute bars into causal 14:00 daily rows."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    timestamp_col = "date" if "date" in frame.columns else "kline_time"
    frame["timestamp"] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    cutoff = pd.Timestamp(f"2000-01-01 {cutoff_time}").time()
    entry_start = pd.Timestamp("2000-01-01 14:50").time()
    entry_end = pd.Timestamp("2000-01-01 14:50").time()
    numeric = ["open", "high", "low", "close", "volume", "amount", "bar_vwap_qfq"]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict] = []
    for trade_date, full_day in frame.groupby(frame["timestamp"].dt.normalize(), sort=True):
        full_day = full_day.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp")
        day = full_day[full_day["timestamp"].dt.time <= cutoff]
        if day.empty:
            continue
        close = day["close"].to_numpy(dtype=float)
        high = day["high"].to_numpy(dtype=float)
        low = day["low"].to_numpy(dtype=float)
        volume = day.get("volume", pd.Series(0.0, index=day.index)).fillna(0.0).to_numpy(dtype=float)
        amount = day.get("amount", pd.Series(0.0, index=day.index)).fillna(0.0).to_numpy(dtype=float)
        bar_vwap = day.get("bar_vwap_qfq", day["close"]).fillna(day["close"]).to_numpy(dtype=float)
        qfq_turnover_value = bar_vwap * volume
        returns = np.diff(np.log(np.where(close > 0, close, np.nan)))
        signs = np.sign(np.nan_to_num(returns, nan=0.0)).astype(int)
        path_length = float(np.nansum(np.abs(returns)))
        net_move = float(abs(math.log(close[-1] / close[0]))) if close[0] > 0 and close[-1] > 0 else float("nan")
        running_vwap = np.cumsum(qfq_turnover_value) / np.where(
            np.cumsum(volume) == 0, np.nan, np.cumsum(volume)
        )
        above_vwap = close > running_vwap
        crossings = int(np.count_nonzero(np.diff(above_vwap.astype(int)) != 0)) if len(close) > 1 else 0
        morning = day[day["timestamp"].dt.time < pd.Timestamp("2000-01-01 12:00").time()]
        afternoon = day[day["timestamp"].dt.time >= pd.Timestamp("2000-01-01 13:00").time()]
        high_pos = int(np.nanargmax(high))
        low_pos = int(np.nanargmin(low))
        realized_var = float(np.nansum(np.square(returns)))
        downside_var = float(np.nansum(np.square(returns[returns < 0])))
        upside_var = float(np.nansum(np.square(returns[returns > 0])))
        ret_std = float(np.nanstd(returns, ddof=1)) if len(returns) > 1 else float("nan")
        jump_count = int(np.count_nonzero(np.abs(returns) > 3.0 * ret_std)) if ret_std > 0 else 0
        signed_volume = float(np.nansum(np.sign(np.r_[0.0, returns]) * volume))
        volume_total = float(np.nansum(volume))
        volume_share = volume / volume_total if volume_total > 0 else np.full(len(volume), np.nan)
        drawdown_series = close / np.maximum.accumulate(close) - 1.0
        max_drawdown_index = int(np.nanargmin(drawdown_series)) if len(drawdown_series) else 0
        slope_30 = _slope(close, 6)
        slope_prev_30 = _slope(close[-12:-6]) if len(close) >= 12 else float("nan")
        slope_morning = _slope(morning["close"].to_numpy(dtype=float)) if len(morning) else float("nan")
        slope_afternoon = _slope(afternoon["close"].to_numpy(dtype=float)) if len(afternoon) else float("nan")
        entry = full_day[
            (full_day["timestamp"].dt.time >= entry_start)
            & (full_day["timestamp"].dt.time <= entry_end)
        ]
        entry_volume = float(entry.get("volume", pd.Series(dtype=float)).sum())
        pre_entry = full_day[
            (full_day["timestamp"].dt.time > cutoff)
            & (full_day["timestamp"].dt.time <= entry_end)
        ]
        through_entry = full_day[full_day["timestamp"].dt.time <= entry_end]
        entry_bar_vwap = entry.get("bar_vwap_qfq", entry.get("close", pd.Series(dtype=float)))
        entry_qfq_value = float(
            (pd.to_numeric(entry_bar_vwap, errors="coerce") * entry.get("volume", 0.0)).sum()
        )
        entry_price = (
            _safe_div(entry_qfq_value, entry_volume)
            if entry_volume > 0 and entry_qfq_value > 0
            else (float(entry["close"].iloc[-1]) if not entry.empty else float("nan"))
        )
        row = {
            "code": str(code)[:6],
            "date": pd.Timestamp(trade_date),
            "open": float(day["open"].iloc[0]),
            "high": float(np.nanmax(high)),
            "low": float(np.nanmin(low)),
            "close": float(close[-1]),
            "volume": float(np.nansum(volume)),
            "amount": float(np.nansum(amount)),
            "vwap": _safe_div(float(np.nansum(qfq_turnover_value)), float(np.nansum(volume))),
            "asof_time": "14:00",
            "cutoff_bar_time": day["timestamp"].iloc[-1].strftime("%H:%M"),
            "bar_count": int(len(day)),
            "is_complete": bool(len(day) >= 36 and day["timestamp"].iloc[-1].strftime("%H:%M") == cutoff_time),
            "signal_last_bar_high": float(high[-1]),
            "signal_last_bar_low": float(low[-1]),
            "signal_last_bar_volume": float(volume[-1]),
            "source": "AmazingData",
            "period": "min5",
            "factor_version": str(day.get("factor_version", pd.Series([""])).iloc[0]),
            "schema_version": config.SCHEMA_VERSION,
            "label_entry_price_1450": entry_price,
            "label_entry_high_1450": float(entry["high"].max()) if not entry.empty else float("nan"),
            "label_entry_low_1450": float(entry["low"].min()) if not entry.empty else float("nan"),
            "label_entry_volume_1450": entry_volume,
            "label_close_1455": float(full_day["close"].iloc[-1]),
            "label_high_to_1450": float(through_entry["high"].max()) if not through_entry.empty else float("nan"),
            "label_low_to_1450": float(through_entry["low"].min()) if not through_entry.empty else float("nan"),
            "label_ret_1400_1450": _safe_div(entry_price, float(close[-1])) - 1.0,
            "label_drawdown_1400_1450": (
                _safe_div(float(pre_entry["low"].min()), float(close[-1])) - 1.0
                if not pre_entry.empty else float("nan")
            ),
            "label_entry_bar_present": bool(not entry.empty),
            "m5_ret_15m": _return(close, 3),
            "m5_ret_30m": _return(close, 6),
            "m5_ret_60m": _return(close, 12),
            "m5_ret_open": float(math.log(close[-1] / day["open"].iloc[0])) if day["open"].iloc[0] > 0 and close[-1] > 0 else float("nan"),
            "m5_ret_morning": float(math.log(morning["close"].iloc[-1] / morning["open"].iloc[0])) if len(morning) else float("nan"),
            "m5_ret_afternoon": float(math.log(afternoon["close"].iloc[-1] / afternoon["open"].iloc[0])) if len(afternoon) else float("nan"),
            "m5_slope_all": _slope(close),
            "m5_slope_30m": slope_30,
            "m5_slope_60m": _slope(close, 12),
            "m5_slope_morning": slope_morning,
            "m5_slope_afternoon": slope_afternoon,
            "m5_accel_30m": slope_30 - slope_prev_30,
            "m5_accel_pm_vs_am": slope_afternoon - slope_morning,
            "m5_norm_speed_30m": _safe_div(_return(close, 6), ret_std * math.sqrt(6.0)) if ret_std > 0 else float("nan"),
            "m5_norm_speed_60m": _safe_div(_return(close, 12), ret_std * math.sqrt(12.0)) if ret_std > 0 else float("nan"),
            "m5_range": _safe_div(float(np.nanmax(high) - np.nanmin(low)), float(close[-1])),
            "m5_close_location": _safe_div(float(close[-1] - np.nanmin(low)), float(np.nanmax(high) - np.nanmin(low))),
            "m5_drawdown_from_high": _safe_div(float(close[-1]), float(np.nanmax(high))) - 1.0,
            "m5_rebound_from_low": _safe_div(float(close[-1]), float(np.nanmin(low))) - 1.0,
            "m5_high_bar_index": high_pos,
            "m5_low_bar_index": low_pos,
            "m5_high_after_low": float(high_pos > low_pos),
            "m5_bars_since_high": int(len(day) - 1 - high_pos),
            "m5_bars_since_low": int(len(day) - 1 - low_pos),
            "m5_new_high_30m": float(high_pos >= max(len(day) - 6, 0)),
            "m5_new_low_30m": float(low_pos >= max(len(day) - 6, 0)),
            "m5_up_bar_ratio": float(np.mean(signs > 0)) if len(signs) else 0.0,
            "m5_down_bar_ratio": float(np.mean(signs < 0)) if len(signs) else 0.0,
            "m5_longest_up_streak": _longest_streak(signs, 1),
            "m5_longest_down_streak": _longest_streak(signs, -1),
            "m5_max_bar_up": float(np.nanmax(returns)) if len(returns) else 0.0,
            "m5_max_bar_down": float(np.nanmin(returns)) if len(returns) else 0.0,
            "m5_path_length": path_length,
            "m5_trend_efficiency": _safe_div(net_move, path_length),
            "m5_return_autocorr_1": _corr(returns[:-1], returns[1:]),
            "m5_sign_persistence": float(np.mean(signs[:-1] == signs[1:])) if len(signs) >= 2 else float("nan"),
            "m5_volume_return_corr": _corr(np.log1p(volume[1:]), returns),
            "m5_volume_lead_return_corr": _corr(np.log1p(volume[:-1]), returns),
            "m5_return_lead_volume_corr": _corr(returns[:-1], np.log1p(volume[2:])),
            "m5_amihud": float(np.nanmean(np.abs(returns) / np.where(amount[1:] > 0, amount[1:], np.nan))) if len(returns) else float("nan"),
            "m5_volume_hhi": float(np.nansum(np.square(volume_share))),
            "m5_max_drawdown_bar_index": max_drawdown_index,
            "m5_bars_since_max_drawdown": int(len(close) - 1 - max_drawdown_index),
            "m5_volume_last_15_share": _safe_div(float(np.nansum(volume[-3:])), volume_total),
            "m5_volume_last_30_share": _safe_div(float(np.nansum(volume[-6:])), float(np.nansum(volume))),
            "m5_volume_last_60_share": _safe_div(float(np.nansum(volume[-12:])), float(np.nansum(volume))),
            "m5_volume_pm_am": _safe_div(float(afternoon.get("volume", pd.Series(dtype=float)).sum()), float(morning.get("volume", pd.Series(dtype=float)).sum())),
            "m5_price_vwap_gap": _safe_div(float(close[-1]), float(running_vwap[-1])) - 1.0,
            "m5_above_vwap_ratio": float(np.nanmean(above_vwap)),
            "m5_vwap_crossings": crossings,
            "m5_vwap_slope_30m": _slope(running_vwap, 6),
            "m5_vwap_slope_60m": _slope(running_vwap, 12),
            "m5_signed_volume_ratio": _safe_div(signed_volume, float(np.nansum(volume))),
            "m5_realized_vol": math.sqrt(max(realized_var, 0.0)),
            "m5_upside_semivar": upside_var,
            "m5_downside_semivar": downside_var,
            "m5_downside_var_ratio": _safe_div(downside_var, realized_var),
            "m5_max_drawdown": _max_drawdown(close),
            "m5_return_skew": float(pd.Series(returns).skew()) if len(returns) >= 3 else float("nan"),
            "m5_return_kurt": float(pd.Series(returns).kurt()) if len(returns) >= 4 else float("nan"),
            "m5_jump_count": jump_count,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_chunk(items: list[tuple[str, pd.DataFrame]], cutoff_time: str) -> pd.DataFrame:
    parts = [aggregate_symbol(frame, code, cutoff_time) for code, frame in items]
    valid = [part for part in parts if part is not None and not part.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def aggregate_many(items: list[tuple[str, pd.DataFrame]], workers: int = 1, cutoff_time: str = "13:55") -> pd.DataFrame:
    if workers <= 1 or len(items) <= 1:
        return _aggregate_chunk(items, cutoff_time)
    worker_count = min(max(int(workers), 1), len(items))
    chunk_size = math.ceil(len(items) / worker_count)
    item_chunks = [items[offset:offset + chunk_size] for offset in range(0, len(items), chunk_size)]
    # spawn prevents CPU workers from inheriting the collector's native SDK session. Chunking
    # avoids serializing and scheduling hundreds of tiny DataFrame futures per request.
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        parts = list(pool.map(_aggregate_chunk, item_chunks, [cutoff_time] * len(item_chunks)))
    valid = [part for part in parts if part is not None and not part.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def add_asof_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute existing daily technical factors on a pure 14:00 series."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy().sort_values(["code", "date"]).reset_index(drop=True)
    grouped = data.groupby("code", sort=False)
    data["ret_1d"] = grouped["close"].pct_change()
    trading_gap_days = grouped["date"].diff().dt.days
    near_limit_up = (data["ret_1d"] >= 0.045).astype(float)
    near_limit_down = (data["ret_1d"] <= -0.045).astype(float)
    flat_intraday = ((data["high"] - data["low"]).abs() <= 0.005).astype(float)
    data["risk_trading_gap_days"] = trading_gap_days
    data["risk_gap_event_count_20"] = (
        (trading_gap_days > 3).astype(float).groupby(data["code"])
        .transform(lambda values: values.rolling(20, min_periods=5).sum())
    )
    data["risk_near_limit_up_count_20"] = near_limit_up.groupby(data["code"]).transform(
        lambda values: values.rolling(20, min_periods=5).sum()
    )
    data["risk_near_limit_down_count_20"] = near_limit_down.groupby(data["code"]).transform(
        lambda values: values.rolling(20, min_periods=5).sum()
    )
    data["risk_flat_intraday_count_20"] = flat_intraday.groupby(data["code"]).transform(
        lambda values: values.rolling(20, min_periods=5).sum()
    )
    rolling_volume_median = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=5).median()
    )
    rolling_amount_median = grouped["amount"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    data["risk_volume_vs_median_20"] = data["volume"] / rolling_volume_median.replace(0, np.nan) - 1.0
    data["risk_amount_vs_median_20"] = data["amount"] / rolling_amount_median.replace(0, np.nan) - 1.0
    for days in (3, 5, 10, 20, 60):
        data[f"ret_{days}d"] = grouped["close"].pct_change(days)
        mean = grouped["close"].transform(lambda values: values.rolling(days).mean())
        data[f"ma_gap_{days}"] = data["close"] / mean - 1.0
    for days in (5, 10, 20):
        data[f"volatility_{days}"] = grouped["ret_1d"].transform(lambda values: values.rolling(days).std())
        mean_volume = grouped["volume"].transform(lambda values: values.rolling(days).mean())
        data[f"volume_ratio_{days}"] = data["volume"] / mean_volume - 1.0
    for days in (10, 20, 60):
        rolling_high = grouped["close"].transform(lambda values: values.rolling(days).max())
        rolling_low = grouped["close"].transform(lambda values: values.rolling(days).min())
        data[f"drawdown_{days}"] = data["close"] / rolling_high - 1.0
        data[f"range_pos_{days}"] = (data["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        rolling_vol = grouped["ret_1d"].transform(lambda values: values.rolling(days).std())
        data[f"ret_vol_adj_{days}"] = data[f"ret_{days}d"] / rolling_vol.replace(0, np.nan)
    from quant.factors import engineering

    for code, indices in data.groupby("code", sort=False).groups.items():
        close = data.loc[indices, "close"]
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        data.loc[indices, "macd_dif"] = dif.to_numpy()
        data.loc[indices, "macd_dea"] = dea.to_numpy()
        data.loc[indices, "macd_hist"] = (dif - dea).to_numpy()
        rule_input = data.loc[indices, ["close", "high", "low", "volume"]].copy()
        rule_input["turnover"] = np.nan
        data.loc[indices, "rule_score"] = engineering._technical_rule_score(rule_input).to_numpy()  # noqa: SLF001
    data["rule_score_chg_5"] = data.groupby("code")["rule_score"].diff(5)
    data["macd_hist_chg_3"] = data.groupby("code")["macd_hist"].diff(3)
    previous_close = data.groupby("code")["close"].shift(1)
    signal_return = data["close"] / previous_close.replace(0, np.nan) - 1.0
    last_high = data.get("signal_last_bar_high", pd.Series(np.nan, index=data.index))
    last_low = data.get("signal_last_bar_low", pd.Series(np.nan, index=data.index))
    last_bar_spread = (
        pd.to_numeric(last_high, errors="coerce")
        - pd.to_numeric(last_low, errors="coerce")
    ).abs()
    flat_bar_ratio = last_bar_spread / previous_close.replace(0, np.nan).abs()
    signal_locked = (flat_bar_ratio <= 0.0005) & (signal_return >= 0.045)
    data["signal_return_1400"] = signal_return
    data["signal_locked_up_1400"] = signal_locked.fillna(False)
    complete = data.get("is_complete", pd.Series(False, index=data.index))
    total_volume = data.get("volume", pd.Series(0.0, index=data.index))
    data["signal_eligible"] = (
        complete.fillna(False).astype(bool)
        & (pd.to_numeric(total_volume, errors="coerce").fillna(0.0) > 0)
        & previous_close.notna()
    )
    true_range = pd.concat([
        (data["high"] - data["low"]).abs(),
        (data["high"] - previous_close).abs(),
        (data["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data["atr14"] = true_range.groupby(data["code"]).transform(
        lambda values: values.rolling(14, min_periods=10).mean()
    )
    data["intraday_range"] = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
    data["amount_chg_5"] = data.groupby("code")["amount"].pct_change(5)
    prior_close = data.groupby("code")["close"].shift(1)
    data["ovn_gap"] = data["open"] / prior_close - 1.0
    data["ovn_gap_5d"] = data.groupby("code")["ovn_gap"].transform(lambda values: values.rolling(5, min_periods=2).mean())
    data["ovn_gap_vol_20d"] = data.groupby("code")["ovn_gap"].transform(lambda values: values.rolling(20, min_periods=5).std())
    median_volume = data.groupby("code")["volume"].transform(lambda values: values.shift(1).rolling(20, min_periods=5).median())
    median_amount = data.groupby("code")["amount"].transform(lambda values: values.shift(1).rolling(20, min_periods=5).median())
    data["m5_volume_vs_20d_median"] = data["volume"] / median_volume - 1.0
    data["m5_amount_vs_20d_median"] = data["amount"] / median_amount - 1.0
    return data


def feature_columns(frame: pd.DataFrame) -> list[str]:
    banned = _PRICE_COLUMNS | _META_COLUMNS
    return [
        column for column in frame.columns
        if column not in banned
        and not column.startswith("target_")
        and not column.startswith("entry_")
        and not column.startswith("exit_")
        and not column.startswith("label_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
