from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    top_n: int = 10
    entry_time: str = "14:50"
    time_exit_signal: str = "14:45"
    roundtrip_cost: float = 0.002
    unsellable_return: float = -0.10
    stop_loss: float = 0.05
    take_profit: float = 0.09
    trailing_arm: float = 0.03
    trailing_drawdown: float = 0.02
    max_exit_sessions: int = 3
    flat_bar_tolerance: float = 0.0005
    limit_return_threshold: float = 0.045
    select_from_buyable_only: bool = False

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.max_exit_sessions <= 0:
            raise ValueError("max_exit_sessions must be positive")
        if self.roundtrip_cost < 0:
            raise ValueError("roundtrip_cost must be nonnegative")
        for value in (
            self.stop_loss,
            self.take_profit,
            self.trailing_arm,
            self.trailing_drawdown,
        ):
            if value < 0:
                raise ValueError("exit thresholds must be nonnegative")


_EXECUTION_COLUMNS = [
    "model",
    "signal_date",
    "code",
    "rank",
    "score",
    "entry_timestamp",
    "entry_price",
    "entry_buyable",
    "entry_reason",
    "exit_timestamp",
    "exit_price",
    "exit_sellable",
    "exit_reason",
    "outcome_observed",
    "gross_return",
    "cost",
    "net_return",
]


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    score_column = next(
        (column for column in ("score", "pred", "ensemble_pred") if column in frame),
        None,
    )
    if score_column is None or not {"code", "date"}.issubset(frame.columns):
        raise ValueError("predictions require code, date, and score/pred/ensemble_pred")
    data = frame[["code", "date", score_column]].rename(columns={score_column: "score"}).copy()
    data["code"] = data["code"].astype(str).str[:6]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    return (
        data.dropna(subset=["code", "date", "score"])
        .drop_duplicates(["code", "date"], keep="last")
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )


def common_prediction_universe(predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not predictions:
        raise ValueError("at least one prediction source is required")
    normalized = {name: normalize_predictions(frame) for name, frame in predictions.items()}
    keys: pd.DataFrame | None = None
    for frame in normalized.values():
        source_keys = frame[["code", "date"]]
        keys = source_keys if keys is None else keys.merge(
            source_keys, on=["code", "date"], how="inner", validate="one_to_one"
        )
    assert keys is not None
    keys = keys.drop_duplicates(["code", "date"])
    return {
        name: frame.merge(keys, on=["code", "date"], how="inner", validate="one_to_one")
        for name, frame in normalized.items()
    }


def select_daily_top(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = normalize_predictions(predictions)
    selected["rank"] = selected.groupby("date")["score"].rank(
        method="first", ascending=False
    ).astype(int)
    return selected[selected["rank"] <= int(top_n)].reset_index(drop=True)


def _metrics(daily: pd.Series) -> dict:
    clean = pd.to_numeric(daily, errors="coerce").dropna().sort_index()
    if clean.empty:
        return {"days": 0}
    std = float(clean.std())
    curve = (1.0 + clean).cumprod()
    return {
        "days": int(len(clean)),
        "mean_return": float(clean.mean()),
        "compound_return": float(curve.iloc[-1] - 1.0),
        "win_rate": float((clean > 0).mean()),
        "sharpe": float(clean.mean() / std * np.sqrt(252)) if std > 0 else None,
        "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
    }


def _paired_bootstrap(values: pd.Series, samples: int = 2000, seed: int = 42) -> list[float] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 10:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare_execution_records(records: pd.DataFrame) -> dict:
    required = {"model", "signal_date", "code", "net_return"}
    if not required.issubset(records.columns):
        raise ValueError(f"execution records missing {sorted(required - set(records.columns))}")
    data = records.copy()
    data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce").dt.normalize()
    grouped = data.groupby(["signal_date", "model"], sort=True)
    daily = grouped["net_return"].mean().where(
        grouped["outcome_observed"].all()
    ).unstack("model")
    models = daily.columns.tolist()
    report = {
        "models": {
            model: {
                **_metrics(daily[model]),
                "mean_names": float(
                    data[data["model"] == model].groupby("signal_date")["code"].nunique().mean()
                ),
                "mean_filled_names": float(
                    data[
                        (data["model"] == model)
                        & data["entry_buyable"].fillna(False)
                    ].groupby("signal_date")["code"].nunique().reindex(
                        daily.index, fill_value=0
                    ).mean()
                ),
                "unbuyable": int(
                    (~data.loc[data["model"] == model, "entry_buyable"].fillna(False)).sum()
                ),
                "unsellable": int((
                    data.loc[data["model"] == model, "entry_buyable"].fillna(False)
                    & data.loc[data["model"] == model, "outcome_observed"].fillna(False)
                    & ~data.loc[data["model"] == model, "exit_sellable"].fillna(False)
                ).sum()),
            }
            for model in models
        },
        "daily_returns": daily.reset_index(),
    }
    pairwise = {}
    for left, right in combinations(models, 2):
        paired = daily[[left, right]].dropna()
        delta = paired[right] - paired[left]
        pairwise[f"{right}_minus_{left}"] = {
            "left": left,
            "right": right,
            "days": int(len(paired)),
            "right_minus_left_mean": float(delta.mean()) if len(delta) else None,
            "right_minus_left_ci95": _paired_bootstrap(delta),
            "right_wins": int((paired[right] > paired[left]).sum()),
            "left_wins": int((paired[left] > paired[right]).sum()),
            "ties": int((paired[left] == paired[right]).sum()),
        }
    report["pairwise"] = pairwise
    if len(models) == 2:
        report["paired"] = next(iter(pairwise.values()))
    return report


def simulate_fixed_exit_race(
    predictions: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    config: ExecutionConfig = ExecutionConfig(),
) -> tuple[pd.DataFrame, dict]:
    required = {"code", "date", "entry_buyable", "target_net_ret_t1"}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels missing {sorted(required - set(labels.columns))}")
    common = common_prediction_universe(predictions)
    outcomes = labels.copy()
    outcomes["code"] = outcomes["code"].astype(str).str[:6]
    outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce").dt.normalize()
    outcomes = outcomes.drop_duplicates(["code", "date"], keep="last")
    outcome_keys = outcomes[["code", "date"]]
    if config.select_from_buyable_only:
        # Buyability is observable at the 14:50 order time, so restricting the
        # ranking pool is point-in-time legal. Without it an unbuyable name
        # consumes a slot and contributes a zero return, which makes the
        # portfolio metrics a function of the fill rate instead of the signal.
        outcome_keys = outcomes.loc[
            outcomes["entry_buyable"].fillna(False).astype(bool), ["code", "date"]
        ]
    common = {
        name: frame.merge(
            outcome_keys, on=["code", "date"], how="inner", validate="one_to_one"
        )
        for name, frame in common.items()
    }
    records = []
    for model_name, frame in common.items():
        selected = select_daily_top(frame, config.top_n)
        selected = selected.merge(outcomes, on=["code", "date"], how="left", validate="one_to_one")
        observed = selected.get(
            "target_outcome_observed_t1",
            selected["target_net_ret_t1"].notna(),
        ).fillna(False).astype(bool)
        buyable = selected["entry_buyable"].fillna(False).astype(bool)
        sellable = selected["target_net_ret_t1"].notna()
        selected["model"] = model_name
        selected["signal_date"] = selected["date"]
        selected["entry_timestamp"] = pd.NaT
        selected["entry_price"] = pd.to_numeric(
            selected.get("label_entry_price_1450"), errors="coerce"
        )
        selected["entry_buyable"] = buyable
        selected["entry_reason"] = np.where(buyable, "filled_1450", "unbuyable_1450")
        selected["exit_timestamp"] = pd.NaT
        selected["exit_price"] = np.nan
        selected["exit_sellable"] = sellable
        selected["exit_reason"] = np.select(
            [~buyable, sellable],
            ["not_entered", "fixed_t1_1450"],
            default="unsellable_t1",
        )
        selected["outcome_observed"] = observed | ~buyable
        selected["cost"] = np.where(buyable, float(config.roundtrip_cost), 0.0)
        selected["net_return"] = pd.to_numeric(selected["target_net_ret_t1"], errors="coerce")
        penalty = buyable & observed & selected["net_return"].isna()
        selected.loc[penalty, "net_return"] = float(config.unsellable_return)
        selected.loc[~buyable, "net_return"] = 0.0
        selected.loc[buyable & ~observed, "net_return"] = np.nan
        selected["gross_return"] = selected["net_return"] + selected["cost"]
        records.append(selected[_EXECUTION_COLUMNS])
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=_EXECUTION_COLUMNS)
    return result, compare_execution_records(result)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = next(
        (column for column in ("timestamp", "date", "kline_time") if column in bars),
        None,
    )
    if timestamp_column is None or "code" not in bars:
        raise ValueError("bars require code and timestamp/date/kline_time")
    data = bars.copy()
    data["code"] = data["code"].astype(str).str[:6]
    data["timestamp"] = pd.to_datetime(data[timestamp_column], errors="coerce")
    data["session"] = data["timestamp"].dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount", "bar_vwap_qfq", "vwap"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return (
        data.dropna(subset=["code", "timestamp", "open", "high", "low", "close"])
        .sort_values(["timestamp", "code"])
        .drop_duplicates(["code", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _bar_price(row: pd.Series) -> float:
    for column in ("bar_vwap_qfq", "vwap"):
        value = row.get(column)
        if pd.notna(value) and float(value) > 0:
            return float(value)
    volume = float(row.get("volume") or 0.0)
    amount = float(row.get("amount") or 0.0)
    if volume > 0 and amount > 0:
        candidate = amount / volume
        close = float(row["close"])
        if 0.5 * close <= candidate <= 2.0 * close:
            return candidate
    return float(row["close"])


def _previous_close(bars: pd.DataFrame, session: pd.Timestamp) -> float | None:
    previous = bars[bars["session"] < session]
    if previous.empty:
        return None
    value = float(previous.iloc[-1]["close"])
    return value if np.isfinite(value) and value > 0 else None


def _locked_bar(row: pd.Series, previous_close: float | None, direction: str, config: ExecutionConfig) -> bool:
    if float(row.get("volume") or 0.0) <= 0:
        return True
    spread = abs(float(row["high"]) - float(row["low"]))
    if not previous_close:
        return False
    if spread / abs(float(previous_close)) > config.flat_bar_tolerance:
        return False
    bar_return = float(row["close"]) / previous_close - 1.0
    if direction == "buy":
        return bar_return >= config.limit_return_threshold
    return bar_return <= -config.limit_return_threshold


def _parse_clock(value: str) -> time:
    return pd.Timestamp(f"2000-01-01 {value}").time()


def _first_sellable_after(
    bars: pd.DataFrame,
    start_timestamp: pd.Timestamp,
    sessions: list[pd.Timestamp],
    signal_session_index: int,
    config: ExecutionConfig,
) -> tuple[pd.Series | None, bool]:
    max_session_index = min(signal_session_index + config.max_exit_sessions - 1, len(sessions) - 1)
    candidates = bars[bars["timestamp"] > start_timestamp]
    for _, row in candidates.iterrows():
        session_index = sessions.index(pd.Timestamp(row["session"]))
        if session_index > max_session_index:
            break
        previous_close = _previous_close(bars, pd.Timestamp(row["session"]))
        if not _locked_bar(row, previous_close, "sell", config):
            return row, True
    return None, False


def _adaptive_exit(
    code_bars: pd.DataFrame,
    entry_price: float,
    entry_session: pd.Timestamp,
    sessions: list[pd.Timestamp],
    config: ExecutionConfig,
) -> dict:
    try:
        entry_session_index = sessions.index(entry_session)
    except ValueError:
        return {"outcome_observed": False}
    if entry_session_index + 1 >= len(sessions):
        return {"outcome_observed": False}
    exit_session_index = entry_session_index + 1
    exit_session = sessions[exit_session_index]
    day = code_bars[code_bars["session"] == exit_session].reset_index(drop=True)
    if day.empty:
        session_end = exit_session + pd.Timedelta(hours=15)
        execution, sellable = _first_sellable_after(
            code_bars,
            session_end,
            sessions,
            exit_session_index,
            config,
        )
        if not sellable or execution is None:
            return {
                "outcome_observed": True,
                "exit_sellable": False,
                "exit_reason": "suspended_t1_blocked",
            }
        exit_price = _bar_price(execution)
        gross_return = exit_price / entry_price - 1.0
        return {
            "outcome_observed": True,
            "exit_sellable": True,
            "exit_reason": "suspended_t1_roll",
            "exit_timestamp": execution["timestamp"],
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": gross_return - config.roundtrip_cost,
        }
    signal_cutoff = _parse_clock(config.time_exit_signal)
    peak = entry_price
    signal_reason = "time_cap"
    signal_index = None
    for index, row in day.iterrows():
        if row["timestamp"].time() > signal_cutoff:
            break
        close = float(row["close"])
        peak = max(peak, float(row["high"]))
        net_move = close / entry_price - 1.0
        if config.stop_loss > 0 and net_move <= -config.stop_loss:
            signal_reason = "stop_loss"
            signal_index = index
            break
        if config.take_profit > 0 and net_move >= config.take_profit:
            signal_reason = "take_profit"
            signal_index = index
            break
        if (
            config.trailing_arm > 0
            and peak / entry_price - 1.0 >= config.trailing_arm
            and config.trailing_drawdown > 0
            and close / peak - 1.0 <= -config.trailing_drawdown
        ):
            signal_reason = "trailing_stop"
            signal_index = index
            break
        signal_index = index
    if signal_index is None:
        return {
            "outcome_observed": True,
            "exit_sellable": False,
            "exit_reason": "missing_exit_signal_bar",
        }
    execution, sellable = _first_sellable_after(
        code_bars,
        pd.Timestamp(day.iloc[signal_index]["timestamp"]),
        sessions,
        exit_session_index,
        config,
    )
    if not sellable or execution is None:
        return {
            "outcome_observed": True,
            "exit_sellable": False,
            "exit_reason": f"{signal_reason}_blocked",
        }
    exit_price = _bar_price(execution)
    gross_return = exit_price / entry_price - 1.0
    return {
        "outcome_observed": True,
        "exit_sellable": True,
        "exit_reason": signal_reason,
        "exit_timestamp": execution["timestamp"],
        "exit_price": exit_price,
        "gross_return": gross_return,
        "net_return": gross_return - config.roundtrip_cost,
    }


def simulate_adaptive_exit_race(
    predictions: dict[str, pd.DataFrame],
    bars: pd.DataFrame,
    config: ExecutionConfig = ExecutionConfig(),
) -> tuple[pd.DataFrame, dict]:
    common = common_prediction_universe(predictions)
    market = _normalize_bars(bars)
    sessions = sorted(pd.Timestamp(value) for value in market["session"].dropna().unique())
    by_code = {
        code: frame.sort_values("timestamp").reset_index(drop=True)
        for code, frame in market.groupby("code", sort=False)
    }
    entry_clock = _parse_clock(config.entry_time)
    records = []
    for model_name, frame in common.items():
        selected = select_daily_top(frame, config.top_n)
        for row in selected.itertuples(index=False):
            signal_date = pd.Timestamp(row.date)
            code_bars = by_code.get(row.code, market.iloc[0:0])
            entry = code_bars[
                (code_bars["session"] == signal_date)
                & (code_bars["timestamp"].dt.time == entry_clock)
            ]
            record = {
                "model": model_name,
                "signal_date": signal_date,
                "code": row.code,
                "rank": int(row.rank),
                "score": float(row.score),
                "entry_timestamp": pd.NaT,
                "entry_price": np.nan,
                "entry_buyable": False,
                "entry_reason": "missing_entry_bar",
                "exit_timestamp": pd.NaT,
                "exit_price": np.nan,
                "exit_sellable": False,
                "exit_reason": "not_entered",
                "outcome_observed": True,
                "gross_return": 0.0,
                "cost": 0.0,
                "net_return": 0.0,
            }
            if not entry.empty:
                entry_row = entry.iloc[0]
                previous_close = _previous_close(code_bars, signal_date)
                if _locked_bar(entry_row, previous_close, "buy", config):
                    record["entry_reason"] = "locked_or_no_volume_1450"
                else:
                    entry_price = _bar_price(entry_row)
                    record.update({
                        "entry_timestamp": entry_row["timestamp"],
                        "entry_price": entry_price,
                        "entry_buyable": True,
                        "entry_reason": "filled_1450",
                        "cost": float(config.roundtrip_cost),
                    })
                    record.update(_adaptive_exit(
                        code_bars,
                        entry_price,
                        signal_date,
                        sessions,
                        config,
                    ))
                    if record["outcome_observed"] and not record["exit_sellable"]:
                        record["net_return"] = float(config.unsellable_return)
                        record["gross_return"] = record["net_return"] + config.roundtrip_cost
            records.append(record)
    result = pd.DataFrame(records, columns=_EXECUTION_COLUMNS)
    return result, {
        "config": asdict(config),
        **compare_execution_records(result),
    }


def required_bar_sessions(signal_dates: Iterable[pd.Timestamp], max_exit_sessions: int = 3) -> int:
    dates = pd.to_datetime(list(signal_dates), errors="coerce")
    if pd.isna(dates).all():
        return 0
    return int(pd.Series(dates).dropna().dt.normalize().nunique()) + max(int(max_exit_sessions), 1)
