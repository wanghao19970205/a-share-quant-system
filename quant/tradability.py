"""可交易口径的单一真源：封板判定、跌停顺延卖出、可实现收益、可买入掩码。

此前这套逻辑内嵌在 ``quant/watchlist_grid.py``（回测选参路径）。为让「训练期 join」
与「回测选参」共用同一份实现、杜绝口径分裂，抽到本模块。``watchlist_grid`` 与
``full_train_batched`` 均从这里取。

口径：
- 默认模式保留历史主板 ±10% 启发式，避免改变现有生产行为。
- ``require_status=True`` 使用逐日 PIT 涨跌停价及停牌/退市整理状态；状态缺失时严格不可交易。
- ``tradable_ret_{h}d``：尾盘 T 买入、T+h 收盘卖出；若卖出日封跌停或不可交易则顺延，
  上限 cap 个交易日（含预定日）；仅封跌停时保留末个可交易日强制平仓策略。
- ``buyable_close``：当日是否可买入（尾盘收盘口径）——涨停封板、零成交量或状态不可用时不可买。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config


def sell_roll_max_days() -> int:
    """跌停顺延卖出的上限交易日数（含预定卖出日）。单一真源在 backtest.bt_sell_roll_max_days()。"""
    return backtest.bt_sell_roll_max_days()


def rolled_sell_close(
    close: np.ndarray,
    sell_blocked: np.ndarray,
    horizon: int,
    cap: int,
    sell_unavailable: np.ndarray | None = None,
) -> np.ndarray:
    """Return the first executable close in the bounded sell window.

    Limit-down rows may still use the last available close as the existing conservative
    cap policy. Rows with no trading volume are never executable; an all-unavailable
    window remains NaN instead of fabricating a fill at a stale close.
    """
    n = close.shape[0]
    unavailable = (
        np.zeros(n, dtype=bool)
        if sell_unavailable is None
        else np.asarray(sell_unavailable, dtype=bool)
    )
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        base = i + horizon
        if base >= n:
            continue  # 未来数据不足，丢尾
        sell_idx = None
        last = min(base + cap - 1, n - 1)
        available_indices = []
        for j in range(base, last + 1):
            if unavailable[j]:
                continue
            available_indices.append(j)
            if not bool(sell_blocked[j]):
                sell_idx = j
                break
        if sell_idx is None and available_indices:
            sell_idx = available_indices[-1]
        if sell_idx is not None:
            out[i] = close[sell_idx]
    return out


def _quant_dir() -> Path:
    return Path(config.QUANT_DIR)


def _align_price_to_trading_calendar(
    px: pd.DataFrame,
    base_dir: Path,
    code: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    calendar_path = base_dir / "trading_calendar.parquet"
    if not calendar_path.exists():
        raise RuntimeError(f"authoritative trading calendar unavailable: {calendar_path}")
    calendar = pd.read_parquet(calendar_path)
    if calendar.empty or list(calendar.columns) != ["date"]:
        raise ValueError("authoritative trading calendar must contain only date")
    dates = pd.to_datetime(calendar["date"], errors="coerce").astype("datetime64[ns]")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("authoritative trading calendar must be unique and increasing")
    observed = pd.DatetimeIndex(px["date"].dropna().unique()).sort_values()
    if observed.empty or not observed.isin(dates).all():
        raise RuntimeError(f"price dates absent from authoritative trading calendar: {code}")
    sessions = dates[(dates >= observed.min()) & (dates <= observed.max())]
    aligned = pd.DataFrame({"date": sessions}).merge(
        px.drop(columns=["code"]), on="date", how="left", validate="one_to_one",
    )
    aligned.insert(0, "code", str(code).zfill(6))
    return aligned, observed


def price_tradability(
    codes: list[str],
    horizons: list[int],
    quant_dir: Path | None = None,
    require_status: bool = False,
    require_calendar: bool = False,
    min_adv20: float | None = None,
    min_listing_sessions: int | None = None,
) -> pd.DataFrame:
    """从 price/{code}.parquet 读 OHLC，返回按 code+date 的可交易口径列。

    产出列（按 horizon 展开）：``target_ret_{h}d``（收盘→收盘乐观口径，供对照/兜底）、
    ``open_ret_{h}d``（次日开盘买入口径，若有 open）、``tradable_ret_{h}d``（跌停顺延实现收益）、
    以及 ``buyable_close`` / ``buyable_next``（可买入掩码）。

    ``code`` 统一 zfill(6)、``date`` 统一 datetime64[ns]，与训练面板 join 键一致。
    """
    base_dir = quant_dir or _quant_dir()
    status_groups: dict[str, pd.DataFrame] = {}
    if require_status:
        status_path = base_dir / "trading_status_history.parquet"
        if not status_path.exists():
            raise RuntimeError(f"PIT trading status unavailable: {status_path}")
        status = pd.read_parquet(status_path)
        required = {
            "code", "date", "high_limit", "low_limit", "is_st",
            "is_suspended", "is_withdrawal", "is_ex_right",
        }
        if status.empty or not required.issubset(status.columns):
            raise ValueError(f"PIT trading status missing columns: {sorted(required - set(status.columns))}")
        status = status.copy()
        status["code"] = status["code"].astype(str).str.zfill(6)
        status["date"] = pd.to_datetime(status["date"], errors="coerce")
        if status[["code", "date"]].isna().any().any() or status.duplicated(["code", "date"]).any():
            raise ValueError("PIT trading status has invalid or duplicate keys")
        wanted = {str(code).zfill(6) for code in codes}
        status = status[status["code"].isin(wanted)]
        status_groups = {
            code: group.drop(columns=["code"]).sort_values("date")
            for code, group in status.groupby("code", sort=False)
        }
    listing_calendar = None
    listing_dates: dict[str, pd.Timestamp] = {}
    if min_listing_sessions is not None:
        if int(min_listing_sessions) < 0:
            raise ValueError("min_listing_sessions must be non-negative")
        calendar_path = base_dir / "trading_calendar.parquet"
        master_path = base_dir / "security_master.parquet"
        if not calendar_path.exists() or not master_path.exists():
            raise RuntimeError("listing-session gate requires calendar and security master")
        calendar = pd.read_parquet(calendar_path)
        master = pd.read_parquet(master_path)
        if calendar.empty or list(calendar.columns) != ["date"]:
            raise ValueError("authoritative trading calendar must contain only date")
        if not {"code", "list_date"}.issubset(master.columns):
            raise ValueError("security master missing code/list_date")
        dates = pd.to_datetime(calendar["date"], errors="coerce").astype("datetime64[ns]")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("authoritative trading calendar must be unique and increasing")
        listing_calendar = pd.DatetimeIndex(dates)
        master = master.copy()
        master["code"] = master["code"].astype(str).str.zfill(6)
        master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce")
        if master["code"].duplicated().any() or master["list_date"].isna().any():
            raise ValueError("security master has invalid listing keys")
        listing_dates = dict(zip(master["code"], master["list_date"]))
    frames: list[pd.DataFrame] = []
    for code in codes:
        path = base_dir / "price" / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            px = pd.read_parquet(
                path,
                columns=["code", "date", "open", "high", "low", "close", "volume", "amount"],
            )
        except Exception:  # noqa: BLE001
            try:
                px = pd.read_parquet(
                    path, columns=["code", "date", "open", "high", "low", "close", "volume"]
                )
            except Exception:  # noqa: BLE001
                # 老价格文件可能没有 open/high/low，退回仅 close
                try:
                    px = pd.read_parquet(path, columns=["code", "date", "close"])
                except Exception:  # noqa: BLE001
                    continue
        if px.empty:
            continue
        if require_calendar and "volume" not in px.columns:
            raise ValueError(f"strict tradability requires volume for {str(code).zfill(6)}")
        if min_adv20 is not None and "amount" not in px.columns:
            raise ValueError(f"ADV20 gate requires amount for {str(code).zfill(6)}")
        px = px.copy()
        px["code"] = px["code"].astype(str).str.zfill(6)
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c in px.columns:
                px[c] = pd.to_numeric(px[c], errors="coerce")
        px = px.dropna(subset=["code", "date", "close"]).sort_values("date")
        if px.empty:
            continue
        observed_dates = None
        if require_calendar:
            px, observed_dates = _align_price_to_trading_calendar(
                px, base_dir, str(code).zfill(6),
            )
        if require_status:
            code_status = status_groups.get(str(code).zfill(6))
            if code_status is None:
                for column in (
                    "high_limit", "low_limit", "is_st", "is_suspended",
                    "is_withdrawal", "is_ex_right",
                ):
                    px[column] = np.nan
                px["status_present"] = False
            else:
                code_status = code_status.copy()
                code_status["status_present"] = True
                px = px.merge(code_status, on="date", how="left", validate="one_to_one")
                px["status_present"] = px["status_present"].eq(True)
        out = px[["code", "date"]].copy()
        has_open = "open" in px.columns and px["open"].notna().any()
        has_hl = "high" in px.columns and "low" in px.columns
        has_volume = "volume" in px.columns
        positive_volume = (
            px["volume"].fillna(0.0).gt(0)
            if has_volume
            else pd.Series(False, index=px.index)
        )
        out["positive_volume"] = positive_volume.to_numpy(dtype=bool)
        listing_ok = pd.Series(True, index=px.index)
        if min_listing_sessions is not None:
            list_date = listing_dates.get(str(code).zfill(6))
            if list_date is None:
                listing_sessions = pd.Series(np.nan, index=px.index)
            else:
                start_pos = listing_calendar.searchsorted(list_date, side="left")
                positions = listing_calendar.searchsorted(px["date"], side="right")
                listing_sessions = pd.Series(positions - start_pos, index=px.index, dtype=float)
                listing_sessions = listing_sessions.where(px["date"] >= list_date)
            listing_ok = listing_sessions.ge(int(min_listing_sessions))
            out["listing_sessions"] = listing_sessions.to_numpy(dtype=float)
            out["listing_pass"] = listing_ok.to_numpy(dtype=bool)
        liquidity_ok = pd.Series(True, index=px.index)
        if min_adv20 is not None:
            if float(min_adv20) < 0:
                raise ValueError("min_adv20 must be non-negative")
            amount = (
                px["amount"] if "amount" in px.columns
                else pd.Series(np.nan, index=px.index)
            )
            adv20 = amount.shift(1).rolling(20, min_periods=20).mean()
            liquidity_ok = adv20.ge(float(min_adv20))
            out["adv20"] = adv20.to_numpy(dtype=float)
            out["liquidity_pass"] = liquidity_ok.to_numpy(dtype=bool)
        close_arr = px["close"].to_numpy(dtype=float)
        status_present = np.ones(len(px), dtype=bool)
        status_unavailable = np.zeros(len(px), dtype=bool)
        if require_status:
            status_present = px["status_present"].to_numpy(dtype=bool)
            is_st = px["is_st"].eq(True).to_numpy(dtype=bool)
            is_suspended = px["is_suspended"].eq(True).to_numpy(dtype=bool)
            is_withdrawal = px["is_withdrawal"].eq(True).to_numpy(dtype=bool)
            is_ex_right = px["is_ex_right"].eq(True).to_numpy(dtype=bool)
            status_unavailable = ~status_present | is_suspended | is_withdrawal
            out["status_present"] = status_present
            out["is_st"] = is_st
            out["is_suspended"] = is_suspended
            out["is_withdrawal"] = is_withdrawal
            out["is_ex_right"] = is_ex_right
        limit_down_seal = None
        if has_hl:
            high_arr = px["high"].to_numpy(dtype=float)
            low_arr = px["low"].to_numpy(dtype=float)
            if require_status:
                high_limit = pd.to_numeric(px["high_limit"], errors="coerce").to_numpy(dtype=float)
                low_limit = pd.to_numeric(px["low_limit"], errors="coerce").to_numpy(dtype=float)
                limit_up_seal = (
                    status_present & np.isclose(close_arr, high_arr, rtol=0.0, atol=1e-8)
                    & np.isclose(close_arr, high_limit, rtol=0.0, atol=1e-6)
                )
                limit_down_seal = (
                    status_present & np.isclose(close_arr, low_arr, rtol=0.0, atol=1e-8)
                    & np.isclose(close_arr, low_limit, rtol=0.0, atol=1e-6)
                )
            else:
                prev_close = px["close"].shift(1).to_numpy(dtype=float)
                with np.errstate(all="ignore"):
                    ret1 = close_arr / prev_close - 1
                limit_up_seal = (close_arr == high_arr) & (ret1 >= 0.095)
                limit_down_seal = (close_arr == low_arr) & (ret1 <= -0.095)
        cap = sell_roll_max_days()
        for horizon in horizons:
            # 收盘口径（保留，供方向统计/兜底）
            out[f"target_ret_{horizon}d"] = px["close"].shift(-horizon) / px["close"] - 1
            # 次日开盘买入、持有 horizon 日后开盘卖出（更贴近实盘）
            if has_open:
                entry = px["open"].shift(-1)
                exit_ = px["open"].shift(-(1 + horizon))
                out[f"open_ret_{horizon}d"] = exit_ / entry - 1
            # 尾盘 T 买入、T+horizon 收盘卖出；若卖出日一字/收盘跌停封板，顺延到下一可卖日收盘，
            # 上限 cap 个交易日，仍封则第 cap 日强制平仓。收益 = 实际卖出日收盘 / 买入日收盘 - 1。
            if limit_down_seal is not None:
                sell_close = rolled_sell_close(
                    close_arr,
                    limit_down_seal,
                    horizon,
                    cap,
                    sell_unavailable=(
                        ~positive_volume.to_numpy(dtype=bool) | status_unavailable
                    ),
                )
                with np.errstate(all="ignore"):
                    out[f"tradable_ret_{horizon}d"] = sell_close / close_arr - 1
        current_status_ok = pd.Series(~status_unavailable, index=px.index)
        next_status_ok = current_status_ok.shift(-1, fill_value=False)
        # 次日是否可买入(次日开盘口径)：一字涨停(high==low 且上涨)当日买不进
        if has_open and has_hl:
            nxt_high = px["high"].shift(-1)
            nxt_low = px["low"].shift(-1)
            nxt_close = px["close"].shift(-1)
            entry = px["open"].shift(-1)
            one_word_up = (nxt_high == nxt_low) & (nxt_close > px["close"])
            out["buyable_next"] = (
                (~one_word_up.fillna(False))
                & entry.notna()
                & positive_volume.shift(-1, fill_value=False)
                & next_status_ok
                & liquidity_ok.shift(-1, fill_value=False)
                & listing_ok.shift(-1, fill_value=False)
            )
        else:
            # 缺少 open/high/low 时无法证明次日可买，严格口径默认不可交易。
            out["buyable_next"] = False
        # 当日是否可买入(尾盘收盘口径)：涨停封板(含一字涨停)当天尾盘买不进
        if limit_down_seal is not None:
            out["buyable_close"] = (
                (~limit_up_seal)
                & positive_volume.to_numpy(dtype=bool)
                & current_status_ok.to_numpy(dtype=bool)
                & liquidity_ok.to_numpy(dtype=bool)
                & listing_ok.to_numpy(dtype=bool)
            )
        else:
            # 缺少高低价无法判定封板，严格口径不得默认可买。
            out["buyable_close"] = False
        if observed_dates is not None:
            out = out[out["date"].isin(observed_dates)].copy()
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
