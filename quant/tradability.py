"""可交易口径的单一真源：封板判定、跌停顺延卖出、可实现收益、可买入掩码。

此前这套逻辑内嵌在 ``quant/watchlist_grid.py``（回测选参路径）。为让「训练期 join」
与「回测选参」共用同一份实现、杜绝口径分裂，抽到本模块。``watchlist_grid`` 与
``full_train_batched`` 均从这里取。

口径（主板 ±10%，一字板是其子集；ST ±5% 为已知盲区，不在此处理）：
- ``limit_up_seal``  = (close==high) & (ret1>=0.095)：涨停封板 → **当天尾盘买不进**（挡买入）。
- ``limit_down_seal``= (close==low)  & (ret1<=-0.095)：跌停封板 → **当天卖不出**（挡卖出）。
- ``tradable_ret_{h}d``：尾盘 T 买入、T+h 收盘卖出；若卖出日封跌停则顺延到下一可卖日收盘，
  上限 cap 个交易日（含预定日），窗口内均封则末日强制平仓。
- ``buyable_close``：当日是否可买入（尾盘收盘口径）——涨停封板(含一字)当天买不进。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config


def sell_roll_max_days() -> int:
    """跌停顺延卖出的上限交易日数（含预定卖出日）。单一真源在 backtest.bt_sell_roll_max_days()。"""
    return backtest.bt_sell_roll_max_days()


def rolled_sell_close(close: np.ndarray, sell_blocked: np.ndarray, horizon: int, cap: int) -> np.ndarray:
    """对每个买入日 i，预定 T+horizon 收盘卖出；若卖出日被封(sell_blocked)则顺延到下一可卖日收盘，
    最多顺延到 T+horizon+cap-1（含预定日共 cap 个交易日窗口），窗口内均封则取窗口末日收盘强制平仓。
    返回与 close 等长的实际卖出价数组；末尾越界(无足够未来数据)置 NaN，与 close.shift(-h) 丢尾一致。"""
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        base = i + horizon
        if base >= n:
            continue  # 未来数据不足，丢尾
        sell_idx = None
        last = min(base + cap - 1, n - 1)
        for j in range(base, last + 1):
            if not bool(sell_blocked[j]):
                sell_idx = j
                break
        if sell_idx is None:
            sell_idx = last  # 窗口内均封，末日强制平仓
        out[i] = close[sell_idx]
    return out


def _quant_dir() -> Path:
    return Path(config.QUANT_DIR)


def price_tradability(codes: list[str], horizons: list[int],
                      quant_dir: Path | None = None) -> pd.DataFrame:
    """从 price/{code}.parquet 读 OHLC，返回按 code+date 的可交易口径列。

    产出列（按 horizon 展开）：``target_ret_{h}d``（收盘→收盘乐观口径，供对照/兜底）、
    ``open_ret_{h}d``（次日开盘买入口径，若有 open）、``tradable_ret_{h}d``（跌停顺延实现收益）、
    以及 ``buyable_close`` / ``buyable_next``（可买入掩码）。

    ``code`` 统一 zfill(6)、``date`` 统一 datetime64[ns]，与训练面板 join 键一致。
    """
    base_dir = quant_dir or _quant_dir()
    frames: list[pd.DataFrame] = []
    for code in codes:
        path = base_dir / "price" / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            px = pd.read_parquet(path, columns=["code", "date", "open", "high", "low", "close"])
        except Exception:  # noqa: BLE001
            # 老价格文件可能没有 open/high/low，退回仅 close
            try:
                px = pd.read_parquet(path, columns=["code", "date", "close"])
            except Exception:  # noqa: BLE001
                continue
        if px.empty:
            continue
        px = px.copy()
        px["code"] = px["code"].astype(str).str.zfill(6)
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        for c in ("open", "high", "low", "close"):
            if c in px.columns:
                px[c] = pd.to_numeric(px[c], errors="coerce")
        px = px.dropna(subset=["code", "date", "close"]).sort_values("date")
        if px.empty:
            continue
        out = px[["code", "date"]].copy()
        has_open = "open" in px.columns and px["open"].notna().any()
        has_hl = "high" in px.columns and "low" in px.columns
        close_arr = px["close"].to_numpy(dtype=float)
        # 收盘封板判定（±10% 主板口径；一字板是其子集）。ST(±5%) 为已知盲区。
        limit_down_seal = None
        if has_hl:
            high_arr = px["high"].to_numpy(dtype=float)
            low_arr = px["low"].to_numpy(dtype=float)
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
                sell_close = rolled_sell_close(close_arr, limit_down_seal, horizon, cap)
                with np.errstate(all="ignore"):
                    out[f"tradable_ret_{horizon}d"] = sell_close / close_arr - 1
        # 次日是否可买入(次日开盘口径)：一字涨停(high==low 且上涨)当日买不进
        if has_open and has_hl:
            nxt_high = px["high"].shift(-1)
            nxt_low = px["low"].shift(-1)
            nxt_close = px["close"].shift(-1)
            entry = px["open"].shift(-1)
            one_word_up = (nxt_high == nxt_low) & (nxt_close > px["close"])
            out["buyable_next"] = (~one_word_up.fillna(False)) & entry.notna()
        else:
            out["buyable_next"] = True
        # 当日是否可买入(尾盘收盘口径)：涨停封板(含一字涨停)当天尾盘买不进
        if limit_down_seal is not None:
            out["buyable_close"] = ~limit_up_seal
        else:
            out["buyable_close"] = True
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
