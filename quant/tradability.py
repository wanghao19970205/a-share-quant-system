"""可交易口径的单一真源：封板判定、跌停顺延卖出、可实现收益、可买入掩码。

此前这套逻辑内嵌在 ``quant/watchlist_grid.py``（回测选参路径）。为让「训练期 join」
与「回测选参」共用同一份实现、杜绝口径分裂，抽到本模块。``watchlist_grid`` 与
``full_train_batched`` 均从这里取。

口径（按板块合法涨跌停档位判定；主板同时考虑 10% 与 ST 的 5%）：
- 默认模式要求当日有成交量；缺少 high/low 或 volume 列时不得默认可交易（fail-closed）。
- ``limit_up_seal``  ：收盘触及涨停价，或一字板上涨 → **当天尾盘买不进**（挡买入）。
- ``limit_down_seal``：收盘触及跌停价，或一字板下跌 → **当天卖不出**（挡卖出）。
- 前收盘缺失或非正时两者均 fail-closed，避免坏价格数据被当成可成交样本。
- ``tradable_ret_{h}d``：尾盘 T 买入、T+h 收盘卖出；若卖出日封跌停则顺延到下一可卖日收盘，
  上限 cap 个交易日（含预定日），窗口内均封则末日强制平仓。
- ``buyable_close``：当日是否可买入（尾盘收盘口径）——涨停封板、零成交量或成交量缺失时不可买。

已知局限：价格文件不含 ST 状态，主板保留 5% 档会把"非 ST 股恰好以当日最高价收在 +5%"
误判为封板。评测口径上宁可保守（少算可成交），彻底解决需要引入 ST 状态数据。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config

# 封板/可交易判定的口径版本。**修改 seal_masks / price_tradability 的判定语义时必须 +1**，
# 否则 full_train_batched 的 window cache 会静默复用旧口径算出的窗口预测。
# v1: (close==high) & ret1>=0.095 的主板启发式。
# v2: 按板块合法涨跌停档位匹配 + 一字板无条件封板 + 前收盘非正 fail-closed。
SEAL_VERSION = 2


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


def limit_tiers(code: str) -> tuple[float, ...]:
    """该股票可能适用的涨跌幅档位。

    代码里拿不到 ST 状态，因此主板同时保留 10% 与 5% 两档（ST 为 5%），
    宁可把可能封板的样本判为不可交易，也不要把买不进的封板票算成可成交。
    """
    c = str(code).zfill(6)
    if c.startswith(("300", "301", "688", "689")):
        return (0.20,)
    # 北交所：430xxx / 83xxxx / 87xxxx / 88xxxx / 920xxx 均为 ±30%
    if c.startswith(("430", "83", "87", "88", "920")):
        return (0.30,)
    return (0.10, 0.05)


def seal_masks(px: pd.DataFrame, code: str) -> tuple[np.ndarray, np.ndarray]:
    """收盘封板判定，返回 (涨停封板, 跌停封板)。

    两条判据：
    - 收盘位于当日最高/最低价，且当日涨跌幅命中该板块的合法涨跌停档位；
    - 一字板（``high == low``）当日有方向性涨跌时，全天单一价格，无论幅度都开不了仓/平不了仓。
    前收盘缺失或非正时 fail-closed，视为封板，避免坏价格数据被当成可成交样本。
    """
    n = len(px)
    close = px["close"].to_numpy(dtype=float)
    high = px["high"].to_numpy(dtype=float)
    low = px["low"].to_numpy(dtype=float)
    prev = px["close"].shift(1).to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        ret1 = np.where(prev > 0, close / prev - 1.0, np.nan)
    # 涨跌停价按分四舍五入，低价股的档位误差更大，留 0.4% 匹配容差。
    tol = 0.004
    at_tier_up = np.zeros(n, dtype=bool)
    at_tier_down = np.zeros(n, dtype=bool)
    for tier in limit_tiers(code):
        with np.errstate(all="ignore"):
            at_tier_up |= np.abs(ret1 - tier) <= tol
            at_tier_down |= np.abs(ret1 + tier) <= tol
    at_high = close >= high - 1e-9
    at_low = close <= low + 1e-9
    one_word = high <= low + 1e-9
    up_move = np.where(np.isfinite(ret1), ret1 > 0, False)
    down_move = np.where(np.isfinite(ret1), ret1 < 0, False)
    bad_prev = ~np.isfinite(ret1)
    limit_up_seal = (at_high & at_tier_up) | (one_word & up_move) | bad_prev
    limit_down_seal = (at_low & at_tier_down) | (one_word & down_move) | bad_prev
    return limit_up_seal, limit_down_seal


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
            px = pd.read_parquet(
                path, columns=["code", "date", "open", "high", "low", "close", "volume"]
            )
        except Exception:  # noqa: BLE001
            # 老价格文件可能没有完整 OHLCV，缺 volume 时保持 fail-closed。
            try:
                px = pd.read_parquet(
                    path, columns=["code", "date", "open", "high", "low", "close"]
                )
            except Exception:  # noqa: BLE001
                try:
                    px = pd.read_parquet(path, columns=["code", "date", "close"])
                except Exception:  # noqa: BLE001
                    continue
        if px.empty:
            continue
        px = px.copy()
        px["code"] = px["code"].astype(str).str.zfill(6)
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        for c in ("open", "high", "low", "close", "volume"):
            if c in px.columns:
                px[c] = pd.to_numeric(px[c], errors="coerce")
        px = px.dropna(subset=["code", "date", "close"]).sort_values("date")
        if px.empty:
            continue
        out = px[["code", "date"]].copy()
        has_open = "open" in px.columns and px["open"].notna().any()
        has_hl = "high" in px.columns and "low" in px.columns
        has_volume = "volume" in px.columns
        positive_volume = (
            px["volume"].fillna(0.0).gt(0).to_numpy(dtype=bool)
            if has_volume
            else np.zeros(len(px), dtype=bool)
        )
        close_arr = px["close"].to_numpy(dtype=float)
        # 收盘封板判定：按板块合法涨跌停档位匹配，并把一字板无条件视为封板。
        limit_down_seal = None
        if has_hl:
            limit_up_seal, limit_down_seal = seal_masks(px, code)
        out["positive_volume"] = positive_volume
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
        # 次日是否可买入(次日开盘口径)：一字涨停(high==low 且上涨)当日买不进。
        # 成交量缺失或为零时 fail-closed，不能把不可执行样本当作可买。
        if has_open and has_hl:
            nxt_high = px["high"].shift(-1)
            nxt_low = px["low"].shift(-1)
            nxt_close = px["close"].shift(-1)
            entry = px["open"].shift(-1)
            one_word_up = (nxt_high == nxt_low) & (nxt_close > px["close"])
            out["buyable_next"] = (
                (~one_word_up.fillna(False))
                & entry.notna()
                & pd.Series(positive_volume, index=px.index).shift(-1, fill_value=False)
            )
        else:
            out["buyable_next"] = False
        # 当日是否可买入(尾盘收盘口径)：涨停封板或无成交量当天尾盘买不进。
        if limit_down_seal is not None:
            out["buyable_close"] = (~limit_up_seal) & positive_volume
        else:
            out["buyable_close"] = positive_volume
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
