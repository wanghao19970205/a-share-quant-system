"""V7 的选股池：60 日波动率的日度截面分位。

研究依据（`quant/lowfreq_backtest.py`，2057 个交易日、往返成本 0.004）：按 60 日
收益率标准差的截面分位，进 ``(0.30, 0.40]``、跌出 ``(0.20, 0.70]`` 才卖、等权持有，
对"全可买等权"基准年化超额 +16.07%（t=10.86），留出期（2023 起）+15.08%（t=6.50），
分年度 9 年全为正。窗口取 60 日是扫描出来的最优点：10/20/40 日排名不稳、换手更高，
120 日信号变钝（gross 从 +14.16% 掉到 +10.79%）。

这里只回答"今天每只票的波动率分位是多少"，进出阈值交给 V7 判定。分位是截面相对量，
所以样本不足的票直接不进结果，不做任何填充——宁可少一个候选，也不能让口径失真。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

VOL_WINDOW = 60
VOL_MIN_PERIODS = 45
# 只读尾部若干行即可算 60 日波动，避免每天全量扫历史。
_TAIL_ROWS = VOL_WINDOW + 10

_cache: dict[tuple, dict[str, float]] = {}


def _price_dir() -> Optional[Path]:
    try:
        from quant import config as _qc
    except Exception:  # noqa: BLE001 - 缺依赖时由调用方降级
        return None
    return Path(getattr(_qc, "PRICE_DIR", Path(_qc.QUANT_DIR) / "price"))


def universe(path: Optional[str] = None) -> list[str]:
    """读股票池文件。每行形如 ``000001 平安银行``，允许 ``#`` 注释。"""
    if path is None:
        try:
            from quant import config as _qc
        except Exception:  # noqa: BLE001
            return []
        path = getattr(_qc, "MAINBOARD_UNIVERSE_FILE", "")
    p = Path(path) if path else None
    if not p or not p.exists():
        return []
    codes: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        token = line.split()[0].strip()
        if token.isdigit():
            codes.append(token.zfill(6))
    return sorted(set(codes))


def _vol_of(px, window: int) -> Optional[float]:
    """尾部 window 段日收益率的标准差，口径与 ``quant.lowfreq_backtest._price_features``
    逐行对齐：先按日期排序、丢掉无效收盘，再 ``pct_change().rolling(window,
    min_periods=VOL_MIN_PERIODS).std()`` 取最后一个值。

    不要"先筛正收盘再算收益"这类看似更稳的写法——那会把停牌/异常价前后的样本拼到
    一起，算出研究口径里不存在的收益，实测会让个别标的的分位偏离 0.8 以上。
    """
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["date", "close"]).sort_values("date")
    if len(px) < 30:
        return None
    sd = px["close"].pct_change().rolling(
        window, min_periods=VOL_MIN_PERIODS).std().iloc[-1]
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        return None
    if sd != sd or sd <= 0:      # NaN 或退化
        return None
    return sd


def volatility(codes: list[str], window: int = VOL_WINDOW) -> dict[str, float]:
    """逐票算 60 日收益波动。读不到、列不全、样本不足的票一律不进结果。"""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    pdir = _price_dir()
    if pdir is None or not pdir.exists():
        return {}
    out: dict[str, float] = {}
    for code in codes:
        f = pdir / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            px = pd.read_parquet(f, columns=["date", "close"])
        except Exception:  # noqa: BLE001 - 单票坏文件不应影响整个截面
            continue
        if px is None or px.empty or "close" not in px.columns:
            continue
        sd = _vol_of(px.tail(_TAIL_ROWS), window)
        if sd is not None:
            out[code] = sd
    return out


def rank_pct(codes: Optional[list[str]] = None, window: int = VOL_WINDOW,
             as_of: Optional[_dt.date] = None) -> dict[str, float]:
    """今天的波动率分位（0<q<=1，越小波动越低），按日期缓存。

    分位口径与研究一致：``rank(pct=True, ascending=True)``，即平均序号除以样本数，
    并列取平均。样本量少于 100 时直接返回空——截面太小，分位没有意义。
    """
    key = (as_of or _dt.date.today(), window, tuple(codes) if codes else None)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    pool = codes if codes is not None else universe()
    vols = volatility(pool, window)
    if len(vols) < 100:
        _cache[key] = {}
        return {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    s = pd.Series(vols)
    ranked = s.rank(pct=True, ascending=True)
    out = {str(k): float(v) for k, v in ranked.items()}
    _cache[key] = out
    return out


def reset_cache() -> None:
    """测试与跨日重算用。"""
    _cache.clear()
