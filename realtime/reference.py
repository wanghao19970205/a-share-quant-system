"""启动期参考数据：为每只订阅票预取盘中纠偏用的静态基准，注入 StrategyContext。

盘中策略需要两类"昨天就能算好"的基准，用来校准模型给的静态价位：
- ATR(14)：日线真实波幅均值，供 ATR 吊灯止损计算跟踪止损线。
- expected_return：模型对该票的预期收益（ridge_pred），供开盘跳空校准判断
  "高开是否已吃掉预期空间"。

只在引擎启动时读一次 price 仓库 + 预测文件，构建 {code: RefRow} 字典；
盘中完全不再碰 quant_data，避免与业务数据链路抢 IO。
缺数据的票优雅跳过（对应策略自动降级/不触发），绝不因个别票缺失崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import RealtimeConfig


@dataclass
class RefRow:
    atr: Optional[float] = None          # ATR(14) 绝对值（元）
    atr_pct: Optional[float] = None      # ATR / 昨收，便于缺 ATR 时按比例兜底
    expected_return: Optional[float] = None  # 模型预期收益（ridge_pred，小数）
    hold_days: Optional[int] = None      # 已持有交易日数（仅持仓票有值；供 HoldingExpiry）


def _atr14(px, n: int = 14) -> tuple[Optional[float], Optional[float]]:
    """从日线 high/low/close 算 ATR(14) 及其相对昨收的百分比。"""
    import pandas as pd

    if px is None or px.empty or len(px) < n + 1:
        return None, None
    high = pd.to_numeric(px.get("high"), errors="coerce")
    low = pd.to_numeric(px.get("low"), errors="coerce")
    close = pd.to_numeric(px.get("close"), errors="coerce")
    if high is None or low is None or close is None:
        return None, None
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean().iloc[-1]
    last_close = close.iloc[-1]
    if atr != atr or atr is None:  # NaN
        return None, None
    atr = float(atr)
    atr_pct = atr / float(last_close) if last_close else None
    return atr, atr_pct


def _load_atr_map(cfg: RealtimeConfig, codes: list[str]) -> dict[str, tuple]:
    """逐票读 price/<code>.parquet 算 ATR。文件缺失/列不全则跳过该票。"""
    try:
        import pandas as pd
        from quant import config as _qc
    except Exception:  # noqa: BLE001
        return {}
    price_dir = Path(getattr(_qc, "PRICE_DIR", Path(_qc.QUANT_DIR) / "price"))
    out: dict[str, tuple] = {}
    for code in codes:
        p = price_dir / f"{code}.parquet"
        if not p.exists():
            continue
        try:
            px = pd.read_parquet(p, columns=["date", "high", "low", "close"])
        except Exception:  # noqa: BLE001
            try:
                px = pd.read_parquet(p)
            except Exception:  # noqa: BLE001
                continue
        if px is None or px.empty or "close" not in px.columns:
            continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px.dropna(subset=["date"]).sort_values("date")
        atr, atr_pct = _atr14(px)
        if atr is not None:
            out[code] = (atr, atr_pct)
    return out


def _load_expected_return(cfg: RealtimeConfig) -> dict[str, float]:
    """从最新一期预测取每票预期收益（优先 ridge_pred，退回 pred）。"""
    path = cfg.predictions_file
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    try:
        df = pd.read_parquet(path, columns=["code", "date", "ridge_pred", "pred"])
    except Exception:  # noqa: BLE001
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return {}
    if "code" not in df.columns:
        return {}
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df[d == d.max()]
    col = "ridge_pred" if "ridge_pred" in df.columns else (
        "pred" if "pred" in df.columns else None)
    if col is None:
        return {}
    out: dict[str, float] = {}
    for code, val in zip(df["code"].astype(str).str.zfill(6), df[col]):
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            out[code] = f
    return out


def _load_hold_days(cfg: RealtimeConfig) -> dict[str, int]:
    """从持仓文件解析每票已持有的交易日数（供 HoldingExpiry 判 T+N 到期）。

    持仓文件格式（每行）：`代码[ 买入日期]`，买入日期支持 YYYY-MM-DD / YYYYMMDD，
    分隔符空白或逗号。示例：
        600519 2026-07-23
        000001,20260722
        002211            # 无日期 → 视为今日买入(hold_days=0)，不触发到期
    以行内注释 # 之后忽略。买入日到今日之间的 A 股交易日数即 hold_days
    （买入当日 = 0；下一交易日 = 1 = T+1）。无 pandas / 无交易日历时按自然日
    近似（周末剔除），保证降级可用。缺日期或解析失败的行 hold_days 记 0。
    """
    path = cfg.holdings_file
    if not path.exists():
        return {}
    import datetime as _dt

    raw: dict[str, Optional[str]] = {}
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            s = ln.split("#", 1)[0].strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            code = parts[0].strip().zfill(6)
            buy = parts[1].strip() if len(parts) > 1 else None
            raw[code] = buy
    except Exception:  # noqa: BLE001
        return {}

    def _parse_date(txt: str) -> Optional[_dt.date]:
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return _dt.datetime.strptime(txt, fmt).date()
            except (TypeError, ValueError):
                continue
        return None

    today = _dt.date.today()
    out: dict[str, int] = {}
    for code, buy_txt in raw.items():
        if not buy_txt:
            out[code] = 0
            continue
        d = _parse_date(buy_txt)
        if d is None or d > today:
            out[code] = 0
            continue
        out[code] = _trading_days_between(d, today)
    return out


def _trading_days_between(start, end) -> int:
    """[start, end] 间的 A 股交易日数（不含 start 当日）。

    优先用项目交易日历；取不到则按工作日近似（周末剔除，不含节假日）。
    """
    try:
        import pandas as pd
        from quant import config as _qc
        cal_path = Path(getattr(_qc, "TRADING_CALENDAR_FILE",
                                Path(_qc.QUANT_DIR) / "trading_calendar.parquet"))
        if cal_path.exists():
            cal = pd.read_parquet(cal_path)
            col = next((c for c in ("date", "trade_date", "cal_date")
                        if c in cal.columns), None)
            if col is not None:
                days = pd.to_datetime(cal[col], errors="coerce").dt.date.dropna()
                n = int(((days > start) & (days <= end)).sum())
                return n
    except Exception:  # noqa: BLE001 - 无日历则退回工作日近似
        pass
    import datetime as _dt
    n, cur = 0, start
    while cur < end:
        cur = cur + _dt.timedelta(days=1)
        if cur.weekday() < 5:  # 0-4 = 周一至周五
            n += 1
    return n


def build(cfg: RealtimeConfig, codes: list[str]) -> dict[str, RefRow]:
    """构建 {code: RefRow}。任何来源缺失都优雅返回可用子集，不抛异常。"""
    atr_map = _load_atr_map(cfg, codes)
    ret_map = _load_expected_return(cfg)
    hold_map = _load_hold_days(cfg)
    ref: dict[str, RefRow] = {}
    for code in codes:
        atr, atr_pct = atr_map.get(code, (None, None))
        ref[code] = RefRow(atr=atr, atr_pct=atr_pct,
                           expected_return=ret_map.get(code),
                           hold_days=hold_map.get(code))
    return ref
