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
    expected_return: Optional[float] = None  # 模型预期收益（ridge_pred 原始小数，供 GapCalibrate 等）
    calibrated_return: Optional[float] = None  # ridge_pred 按历史分档重标定的实际兑现均值（仅展示用）
    win_rate: Optional[float] = None     # 该分档历史 (target>0) 胜率（仅展示用）
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
    """从最新一期预测取每票预期收益，只认 ridge_pred（Ridge 对未来收益率的回归预测，有量纲小数）。

    注意：绝不退回融合分 `pred`。`pred` 是各腿 z-score 的加权排序分（无量纲），
    与「预期收益率」量纲不同；GapCalibrate 用 expected_return 算 gap/exp 与判 exp<=0，
    若拿排序分冒充收益率会产生错误的跳空纠偏告警。缺 ridge_pred 时返回空 map
    → expected_return=None → 相关策略自动降级不触发。
    """
    path = cfg.predictions_file
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    try:
        df = pd.read_parquet(path, columns=["code", "date", "ridge_pred"])
    except Exception:  # noqa: BLE001
        # 列裁剪失败（老/变体文件无 ridge_pred 列）→ 退回读全表，但仍只认 ridge_pred。
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return {}
    if "code" not in df.columns or "ridge_pred" not in df.columns:
        return {}
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df[d == d.max()]
    out: dict[str, float] = {}
    for code, val in zip(df["code"].astype(str).str.zfill(6), df["ridge_pred"]):
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


def _build_calibration(cfg: RealtimeConfig):
    """从历史预测表建 ridge_pred → 实际兑现（均值/胜率）的分档查找函数。

    背景：ridge_pred 是 Ridge 对 T+N 收益率小数的回归拟合，强正则收缩 + 短线实现分布
    右偏厚尾 → 点估计系统性偏保守（展示「预期+1%」的票日内常涨更多）。这里用同一份
    预测文件的历史行（含已实现 target_ret_{h}d）按 ridge_pred 等频分档，求每档实际
    兑现的平均收益与胜率，把裸 ridge_pred 重标定成「该档历史真实兑现」用于展示。

    数据源 = cfg.predictions_file（即 _load_expected_return 读的同一份，零新增依赖）。
    只读两列，丢最新未实现日（target NaN）。样本不足/无 target 列 → 返回 None（优雅降级，
    展示回退原始 ridge_pred）。返回 lookup(ridge)->(cal_return, win_rate)。
    """
    if not getattr(cfg, "calib_enabled", True):
        return None
    path = cfg.predictions_file
    if not path.exists():
        return None
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    horizon = max(1, int(getattr(cfg, "sell_horizon", 1)))
    # 与实盘卖点口径一致：优先 close→close 的 target_ret_{h}d，退回可交易口径。
    candidates = [f"target_ret_{horizon}d", f"tradable_ret_{horizon}d", f"open_ret_{horizon}d"]
    try:
        head = pd.read_parquet(path, columns=None).head(0)
        cols = set(head.columns)
    except Exception:  # noqa: BLE001
        return None
    if "ridge_pred" not in cols:
        return None
    target_col = next((c for c in candidates if c in cols), None)
    if target_col is None:
        return None
    try:
        df = pd.read_parquet(path, columns=["ridge_pred", target_col])
    except Exception:  # noqa: BLE001
        return None
    df = df.dropna(subset=["ridge_pred", target_col])
    if len(df) < 500:  # 样本太少，校准不稳，降级
        return None
    bins = max(4, int(getattr(cfg, "calib_bins", 20)))
    try:
        # 等频分位分档；重复边界（ridge_pred 大量并列）时 duplicates="drop" 自动减档。
        codes, edges = pd.qcut(df["ridge_pred"], q=bins, labels=False,
                               retbins=True, duplicates="drop")
    except Exception:  # noqa: BLE001
        return None
    df = df.assign(_bin=codes)
    grp = df.groupby("_bin")[target_col]
    means = grp.mean()
    wins = df.assign(_w=(df[target_col] > 0)).groupby("_bin")["_w"].mean()
    # 按分档序号排序后对均值做累积最大值 → 保证「ridge 越高，校准收益不降」的单调性
    # （零依赖替代保序回归；短线个别档因样本噪声反转时抹平）。
    idx = sorted(means.index)
    mono = np.maximum.accumulate([float(means[i]) for i in idx])
    cal_by_bin = {b: mono[k] for k, b in enumerate(idx)}
    wr_by_bin = {b: float(wins[b]) for b in idx}
    inner_edges = list(edges[1:-1])  # 用于 np.searchsorted 定位分档

    def lookup(ridge):
        if ridge is None or ridge != ridge:  # None/NaN
            return None, None
        b = int(np.searchsorted(inner_edges, float(ridge), side="right"))
        b = min(b, idx[-1])  # 落在最右开区间外时归入最高档
        return cal_by_bin.get(b), wr_by_bin.get(b)

    lookup.n_bins = len(idx)  # type: ignore[attr-defined]
    lookup.n_rows = int(len(df))  # type: ignore[attr-defined]
    lookup.target_col = target_col  # type: ignore[attr-defined]
    return lookup


def build(cfg: RealtimeConfig, codes: list[str]) -> dict[str, RefRow]:
    """构建 {code: RefRow}。任何来源缺失都优雅返回可用子集，不抛异常。"""
    atr_map = _load_atr_map(cfg, codes)
    ret_map = _load_expected_return(cfg)
    hold_map = _load_hold_days(cfg)
    calib = None
    try:
        calib = _build_calibration(cfg)
    except Exception as e:  # noqa: BLE001 - 校准失败只降级展示，不拦启动
        print(f"[reference] 校准表构建失败(展示回退原始 ridge_pred)：{type(e).__name__}", flush=True)
    if calib is not None:
        print(f"[reference] 预期收益校准就绪：{calib.n_bins} 档 / {calib.n_rows} 行历史 "
              f"（口径 {calib.target_col}）", flush=True)
    ref: dict[str, RefRow] = {}
    for code in codes:
        atr, atr_pct = atr_map.get(code, (None, None))
        exp = ret_map.get(code)
        cal, wr = (calib(exp) if calib is not None else (None, None))
        ref[code] = RefRow(atr=atr, atr_pct=atr_pct,
                           expected_return=exp,
                           calibrated_return=cal, win_rate=wr,
                           hold_days=hold_map.get(code))
    return ref
