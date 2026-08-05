"""启动期参考数据：为每只订阅票预取盘中纠偏用的静态基准，注入 StrategyContext。

盘中策略需要两类"昨天就能算好"的基准，用来校准模型给的静态价位：
- ATR(14)：日线真实波幅均值，供 ATR 吊灯止损计算跟踪止损线。
- expected_return：Ridge/ElasticNet/ExtraTrees 同量纲融合收益率，供成本门和收益展示。
- model_rank_pct：现役融合 pred 的当日全A百分位，供实时候选主序。

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
    expected_return: Optional[float] = None  # 三模型融合收益率（成本门/收益展示）
    return_components: Optional[dict] = None  # Ridge/Elastic/ExtraTrees 各腿及归一化权重
    model_score: Optional[float] = None       # 现役融合 pred 原始分（无量纲，仅审计）
    model_rank_pct: Optional[float] = None    # 融合 pred 当日全A百分位（实时排序主序）
    calibrated_return: Optional[float] = None  # ridge_pred 历史分档的实际毛收益均值
    calibrated_net_return: Optional[float] = None  # 按模拟盘成交成本扣费后的历史净收益
    win_rate: Optional[float] = None     # 该分档历史 (target>0) 胜率（仅展示用）
    hold_days: Optional[int] = None      # 已持有交易日数（仅持仓票有值；供 HoldingExpiry）
    prediction_date: Optional[str] = None  # 当前 expected_return 对应预测日期（YYYY-MM-DD）


def net_return_after_cost(gross_return: float, roundtrip_cost: float) -> float:
    """按模拟盘买卖各半成本模型，把毛收益换算为净收益。"""
    half = max(0.0, float(roundtrip_cost)) / 2.0
    return (1.0 + float(gross_return)) * (1.0 - half) / (1.0 + half) - 1.0


def expected_return_text(ref: Optional[RefRow], raw_return: float,
                         roundtrip_cost: float) -> str:
    """同时展示原始模型净收益和历史校准收益，避免两种口径混成一个数。"""
    calibrated = getattr(ref, "calibrated_return", None) if ref is not None else None
    win_rate = getattr(ref, "win_rate", None) if ref is not None else None
    raw = float(raw_return)
    raw_net = net_return_after_cost(raw, roundtrip_cost)
    if calibrated is None:
        return f"模型净{raw_net:+.2%}(原始毛{raw:+.2%})"
    calibrated_gross = float(calibrated)
    win_text = f" 胜率{win_rate:.0%}" if win_rate is not None else ""
    return (f"模型净{raw_net:+.2%}(原始毛{raw:+.2%};"
            f"历史校准{calibrated_gross:+.2%}{win_text})")


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


_RETURN_MODEL_COLUMNS = ("ridge_pred", "elastic_pred", "extra_trees_pred")


def _return_model_weights(cfg: RealtimeConfig) -> dict[str, float]:
    """返回同量纲收益模型权重；关闭融合时只使用 Ridge。"""
    if not getattr(cfg, "ensemble_return_enabled", True):
        return {"ridge_pred": 1.0}
    weights = {
        "ridge_pred": float(getattr(cfg, "ensemble_ridge_weight", 0.30)),
        "elastic_pred": float(getattr(cfg, "ensemble_elastic_weight", 0.20)),
        "extra_trees_pred": float(getattr(cfg, "ensemble_extra_trees_weight", 0.50)),
    }
    positive = {key: max(0.0, value) for key, value in weights.items() if value > 0}
    return positive or {"ridge_pred": 1.0}


def _ensemble_return_series(df, cfg: RealtimeConfig):
    """逐行融合同一收益标签的模型；缺值时按该行可用权重重新归一化。"""
    import pandas as pd

    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for column, weight in _return_model_weights(cfg).items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        available = values.notna()
        numerator = numerator.add(values.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(available.astype(float) * weight, fill_value=0.0)
    return numerator.div(denominator.where(denominator > 0))


def _load_expected_return(
        cfg: RealtimeConfig,
) -> tuple[dict[str, float], Optional[str], dict[str, dict],
           dict[str, float], dict[str, float]]:
    """读取同量纲融合收益和现役融合 pred 的当日全A百分位。

    `expected_return` 只融合 Ridge/ElasticNet/ExtraTrees 三条 target_ret 回归腿；`pred` 包含
    LightGBM/IC 等无量纲信号，只转换成 [0,1] 百分位用于排序主序。
    """
    path = cfg.predictions_file
    if not path.exists():
        return {}, None, {}, {}, {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}, None, {}, {}, {}
    requested = ["code", "date", "pred", *_RETURN_MODEL_COLUMNS]
    try:
        df = pd.read_parquet(path, columns=requested)
    except Exception:  # noqa: BLE001
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return {}, None, {}, {}, {}
    if "code" not in df.columns or not any(
            column in df.columns for column in _RETURN_MODEL_COLUMNS):
        return {}, None, {}, {}, {}
    prediction_date = None
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        latest = d.max()
        if latest == latest:
            prediction_date = latest.strftime("%Y-%m-%d")
            df = df[d == latest].copy()
    expected_returns = _ensemble_return_series(df, cfg)
    raw_scores = (pd.to_numeric(df["pred"], errors="coerce")
                  if "pred" in df.columns else pd.Series(float("nan"), index=df.index))
    rank_pct = raw_scores.rank(method="average", pct=True)
    configured_weights = _return_model_weights(cfg)
    returns: dict[str, float] = {}
    components: dict[str, dict] = {}
    model_scores: dict[str, float] = {}
    model_rank_pct: dict[str, float] = {}
    codes = df["code"].astype(str).str.zfill(6)
    for index, (code, expected, model_score, percentile) in enumerate(zip(
            codes, expected_returns, raw_scores, rank_pct)):
        if expected != expected:
            continue
        row = df.iloc[index]
        model_returns = {}
        available_weights = {}
        for column, weight in configured_weights.items():
            raw = row.get(column)
            if raw is not None and raw == raw:
                model_returns[column] = float(raw)
                available_weights[column] = float(weight)
        total = sum(available_weights.values())
        normalized_weights = ({key: value / total
                               for key, value in available_weights.items()}
                              if total > 0 else {})
        returns[code] = float(expected)
        components[code] = {
            "source": ("same_target_weighted" if len(model_returns) > 1
                       else "single_model_fallback"),
            "returns": model_returns,
            "weights": normalized_weights,
        }
        if model_score == model_score and percentile == percentile:
            model_scores[code] = float(model_score)
            model_rank_pct[code] = float(percentile)
    return returns, prediction_date, components, model_scores, model_rank_pct


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
    """从历史预测表建融合收益 → 实际兑现（均值/胜率）的分档查找函数。

    使用与实时收益门完全相同的 Ridge/ElasticNet/ExtraTrees 权重逐行融合，再按融合收益
    等频分档，统计 target_ret_{h}d 的实际均值和胜率。模型列缺失时与实时加载一致，按
    该行可用权重归一化；样本不足时返回 None，展示直接回退融合原始收益。
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
    return_columns = [column for column in _RETURN_MODEL_COLUMNS if column in cols]
    if not return_columns:
        return None
    target_col = next((c for c in candidates if c in cols), None)
    if target_col is None:
        return None
    try:
        df = pd.read_parquet(path, columns=[*return_columns, target_col])
    except Exception:  # noqa: BLE001
        return None
    df["_expected_return"] = _ensemble_return_series(df, cfg)
    df = df.dropna(subset=["_expected_return", target_col])
    if len(df) < 500:  # 样本太少，校准不稳，降级
        return None
    bins = max(4, int(getattr(cfg, "calib_bins", 20)))
    try:
        # 等频分位分档；重复边界较多时 duplicates="drop" 自动减档。
        codes, edges = pd.qcut(df["_expected_return"], q=bins, labels=False,
                               retbins=True, duplicates="drop")
    except Exception:  # noqa: BLE001
        return None
    df = df.assign(_bin=codes)
    grp = df.groupby("_bin")[target_col]
    means = grp.mean()
    wins = df.assign(_w=(df[target_col] > 0)).groupby("_bin")["_w"].mean()
    # 按分档序号排序后对均值做累积最大值 → 保证「融合收益越高，校准收益不降」的单调性
    # （零依赖替代保序回归；短线个别档因样本噪声反转时抹平）。
    idx = sorted(means.index)
    mono = np.maximum.accumulate([float(means[i]) for i in idx])
    cal_by_bin = {b: mono[k] for k, b in enumerate(idx)}
    wr_by_bin = {b: float(wins[b]) for b in idx}
    inner_edges = list(edges[1:-1])  # 用于 np.searchsorted 定位分档

    def lookup(expected_return):
        if expected_return is None or expected_return != expected_return:  # None/NaN
            return None, None
        b = int(np.searchsorted(inner_edges, float(expected_return), side="right"))
        b = min(b, idx[-1])  # 落在最右开区间外时归入最高档
        return cal_by_bin.get(b), wr_by_bin.get(b)

    lookup.n_bins = len(idx)  # type: ignore[attr-defined]
    lookup.n_rows = int(len(df))  # type: ignore[attr-defined]
    lookup.target_col = target_col  # type: ignore[attr-defined]
    return lookup


def build(cfg: RealtimeConfig, codes: list[str]) -> dict[str, RefRow]:
    """构建 {code: RefRow}。任何来源缺失都优雅返回可用子集，不抛异常。"""
    atr_map = _load_atr_map(cfg, codes)
    ret_map, prediction_date, return_components, model_scores, model_rank_pct = (
        _load_expected_return(cfg))
    hold_map = _load_hold_days(cfg)
    calib = None
    try:
        calib = _build_calibration(cfg)
    except Exception as e:  # noqa: BLE001 - 校准失败只降级展示，不拦启动
        print(f"[reference] 校准表构建失败(展示回退融合原始收益)：{type(e).__name__}", flush=True)
    if calib is not None:
        print(f"[reference] 预期收益校准就绪：{calib.n_bins} 档 / {calib.n_rows} 行历史 "
              f"（口径 {calib.target_col}）", flush=True)
    ref: dict[str, RefRow] = {}
    for code in codes:
        atr, atr_pct = atr_map.get(code, (None, None))
        exp = ret_map.get(code)
        cal, wr = (calib(exp) if calib is not None else (None, None))
        net = (net_return_after_cost(cal, cfg.paper_cost) if cal is not None else None)
        ref[code] = RefRow(atr=atr, atr_pct=atr_pct,
                           expected_return=exp,
                           return_components=return_components.get(code),
                           model_score=model_scores.get(code),
                           model_rank_pct=model_rank_pct.get(code),
                           calibrated_return=cal, calibrated_net_return=net,
                           win_rate=wr, hold_days=hold_map.get(code),
                           prediction_date=prediction_date)
    return ref
