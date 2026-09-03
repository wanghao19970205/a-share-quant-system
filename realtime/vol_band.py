"""V7/V8 的选股池：日度横截面分位（60 日波动率 / 20 日成交额）。

两条腿共用同一套机制——按某个横截面指标取一条分位带，进带买、跌出更宽的出带才卖、
等权持有，除分位带之外没有任何出场规则。区别只在指标：

- ``vol60``（V7）：60 日收益率标准差，进 ``(0.30, 0.40]``、出 ``(0.20, 0.70]``。
  研究依据（`quant/lowfreq_backtest.py`，2058 个交易日、往返成本 0.004、n=20）：
  年化超额 +15.89%，IS +15.29%（t 3.92）/ OOS +16.70%（t 3.60）。窗口 60 日是扫出来
  的最优点：10/20/40 日排名不稳、换手更高，120 日信号变钝。
- ``dollar_vol20``（V8）：20 日均成交额（``volume × close``），进 ``(0.10, 0.20]``、
  出 ``(0.05, 0.50]``。年化超额 +17.97%，IS +17.85%（t 4.32）/ OOS +17.23%（t 2.87）。

两条腿几乎不重合，这是开第二个账户的唯一理由：每个调仓日候选前 20 的 Jaccard 均值
0.0032（中位 0），日度超额相关系数 0.174，五五等权合并后 IR 从 1.86/1.78 升到 2.37。
低成交额那一档听起来像"买不进去"，实测不是：带内 20 日均成交额中位 4237 万元、最小
3455 万元，单笔 5000 元只占 0.012%，99.1% 的标的买得起 1 手。

上面那些超额数字要往下打一档。60 次截面内置换（gross 口径）测出：随机信号在同一套
机制下对全池基准也有 +5.46% 年化，其中 +0.79% 是池口径差（策略要求指标非空），
约 +4.4% 是持仓惯性——调仓日 1.99% 的样本不可买（多为涨停封板），次日收益均值
+1.776%（可买样本 -0.010%），而 `simulate()` 的已持仓不要求可买、基准却剔掉它们。
所以可归因于指标本身的 gross 超额是 19.94% - 5.46% ≈ 14.5%、扣成本约 10.5%。
观测值仍显著（固定带 z 4.6，7 组取最优 z 6.1，经验 p < 1/60），只是幅度没那么大。
注意带成本的置换检验没有意义：随机信号换手 0.6624/日（实测 0.0336），零分布会被成本
拖到 -46%，检验的是换手不是选股。

这里只回答"今天每只票在某个指标上的分位是多少"，进出阈值交给 V7/V8 判定。分位是截面
相对量，所以样本不足的票直接不进结果，不做任何填充——宁可少一个候选，也不能让口径失真。

与研究口径的差别不是"滞后一天"——价格文件里已经有当日的 bar，实盘 14:50 读到的是
**当日尚未走完的 bar**，研究口径用的是当日完整收盘。所以既没有前视，也不是 T-1：
当日成交量与 20 日均量之比中位 0.809，说明这根 bar 大约走了八成。实测（2026-09-03
全池 3191 只，vol60）两种口径带内成员 319 vs 319、交集 284、Jaccard 0.8023，全池分位
Spearman 0.998652，但最大绝对差 0.1377——分位在带下沿很密，"带内最低的 40 只"只重合
16/40。这不影响策略预期：同一条带下 target_n 从 20 扫到 319，年化超额
15.89%/17.52%/15.33%/16.07%/14.53%，留出期 t 全在 3.6 以上，说明超额来自"进了这条带"
而不是带内的细微排序，未走完的 bar 打乱的只是排序。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

VOL_WINDOW = 60
VOL_MIN_PERIODS = 45

# 每个指标的窗口、最短样本、需要的列和磁盘快照前缀。窗口与 min_periods 必须和
# ``quant.lowfreq_backtest._price_features`` 逐字一致，否则实盘选出来的不是研究里
# 那批标的（vol60 曾因为多筛了一次正收盘而让个别标的分位偏离 0.8 以上）。
METRICS = {
    "vol60": {"window": VOL_WINDOW, "min_periods": VOL_MIN_PERIODS,
              "cols": ["date", "close"], "file": "vol_rank"},
    "dollar_vol20": {"window": 20, "min_periods": 15,
                     "cols": ["date", "close", "volume"], "file": "dv_rank"},
}
DEFAULT_METRIC = "vol60"

_cache: dict[tuple, dict] = {}


def _price_dir() -> Optional[Path]:
    try:
        from quant import config as _qc
    except Exception:  # noqa: BLE001 - 缺依赖时由调用方降级
        return None
    return Path(getattr(_qc, "PRICE_DIR", Path(_qc.QUANT_DIR) / "price"))


def universe(path: Optional[str] = None) -> list:
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
    codes: list = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        token = line.split()[0].strip()
        if token.isdigit():
            codes.append(token.zfill(6))
    return sorted(set(codes))


def _clean(px):
    """与研究口径逐行对齐的预处理：按日期排序、丢掉无效日期/收盘。

    不要"先筛正收盘再算收益/成交额"这类看似更稳的写法——那会把停牌、异常价前后的
    样本拼到一起，算出研究口径里不存在的值。
    """
    import pandas as pd
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["date", "close"]).sort_values("date")
    return px if len(px) >= 30 else None


def _tail_value(px, metric: str) -> Optional[float]:
    """该票在给定指标上的最新值；样本不足、退化或缺列都返回 None。"""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    spec = METRICS[metric]
    px = _clean(px)
    if px is None:
        return None
    if metric == "vol60":
        value = px["close"].pct_change().rolling(
            spec["window"], min_periods=spec["min_periods"]).std().iloc[-1]
    else:
        if "volume" not in px.columns:
            return None
        vol = pd.to_numeric(px["volume"], errors="coerce")
        value = (vol * px["close"]).rolling(
            spec["window"], min_periods=spec["min_periods"]).mean().iloc[-1]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value <= 0:      # NaN 或退化
        return None
    return value


def metric_values(codes: list, metric: str = DEFAULT_METRIC) -> dict:
    """逐票算指标值。读不到、列不全、样本不足的票一律不进结果。"""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    spec = METRICS[metric]
    pdir = _price_dir()
    if pdir is None or not pdir.exists():
        return {}
    tail = spec["window"] + 10
    out: dict = {}
    for code in codes:
        f = pdir / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            px = pd.read_parquet(f, columns=spec["cols"])
        except Exception:  # noqa: BLE001 - 单票坏文件不应影响整个截面
            continue
        if px is None or px.empty or "close" not in px.columns:
            continue
        value = _tail_value(px.tail(tail), metric)
        if value is not None:
            out[code] = value
    return out


def volatility(codes: list, window: int = VOL_WINDOW) -> dict:
    """兼容入口：60 日波动率。新代码请直接用 ``metric_values``。"""
    return metric_values(codes, "vol60")


def _disk_cache_path(cache_dir, day: _dt.date, metric: str) -> Optional[Path]:
    if not cache_dir:
        return None
    spec = METRICS[metric]
    return Path(cache_dir) / f"{spec['file']}_{day:%Y%m%d}_{spec['window']}.json"


def _prune_disk_cache(path: Path, keep: int = 5) -> None:
    """只留最近几天的分位快照，避免 logs 目录无限增长。"""
    try:
        prefix, _, window = path.stem.rsplit("_", 2)
        for old in sorted(path.parent.glob(f"{prefix}_*_{window}.json"))[:-keep]:
            old.unlink()
    except (OSError, ValueError):
        pass


def rank_pct(codes: Optional[list] = None, metric: str = DEFAULT_METRIC,
             as_of: Optional[_dt.date] = None, cache_dir=None) -> dict:
    """今天该指标的横截面分位（0<q<=1，越小值越低），按日期缓存。

    分位口径与研究一致：``rank(pct=True, ascending=True)``，即平均序号除以样本数，
    并列取平均。样本量少于 100 时直接返回空——截面太小，分位没有意义。

    ``cache_dir`` 给出时额外落一份当日快照到磁盘。全池 3191 只要读 3000 多个
    parquet 尾部、实测 29 秒；进程内缓存救不了盘中 execv 重启，而重启可能正好落在
    14:50-14:55 买窗里，把所有账户的建仓机会拖过去。只有全池口径（codes 为空）
    才用磁盘缓存，指定 codes 时的截面语义不同，不能混用同一份文件。
    """
    if metric not in METRICS:
        raise ValueError(f"未知指标 {metric}，可选 {sorted(METRICS)}")
    day = as_of or _dt.date.today()
    key = (day, metric, tuple(codes) if codes else None)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    disk = _disk_cache_path(cache_dir, day, metric) if codes is None else None
    if disk is not None and disk.exists():
        try:
            import json
            cached = json.loads(disk.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and len(cached) >= 100:
                out = {str(k): float(v) for k, v in cached.items()}
                _cache[key] = out
                return out
        except Exception:  # noqa: BLE001 - 缓存坏了就重算，不影响交易
            pass
    pool = codes if codes is not None else universe()
    values = metric_values(pool, metric)
    if len(values) < 100:
        _cache[key] = {}
        return {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    ranked = pd.Series(values).rank(pct=True, ascending=True)
    out = {str(k): float(v) for k, v in ranked.items()}
    _cache[key] = out
    if disk is not None:
        try:
            import json
            import os
            disk.parent.mkdir(parents=True, exist_ok=True)
            tmp = disk.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(out), encoding="utf-8")
            os.replace(tmp, disk)
            _prune_disk_cache(disk)
        except Exception:  # noqa: BLE001 - 落盘失败只是慢一点，不该拦交易
            pass
    return out


def reset_cache() -> None:
    """测试与跨日重算用。"""
    _cache.clear()
