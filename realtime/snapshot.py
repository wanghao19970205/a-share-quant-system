"""Snapshot 数据模型 + 字段映射适配层。

设计要点：
- 手册附录 4.2.1 给出的 Snapshot 字段是我们的标准字段名；但 SDK 实际回调对象
  的属性名可能有细微差异（如 last vs last_price）。本模块用一层容错映射吸收这种
  差异——下个交易日盘中拿到真实回调后，只需在 _FIELD_ALIASES 里补映射，不动上层策略。
- from_obj(): 从 onSnapshot 回调对象（属性访问）提取，兼容 getattr。
- from_mapping(): 从 dict / DataFrame 行（query_snapshot 验证用）提取。
- 缺失字段填 None，上层策略自行判空，不因个别字段缺失崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional


# 标准字段 -> 可能的别名列表（第一个命中的取值）。真实字段名确认后在此补充。
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "code": ("code", "symbol", "security_id", "windcode", "ticker"),
    "trade_time": ("trade_time", "time", "datetime", "update_time", "timestamp"),
    "last": ("last", "last_price", "price", "current", "now"),
    "pre_close": ("pre_close", "prev_close", "preclose"),
    "open": ("open", "open_price"),
    "high": ("high", "high_price"),
    "low": ("low", "low_price"),
    "volume": ("volume", "vol", "cum_volume"),
    "amount": ("amount", "turnover", "cum_amount"),
    "high_limited": ("high_limited", "high_limit", "limit_up", "upper_limit"),
    "low_limited": ("low_limited", "low_limit", "limit_down", "lower_limit"),
    "bid_price1": ("bid_price1", "bid1", "bid_px1", "buy_price1"),
    "ask_price1": ("ask_price1", "ask1", "ask_px1", "sell_price1", "offer_price1"),
    "bid_volume1": ("bid_volume1", "bid_vol1", "buy_volume1"),
    "ask_volume1": ("ask_volume1", "ask_vol1", "sell_volume1", "offer_volume1"),
}


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # 屏蔽券商用极值/0 表示的空档位
    if f != f:  # NaN
        return None
    return f


@dataclass
class Snapshot:
    """一条标准化 Level-1 快照。字段名以手册附录 4.2.1 为准。"""

    code: Optional[str] = None
    trade_time: Optional[str] = None
    last: Optional[float] = None
    pre_close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None
    high_limited: Optional[float] = None
    low_limited: Optional[float] = None
    bid_price1: Optional[float] = None
    ask_price1: Optional[float] = None
    bid_volume1: Optional[float] = None
    ask_volume1: Optional[float] = None
    # 保留原始对象，方便策略层临时取未建模字段（调试期用）。
    raw: Any = None

    # ---- 派生量（策略常用，惰性计算避免污染字段）--------------------------
    @property
    def pct_change(self) -> Optional[float]:
        """相对昨收涨跌幅（小数，如 0.05=+5%）。"""
        if self.last is None or not self.pre_close:
            return None
        return self.last / self.pre_close - 1.0

    @property
    def is_limit_up(self) -> bool:
        """是否封涨停：最新价 >= 涨停价（留 1 厘容差）。"""
        if self.last is None or self.high_limited is None:
            return False
        return self.last >= self.high_limited - 0.001

    @property
    def is_limit_down(self) -> bool:
        if self.last is None or self.low_limited is None:
            return False
        return self.last <= self.low_limited + 0.001

    @property
    def bid_ask_imbalance(self) -> Optional[float]:
        """买一/卖一量的失衡度：>0 买盘强，<0 卖盘强。范围约 [-1, 1]。"""
        b, a = self.bid_volume1, self.ask_volume1
        if b is None or a is None or (b + a) == 0:
            return None
        return (b - a) / (b + a)


_NUMERIC_FIELDS = {
    "last", "pre_close", "open", "high", "low", "volume", "amount",
    "high_limited", "low_limited", "bid_price1", "ask_price1",
    "bid_volume1", "ask_volume1",
}
_STD_FIELDS = {f.name for f in fields(Snapshot)} - {"raw"}


def _pick(getter, std_name: str):
    """按别名列表依次尝试取值，返回第一个非 None。"""
    for alias in _FIELD_ALIASES.get(std_name, (std_name,)):
        val = getter(alias)
        if val is not None:
            return val
    return None


def from_obj(obj: Any) -> Snapshot:
    """从 onSnapshot 回调对象（属性访问）构造 Snapshot。"""
    def getter(name):
        return getattr(obj, name, None)

    kwargs: dict[str, Any] = {"raw": obj}
    for std in _STD_FIELDS:
        val = _pick(getter, std)
        kwargs[std] = _coerce_float(val) if std in _NUMERIC_FIELDS else (
            str(val) if val is not None else None)
    return Snapshot(**kwargs)


def from_mapping(row: Mapping[str, Any]) -> Snapshot:
    """从 dict / DataFrame 行构造 Snapshot（query_snapshot 验证用）。"""
    def getter(name):
        return row.get(name)

    kwargs: dict[str, Any] = {"raw": dict(row)}
    for std in _STD_FIELDS:
        val = _pick(getter, std)
        kwargs[std] = _coerce_float(val) if std in _NUMERIC_FIELDS else (
            str(val) if val is not None else None)
    return Snapshot(**kwargs)
