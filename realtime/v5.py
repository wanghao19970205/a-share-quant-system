"""V5 模拟盘：完整继承 V4，仅增加行业 ETF 相对强度动态仓位。"""
from __future__ import annotations

import math
from typing import Optional

from .v4 import V4PaperTrader


class V5PaperTrader(V4PaperTrader):
    """V5 独立赛马账户：V4 入场过滤 + 可审计的行业强度仓位系数。"""

    _FILE_SUFFIX = "_v5"
    _VERSION = 5
    _EVENT_TYPE = "paper_buy_decision_v5"
    _EVENT_ID_PREFIX = "paper-buy-decision-v5"
    _POSITION_ID_PREFIX = "paper-pos-v5"
    _PAPER_TITLE = "模拟盘V5"

    def __init__(self, cfg, ctx, notifier, sector_ctx, name_map=None):
        self._sector_allocation_cache: dict[str, dict] = {}
        super().__init__(cfg, ctx, notifier, sector_ctx, name_map)
        self._sector_lagging_factor = float(getattr(
            cfg, "paper_v5_sector_lagging_factor", 0.85))
        self._sector_neutral_factor = float(getattr(
            cfg, "paper_v5_sector_neutral_factor", 1.00))
        self._sector_strong_factor = float(getattr(
            cfg, "paper_v5_sector_strong_factor", 1.15))

    def _prefix(self) -> str:
        return "[paper_v5]"

    def _allocation_factor(self, code: str) -> tuple[float, Optional[str]]:
        concentration_factor, concentration_reason = super()._allocation_factor(code)
        if concentration_factor <= 0:
            return concentration_factor, concentration_reason
        sector = (self._sector_entry_cache.get(code)
                  or self._sector_ctx.assessment_for_stock(code))
        confidence = float(sector.get("mapping_confidence") or 0.0)
        status = str(sector.get("status") or "unavailable")
        raw_excess = sector.get("excess_return")
        try:
            excess = float(raw_excess)
        except (TypeError, ValueError):
            excess = None
        if excess is not None and not math.isfinite(excess):
            excess = None

        if (confidence < self._sector_ctx.mapping_min_confidence or
                status == "unavailable" or excess is None):
            bucket = "fallback_neutral"
            factor = self._sector_neutral_factor
        elif status == "strong":
            bucket = "strong"
            factor = self._sector_strong_factor
        elif excess < 0:
            bucket = "lagging"
            factor = self._sector_lagging_factor
        else:
            bucket = "neutral"
            factor = self._sector_neutral_factor

        detail = {
            "rule": "sector_relative_allocation_v1",
            "bucket": bucket,
            "factor": factor,
            "status": status,
            "excess_return": excess,
            "mapping_confidence": confidence,
            "sector": sector.get("sector"),
            "etf_code": sector.get("etf_code"),
        }
        self._sector_allocation_cache[code] = detail
        return concentration_factor * factor, f"行业强度仓位({bucket} x{factor:.2f})"

    def _entry_extra(self, code: str, quote: dict) -> dict:
        extra = super()._entry_extra(code, quote)
        extra["sector_allocation"] = self._sector_allocation_cache.get(code)
        return extra

    def _trade_extra(self, pos: dict) -> dict:
        extra = super()._trade_extra(pos)
        extra["sector_allocation_at_entry"] = pos.get("sector_allocation")
        return extra

    def _paper_config_extra(self) -> dict:
        extra = super()._paper_config_extra()
        extra.update({
            "sector_allocation_rule": "relative_strength_v1",
            "sector_lagging_factor": self._sector_lagging_factor,
            "sector_neutral_factor": self._sector_neutral_factor,
            "sector_strong_factor": self._sector_strong_factor,
        })
        return extra
