"""V4 模拟盘：完整继承 V3，仅增加行业 ETF 相对弱势入场过滤。"""
from __future__ import annotations

from typing import Optional

from .sector_etf import SectorETFContext
from .v3 import V3PaperTrader


class V4PaperTrader(V3PaperTrader):
    """V4 独立赛马账户：V3 可成交框架 + 弱板块回避。"""

    _FILE_SUFFIX = "_v4"
    _VERSION = 4
    _EVENT_TYPE = "paper_buy_decision_v4"
    _EVENT_ID_PREFIX = "paper-buy-decision-v4"
    _POSITION_ID_PREFIX = "paper-pos-v4"
    _PAPER_TITLE = "模拟盘V4"

    def __init__(self, cfg, ctx, notifier, sector_ctx: SectorETFContext, name_map=None):
        self._sector_ctx = sector_ctx
        self._sector_entry_cache: dict[str, dict] = {}
        super().__init__(cfg, ctx, notifier, name_map)

    def _prefix(self) -> str:
        return "[paper_v4]"

    def _entry_quote_detail(self, code: str, exp: float) -> tuple[Optional[str], Optional[dict]]:
        skip, quote = super()._entry_quote_detail(code, exp)
        if skip or quote is None:
            return skip, quote
        sector = self._sector_ctx.assessment_for_stock(code)
        self._sector_entry_cache[code] = sector
        confidence = float(sector.get("mapping_confidence") or 0.0)
        if (sector.get("status") == "weak" and
                confidence >= self._sector_ctx.mapping_min_confidence):
            excess = sector.get("excess_return")
            text = "缺失" if excess is None else f"{excess:+.2%}"
            return f"行业ETF相对弱势({sector.get('sector') or '未知'} {text})", None
        quote["sector_etf"] = sector
        return None, quote

    def _market_audit(self, code: str) -> dict:
        audit = super()._market_audit(code)
        audit["sector_etf"] = (
            self._sector_entry_cache.get(code)
            or self._sector_ctx.assessment_for_stock(code)
        )
        return audit

    def _entry_extra(self, code: str, quote: dict) -> dict:
        return {"sector_etf": quote.get("sector_etf") or self._sector_entry_cache.get(code)}

    def _trade_extra(self, pos: dict) -> dict:
        return {"sector_etf_at_entry": pos.get("sector_etf")}

    def _paper_config_extra(self) -> dict:
        return {
            "sector_entry_rule": "reject_weak",
            "sector_weak_excess": self._sector_ctx.weak_threshold,
            "sector_strong_excess": self._sector_ctx.strong_threshold,
            "sector_benchmark": self._sector_ctx.benchmark,
            "sector_mapping_version": self._sector_ctx.mapping_version,
            "sector_mapping_min_confidence": self._sector_ctx.mapping_min_confidence,
        }
