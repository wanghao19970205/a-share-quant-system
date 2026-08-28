"""V6 模拟盘：日线候选池 + 分钟级盘中入场择时。"""
from __future__ import annotations

import math
from typing import Optional

from .rerank import RerankScorer, _clamp
from .strategy import StrategyContext
from .v5 import V5PaperTrader


class V6RerankScorer(RerankScorer):
    """在 V5 盘中重排基础上增加有界的当前分钟稳定性评分。"""

    def __init__(self, cfg, ctx: StrategyContext):
        super().__init__(cfg, ctx)
        self._w_minute_speed = max(
            0.0, float(getattr(cfg, "paper_v6_minute_speed_weight", 0.20)))
        self._minute_move_scale = max(
            0.0005, float(getattr(cfg, "paper_v6_minute_move_scale", 0.006)))

    def _intraday_adj(self, code: str, exp: float, px: Optional[float]) -> tuple[float, list]:
        adj, reasons = super()._intraday_adj(code, exp, px)
        minute_return = self._ctx.minute_return_of(code)
        if minute_return is None or not math.isfinite(minute_return):
            return adj, reasons

        # 盘中择时第一版只惩罚当前分钟的急变，避免把瞬时拉升/跳水当作稳定入场点。
        # 不对缺失或普通波动做硬过滤，且总位移仍受 RerankScorer cap 约束。
        speed = _clamp(abs(float(minute_return)) / self._minute_move_scale, 0.0, 1.0)
        if speed >= 1e-4:
            contrib = -self._w_minute_speed * speed
            adj = _clamp(adj + contrib, -self._cap, self._cap)
            reasons.append(("分钟急变" if minute_return > 0 else "分钟走弱", contrib))
        return adj, reasons


class V6PaperTrader(V5PaperTrader):
    """V6 独立赛马账户：V5 规则 + 当前分钟入场择时。"""

    _FILE_SUFFIX = "_v6"
    _VERSION = 6
    _EVENT_TYPE = "paper_buy_decision_v6"
    _EVENT_ID_PREFIX = "paper-buy-decision-v6"
    _POSITION_ID_PREFIX = "paper-pos-v6"
    _PAPER_TITLE = "模拟盘V6"

    def __init__(self, cfg, ctx, notifier, sector_ctx, name_map=None):
        super().__init__(cfg, ctx, notifier, sector_ctx, name_map)
        self._scorer = V6RerankScorer(cfg, ctx)

    def _prefix(self) -> str:
        return "[paper_v6]"

    def _market_audit(self, code: str) -> dict:
        audit = super()._market_audit(code)
        audit.update({
            "minute_return": self._ctx.minute_return_of(code),
            "minute_amount_delta": self._ctx.minute_amount_delta_of(code),
        })
        return audit

    def _paper_config_extra(self) -> dict:
        extra = super()._paper_config_extra()
        extra.update({
            "entry_timing_rule": "current_minute_move_penalty_v1",
            "minute_speed_weight": self._scorer._w_minute_speed,
            "minute_move_scale": self._scorer._minute_move_scale,
            "minute_feature_source": "realtime_level1_minute_bucket",
        })
        return extra

    def _entry_extra(self, code: str, quote: dict) -> dict:
        extra = super()._entry_extra(code, quote)
        extra.update({
            "minute_return": self._ctx.minute_return_of(code),
            "minute_amount_delta": self._ctx.minute_amount_delta_of(code),
        })
        return extra

    def _trade_extra(self, pos: dict) -> dict:
        extra = super()._trade_extra(pos)
        extra.update({
            "minute_return_at_entry": pos.get("minute_return"),
            "minute_amount_delta_at_entry": pos.get("minute_amount_delta"),
        })
        return extra
