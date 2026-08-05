"""盘中动态重排打分器（RerankScorer）：RankBoard 与 PaperTrader 共用的单一排序入口。

背景：此前 RankBoard._rank 与 PaperTrader._rank 各写一套「按 expected_return 降序」逻辑，
容易漂移。这里抽成唯一打分器，两处复用——保证「你看到的榜单」与「模拟盘实际买的」是同一序。

设计原则（守 strategy/RankBoard 既定「模型选强票入池，盘中只做纠偏」）：
  - 候选池先用 Ridge/ElasticNet/ExtraTrees 融合收益覆盖成本和安全边际，再按现役融合 pred 的当日全A百分位
    取 Top-{rank_pool_n}（默认30）；历史校准净收益仅供展示/审计。
  - 重排主序 = 融合 pred 百分位 + 有界盘中位移。盘口只在相邻模型候选间纠偏，不把池外票
    拉进来；旧预测缺 pred 时退回 ridge_pred * (1 + adj) 的原行为。

盘中调整分 adj（各分项先归一到 [-1,1]，再加权求和，最后 clamp 到 ±cap）——只用现成的、
只读 ctx 内存态的量，缺某项即跳过该项（不崩、不误判）：
  - VWAP 位置：现价低于 VWAP（便宜）→ 加分；高于（偏贵/追高）→ 减分。
  - 买卖失衡：买一强 → 加分；卖一强 → 减分（L1 bid_ask_imbalance，已有）。
  - 高开吃预期：开盘跳空已吃掉模型预期越多 → 减分越狠（追高惩罚）。
  - 盘口价差：价差窄（流动性好）→ 轻微加分；价差宽 → 减分（方向4，L1 spread_pct）。

只读 ctx（snapshot_of / vwap_of / ref_of / all_refs），盘中绝不碰 quant_data。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .reference import net_return_after_cost
from .strategy import StrategyContext, _digits


@dataclass
class RankRow:
    """一只票的重排结果。score 为重排后主序键，exp 为原始模型预期（展示/兜底用）。"""

    code: str
    exp: float                        # 三模型融合 expected_return（收益门/展示），恒保留
    score: float                      # 重排后排序键（融合百分位主序；缺失时回退 Ridge）
    adj: float                        # 盘中调整幅度（已 clamp 到 ±cap）
    model_score: Optional[float] = None     # 现役融合 pred 原始分（无量纲）
    model_rank_pct: Optional[float] = None  # 融合 pred 当日全A百分位
    reasons: list = field(default_factory=list)  # 命中的加减分项（供展示「为何上/下移」）
    px: Optional[float] = None        # 最新实时价（require_price 时必有，否则可能 None）


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class RerankScorer:
    def __init__(self, cfg, ctx: StrategyContext):
        self._cfg = cfg
        self._ctx = ctx
        self._enabled = bool(getattr(cfg, "rerank_enabled", True))
        self._pool_n = max(1, int(getattr(cfg, "rank_pool_n", 30)))
        self._cap = max(0.0, float(getattr(cfg, "rerank_cap", 0.30)))
        self._w_vwap = float(getattr(cfg, "rerank_w_vwap", 0.35))
        self._w_imb = float(getattr(cfg, "rerank_w_imb", 0.30))
        self._w_gap = float(getattr(cfg, "rerank_w_gap", 0.0))
        self._w_spread = float(getattr(cfg, "rerank_w_spread", 0.15))
        self._rank_scale = max(
            0.0, float(getattr(cfg, "rerank_intraday_rank_scale", 0.10)))
        self._cost = max(0.0, float(getattr(cfg, "paper_cost", 0.002)))
        self._raw_safety = max(0.0, float(getattr(cfg, "rank_raw_safety_margin", 0.001)))
        configured_min = max(0.0, float(getattr(cfg, "rank_min_raw_return", 0.0)))
        self._min_raw = max(configured_min, self._cost + self._raw_safety)

    # ---- 盘中调整分 --------------------------------------------------------
    def _intraday_adj(self, code: str, exp: float, px: Optional[float]) -> tuple[float, list]:
        """算该票盘中调整分 adj∈[-cap,+cap] 及命中原因列表。缺量的分项自动跳过。"""
        snap = self._ctx.snapshot_of(code)
        reasons: list[str] = []
        raw = 0.0

        last = px if px is not None else (getattr(snap, "last", None) if snap is not None else None)

        # 1) VWAP 位置：(vwap-last)/vwap，>0=现价低于均价=便宜=加分
        vwap = self._ctx.vwap_of(code)
        if vwap and vwap > 0 and last is not None:
            rel = _clamp((vwap - last) / vwap, -1.0, 1.0)
            if abs(rel) >= 1e-4:
                contrib = self._w_vwap * rel
                raw += contrib
                reasons.append(("便宜" if rel > 0 else "偏贵", contrib))

        # 2) 买卖失衡（L1）：>0 买盘强=加分
        imb = getattr(snap, "bid_ask_imbalance", None) if snap is not None else None
        if imb is not None:
            imb = _clamp(float(imb), -1.0, 1.0)
            if abs(imb) >= 1e-4:
                contrib = self._w_imb * imb
                raw += contrib
                reasons.append(("买盘强" if imb > 0 else "卖盘强", contrib))

        # 3) 高开吃预期（追高惩罚）：gap/exp 越大 → 减分越狠
        if self._w_gap > 0 and snap is not None and exp and exp > 0:
            open_px = getattr(snap, "open", None)
            pre_close = getattr(snap, "pre_close", None)
            if open_px and pre_close:
                gap = open_px / pre_close - 1.0
                eaten = _clamp(gap / exp, -1.0, 1.0)  # 归一：吃满预期=1
                if eaten >= 1e-4:  # 只惩罚高开吃预期，不奖励低开（低开由 VWAP 便宜项已捕捉）
                    contrib = -self._w_gap * eaten
                    raw += contrib
                    reasons.append(("追高", contrib))

        # 4) 盘口价差（方向4，L1）：价差窄=流动性好=轻微加分；宽=减分
        spread = getattr(snap, "spread_pct", None) if snap is not None else None
        if spread is not None and spread >= 0:
            # 以 0.5% 为一档标尺归一：价差越大越接近 -1；0 价差 → +1（最好）
            norm = _clamp(1.0 - float(spread) / 0.005, -1.0, 1.0)
            contrib = self._w_spread * norm
            raw += contrib
            reasons.append(("盘口紧" if norm > 0 else "盘口宽", contrib))

        return _clamp(raw, -self._cap, self._cap), reasons

    # ---- 主入口 ------------------------------------------------------------
    def ranked(self, exclude: Optional[set] = None,
               require_price: bool = False, drop_limit_up: bool = False,
               trace: Optional[list[dict]] = None) -> list[RankRow]:
        """返回按 score 降序的 RankRow 列表。

        - 候选池：同量纲融合收益覆盖成本和安全边际，再按融合 pred 百分位取 Top-{rank_pool_n}；历史校准只做展示/审计。
        - require_price=True：仅保留有实时价的票（买入腿用，买不了的剔除）。
        - drop_limit_up=True：剔除封涨停票（14:50 买不进）。
        - exclude：按 6 位纯代码排除（如已持仓）。
        - rerank 关闭时退化为纯模型序（adj=0），保持旧行为。
        - trace：可选旁路审计容器；只记录既有分支结果，不改变筛选和排序。
        """
        exclude = exclude or set()
        ref = self._ctx.all_refs() or {}
        audit: dict[str, dict] = {}

        def _mark(code: str, status: str, **values) -> None:
            if trace is not None:
                audit[_digits(code)] = {"code": _digits(code), "status": status, **values}

        # 同量纲融合收益先做可交易门；池内主序使用已批准融合 pred 的当日全A百分位。
        pool: list[tuple[str, float, Optional[float], Optional[float]]] = []
        for code, r in ref.items():
            clean = _digits(code)
            if clean in exclude:
                _mark(code, "excluded_held")
                continue
            exp = getattr(r, "expected_return", None)
            if exp is None:
                _mark(code, "excluded_missing_expected_return")
                continue
            if exp <= 0:
                _mark(code, "excluded_nonpositive_expected_return",
                      expected_return=float(exp))
                continue
            calibrated = getattr(r, "calibrated_return", None)
            calibrated_net = getattr(r, "calibrated_net_return", None)
            if calibrated_net is None:
                gross = float(calibrated) if calibrated is not None else float(exp)
                calibrated_net = net_return_after_cost(gross, self._cost)
            model_score = getattr(r, "model_score", None)
            model_rank_pct = getattr(r, "model_rank_pct", None)
            values = {
                "expected_return": float(exp),
                "raw_net_return": float(net_return_after_cost(float(exp), self._cost)),
                "raw_min_return": float(self._min_raw),
                "return_components": getattr(r, "return_components", None),
                "model_score": (float(model_score) if model_score is not None else None),
                "model_rank_pct": (float(model_rank_pct)
                                   if model_rank_pct is not None else None),
                "calibrated_return": (float(calibrated) if calibrated is not None else None),
                "calibrated_net_return": float(calibrated_net),
            }
            if float(exp) < self._min_raw:
                _mark(code, "excluded_raw_return_gate", **values)
                continue
            pool.append((code, float(exp), model_score, model_rank_pct))
            _mark(code, "model_pool", **values)
        pool.sort(key=lambda item: (
            item[3] is not None,
            float(item[3]) if item[3] is not None else float(item[1]),
            float(item[1]),
        ), reverse=True)
        for code, _, _, _ in pool[self._pool_n:]:
            if trace is not None:
                audit[_digits(code)]["status"] = "excluded_outside_model_top_pool"
        pool = pool[: self._pool_n]

        rows: list[RankRow] = []
        for code, exp, model_score, model_rank_pct in pool:
            px = self._price_of(code)
            if require_price and px is None:
                if trace is not None:
                    audit[_digits(code)]["status"] = "excluded_missing_realtime_price"
                continue
            if drop_limit_up:
                snap = self._ctx.snapshot_of(code)
                if snap is not None and getattr(snap, "is_limit_up", False):
                    if trace is not None:
                        audit[_digits(code)]["status"] = "excluded_limit_up"
                    continue
            if self._enabled:
                adj, reasons = self._intraday_adj(code, exp, px)
            else:
                adj, reasons = 0.0, []
            if model_rank_pct is not None:
                score = float(model_rank_pct) + self._rank_scale * adj
            else:
                score = exp * (1.0 + adj)
            rows.append(RankRow(
                code=code, exp=exp, score=score, adj=adj,
                model_score=(float(model_score) if model_score is not None else None),
                model_rank_pct=(float(model_rank_pct)
                                if model_rank_pct is not None else None),
                reasons=reasons, px=px))
        # 主序 score 降序；并列时依次回退融合百分位和 Ridge 收益率。
        rows.sort(key=lambda row: (
            row.score,
            row.model_rank_pct if row.model_rank_pct is not None else -1.0,
            row.exp,
        ), reverse=True)
        if trace is not None:
            for rank, row in enumerate(rows, 1):
                audit[_digits(row.code)].update({
                    "status": "eligible_ranked", "rank": rank,
                    "score": float(row.score), "adj": float(row.adj),
                    "model_score": row.model_score,
                    "model_rank_pct": row.model_rank_pct,
                    "reasons": [[name, float(value)] for name, value in row.reasons],
                    "price": row.px,
                })
            trace.extend(audit.values())
        return rows

    def _price_of(self, code: str) -> Optional[float]:
        snap = self._ctx.snapshot_of(code)
        last = getattr(snap, "last", None) if snap is not None else None
        try:
            return float(last) if last else None
        except (TypeError, ValueError):
            return None
