"""盘中动态重排打分器（RerankScorer）：RankBoard 与 PaperTrader 共用的单一排序入口。

背景：此前 RankBoard._rank 与 PaperTrader._rank 各写一套「按 expected_return 降序」逻辑，
容易漂移。这里抽成唯一打分器，两处复用——保证「你看到的榜单」与「模拟盘实际买的」是同一序。

设计原则（守 strategy/RankBoard 既定「模型选强票入池，盘中只做纠偏」）：
  - 候选池恒 = 模型 expected_return(ridge_pred) > 0 的票，按模型分取 Top-{rank_pool_n}（默认30）。
    盘中信号【绝不】把池外的票拉进来、【绝不】造新 alpha。
  - 重排分 score = exp * (1 + clamp(adj, -cap, +cap))：模型分是主序标尺，盘中 adj 只在
    有界范围（默认 ±30%）里对同池票微调名次——模型分差距大的票次序翻不动，差距小的才被盘中量分出高下。

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

from .strategy import StrategyContext, _digits


@dataclass
class RankRow:
    """一只票的重排结果。score 为重排后主序键，exp 为原始模型预期（展示/兜底用）。"""

    code: str
    exp: float                        # 原始 expected_return（ridge_pred），恒保留
    score: float                      # 重排后排序键 = exp*(1+clamp(adj))
    adj: float                        # 盘中调整幅度（已 clamp 到 ±cap）
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
        self._w_gap = float(getattr(cfg, "rerank_w_gap", 0.20))
        self._w_spread = float(getattr(cfg, "rerank_w_spread", 0.15))

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
        if snap is not None and exp and exp > 0:
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
               require_price: bool = False, drop_limit_up: bool = False) -> list[RankRow]:
        """返回按 score 降序的 RankRow 列表。

        - 候选池：expected_return>0 的模型票，先按模型分取 Top-{rank_pool_n}（重排作用域）。
        - require_price=True：仅保留有实时价的票（买入腿用，买不了的剔除）。
        - drop_limit_up=True：剔除封涨停票（14:50 买不进）。
        - exclude：按 6 位纯代码排除（如已持仓）。
        - rerank 关闭时退化为纯模型序（adj=0），保持旧行为。
        """
        exclude = exclude or set()
        ref = self._ctx.all_refs() or {}

        # 先取模型池（expected_return>0，按模型分降序 Top-pool_n）
        pool: list[tuple[str, float]] = []
        for code, r in ref.items():
            if _digits(code) in exclude:
                continue
            exp = getattr(r, "expected_return", None)
            if exp is None or exp <= 0:
                continue
            pool.append((code, float(exp)))
        pool.sort(key=lambda kv: kv[1], reverse=True)
        pool = pool[: self._pool_n]

        rows: list[RankRow] = []
        for code, exp in pool:
            px = self._price_of(code)
            if require_price and px is None:
                continue
            if drop_limit_up:
                snap = self._ctx.snapshot_of(code)
                if snap is not None and getattr(snap, "is_limit_up", False):
                    continue
            if self._enabled:
                adj, reasons = self._intraday_adj(code, exp, px)
            else:
                adj, reasons = 0.0, []
            score = exp * (1.0 + adj)
            rows.append(RankRow(code=code, exp=exp, score=score, adj=adj,
                                reasons=reasons, px=px))
        # 主序 score 降序；score 相等时回退模型分（稳定）
        rows.sort(key=lambda r: (r.score, r.exp), reverse=True)
        return rows

    def _price_of(self, code: str) -> Optional[float]:
        snap = self._ctx.snapshot_of(code)
        last = getattr(snap, "last", None) if snap is not None else None
        try:
            return float(last) if last else None
        except (TypeError, ValueError):
            return None
