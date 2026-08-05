"""实时买入候选榜（RankBoard）：跨票聚合器，盘中定期推一条「模型 Top-N 买入候选」digest。

与逐票策略（strategy.py）的区别：策略是「一条快照→一条离散 alert」，走 notifier.notify
过白名单+冷却；RankBoard 是【跨票汇总】，主动按模型预期收益排名并配盘中量标注，走
notifier.push 低层派发（不过白名单），自带节奏控制 + 指纹去重防刷屏。

排序主序 = 盘中重排后 score（RerankScorer：现役融合 pred 百分位锚定 + 盘中有界微调）；
候选池先通过三模型融合收益成本安全边际门，再按融合模型取 Top-rank_pool_n；重排只在池内微调，
不造新 alpha。

只读 ctx 内存状态（最新快照 + VWAP + ref），盘中绝不碰 quant_data。缺 expected_return
的票自动落榜。仅当 Top-N 榜单指纹（代码序 + 重排幅度 + 标注）变化才推。
"""
from __future__ import annotations

import time
from typing import Optional

from .notifier import Notifier
from .reference import expected_return_text
from .rerank import RerankScorer
from .strategy import StrategyContext


class RankBoard:
    def __init__(self, cfg, ctx: StrategyContext, notifier: Notifier,
                 name_map: Optional[dict] = None):
        self._cfg = cfg
        self._ctx = ctx
        self._notifier = notifier
        self._name_map = name_map or {}
        self._top_n = max(1, getattr(cfg, "rank_top_n", 5))
        self._interval = max(30, getattr(cfg, "rank_interval_sec", 300))
        self._last_emit = 0.0
        self._last_fingerprint: Optional[str] = None
        self._scorer = RerankScorer(cfg, ctx)  # 与 PaperTrader 共用的盘中重排打分器

    # ---- 展示辅助 ------------------------------------------------------------
    def _label(self, code: str) -> str:
        """代码 + 中文简称；name_map 以 6 位纯代码为 key，code 可能带券商后缀，两种口径都查。"""
        name = self._name_map.get(code)
        if name is None:
            digits = str(code).split(".", 1)[0].strip().zfill(6)
            name = self._name_map.get(digits)
        return f"{code} {name}" if name else str(code)

    def _tags(self, code: str, _exp: float) -> tuple[list[str], str]:
        """按盘中量给一只票拼标注，返回 (标注列表, 用于指纹的稳定摘要串)。

        只用现成的 Level-1 派生量：现价 vs VWAP（便宜/贵）、买一卖一失衡（买盘强弱）、
        距涨停空间和当日涨幅。缺量的标注自动省略，不崩。
        """
        tags: list[str] = []
        fp_parts: list[str] = []
        snap = self._ctx.snapshot_of(code)
        if snap is None:
            tags.append("待开盘")
            return tags, "wait"

        last = snap.last
        # 当日涨幅
        pct = snap.pct_change
        if pct is not None:
            tags.append(f"日内{pct:+.1%}")
            fp_parts.append(f"p{round(pct, 3)}")

        # 现价 vs VWAP：便宜/贵（入场时机）
        vwap = self._ctx.vwap_of(code)
        if vwap and last is not None and vwap > 0:
            rel = (last - vwap) / vwap
            if rel <= -0.01:
                tags.append("便宜(低VWAP)"); fp_parts.append("cheap")
            elif rel >= 0.01:
                tags.append("偏贵(高VWAP)"); fp_parts.append("rich")
            else:
                tags.append("近VWAP"); fp_parts.append("near")

        # 买盘强弱（买一/卖一失衡）——现成没人用的信号
        imb = snap.bid_ask_imbalance
        if imb is not None:
            if imb >= 0.2:
                tags.append("买盘强"); fp_parts.append("bid+")
            elif imb <= -0.2:
                tags.append("卖盘强"); fp_parts.append("ask+")

        # 距涨停空间
        if last is not None and snap.high_limited:
            room = (snap.high_limited - last) / snap.high_limited
            if room <= 0.001:
                tags.append("已封涨停"); fp_parts.append("lu")
            elif room <= 0.03:
                tags.append(f"距涨停{room:.1%}"); fp_parts.append("near_lu")

        return tags, "|".join(fp_parts)

    # ---- 主入口 --------------------------------------------------------------
    def maybe_emit(self, force: bool = False) -> bool:
        """到间隔则算榜；指纹变化（或 force）才推。返回是否实际推送。"""
        if not getattr(self._cfg, "rank_board_enabled", True):
            return False
        now = time.time()
        if not force and now - self._last_emit < self._interval:
            return False
        self._last_emit = now

        ranked = self._rank()
        if not ranked:
            return False
        title, body, fingerprint = self._render(ranked)
        if not force and fingerprint == self._last_fingerprint:
            return False  # 榜单没变，不刷屏
        self._last_fingerprint = fingerprint
        self._notifier.push(title, body)
        return True

    def _rank(self) -> list:
        """盘中重排后取 Top-N。候选池先过三模型融合收益门，再按融合 pred 取 Top-rank_pool_n，
        经 RerankScorer 盘中微调后按 score 降序，取前 rank_top_n 展示。

        排序主序 = 重排后 score（融合百分位锚定 + 盘中有界微调），预期%仍用 Ridge 口径。

        与 PaperTrader 买入腿同口径 drop_limit_up=True：封涨停/一字板当下买不进，
        摆在榜首无意义（且 last 顶死涨停价恒定 → 指纹不变 → 榜单假性「不更新」）。
        剔掉后榜单由真正可买候选填充，随盘中量刷新；炸板则自然回榜。
        """
        rows = self._scorer.ranked(drop_limit_up=True)
        return rows[: self._top_n]

    def _exp_str(self, code: str, exp: float) -> str:
        """展示扣除模拟盘 round-trip 成本后的净收益，并保留毛收益口径。"""
        return expected_return_text(
            self._ctx.ref_of(code), exp, getattr(self._cfg, "paper_cost", 0.002))

    def _rerank_str(self, row) -> str:
        """把重排原因拼成短串（▲加分项 / ▼减分项），无调整则空串。"""
        if not getattr(row, "reasons", None) or abs(getattr(row, "adj", 0.0)) < 1e-4:
            return ""
        ups = [name for name, c in row.reasons if c > 0]
        downs = [name for name, c in row.reasons if c < 0]
        parts = []
        if ups:
            parts.append("▲" + "+".join(dict.fromkeys(ups)))
        if downs:
            parts.append("▼" + "+".join(dict.fromkeys(downs)))
        return (" " + " ".join(parts)) if parts else ""

    def _render(self, ranked: list) -> tuple[str, str, str]:
        """把重排后 Top-N（RankRow 列表）拼成 (title, body, fingerprint)。"""
        from datetime import datetime

        marks = "①②③④⑤⑥⑦⑧⑨⑩"
        lines: list[str] = []
        fp_rows: list[str] = []
        for i, row in enumerate(ranked):
            code, exp = row.code, row.exp
            tags, fp = self._tags(code, exp)
            mark = marks[i] if i < len(marks) else f"{i + 1}."
            tag_str = (" " + " ".join(tags)) if tags else ""
            lines.append(f"{mark} {self._label(code)} {self._exp_str(code, exp)}"
                         f"{self._rerank_str(row)}{tag_str}")
            # 指纹并入 adj 粗分桶（0.05 一档），使盘中重排变化能触发推送但不因微抖动刷屏。
            adj_bucket = round(getattr(row, "adj", 0.0) / 0.05)
            fp_rows.append(f"{code}:{round(exp, 3)}:a{adj_bucket}:{fp}")
        pool_n = getattr(self._cfg, "rank_pool_n", 30)
        title = f"[实时榜] 重排Top{len(ranked)}（模型Top{pool_n}池） {datetime.now():%H:%M}"
        body = "\n".join(lines)
        fingerprint = ";".join(fp_rows)
        return title, body, fingerprint
