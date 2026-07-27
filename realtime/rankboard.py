"""实时买入候选榜（RankBoard）：跨票聚合器，盘中定期推一条「模型 Top-N 买入候选」digest。

与逐票策略（strategy.py）的区别：策略是「一条快照→一条离散 alert」，走 notifier.notify
过白名单+冷却；RankBoard 是【跨票汇总】，主动按模型预期收益排名并配盘中量标注，走
notifier.push 低层派发（不过白名单），自带节奏控制 + 指纹去重防刷屏。

排序主序 = 模型 expected_return（ridge_pred，启动期已由 reference 加载进 ctx.ref）；
盘中信号（VWAP 偏离 / 买盘失衡 / 距涨停 / 当日涨幅）只做【标注】，不改排序、不造新 alpha
——守 strategy 既定原则「模型选强票入池，盘中只做纠偏」。

只读 ctx 内存状态（最新快照 + VWAP + ref），盘中绝不碰 quant_data。缺 expected_return
的票自动落榜。仅当 Top-N 榜单指纹（代码序 + 标注）变化才推。
"""
from __future__ import annotations

import time
from typing import Optional

from .notifier import Notifier
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

    # ---- 展示辅助 ------------------------------------------------------------
    def _label(self, code: str) -> str:
        """代码 + 中文简称；name_map 以 6 位纯代码为 key，code 可能带券商后缀，两种口径都查。"""
        name = self._name_map.get(code)
        if name is None:
            digits = str(code).split(".", 1)[0].strip().zfill(6)
            name = self._name_map.get(digits)
        return f"{code} {name}" if name else str(code)

    def _tags(self, code: str, exp: float) -> tuple[list[str], str]:
        """按盘中量给一只票拼标注，返回 (标注列表, 用于指纹的稳定摘要串)。

        只用现成的 Level-1 派生量：现价 vs VWAP（便宜/贵）、买一卖一失衡（买盘强弱）、
        距涨停空间、当日涨幅、开盘是否已吃预期。缺量的标注自动省略，不崩。
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

        # 开盘跳空是否已吃掉预期（追高风险）
        if snap.open is not None and snap.pre_close and exp > 0:
            gap = snap.open / snap.pre_close - 1.0
            eaten = gap / exp if exp else 0.0
            if eaten >= 0.6:
                tags.append(f"高开已吃预期{eaten:.0%}谨慎"); fp_parts.append("eaten")

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

    def _rank(self) -> list[tuple[str, float]]:
        """取 expected_return>0 的票，按预期降序 Top-N。缺 ref/预期的票落榜。

        排序主序恒为原始 ridge_pred（expected_return），校准只改展示不改序。
        """
        ref = self._ctx.all_refs() or {}
        rows: list[tuple[str, float]] = []
        for code, r in ref.items():
            exp = getattr(r, "expected_return", None)
            if exp is None or exp <= 0:
                continue
            rows.append((code, float(exp)))
        rows.sort(key=lambda kv: kv[1], reverse=True)
        return rows[: self._top_n]

    def _exp_str(self, code: str, exp: float) -> str:
        """展示用预期字符串：优先历史校准值 + 胜率，缺校准回退原始 ridge_pred。"""
        r = self._ctx.ref_of(code)
        cal = getattr(r, "calibrated_return", None) if r is not None else None
        wr = getattr(r, "win_rate", None) if r is not None else None
        if cal is not None:
            wr_str = f"(胜率{wr:.0%})" if wr is not None else ""
            return f"预期{cal:+.1%}{wr_str}"
        return f"预期{exp:+.1%}"

    def _render(self, ranked: list[tuple[str, float]]) -> tuple[str, str, str]:
        """把 Top-N 拼成 (title, body, fingerprint)。"""
        from datetime import datetime

        marks = "①②③④⑤⑥⑦⑧⑨⑩"
        lines: list[str] = []
        fp_rows: list[str] = []
        for i, (code, exp) in enumerate(ranked):
            tags, fp = self._tags(code, exp)
            mark = marks[i] if i < len(marks) else f"{i + 1}."
            tag_str = (" " + " ".join(tags)) if tags else ""
            lines.append(f"{mark} {self._label(code)} {self._exp_str(code, exp)}{tag_str}")
            fp_rows.append(f"{code}:{round(exp, 3)}:{fp}")
        title = f"[实时榜] 模型Top{len(ranked)}买入候选 {datetime.now():%H:%M}"
        body = "\n".join(lines)
        fingerprint = ";".join(fp_rows)
        return title, body, fingerprint
