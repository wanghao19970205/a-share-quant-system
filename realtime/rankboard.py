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

from datetime import datetime
import math
import re
import time
from typing import Optional

from .notifier import Notifier
from .reference import (expected_return_text, load_actual_cash,
                        load_actual_holdings, net_return_after_cost)
from .rerank import RerankScorer
from .strategy import StrategyContext


class RankBoard:
    def __init__(self, cfg, ctx: StrategyContext, notifier: Notifier,
                 name_map: Optional[dict] = None, sector_ctx=None):
        self._cfg = cfg
        self._ctx = ctx
        self._notifier = notifier
        self._name_map = name_map or {}
        self._sector_ctx = sector_ctx
        self._actual_holdings = load_actual_holdings(cfg)
        self._actual_cash = load_actual_cash(cfg)
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

    @staticmethod
    def _compact_items(value, limit: int = 2, item_chars: int = 10) -> list[str]:
        """把本地元数据压成少量短标签，避免 Top5 推送正文过长。"""
        text = "" if value is None else str(value).strip()
        if not text or text.lower() in {"nan", "<na>"}:
            return []
        items: list[str] = []
        for raw in re.split(r"[、,，;；|/]+", text):
            item = raw.strip()
            if item and item not in items:
                items.append(item[:item_chars])
            if len(items) >= limit:
                break
        return items

    def _stock_meta_str(self, code: str) -> str:
        """返回内存中的行业/概念展示串；元数据缺失时保持原榜单格式。"""
        if self._sector_ctx is None:
            return ""
        try:
            meta = self._sector_ctx.assessment_for_stock(code)
        except Exception:  # noqa: BLE001 - 展示元数据异常不能阻断 RankBoard
            return ""
        primary = self._compact_items(meta.get("stock_industry"), limit=1, item_chars=12)
        hierarchy = self._compact_items(meta.get("stock_industries"), limit=3, item_chars=12)
        industries = list(primary)
        for item in hierarchy:
            if item not in industries:
                industries.append(item)
            if len(industries) >= 2:
                break
        concepts = self._compact_items(meta.get("stock_concepts"), limit=2, item_chars=10)
        parts = []
        if industries:
            parts.append(f"细分行业:{'/'.join(industries)}")
        if concepts:
            parts.append(f"概念:{'/'.join(concepts)}")
        sector = meta.get("sector")
        etf_code = meta.get("etf_code")
        if sector and etf_code:
            parts.append(f"比较ETF:{sector}({etf_code})")
        return (" [" + " ".join(parts) + "]") if parts else ""

    def _actual_advice(self) -> tuple[str, str]:
        """生成实际账户的只读调仓提示；不调用任何交易或模拟盘接口。"""
        if (not getattr(self._cfg, "actual_advice_enabled", True) or
                not self._actual_holdings):
            return "", ""
        rows = []
        total_value = self._actual_cash
        for code, holding in self._actual_holdings.items():
            snap = self._ctx.snapshot_of(code)
            price = getattr(snap, "last", None) if snap is not None else None
            value = (float(price) * holding.shares
                     if price is not None and price > 0 and holding.shares else None)
            if value is not None:
                total_value += value
            rows.append((code, holding, snap, price, value))

        lines = ["[实际持仓怎么处理（只提醒，不会自动买卖）]"]
        fingerprints = []
        max_weight = float(getattr(self._cfg, "actual_advice_max_weight", 0.35))
        profit_lock = float(getattr(self._cfg, "actual_advice_profit_lock", 0.15))
        loss_review = float(getattr(self._cfg, "actual_advice_loss_review", -0.15))
        for code, holding, snap, price, value in rows:
            name = holding.name or self._name_map.get(code, "")
            label = f"{code} {name}".strip()
            if price is None or price <= 0 or not holding.shares:
                lines.append(f"- {label}：暂时不操作，还没有收到实时价格或持股数量。")
                fingerprints.append(f"{code}:wait")
                continue
            weight = value / total_value if total_value > 0 and value is not None else None
            pnl = ((float(price) / holding.cost - 1.0)
                   if holding.cost is not None and holding.cost > 0 else None)
            ref = self._ctx.ref_of(code)
            expected = getattr(ref, "expected_return", None) if ref is not None else None
            expected_net = (net_return_after_cost(
                float(expected), getattr(self._cfg, "paper_cost", 0.002))
                if expected is not None else None)
            sector = (self._sector_ctx.assessment_for_stock(code)
                      if self._sector_ctx is not None else {})
            sector_status = sector.get("status")
            too_large = weight is not None and weight > max_weight
            profit_high = pnl is not None and pnl >= profit_lock
            loss_large = pnl is not None and pnl <= loss_review
            model_weak = expected_net is not None and expected_net <= 0
            sector_weak = sector_status == "weak"

            reduce_shares = 0
            if too_large:
                target_value = total_value * max_weight
                excess_value = max(0.0, value - target_value)
                reduce_shares = int(math.ceil(excess_value / float(price) / 100.0) * 100)
                reduce_shares = min(reduce_shares, holding.available or holding.shares)

            if reduce_shares > 0:
                action = (f"建议卖出约{reduce_shares}股，把这只股票降到总资产的"
                          f"{max_weight:.0%}左右。")
            elif loss_large and (model_weak or sector_weak):
                action = "这只已经亏得比较多，短期信号也偏弱。建议考虑卖出一部分或全部，暂时不要补仓。"
            elif loss_large:
                action = "这只已经亏得比较多，先不要补仓；如果不想继续承担下跌，建议卖出一部分。"
            elif profit_high:
                action = "这只已经赚得比较多，建议卖出一部分先把利润落袋，剩余继续观察。"
            elif model_weak or sector_weak:
                action = "暂时不要加仓；如果下午继续走弱，建议卖出一部分。"
            else:
                action = "可以继续持有，暂时不用调整。"

            facts = []
            if weight is not None:
                facts.append(f"占总资产{weight:.1%}")
            if pnl is not None:
                verb = "赚" if pnl >= 0 else "亏"
                facts.append(f"相对成本{verb}{abs(pnl):.1%}")
            if expected_net is not None:
                facts.append(f"模型预计下一交易日扣成本后{expected_net:+.1%}")
            if sector_weak:
                facts.append("所属行业今天偏弱")
            primary = self._compact_items(
                sector.get("stock_industry"), limit=1, item_chars=12)
            hierarchy = self._compact_items(
                sector.get("stock_industries"), limit=3, item_chars=12)
            industry_labels = list(primary)
            for item in hierarchy:
                if item not in industry_labels:
                    industry_labels.append(item)
                if len(industry_labels) >= 2:
                    break
            if industry_labels:
                facts.append(f"细分行业:{'/'.join(industry_labels)}")
            compare_sector = sector.get("sector")
            compare_etf = sector.get("etf_code")
            if compare_sector and compare_etf:
                facts.append(f"走势比较:{compare_sector}ETF({compare_etf})")
            updated_at = sector.get("stock_meta_updated_at")
            if updated_at:
                try:
                    updated_date = datetime.fromisoformat(str(updated_at)).date()
                    if (datetime.now().date() - updated_date).days > 7:
                        facts.append(
                            f"行业资料更新于{updated_date:%m/%d}，可能漏掉近期热点")
                except (TypeError, ValueError):
                    pass
            lines.append(f"- {label}：{action}\n  原因：{'，'.join(facts)}")
            fingerprints.append(
                f"{code}:{action}:w{round((weight or 0) / 0.05)}:p{round((pnl or 0) / 0.05)}")
        return "\n".join(lines), ";".join(fingerprints)

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
        # 榜单必须基于本进程已收到的实时价；启动/热重载后无快照的“待开盘”票
        # 无法判定是否封板，不能先放行再等下一轮纠正。
        rows = self._scorer.ranked(require_price=True, drop_limit_up=True)
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
        marks = "①②③④⑤⑥⑦⑧⑨⑩"
        lines: list[str] = []
        fp_rows: list[str] = []
        for i, row in enumerate(ranked):
            code, exp = row.code, row.exp
            tags, fp = self._tags(code, exp)
            mark = marks[i] if i < len(marks) else f"{i + 1}."
            tag_str = (" " + " ".join(tags)) if tags else ""
            meta_str = self._stock_meta_str(code)
            lines.append(f"{mark} {self._label(code)}{meta_str} {self._exp_str(code, exp)}"
                         f"{self._rerank_str(row)}{tag_str}")
            # 指纹并入 adj 粗分桶（0.05 一档），使盘中重排变化能触发推送但不因微抖动刷屏。
            adj_bucket = round(getattr(row, "adj", 0.0) / 0.05)
            fp_rows.append(f"{code}:{round(exp, 3)}:a{adj_bucket}:{fp}")
        pool_n = getattr(self._cfg, "rank_pool_n", 30)
        title = f"[实时榜] 重排Top{len(ranked)}（模型Top{pool_n}池） {datetime.now():%H:%M}"
        body = "\n".join(lines)
        advice, advice_fingerprint = self._actual_advice()
        if advice:
            body = f"{body}\n\n{advice}"
        fingerprint = ";".join(fp_rows) + f"|actual:{advice_fingerprint}"
        return title, body, fingerprint
