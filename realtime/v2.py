"""V2 模拟盘（PaperTrader 优化版）：在原账户旁并行运行，独立状态与流水。

与 V1（paper_trader.PaperTrader）同引擎、同 ctx、同 notifier，但：
  - 读取不同的状态/流水/审计文件（_v2 后缀），起始资金一致；
  - 保持模型的 close→close 口径和风险参数默认值，
    同时在买入分配、卖出跌停、保护性止盈等七项上做定向优化。

设计意图是"赛马"：V1 不动，V2 并行积累新样本，前向比较两版净值和损益结构。
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Optional

from .paper_trader import PaperTrader as _V1, _EXIT_LABEL, _today
from .reference import _trading_days_between
from .strategy import _digits


# V2 新增退出原因（保留 V1 全部五级，只加三项）。
_EXIT_LABEL_V2 = {**_EXIT_LABEL,
                  "breakeven_stop": "保本止盈",
                  "limit_open": "炸板卖出",
                  "limit_down_blocked": "跌停阻塞(顺延)",
                  }


class V2PaperTrader(_V1):
    """V2 模拟盘，继承 V1 全部状态/估值/审计/反事实，仅覆盖关键决策方法。"""

    # 声明后缀即让父类在任何 I/O 前把四个账户文件定向到 _v2 副本。
    _FILE_SUFFIX = "_v2"

    def __init__(self, cfg, ctx, notifier, name_map=None):
        super().__init__(cfg, ctx, notifier, name_map)
        # V2 新增参数（在 super().__init__ 之后读取，cfg 为不可变 dataclass）。
        self._max_positions = int(getattr(cfg, "paper_max_positions", 4))
        self._buy_end = max(1455, min(int(getattr(cfg, "paper_buy_end", 1457)), 1500))
        self._breakeven_arm = float(getattr(cfg, "paper_breakeven_arm", 0.03))
        self._breakeven_margin = float(getattr(cfg, "paper_breakeven_margin", 0.005))
        self._take_profit_tighten = float(getattr(
            cfg, "paper_take_profit_tighten", 0.03))
        self._limit_down_roll_max = int(getattr(cfg, "paper_limit_down_roll_max", 3))
        self._now_hhmm: Optional[int] = None  # 本轮心跳时刻，由 maybe_trade 注入

    # ── 标记 ---------------------------------------------------------------
    def _prefix(self) -> str:
        return "[paper_v2]"

    def maybe_trade(self, now_hhmm: Optional[int] = None) -> None:
        """记录本轮时刻后交给 V1 主流程，使买窗判断与卖出腿共用同一时间源。"""
        self._now_hhmm = now_hhmm if now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        super().maybe_trade(now_hhmm)

    # ── 卖出：跌停阻塞 + 保护性止盈 + 炸板 + 时间加权止盈 ──────────────
    def _exit_decision(self, pos: dict, px: float, held: int,
                       allow_time_cap: bool = True) -> Optional[str]:
        buy_price = pos.get("buy_price") or 0.0
        ret = (px / buy_price - 1.0) if buy_price else 0.0
        peak = pos.get("peak") or buy_price
        has_profit = peak > buy_price

        # 1) 硬止损（同 V1）
        if self._stop_loss > 0 and ret <= -self._stop_loss:
            return "stop_loss"

        # 1.5) 保护性止盈：浮盈达到 arm 后，跌破买入价附近即退出
        if has_profit and self._breakeven_arm > 0:
            arm_price = round(buy_price * (1 + self._breakeven_arm), 8)
            margin_price = round(buy_price * (1 + self._breakeven_margin), 8)
            if peak >= arm_price and round(px, 8) <= margin_price:
                return "breakeven_stop"

        # 2) 止盈上限 + 时间加权：持有多日后按紧缩量收窄止盈阈值
        tp = self._take_profit
        if tp > 0 and held >= 2 and self._take_profit_tighten > 0:
            tp = max(0.005, self._take_profit - self._take_profit_tighten * (held - 1))
        if tp > 0 and ret >= tp:
            return "take_profit"

        # 3) 移动止盈（ATR 吊灯，同 V1）：仅在曾有浮盈时启用
        atr = self._atr_of(pos["code"])
        if atr and has_profit:
            if px <= peak - self._trail_k * atr:
                return "trailing_stop"

        # 4) VWAP 破位（同 V1）
        if self._vwap_break > 0:
            vwap = self._ctx.vwap_of(pos["code"])
            if vwap and vwap > 0 and px < vwap * (1 - self._vwap_break):
                return "vwap_break"

        # 5) time_cap（仅 14:50 后）
        if allow_time_cap and held >= self._horizon:
            return "time_cap"
        return None

    def _run_sells(self, now_hhmm: Optional[int] = None) -> None:
        """同 V1 但增加跌停阻塞 + 炸板卖出的前置判断。"""
        t = 1500 if now_hhmm is None else int(now_hhmm)
        allow_time_cap = t >= self._time_cap_start
        today = _dt.date.today()
        changed = False
        sold = False
        for pos in list(self._state.get("positions", [])):
            px = self._price_of(pos["code"])
            if px is None:
                continue
            prev_peak = pos.get("peak") or pos.get("buy_price", 0.0)
            if px > prev_peak:
                pos["peak"] = round(px, 3)
                changed = True
            buy_d = self._parse_date(pos.get("buy_date"))
            held = _trading_days_between(buy_d, today) if buy_d else 0
            if held < 1:
                continue

            snap = self._ctx.snapshot_of(pos["code"])

            # V2 新增：跌停阻塞 —— 封跌停时保留持仓待次日
            if snap is not None and getattr(snap, "is_limit_down", False):
                # 顺延按交易日计数，不按心跳：同日多次心跳只阻塞、不重复累加。
                today_str = _today()
                if pos.get("_ld_date") != today_str:
                    pos["_ld_date"] = today_str
                    pos["_ld_rolls"] = pos.get("_ld_rolls", 0) + 1
                    changed = True
                rolls = pos.get("_ld_rolls", 1)
                if rolls <= self._limit_down_roll_max:
                    print(f"{self._prefix()} 跌停阻塞 {pos['code']}（"
                          f"顺延第{rolls}日）", flush=True)
                    continue
                print(f"{self._prefix()} 跌停强制平仓 {pos['code']}（已顺延{rolls}日）",
                      flush=True)
            elif pos.pop("_ld_rolls", None) is not None:
                pos.pop("_ld_date", None)
                changed = True

            # V2 新增：炸板退出 —— 曾封涨停后开板且有一定浮盈
            if snap is not None and self._take_profit > 0:
                ever_limit_up = self._ctx.ever_limit_up_of(pos["code"])
                if ever_limit_up and not getattr(snap, "is_limit_up", False):
                    arm_price = pos.get("buy_price", 0.0) * (1 + self._breakeven_arm)
                    if pos.get("peak", 0.0) >= arm_price:
                        self._remove_and_settle(pos, px, held, "limit_open")
                        changed = True
                        sold = True
                        continue

            reason = self._exit_decision(pos, px, held, allow_time_cap=allow_time_cap)
            if reason is None:
                continue
            self._remove_and_settle(pos, px, held, reason)
            changed = True
            sold = True
        if changed:
            self._save_state()
        if sold:
            self._refresh_counterfactuals()

    def _remove_and_settle(self, pos: dict, px: float, held: int, reason: str) -> None:
        try:
            self._state["positions"].remove(pos)
        except ValueError:
            pass
        buy_price = pos.get("buy_price", 0.0)
        proceeds = px * pos["shares"] * (1 - self._cost / 2.0)
        cost_basis = pos.get("cost_basis", buy_price * pos["shares"])
        pnl = proceeds - cost_basis
        ret = (pnl / cost_basis) if cost_basis else 0.0
        self._state["cash"] = self._state.get("cash", 0.0) + proceeds
        self._state["realized_pnl"] = self._state.get("realized_pnl", 0.0) + pnl
        sell_time = time.strftime("%Y-%m-%d %H:%M:%S")
        trade_rec = {
            "action": "sell", "time": sell_time,
            "trade_id": str(pos.get("position_id") or self._stable_id(
                "paper-pos", pos["code"], pos.get("buy_date"),
                pos.get("buy_time"), pos["buy_price"], pos["shares"])),
            "code": pos["code"],
            "name": self._name_map.get(_digits(pos["code"]), ""),
            "buy_date": pos.get("buy_date"), "buy_time": pos.get("buy_time"),
            "buy_price": pos["buy_price"], "sell_date": _today(),
            "sell_price": round(px, 3), "held_days": held,
            "exit_reason": reason, "peak": pos.get("peak"),
            "exp": pos.get("exp"),
            "shares": pos["shares"], "pnl": round(pnl, 2),
            "return": round(ret, 4),
            "equity_after": round(self._equity(), 2),
        }
        self._append_trade(trade_rec)
        self._notifier.push(
            f"[模拟盘V2] 卖出 {self._label(pos['code'])} @{px:.2f} "
            f"{_EXIT_LABEL_V2.get(reason, reason)}",
            f"持有T+{held} 收益{ret:+.2%} 盈亏¥{pnl:,.0f}\n{self.summary()}")

    # ── 买入：动态分配 + 上限 + 买窗到 14:57 ────────────────────────────
    def _run_buys(self) -> None:
        if self._state.get("last_buy_date") == _today():
            return
        positions = self._state.get("positions", [])
        free_slots = self._max_positions - len(positions)
        if free_slots <= 0:
            self._state["last_buy_date"] = _today()
            self._save_state()
            return
        held_codes = {_digits(p["code"]) for p in positions}
        account_before = {
            "cash": self._state.get("cash", 0.0),
            "position_count": len(positions), "held_codes": sorted(held_codes),
        }
        # 目标只数受持仓上限收缩，避免接近上限时按 buy_n 均分而闲置现金。
        target_n = min(self._buy_n, free_slots)
        trace: list[dict] = []
        ranked_rows = self._scorer.ranked(
            exclude=held_codes, require_price=True, drop_limit_up=True, trace=trace)
        trace_by_code = {rec["code"]: rec for rec in trace}
        # 买窗判断用本轮心跳时刻（maybe_trade 注入），缺省回退系统时钟。
        t = self._now_hhmm if self._now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        past_window = t > self._buy_end
        remaining_slots = target_n
        bought: list[str] = []
        for row in ranked_rows:
            code, exp, px = row.code, row.exp, row.px
            audit = trace_by_code.get(_digits(code), {})
            if len(bought) >= target_n:
                break
            if past_window:
                audit["entry_decision"] = "past_buy_window"
                continue
            skip = self._entry_skip(code, exp, px)
            if skip:
                audit.update({"entry_decision": "filtered", "entry_filter_reason": skip})
                print(f"{self._prefix()} 跳过候选 {code}：{skip}", flush=True)
                continue
            # V2 动态分配：剩余资金 / 剩余名额
            alloc = self._state.get("cash", 0.0) / max(1, remaining_slots)
            budget = min(alloc, self._state.get("cash", 0.0))
            shares = int(budget / (px * (1 + self._cost / 2.0)) // 100) * 100
            if shares <= 0:
                audit["entry_decision"] = "insufficient_lot_cash"
                continue
            cost_basis = px * shares * (1 + self._cost / 2.0)
            if cost_basis > self._state.get("cash", 0.0):
                audit["entry_decision"] = "insufficient_cash"
                continue
            buy_time = time.strftime("%H:%M:%S")
            position_id = self._stable_id(
                "paper-pos-v2", code, _today(), buy_time, round(px, 3), shares)
            self._state["cash"] -= cost_basis
            self._state.setdefault("positions", []).append({
                "position_id": position_id,
                "code": code, "name": self._name_map.get(_digits(code), ""),
                "buy_date": _today(), "buy_time": buy_time,
                "buy_price": round(px, 3), "shares": shares, "peak": round(px, 3),
                "cost_basis": round(cost_basis, 2), "exp": round(exp, 4)})
            remaining_slots -= 1
            audit.update({
                "entry_decision": "bought", "position_id": position_id,
                "shares": shares, "fill_price": round(px, 3),
                "allocated_cash": alloc, "cost_basis": round(cost_basis, 2),
            })
            bought.append(
                f"{self._label(code)} @{px:.2f} {self._exp_str(code, exp)} {shares}股")
            if len(self._state["positions"]) >= self._max_positions:
                break
        self._state["last_buy_date"] = _today()
        self._save_state()
        self._record_buy_decision(trace, account_before, len(bought),
                                  target_n=target_n)
        if bought:
            self._notifier.push(
                f"[模拟盘V2] 买入 {len(bought)}只 {_dt.datetime.now():%m-%d %H:%M}",
                "\n".join(f"{'①②③④⑤'[i] if i < 5 else i + 1} {b}"
                          for i, b in enumerate(bought)) + f"\n{self.summary()}")

    # ── 审计：记录 V2 特有配置 ───────────────────────────────────────────
    def _record_buy_decision(self, candidates: list[dict], account_before: dict,
                             bought_count: int, target_n: Optional[int] = None) -> None:
        """写 V2 决策快照；target_n 为受持仓上限收缩后的当轮目标只数。"""
        target = self._buy_n if target_n is None else int(target_n)
        for rec in candidates:
            rec["name"] = self._name_map.get(_digits(rec.get("code", "")), "")
            rec["market"] = self._market_audit(rec.get("code", ""))
        eligible = [r for r in candidates if r.get("status") == "eligible_ranked"]
        if bought_count:
            status = "bought" if bought_count >= target else "partial_fill"
        elif eligible and all(r.get("entry_decision") == "filtered" for r in eligible):
            status = "all_candidates_filtered"
        elif eligible:
            status = "insufficient_cash_or_lot"
        else:
            status = "no_ranked_candidate"
        record = {
            "schema_version": 2, "event_type": "paper_buy_decision_v2",
            "event_id": f"paper-buy-decision-v2:{_today()}",
            "trade_date": _today(), "decision_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decision_status": status,
            "paper_config": {
                "buy_n": self._buy_n, "target_n": target,
                "buy_start": self._buy_start,
                "buy_end": self._buy_end,
                "time_cap_start": self._time_cap_start, "cost_roundtrip": self._cost,
                "max_positions": self._max_positions,
                "breakeven_arm": self._breakeven_arm,
                "breakeven_margin": self._breakeven_margin,
                "take_profit": self._take_profit,
                "take_profit_tighten": self._take_profit_tighten,
                "limit_down_roll_max": self._limit_down_roll_max,
                "rank_pool_n": getattr(self._cfg, "rank_pool_n", 30),
                "rank_min_net_return": getattr(self._cfg, "rank_min_net_return", 0.0),
                "entry_rich": self._entry_rich,
                "entry_ask_strong": self._entry_ask_strong,
                "entry_spread": self._entry_spread,
            },
            "account_before": account_before,
            "account_after": {
                "cash": self._state.get("cash", 0.0),
                "position_count": len(self._state.get("positions", [])),
                "bought_count": bought_count,
            },
            "candidates": candidates,
        }
        self._upsert_jsonl(self._decisions_file, "event_id", record)
