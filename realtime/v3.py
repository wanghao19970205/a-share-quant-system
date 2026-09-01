"""V3 模拟盘：在 V1/V2 旁验证执行确认与 ATR 自适应出场。

V3 保持 V2 的模型候选池、动态资金分配、买入窗口、持仓上限和 T+1 口径，
只改变可明确归因的执行规则：当日预测 + 新鲜盘口确认后按 ask1 买入，按 bid1
卖出；无有效买一或封跌停时绝不虚拟成交；风险退出统一使用 ATR 风险单位。
"""
from __future__ import annotations

import datetime as _dt
import math
import time
from typing import Optional

from .paper_trader import _today
from .reference import _trading_days_between
from .strategy import _digits
from .v2 import V2PaperTrader


_EXIT_LABEL_V3 = {
    "atr_stop": "ATR止损",
    "atr_take_profit": "ATR止盈",
    "atr_trailing": "ATR移动止盈",
    "time_cap": "到期(T+N)",
}


class V3PaperTrader(V2PaperTrader):
    """V3 独立赛马账户：可成交报价入场/出场 + ATR 单位化风控。"""

    _FILE_SUFFIX = "_v3"
    _VERSION = 3
    _EVENT_TYPE = "paper_buy_decision_v3"
    _EVENT_ID_PREFIX = "paper-buy-decision-v3"
    _POSITION_ID_PREFIX = "paper-pos-v3"
    _PAPER_TITLE = "模拟盘V3"

    def __init__(self, cfg, ctx, notifier, name_map=None):
        super().__init__(cfg, ctx, notifier, name_map)
        self._quote_max_age = float(getattr(cfg, "paper_v3_quote_max_age_sec", 90.0))
        self._atr_k = float(getattr(cfg, "paper_v3_atr_k", 2.0))

    def _prefix(self) -> str:
        return "[paper_v3]"

    def _exit_label(self, reason: str) -> str:
        return _EXIT_LABEL_V3.get(reason, reason)

    def _prediction_date(self, code: str) -> Optional[str]:
        ref = self._ctx.ref_of(code) or self._ctx.ref_of(_digits(code))
        value = getattr(ref, "prediction_date", None) if ref is not None else None
        return str(value) if value else None

    def _quote(self, code: str) -> tuple:
        snap = self._ctx.snapshot_of(code)
        age = self._ctx.quote_age_of(code)
        if snap is None:
            return None, age, None, None

        def _positive(value) -> Optional[float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and number > 0 else None

        return snap, age, _positive(getattr(snap, "bid_price1", None)), _positive(
            getattr(snap, "ask_price1", None))

    def _price_of(self, code: str) -> Optional[float]:
        """V3 按可卖买一价估值；无买一时退回 None，由账户估值使用成本价。"""
        _, _, bid, _ = self._quote(code)
        return bid

    def _mark_detail(self, code: str) -> tuple[Optional[float], Optional[float], str]:
        """V3-V6 按可成交 bid1 估值，并同时记录盘口年龄。"""
        _, age, bid, _ = self._quote(code)
        return bid, age, "bid1"

    def _entry_quote_detail(self, code: str, exp: float) -> tuple[Optional[str], Optional[dict]]:
        """一次性读取并验证入场盘口，避免校验与记账跨越两条行情。"""
        prediction_date = self._prediction_date(code)
        if prediction_date != _today():
            return f"预测非当日({prediction_date or '缺失'})", None
        snap, age, bid, ask = self._quote(code)
        if age is None or age > self._quote_max_age:
            age_text = "缺失" if age is None else f"{age:.1f}s"
            return f"行情过期({age_text})", None
        if snap is None or bid is None or ask is None or ask < bid:
            return "买卖一报价无效", None
        if getattr(snap, "is_limit_up", False):
            return "封涨停不可买", None
        high_limit = getattr(snap, "high_limited", None)
        if high_limit and ask >= float(high_limit) - 0.001:
            return "卖一已到涨停价", None
        bid_volume = getattr(snap, "bid_volume1", None)
        ask_volume = getattr(snap, "ask_volume1", None)
        try:
            bid_volume = float(bid_volume)
            ask_volume = float(ask_volume)
        except (TypeError, ValueError):
            return "一档挂单量无效", None
        if (not math.isfinite(bid_volume) or not math.isfinite(ask_volume) or
                bid_volume <= 0 or ask_volume <= 0):
            return "一档挂单量无效", None
        imbalance = getattr(snap, "bid_ask_imbalance", None)
        try:
            imbalance = float(imbalance)
        except (TypeError, ValueError):
            return "买盘未确认(缺失)", None
        if not math.isfinite(imbalance) or imbalance < 0:
            return f"买盘未确认({imbalance})", None
        inherited = super()._entry_skip(code, exp, ask)
        if inherited:
            return inherited, None
        return None, {
            "snap": snap, "age": age, "bid": bid, "ask": ask,
            "bid_volume1": float(bid_volume), "ask_volume1": float(ask_volume),
        }

    def _entry_quote(self, code: str, exp: float) -> tuple[Optional[str], Optional[float]]:
        """兼容测试/诊断入口：返回过滤原因和可执行卖一价。"""
        reason, quote = self._entry_quote_detail(code, exp)
        return reason, (quote["ask"] if quote is not None else None)

    def _market_audit(self, code: str) -> dict:
        audit = super()._market_audit(code)
        snap, age, bid, ask = self._quote(code)
        audit.update({
            "quote_age_sec": age,
            "prediction_date": self._prediction_date(code),
            "bid_price1": bid,
            "ask_price1": ask,
            "bid_volume1": getattr(snap, "bid_volume1", None) if snap else None,
            "ask_volume1": getattr(snap, "ask_volume1", None) if snap else None,
        })
        return audit

    def _entry_extra(self, code: str, quote: dict) -> dict:
        """子版本可追加持仓与买入审计字段；V3 默认无附加项。"""
        return {}

    def _paper_config_extra(self) -> dict:
        """子版本可追加独立参数快照；V3 默认无附加项。"""
        return {}

    def _trade_extra(self, pos: dict) -> dict:
        """子版本可把入场上下文复制到卖出流水；V3 默认无附加项。"""
        return {}

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
        target_n = min(self._buy_n, free_slots)
        trace: list[dict] = []
        ranked_rows = self._scorer.ranked(
            exclude=held_codes, require_price=True, drop_limit_up=True, trace=trace)
        trace_by_code = {rec["code"]: rec for rec in trace}
        t = self._now_hhmm if self._now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        past_window = t > self._buy_end
        remaining_slots = target_n
        bought: list[str] = []
        for row in ranked_rows:
            code, exp = row.code, row.exp
            audit = trace_by_code.get(_digits(code), {})
            if len(bought) >= target_n:
                break
            if past_window:
                audit["entry_decision"] = "past_buy_window"
                continue
            skip, quote = self._entry_quote_detail(code, exp)
            if skip or quote is None:
                audit.update({"entry_decision": "filtered", "entry_filter_reason": skip})
                print(f"{self._prefix()} 跳过候选 {code}：{skip}", flush=True)
                continue
            fill_price = quote["ask"]
            budget, allocation = self._allocation_budget(
                code, exp, fill_price, remaining_slots)
            audit["allocation"] = allocation
            if budget <= 0:
                audit.update({"entry_decision": "allocation_blocked",
                              "entry_filter_reason": allocation.get(
                                  "allocation_factor_reason") or "风险预算为零"})
                continue
            shares = int(budget / (fill_price * (1 + self._cost / 2.0)) // 100) * 100
            if shares <= 0:
                audit["entry_decision"] = "insufficient_lot_cash"
                continue
            cost_basis = fill_price * shares * (1 + self._cost / 2.0)
            if cost_basis > self._state.get("cash", 0.0):
                audit["entry_decision"] = "insufficient_cash"
                continue
            buy_time = time.strftime("%H:%M:%S")
            position_id = self._stable_id(
                self._POSITION_ID_PREFIX, code, _today(), buy_time, round(fill_price, 3), shares)
            age, bid = quote["age"], quote["bid"]
            entry_atr = self._atr_of(code)
            self._state["cash"] -= cost_basis
            position = {
                "position_id": position_id,
                "code": code, "name": self._name_map.get(_digits(code), ""),
                "buy_date": _today(), "buy_time": buy_time,
                "buy_price": round(fill_price, 3), "shares": shares,
                "peak": round(fill_price, 3), "peak_bid": round(bid or fill_price, 3),
                "cost_basis": round(cost_basis, 2), "exp": round(exp, 4),
                "entry_atr": entry_atr, "prediction_date": self._prediction_date(code),
                "buy_quote_age_sec": age, "buy_fill_source": "ask1",
                "buy_ask_volume1_raw": quote["ask_volume1"],
                # 与卖出侧同理：留下成交瞬间的 last/bid1/ask1 才能事后拆分价差与漂移。
                # 注意 V3 的 _price_of 已被改成返回 bid1，取 last 必须走基类实现。
                "buy_last": super()._price_of(code), "buy_bid1": bid,
                "buy_ask1": quote.get("ask"),
            }
            position.update(self._entry_extra(code, quote))
            self._state.setdefault("positions", []).append(position)
            remaining_slots -= 1
            audit.update({
                "entry_decision": "bought", "position_id": position_id,
                "shares": shares, "fill_price": round(fill_price, 3),
                "fill_source": "ask1", "quote_age_sec": age,
                "quoted_ask_volume1_raw": quote["ask_volume1"], "entry_atr": entry_atr,
                "allocated_cash": budget, "cost_basis": round(cost_basis, 2),
                **self._entry_extra(code, quote),
            })
            bought.append(
                f"{self._label(code)} @{fill_price:.2f} {self._exp_str(code, exp)} {shares}股")
            if len(self._state["positions"]) >= self._max_positions:
                break
        self._finish_buy_attempt(len(bought))
        self._save_state()
        self._record_buy_decision(trace, account_before, len(bought), target_n=target_n)
        if bought:
            self._notifier.push(
                f"[{self._PAPER_TITLE}] 买入 {len(bought)}只 {_dt.datetime.now():%m-%d %H:%M}",
                "\n".join(f"{'①②③④⑤'[i] if i < 5 else i + 1} {b}"
                          for i, b in enumerate(bought)) + f"\n{self.summary()}")

    def _record_buy_decision(self, candidates: list[dict], account_before: dict,
                             bought_count: int, target_n: Optional[int] = None) -> None:
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
        stage = self._current_buy_stage or "direct"
        paper_config = {
            "buy_n": self._buy_n, "target_n": target,
            "buy_start": self._buy_start, "buy_retry_start": self._buy_retry_start,
            "buy_end": self._buy_end, "time_cap_start": self._time_cap_start,
            "cost_roundtrip": self._cost, "max_positions": self._max_positions,
            "quote_max_age_sec": self._quote_max_age, "atr_k": self._atr_k,
            "entry_imbalance_min": 0.0, "fill_buy": "ask1", "fill_sell": "bid1",
            "rank_pool_n": getattr(self._cfg, "rank_pool_n", 30),
            "rank_min_raw_return": getattr(self._cfg, "rank_min_raw_return", 0.0),
            "rank_raw_safety_margin": getattr(self._cfg, "rank_raw_safety_margin", 0.001),
            "entry_rich": self._entry_rich,
            "entry_spread": self._entry_spread,
            "risk_per_trade": self._risk_per_trade,
            "max_position_weight": self._max_position_weight,
            "allocation_atr_k": self._allocation_atr_k,
            "allocation_target_return": self._allocation_target_return,
        }
        paper_config.update(self._paper_config_extra())
        record = {
            "schema_version": self._VERSION, "event_type": self._EVENT_TYPE,
            "event_id": f"{self._EVENT_ID_PREFIX}:{_today()}:{stage}",
            "trade_date": _today(), "attempt_stage": stage,
            "decision_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decision_status": status,
            "paper_config": paper_config,
            "account_before": account_before,
            "account_after": {
                "cash": self._state.get("cash", 0.0),
                "position_count": len(self._state.get("positions", [])),
                "bought_count": bought_count,
            },
            "candidates": candidates,
        }
        self._upsert_jsonl(self._decisions_file, "event_id", record)

    def _exit_decision(self, pos: dict, px: float, held: int,
                       allow_time_cap: bool = True) -> Optional[str]:
        entry = float(pos.get("buy_price") or 0.0)
        atr = pos.get("entry_atr") or self._atr_of(pos["code"])
        peak_bid = float(pos.get("peak_bid") or pos.get("peak") or entry)
        if entry > 0 and atr and atr > 0:
            risk = self._atr_k * atr
            if px <= entry - risk:
                return "atr_stop"
            if px >= entry + 2.0 * risk:
                return "atr_take_profit"
            if peak_bid >= entry + risk and px <= peak_bid - risk:
                return "atr_trailing"
        if allow_time_cap and held >= self._horizon:
            return "time_cap"
        return None

    def _set_blocked(self, pos: dict, reason: str) -> bool:
        marker = {"date": _today(), "reason": reason}
        if pos.get("v3_sell_blocked") == marker:
            return False
        pos["v3_sell_blocked"] = marker
        print(f"{self._prefix()} 暂缓卖出 {pos['code']}：{reason}", flush=True)
        return True

    def _run_sells(self, now_hhmm: Optional[int] = None) -> None:
        t = 1500 if now_hhmm is None else int(now_hhmm)
        allow_time_cap = t >= self._time_cap_start
        today = _dt.date.today()
        changed = False
        sold = False
        for pos in list(self._state.get("positions", [])):
            snap, age, bid, _ = self._quote(pos["code"])
            if age is None or age > self._quote_max_age:
                changed = self._set_blocked(pos, "行情过期") or changed
                continue
            if snap is None or bid is None:
                changed = self._set_blocked(pos, "无有效买一价") or changed
                continue
            bid_volume = getattr(snap, "bid_volume1", None)
            if bid_volume is None or bid_volume <= 0:
                reason = "跌停无买盘承接" if getattr(snap, "is_limit_down", False) else "买一挂单量无效"
                changed = self._set_blocked(pos, reason) or changed
                continue
            if pos.pop("v3_sell_blocked", None) is not None:
                changed = True
            prev_peak = float(pos.get("peak_bid") or pos.get("buy_price", 0.0))
            if bid > prev_peak:
                pos["peak_bid"] = round(bid, 3)
                pos["peak"] = round(bid, 3)
                changed = True
            buy_d = self._parse_date(pos.get("buy_date"))
            held = _trading_days_between(buy_d, today) if buy_d else 0
            if held < 1:
                continue
            reason = self._exit_decision(pos, bid, held, allow_time_cap=allow_time_cap)
            if reason is None:
                continue
            self._remove_and_settle(pos, bid, held, reason, bid_volume=float(bid_volume))
            changed = True
            sold = True
        if changed:
            self._save_state()
        if sold:
            self._refresh_counterfactuals()

    def _remove_and_settle(self, pos: dict, px: float, held: int, reason: str,
                           bid_volume: Optional[float] = None) -> None:
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
        # 成交瞬间的最新价与买卖一价：回测按收盘计价，实盘按 bid1/ask1 成交，
        # 两者之差既含买卖价差也含 14:50 到收盘的漂移。只有把 last 一起记下来，
        # 事后才能把这两个成分拆开——否则成本假设只能靠猜。
        snap, sell_age, sell_bid, sell_ask = self._quote(pos["code"])
        trade_rec = {
            "action": "sell", "time": sell_time,
            "trade_id": str(pos.get("position_id") or self._stable_id(
                self._POSITION_ID_PREFIX, pos["code"], pos.get("buy_date"),
                pos.get("buy_time"), pos["buy_price"], pos["shares"])),
            "code": pos["code"], "name": self._name_map.get(_digits(pos["code"]), ""),
            "buy_date": pos.get("buy_date"), "buy_time": pos.get("buy_time"),
            "buy_price": pos["buy_price"], "sell_date": _today(),
            "sell_price": round(px, 3), "sell_fill_source": "bid1",
            "sell_bid_volume1_raw": bid_volume, "held_days": held,
            "sell_last": super()._price_of(pos["code"]), "sell_bid1": sell_bid,
            "sell_ask1": sell_ask, "sell_quote_age_sec": sell_age,
            "exit_reason": reason, "peak": pos.get("peak_bid"), "exp": pos.get("exp"),
            "entry_atr": pos.get("entry_atr"), "prediction_date": pos.get("prediction_date"),
            "shares": pos["shares"], "pnl": round(pnl, 2), "return": round(ret, 4),
            "equity_after": round(self._equity(), 2),
            **self._trade_extra(pos),
        }
        self._append_trade(trade_rec)
        self._notifier.push(
            f"[{self._PAPER_TITLE}] 卖出 {self._label(pos['code'])} @{px:.2f} "
            f"{_EXIT_LABEL_V3.get(reason, reason)}",
            self._sell_notice(pos, px, held, pnl, ret, reason, proceeds))
