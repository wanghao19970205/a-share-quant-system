"""V7 模拟盘：等权低波动带账户。

与 V1-V6 的取向刻意相反。V1-V6 是"少而精 + 主动风控"——按模型预期收益选 4 只、
ATR 风险定仓、止损/止盈/移动止盈层层设防。V7 是"多而散 + 被动分位带"：不看模型，
只按 60 日波动率的截面分位进出，等权持有 20 只，除了分位带之外没有任何出场规则。

研究依据见 ``realtime/vol_band.py`` 模块说明。两个必须保留的设计约束：

1. **不叠加 V2 的保本/动态止盈与 V3 的 ATR 出场。** 这条策略的超额来自极低换手
   （研究口径日均 0.033），任何额外出场规则都会把换手推上去，那就不是被验证过的
   那个策略了。也因此 V7 不使用 ``time_cap``——研究规则没有持有期上限。
2. **等权而不是风险定仓。** 研究结论建立在等权之上，换成 ATR 风险定仓结果不转移。

持仓数定 20 而不是研究最优的 50：10 万本金下每只 5000 元，带内 96.9% 的标的买得起
1 手，整手取整后实际投入/目标中位 0.92；n=50 每只只有 2000 元，17% 的标的买不起，
会引入研究里不存在的价格倾斜。
"""
from __future__ import annotations

import datetime as _dt
import math
import time
from typing import Optional

from . import vol_band
from .paper_trader import PaperTrader, _today
from .strategy import _digits
from .v3 import V3PaperTrader

_EXIT_LABEL_V7 = {"vol_band_exit": "跌出波动率分位带"}


class V7PaperTrader(V3PaperTrader):
    _FILE_SUFFIX = "_v7"
    _VERSION = 7
    _EVENT_TYPE = "paper_buy_decision_v7"
    _EVENT_ID_PREFIX = "paper-buy-decision-v7"
    _POSITION_ID_PREFIX = "paper-pos-v7"
    _PAPER_TITLE = "模拟盘V7"

    def __init__(self, cfg, ctx, notifier, *args, **kwargs):
        super().__init__(cfg, ctx, notifier, *args, **kwargs)
        self._entry_lo = float(getattr(cfg, "paper_v7_entry_lo", 0.30))
        self._entry_hi = float(getattr(cfg, "paper_v7_entry_hi", 0.40))
        self._exit_lo = float(getattr(cfg, "paper_v7_exit_lo", 0.20))
        self._exit_hi = float(getattr(cfg, "paper_v7_exit_hi", 0.70))
        self._target_positions = max(1, int(getattr(cfg, "paper_v7_positions", 20)))
        if not (self._exit_lo <= self._entry_lo < self._entry_hi <= self._exit_hi):
            raise ValueError(
                f"V7 分位带非法：进 ({self._entry_lo},{self._entry_hi}] "
                f"必须被出 ({self._exit_lo},{self._exit_hi}] 包住")
        # V7 自己管持仓上限，不沿用 paper_max_positions（那是给 4 只票的账户设的）。
        self._max_positions = self._target_positions

    def _prefix(self) -> str:
        return "[paper_v7]"

    # ---------- 选股池 ----------

    def _ranks(self) -> dict:
        """今天的波动率分位表；取不到就返回空，由调用方按"不动"处理。"""
        try:
            return vol_band.rank_pct(cache_dir=getattr(self._cfg, "ledger_dir", None))
        except Exception as e:  # noqa: BLE001 - 选股池算不出来不该拖垮引擎
            print(f"{self._prefix()} 波动率分位不可用: {type(e).__name__}: {e}", flush=True)
            return {}

    def _exit_label(self, reason: str) -> str:
        return _EXIT_LABEL_V7.get(reason, super()._exit_label(reason))

    # ---------- 出场 ----------

    def _exit_decision(self, pos: dict, px: float, held: int,
                       allow_time_cap: bool = True) -> Optional[str]:
        """只有跌出退出带才卖。

        分位取不到时选择继续持有：数据缺失不等于标的变差，宁可下一轮再评估，
        也不要因为一次读数失败就制造一次不该有的换手。``allow_time_cap`` 被
        刻意忽略——研究规则没有持有期上限，加上它会把换手推高一个数量级。
        """
        q = self._ranks().get(_digits(pos["code"]))
        if q is None:
            return None
        if self._exit_lo < float(q) <= self._exit_hi:
            return None
        return "vol_band_exit"

    # ---------- 入场 ----------

    def _v7_entry_skip(self, code: str) -> tuple[Optional[str], Optional[dict]]:
        """V7 自己的入场门：查报价年龄、买卖一价、一档挂单量、涨停封板，不查模型预测。

        不能复用 V3 的 ``_entry_quote_detail``——它会走到基类的预测新鲜度检查，
        而 V7 没有模型预测，那条链会把所有候选拒掉。
        """
        snap, age, bid, ask = self._quote(code)
        if snap is None:
            return "无行情快照", None
        if age is None or not math.isfinite(age) or age > self._quote_max_age:
            return f"行情过期({age})", None
        if not ask or not math.isfinite(ask) or ask <= 0:
            return "卖一价无效", None
        if not bid or not math.isfinite(bid) or bid <= 0:
            return "买一价无效", None
        ask_volume = getattr(snap, "ask_volume1", None)
        try:
            ask_volume = float(ask_volume)
        except (TypeError, ValueError):
            return "卖一挂单量无效", None
        if not math.isfinite(ask_volume) or ask_volume <= 0:
            return "卖一挂单量无效", None
        # 涨停判定沿用 V3 的口径（快照自带 high_limited），不自己按 pre_close 推档位：
        # 两套口径若不一致，V3/V7 的"买不进"样本就无法对比。
        if getattr(snap, "is_limit_up", False):
            return "封涨停不可买", None
        high_limit = getattr(snap, "high_limited", None)
        if high_limit and ask >= float(high_limit) - 0.001:
            return "卖一已到涨停价", None
        return None, {"snap": snap, "age": age, "bid": bid, "ask": ask,
                      "ask_volume1": ask_volume}

    def _run_buys(self) -> None:
        if self._state.get("last_buy_date") == _today():
            return
        positions = self._state.get("positions", [])
        free_slots = self._target_positions - len(positions)
        if free_slots <= 0:
            self._state["last_buy_date"] = _today()
            self._save_state()
            return
        t = self._now_hhmm if self._now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        if t > self._buy_end:
            return
        ranks = self._ranks()
        if not ranks:
            print(f"{self._prefix()} 波动率分位为空，本轮不买入", flush=True)
            return
        held_codes = {_digits(p["code"]) for p in positions}
        account_before = {
            "cash": self._state.get("cash", 0.0),
            "position_count": len(positions), "held_codes": sorted(held_codes),
        }
        cands = sorted(
            ((c, q) for c, q in ranks.items()
             if self._entry_lo < q <= self._entry_hi and c not in held_codes),
            key=lambda kv: kv[1])
        # 等权：每只的预算按目标持仓数摊，不按剩余槽位摊，否则先买的会拿到过大权重。
        budget = self._equity() / self._target_positions
        trace: list[dict] = []
        bought: list[str] = []
        for code, q in cands:
            if len(bought) >= free_slots:
                break
            audit = {"code": code, "status": "eligible_ranked",
                     "vol_rank_pct": round(float(q), 4)}
            trace.append(audit)
            skip, quote = self._v7_entry_skip(code)
            if skip or quote is None:
                audit.update({"entry_decision": "filtered", "entry_filter_reason": skip})
                continue
            fill_price = quote["ask"]
            unit = fill_price * (1 + self._cost / 2.0)
            shares = int(budget / unit // 100) * 100
            if shares <= 0:
                audit["entry_decision"] = "insufficient_lot_cash"
                continue
            cost_basis = fill_price * shares * (1 + self._cost / 2.0)
            if cost_basis > self._state.get("cash", 0.0):
                audit["entry_decision"] = "insufficient_cash"
                continue
            buy_time = time.strftime("%H:%M:%S")
            position_id = self._stable_id(
                self._POSITION_ID_PREFIX, code, _today(), buy_time,
                round(fill_price, 3), shares)
            self._state["cash"] -= cost_basis
            position = {
                "position_id": position_id,
                "code": code, "name": self._name_map.get(_digits(code), ""),
                "buy_date": _today(), "buy_time": buy_time,
                "buy_price": round(fill_price, 3), "shares": shares,
                "peak": round(fill_price, 3),
                "peak_bid": round(quote["bid"] or fill_price, 3),
                "cost_basis": round(cost_basis, 2),
                "entry_vol_rank_pct": round(float(q), 4),
                "buy_quote_age_sec": quote["age"], "buy_fill_source": "ask1",
                "buy_ask_volume1_raw": quote["ask_volume1"],
                # V3 起 _price_of 返回 bid1，取"最新价"必须显式走基类实现，
                # 否则 buy_last 会等于 buy_bid1，价差与漂移就拆不开了。
                "buy_last": PaperTrader._price_of(self, code),
                "buy_bid1": quote["bid"], "buy_ask1": quote["ask"],
            }
            self._state.setdefault("positions", []).append(position)
            audit.update({
                "entry_decision": "bought", "position_id": position_id,
                "shares": shares, "fill_price": round(fill_price, 3),
                "fill_source": "ask1", "quote_age_sec": quote["age"],
                "allocated_cash": round(budget, 2), "cost_basis": round(cost_basis, 2),
            })
            bought.append(f"{self._label(code)} @{fill_price:.2f} "
                          f"分位{q:.3f} {shares}股")
            if len(self._state["positions"]) >= self._target_positions:
                break
        self._finish_buy_attempt(len(bought))
        self._save_state()
        self._record_buy_decision(trace, account_before, len(bought),
                                  target_n=free_slots)
        if bought:
            self._notifier.push(
                f"[{self._PAPER_TITLE}] 买入 {len(bought)}只 {_dt.datetime.now():%m-%d %H:%M}",
                "\n".join(f"{i + 1} {b}" for i, b in enumerate(bought))
                + f"\n{self.summary()}")
