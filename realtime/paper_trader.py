"""实时模拟盘（PaperTrader）：把模型 Top-N 买入候选跑成一个持久化的纸上账户，
验证「收盘前买入 + 完整出场策略」在真实盘中价上的累计收益。

与 RankBoard（只推送不下单）的区别：PaperTrader 真正维护一个虚拟账户
（现金 + 持仓），每交易日在收盘前 10 分钟按重排后 score 降序买 Top-N，持仓则在
【全交易时段】每轮按完整出场策略评估卖出，全部按【触发时的实时价】成交，逐笔落盘复盘。

出场策略（先触发先走，缺某项数据即跳过该项、绝不误卖）：
  1. 硬止损     ret <= -stop_loss（默认 -5%）          → 控制单笔风险
  2. 止盈上限   ret >= take_profit（默认 +9%）          → 锁涨停附近冲高
  3. 移动止盈   last <= peak - k*ATR（吊灯，k=3.0）     → 回撤保护，随高点上移
  4. 破位       last < vwap*(1 - vwap_break)（默认 -2%） → 跌破日内均价走弱
  5. 时间上限   持有达 T+sell_horizon（默认 T+1）        → 最终兜底，恒可用
排序主序 = 盘中重排后 score（RerankScorer：模型 expected_return 锚定 + 盘中有界微调），
候选池恒 = 模型 Top-N 池，盘中信号只做择时/择序/出场，不造新 alpha。

设计原则（与 strategy/RankBoard 一致）：
  - 只读 ctx 内存态（最新快照 + VWAP + ref），绝不碰 quant_data。
  - 成本用 round-trip（默认 0.002），买卖各计一半。
  - 状态持久化到挂载盘 JSON：引擎日拱、盘中 execv 重启、跨交易日都要延续持仓。
    任何一步缺实时价/异常都优雅跳过（当日不动，次日再处理），绝不崩、不空跑。

落盘两份（均在 logs/realtime，容器已挂载）：
  - paper_state.json ：账户当前态（现金 + 持仓列表[含持仓期最高价 peak] + 最近买入日）。
  - paper_trades.jsonl：每笔平仓一行审计流水（买卖价/出场原因/持有交易日/收益率/累计净值）。
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path
from typing import Optional

from .notifier import Notifier
from .reference import _trading_days_between
from .rerank import RerankScorer
from .strategy import StrategyContext, _digits


def _today() -> str:
    return _dt.date.today().strftime("%Y-%m-%d")


# 出场原因 -> 中文播报短语（供卖出推送与流水审计）。
_EXIT_LABEL = {
    "stop_loss": "硬止损",
    "take_profit": "止盈",
    "trailing_stop": "移动止盈",
    "vwap_break": "破位(跌破VWAP)",
    "time_cap": "到期(T+N)",
}


class PaperTrader:
    def __init__(self, cfg, ctx: StrategyContext, notifier: Notifier,
                 name_map: Optional[dict] = None):
        self._cfg = cfg
        self._ctx = ctx
        self._notifier = notifier
        self._name_map = name_map or {}
        self._buy_n = max(1, getattr(cfg, "paper_buy_n", 2))
        self._buy_start = int(getattr(cfg, "paper_buy_start", 1450))
        self._start_equity = float(getattr(cfg, "paper_start_equity", 100000.0))
        self._cost = float(getattr(cfg, "paper_cost", 0.002))
        self._horizon = max(1, int(getattr(cfg, "sell_horizon", 1)))
        # 出场阈值（全 env 可调，见 config.py）。
        self._stop_loss = float(getattr(cfg, "paper_stop_loss", 0.05))
        self._take_profit = float(getattr(cfg, "paper_take_profit", 0.09))
        self._trail_k = float(getattr(cfg, "paper_trail_k", 3.0))
        self._vwap_break = float(getattr(cfg, "paper_vwap_break", 0.02))
        # 买入择时过滤阈值（方向2；0 即关闭该项）。
        self._entry_gap_eaten = float(getattr(cfg, "paper_entry_gap_eaten", 0.6))
        self._entry_rich = float(getattr(cfg, "paper_entry_rich", 0.01))
        self._entry_ask_strong = float(getattr(cfg, "paper_entry_ask_strong", 0.2))
        self._scorer = RerankScorer(cfg, ctx)  # 与 RankBoard 共用的盘中重排打分器
        self._state_file = Path(getattr(cfg, "paper_state_file", "")
                                or (Path(getattr(cfg, "ledger_dir", ".")) / "paper_state.json"))
        self._trades_file = self._state_file.with_name("paper_trades.jsonl")
        self._acted_today = False  # 当日买入腿是否已跑过（进程内哨兵，防同日重复建仓）
        self._state = self._load_state()

    # ---- 展示辅助 ------------------------------------------------------------
    def _label(self, code: str) -> str:
        """代码 + 中文简称；name_map 以 6 位纯代码为 key，两种口径都查。"""
        name = self._name_map.get(code) or self._name_map.get(_digits(code))
        return f"{code} {name}" if name else str(code)

    # ---- 状态持久化 ----------------------------------------------------------
    def _load_state(self) -> dict:
        """读 paper_state.json；缺失/损坏则起一个全现金空仓的新账户。"""
        if self._state_file.exists():
            try:
                s = json.loads(self._state_file.read_text(encoding="utf-8"))
                if isinstance(s, dict) and "cash" in s:
                    s.setdefault("positions", [])
                    s.setdefault("start_equity", self._start_equity)
                    s.setdefault("realized_pnl", 0.0)
                    s.setdefault("last_buy_date", "")
                    # 兼容旧状态：老持仓无 peak 字段 → 用买入价初始化。
                    for pos in s.get("positions", []):
                        pos.setdefault("peak", pos.get("buy_price", 0.0))
                    return s
            except Exception as e:  # noqa: BLE001 - 状态损坏不拦启动，重开新账户
                print(f"[paper] 读状态失败(重开新账户)：{type(e).__name__}", flush=True)
        return {"cash": self._start_equity, "start_equity": self._start_equity,
                "realized_pnl": 0.0, "last_buy_date": "", "positions": []}

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self._state_file)  # 原子替换，避免半写
        except Exception as e:  # noqa: BLE001 - 落盘失败不拖垮主循环
            print(f"[paper] 写状态失败：{type(e).__name__}", flush=True)

    def _append_trade(self, rec: dict) -> None:
        try:
            self._trades_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._trades_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[paper] 写流水失败：{type(e).__name__}", flush=True)

    # ---- 估值 ----------------------------------------------------------------
    def _price_of(self, code: str) -> Optional[float]:
        """该票最新实时价（无快照/无价则 None）。"""
        snap = self._ctx.snapshot_of(code)
        last = getattr(snap, "last", None) if snap is not None else None
        return float(last) if last else None

    def _atr_of(self, code: str) -> Optional[float]:
        """该票 ATR 绝对值（元）：优先 ref.atr，缺则 atr_pct*pre_close 兜底，都无则 None。

        复用 ChandelierStop 的兜底口径（strategy.py），供移动止盈吊灯线计算。
        """
        ref = self._ctx.ref_of(code)
        if ref is None:
            ref = self._ctx.ref_of(_digits(code))
        if ref is None:
            return None
        atr = getattr(ref, "atr", None)
        if atr:
            return float(atr)
        atr_pct = getattr(ref, "atr_pct", None)
        snap = self._ctx.snapshot_of(code)
        pre_close = getattr(snap, "pre_close", None) if snap is not None else None
        if atr_pct and pre_close:
            return float(atr_pct) * float(pre_close)
        return None

    def _equity(self) -> float:
        """当前总净值 = 现金 + 持仓按最新价市值（缺价的持仓用成本价兜底）。"""
        eq = float(self._state.get("cash", 0.0))
        for pos in self._state.get("positions", []):
            px = self._price_of(pos["code"]) or pos.get("buy_price", 0.0)
            eq += px * pos.get("shares", 0.0)
        return eq

    def summary(self) -> str:
        eq = self._equity()
        base = self._state.get("start_equity", self._start_equity) or 1.0
        ret = eq / base - 1.0
        return f"净值¥{eq:,.0f}（{ret:+.2%}）持仓{len(self._state.get('positions', []))}只"

    # ---- 主入口 --------------------------------------------------------------
    def maybe_trade(self, now_hhmm: Optional[int] = None) -> None:
        """每轮心跳跑一次：卖出腿【全交易时段】评估出场，买入腿仅收盘前窗口建仓。"""
        if not getattr(self._cfg, "paper_trade_enabled", True):
            return
        t = now_hhmm if now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        # 卖出腿：全交易时段每轮评估（止损/止盈/移动止盈/破位/到期，先触发先走）。
        self._run_sells()
        # 买入腿：仍只在收盘前 [buy_start, 收盘] 窗内、且当日未建过仓时建仓。
        if self._buy_start <= t <= 1500:
            if not (self._acted_today and self._state.get("last_buy_date") == _today()):
                self._run_buys()
                self._acted_today = True

    # ---- 出场评估 ------------------------------------------------------------
    def _exit_decision(self, pos: dict, px: float, held: int) -> Optional[str]:
        """按优先级判该持仓是否出场，返回出场原因 key（None=继续持有）。先触发先走。

        缺某项数据（ATR/VWAP）即跳过该项判定，不误卖；时间上限恒可用作最终兜底。
        """
        buy_price = pos.get("buy_price") or 0.0
        ret = (px / buy_price - 1.0) if buy_price else 0.0
        # 1) 硬止损
        if self._stop_loss > 0 and ret <= -self._stop_loss:
            return "stop_loss"
        # 2) 止盈上限
        if self._take_profit > 0 and ret >= self._take_profit:
            return "take_profit"
        # 3) 移动止盈（ATR 吊灯）：仅在已有浮盈（peak 高于买入价）时启用，避免刚建仓即被小波动扫出
        peak = pos.get("peak") or buy_price
        atr = self._atr_of(pos["code"])
        if atr and peak > buy_price:
            stop_line = peak - self._trail_k * atr
            if px <= stop_line:
                return "trailing_stop"
        # 4) 破位：跌破当日 VWAP 一定幅度
        if self._vwap_break > 0:
            vwap = self._ctx.vwap_of(pos["code"])
            if vwap and vwap > 0 and px < vwap * (1 - self._vwap_break):
                return "vwap_break"
        # 5) 时间上限（T+N 到期兜底）
        if held >= self._horizon:
            return "time_cap"
        return None

    def _run_sells(self) -> None:
        """全时段评估每个持仓的出场；命中即按实时价平仓，缺实时价则保留待下轮/次日。

        A股 T+1：当日买入（held<1）的持仓【不可卖】，跳过全部出场评估（仍持有、仍刷 peak），
        次日 held>=1 起才进出场逻辑——否则会出现 14:50 建仓后价格触止损/破位即当日平仓的 T+0 违规。

        迭代持仓列表的副本，成交时【先从持仓移除再把所得计入现金】，使 _equity()/summary()
        在写流水/推送那一刻即自洽（否则已卖出持仓仍留在列表里被重复计市值 → 净值虚高一份）。
        """
        today = _dt.date.today()
        changed = False
        for pos in list(self._state.get("positions", [])):  # 迭代副本，循环内可安全移除
            px = self._price_of(pos["code"])
            if px is None:  # 无实时价 → 无法评估/成交，保留，下轮再来
                continue
            # 刷持仓期最高价（移动止盈锚点），持久化跨日/跨 execv 重启。
            prev_peak = pos.get("peak") or pos.get("buy_price", 0.0)
            if px > prev_peak:
                pos["peak"] = round(px, 3)
                changed = True
            buy_d = self._parse_date(pos.get("buy_date"))
            held = _trading_days_between(buy_d, today) if buy_d else 0
            # A股 T+1：当日买入不可卖出，跳过出场评估（time_cap 也自然要到 held>=horizon 才触发）。
            if held < 1:
                continue
            reason = self._exit_decision(pos, px, held)
            if reason is None:
                continue
            # 先移除持仓、再把所得入现金 → _equity()/summary() 立即反映平仓后的正确净值。
            try:
                self._state["positions"].remove(pos)
            except ValueError:
                pass
            proceeds = px * pos["shares"] * (1 - self._cost / 2.0)
            cost_basis = pos.get("cost_basis", pos["buy_price"] * pos["shares"])
            pnl = proceeds - cost_basis
            ret = (pnl / cost_basis) if cost_basis else 0.0
            self._state["cash"] = self._state.get("cash", 0.0) + proceeds
            self._state["realized_pnl"] = self._state.get("realized_pnl", 0.0) + pnl
            changed = True
            self._append_trade({
                "action": "sell", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code": pos["code"], "name": self._name_map.get(_digits(pos["code"]), ""),
                "buy_date": pos.get("buy_date"), "buy_price": pos["buy_price"],
                "sell_date": _today(), "sell_price": round(px, 3), "held_days": held,
                "exit_reason": reason, "peak": pos.get("peak"), "exp": pos.get("exp"),
                "shares": pos["shares"], "pnl": round(pnl, 2), "return": round(ret, 4),
                "equity_after": round(self._equity(), 2),
            })
            self._notifier.push(
                f"[模拟盘] 卖出 {self._label(pos['code'])} @{px:.2f} {_EXIT_LABEL.get(reason, reason)}",
                f"持有T+{held} 收益{ret:+.2%} 盈亏¥{pnl:,.0f}\n{self.summary()}")
        if changed:
            self._save_state()

    def _run_buys(self) -> None:
        """当日尚未建仓则按【重排后 score】降序买 Top-N，买前过滤追高票，均分现金。"""
        if self._state.get("last_buy_date") == _today():
            return
        held_codes = {_digits(p["code"]) for p in self._state.get("positions", [])}
        ranked = self._rank(exclude=held_codes)
        if not ranked:
            self._state["last_buy_date"] = _today()  # 无候选也标记，避免反复重算
            self._save_state()
            return
        alloc = self._state.get("cash", 0.0) / self._buy_n  # 按目标只数均分（留现金给不足的腿）
        bought: list[str] = []
        for code, exp, px in ranked:
            if len(bought) >= self._buy_n:
                break
            skip = self._entry_skip(code, exp, px)  # 入场择时过滤：追高/偏贵/卖盘强则跳过
            if skip:
                print(f"[paper] 跳过追高候选 {code}：{skip}", flush=True)
                continue
            budget = min(alloc, self._state.get("cash", 0.0))
            shares = int(budget / (px * (1 + self._cost / 2.0)) // 100) * 100  # A股整百手
            if shares <= 0:
                continue
            cost_basis = px * shares * (1 + self._cost / 2.0)
            if cost_basis > self._state.get("cash", 0.0):
                continue
            self._state["cash"] -= cost_basis
            self._state.setdefault("positions", []).append({
                "code": code, "name": self._name_map.get(_digits(code), ""),
                "buy_date": _today(), "buy_time": time.strftime("%H:%M:%S"),
                "buy_price": round(px, 3), "shares": shares, "peak": round(px, 3),
                "cost_basis": round(cost_basis, 2), "exp": round(exp, 4)})
            bought.append(f"{self._label(code)} @{px:.2f} {self._exp_str(code, exp)} {shares}股")
        self._state["last_buy_date"] = _today()
        self._save_state()
        if bought:
            self._notifier.push(
                f"[模拟盘] 买入 {len(bought)}只 {_dt.datetime.now():%m-%d %H:%M}",
                "\n".join(f"{'①②③④⑤'[i] if i < 5 else i + 1} {b}"
                          for i, b in enumerate(bought)) + f"\n{self.summary()}")

    def _entry_skip(self, code: str, exp: float, px: float) -> Optional[str]:
        """买入择时过滤：命中任一追高信号则返回原因串（跳过该票），否则 None。

        缺某项数据即不拦（不误跳）。全 env 可关（阈值设 0）。
        """
        snap = self._ctx.snapshot_of(code)
        # 1) 高开已吃掉模型预期太多 → 追高
        if self._entry_gap_eaten > 0 and snap is not None and exp and exp > 0:
            open_px = getattr(snap, "open", None)
            pre_close = getattr(snap, "pre_close", None)
            if open_px and pre_close:
                eaten = (open_px / pre_close - 1.0) / exp
                if eaten >= self._entry_gap_eaten:
                    return f"高开已吃预期{eaten:.0%}"
        # 2) 现价高于 VWAP 太多 → 偏贵
        if self._entry_rich > 0:
            vwap = self._ctx.vwap_of(code)
            if vwap and vwap > 0 and px > vwap * (1 + self._entry_rich):
                return f"高于VWAP{(px / vwap - 1):.1%}"
        # 3) 卖盘明显强 → 抛压大
        if self._entry_ask_strong > 0 and snap is not None:
            imb = getattr(snap, "bid_ask_imbalance", None)
            if imb is not None and imb <= -self._entry_ask_strong:
                return f"卖盘强{imb:.2f}"
        return None

    def _exp_str(self, code: str, exp: float) -> str:
        """展示用预期字符串：优先历史校准值 + 胜率，缺校准回退原始 ridge_pred。"""
        r = self._ctx.ref_of(code) or self._ctx.ref_of(_digits(code))
        cal = getattr(r, "calibrated_return", None) if r is not None else None
        wr = getattr(r, "win_rate", None) if r is not None else None
        if cal is not None:
            wr_str = f" 胜率{wr:.0%}" if wr is not None else ""
            return f"预期{cal:+.1%}{wr_str}"
        return f"预期{exp:+.1%}"

    def _rank(self, exclude: Optional[set] = None) -> list:
        """按【重排后 score】降序返回可买候选 [(code, exp, price)]。

        候选池 = 模型 expected_return>0 的 Top-rank_pool_n，经 RerankScorer 盘中微调
        （模型分锚定 + 盘中有界纠偏），并要求有实时价、剔除封涨停、排除已持仓。
        返回三元组兼容 _run_buys；exp 恒为原始 ridge_pred（记账/展示用）。
        """
        rows = self._scorer.ranked(exclude=exclude or set(),
                                   require_price=True, drop_limit_up=True)
        return [(r.code, r.exp, r.px) for r in rows]

    @staticmethod
    def _parse_date(txt) -> Optional[_dt.date]:
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
            try:
                return _dt.datetime.strptime(str(txt), fmt).date()
            except (TypeError, ValueError):
                continue
        return None
