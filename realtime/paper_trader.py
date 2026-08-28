"""实时模拟盘（PaperTrader）：把模型 Top-N 买入候选跑成一个持久化的纸上账户，
验证「收盘前买入 + 完整出场策略」在真实盘中价上的累计收益。

与 RankBoard（只推送不下单）的区别：PaperTrader 真正维护一个虚拟账户
（现金 + 持仓），每交易日在 14:55 后按重排 score 买 Top-N；持仓风险退出全天有效，
T+N 到期腿在 14:50 后执行，全部按【触发时的实时价】成交，逐笔落盘复盘。

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

落盘四份（均在 logs/realtime，容器已挂载）：
  - paper_state.json：账户当前态（现金 + 持仓列表[含持仓期最高价 peak] + 最近买入日）。
  - paper_position_snapshots.jsonl：按心跳记录逐票成本、估值价、行情年龄和浮动盈亏。
  - paper_trades.jsonl：每笔平仓一行不可变成交流水，包含成本、卖出净额和盈亏。
  - paper_buy_decisions.jsonl：每日模型池、重排、过滤和成交决策快照。
  - paper_sell_counterfactuals.jsonl：卖出日及后续 3 日的收益/机会损益标记。
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from quant import warehouse

from .notifier import Notifier
from .reference import _trading_days_between, expected_return_text
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
    # 子类可设文件名后缀（如 V2 赛马账户设 "_v2"），使状态/流水/审计在构造前即完全隔离。
    _FILE_SUFFIX = ""

    @classmethod
    def _suffixed(cls, base: Path) -> Path:
        """按 _FILE_SUFFIX 派生同目录同后缀的账户文件名；空后缀时原样返回。"""
        if not cls._FILE_SUFFIX:
            return base
        return base.parent / f"{base.stem}{cls._FILE_SUFFIX}{base.suffix}"

    def __init__(self, cfg, ctx: StrategyContext, notifier: Notifier,
                 name_map: Optional[dict] = None):
        self._cfg = cfg
        self._ctx = ctx
        self._notifier = notifier
        self._name_map = name_map or {}
        self._buy_n = max(1, getattr(cfg, "paper_buy_n", 2))
        self._buy_start = int(getattr(cfg, "paper_buy_start", 1450))
        self._buy_end = max(
            self._buy_start, min(int(getattr(cfg, "paper_buy_end", 1455)), 1500))
        self._buy_retry_start = max(
            self._buy_start,
            min(int(getattr(cfg, "paper_buy_retry_start", 1453)), self._buy_end))
        self._current_buy_stage: Optional[str] = None
        self._time_cap_start = int(getattr(cfg, "paper_time_cap_start", 1450))
        self._start_equity = float(getattr(cfg, "paper_start_equity", 100000.0))
        self._cost = float(getattr(cfg, "paper_cost", 0.002))
        self._horizon = max(1, int(getattr(cfg, "sell_horizon", 1)))
        # 出场阈值（全 env 可调，见 config.py）。
        self._stop_loss = float(getattr(cfg, "paper_stop_loss", 0.05))
        self._take_profit = float(getattr(cfg, "paper_take_profit", 0.09))
        self._trail_k = float(getattr(cfg, "paper_trail_k", 3.0))
        self._vwap_break = float(getattr(cfg, "paper_vwap_break", 0.02))
        # 买入择时过滤阈值（方向2；0 即关闭该项）。
        self._entry_gap_eaten = float(getattr(cfg, "paper_entry_gap_eaten", 0.0))
        self._entry_rich = float(getattr(cfg, "paper_entry_rich", 0.01))
        self._entry_ask_strong = float(getattr(cfg, "paper_entry_ask_strong", 0.2))
        self._entry_spread = float(getattr(cfg, "paper_entry_spread", 0.006))
        # 仓位预算只属于实时模拟盘，不回写模型或例行训练配置。
        self._risk_per_trade = max(0.0, float(
            getattr(cfg, "paper_risk_per_trade", 0.015)))
        self._max_position_weight = min(1.0, max(0.01, float(
            getattr(cfg, "paper_max_position_weight", 0.40))))
        self._allocation_atr_k = max(0.1, float(
            getattr(cfg, "paper_allocation_atr_k", 2.0)))
        self._allocation_target_return = max(0.0001, float(
            getattr(cfg, "paper_allocation_target_return", 0.02)))
        self._scorer = RerankScorer(cfg, ctx)  # 与 RankBoard 共用的盘中重排打分器
        base_state = Path(getattr(cfg, "paper_state_file", "")
                          or (Path(getattr(cfg, "ledger_dir", ".")) / "paper_state.json"))
        # 四个账户文件在任何 I/O 之前一次性定型，子类只需声明 _FILE_SUFFIX 即完全隔离。
        self._state_file = self._suffixed(base_state)
        self._trades_file = self._suffixed(base_state.with_name("paper_trades.jsonl"))
        self._decisions_file = self._suffixed(
            base_state.with_name("paper_buy_decisions.jsonl"))
        self._position_snapshots_file = self._suffixed(
            base_state.with_name("paper_position_snapshots.jsonl"))
        self._counterfactuals_file = self._suffixed(
            base_state.with_name("paper_sell_counterfactuals.jsonl"))
        self._acted_today = False  # 当日买入腿是否已跑过（进程内哨兵，防同日重复建仓）
        self._state = self._load_state()
        self._refresh_counterfactuals()

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
                    s.setdefault("buy_attempt_date", "")
                    s.setdefault("buy_attempt_stages", [])
                    # 兼容旧状态：老持仓无 peak 字段 → 用买入价初始化。
                    for pos in s.get("positions", []):
                        pos.setdefault("peak", pos.get("buy_price", 0.0))
                    return s
            except Exception as e:  # noqa: BLE001 - 状态损坏不拦启动，重开新账户
                print(f"[paper] 读状态失败(重开新账户)：{type(e).__name__}", flush=True)
        return {"cash": self._start_equity, "start_equity": self._start_equity,
                "realized_pnl": 0.0, "last_buy_date": "", "buy_attempt_date": "",
                "buy_attempt_stages": [], "positions": []}

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            # 唯一临时名 + 原子替换：避免多进程/重启窗口下互相覆盖同一 .tmp。
            tmp = self._state_file.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self._state_file)
        except Exception as e:  # noqa: BLE001 - 落盘失败不拖垮主循环
            print(f"[paper] 写状态失败：{type(e).__name__}", flush=True)

    def _append_trade(self, rec: dict) -> None:
        try:
            self._trades_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._trades_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[paper] 写流水失败：{type(e).__name__}", flush=True)

    @staticmethod
    def _stable_id(prefix: str, *values) -> str:
        raw = "|".join(str(v) for v in values).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20]}"

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    out.append(rec)
            except json.JSONDecodeError:
                continue
        return out

    def _upsert_jsonl(self, path: Path, key: str, record: dict) -> None:
        self._upsert_many_jsonl(path, key, [record])

    def _upsert_many_jsonl(self, path: Path, key: str, records: list[dict]) -> None:
        """按逻辑主键批量原子 upsert JSONL；审计失败不影响交易。"""
        if not records:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_suffix(path.suffix + ".lock")
            with open(lock_path, "a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                by_key = {
                    str(rec.get(key)): rec for rec in self._read_jsonl(path)
                    if rec.get(key) is not None
                }
                by_key.update({str(rec[key]): rec for rec in records})
                tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    for rec in by_key.values():
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 - 审计失败不阻断交易
            print(f"[paper] 写决策审计失败：{type(e).__name__}", flush=True)

    def _trade_id(self, trade: dict) -> str:
        return str(trade.get("trade_id") or self._stable_id(
            "paper-sell", trade.get("code"), trade.get("buy_date"),
            trade.get("buy_price"), trade.get("shares"), trade.get("sell_date"),
            trade.get("sell_price")))

    def _refresh_counterfactuals(self) -> None:
        """基于日线幂等补齐每笔卖出的当日及后续 3 日反事实。"""
        records: list[dict] = []
        for trade in self._read_jsonl(self._trades_file):
            if trade.get("action") != "sell":
                continue
            try:
                sell_date = pd.Timestamp(trade["sell_date"]).normalize()
                price = warehouse.load_price_tail(
                    _digits(trade.get("code", "")), sell_date, warmup_rows=0)
                if price.empty or "date" not in price.columns:
                    continue
                price = price.copy()
                price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
                price = price.dropna(subset=["date"]).sort_values("date").drop_duplicates(
                    "date", keep="last")
                dates = price["date"].tolist()
                if sell_date not in dates:
                    future = price.iloc[0:0]
                else:
                    start = dates.index(sell_date)
                    future = price.iloc[start:start + 4]
                marks: list[dict] = []
                buy_price = float(trade.get("buy_price") or 0.0)
                sell_price = float(trade.get("sell_price") or 0.0)
                shares = float(trade.get("shares") or 0.0)
                basis = buy_price * shares * (1 + self._cost / 2.0)
                for day_index, (_, row) in enumerate(future.iterrows()):
                    close = float(row["close"])
                    high = float(row["high"]) if pd.notna(row.get("high")) else None
                    close_pnl = close * shares * (1 - self._cost / 2.0) - basis
                    marks.append({
                        "day": day_index, "date": row["date"].strftime("%Y-%m-%d"),
                        "close": close, "high": high,
                        "close_vs_sell": (close / sell_price - 1.0) if sell_price else None,
                        "high_vs_sell": (high / sell_price - 1.0) if high and sell_price else None,
                        "hold_close_return": (close_pnl / basis) if basis else None,
                        "hold_close_pnl": close_pnl,
                        "opportunity_pnl": close_pnl - float(trade.get("pnl") or 0.0),
                    })
                records.append({
                    "schema_version": 1, "event_type": "paper_sell_counterfactual",
                    "trade_id": self._trade_id(trade),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sell": trade, "markouts": marks,
                })
            except Exception as e:  # noqa: BLE001 - 单票数据问题不影响其它交易/实时引擎
                print(f"[paper] 反事实补齐跳过 {trade.get('code')}: {type(e).__name__}", flush=True)
        self._upsert_many_jsonl(
            self._counterfactuals_file, "trade_id", records)

    # ---- 估值 ----------------------------------------------------------------
    def _price_of(self, code: str) -> Optional[float]:
        """该票最新实时价（无快照/无价则 None）。"""
        snap = self._ctx.snapshot_of(code)
        last = getattr(snap, "last", None) if snap is not None else None
        try:
            price = float(last)
        except (TypeError, ValueError):
            return None
        return price if math.isfinite(price) and price > 0 else None

    def _mark_detail(self, code: str) -> tuple[Optional[float], Optional[float], str]:
        """返回估值价、行情年龄和估值来源；子类可改为可成交的 bid1。"""
        return self._price_of(code), self._ctx.quote_age_of(code), "last"

    def position_details(self) -> list[dict]:
        """生成当前持仓逐票估值明细，缺行情时保留空估值，不虚构盈亏。"""
        details = []
        for pos in self._state.get("positions", []):
            shares = float(pos.get("shares") or 0.0)
            cost_basis = float(pos.get("cost_basis") or 0.0)
            mark, age, source = self._mark_detail(pos["code"])
            market_value = mark * shares if mark is not None else None
            pnl = (market_value * (1 - self._cost / 2.0) - cost_basis
                   if market_value is not None else None)
            details.append({
                "code": pos["code"], "name": pos.get("name") or self._name_map.get(
                    _digits(pos["code"]), ""),
                "buy_date": pos.get("buy_date"), "buy_price": pos.get("buy_price"),
                "shares": int(shares), "cost_basis": round(cost_basis, 2),
                "mark_price": round(mark, 3) if mark is not None else None,
                "mark_source": source, "quote_age_sec": round(age, 1) if age is not None else None,
                "market_value": round(market_value, 2) if market_value is not None else None,
                "unrealized_pnl": round(pnl, 2) if pnl is not None else None,
                "unrealized_return": round(pnl / cost_basis, 6)
                if pnl is not None and cost_basis else None,
            })
        return details

    def record_position_snapshot(self) -> None:
        """按心跳追加逐票估值，供盘中查看和复盘，不影响交易流程。"""
        try:
            self._position_snapshots_file.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "schema_version": 1, "event_type": "paper_position_snapshot",
                "account": self._FILE_SUFFIX or "v1",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": self.summary(), "positions": self.position_details(),
            }
            with open(self._position_snapshots_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 - 观测失败不影响交易
            print(f"[paper] 写持仓明细失败：{type(e).__name__}", flush=True)

    def _exit_label(self, reason: str) -> str:
        return _EXIT_LABEL.get(reason, reason)

    def _sell_notice(self, pos: dict, px: float, held: int, pnl: float,
                     ret: float, reason: str, proceeds: float) -> str:
        cost_basis = float(pos.get("cost_basis") or pos["buy_price"] * pos["shares"])
        return (f"买入成本¥{cost_basis:,.2f}（{pos['buy_price']:.3f}×{int(pos['shares'])}股）"
                f"\\n卖出价¥{px:.3f} 卖出净额¥{proceeds:,.2f}"
                f"\\n持有T+{held} 盈亏¥{pnl:,.2f}（{ret:+.2%}）"
                f"\\n原因：{self._exit_label(reason)}\\n{self.summary()}")

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
        try:
            numeric_atr = float(atr)
        except (TypeError, ValueError):
            numeric_atr = 0.0
        if math.isfinite(numeric_atr) and numeric_atr > 0:
            return numeric_atr
        atr_pct = getattr(ref, "atr_pct", None)
        snap = self._ctx.snapshot_of(code)
        pre_close = getattr(snap, "pre_close", None) if snap is not None else None
        try:
            fallback = float(atr_pct) * float(pre_close)
        except (TypeError, ValueError):
            return None
        return fallback if math.isfinite(fallback) and fallback > 0 else None

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
    def _buy_attempts_today(self) -> set[str]:
        if self._state.get("buy_attempt_date") != _today():
            return set()
        return {str(value) for value in self._state.get("buy_attempt_stages", [])}

    def _buy_stage(self, t: int) -> Optional[str]:
        if not self._buy_start <= t <= self._buy_end:
            return None
        return "primary" if t < self._buy_retry_start else "retry"

    def _finish_buy_attempt(self, bought_count: int) -> None:
        """记录本轮阶段；成交或完成 retry 后关闭当日买入腿。"""
        stage = self._current_buy_stage or "direct"
        stages = self._buy_attempts_today()
        stages.add(stage)
        self._state["buy_attempt_date"] = _today()
        self._state["buy_attempt_stages"] = sorted(stages)
        if bought_count > 0 or stage in {"retry", "direct"}:
            self._state["last_buy_date"] = _today()

    def maybe_trade(self, now_hhmm: Optional[int] = None) -> None:
        """每轮心跳评估卖出；买入在 14:50 初筛、14:53 对未成交账户复评。"""
        if not getattr(self._cfg, "paper_trade_enabled", True):
            return
        t = now_hhmm if now_hhmm is not None else (
            _dt.datetime.now().hour * 100 + _dt.datetime.now().minute)
        # 风险退出全天有效；纯到期退出只在收盘前执行，保持训练的 close→close 口径。
        self._run_sells(t)
        stage = self._buy_stage(t)
        if stage is None or self._state.get("last_buy_date") == _today():
            return
        if stage in self._buy_attempts_today():
            return
        self._current_buy_stage = stage
        try:
            self._run_buys()
        finally:
            self._current_buy_stage = None
        self._acted_today = self._state.get("last_buy_date") == _today()

    # ---- 出场评估 ------------------------------------------------------------
    def _exit_decision(self, pos: dict, px: float, held: int,
                       allow_time_cap: bool = True) -> Optional[str]:
        """按优先级判持仓是否出场；风险退出全天有效，到期退出由调用方按时段放行。"""
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
        # 5) 时间上限（T+N 到期兜底）：只在收盘前放行，避免把 close→close 做成 next-open。
        if allow_time_cap and held >= self._horizon:
            return "time_cap"
        return None

    def _run_sells(self, now_hhmm: Optional[int] = None) -> None:
        """评估持仓出场；风险规则全天生效，纯到期规则只在 time_cap_start 后生效。

        A股 T+1：当日买入（held<1）的持仓不可卖，跳过全部出场评估（仍持有、仍刷 peak）。
        缺实时价时保留待下轮；直接调用未传时刻时按收盘时段处理，保持测试与旧内部调用兼容。
        """
        t = 1500 if now_hhmm is None else int(now_hhmm)
        allow_time_cap = t >= self._time_cap_start
        today = _dt.date.today()
        changed = False
        sold = False
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
            reason = self._exit_decision(pos, px, held, allow_time_cap=allow_time_cap)
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
            sell_time = time.strftime("%Y-%m-%d %H:%M:%S")
            trade_rec = {
                "action": "sell", "time": sell_time,
                "trade_id": str(pos.get("position_id") or self._stable_id(
                    "paper-pos", pos["code"], pos.get("buy_date"), pos.get("buy_time"),
                    pos["buy_price"], pos["shares"])),
                "code": pos["code"], "name": self._name_map.get(_digits(pos["code"]), ""),
                "buy_date": pos.get("buy_date"), "buy_time": pos.get("buy_time"),
                "buy_price": pos["buy_price"], "sell_date": _today(),
                "sell_price": round(px, 3), "held_days": held,
                "exit_reason": reason, "peak": pos.get("peak"), "exp": pos.get("exp"),
                "shares": pos["shares"], "pnl": round(pnl, 2), "return": round(ret, 4),
                "equity_after": round(self._equity(), 2),
            }
            self._append_trade(trade_rec)
            sold = True
            self._notifier.push(
                f"[模拟盘] 卖出 {self._label(pos['code'])} @{px:.2f} {_EXIT_LABEL.get(reason, reason)}",
                self._sell_notice(pos, px, held, pnl, ret, reason, proceeds))
        if changed:
            self._save_state()
        if sold:
            self._refresh_counterfactuals()

    def _market_audit(self, code: str) -> dict:
        snap = self._ctx.snapshot_of(code)
        return {
            "last": getattr(snap, "last", None) if snap is not None else None,
            "pre_close": getattr(snap, "pre_close", None) if snap is not None else None,
            "open": getattr(snap, "open", None) if snap is not None else None,
            "vwap": self._ctx.vwap_of(code),
            "bid_ask_imbalance": (
                getattr(snap, "bid_ask_imbalance", None) if snap is not None else None),
            "spread_pct": getattr(snap, "spread_pct", None) if snap is not None else None,
            "is_limit_up": bool(
                getattr(snap, "is_limit_up", False)) if snap is not None else False,
        }

    def _record_buy_decision(self, candidates: list[dict], account_before: dict,
                             bought_count: int) -> None:
        for rec in candidates:
            rec["name"] = self._name_map.get(_digits(rec.get("code", "")), "")
            rec["market"] = self._market_audit(rec.get("code", ""))
        eligible = [r for r in candidates if r.get("status") == "eligible_ranked"]
        if bought_count:
            status = "bought" if bought_count == self._buy_n else "partial_fill"
        elif eligible and all(r.get("entry_decision") == "filtered" for r in eligible):
            status = "all_candidates_filtered"
        elif eligible:
            status = "insufficient_cash_or_lot"
        else:
            status = "no_ranked_candidate"
        stage = self._current_buy_stage or "direct"
        record = {
            "schema_version": 1, "event_type": "paper_buy_decision",
            "event_id": f"paper-buy-decision:{_today()}:{stage}",
            "trade_date": _today(), "attempt_stage": stage,
            "decision_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decision_status": status,
            "paper_config": {
                "buy_n": self._buy_n, "buy_start": self._buy_start,
                "buy_retry_start": self._buy_retry_start, "buy_end": self._buy_end,
                "time_cap_start": self._time_cap_start, "cost_roundtrip": self._cost,
                "rank_pool_n": getattr(self._cfg, "rank_pool_n", 30),
                "rank_min_raw_return": getattr(self._cfg, "rank_min_raw_return", 0.0),
                "rank_raw_safety_margin": getattr(self._cfg, "rank_raw_safety_margin", 0.001),
                "entry_rich": self._entry_rich,
                "entry_ask_strong": self._entry_ask_strong,
                "entry_spread": self._entry_spread,
                "risk_per_trade": self._risk_per_trade,
                "max_position_weight": self._max_position_weight,
                "allocation_atr_k": self._allocation_atr_k,
                "allocation_target_return": self._allocation_target_return,
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

    def _allocation_factor(self, code: str) -> tuple[float, Optional[str]]:
        """子版本可按组合上下文缩放预算；0 表示阻断该候选。"""
        return 1.0, None

    def _allocation_budget(self, code: str, exp: float, px: float,
                           remaining_slots: int) -> tuple[float, dict]:
        """按组合净值、ATR 风险和模型预期计算可执行的单票现金预算。"""
        cash = max(0.0, float(self._state.get("cash", 0.0)))
        equity = max(0.0, float(self._equity()))
        equal_budget = cash / max(1, int(remaining_slots))
        position_cap = equity * self._max_position_weight
        atr = self._atr_of(code)
        if atr is not None and px > 0:
            risk_pct = max(self._cost, self._allocation_atr_k * atr / px)
            risk_source = "atr"
        else:
            risk_pct = max(self._cost, self._stop_loss if self._stop_loss > 0 else 0.05)
            risk_source = "stop_loss_fallback"
        signal_factor = min(1.25, max(0.75, math.sqrt(
            max(0.0, float(exp)) / self._allocation_target_return)))
        risk_budget_cash = equity * self._risk_per_trade * signal_factor
        risk_position_cap = risk_budget_cash / risk_pct if risk_pct > 0 else position_cap
        allocation_factor, factor_reason = self._allocation_factor(code)
        adjusted_risk_cap = risk_position_cap * max(0.0, allocation_factor)
        budget = max(0.0, min(cash, equal_budget, position_cap, adjusted_risk_cap))
        return budget, {
            "allocation_method": "risk_budget_v1",
            "equal_budget": round(equal_budget, 2),
            "position_cap": round(position_cap, 2),
            "risk_position_cap": round(risk_position_cap, 2),
            "adjusted_risk_cap": round(adjusted_risk_cap, 2),
            "risk_budget_cash": round(risk_budget_cash, 2),
            "risk_pct": round(risk_pct, 6), "risk_source": risk_source,
            "allocation_atr": atr, "signal_factor": round(signal_factor, 4),
            "allocation_factor": round(max(0.0, allocation_factor), 4),
            "allocation_factor_reason": factor_reason,
            "budget": round(budget, 2),
        }

    def _run_buys(self) -> None:
        """按重排主序买入，并以风险预算、单票上限和整手约束确定仓位。"""
        if self._state.get("last_buy_date") == _today():
            return
        positions = self._state.get("positions", [])
        held_codes = {_digits(p["code"]) for p in positions}
        account_before = {
            "cash": self._state.get("cash", 0.0),
            "position_count": len(positions), "held_codes": sorted(held_codes),
        }
        trace: list[dict] = []
        ranked_rows = self._scorer.ranked(
            exclude=held_codes, require_price=True, drop_limit_up=True, trace=trace)
        trace_by_code = {rec["code"]: rec for rec in trace}
        remaining_slots = self._buy_n
        bought: list[str] = []
        for row in ranked_rows:
            code, exp, px = row.code, row.exp, row.px
            audit = trace_by_code.get(_digits(code), {})
            if len(bought) >= self._buy_n:
                audit["entry_decision"] = "not_selected_below_buy_n"
                continue
            skip = self._entry_skip(code, exp, px)
            if skip:
                audit.update({"entry_decision": "filtered", "entry_filter_reason": skip})
                print(f"[paper] 跳过追高候选 {code}：{skip}", flush=True)
                continue
            budget, allocation = self._allocation_budget(
                code, exp, px, remaining_slots)
            audit["allocation"] = allocation
            if budget <= 0:
                audit.update({"entry_decision": "allocation_blocked",
                              "entry_filter_reason": allocation.get(
                                  "allocation_factor_reason") or "风险预算为零"})
                continue
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
                "paper-pos", code, _today(), buy_time, round(px, 3), shares)
            self._state["cash"] -= cost_basis
            self._state.setdefault("positions", []).append({
                "position_id": position_id,
                "code": code, "name": self._name_map.get(_digits(code), ""),
                "buy_date": _today(), "buy_time": buy_time,
                "buy_price": round(px, 3), "shares": shares, "peak": round(px, 3),
                "cost_basis": round(cost_basis, 2), "exp": round(exp, 4)})
            audit.update({
                "entry_decision": "bought", "position_id": position_id,
                "shares": shares, "fill_price": round(px, 3),
                "allocated_cash": budget, "cost_basis": round(cost_basis, 2),
            })
            remaining_slots -= 1
            bought.append(f"{self._label(code)} @{px:.2f} {self._exp_str(code, exp)} {shares}股")
        self._finish_buy_attempt(len(bought))
        self._save_state()
        self._record_buy_decision(trace, account_before, len(bought))
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
        # 4) 盘口价差过宽 → 流动性差(薄盘口)，建仓滑点大
        if self._entry_spread > 0 and snap is not None:
            spread = getattr(snap, "spread_pct", None)
            if spread is not None and spread >= self._entry_spread:
                return f"盘口宽{spread:.2%}"
        return None

    def _exp_str(self, code: str, exp: float) -> str:
        """展示扣除模拟盘 round-trip 成本后的净收益，并保留毛收益口径。"""
        r = self._ctx.ref_of(code) or self._ctx.ref_of(_digits(code))
        return expected_return_text(r, exp, self._cost)

    def _rank(self, exclude: Optional[set] = None) -> list:
        """按【重排后 score】降序返回可买候选 [(code, exp, price)]。

        候选池先过三模型融合收益门，再按融合 pred 百分位取 Top-rank_pool_n，经
        RerankScorer 盘中有界纠偏，并要求有实时价、剔除封涨停、排除已持仓。
        返回三元组兼容 _run_buys；exp 为同量纲融合收益率（成本门/记账/展示用）。
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
