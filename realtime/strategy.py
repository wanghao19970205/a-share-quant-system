"""信号策略：可插拔骨架 + 一组实时流策略。

策略分三类：
  异动监控类（提醒你盯的这批票发生了什么）：
    - LimitMoveWatch   封板/开板/炸板 + 逼近涨跌停
    - SurgeWatch       急速拉升 / 跳水（短时涨跌速）
    - VolumeSurge      放量异动（日内成交速率突破基线）
  买卖点纠偏类（用盘中基准校准模型的静态价位）：
    - VWAPDeviation    现价相对当日 VWAP 偏离（便宜/贵）
    - ChandelierStop   ATR 吊灯跟踪止损（随最高价上移，锁利）
    - GapCalibrate     仅保留显式实验；高开经离线 IC 验证为正向动量，不再默认装配
  持仓管理类（按已定卖点纪律择时了结）：
    - HoldingExpiry    持有期到期卖出（Phase1 对比表定档 T+1，次日提示了结）

设计原则：
  - 模型给的价位只用于"选强票入池"，真正买卖点交给盘中规则纠偏。
  - 卖点时机由 Phase1 holdout 对比表定档 T+1（次日卖：年化/Sharpe/回撤三项全优，
    持有越久 alpha 衰减越快）；HoldingExpiry 把这条纪律落到实时提醒。
  - 每条信号 fires-once / 带内部去重，冷却在 notifier 层统一兜底。
  - 只用 Level-1 快照能得到的量（价、累计量额、5 档盘口）+ 启动期基准
    （ATR、预期收益、持有交易日数，见 reference.RefRow），不假设逐笔/Level-2 数据。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .snapshot import Snapshot


@dataclass
class Signal:
    code: str
    kind: str                     # 信号类型，如 "limit_up" / "surge_up" / "vwap_cheap"
    level: str = "info"           # info / warn / strong
    reason: str = ""              # 人类可读理由（进推送正文）
    metrics: dict = field(default_factory=dict)  # 关键数值（价格/涨跌幅/失衡度…）
    ts: float = field(default_factory=time.time)


@dataclass
class _CodeState:
    last_snap: Optional[Snapshot] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    ever_limit_up: bool = False
    # ---- 盘中累计/派生（供纠偏类策略）--------------------------------------
    open_price: Optional[float] = None          # 当日开盘价（首笔快照的 open 或 last）
    cur_ts: Optional[float] = None              # 当前快照接收时刻
    prev_amount: Optional[float] = None         # 上一条快照的累计成交额
    prev_volume: Optional[float] = None         # 上一条快照的累计成交量
    prev_ts: Optional[float] = None             # 上一条快照的接收时刻（算涨速）
    prev_last: Optional[float] = None           # 上一条快照的最新价（算涨速）
    vol_rate_ref: Optional[float] = None        # 成交速率基线（首个正的额增量）

    @property
    def vwap(self) -> Optional[float]:
        """当日 VWAP=累计成交额/累计成交量。带量纲护栏：真实 VWAP 与现价之比恒在
        [0.5,2.0] 内（A股日内涨跌停封顶±10~20%），越界即判 amount/volume 量纲不一致
        （如 SDK volume 回「手」而 amount 回「元」→ VWAP≈真实均价×100），返回 None 让
        上层跳过 VWAP 类判定（破位/便宜贵），绝不据坏值误卖。护栏对合法 VWAP 完全透明。"""
        s = self.last_snap
        if s is None or not s.amount or not s.volume:
            return None
        try:
            v = s.amount / s.volume
        except ZeroDivisionError:
            return None
        last = s.last
        if last and last > 0 and not (0.5 <= v / last <= 2.0):
            return None  # 量纲异常（多半是 手/股 或 元/千元 不一致），不返回坏 VWAP
        return v


def _digits(code) -> str:
    """把任意口径代码规范成 6 位纯数字（剥券商后缀）：603956.SH -> 603956。"""
    return str(code or "").split(".", 1)[0].strip().zfill(6)


class StrategyContext:
    """跨快照的轻量状态容器（单进程内存，日内有效）。

    可选注入启动期参考数据 ref（{code: RefRow}），供纠偏类策略取 ATR/预期收益。

    重要时序：引擎在跑策略【前】先调 update(snap)。因此 update 必须把「上一条」
    快照的值滚存进 prev_*（在覆盖 last_snap 之前），策略随后才能拿到 current(snap)
    vs prev_* 的差分。否则 SurgeWatch/VolumeSurge 会看到 prev==current（差分恒 0）。
    """

    def __init__(self, ref: Optional[dict] = None) -> None:
        self._by_code: dict[str, _CodeState] = {}
        self._by_digits: dict[str, _CodeState] = {}  # 6 位纯代码 -> 同一 state（供无后缀口径反查）
        self._ref = ref or {}

    def ref_of(self, code: str):
        return self._ref.get(code)

    def all_refs(self) -> dict:
        """全部 {code: RefRow} 只读视图；供 RankBoard 按 expected_return 排名。"""
        return self._ref

    def _state(self, code):
        """按原始 code 查，未命中再按 6 位纯代码反查（吸收带/不带后缀口径差异）。"""
        st = self._by_code.get(code or "?")
        if st is None:
            st = self._by_digits.get(_digits(code))
        return st

    def snapshot_of(self, code):
        """该票最新一条快照（无则 None）；供 RankBoard 只读取盘中量。"""
        st = self._state(code)
        return st.last_snap if st is not None else None

    def vwap_of(self, code):
        """该票当日 VWAP（累计额/累计量，无则 None）；供 RankBoard 只读。"""
        st = self._state(code)
        return st.vwap if st is not None else None

    def state_of(self, code: str) -> _CodeState:
        return self._by_code.setdefault(code or "?", _CodeState())

    def update(self, snap: Snapshot) -> _CodeState:
        st = self._by_code.setdefault(snap.code or "?", _CodeState())
        self._by_digits[_digits(snap.code)] = st  # 维护 6 位索引，供 RankBoard 无后缀口径反查
        now = time.time()
        old = st.last_snap
        # 先把「上一条」的值滚存进 prev_*（务必在覆盖 last_snap 之前）
        if old is not None:
            st.prev_last = old.last
            st.prev_amount = old.amount
            st.prev_volume = old.volume
            st.prev_ts = st.cur_ts
        # 成交速率基线：首个正的额增量（current - prev）作参考
        if snap.amount is not None and st.prev_amount is not None:
            d_amt = snap.amount - st.prev_amount
            if d_amt > 0 and st.vol_rate_ref is None:
                st.vol_rate_ref = d_amt
        # 日内高低 + 开盘价
        if snap.last is not None:
            st.day_high = snap.last if st.day_high is None else max(st.day_high, snap.last)
            st.day_low = snap.last if st.day_low is None else min(st.day_low, snap.last)
            if st.open_price is None:
                st.open_price = snap.open if snap.open is not None else snap.last
        if snap.is_limit_up:
            st.ever_limit_up = True
        st.cur_ts = now
        st.last_snap = snap
        return st


class Strategy:
    """策略抽象基类。子类实现 on_snapshot 返回 0..N 条信号。"""

    name = "base"

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:  # pragma: no cover
        raise NotImplementedError


# ==== 异动监控类 =============================================================

class LimitMoveWatch(Strategy):
    """封板/开板/炸板 + 逼近涨跌停。

    - 距涨停 <= near_pct（默认 1%）→ near_limit_up（warn）
    - 首次封涨停 → limit_up（strong）
    - 封过又打开（炸板）→ limit_open（strong，仅在曾封板后触发一次）
    - 跌停对称处理。
    """

    name = "limit_move_watch"

    def __init__(self, near_pct: float = 0.01):
        self._near = near_pct
        self._fired_up: set[str] = set()
        self._fired_down: set[str] = set()
        self._fired_open: set[str] = set()

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        last, hi, lo = snap.last, snap.high_limited, snap.low_limited
        if last is None:
            return out

        if snap.is_limit_up:
            if code not in self._fired_up:
                self._fired_up.add(code)
                out.append(Signal(code, "limit_up", "strong", f"封涨停 {last}",
                                  {"last": last, "high_limited": hi}))
        else:
            # 曾封涨停、现已打开 → 炸板（只提示一次）
            if code in self._fired_up and code not in self._fired_open:
                self._fired_open.add(code)
                out.append(Signal(code, "limit_open", "strong", f"炸板打开 {last}/{hi}",
                                  {"last": last, "high_limited": hi}))
            elif hi and last >= hi * (1 - self._near):
                out.append(Signal(code, "near_limit_up", "warn", f"逼近涨停 {last}/{hi}",
                                  {"last": last, "high_limited": hi,
                                   "gap_pct": round((hi - last) / hi, 4)}))

        if snap.is_limit_down:
            if code not in self._fired_down:
                self._fired_down.add(code)
                out.append(Signal(code, "limit_down", "strong", f"封跌停 {last}",
                                  {"last": last, "low_limited": lo}))
        elif lo and last <= lo * (1 + self._near):
            out.append(Signal(code, "near_limit_down", "warn", f"逼近跌停 {last}/{lo}",
                              {"last": last, "low_limited": lo}))
        return out


class SurgeWatch(Strategy):
    """急速拉升 / 跳水：短时窗口内价格变动速率超阈值。

    用相邻两条快照的 (Δprice/price)/Δt 估瞬时涨速，>= surge_pct 且时间间隔
    在 [min_dt, max_dt] 内才算（防止长间隔累积误判为急拉）。按方向各带冷却。
    """

    name = "surge_watch"

    def __init__(self, surge_pct: float = 0.02, min_dt: float = 1.0,
                 max_dt: float = 60.0, cooldown: float = 60.0):
        self._surge = surge_pct
        self._min_dt, self._max_dt = min_dt, max_dt
        self._cooldown = cooldown
        self._last_fire: dict[tuple[str, str], float] = {}

    def _cooled(self, code: str, direction: str) -> bool:
        now = time.time()
        key = (code, direction)
        if now - self._last_fire.get(key, 0.0) < self._cooldown:
            return False
        self._last_fire[key] = now
        return True

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        st = ctx.state_of(code)
        last = snap.last
        prev, prev_ts = st.prev_last, st.prev_ts
        if last is None or prev is None or prev_ts is None or not prev:
            return out
        dt = time.time() - prev_ts
        if dt < self._min_dt or dt > self._max_dt:
            return out
        move = (last - prev) / prev
        if move >= self._surge and self._cooled(code, "up"):
            out.append(Signal(code, "surge_up", "warn",
                              f"急速拉升 {prev}->{last} ({move:+.2%}/{dt:.0f}s)",
                              {"last": last, "prev": prev, "move": round(move, 4),
                               "dt": round(dt, 1)}))
        elif move <= -self._surge and self._cooled(code, "down"):
            out.append(Signal(code, "surge_down", "warn",
                              f"急速跳水 {prev}->{last} ({move:+.2%}/{dt:.0f}s)",
                              {"last": last, "prev": prev, "move": round(move, 4),
                               "dt": round(dt, 1)}))
        return out


class VolumeSurge(Strategy):
    """放量异动：当前成交额增量速率 >= 基线速率 * mult。

    基线取 ctx 记录的首个正额增量（vol_rate_ref）。仅在有基线且当前增量为正时判定。
    每票只报一次（放量是状态切换，无需 tick 刷）。
    """

    name = "volume_surge"

    def __init__(self, mult: float = 3.0):
        self._mult = mult
        self._fired: set[str] = set()

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        if code in self._fired:
            return out
        st = ctx.state_of(code)
        ref = st.vol_rate_ref
        if ref is None or ref <= 0 or snap.amount is None or st.prev_amount is None:
            return out
        d_amt = snap.amount - st.prev_amount
        if d_amt <= 0:
            return out
        if d_amt >= ref * self._mult:
            self._fired.add(code)
            out.append(Signal(code, "volume_surge", "warn",
                              f"放量异动 增量额×{d_amt / ref:.1f} 基线",
                              {"d_amount": round(d_amt, 2), "ref": round(ref, 2),
                               "mult": round(d_amt / ref, 2), "last": snap.last}))
        return out


# ==== 买卖点纠偏类 ===========================================================

class VWAPDeviation(Strategy):
    """现价相对当日 VWAP 的偏离：便宜（低于 VWAP dev）→买点确认；贵（高于）→卖点确认。

    VWAP = 累计成交额 / 累计成交量（Level-1 够）。开盘头 warmup_sec 秒内 VWAP
    抖动大，跳过。各方向带冷却。
    """

    name = "vwap_deviation"

    def __init__(self, dev: float = 0.01, cooldown: float = 300.0):
        self._dev = dev
        self._cooldown = cooldown
        self._last_fire: dict[tuple[str, str], float] = {}

    def _cooled(self, code: str, side: str) -> bool:
        now = time.time()
        key = (code, side)
        if now - self._last_fire.get(key, 0.0) < self._cooldown:
            return False
        self._last_fire[key] = now
        return True

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        st = ctx.state_of(code)
        vwap = st.vwap
        last = snap.last
        if vwap is None or last is None or vwap <= 0:
            return out
        rel = (last - vwap) / vwap
        if rel <= -self._dev and self._cooled(code, "cheap"):
            out.append(Signal(code, "vwap_cheap", "info",
                              f"低于VWAP {last}<{vwap:.2f} ({rel:+.2%}) 相对便宜",
                              {"last": last, "vwap": round(vwap, 3), "rel": round(rel, 4)}))
        elif rel >= self._dev and self._cooled(code, "rich"):
            out.append(Signal(code, "vwap_rich", "info",
                              f"高于VWAP {last}>{vwap:.2f} ({rel:+.2%}) 相对偏贵",
                              {"last": last, "vwap": round(vwap, 3), "rel": round(rel, 4)}))
        return out


class ChandelierStop(Strategy):
    """ATR 吊灯跟踪止损：止损线 = 当日最高价 - k*ATR，随最高价上移只涨不跌。

    现价跌破止损线 → chandelier_stop（strong）。ATR 取自启动期 reference.RefRow；
    缺 ATR 时按 atr_pct 兜底，都没有则不启用（该票降级）。每票触发一次。
    """

    name = "chandelier_stop"

    def __init__(self, k: float = 3.0):
        self._k = k
        self._fired: set[str] = set()

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        if code in self._fired:
            return out
        last = snap.last
        if last is None:
            return out
        st = ctx.state_of(code)
        ref = ctx.ref_of(code)
        if ref is None or st.day_high is None:
            return out
        atr = getattr(ref, "atr", None)
        if atr is None:
            atr_pct = getattr(ref, "atr_pct", None)
            if atr_pct and snap.pre_close:
                atr = atr_pct * snap.pre_close
        if not atr:
            return out
        stop_line = st.day_high - self._k * atr
        if last <= stop_line:
            self._fired.add(code)
            out.append(Signal(code, "chandelier_stop", "strong",
                              f"跌破吊灯止损 {last}<={stop_line:.2f} "
                              f"(高{st.day_high:.2f}-{self._k}*ATR{atr:.2f})",
                              {"last": last, "day_high": round(st.day_high, 3),
                               "atr": round(atr, 3), "stop_line": round(stop_line, 3)}))
        return out


class GapCalibrate(Strategy):
    """开盘跳空校准：高开已吃掉模型预期收益的大部分 → 提示别追（gap_eaten）。

    开盘涨幅 / 预期收益 >= eat_ratio（默认 0.6，吃掉 60%+）即触发。低开对称
    给"折价补涨空间"提示。依赖 reference 的 expected_return，缺则不启用。每票一次。
    """

    name = "gap_calibrate"

    def __init__(self, eat_ratio: float = 0.6):
        self._eat = eat_ratio
        self._fired: set[str] = set()

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        if code in self._fired:
            return out
        st = ctx.state_of(code)
        ref = ctx.ref_of(code)
        if ref is None or st.open_price is None or not snap.pre_close:
            return out
        exp = getattr(ref, "expected_return", None)
        if not exp or exp <= 0:
            return out
        gap = st.open_price / snap.pre_close - 1.0
        eaten = gap / exp
        if eaten >= self._eat:
            self._fired.add(code)
            out.append(Signal(code, "gap_eaten", "warn",
                              f"高开{gap:+.2%}已吃预期{exp:+.2%}的{eaten:.0%}，追高需谨慎",
                              {"open": st.open_price, "pre_close": snap.pre_close,
                               "gap": round(gap, 4), "expected": round(exp, 4),
                               "eaten_ratio": round(eaten, 3)}))
        return out


# ==== 持仓管理类 =============================================================

class HoldingExpiry(Strategy):
    """持有期到期卖出：按 Phase1 对比表定档的卖点纪律（默认 T+1，次日卖）提示了结。

    依赖 reference 注入的 RefRow.hold_days（该持仓已持有的交易日数，启动期按
    买入日期 + 交易日历算好）。当 hold_days >= sell_horizon（默认 1）即提示卖出。

    - 只对「有买入记录」的持仓票触发（ref_of(code).hold_days is not None）；
      纯监控票（选股清单里未持仓的）没有 hold_days，不触发，避免误报。
    - 每票每日一次（fires-once）。真正到期是"状态"，不需要每 tick 刷。
    - sell_horizon 可被环境变量 REALTIME_SELL_HORIZON 覆盖，便于对比表结论更新后
      不改代码即可调（当前定档 T+1）。
    """

    name = "holding_expiry"

    def __init__(self, sell_horizon: Optional[int] = None):
        # 默认 1 = T+1（Phase1 holdout 对比：年化/Sharpe/回撤三项全优）。
        if sell_horizon is None:
            try:
                sell_horizon = int(os.environ.get("REALTIME_SELL_HORIZON", "1"))
            except (TypeError, ValueError):
                sell_horizon = 1
        self._horizon = max(1, sell_horizon)
        self._fired: set[str] = set()

    def on_snapshot(self, snap: Snapshot, ctx: StrategyContext) -> list[Signal]:
        out: list[Signal] = []
        code = snap.code or "?"
        if code in self._fired:
            return out
        ref = ctx.ref_of(code)
        hold_days = getattr(ref, "hold_days", None) if ref is not None else None
        if hold_days is None:          # 非持仓票（无买入记录）→ 不管
            return out
        if hold_days >= self._horizon:
            self._fired.add(code)
            last = snap.last
            out.append(Signal(code, "holding_expiry", "strong",
                              f"持有已达 T+{hold_days}（定档 T+{self._horizon}），"
                              f"到期建议了结 现价{last}",
                              {"hold_days": hold_days, "sell_horizon": self._horizon,
                               "last": last}))
        return out


def default_strategies() -> list[Strategy]:
    """默认装配：异动监控 3 条 + 已验证纠偏 2 条 + 持仓管理 1 条。"""
    return [
        LimitMoveWatch(),
        SurgeWatch(),
        VolumeSurge(),
        VWAPDeviation(),
        ChandelierStop(),
        HoldingExpiry(),
    ]
