"""主循环 + 生命周期：装配 feed/strategy/notifier/ledger，按交易时段驻场运行。

生命周期（本地时间）：
    < session_start        : 未开盘，休眠等待
    [start, morning_close] : 盘中，订阅活跃
    (morning_close, afternoon_open): 午休，快照静默（订阅保持，无推送属正常）
    [afternoon_open, end]  : 盘中，订阅活跃
    > session_end          : 收盘，停订阅并退出
    [avoid_start, avoid_end]: daily-light 规避窗，暂停订阅避免抢券商连接

信号处理链：feed → strategy.on_snapshot → notifier.notify → ledger.record

先搭骨架：真实策略后续往 strategy 里塞；引擎不需要改。
"""
from __future__ import annotations

import signal as _signal
import threading
import time
from datetime import datetime

from . import feed as feed_mod
from . import reference as ref_mod
from . import watchlist as wl_mod
from .config import RealtimeConfig, load
from .ledger import Ledger
from .notifier import Notifier
from .snapshot import Snapshot
from .strategy import Strategy, StrategyContext, default_strategies


def _hhmm(now: datetime | None = None) -> int:
    now = now or datetime.now()
    return now.hour * 100 + now.minute


class Engine:
    def __init__(self, cfg: RealtimeConfig | None = None,
                 strategies: list[Strategy] | None = None):
        self._cfg = cfg or load()
        self._strategies = strategies if strategies is not None else default_strategies()
        self._ctx = StrategyContext()
        self._notifier = Notifier(self._cfg)
        self._ledger = Ledger(self._cfg)
        self._feed: feed_mod.BaseFeed | None = None
        self._stop = threading.Event()
        self._active = False          # 当前是否处于活跃订阅态
        self._recv = 0                # 收到快照计数（心跳打印用）
        self._signals = 0

    # ---- 生命周期判定 --------------------------------------------------------
    def _in_avoid_window(self, t: int) -> bool:
        a, b = self._cfg.avoid_start, self._cfg.avoid_end
        return a and b and a <= t <= b

    def _in_session(self, t: int) -> bool:
        c = self._cfg
        if self._in_avoid_window(t):
            return False
        morning = c.session_start <= t <= c.morning_close
        afternoon = c.afternoon_open <= t <= c.session_end
        return morning or afternoon

    def _past_session(self, t: int) -> bool:
        return t > self._cfg.session_end

    # ---- 快照处理链 ----------------------------------------------------------
    def _on_snapshot(self, snap: Snapshot) -> None:
        self._recv += 1
        self._ctx.update(snap)
        for strat in self._strategies:
            try:
                sigs = strat.on_snapshot(snap, self._ctx)
            except Exception as e:  # noqa: BLE001 - 单策略异常不拖垮全局
                print(f"[engine] 策略 {strat.name} 异常: {type(e).__name__}", flush=True)
                continue
            for sig in sigs:
                self._signals += 1
                notified = self._notifier.notify(sig)
                self._ledger.record(sig, notified)

    # ---- 订阅启停 ------------------------------------------------------------
    def _start_feed(self, codes: list[str]) -> None:
        if self._active:
            return
        try:
            self._feed = feed_mod.make_feed(self._cfg, codes, self._on_snapshot)
            self._feed.start()
            self._active = True
            print(f"[engine] 订阅已启动，{len(codes)} 只 "
                  f"(dry_run={self._cfg.dry_run})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[engine] 启动订阅失败: {type(e).__name__}: {e}", flush=True)
            self._feed = None
            self._active = False

    def _stop_feed(self, reason: str = "") -> None:
        if not self._active:
            return
        try:
            if self._feed:
                self._feed.stop()
        finally:
            self._feed = None
            self._active = False
            print(f"[engine] 订阅已停止{('：' + reason) if reason else ''}", flush=True)

    # ---- 主循环 --------------------------------------------------------------
    def run(self) -> None:
        self._install_signal_handlers()
        codes = wl_mod.load_codes(self._cfg)
        if not codes:
            print("[engine] 订阅清单为空（选股清单/持仓/兜底池都没读到），退出。", flush=True)
            return
        # 启动期预取盘中纠偏基准（ATR/预期收益），注入策略上下文；盘中不再碰 quant_data。
        try:
            ref = ref_mod.build(self._cfg, codes)
            self._ctx = StrategyContext(ref=ref)
            n_atr = sum(1 for r in ref.values() if r.atr is not None)
            n_ret = sum(1 for r in ref.values() if r.expected_return is not None)
            print(f"[engine] 参考基准就绪：{len(ref)} 只（ATR {n_atr}，预期收益 {n_ret}）",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - 基准缺失只降级纠偏类，不拦启动
            print(f"[engine] 参考基准构建失败(降级)：{type(e).__name__}: {e}", flush=True)
        print(f"[engine] 启动实时层：清单 {len(codes)} 只，"
              f"时段 {self._cfg.session_start}-{self._cfg.session_end}，"
              f"策略 {[s.name for s in self._strategies]}", flush=True)

        last_heartbeat = 0.0
        while not self._stop.is_set():
            t = _hhmm()
            if self._past_session(t):
                self._stop_feed("收盘")
                print("[engine] 已过收盘时间，退出主循环。", flush=True)
                break
            if self._in_session(t):
                self._start_feed(codes)
            else:
                self._stop_feed("非交易时段/规避窗")

            now = time.time()
            if now - last_heartbeat >= self._cfg.heartbeat_sec:
                last_heartbeat = now
                print(f"[engine] 心跳 {datetime.now():%H:%M:%S} "
                      f"active={self._active} recv={self._recv} signals={self._signals}",
                      flush=True)
            self._stop.wait(min(self._cfg.heartbeat_sec, 30))

        self._stop_feed("退出")
        print(f"[engine] 结束。累计快照={self._recv} 信号={self._signals}", flush=True)

    # ---- 优雅退出 ------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            print(f"[engine] 收到信号 {signum}，准备退出 ...", flush=True)
            self._stop.set()
        for s in (_signal.SIGINT, _signal.SIGTERM):
            try:
                _signal.signal(s, _handler)
            except (ValueError, OSError):
                pass  # 非主线程时忽略


def main() -> None:
    Engine().run()


if __name__ == "__main__":
    main()
