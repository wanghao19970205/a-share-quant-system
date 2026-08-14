"""主循环 + 生命周期：装配 feed/strategy/notifier/ledger，按交易时段驻场运行。

生命周期（本地时间）：
    < session_start        : 未开盘，休眠等待
    [start, morning_close] : 盘中，订阅活跃
    (morning_close, afternoon_open): 午休，快照静默（订阅保持，无推送属正常）
    [afternoon_open, end]  : 盘中，订阅活跃
    > session_end          : 收盘，停订阅并退出
    [avoid_start, avoid_end]: daily-light 规避窗，暂停订阅避免抢券商连接

信号处理链：feed → strategy.on_snapshot → 有界队列 → notifier.notify → ledger.record。
策略和上下文更新保持同步顺序，网络与文件副作用由单消费者异步执行。
"""
from __future__ import annotations

import os
import queue
import signal as _signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import feed as feed_mod
from . import names as names_mod
from . import reference as ref_mod
from . import watchlist as wl_mod
from .config import RealtimeConfig, load
from .ledger import Ledger
from .notifier import Notifier
from .paper_trader import PaperTrader
from .v2 import V2PaperTrader
from .v3 import V3PaperTrader
from .v4 import V4PaperTrader
from .v5 import V5PaperTrader
from .rankboard import RankBoard
from .sector_etf import SectorETFContext
from .snapshot import Snapshot
from .strategy import Signal, Strategy, StrategyContext, default_strategies


def _hhmm(now: datetime | None = None) -> int:
    now = now or datetime.now()
    return now.hour * 100 + now.minute


def _mtime(path: Path) -> float:
    """文件 mtime（不存在返回 0.0）；用于低成本探测名单/持仓文件是否变化。"""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


class Engine:
    def __init__(self, cfg: RealtimeConfig | None = None,
                 strategies: list[Strategy] | None = None):
        self._cfg = cfg or load()
        self._strategies = strategies if strategies is not None else default_strategies()
        self._ctx = StrategyContext()
        self._notifier = Notifier(self._cfg)
        self._ledger = Ledger(self._cfg)
        self._rankboard = None  # 实时买入候选榜（run() 就绪后装配）
        self._paper = None     # 实时模拟盘 V1（run() 就绪后装配）
        self._paper_v2 = None  # 实时模拟盘 V2（赛马对照）
        self._paper_v3 = None  # 实时模拟盘 V3（执行确认 + ATR 出场）
        self._paper_v4 = None  # 实时模拟盘 V4（V3 + 行业 ETF 弱势回避）
        self._paper_v5 = None  # 实时模拟盘 V5（V4 + 行业相对强度动态仓位）
        self._sector_ctx = SectorETFContext(self._cfg)
        self._sector_recv = 0
        self._feed: feed_mod.BaseFeed | None = None
        self._stop = threading.Event()
        self._active = False          # 当前是否处于活跃订阅态
        self._recv = 0                # 收到快照计数（心跳打印用）
        self._signals = 0
        self._effect_q: queue.Queue = queue.Queue(maxsize=self._cfg.effect_queue_size)
        self._effect_stop = threading.Event()
        self._effect_thread: threading.Thread | None = None
        self._effect_start_lock = threading.Lock()
        self._effects_dispatched = 0
        self._effects_dropped = 0
        self._codes_key: frozenset = frozenset()  # 当前订阅码集指纹（供盘中重载比对）
        self._src_mtime: tuple[float, ...] = ()   # 名单/持仓源上次读到的 mtime

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
    def _ensure_effect_dispatcher(self) -> None:
        """按需启动单消费者，保持信号通知和账本的 FIFO 顺序。"""
        if self._effect_thread is not None and self._effect_thread.is_alive():
            return
        with self._effect_start_lock:
            if self._effect_thread is not None and self._effect_thread.is_alive():
                return
            if self._effect_stop.is_set():
                return
            self._effect_thread = threading.Thread(
                target=self._effect_loop, name="realtime-effects", daemon=True)
            self._effect_thread.start()

    def _effect_loop(self) -> None:
        while not self._effect_stop.is_set() or not self._effect_q.empty():
            try:
                sig = self._effect_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                notified = self._notifier.notify(sig)
                self._ledger.record(sig, notified)
                self._effects_dispatched += 1
            except Exception as e:  # noqa: BLE001 - 副作用异常不终止消费者
                print(f"[engine] 信号副作用异常: {type(e).__name__}", flush=True)
            finally:
                self._effect_q.task_done()

    def _dispatch_effect(self, sig: Signal) -> None:
        self._ensure_effect_dispatcher()
        try:
            self._effect_q.put_nowait(sig)
        except queue.Full:
            self._effects_dropped += 1
            # 首次及每 100 次打印，避免通知端故障时日志本身成为新瓶颈。
            if self._effects_dropped == 1 or self._effects_dropped % 100 == 0:
                print(f"[engine] 信号副作用队列已满，累计丢弃={self._effects_dropped} "
                      f"capacity={self._effect_q.maxsize}", flush=True)

    def _drain_effects(self, timeout: float | None = None) -> bool:
        """有限等待已入队任务完成；返回是否全部排空。"""
        grace = self._cfg.effect_shutdown_grace_sec if timeout is None else max(0.0, timeout)
        deadline = time.monotonic() + grace
        while self._effect_q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return self._effect_q.unfinished_tasks == 0

    def _shutdown_effect_dispatcher(self, reason: str) -> None:
        """停止接收并尽力排空；通知超时不能阻止收盘或热重载。"""
        self._effect_stop.set()
        drained = self._drain_effects()
        thread = self._effect_thread
        if thread is not None and drained:
            thread.join(timeout=0.2)
        if not drained:
            print(f"[engine] 信号副作用排空超时({reason})：pending="
                  f"{self._effect_q.unfinished_tasks} grace="
                  f"{self._cfg.effect_shutdown_grace_sec:.1f}s", flush=True)

    def _on_snapshot(self, snap: Snapshot) -> None:
        self._recv += 1
        if self._sector_ctx.is_sector_code(snap.code):
            self._sector_recv += 1
            for sig in self._sector_ctx.update(snap):
                self._signals += 1
                self._dispatch_effect(sig)
            return
        self._ctx.update(snap)
        for strat in self._strategies:
            try:
                sigs = strat.on_snapshot(snap, self._ctx)
            except Exception as e:  # noqa: BLE001 - 单策略异常不拖垮全局
                print(f"[engine] 策略 {strat.name} 异常: {type(e).__name__}", flush=True)
                continue
            for sig in sigs:
                self._signals += 1
                self._dispatch_effect(sig)

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

    # ---- 盘中名单自动重载 ----------------------------------------------------
    def _src_mtimes(self) -> tuple[float, ...]:
        """名单、预测和持仓源 mtime 指纹；变化后重建订阅及参考数据。"""
        return (
            _mtime(self._cfg.mobile_snapshot_file),
            _mtime(self._cfg.predictions_file),
            _mtime(self._cfg.holdings_file),
            *(_mtime(p) for p in wl_mod._paper_state_files(self._cfg)),
            _mtime(self._cfg.sector_meta_file),
        )

    def _maybe_reload(self) -> None:
        """盘中检测预测、名单和持仓更新；预测或码集变化时 execv 自我重启。

        AmazingData SDK 无可靠优雅 stop（见 feed.AmazingDataFeed.stop 注释），进程内换订阅
        会残留旧订阅线程 → 重复推送 + 抢券商连接。故用 os.execv 整体替换进程映像：daemon
        订阅线程与券商连接随 exec 消失，新进程重新登录+订阅最新名单。PID 不变，cron 幂等守卫
        （扫 /proc 找 realtime.engine）仍视其在跑，不会重复拉起。
        """
        if not self._cfg.watchlist_reload:
            return
        cur = self._src_mtimes()
        if cur == self._src_mtime:
            return  # 文件都没动，跳过 IO/解析（每心跳只做一次廉价 stat）
        previous = self._src_mtime
        self._src_mtime = cur
        prediction_changed = bool(previous) and cur[1] != previous[1]
        sector_meta_changed = bool(previous) and cur[-1] != previous[-1]
        try:
            new_codes = wl_mod.load_codes(self._cfg)
        except Exception as e:  # noqa: BLE001 - 读清单异常不重启，维持现订阅
            print(f"[engine] 重载检查读清单失败(维持现订阅)：{type(e).__name__}", flush=True)
            return
        new_key = frozenset(new_codes)
        if not new_key:
            return  # 空清单不切，防误清空
        if new_key == self._codes_key and not prediction_changed and not sector_meta_changed:
            return
        if prediction_changed:
            reason = "预测更新"
        elif sector_meta_changed:
            reason = "行业映射更新"
        else:
            reason = "订阅清单更新"
        print(f"[engine] {reason}（代码数 {len(self._codes_key)}→{len(new_key)}），"
              "重启引擎刷新参考数据和订阅 ...", flush=True)
        # AmazingDataFeed.stop() 在真实盘中连接上偶发 SIGSEGV，会使进程在 execv 前消失。
        # 直接替换进程映像，由内核关闭旧连接和线程；只先排空我们自己的异步副作用队列。
        self._shutdown_effect_dispatcher("清单热重载")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, "-m", "realtime.engine"])

    # ---- 主循环 --------------------------------------------------------------
    def run(self) -> None:
        self._install_signal_handlers()
        # Validate before prediction fallback can populate subscriptions or references.
        ref_mod.assert_prediction_fresh(self._cfg)
        codes = wl_mod.load_codes(self._cfg)
        if not codes:
            print("[engine] 订阅清单为空（选股清单/持仓/兜底池都没读到），退出。", flush=True)
            return
        sector_codes = self._sector_ctx.subscription_codes()
        subscription_codes = list(codes)
        seen_subscription = {str(c).upper() for c in subscription_codes}
        subscription_codes.extend(
            c for c in sector_codes if str(c).upper() not in seen_subscription)
        # 启动期构建 code->中文简称映射（离线读本地 meta），注入 notifier 让推送带股票名。
        name_map = {}
        try:
            name_map = names_mod.load_name_map(codes)
            self._notifier = Notifier(self._cfg, name_map)
            print(f"[engine] 名称映射就绪：{len(name_map)}/{len(codes)} 只有中文简称",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - 名称缺失只影响展示，退回纯代码
            print(f"[engine] 名称映射构建失败(降级为纯代码)：{type(e).__name__}", flush=True)
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
        # 装配实时买入候选榜（跨票聚合器）：按模型预期收益排名 + 盘中量标注，主循环内定期推 digest。
        if self._cfg.rank_board_enabled:
            self._rankboard = RankBoard(
                self._cfg, self._ctx, self._notifier, name_map, self._sector_ctx)
            print(f"[engine] 实时候选榜就绪：每 {self._cfg.rank_interval_sec}s 推 Top{self._cfg.rank_top_n}"
                  f"（仅榜单变化时）", flush=True)
        # 装配实时模拟盘：收盘前先执行 T+N 到期腿，再按模型 Top-N 建新仓。
        if self._cfg.paper_trade_enabled:
            self._paper = PaperTrader(self._cfg, self._ctx, self._notifier, name_map)
            print(f"[engine] 模拟盘就绪：{self._paper.summary()}，"
                  f"{self._cfg.paper_time_cap_start} 后到期卖、买窗"
                  f"{self._paper._buy_start}-{self._paper._buy_end} "
                  f"Top{self._cfg.paper_buy_n}（T+{self._cfg.sell_horizon}）", flush=True)
        if getattr(self._cfg, "paper_v2_enabled", True):
            # V2 是赛马对照账户，其初始化异常绝不能拖垮已装配好的 V1 现役策略。
            try:
                self._paper_v2 = V2PaperTrader(
                    self._cfg, self._ctx, self._notifier, name_map)
                print(f"[engine] 模拟盘V2就绪：{self._paper_v2.summary()}，"
                      f"保护止盈{self._paper_v2._breakeven_arm:+.0%} | "
                      f"持仓上限{self._paper_v2._max_positions} | "
                      f"动态分配 | 买窗{self._cfg.paper_buy_start}-{self._paper_v2._buy_end}",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - V2 降级不影响 V1
                self._paper_v2 = None
                print(f"[engine] 模拟盘V2装配失败(降级跳过，V1 不受影响)："
                      f"{type(e).__name__}: {e}", flush=True)
        if getattr(self._cfg, "paper_v3_enabled", True):
            try:
                self._paper_v3 = V3PaperTrader(
                    self._cfg, self._ctx, self._notifier, name_map)
                print(f"[engine] 模拟盘V3就绪：{self._paper_v3.summary()}，"
                      f"ask1买/bid1卖 | 新鲜度≤{self._paper_v3._quote_max_age:.0f}s | "
                      f"ATR风险单位={self._paper_v3._atr_k:g}x",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - V3 独立降级，不影响 V1/V2
                self._paper_v3 = None
                print(f"[engine] 模拟盘V3装配失败(降级跳过，V1/V2 不受影响)："
                      f"{type(e).__name__}: {e}", flush=True)
        if getattr(self._cfg, "paper_v4_enabled", True):
            try:
                self._paper_v4 = V4PaperTrader(
                    self._cfg, self._ctx, self._notifier, self._sector_ctx, name_map)
                print(f"[engine] 模拟盘V4就绪：{self._paper_v4.summary()}，"
                      f"V3执行/ATR规则 | 行业ETF弱势门槛≤"
                      f"{self._sector_ctx.weak_threshold:+.2%} | 精确映射"
                      f"{self._sector_ctx.mapped_count(codes)}/{len(codes)} "
                      f"({self._sector_ctx.mapping_version})", flush=True)
            except Exception as e:  # noqa: BLE001 - V4 独立降级，不影响 V1/V2/V3
                self._paper_v4 = None
                print(f"[engine] 模拟盘V4装配失败(降级跳过，V1/V2/V3 不受影响)："
                      f"{type(e).__name__}: {e}", flush=True)
        if getattr(self._cfg, "paper_v5_enabled", True):
            try:
                self._paper_v5 = V5PaperTrader(
                    self._cfg, self._ctx, self._notifier, self._sector_ctx, name_map)
                print(f"[engine] 模拟盘V5就绪：{self._paper_v5.summary()}，"
                      f"V4规则 | 行业仓位系数"
                      f"{self._paper_v5._sector_lagging_factor:.2f}/"
                      f"{self._paper_v5._sector_neutral_factor:.2f}/"
                      f"{self._paper_v5._sector_strong_factor:.2f}", flush=True)
            except Exception as e:  # noqa: BLE001 - V5 独立降级，不影响 V1-V4
                self._paper_v5 = None
                print(f"[engine] 模拟盘V5装配失败(降级跳过，V1-V4 不受影响)："
                      f"{type(e).__name__}: {e}", flush=True)
        print(f"[engine] 启动实时层：股票 {len(codes)} 只 + 行业ETF/基准 "
              f"{len(sector_codes)} 只，同一会话共 {len(subscription_codes)} 只；"
              f"时段 {self._cfg.session_start}-{self._cfg.session_end}，"
              f"策略 {[s.name for s in self._strategies]}", flush=True)

        # 记录初始订阅码集指纹 + 名单/持仓源 mtime，供盘中自动重载比对。
        self._codes_key = frozenset(codes)
        self._src_mtime = self._src_mtimes()

        last_heartbeat = 0.0
        while not self._stop.is_set():
            t = _hhmm()
            if self._past_session(t):
                self._stop_feed("收盘")
                print("[engine] 已过收盘时间，退出主循环。", flush=True)
                break
            if self._in_session(t):
                self._start_feed(subscription_codes)
                self._maybe_reload()  # 盘中检测名单/持仓变化 → execv 重启换订阅
                if self._rankboard is not None:
                    try:
                        self._rankboard.maybe_emit()  # 到间隔且榜单变化则推候选榜 digest
                    except Exception as e:  # noqa: BLE001 - 榜单异常不拖垮主循环
                        print(f"[engine] 候选榜推送异常: {type(e).__name__}", flush=True)
                if self._paper is not None:
                    try:
                        self._paper.maybe_trade(t)  # 收盘前交易窗内买 Top-N / 卖到期
                    except Exception as e:  # noqa: BLE001 - 模拟盘异常不拖垮主循环
                        print(f"[engine] 模拟盘交易异常: {type(e).__name__}", flush=True)
                if self._paper_v2 is not None:
                    try:
                        self._paper_v2.maybe_trade(t)
                    except Exception as e:  # noqa: BLE001
                        print(f"[engine] 模拟盘V2交易异常: {type(e).__name__}", flush=True)
                if self._paper_v3 is not None:
                    try:
                        self._paper_v3.maybe_trade(t)
                    except Exception as e:  # noqa: BLE001
                        print(f"[engine] 模拟盘V3交易异常: {type(e).__name__}", flush=True)
                if self._paper_v4 is not None:
                    try:
                        self._paper_v4.maybe_trade(t)
                    except Exception as e:  # noqa: BLE001
                        print(f"[engine] 模拟盘V4交易异常: {type(e).__name__}", flush=True)
                if self._paper_v5 is not None:
                    try:
                        self._paper_v5.maybe_trade(t)
                    except Exception as e:  # noqa: BLE001
                        print(f"[engine] 模拟盘V5交易异常: {type(e).__name__}", flush=True)
            else:
                self._stop_feed("非交易时段/规避窗")

            now = time.time()
            if now - last_heartbeat >= self._cfg.heartbeat_sec:
                last_heartbeat = now
                print(f"[engine] 心跳 {datetime.now():%H:%M:%S} "
                      f"active={self._active} recv={self._recv} sector_recv={self._sector_recv} "
                      f"sector={self._sector_ctx.summary()} signals={self._signals} "
                      f"effects_pending={self._effect_q.unfinished_tasks} "
                      f"effects_done={self._effects_dispatched} "
                      f"effects_dropped={self._effects_dropped}", flush=True)
            self._stop.wait(min(self._cfg.heartbeat_sec, 30))

        self._stop_feed("退出")
        self._shutdown_effect_dispatcher("退出")
        print(f"[engine] 结束。累计快照={self._recv} 信号={self._signals} "
              f"副作用完成={self._effects_dispatched} 丢弃={self._effects_dropped}", flush=True)

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
