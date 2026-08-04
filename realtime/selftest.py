"""实时层骨架自测：不依赖券商连接、不依赖当前时钟，纯验证数据链路通不通。

跑法（容器内 Python 3.12）：
    python3 -m realtime.selftest
覆盖：
    1) 各模块 import
    2) Snapshot 字段映射 + 派生量（涨停判定/失衡度）
    3) 订阅清单加载（读到几只）
    4) DummyFeed → 策略 → notifier(干跑) → ledger 全链路
    5) 直接喂"封涨停"快照，验证信号产出 + 账本落盘
全程 dry_run，不登录券商、不发网络请求、不碰 quant_data。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"[selftest][OK] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[selftest][FAIL] {msg}", flush=True)


def main() -> int:
    errors = 0

    # 用临时账本目录 + 干跑，避免污染 logs
    tmp = Path(tempfile.mkdtemp(prefix="rt_selftest_"))
    os.environ["REALTIME_DRY_RUN"] = "1"
    os.environ["REALTIME_LEDGER_DIR"] = str(tmp)
    os.environ["REALTIME_NOTIFY_COOLDOWN"] = "0"

    # 1) import
    try:
        from realtime import (config, snapshot, watchlist, feed,  # noqa: F401
                              strategy, notifier, ledger, engine)
        _ok("import 全部模块")
    except Exception as e:  # noqa: BLE001
        _fail(f"import 失败: {type(e).__name__}: {e}")
        return 1

    from realtime.config import load
    from realtime.snapshot import Snapshot, from_mapping
    from realtime.strategy import (LimitMoveWatch, Signal, SurgeWatch, VolumeSurge,
                                    VWAPDeviation, ChandelierStop, GapCalibrate,
                                    HoldingExpiry, StrategyContext, default_strategies)
    from realtime.reference import RefRow
    from realtime.engine import Engine

    cfg = load()

    # 2) Snapshot 映射 + 派生量
    row = {"symbol": "600519", "last_price": 11.0, "prev_close": 10.0,
           "limit_up": 11.0, "limit_down": 9.0, "bid_vol1": 800, "ask_vol1": 200}
    s = from_mapping(row)
    if s.code == "600519" and abs((s.pct_change or 0) - 0.1) < 1e-6 and s.is_limit_up \
       and s.bid_ask_imbalance and s.bid_ask_imbalance > 0:
        _ok(f"字段映射 last={s.last} pct={s.pct_change:.2%} limit_up={s.is_limit_up} "
            f"imbalance={s.bid_ask_imbalance:.2f}")
    else:
        _fail(f"字段映射异常: {s}")
        errors += 1

    # 3) 订阅清单
    try:
        codes = watchlist.load_codes(cfg)
        _ok(f"订阅清单加载 {len(codes)} 只（前5 {codes[:5]}）")
    except Exception as e:  # noqa: BLE001
        _fail(f"清单加载失败: {type(e).__name__}: {e}")
        errors += 1

    # 4+5) 全链路：直接喂封涨停快照，验证信号 + 账本
    eng = Engine(cfg, strategies=[LimitMoveWatch()])
    limit_snap = Snapshot(code="000001", last=11.0, pre_close=10.0,
                          high_limited=11.0, low_limited=9.0,
                          bid_price1=10.99, ask_price1=11.0,
                          bid_volume1=500, ask_volume1=0)
    eng._on_snapshot(limit_snap)  # noqa: SLF001 - 自测直连内部处理链
    if not eng._drain_effects(timeout=1.0):  # noqa: SLF001
        _fail("信号副作用队列未在 1 秒内排空")
        errors += 1

    files = list(tmp.glob("signals_*.jsonl"))
    if not files:
        _fail("账本未生成文件")
        errors += 1
    else:
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        recs = [json.loads(x) for x in lines]
        up = [r for r in recs if r["kind"] == "limit_up"]
        if up:
            _ok(f"信号链路通：账本 {len(recs)} 条，含 limit_up（{up[0]['reason']}）")
        else:
            _fail(f"未产出 limit_up 信号，账本内容={recs}")
            errors += 1
    eng._shutdown_effect_dispatcher("自测")  # noqa: SLF001

    # 5a) 慢通知必须脱离行情回调，且通知完成后仍按顺序记账。
    gate = threading.Event()
    started = threading.Event()
    recorded: list[tuple[str, bool]] = []

    class _SlowNotifier:
        def notify(self, sig: Signal) -> bool:
            started.set()
            gate.wait(timeout=1.0)
            return True

    class _CaptureLedger:
        def record(self, sig: Signal, notified: bool) -> None:
            recorded.append((sig.kind, notified))

    async_eng = Engine(cfg, strategies=[LimitMoveWatch()])
    async_eng._notifier = _SlowNotifier()  # noqa: SLF001
    async_eng._ledger = _CaptureLedger()  # noqa: SLF001
    begin = time.monotonic()
    async_eng._on_snapshot(limit_snap)  # noqa: SLF001
    callback_elapsed = time.monotonic() - begin
    worker_started = started.wait(timeout=0.5)
    not_blocked = callback_elapsed < 0.1 and worker_started and not recorded
    gate.set()
    drained = async_eng._drain_effects(timeout=1.0)  # noqa: SLF001
    async_eng._shutdown_effect_dispatcher("异步自测")  # noqa: SLF001
    if not_blocked and drained and recorded == [("limit_up", True)]:
        _ok(f"慢通知已脱离行情回调（callback={callback_elapsed:.4f}s）")
    else:
        _fail(f"异步副作用异常：callback={callback_elapsed:.4f}s "
              f"started={worker_started} drained={drained} recorded={recorded}")
        errors += 1

    # 5b) 各实时策略单测：构造能触发的快照序列，验证 kind 产出
    def _check(label: str, kinds: list, want: str) -> int:
        if want in kinds:
            _ok(f"{label} 触发 {kinds}")
            return 0
        _fail(f"{label} 未触发 {want}: {kinds}")
        return 1

    # SurgeWatch：间隔>=1s、涨幅>=2% → surge_up
    sw = SurgeWatch(surge_pct=0.02, min_dt=0.0, cooldown=0.0)
    ctx_sw = StrategyContext()
    ctx_sw.update(Snapshot(code="600000", last=10.0, pre_close=10.0))
    ctx_sw.state_of("600000").prev_ts = time.time() - 5   # 伪造 5s 前
    ctx_sw.state_of("600000").prev_last = 10.0
    surge_kinds = [sig.kind for sig in sw.on_snapshot(
        Snapshot(code="600000", last=10.3, pre_close=10.0), ctx_sw)]
    errors += _check("SurgeWatch", surge_kinds, "surge_up")

    # VWAPDeviation：last 远低于 vwap(=amount/volume=10) → vwap_cheap
    vd = VWAPDeviation(dev=0.01, cooldown=0.0)
    ctx_vd = StrategyContext()
    sv = Snapshot(code="600001", last=9.5, pre_close=10.0, amount=10000.0, volume=1000.0)
    ctx_vd.update(sv)
    vwap_kinds = [sig.kind for sig in vd.on_snapshot(sv, ctx_vd)]
    errors += _check("VWAPDeviation", vwap_kinds, "vwap_cheap")

    # ChandelierStop：day_high 10.5，ATR 0.2，k=3 → stop=9.9；last=9.8 跌破
    cs = ChandelierStop(k=3.0)
    ctx_cs = StrategyContext(ref={"600002": RefRow(atr=0.2, atr_pct=0.02,
                                                   expected_return=0.05)})
    ctx_cs.update(Snapshot(code="600002", last=10.5, pre_close=10.0))  # 抬 day_high
    stop_kinds = [sig.kind for sig in cs.on_snapshot(
        Snapshot(code="600002", last=9.8, pre_close=10.0), ctx_cs)]
    errors += _check("ChandelierStop", stop_kinds, "chandelier_stop")

    # GapCalibrate：预期收益 5%，高开 4% → 吃掉 80% → gap_eaten
    gc = GapCalibrate(eat_ratio=0.6)
    ctx_gc = StrategyContext(ref={"600003": RefRow(expected_return=0.05)})
    sg = Snapshot(code="600003", last=10.4, pre_close=10.0, open=10.4)
    ctx_gc.update(sg)
    gap_kinds = [sig.kind for sig in gc.on_snapshot(sg, ctx_gc)]
    errors += _check("GapCalibrate", gap_kinds, "gap_eaten")

    # HoldingExpiry：持有 1 交易日、定档 T+1 → holding_expiry
    he = HoldingExpiry(sell_horizon=1)
    ctx_he = StrategyContext(ref={"600004": RefRow(hold_days=1)})
    sh = Snapshot(code="600004", last=10.2, pre_close=10.0)
    ctx_he.update(sh)
    he_kinds = [sig.kind for sig in he.on_snapshot(sh, ctx_he)]
    errors += _check("HoldingExpiry", he_kinds, "holding_expiry")

    # HoldingExpiry 负例：非持仓票(hold_days=None) 不应触发
    ctx_he2 = StrategyContext(ref={"600005": RefRow(expected_return=0.05)})
    sh2 = Snapshot(code="600005", last=10.2, pre_close=10.0)
    ctx_he2.update(sh2)
    if not HoldingExpiry(sell_horizon=1).on_snapshot(sh2, ctx_he2):
        _ok("HoldingExpiry 非持仓票不误报")
    else:
        _fail("HoldingExpiry 对非持仓票误触发")
        errors += 1

    # 高开惩罚已判负，默认只装配 6 条已验证策略；GapCalibrate 类仍保留供显式实验。
    ds = default_strategies()
    ds_names = [s.name for s in ds]
    if len(ds) == 6 and "gap_calibrate" not in ds_names:
        _ok(f"default_strategies 装配 6 条已验证策略：{ds_names}")
    else:
        _fail(f"default_strategies 装配异常: {ds_names}")
        errors += 1

    # 6) DummyFeed 短跑（验证 feed 线程 + 回调）
    got = {"n": 0}
    df = feed.DummyFeed(["000001", "000002"], lambda s: got.__setitem__("n", got["n"] + 1),
                        tick_interval=0.05)
    df.start()
    time.sleep(0.3)
    df.stop()
    if got["n"] > 0:
        _ok(f"DummyFeed 回调 {got['n']} 条")
    else:
        _fail("DummyFeed 无回调")
        errors += 1

    # 清理临时账本
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()

    print(f"[selftest] {'全部通过' if errors == 0 else str(errors) + ' 项失败'}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
