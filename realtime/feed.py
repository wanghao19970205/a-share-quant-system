"""订阅流封装：统一把 AmazingData onSnapshot 回调 / dry-run 假流 转成 Snapshot 回调。

对上层（engine）暴露统一接口：
    feed = make_feed(cfg, codes, on_snapshot=callback)
    feed.start()   # 非阻塞：内部起线程跑 sub.run() / 假流
    feed.stop()

真实流走 AmazingData SubscribeData（Push 通道，仅盘中有推送）；
dry-run 走 DummyFeed，按固定节奏喂造快照，用于本地验证骨架不依赖券商连接。

字段名以手册 4.2.1 为准，经 snapshot.from_obj() 的别名映射吸收实际差异。
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable, Sequence

from . import snapshot as snap_mod
from .config import RealtimeConfig
from .snapshot import Snapshot

OnSnapshot = Callable[[Snapshot], None]


def subscription_code(code: str, converter) -> str:
    """保留 SDK 返回的完整代码；仅对旧的 6 位股票代码推断交易所。"""
    value = str(code or "").strip().upper()
    return value if "." in value else converter(value)


class BaseFeed:
    def start(self) -> None:  # pragma: no cover - 接口
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - 接口
        raise NotImplementedError


class DummyFeed(BaseFeed):
    """假流：按 tick_interval 给每只票喂一条随机游走快照，供本地跑通骨架。"""

    def __init__(self, codes: Sequence[str], on_snapshot: OnSnapshot,
                 tick_interval: float = 1.0):
        self._codes = list(codes)
        self._cb = on_snapshot
        self._interval = tick_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, float] = {c: 10.0 for c in self._codes}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            for c in self._codes:
                base = self._state[c]
                last = round(base * (1 + random.uniform(-0.02, 0.02)), 2)
                self._state[c] = last
                pre = round(base, 2)
                s = Snapshot(
                    code=c,
                    trade_time=time.strftime("%H:%M:%S"),
                    last=last, pre_close=pre,
                    high_limited=round(pre * 1.1, 2),
                    low_limited=round(pre * 0.9, 2),
                    bid_price1=round(last - 0.01, 2), ask_price1=round(last + 0.01, 2),
                    bid_volume1=random.randint(1, 1000),
                    ask_volume1=random.randint(1, 1000),
                    raw={"dummy": True},
                )
                try:
                    self._cb(s)
                except Exception:  # noqa: BLE001 - 假流不因回调异常中断
                    pass
                if self._stop.is_set():
                    return
            self._stop.wait(self._interval)


class AmazingDataFeed(BaseFeed):
    """真实订阅流：AmazingData SubscribeData.onSnapshot（Push 通道）。

    登录复用 stock_analyzer.amazingdata_source 的凭证与 _to_broker_code。
    sub.run() 阻塞，故放独立守护线程；stop() 依赖进程退出回收（SDK 无优雅 stop 时）。
    """

    def __init__(self, codes: Sequence[str], on_snapshot: OnSnapshot):
        self._codes = list(codes)
        self._cb = on_snapshot
        self._thread: threading.Thread | None = None
        self._sub = None
        self._ad = None

    def start(self) -> None:
        import AmazingData as ad
        from stock_analyzer import amazingdata_source as ads

        ads._load_env_if_empty()
        c = ads._CREDS
        if not all([c.get("username"), c.get("password"), c.get("host"), c.get("port")]):
            raise RuntimeError("未配置券商账号，无法启动订阅流")
        ad.login(username=c["username"], password=c["password"],
                 host=c["host"], port=int(c["port"]))

        broker_codes = [subscription_code(x, ads._to_broker_code) for x in self._codes]
        self._ad = ad
        sub = ad.SubscribeData()
        cb = self._cb

        @sub.register(code_list=broker_codes,
                      period=ad.constant.Period.snapshot.value)
        def _on(data, period):  # noqa: ARG001
            try:
                cb(snap_mod.from_obj(data))
            except Exception:  # noqa: BLE001 - 单条异常不拖垮订阅
                pass

        self._sub = sub
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._sub.run()
        except Exception as e:  # noqa: BLE001
            print(f"[feed] 订阅 run() 异常: {type(e).__name__}", flush=True)

    def stop(self) -> None:
        # SDK 无公开 stop；订阅线程为 daemon，随进程退出回收。
        stop = getattr(self._sub, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:  # noqa: BLE001
                pass


def make_feed(cfg: RealtimeConfig, codes: Sequence[str],
              on_snapshot: OnSnapshot) -> BaseFeed:
    if cfg.dry_run:
        return DummyFeed(codes, on_snapshot)
    return AmazingDataFeed(codes, on_snapshot)
