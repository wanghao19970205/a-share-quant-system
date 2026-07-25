"""推送通道：Bark(iOS 首选) / Server酱 / PushDeer，带按 (code, kind) 的冷却节流。

- 无凭证时进入"干跑"模式：只打印 + 交给账本记录，不发网络请求。
- 网络请求失败不抛异常（实时层不能因推送失败而中断），只打印告警。
- 冷却：同一 (code, kind) 在 cooldown 秒内只推一次，防止逼近涨停每 tick 刷屏。
- 多通道可同时配置；任一成功即视为已发出。
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request

from .config import RealtimeConfig
from .strategy import Signal


class Notifier:
    def __init__(self, cfg: RealtimeConfig):
        self._cfg = cfg
        self._last_sent: dict[tuple[str, str], float] = {}

    # ---- 节流 ----------------------------------------------------------------
    def _cooled(self, sig: Signal) -> bool:
        key = (sig.code, sig.kind)
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if now - last < self._cfg.notify_cooldown_sec:
            return False
        self._last_sent[key] = now
        return True

    # ---- 对外 ----------------------------------------------------------------
    def notify(self, sig: Signal) -> bool:
        """推送一条信号。返回是否实际发出（被节流/干跑则 False）。"""
        if not self._cooled(sig):
            return False
        title = f"[{sig.level}] {sig.code} {sig.kind}"
        body = sig.reason or sig.kind
        sent = False
        if self._cfg.bark_key:
            sent = self._send_bark(title, body) or sent
        if self._cfg.serverchan_key:
            sent = self._send_serverchan(title, body) or sent
        if self._cfg.pushdeer_key:
            sent = self._send_pushdeer(title, body) or sent
        if not (self._cfg.bark_key or self._cfg.serverchan_key or self._cfg.pushdeer_key):
            print(f"[notify:dry] {title} | {body}", flush=True)
        return sent

    # ---- 通道实现 ------------------------------------------------------------
    def _post(self, url: str, data: dict) -> bool:
        try:
            payload = urllib.parse.urlencode(data).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                return 200 <= resp.status < 300
        except Exception as e:  # noqa: BLE001 - 推送失败不影响主流程
            print(f"[notify] 推送失败({url.split('//')[-1].split('/')[0]}): {type(e).__name__}", flush=True)
            return False

    def _send_bark(self, title: str, body: str) -> bool:
        # Bark: POST <endpoint>/<device_key>，表单 title/body；group 便于在通知中心分组。
        url = f"{self._cfg.bark_endpoint.rstrip('/')}/{self._cfg.bark_key}"
        return self._post(url, {"title": title, "body": body, "group": "A股实时"})

    def _send_serverchan(self, title: str, body: str) -> bool:
        # Server酱 Turbo: https://sctapi.ftqq.com/<SCKEY>.send
        url = f"https://sctapi.ftqq.com/{self._cfg.serverchan_key}.send"
        return self._post(url, {"title": title, "desp": body})

    def _send_pushdeer(self, title: str, body: str) -> bool:
        url = f"{self._cfg.pushdeer_endpoint.rstrip('/')}/message/push"
        return self._post(url, {"pushkey": self._cfg.pushdeer_key,
                                "text": title, "desp": body, "type": "markdown"})
