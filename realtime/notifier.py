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
    def __init__(self, cfg: RealtimeConfig, name_map: dict[str, str] | None = None):
        self._cfg = cfg
        self._name_map = name_map or {}
        self._last_sent: dict[tuple[str, str], float] = {}

    # ---- 展示 ----------------------------------------------------------------
    def _label(self, code: str) -> str:
        """代码 + 中文简称（如 "600519 贵州茅台"）；无名称则退回纯代码。

        name_map 以 6 位纯代码为 key，但信号里的 code 是券商回调的 broker 格式
        （带交易所后缀，如 "603956.SH"）。故先按原样查，miss 再退回规范化的 6 位
        （去后缀、zfill）查一次，两种口径都能命中。
        """
        name = self._name_map.get(code)
        if name is None:
            digits = str(code).split(".", 1)[0].strip().zfill(6)
            name = self._name_map.get(digits)
        return f"{code} {name}" if name else code

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
        """推送一条通用信号；关闭时返回 False，调用方仍会将信号写入账本。"""
        if not getattr(self._cfg, "signal_push_enabled", False):
            return False
        # 白名单过滤：只推 notify_kinds 内的 kind（空集=不过滤）。被挡的仍会记账。
        if self._cfg.notify_kinds and sig.kind not in self._cfg.notify_kinds:
            return False
        if not self._cooled(sig):
            return False
        title = f"[{sig.level}] {self._label(sig.code)} {sig.kind}"
        body = sig.reason or sig.kind
        return self.push(title, body)

    def push(self, title: str, body: str) -> bool:
        """低层推送：把 (title, body) 派发到所有已配置通道，任一成功即 True。

        不过白名单、不做 (code, kind) 冷却——供 RankBoard 等主动汇总类推送直接调用。
        无凭证则干跑打印。
        """
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
