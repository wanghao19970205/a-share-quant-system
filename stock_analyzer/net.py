"""代理路由工具（借鉴开源量化项目的思路）。

核心：用 unittest.mock.patch 无侵入地把 AKShare 内部的 requests.get 替换成
「带代理轮转 + 会话净化 + urllib3 重试」的 getter，从而在配置了代理时，
让东财等被本地网络屏蔽的数据源恢复可用；未配置代理时为 no-op，行为不变。
"""
from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (ChunkedEncodingError, ConnectionError,
                                 ConnectTimeout, ProxyError, ReadTimeout, SSLError)

try:
    from urllib3.util.retry import Retry
except Exception:  # noqa: BLE001
    Retry = None

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

_RETRIABLE = (ProxyError, ConnectTimeout, ReadTimeout, SSLError,
              ConnectionError, ChunkedEncodingError)

# 代理 URL 列表，如 ["http://127.0.0.1:7890", "http://user:pass@ip:port"]
_PROXIES: list[str] = []

# 代理补丁的线程安全装载状态（引用计数 + 锁）
_patch_lock = threading.Lock()
_patch_depth = 0
_orig_get = None


def set_proxies(proxies: list[str]) -> None:
    """设置代理列表（供 UI/环境变量配置，作用于 akshare 的东财等请求）。"""
    global _PROXIES
    _PROXIES = [p.strip() for p in (proxies or []) if p and p.strip()]


def has_proxy() -> bool:
    return bool(_PROXIES)


def _proxy_dicts() -> list:
    return [{"http": p, "https": p} for p in _PROXIES]


def _build_rotating_get(timeout: int = 15, per_proxy_retries: int = 1,
                        include_direct: bool = True):
    def rotating_get(url, **kwargs):
        pool = _proxy_dicts()
        if include_direct:
            pool = pool + [None]          # 追加直连兜底
        random.shuffle(pool)
        last_exc = None
        for proxy in pool:
            s = requests.Session()
            try:
                s.trust_env = False       # 不使用系统环境代理，避免干扰
                s.headers.update({
                    "User-Agent": _UA, "Accept": "*/*",
                    "Connection": "close", "Referer": "https://quote.eastmoney.com/",
                })
                if proxy:
                    s.proxies.update(proxy)
                if Retry is not None:
                    retry = Retry(
                        total=per_proxy_retries, connect=per_proxy_retries,
                        read=per_proxy_retries, backoff_factor=0.5,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET"], respect_retry_after_header=True,
                        raise_on_status=False)
                    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
                    s.mount("http://", adapter)
                    s.mount("https://", adapter)
                kw = dict(kwargs)
                kw.setdefault("timeout", timeout)
                resp = s.get(url, **kw)
                resp.raise_for_status()
                return resp
            except _RETRIABLE as e:  # noqa: PERF203 逐代理切换
                last_exc = e
                time.sleep(random.uniform(0.3, 0.8))
                continue
            finally:
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
        if last_exc:
            raise last_exc
        raise RuntimeError("请求失败且无可用代理")

    return rotating_get


@contextmanager
def akshare_proxied():
    """上下文内把 requests.get 路由到代理 getter；未配置代理则为 no-op。

    线程安全：多线程并发拉取时，用锁 + 引用计数保证代理 getter 只装载一次、
    在最后一个使用者退出时才还原，避免 ``mock.patch`` 全局替换在并发下相互干扰
    （否则可能出现某线程提前还原、或补丁泄漏未还原）。
    """
    if not _PROXIES:
        yield
        return
    global _patch_depth, _orig_get
    with _patch_lock:
        if _patch_depth == 0:
            _orig_get = requests.get
            requests.get = _build_rotating_get()
        _patch_depth += 1
    try:
        yield
    finally:
        with _patch_lock:
            _patch_depth -= 1
            if _patch_depth == 0:
                requests.get = _orig_get
                _orig_get = None
