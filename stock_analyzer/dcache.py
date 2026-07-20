"""磁盘持久化缓存：带 TTL 的函数结果缓存，跨进程 / 重启依然有效。

与 ``functools.lru_cache`` 的区别：
- 结果 pickle 落盘（默认 ``./.cache``，可用环境变量 CACHE_DIR 覆盖），进程重启后仍命中；
- 带 TTL（存活时间），到期自动重新拉取，避免数据陈旧——时效性强的（新闻）用短 TTL，
  历史行情（K线）可用长 TTL；
- 只缓存**成功返回值**，函数抛异常时不写缓存、异常照常向上抛出（不会把失败固化）。

这是纯缓存层：不改变数据内容、不丢弃任何数据，只是把「已经拉过的」在 TTL 内复用。
"""
from __future__ import annotations

import functools
import hashlib
import os
import pickle
import tempfile
import time


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# 各类数据的默认 TTL（秒），均可用环境变量覆盖：
#   K线：30 分钟——盘中会形成新bar，避免长时间冻结最新价；重复查询仍瞬时。
#   新闻：30 分钟——保证盘中新闻不会长时间遗漏（比原来进程内永不刷新的 lru_cache 更新鲜）。
#   外围/板块：30 分钟。
# 想要「整日缓存」，把对应环境变量设为 86400 即可。
def kline_ttl() -> int:
    return _env_int("CACHE_TTL_KLINE", 1800)


def news_ttl() -> int:
    return _env_int("CACHE_TTL_NEWS", 1800)


def market_ttl() -> int:
    return _env_int("CACHE_TTL_MARKET", 1800)


def cache_dir() -> str:
    d = os.environ.get("CACHE_DIR", ".cache")
    os.makedirs(d, exist_ok=True)
    return d


def _key(tag: str, args: tuple, kwargs: dict) -> str:
    raw = repr((tag, args, tuple(sorted(kwargs.items()))))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def disk_cache(ttl, name: str = ""):
    """装饰器：把函数结果按 (函数名+参数) 落盘缓存，TTL 秒内命中。

    Args:
        ttl: 存活秒数，或返回秒数的可调用对象（延迟读取环境变量）。
        name: 缓存标签（默认用函数名），用于文件名前缀与 key。
    """
    def deco(fn):
        tag = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            secs = ttl() if callable(ttl) else ttl
            path = os.path.join(cache_dir(), f"{tag}_{_key(tag, args, kwargs)}.pkl")
            try:
                if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < secs:
                    with open(path, "rb") as f:
                        return pickle.load(f)
            except Exception:  # noqa: BLE001 缓存损坏则忽略，重新计算
                pass
            val = fn(*args, **kwargs)   # 异常自然抛出，不缓存失败
            try:
                fd, tmp = tempfile.mkstemp(dir=cache_dir())
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(val, f)
                os.replace(tmp, path)  # 原子替换，多线程/多进程安全
            except Exception:  # noqa: BLE001 落盘失败不影响返回
                pass
            return val

        wrapper.cache_path_prefix = tag  # 便于排查

        def cache_clear():
            """删除该函数在磁盘上的所有缓存文件（用于代理/账号变更后强制重取）。"""
            try:
                d = cache_dir()
                for fn_ in os.listdir(d):
                    if fn_.startswith(tag + "_") and fn_.endswith(".pkl"):
                        try:
                            os.remove(os.path.join(d, fn_))
                        except OSError:
                            pass
            except Exception:  # noqa: BLE001
                pass

        wrapper.cache_clear = cache_clear
        return wrapper

    return deco
