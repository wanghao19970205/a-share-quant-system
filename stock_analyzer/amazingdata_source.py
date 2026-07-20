"""银河证券 星耀数智 AmazingData(tgw SDK) 数据源适配层（可选）。

作为 A股行情的**优先数据源**：SDK 已安装且登录成功时使用（官方、权威、稳定，
可绕过东财被屏蔽的问题）；否则自动降级到 akshare(新浪/腾讯/东财) 免费源。

前置条件（需你自行完成，SDK 非 pip 公共包）：
1. 从银河网盘下载对应 Python 版本的 wheel（本机 Python 3.9 → 选 cp39）：
   https://cloud.chinastock.com.cn/p/DSG36jYQx2IY_Y8CIAA
2. 安装：
   pip install tgw-1.7.1-py3-none-any.whl
   pip install AmazingData-1.0.0-cp39-none-any.whl
3. 配置账号（营业部申请开通）——通过环境变量或 set_credentials()：
   AMAZINGDATA_USER / AMAZINGDATA_PASSWORD / AMAZINGDATA_HOST / AMAZINGDATA_PORT

文档 API 摘要：
    import AmazingData as ad
    ad.login(username=?, password=?, host=IP, port=PORT)
    base = ad.BaseData(); cal = base.get_calendar()
    md = ad.MarketData(cal)
    kl = md.query_kline([code], begin_date=YYYYMMDD, end_date=YYYYMMDD,
                        period=ad.constant.Period.day.value)
    df = kl[code]
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import pandas as pd

try:
    import AmazingData as _ad  # noqa: N813
    _import_error = ""
except Exception as e:  # noqa: BLE001 未安装或原生库加载失败
    _ad = None
    _import_error = f"{type(e).__name__}: {e}"

_CREDS = {"username": "", "password": "", "host": "", "port": 0}
_login_lock = threading.Lock()
# 单次 SDK 调用超时（秒）：个别请求无返回时不再永久挂起 UI，可用环境变量覆盖。
_SDK_TIMEOUT = float(os.environ.get("AMAZINGDATA_TIMEOUT", "15") or 15)
# 券商基本面一次批量调用（多个 InfoData 接口串行）的整体超时。
_BROKER_TIMEOUT = float(os.environ.get("AMAZINGDATA_BROKER_TIMEOUT", "25") or 25)
_logged_in = False
_login_failed = False
_last_error = ""
_base = None
_market = None


def sdk_call(fn, *args, timeout: float | None = None, **kwargs):
    """在独立守护线程内执行阻塞式 SDK 调用并施加超时。

    背景：SDK 底层为原生库，网络异常时个别请求可能永不返回。之前用全局锁串行化
    SDK 访问，结果是「K线预热挂起 -> 一直占锁 -> 基本面永远拿不到锁」的连锁阻塞
    （表现为多栏目一起卡在 135s+ 且持续增长）。改为无锁 + 逐调用超时：挂起的调用
    在超时后被放弃（守护线程随进程回收），调用方立即抛 TimeoutError 走兜底。
    """
    to = _SDK_TIMEOUT if timeout is None else timeout
    box: dict = {}

    def _worker():
        try:
            box["val"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(to)
    if t.is_alive():
        raise TimeoutError(f"AmazingData 调用超时（>{to:.0f}s 无返回）")
    if "err" in box:
        raise box["err"]
    return box.get("val")



def set_credentials(username: str = "", password: str = "",
                    host: str = "", port: int = 0) -> None:
    """设置券商账号（供 UI/环境变量配置）。变更会强制下次重新登录。"""
    global _logged_in, _login_failed, _last_error, _base, _market
    _CREDS.update({
        "username": (username or os.environ.get("AMAZINGDATA_USER", "")).strip(),
        "password": (password or os.environ.get("AMAZINGDATA_PASSWORD", "")).strip(),
        "host": (host or os.environ.get("AMAZINGDATA_HOST", "")).strip(),
        "port": int(port or os.environ.get("AMAZINGDATA_PORT", 0) or 0),
    })
    _logged_in = False
    _login_failed = False
    _last_error = ""
    _base = _market = None


def _load_env_if_empty():
    if not _CREDS["username"]:
        set_credentials()  # 从环境变量补齐


def _auto_login_enabled() -> bool:
    return os.environ.get("AMAZINGDATA_AUTO_LOGIN", "1").strip().lower() not in {"0", "false", "no", "off"}


def available() -> bool:
    """SDK 已安装且账号信息齐全，且允许自动登录。"""
    if not _auto_login_enabled():
        return False
    _load_env_if_empty()
    return _ad is not None and all(
        [_CREDS["username"], _CREDS["password"], _CREDS["host"], _CREDS["port"]])


def _ensure_login() -> bool:
    global _logged_in, _login_failed, _last_error, _base, _market
    if _logged_in:
        return True
    if _login_failed:            # 上次已失败，账号变更前不重复尝试
        return False
    if not available():
        _last_error = "SDK 未安装或账号未配置"
        return False
    with _login_lock:
        if _logged_in:
            return True
        try:
            _ad.login(username=_CREDS["username"], password=_CREDS["password"],
                      host=_CREDS["host"], port=_CREDS["port"])
            _base = _ad.BaseData()
            _market = _ad.MarketData(_base.get_calendar())
            _logged_in = True
            _last_error = ""
            return True
        except Exception as e:  # noqa: BLE001
            _last_error = f"{type(e).__name__}: {e}"
            _login_failed = True
            return False


def status() -> str:
    """人类可读的券商数据源状态，用于 UI 展示。"""
    if not _auto_login_enabled():
        return "⚪ AmazingData 自动登录已关闭，基本面使用免费公开源兜底"
    if _ad is None:
        return f"❌ SDK 未加载：{_import_error or '未安装'}（仅 Linux x86_64 可用）"
    _load_env_if_empty()
    if not all([_CREDS["username"], _CREDS["password"], _CREDS["host"], _CREDS["port"]]):
        return "⚪ 未配置账号"
    if _logged_in:
        return "✅ 已登录，A股行情优先走券商"
    if _login_failed:
        return f"❌ 登录失败：{_last_error}"
    ok = _ensure_login()
    return "✅ 已登录，A股行情优先走券商" if ok else f"❌ 登录失败：{_last_error}"


def _to_broker_code(code: str) -> str:
    """6 位代码 -> 券商代码格式（如 600000.SH / 000001.SZ / 830799.BJ）。"""
    code = code.strip()
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SH"


# 券商 K线可能的列名 -> 标准列名（容错映射）
_KL_COLMAP = {
    "date": "date", "time": "date", "trade_date": "date", "kline_time": "date",
    "datetime": "date", "日期": "date",
    "open": "open", "开盘": "open", "high": "high", "最高": "high",
    "low": "low", "最低": "low", "close": "close", "收盘": "close",
    "volume": "volume", "成交量": "volume", "amount": "amount", "成交额": "amount",
    "turnover": "turnover", "换手率": "turnover",
    "pct_change": "pct_change", "涨跌幅": "pct_change",
}


@lru_cache(maxsize=256)
def stock_name(symbol: str) -> str:
    """用券商 InfoData.get_stock_basic 取证券简称，失败返回空串。"""
    if _ad is None or not _ensure_login():
        return ""
    try:
        code = _to_broker_code(symbol)

        def _query():
            info = _ad.InfoData()
            return info.get_stock_basic([code])

        r = sdk_call(_query, timeout=min(_SDK_TIMEOUT, 6.0))
        df = r[code] if isinstance(r, dict) else r
        if df is None or len(df) == 0:
            return ""
        row = df.iloc[0]
        for col in ("SECURITY_NAME", "SEC_NAME", "NAME", "名称", "证券简称", "SECURITY_ABBR"):
            if col in df.columns and str(row.get(col, "")).strip() not in ("", "nan"):
                return str(row[col]).strip()
    except Exception:  # noqa: BLE001
        return ""
    return ""


def raw_kline(symbol: str, start_date: str, end_date: str):
    """返回券商原始 K线 DataFrame（用于首次联调时查看真实列名）。"""
    if not _ensure_login():
        raise RuntimeError(f"AmazingData 不可用：{_last_error or '未安装 SDK 或账号未配置'}")
    code = _to_broker_code(symbol)
    kl = sdk_call(_market.query_kline, [code], begin_date=int(start_date),
                  end_date=int(end_date), period=_ad.constant.Period.day.value)
    return kl[code]


def fetch_daily(symbol: str, start_date: str, end_date: str,
                adjust: str = "qfq") -> pd.DataFrame | None:
    """拉取日线并标准化为项目通用列（date/open/high/low/close/volume/...）。

    注：复权在 SDK 中为独立的复权因子接口，这里先取原始行情；adjust 暂忽略。
    列名做了容错映射，首次联调后如有出入可据 raw_kline() 结果微调。
    """
    df = raw_kline(symbol, start_date, end_date)
    if df is None or len(df) == 0:
        return None
    df = df.reset_index() if df.index.name else df.copy()
    df = df.rename(columns={c: _KL_COLMAP.get(str(c).lower(), _KL_COLMAP.get(str(c), c))
                            for c in df.columns})
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"K线列不匹配，实际列：{list(df.columns)}（请用 raw_kline 联调）")
    if "date" not in df.columns:
        df["date"] = pd.RangeIndex(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
