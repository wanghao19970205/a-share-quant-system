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
import shutil
import tempfile
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
_BROKER_TIMEOUT = float(os.environ.get("AMAZINGDATA_BROKER_TIMEOUT", "90") or 90)
# 复权因子接口（get_backward_factor）从网络拉取后会用 HDF5 落地本地再读取。
# local_path 必须是容器内可写目录；默认放到缓存目录下，避免每次都重新联网拉因子。
_FACTOR_TIMEOUT = float(os.environ.get("AMAZINGDATA_FACTOR_TIMEOUT", "120") or 120)
# 券商【批量】K线 query_kline 冷启动超时：200 只首批冷启动实测 ~24s，
# 默认 15s(_SDK_TIMEOUT) 必超→同进程中毒→重试挂死，故单独放宽到 90s。
_KLINE_TIMEOUT = float(os.environ.get("AMAZINGDATA_KLINE_TIMEOUT", "90") or 90)
_FACTOR_LOCAL_PATH = os.environ.get(
    "AMAZINGDATA_FACTOR_DIR",
    os.path.join(os.environ.get("CACHE_DIR", ".cache"), "ad_factor"),
)
# 复权因子临时落地根目录：每次 get_backward_factor 用独立子目录，避免所有调用
# 共用同一 backward_factor.h5 反复覆写致损（block0_items_variety / already opened）。
_FACTOR_TMP_ROOT = os.path.dirname(_FACTOR_LOCAL_PATH.rstrip("/")) or "."
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


def _normalize_kline(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    out = df.reset_index() if df.index.name else df.copy()
    out = out.rename(columns={
        c: _KL_COLMAP.get(str(c).lower(), _KL_COLMAP.get(str(c), c))
        for c in out.columns
    })
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(out.columns)):
        raise ValueError(f"K线列不匹配，实际列：{list(out.columns)}（请用 raw_kline 联调）")
    if "date" not in out.columns:
        out["date"] = pd.RangeIndex(len(out))
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def fetch_daily_batch(symbols: list[str], start_date: str, end_date: str,
                      adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    """Fetch daily bars in bounded SDK requests and merge results by symbol.

    与 fetch_daily 口径一致：原始价按 ``adjust`` 用后复权因子换算（默认前复权）。
    因子逐票拉取并进程内缓存；某票取不到因子时该票回退不复权（不影响其它票）。
    """
    if not _ensure_login():
        raise RuntimeError(f"AmazingData 不可用：{_last_error or '未安装 SDK 或账号未配置'}")
    mapping = {str(symbol).strip()[:6]: _to_broker_code(str(symbol)) for symbol in symbols}
    result: dict[str, pd.DataFrame] = {}
    batch_size = max(int(os.environ.get("AMAZINGDATA_KLINE_BATCH_SIZE", "200") or 200), 1)
    broker_items = list(mapping.items())
    import time as _time  # 局部导入：模块未导入 time，避免动其它 import
    n_batches = (len(broker_items) + batch_size - 1) // batch_size
    loop_t0 = _time.perf_counter()
    for offset in range(0, len(broker_items), batch_size):
        batch_idx = offset // batch_size
        chunk = broker_items[offset:offset + batch_size]
        _k_t0 = _time.perf_counter()
        raw = sdk_call(
            _market.query_kline,
            [broker_code for _, broker_code in chunk],
            begin_date=int(start_date),
            end_date=int(end_date),
            period=_ad.constant.Period.day.value,
            timeout=_KLINE_TIMEOUT,
        )
        _k_el = _time.perf_counter() - _k_t0
        if not isinstance(raw, dict):
            raise TypeError(f"AmazingData 批量 K 线返回类型异常：{type(raw).__name__}")
        # 复权因子对整批一次性拉取（一个请求覆盖 chunk 内全部代码），逐票 O(1) 取列
        _f_t0 = _time.perf_counter()
        factor_frame = _get_factor_frame(tuple(bc for _, bc in chunk)) if adjust else None
        _f_el = _time.perf_counter() - _f_t0
        _f_stat = "off" if not adjust else ("ok" if factor_frame is not None else "miss")
        print(f"[kline_batch] {batch_idx + 1}/{n_batches} codes={len(chunk)} "
              f"kline={_k_el:.1f}s factor={_f_el:.1f}s({_f_stat}) "
              f"batch={_time.perf_counter() - _k_t0:.1f}s "
              f"cum={_time.perf_counter() - loop_t0:.1f}s", flush=True)
        for code, broker_code in chunk:
            frame = _normalize_kline(raw.get(broker_code))
            factor = _factor_series(factor_frame, broker_code) if adjust else None
            frame = _apply_adjust(frame, factor, adjust)
            if frame is not None and not frame.empty:
                result[code] = frame
    return result


def _apply_adjust(frame: pd.DataFrame | None, factor: "pd.Series | None",
                  adjust: str) -> pd.DataFrame | None:
    """把原始 K 线按复权方式换算（qfq 前复权 / hfq 后复权 / 空=不复权）。

    券商 query_kline 只返回原始价（手册确认无复权参数），复权走独立的
    ``get_backward_factor`` 后复权因子（``factor``：index=交易日, 值=因子）：
      - 后复权 hfq：raw * factor
      - 前复权 qfq：raw * factor / factor[最新交易日]
        （归一化到最新日 → 最新价保持真实，历史被连续缩放，符合前复权语义）
    因子只作用于价格列（open/high/low/close），成交量/额不动。
    factor 为空时回退原始价（宁可不复权也不返回错价）。
    """
    if frame is None or frame.empty or not adjust:
        return frame
    if factor is None or factor.empty:
        return frame
    out = frame.copy()
    dates = pd.to_datetime(out["date"])
    # 因子按交易日 ffill 对齐到 K 线日期（停牌日无因子则沿用前值）
    aligned = factor.reindex(factor.index.union(dates.values)).ffill().reindex(dates.values)
    aligned = aligned.to_numpy()
    if pd.isna(aligned).all():
        return frame
    if adjust == "qfq":
        # 归一化基准：因子表内最新交易日（而非本窗口末尾），保证不同起止窗口口径一致
        base = float(factor.dropna().iloc[-1])
        scale = aligned / base
    else:  # hfq
        scale = aligned
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].to_numpy() * scale
    return out


def _factor_series(frame: "pd.DataFrame | None", broker_code: str) -> "pd.Series | None":
    """从 get_backward_factor 返回的宽表里取出单只股票的因子序列。"""
    if frame is None or len(frame) == 0 or broker_code not in frame.columns:
        return None
    s = frame[broker_code].copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _get_factor_frame(broker_codes: tuple[str, ...]) -> "pd.DataFrame | None":
    """批量拉取后复权因子宽表（index=交易日, columns=券商代码）。

    一次请求覆盖整批代码——务必批量，逐票单请求会把全市场日更拖垮。
    is_local=False：联网取最新并落地 HDF5（当日新除权也能反映）。
    失败记录 _last_error 返回 None（调用方回退不复权）。
    """
    if _base is None or not broker_codes:
        return None
    global _last_error
    # 参考 AmazingData-main：get_backward_factor(is_local=False) 直接返回内存宽表，
    # HDF5 落地只是 SDK 的副作用缓存，本函数用其内存返回值、从不 pd.read_hdf 读回。
    # 旧实现所有调用共用同一 _FACTOR_LOCAL_PATH：SDK 写完 h5 不释放句柄，同进程下一
    # chunk / 下一轮再往同一 backward_factor.h5 写就撞 already opened、或写出半截 pandas
    # fixed 格式，读回报 block0_items_variety。改为每次落到独立临时目录，调用间互不
    # 干扰，用完即删；只依赖内存返回值，磁盘 h5 读回与否都不影响结果。
    os.makedirs(_FACTOR_TMP_ROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ad_factor_", dir=_FACTOR_TMP_ROOT)
    try:
        return sdk_call(_base.get_backward_factor, list(broker_codes),
                        local_path=tmp.rstrip("/") + "/", is_local=False,
                        timeout=_FACTOR_TIMEOUT)
    except Exception as e:  # noqa: BLE001 因子拉取失败不阻断行情，回退不复权
        _last_error = f"复权因子获取失败({len(broker_codes)}只): {type(e).__name__}: {e}"
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@lru_cache(maxsize=8192)
def _backward_factor(broker_code: str) -> "pd.Series | None":
    """单只股票的后复权因子序列（供 fetch_daily 单票路径用，进程内缓存）。"""
    return _factor_series(_get_factor_frame((broker_code,)), broker_code)


def fetch_daily(symbol: str, start_date: str, end_date: str,
                adjust: str = "qfq") -> pd.DataFrame | None:
    """拉取日线并标准化为项目通用列（date/open/high/low/close/volume/...）。

    券商 query_kline 返回原始价；这里按 ``adjust`` 用后复权因子换算：
    qfq=前复权（默认，与免费源口径一致）/ hfq=后复权 / ""=不复权。
    列名做了容错映射，首次联调后如有出入可据 raw_kline() 结果微调。
    """
    frame = _normalize_kline(raw_kline(symbol, start_date, end_date))
    factor = _backward_factor(_to_broker_code(symbol)) if adjust else None
    return _apply_adjust(frame, factor, adjust)
