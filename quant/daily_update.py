"""Daily incremental update for quant data and stock_analyzer snapshots.

This module is intentionally conservative for scheduled runs:
- price/valuation are updated per code from the local last date forward;
- event tables are refreshed for a recent rolling window because disclosures can lag;
- watchlist snapshots are recorded once per run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")


def _quiet_mini_racer_del() -> None:
    try:
        from py_mini_racer import py_mini_racer

        def _safe_del(self):
            try:
                ext = getattr(self, "ext", None)
                ctx = getattr(self, "ctx", None)
                if ext is not None and ctx is not None:
                    ext.mr_free_context(ctx)
            except Exception:  # noqa: BLE001
                pass

        py_mini_racer.MiniRacer.__del__ = _safe_del
    except Exception:  # noqa: BLE001
        pass


_quiet_mini_racer_del()

from quant import config, datafeed, warehouse


_LOCK = threading.Lock()


def _yyyymmdd(d: dt.date | pd.Timestamp) -> str:
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return d.strftime("%Y%m%d")


def _last_date(df: pd.DataFrame, col: str = "date") -> dt.date | None:
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return s.max().date()


def _probe_latest_price_date(codes: list[str], lookback_days: int) -> dt.date | None:
    import time as _time

    preferred = ["000001", "600000", "300750", "600519", "000333"]
    probes = [c for c in preferred if c in codes]
    if not probes:
        probes = codes[:1]
    if not probes:
        return None
    start = _yyyymmdd(max(
        dt.date.today() - dt.timedelta(days=max(lookback_days, 5)),
        dt.date(2018, 1, 1),
    ))
    end = _yyyymmdd(dt.date.today())
    started = _time.perf_counter()
    print(f"[price:probe] start codes={len(probes)} start={start}", flush=True)
    try:
        frames = datafeed.broker_daily_prices(probes, start, end, adjust="")
        latest_dates = [
            latest
            for latest in (_last_date(frames.get(code)) for code in probes)
            if latest is not None
        ]
        latest = max(latest_dates) if latest_dates else None
        print(
            f"[price:probe] done latest={latest or ''} "
            f"seconds={_time.perf_counter() - started:.1f}",
            flush=True,
        )
        return latest
    except Exception as error:  # noqa: BLE001
        print(
            f"[price:probe] failed={type(error).__name__} "
            f"seconds={_time.perf_counter() - started:.1f}; use local latest date",
            flush=True,
        )
        return None


def _probe_latest_valuation_date(codes: list[str]) -> dt.date | None:
    preferred = ["000001", "600000", "300750", "600519", "000333"]
    probes = [c for c in preferred if c in codes] + [c for c in codes[:20] if c not in preferred]
    latest_dates: list[dt.date] = []
    for code in probes:
        try:
            latest = _last_date(datafeed.valuation(code))
            if latest is not None:
                latest_dates.append(latest)
        except Exception:  # noqa: BLE001
            continue
    return max(latest_dates) if latest_dates else None


def _stale_codes(codes: list[str], load_fn, latest_date: dt.date | None, label: str,
                 force_latest: bool = False,
                 local_dates_out: dict[str, dt.date | None] | None = None) -> list[str]:
    local_dates: dict[str, dt.date | None] = {}
    local_max: dt.date | None = None
    for code in codes:
        local_latest = _last_date(load_fn(code))
        local_dates[code] = local_latest
        if local_dates_out is not None:
            local_dates_out[code] = local_latest
        if local_latest is not None and (local_max is None or local_latest > local_max):
            local_max = local_latest

    effective_latest = max([d for d in (latest_date, local_max) if d is not None], default=None)
    if effective_latest is None:
        print(f"[{label}] fresh-skip disabled: unable to determine latest date")
        return codes

    stale = []
    for code, local_latest in local_dates.items():
        # force_latest: 收盘后强制重抓最新交易日那根 K 线，覆盖盘中/午间跑批写入的临时值。
        # 仅看日期的新鲜度判断无法区分「盘中快照」与「收盘价」，故同日也需重抓。
        if local_latest is None or local_latest < effective_latest or (
            force_latest and local_latest <= effective_latest
        ):
            stale.append(code)
    print(
        f"[{label}] source_latest={latest_date or ''} local_latest={local_max or ''} "
        f"effective_latest={effective_latest} force_latest={force_latest} "
        f"stale={len(stale)} skipped_fresh={len(codes) - len(stale)}"
    )
    return stale


def _merge_time_series(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if new is None or new.empty:
        return old if old is not None else pd.DataFrame()
    merged = pd.concat([old, new], ignore_index=True) if old is not None and not old.empty else new.copy()
    merged = merged.drop_duplicates(subset=keys, keep="last")
    if "date" in merged.columns:
        merged = merged.sort_values("date")
    return merged.reset_index(drop=True)


def _spot_value(row: pd.Series, name: str, default=None):
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def _update_one_intraday_spot(code: str, row: pd.Series, trade_date: dt.date) -> tuple[str, str, int]:
    try:
        old = warehouse.load_price(code)
        close = _spot_value(row, "最新价")
        if close is None or close <= 0:
            return code, "InvalidSpot", 0
        previous = _spot_value(row, "昨收", close)
        item = {
            "code": code,
            "date": pd.Timestamp(trade_date),
            "open": _spot_value(row, "今开", close),
            "high": _spot_value(row, "最高", close),
            "low": _spot_value(row, "最低", close),
            "close": close,
            "volume": _spot_value(row, "成交量", 0.0),
            "amount": _spot_value(row, "成交额", 0.0),
            "turnover": _spot_value(row, "换手率", 0.0),
            "pct_change": _spot_value(row, "涨跌幅", (close / previous - 1) * 100 if previous else 0.0),
        }
        merged = _merge_time_series(old, pd.DataFrame([item]), keys=["code", "date"])
        warehouse.save_price(code, merged)
        return code, "ok", 1
    except Exception as e:  # noqa: BLE001
        return code, type(e).__name__, 0


def _merge_broker_daily(
    code: str,
    frame: pd.DataFrame | None,
) -> tuple[str, str, int]:
    if frame is None or frame.empty:
        return code, "EmptyBrokerData", 0
    try:
        old = warehouse.load_price(code)
        new = frame.copy()
        if "code" in new.columns:
            new = new.drop(columns=["code"])
        new.insert(0, "code", code)
        merged = _merge_time_series(old, new, keys=["code", "date"])
        warehouse.save_price(code, merged)
        return code, "ok", len(new)
    except Exception as e:  # noqa: BLE001
        return code, type(e).__name__, 0


def _update_intraday_from_broker(
    codes: list[str],
    workers: int,
    trade_date: dt.date,
) -> dict | None:
    if not datafeed.broker_available():
        print("[intraday] AmazingData unavailable=config-or-sdk", flush=True)
        return None
    start = _yyyymmdd(trade_date - dt.timedelta(days=5))
    end = _yyyymmdd(trade_date)
    try:
        frames = datafeed.broker_daily_prices(codes, start, end)
    except Exception as e:  # noqa: BLE001
        print(
            f"[intraday] AmazingData batch failed={type(e).__name__}; "
            "falling back to whole-market free snapshot",
            flush=True,
        )
        return None
    print(
        f"[intraday] source=AmazingData batch_frames={len(frames)}/{len(codes)}",
        flush=True,
    )
    result = _run_batch(
        codes,
        lambda code: _merge_broker_daily(code, frames.get(code)),
        workers,
        "intraday-amazingdata",
    )
    coverage = result["ok"] / max(len(codes), 1)
    print(
        f"[intraday] AmazingData coverage={coverage:.1%} "
        f"ok={result['ok']} fail={result['fail']}",
        flush=True,
    )
    if coverage < 0.80:
        print(
            "[intraday] AmazingData coverage below 80%; "
            "falling back to whole-market free snapshot",
            flush=True,
        )
        return None
    result["source"] = "AmazingData"
    result["trade_date"] = str(trade_date)
    return result


def update_intraday_spot(codes: list[str], workers: int = 12) -> dict:
    """Prefer AmazingData, then use the free whole-market intraday snapshot."""
    trade_date = dt.date.today()
    broker = _update_intraday_from_broker(codes, workers, trade_date)
    if broker is not None:
        return broker

    print("[intraday] source=free-whole-market-spot", flush=True)
    spot = datafeed.market_spot()
    if spot is None or spot.empty or "代码" not in spot.columns:
        raise RuntimeError("whole-market intraday snapshot is empty")
    spot = spot.copy()
    spot["代码"] = spot["代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    spot = spot.dropna(subset=["代码"]).drop_duplicates("代码", keep="last").set_index("代码")
    selected = [code for code in codes if code in spot.index]
    coverage = len(selected) / max(len(codes), 1)
    print(f"[intraday-spot] source_rows={len(spot)} matched={len(selected)}/{len(codes)} coverage={coverage:.1%}")
    if coverage < 0.80:
        raise RuntimeError(f"intraday snapshot coverage too low: {coverage:.1%}")
    trade_date = dt.date.today()
    return _run_batch(
        selected,
        lambda code: _update_one_intraday_spot(code, spot.loc[code], trade_date),
        workers,
        "intraday-spot",
    )


def _update_one_price(code: str, lookback_days: int = 5) -> tuple[str, str, int]:
    old = warehouse.load_price(code)
    last = _last_date(old)
    start = dt.date(2018, 1, 1) if last is None else max(last - dt.timedelta(days=lookback_days), dt.date(2018, 1, 1))
    try:
        new = datafeed.daily_price(code, _yyyymmdd(start))
        merged = _merge_time_series(old, new, keys=["code", "date"])
        if not merged.empty:
            warehouse.save_price(code, merged)
        return code, "ok", len(new) if new is not None else 0
    except Exception as e:  # noqa: BLE001
        return code, type(e).__name__, 0


def _update_one_valuation(code: str, lookback_days: int = 5) -> tuple[str, str, int]:
    old = warehouse.load_valuation(code)
    last = _last_date(old)
    try:
        new = datafeed.valuation(code)
        if last is not None and new is not None and not new.empty and "date" in new.columns:
            cutoff = pd.Timestamp(last - dt.timedelta(days=lookback_days))
            new = new[pd.to_datetime(new["date"], errors="coerce") >= cutoff].copy()
        merged = _merge_time_series(old, new, keys=["code", "date"])
        if not merged.empty:
            warehouse.save_valuation(code, merged)
        return code, "ok", len(new) if new is not None else 0
    except Exception as e:  # noqa: BLE001
        return code, type(e).__name__, 0


def _warmup_broker(retries: int = 3) -> bool:
    """批量前预热券商 TGW 连接：先用单只轻量 K 线把 push server 拉起来再发大批量。

    诊断实测(7-23)：进程冷启动即发 200 只大批量会撞在连接未就绪上首发 TimeoutError，
    重登录也救不回；而单只渐进调用能让连接就绪(单只 1.3s 成功后批量 30 只稳定 <3.4s，
    逐股兜底能成也是同一个渐进预热机制)。预热失败(券商真不可用)返回 False，主路径照常
    进批量循环并在失败时降级逐股免费源兜底(与不预热行为一致，零丢失)。"""
    import time as _time
    from stock_analyzer import amazingdata_source
    end = _yyyymmdd(dt.date.today())
    start = _yyyymmdd(dt.date.today() - dt.timedelta(days=10))
    for attempt in range(retries):
        try:
            amazingdata_source.raw_kline("000001", start, end)
            return True
        except Exception:  # noqa: BLE001 未就绪 -> 同一会话退避再探，耗尽则让主路径降级
            # sdk_call 超时后底层守护线程可能仍持有服务端连接。强制重登录会叠加
            # 僵持连接并触发账户连接数上限，因此重试必须复用当前会话。
            _time.sleep(2.0 * (attempt + 1))
    return False


def _update_prices_batched(codes, lookback_days: int = 5, workers: int = 12,
                           batch_size: int | None = None, gap_threshold_days: int = 45,
                           batch_retries: int = 2,
                           local_dates: dict[str, dt.date | None] | None = None) -> dict:
    """价格更新：券商批量 K 线做主路径 + 逐股缺口/兜底回退。

    保留逐股路径三个保护：①免费源兜底(批量缺票转单只)②列标准化(_standardize
    补齐 amount/turnover/pct_change)③缺口恢复(缺口超阈值或无本地数据的票走单只全量)。
    窗口数据驱动：批量起点 = 今天 -(本批最大缺口 + lookback_days 重叠)，平时缺口小
    窗口自动收窄、长假后自动张开，不猜节假日。批量前先 _warmup_broker 预热券商连接
    (治冷启动竞态)，块超时/失败在当前会话退避重试 batch_retries 次后降级逐股。
    实测批量 65ms/只 vs 逐只 3.15s/只(49x)，全市场拉数 ~40min 压到 ~4min。"""
    import time as _time
    from stock_analyzer.data import _standardize
    if batch_size is None:
        batch_size = int(os.environ.get("AMAZINGDATA_KLINE_BATCH_SIZE", "200") or 200)
    batch_size = max(int(batch_size), 1)
    today = dt.date.today()
    # 按本地最后日期分流：缺口在阈值内走批量(数据驱动窗口)；无本地数据或缺口过大走逐股全量补
    batch_codes, gap_codes, max_gap = [], [], lookback_days
    for code in codes:
        last = (
            local_dates[code]
            if local_dates is not None and code in local_dates
            else _last_date(warehouse.load_price(code))
        )
        if last is None:
            gap_codes.append(code)
            continue
        gap = (today - last).days
        if gap > gap_threshold_days:
            gap_codes.append(code)
        else:
            batch_codes.append(code)
            if gap > max_gap:
                max_gap = gap
    # 窗口只需覆盖 max_gap + lookback_days 重叠(纠正盘中/复权)，平时自动收到 ~7 天
    window = max_gap + lookback_days
    start = _yyyymmdd(max(today - dt.timedelta(days=window), dt.date(2018, 1, 1)))
    end = _yyyymmdd(today)
    print(f"[price] batch codes={len(batch_codes)} gap_fallback={len(gap_codes)} "
          f"window_days={window} start={start}", flush=True)
    if batch_codes and not _warmup_broker():
        print("[price] broker warmup failed; 批量可能整片降级逐股", flush=True)
    ok = rows = 0
    fallback = list(gap_codes)
    total_batches = (len(batch_codes) + batch_size - 1) // batch_size
    for i in range(0, len(batch_codes), batch_size):
        chunk = batch_codes[i:i + batch_size]
        batch_idx = i // batch_size
        frames = None
        for attempt in range(batch_retries + 1):
            try:
                frames = datafeed.broker_daily_prices(
                    chunk, start, end,
                    progress_offset=batch_idx,
                    progress_total=total_batches,
                )
                break
            except Exception as e:  # noqa: BLE001 批量块失败 -> 当前会话退避重试，仍失败才逐股
                if attempt < batch_retries:
                    print(f"[price] batch chunk={i // batch_size} attempt={attempt + 1} "
                          f"{type(e).__name__}; 当前会话退避重试", flush=True)
                    _time.sleep(2.0 * (attempt + 1))
                else:
                    print(f"[price] batch chunk={i // batch_size} failed={type(e).__name__} "
                          f"after {batch_retries + 1} tries; per-stock fallback", flush=True)
                    fallback.extend(chunk)
        if frames is None:
            continue
        for code in chunk:
            fr = frames.get(code)
            if fr is None or getattr(fr, "empty", True):
                fallback.append(code)
                continue
            try:
                new = _standardize(fr.copy())
                if "code" in new.columns:
                    new = new.drop(columns=["code"])
                new.insert(0, "code", code)
                merged = _merge_time_series(warehouse.load_price(code), new, keys=["code", "date"])
                if not merged.empty:
                    warehouse.save_price(code, merged)
                ok += 1
                rows += len(new)
            except Exception:  # noqa: BLE001 标准化/落盘异常 -> 转单只兜底重试
                fallback.append(code)
    failures: list[str] = []
    if fallback:
        print(f"[price] batch ok={ok}; per-stock fallback={len(fallback)}", flush=True)
        fb = _run_batch(fallback, lambda c: _update_one_price(c, lookback_days), workers, "price-fallback")
        ok += fb["ok"]
        rows += fb["rows"]
        failures = fb["failures"]
    fail = len(codes) - ok
    print(f"[price] ok={ok} fail={fail} fetched_rows={rows}" + (f" failures={failures}" if failures else ""))
    return {"ok": ok, "fail": fail, "rows": rows, "failures": failures}


def _run_batch(codes: list[str], fn, workers: int, label: str) -> dict:
    ok = fail = rows = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, status, n in ex.map(fn, codes):
            if status == "ok":
                ok += 1
                rows += int(n or 0)
            else:
                fail += 1
                if len(failures) < 20:
                    failures.append(f"{code}:{status}")
    print(f"[{label}] ok={ok} fail={fail} fetched_rows={rows}" + (f" failures={failures}" if failures else ""))
    return {"ok": ok, "fail": fail, "rows": rows, "failures": failures}


def _recent_report_dates(window_days: int) -> list[str]:
    today = dt.date.today()
    start_year = today.year - 1 if window_days > 120 else today.year
    dates = []
    for y in range(start_year, today.year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            d = dt.datetime.strptime(f"{y}{md}", "%Y%m%d").date()
            if d <= today:
                dates.append(f"{y}{md}")
    return dates


def update_recent_events(start: str, end: str, workers: int = 4, fundamentals: bool = True) -> None:
    print(f"[events] window={start}-{end}")
    for name, fn in (("block_trades", datafeed.block_trades), ("lhb", datafeed.lhb)):
        try:
            df = fn(start, end)
            keys = [k for k in ("code", "date") if k in df.columns]
            warehouse.upsert(name, df, keys=keys or list(df.columns[:2]))
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] failed: {type(e).__name__}")
        print(f"[{name}] rows={len(warehouse.load(name))}")

    try:
        warehouse.upsert("margin_sse", datafeed.margin_sse(start, end), keys=["date"])
    except Exception as e:  # noqa: BLE001
        print(f"[margin_sse] failed: {type(e).__name__}")
    print(f"[margin_sse] rows={len(warehouse.load('margin_sse'))}")

    for name, fn in (("margin_szse", datafeed.margin_szse), ("margin_underlying_szse", datafeed.margin_underlying_szse)):
        try:
            df = fn(end)
            keys = ["code", "date"] if "code" in df.columns else ["date"]
            warehouse.upsert(name, df, keys=keys)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] failed: {type(e).__name__}")
        print(f"[{name}] rows={len(warehouse.load(name))}")

    if not fundamentals:
        return

    qdates = _recent_report_dates(180)
    for qd in qdates:
        for name, fn in (("financial_yjbb", datafeed.financial_yjbb),
                         ("performance_forecast", datafeed.performance_forecast),
                         ("holder_num", datafeed.holder_num),
                         ("dividend", datafeed.dividend)):
            try:
                df = fn(qd)
                keys = ["code", "report_date"]
                if name == "performance_forecast" and "预测指标" in df.columns:
                    keys.append("预测指标")
                warehouse.upsert(name, df, keys=keys)
            except Exception as e:  # noqa: BLE001
                print(f"[{name}:{qd}] failed: {type(e).__name__}")
    print("[fundamentals] recent report tables updated")


def _read_watchlist(snapshot_dir: str) -> list[str]:
    path = os.path.join(snapshot_dir, "watchlist.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return sorted(set(re.findall(r"\d{6}", f.read())))


def run_snapshots(snapshot_dir: str, codes: list[str] | None = None) -> int:
    codes = codes or _read_watchlist(snapshot_dir)
    if not codes:
        print("[snapshots] no watchlist codes")
        return 0
    env = os.environ.copy()
    env["SNAPSHOT_DIR"] = snapshot_dir
    cmd = [sys.executable, "-m", "stock_analyzer.snapshot_batch", *codes]
    print(f"[snapshots] running {len(codes)} codes")
    res = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env, check=False)
    return int(res.returncode)


def refresh_trading_calendar() -> dict:
    """Refresh the authoritative AmazingData open-session calendar when available."""
    if not datafeed.broker_available():
        return {
            "status": "broker-unavailable",
            "path": config.TRADING_CALENDAR_FILE,
            "exists": os.path.isfile(config.TRADING_CALENDAR_FILE),
        }
    calendar = datafeed.broker_trading_calendar().copy()
    if list(calendar.columns) != ["date"]:
        raise ValueError("broker trading calendar must contain only the date column")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").astype("datetime64[ns]")
    if calendar.empty or calendar["date"].isna().any():
        raise ValueError("broker trading calendar is empty or invalid")
    if calendar["date"].duplicated().any() or not calendar["date"].is_monotonic_increasing:
        raise ValueError("broker trading calendar must be unique and increasing")
    warehouse.save("trading_calendar", calendar)
    return {
        "status": "refreshed",
        "path": config.TRADING_CALENDAR_FILE,
        "rows": int(len(calendar)),
        "first_date": str(calendar["date"].iloc[0].date()),
        "last_date": str(calendar["date"].iloc[-1].date()),
    }


def refresh_pit_reference_data(index_codes: list[str] | None = None) -> dict:
    """Refresh PIT security-master and index-membership inputs without activating them."""
    if not datafeed.broker_available():
        return {
            "status": "broker-unavailable",
            "security_master_path": config.SECURITY_MASTER_FILE,
            "index_history_path": config.INDEX_CONSTITUENT_HISTORY_FILE,
        }
    requested = index_codes or ["000300.SH", "000905.SH", "000852.SH"]
    security_master = datafeed.broker_security_master()
    index_history = datafeed.broker_index_constituent_history(requested)
    warehouse.save("security_master", security_master)
    warehouse.save("index_constituent_history", index_history)
    return {
        "status": "refreshed",
        "security_master_path": config.SECURITY_MASTER_FILE,
        "security_master_rows": int(len(security_master)),
        "security_master_first_list_date": str(security_master["list_date"].min().date()),
        "index_history_path": config.INDEX_CONSTITUENT_HISTORY_FILE,
        "index_history_rows": int(len(index_history)),
        "index_codes": sorted(index_history["index_code"].astype(str).unique().tolist()),
        "index_history_first_in_date": str(index_history["in_date"].min().date()),
    }


def refresh_trading_status_reference(
    codes: list[str],
    batch_size: int = 0,
) -> dict:
    """Refresh selected PIT status histories without activating trading filters."""
    normalized = sorted({
        datafeed._norm(code) for code in codes if datafeed._norm(code).isdigit()
    })
    if not datafeed.broker_available():
        return {
            "status": "broker-unavailable",
            "path": config.TRADING_STATUS_HISTORY_FILE,
            "requested_codes": len(normalized),
        }
    effective_batch_size = int(batch_size)
    if effective_batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if effective_batch_size:
        batches = [
            normalized[offset:offset + effective_batch_size]
            for offset in range(0, len(normalized), effective_batch_size)
        ]
        frames = [datafeed.broker_history_stock_status(batch) for batch in batches]
        history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        batches = [normalized]
        history = datafeed.broker_history_stock_status(normalized)
    warehouse.save("trading_status_history", history)
    return {
        "status": "refreshed",
        "path": config.TRADING_STATUS_HISTORY_FILE,
        "rows": int(len(history)),
        "codes": int(history["code"].nunique()),
        "batches": int(len(batches)),
        "batch_size": effective_batch_size,
        "st_rows": int(history["is_st"].fillna(False).astype(bool).sum()),
        "suspended_rows": int(history["is_suspended"].fillna(False).astype(bool).sum()),
        "withdrawal_rows": int(history["is_withdrawal"].fillna(False).astype(bool).sum()),
    }

    history = datafeed.broker_history_stock_status(normalized)
    warehouse.save("trading_status_history", history)
    return {
        "status": "refreshed",
        "path": config.TRADING_STATUS_HISTORY_FILE,
        "requested_codes": len(normalized),
        "rows": int(len(history)),
        "first_date": str(history["date"].min().date()),
        "last_date": str(history["date"].max().date()),
        "st_rows": int(history["is_st"].sum()),
        "suspended_rows": int(history["is_suspended"].sum()),
        "withdrawal_rows": int(history["is_withdrawal"].sum()),
    }


def _refresh_mainboard_universe_isolated() -> None:
    """在子进程里刷新主板股票池，隔离 akshare 抓表对 TGW 连接的进程级污染。

    诊断坐实(券商连接诊断.txt 第3-5轮)：price 前 refresh_mainboard_universe() 调
    akshare ak.stock_info_a_code_name() 抓全A代码表，会毒化本进程网络状态，之后
    券商 TGW push server 再也 init 不起来，price 批量/单只全 15s 超时。socket 复位、
    TGW 断开重连都救不回——唯一可靠修法是把抓表关进子进程，副作用随子进程回收，
    主进程 TGW 永不被碰(第5轮 isolate_verify 已实测：子进程抓表后主进程 30/30 全成)。
    子进程内 refresh_mainboard_universe() 把股票池写进 config.MAINBOARD_UNIVERSE_FILE，
    主进程随后照常 datafeed.universe 读该文件，无需回传数据。抓表每轮仅一次(~10s)，
    子进程额外启动开销 ~2s，对整轮日更(目标<10min)可忽略。子进程失败则沿用现有
    universe 文件(股票池变化极慢，晚一轮无害)，不阻断日更。"""
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "from quant import datafeed; datafeed.refresh_mainboard_universe()"],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as e:  # noqa: BLE001 子进程异常(超时等) -> 沿用现有 universe 文件
        print(f"[universe] 子进程刷新异常 {type(e).__name__}; 沿用现有 universe 文件", flush=True)
        return
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        print(f"[universe] 子进程刷新失败 rc={r.returncode}; 沿用现有 universe 文件. {tail}", flush=True)


def run(universe: str = "mainboard_active", workers: int = 12, lookback_days: int = 5,
        event_window_days: int = 30, snapshot_dir: str | None = None,
        skip_price: bool = False, skip_valuation: bool = False,
        skip_events: bool = False, skip_fundamentals: bool = False,
        skip_snapshots: bool = False, limit: int = 0, force_latest: bool = False,
        intraday_spot: bool = False, codes_file: str | None = None,
        refresh_pit_reference: bool = False) -> dict:
    config.ensure_dirs()
    if refresh_pit_reference:
        pit_summary = refresh_pit_reference_data()
        print(f"[pit] {pit_summary}", flush=True)
        if pit_summary.get("status") != "refreshed":
            raise RuntimeError(f"PIT reference refresh unavailable: {pit_summary}")
    calendar_summary = refresh_trading_calendar()
    u = config.UNIVERSES[universe]
    if codes_file:
        with open(codes_file, encoding="utf-8") as fh:
            codes = sorted({line.strip() for line in fh if re.fullmatch(r"\d{6}", line.strip())})
        if not codes:
            raise ValueError(f"codes file contains no valid six-digit codes: {codes_file}")
    else:
        if u["kind"] == "mainboard_active":
            _refresh_mainboard_universe_isolated()
        codes = datafeed.universe(u["kind"], u["arg"])
    if limit:
        codes = codes[:limit]
    print(f"[universe] {universe} codes={len(codes)} quant_dir={config.QUANT_DIR}")

    summary: dict = {
        "universe": universe,
        "n_codes": len(codes),
        "trading_calendar": calendar_summary,
    }
    if not skip_price:
        if intraday_spot:
            summary["price"] = update_intraday_spot(codes, workers=workers)
            summary["price"]["mode"] = (
                "broker-daily-bars"
                if summary["price"].get("source") == "AmazingData"
                else "whole-market-spot"
            )
        else:
            if force_latest:
                price_latest = None
                print(
                    "[price:probe] skipped because force_latest refreshes every code",
                    flush=True,
                )
            else:
                price_latest = _probe_latest_price_date(codes, lookback_days)
            price_local_dates: dict[str, dt.date | None] = {}
            price_codes = _stale_codes(
                codes,
                warehouse.load_price,
                price_latest,
                "price",
                force_latest=force_latest,
                local_dates_out=price_local_dates,
            )
            summary["price"] = _update_prices_batched(
                price_codes,
                lookback_days=lookback_days,
                workers=workers,
                local_dates=price_local_dates,
            )
            summary["price"]["skipped_fresh"] = len(codes) - len(price_codes)
            summary["price"]["source_latest"] = str(price_latest or "")
    if not skip_valuation:
        valuation_latest = _probe_latest_valuation_date(codes)
        valuation_codes = _stale_codes(codes, warehouse.load_valuation, valuation_latest, "valuation")
        summary["valuation"] = _run_batch(valuation_codes, lambda c: _update_one_valuation(c, lookback_days), workers, "valuation")
        summary["valuation"]["skipped_fresh"] = len(codes) - len(valuation_codes)
        summary["valuation"]["source_latest"] = str(valuation_latest or "")
    if not skip_events:
        end = dt.date.today()
        start = end - dt.timedelta(days=event_window_days)
        update_recent_events(_yyyymmdd(start), _yyyymmdd(end), workers=workers,
                             fundamentals=not skip_fundamentals)
    if not skip_snapshots:
        snapshot_dir = snapshot_dir or os.environ.get(
            "SNAPSHOT_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots"),
        )
        summary["snapshot_returncode"] = run_snapshots(snapshot_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="每日增量更新全A量化数据和白名单快照")
    ap.add_argument("--universe", default="mainboard_active", choices=list(config.UNIVERSES))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lookback-days", type=int, default=5, help="价格/估值回看天数，用于覆盖迟到或修正数据")
    ap.add_argument("--event-window-days", type=int, default=30, help="事件表滚动刷新窗口")
    ap.add_argument("--snapshot-dir", default=os.environ.get("SNAPSHOT_DIR", ""))
    ap.add_argument("--skip-price", action="store_true")
    ap.add_argument("--skip-valuation", action="store_true")
    ap.add_argument("--skip-events", action="store_true")
    ap.add_argument("--skip-fundamentals", action="store_true", help="事件更新时跳过财报/业绩预告/股东户数/分红等低频宽表")
    ap.add_argument("--skip-snapshots", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="调试用，仅更新前 N 只股票")
    ap.add_argument("--force-latest", action="store_true",
                    help="收盘后强制重抓最新交易日那根K线，覆盖盘中/午间跑批写入的临时值")
    ap.add_argument("--intraday-spot", action="store_true",
                    help="盘中使用一次全市场实时快照覆盖当日临时K线，避免逐股网络请求")
    ap.add_argument("--codes-file", default="",
                    help="仅更新文件中列出的6位股票代码，用于崩溃后的缺口恢复")
    ap.add_argument("--refresh-pit-reference", action="store_true",
                    help="训练前刷新证券主数据和指数成分历史；不可用时 fail-closed")
    args = ap.parse_args()
    res = run(universe=args.universe, workers=args.workers, lookback_days=args.lookback_days,
              event_window_days=args.event_window_days, snapshot_dir=args.snapshot_dir or None,
              skip_price=args.skip_price, skip_valuation=args.skip_valuation,
              skip_events=args.skip_events, skip_fundamentals=args.skip_fundamentals,
              skip_snapshots=args.skip_snapshots, limit=args.limit, force_latest=args.force_latest,
              intraday_spot=args.intraday_spot, codes_file=args.codes_file or None,
              refresh_pit_reference=args.refresh_pit_reference)
    print("[done]", res)
    # 券商 tgw 原生库在解释器退出（atexit/析构）时可能段错误(SIGSEGV)，此时数据已全部落盘。
    # 用 os._exit(0) 跳过 native 析构直接干净退出，避免非零退出码被上游 check=True 误判为失败、
    # 从而中断后续训练。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
