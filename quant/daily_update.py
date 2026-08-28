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
import time
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
from quant.logging_utils import install_timestamped_stdout


install_timestamped_stdout()

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
    preferred = ["000001", "600000", "300750", "600519", "000333"]
    probes = [c for c in preferred if c in codes] + [c for c in codes[:20] if c not in preferred]
    start = _yyyymmdd(max(dt.date.today() - dt.timedelta(days=max(lookback_days, 5)), dt.date(2018, 1, 1)))
    latest_dates: list[dt.date] = []
    for code in probes:
        try:
            latest = _last_date(datafeed.daily_price(code, start))
            if latest is not None:
                latest_dates.append(latest)
        except Exception:  # noqa: BLE001
            continue
    return max(latest_dates) if latest_dates else None


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


def _stale_codes(
    codes: list[str],
    load_fn,
    latest_date: dt.date | None,
    label: str,
    force_latest: bool = False,
    local_dates_out: dict[str, dt.date | None] | None = None,
    last_date_fn=None,
) -> list[str]:
    local_dates: dict[str, dt.date | None] = {}
    local_max: dt.date | None = None
    scan_started = time.perf_counter()
    print(
        f"[{label}:scan-start] codes={len(codes)} force_latest={force_latest}",
        flush=True,
    )
    for index, code in enumerate(codes, start=1):
        if index == 1 or index % 100 == 0:
            print(
                f"[{label}:scan-current] index={index}/{len(codes)} code={code} "
                f"elapsed={time.perf_counter() - scan_started:.2f}s",
                flush=True,
            )
        local_latest = (
            last_date_fn(code)
            if last_date_fn is not None
            else _last_date(load_fn(code))
        )
        local_dates[code] = local_latest
        if local_dates_out is not None:
            local_dates_out[code] = local_latest
        if local_latest is not None and (local_max is None or local_latest > local_max):
            local_max = local_latest

    print(
        f"[{label}:scan-done] codes={len(codes)} "
        f"seconds={time.perf_counter() - scan_started:.2f}",
        flush=True,
    )
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


def _price_merge_changed(old: pd.DataFrame, merged: pd.DataFrame) -> bool:
    """Return whether a price merge changes persisted rows.

    force_latest still fetches and compares the current trading day, but an
    unchanged response does not need to rewrite the full historical parquet.
    """
    if old is None or old.empty:
        return True
    if merged is None or len(old) != len(merged) or list(old.columns) != list(merged.columns):
        return True
    try:
        left = old.sort_values(["code", "date"]).reset_index(drop=True)
        right = merged.sort_values(["code", "date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        return False
    except (AssertionError, KeyError, TypeError, ValueError):
        return True


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
        if _price_merge_changed(old, merged):
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
        if _price_merge_changed(old, merged):
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


def _sync_trading_calendar() -> dict:
    """从 AmazingData 获取真实交易日历并原子落盘；失败时保留旧文件。"""
    from stock_analyzer import amazingdata_source

    path = config.TRADING_CALENDAR_FILE
    try:
        calendar = amazingdata_source.trading_calendar()
        latest = pd.Timestamp(calendar["date"].max()).date()
        today = dt.date.today()
        # 工作日不能接受落后于今天的日历；节假日/周末允许沿用最近一个交易日。
        if today.weekday() < 5 and latest < today:
            raise RuntimeError(f"SDK 日历落后：latest={latest} today={today}")
        warehouse.save_trading_calendar(calendar)
        print(f"[calendar] updated path={path} dates={len(calendar)} latest={latest}", flush=True)
        return {"status": "updated", "path": path, "latest": str(latest),
                "dates": len(calendar)}
    except Exception as e:  # noqa: BLE001 - 日历同步失败不阻断价格/训练主链路
        print(f"[calendar] sync_failed path={path} error={type(e).__name__}: {e}; keep_existing", flush=True)
        return {"status": "kept_existing", "path": path, "error": type(e).__name__}


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
        except Exception:  # noqa: BLE001 未就绪 -> 重登录+退避再探，耗尽则让主路径降级
            amazingdata_source._logged_in = False
            _time.sleep(2.0 * (attempt + 1))
    return False


def _update_prices_batched(
    codes,
    lookback_days: int = 5,
    workers: int = 12,
    batch_size: int | None = None,
    gap_threshold_days: int = 45,
    batch_retries: int = 2,
    local_dates: dict[str, dt.date | None] | None = None,
) -> dict:
    """价格更新：券商批量 K 线做主路径 + 逐股缺口/兜底回退。

    未显式传入批次时读取 AMAZINGDATA_KLINE_BATCH_SIZE，保证 scheduler 的
    部署配置能够贯穿到外层 daily_update，而不是只影响 SDK 内层分块。

    保留逐股路径三个保护：①免费源兜底(批量缺票转单只)②列标准化(_standardize
    补齐 amount/turnover/pct_change)③缺口恢复(缺口超阈值或无本地数据的票走单只全量)。
    窗口数据驱动：批量起点 = 今天 -(本批最大缺口 + lookback_days 重叠)，平时缺口小
    窗口自动收窄、长假后自动张开，不猜节假日。批量前先 _warmup_broker 预热券商连接
    (治冷启动竞态)，块超时/失败再重试 batch_retries 次(重登录 + 退避)后降级逐股。
    实测批量 65ms/只 vs 逐只 3.15s/只(49x)，全市场拉数 ~40min 压到 ~4min。"""
    import time as _time
    from stock_analyzer import amazingdata_source
    from stock_analyzer.data import _standardize
    if batch_size is None:
        try:
            batch_size = int(os.environ.get("AMAZINGDATA_KLINE_BATCH_SIZE", "200") or 200)
        except (TypeError, ValueError):
            batch_size = 200
    batch_size = max(1, int(batch_size))
    today = dt.date.today()
    # 按本地最后日期分流：缺口在阈值内走批量(数据驱动窗口)；无本地数据或缺口过大走逐股全量补
    batch_codes, gap_codes, max_gap = [], [], lookback_days
    for code in codes:
        last = (
            local_dates.get(code)
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
    local_load_seconds = 0.0
    local_merge_sort_seconds = 0.0
    local_save_seconds = 0.0
    local_compare_seconds = 0.0
    fallback = list(gap_codes)
    total_batches = (len(batch_codes) + batch_size - 1) // batch_size
    for i in range(0, len(batch_codes), batch_size):
        chunk = batch_codes[i:i + batch_size]
        batch_no = i // batch_size + 1
        batch_started = _time.perf_counter()
        load_before = local_load_seconds
        merge_before = local_merge_sort_seconds
        save_before = local_save_seconds
        print(
            f"[price:batch-start] batch={batch_no}/{total_batches} "
            f"codes={len(chunk)}",
            flush=True,
        )
        frames = None
        for attempt in range(batch_retries + 1):
            try:
                frames = datafeed.broker_daily_prices(chunk, start, end)
                break
            except Exception as e:  # noqa: BLE001 批量块失败 -> 退避+重登录重试，仍失败才逐股
                if attempt < batch_retries:
                    print(f"[price] batch chunk={i // batch_size} attempt={attempt + 1} "
                          f"{type(e).__name__}; 重登录后重试", flush=True)
                    amazingdata_source._logged_in = False  # 强制下次调用重新登录，冲掉卡死连接
                    _time.sleep(2.0 * (attempt + 1))
                else:
                    print(f"[price] batch chunk={i // batch_size} failed={type(e).__name__} "
                          f"after {batch_retries + 1} tries; per-stock fallback", flush=True)
                    fallback.extend(chunk)
        print(
            f"[price:batch-sdk-done] batch={batch_no}/{total_batches} "
            f"seconds={_time.perf_counter() - batch_started:.2f} "
            f"returned={len(frames) if frames is not None else 0}",
            flush=True,
        )
        if frames is None:
            print(
                f"[price:batch-done] batch={batch_no}/{total_batches} status=fallback "
                f"seconds={_time.perf_counter() - batch_started:.2f}",
                flush=True,
            )
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
                load_started = _time.perf_counter()
                old = warehouse.load_price(code)
                local_load_seconds += _time.perf_counter() - load_started
                merge_started = _time.perf_counter()
                merged = _merge_time_series(old, new, keys=["code", "date"])
                local_merge_sort_seconds += _time.perf_counter() - merge_started
                compare_started = _time.perf_counter()
                changed = _price_merge_changed(old, merged)
                local_compare_seconds += _time.perf_counter() - compare_started
                if not merged.empty and changed:
                    save_started = _time.perf_counter()
                    warehouse.save_price(code, merged)
                    local_save_seconds += _time.perf_counter() - save_started
                ok += 1
                rows += len(new)
            except Exception:  # noqa: BLE001 标准化/落盘异常 -> 转单只兜底重试
                fallback.append(code)
        print(
            f"[price:batch-done] batch={batch_no}/{total_batches} status=ok "
            f"seconds={_time.perf_counter() - batch_started:.2f} "
            f"local_load={local_load_seconds - load_before:.2f}s "
            f"local_merge_sort={local_merge_sort_seconds - merge_before:.2f}s "
            f"local_save={local_save_seconds - save_before:.2f}s",
            flush=True,
        )
    failures: list[str] = []
    if fallback:
        print(f"[price] batch ok={ok}; per-stock fallback={len(fallback)}", flush=True)
        fb = _run_batch(fallback, lambda c: _update_one_price(c, lookback_days), workers, "price-fallback")
        ok += fb["ok"]
        rows += fb["rows"]
        failures = fb["failures"]
    fail = len(codes) - ok
    print(
        f"[price:timing] batch_local_load={local_load_seconds:.2f}s "
        f"batch_local_merge_sort={local_merge_sort_seconds:.2f}s "
        f"batch_local_compare={local_compare_seconds:.2f}s "
        f"batch_local_save={local_save_seconds:.2f}s "
        f"batch_codes={len(batch_codes)}",
        flush=True,
    )
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
    events_started = time.perf_counter()
    print(f"[events] window={start}-{end}", flush=True)
    for name, fn in (("block_trades", datafeed.block_trades), ("lhb", datafeed.lhb)):
        item_started = time.perf_counter()
        try:
            fetch_started = time.perf_counter()
            df = fn(start, end)
            fetch_seconds = time.perf_counter() - fetch_started
            keys = [k for k in ("code", "date") if k in df.columns]
            upsert_started = time.perf_counter()
            merged = warehouse.upsert(name, df, keys=keys or list(df.columns[:2]))
            upsert_seconds = time.perf_counter() - upsert_started
            rows = len(merged)
        except Exception as e:  # noqa: BLE001
            fetch_seconds = upsert_seconds = 0.0
            rows = 0
            print(f"[{name}] failed: {type(e).__name__}", flush=True)
        print(
            f"[events:timing] name={name} fetch={fetch_seconds:.2f}s "
            f"upsert_write={upsert_seconds:.2f}s total={time.perf_counter() - item_started:.2f}s rows={rows}",
            flush=True,
        )

    for name, fetch in (("margin_sse", lambda: datafeed.margin_sse(start, end)),
                        ("margin_szse", lambda: datafeed.margin_szse(end)),
                        ("margin_underlying_szse", lambda: datafeed.margin_underlying_szse(end))):
        item_started = time.perf_counter()
        try:
            fetch_started = time.perf_counter()
            df = fetch()
            fetch_seconds = time.perf_counter() - fetch_started
            keys = ["code", "date"] if "code" in df.columns else ["date"]
            upsert_started = time.perf_counter()
            merged = warehouse.upsert(name, df, keys=keys)
            upsert_seconds = time.perf_counter() - upsert_started
            rows = len(merged)
        except Exception as e:  # noqa: BLE001
            fetch_seconds = upsert_seconds = 0.0
            rows = 0
            print(f"[{name}] failed: {type(e).__name__}", flush=True)
        print(
            f"[events:timing] name={name} fetch={fetch_seconds:.2f}s "
            f"upsert_write={upsert_seconds:.2f}s total={time.perf_counter() - item_started:.2f}s rows={rows}",
            flush=True,
        )

    print(f"[events:timing] total={time.perf_counter() - events_started:.2f}s", flush=True)

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
    started = time.perf_counter()
    print("[universe-refresh:start] isolated=true timeout=180s", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "from quant import datafeed; datafeed.refresh_mainboard_universe()"],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as e:  # noqa: BLE001 子进程异常(超时等) -> 沿用现有 universe 文件
        print(
            f"[universe-refresh:done] seconds={time.perf_counter() - started:.2f} "
            f"status=exception error={type(e).__name__}; 沿用现有 universe 文件",
            flush=True,
        )
        return
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
    print(
        f"[universe-refresh:done] seconds={time.perf_counter() - started:.2f} "
        f"rc={r.returncode} output={tail}",
        flush=True,
    )
    if r.returncode != 0:
        print(f"[universe] 子进程刷新失败 rc={r.returncode}; 沿用现有 universe 文件. {tail}", flush=True)


def run(universe: str = "mainboard_active", workers: int = 12, lookback_days: int = 5,
        event_window_days: int = 30, snapshot_dir: str | None = None,
        skip_price: bool = False, skip_valuation: bool = False,
        skip_events: bool = False, skip_fundamentals: bool = False,
        skip_snapshots: bool = False, limit: int = 0, force_latest: bool = False,
        intraday_spot: bool = False, codes_file: str | None = None) -> dict:
    config.ensure_dirs()
    summary: dict = {"universe": universe, "calendar": _sync_trading_calendar()}
    u = config.UNIVERSES[universe]
    if codes_file:
        with open(codes_file, encoding="utf-8") as fh:
            codes = sorted({line.strip() for line in fh if re.fullmatch(r"\\d{6}", line.strip())})
    else:
        if u["kind"] == "mainboard_active":
            universe_path = config.MAINBOARD_UNIVERSE_FILE
            try:
                file_date = dt.datetime.fromtimestamp(os.path.getmtime(universe_path)).date()
            except OSError:
                file_date = None
            if file_date == dt.date.today():
                print(
                    f"[universe-refresh:skip] reason=fresh-today path={universe_path}",
                    flush=True,
                )
            else:
                _refresh_mainboard_universe_isolated()
        codes = datafeed.universe(u["kind"], u["arg"])
    if limit:
        codes = codes[:limit]
    print(f"[universe] {universe} codes={len(codes)} quant_dir={config.QUANT_DIR}")

    summary["n_codes"] = len(codes)
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
                # force_latest already refreshes every locally present code. Avoid
                # 20 serial probe requests whose date result cannot change the stale set.
                price_latest = None
                print("[price:probe-skip] reason=force-latest", flush=True)
            else:
                price_latest = _probe_latest_price_date(codes, lookback_days)
            local_price_dates: dict[str, dt.date | None] = {}
            price_codes = _stale_codes(
                codes,
                warehouse.load_price,
                price_latest,
                "price",
                force_latest=force_latest,
                local_dates_out=local_price_dates,
                last_date_fn=warehouse.latest_price_date,
            )
            summary["price"] = _update_prices_batched(
                price_codes,
                lookback_days=lookback_days,
                workers=workers,
                local_dates=local_price_dates,
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
    args = ap.parse_args()
    res = run(universe=args.universe, workers=args.workers, lookback_days=args.lookback_days,
              event_window_days=args.event_window_days, snapshot_dir=args.snapshot_dir or None,
              skip_price=args.skip_price, skip_valuation=args.skip_valuation,
              skip_events=args.skip_events, skip_fundamentals=args.skip_fundamentals,
              skip_snapshots=args.skip_snapshots, limit=args.limit, force_latest=args.force_latest,
              intraday_spot=args.intraday_spot, codes_file=args.codes_file or None)
    print("[done]", res)
    # 券商 tgw 原生库在解释器退出（atexit/析构）时可能段错误(SIGSEGV)，此时数据已全部落盘。
    # 用 os._exit(0) 跳过 native 析构直接干净退出，避免非零退出码被上游 check=True 误判为失败、
    # 从而中断后续训练。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
