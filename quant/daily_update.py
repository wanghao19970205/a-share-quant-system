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


def _stale_codes(codes: list[str], load_fn, latest_date: dt.date | None, label: str,
                 force_latest: bool = False) -> list[str]:
    local_dates: dict[str, dt.date | None] = {}
    local_max: dt.date | None = None
    for code in codes:
        local_latest = _last_date(load_fn(code))
        local_dates[code] = local_latest
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


def run(universe: str = "mainboard_active", workers: int = 12, lookback_days: int = 5,
        event_window_days: int = 30, snapshot_dir: str | None = None,
        skip_price: bool = False, skip_valuation: bool = False,
        skip_events: bool = False, skip_fundamentals: bool = False,
        skip_snapshots: bool = False, limit: int = 0, force_latest: bool = False,
        intraday_spot: bool = False, codes_file: str | None = None) -> dict:
    config.ensure_dirs()
    u = config.UNIVERSES[universe]
    if codes_file:
        with open(codes_file, encoding="utf-8") as fh:
            codes = sorted({line.strip() for line in fh if re.fullmatch(r"\\d{6}", line.strip())})
    else:
        if u["kind"] == "mainboard_active":
            datafeed.refresh_mainboard_universe()
        codes = datafeed.universe(u["kind"], u["arg"])
    if limit:
        codes = codes[:limit]
    print(f"[universe] {universe} codes={len(codes)} quant_dir={config.QUANT_DIR}")

    summary: dict = {"universe": universe, "n_codes": len(codes)}
    if not skip_price:
        if intraday_spot:
            summary["price"] = update_intraday_spot(codes, workers=workers)
            summary["price"]["mode"] = (
                "broker-daily-bars"
                if summary["price"].get("source") == "AmazingData"
                else "whole-market-spot"
            )
        else:
            price_latest = _probe_latest_price_date(codes, lookback_days)
            price_codes = _stale_codes(codes, warehouse.load_price, price_latest, "price",
                                       force_latest=force_latest)
            summary["price"] = _run_batch(price_codes, lambda c: _update_one_price(c, lookback_days), workers, "price")
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
