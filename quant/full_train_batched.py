"""Memory-conscious full-A training entry.

This script avoids loading the whole full-A factor panel at once while building it:
- build raw factors by stock batches;
- store raw rows by month;
- prepare continuous features month by month;
- train walk-forward windows by loading only the months needed for that window.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, config, model as qmodel, select as factor_select, tradability, warehouse
from quant.factors import engineering
from quant.logging_utils import install_timestamped_stdout


install_timestamped_stdout()

MODEL_NAME = "ridge_lightgbm_ranker_ensemble"
# Bump when the per-window prediction computation changes in a way that invalidates
# previously cached window outputs (see _recipe_signature / window_cache_dir).
WINDOW_CACHE_VERSION = 1


def _recipe_signature_payload(ignore_universe: bool, **kwargs) -> str:
    payload = {"_v": WINDOW_CACHE_VERSION, "model": MODEL_NAME}
    for key, value in kwargs.items():
        if ignore_universe and key == "universe_codes":
            continue
        payload[key] = sorted(map(str, value)) if key == "factors" else value
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _recipe_signature(**kwargs) -> str:
    """Stable hash of parameters that change a window's stored predictions.

    Universe membership is applied when cached predictions are read. The monthly
    panel file signature already invalidates windows whose actual input rows changed,
    so raw universe-file churn must not invalidate every historical window.
    """
    return _recipe_signature_payload(True, **kwargs)


def _legacy_recipe_signature(**kwargs) -> str:
    """Return the pre-optimization signature so existing caches can be promoted."""
    return _recipe_signature_payload(False, **kwargs)


def _window_month_signature(prepared_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Identity of the monthly parquet parts feeding a window: (name, size, mtime_ns).

    Daily refresh rewrites only the latest month, so historical windows keep a stable
    signature (cache hit) while any window touching a refreshed month misses and recomputes.
    """
    sig = []
    for p in _prepared_files(prepared_dir):
        month = pd.Timestamp(p.stem + "-01")
        month_end = month + pd.offsets.MonthEnd(1)
        if month_end < start or month >= end:
            continue
        st = p.stat()
        sig.append([p.name, int(st.st_size), int(st.st_mtime_ns)])
    return sig


def _window_cache_key(prepared_dir: Path, recipe_sig: str, train_start: pd.Timestamp,
                      valid_start: pd.Timestamp, current: pd.Timestamp,
                      test_end: pd.Timestamp) -> str:
    payload = {
        "recipe": recipe_sig,
        "train_start": str(pd.Timestamp(train_start).date()),
        "valid_start": str(pd.Timestamp(valid_start).date()),
        "current": str(pd.Timestamp(current).date()),
        "test_end": str(pd.Timestamp(test_end).date()),
        "months": _window_month_signature(prepared_dir, train_start, test_end),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _cached_prediction_valid(
    frame: pd.DataFrame,
    current: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[bool, str]:
    """Guard window-cache reuse against stale or malformed prediction files."""
    required = {"code", "date"}
    if frame.empty:
        return False, "empty"
    if not required.issubset(frame.columns):
        return False, "missing-code-date"
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        return False, "invalid-date"
    if (dates < pd.Timestamp(current)).any() or (dates >= pd.Timestamp(test_end)).any():
        return False, "date-out-of-window"
    if frame.duplicated(["code", "date"]).any():
        return False, "duplicate-code-date"
    return True, "ok"


def _chunks(items: list[str], n: int):
    n = max(int(n), 1)
    for i in range(0, len(items), n):
        yield i // n, items[i:i + n]


def _daily_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if not std or not pd.notna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _panel_dirs(name: str) -> tuple[Path, Path, Path]:
    root = Path(config.QUANT_DIR) / f"{name}_parts"
    return root, root / "raw_monthly", root / "prepared_monthly"


def _month_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m")


def _prepared_files(prepared_dir: Path) -> list[Path]:
    return sorted(prepared_dir.glob("*.parquet"))


def _prepared_source_signatures(prepared_dir: Path) -> dict[str, str]:
    """Return stable monthly versions for persisted factor-IC cache entries."""
    signatures = {}
    for path in _prepared_files(prepared_dir):
        try:
            stat = path.stat()
            signatures[path.stem] = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            continue
    return signatures


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _latest_price_month(codes: list[str]) -> pd.Timestamp | None:
    latest: pd.Timestamp | None = None
    for code in codes:
        df = warehouse.load_price(code)
        if df.empty or "date" not in df.columns:
            continue
        d = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.notna(d) and (latest is None or d > latest):
            latest = d
    return _month_floor(latest) if latest is not None else None


def _prepared_date_count(files: list[Path]) -> int:
    vals = []
    for path in files:
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        vals.append(pd.to_datetime(df["date"], errors="coerce"))
    if not vals:
        return 0
    return int(pd.concat(vals, ignore_index=True).dropna().nunique())


def _refresh_month_keys(
    codes: list[str],
    refresh_months: int,
    existing_prepared: list[Path] | None = None,
) -> set[str]:
    if refresh_months <= 0:
        return set()
    latest_month = _latest_price_month(codes)
    if latest_month is None:
        return set()

    months = set(
        str(m)
        for m in pd.period_range(
            latest_month - pd.DateOffset(months=refresh_months - 1),
            latest_month,
            freq="M",
        )
    )
    existing_keys = {
        path.stem for path in (existing_prepared or [])
        if len(path.stem) == 7 and path.stem[4] == "-"
    }
    if existing_keys:
        first_prepared = pd.Period(min(existing_keys), freq="M")
        latest_available = latest_month.to_period("M")
        months.update(
            str(m)
            for m in pd.period_range(
                first_prepared,
                latest_available,
                freq="M",
            )
            if str(m) not in existing_keys
        )
    return months


class MonthFrameCache:
    def __init__(self, max_months: int = 0):
        self.max_months = max(int(max_months), 0)
        self._frames: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def read(self, path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if self.max_months <= 0:
            self.misses += 1
            return pd.read_parquet(path, columns=columns)
        if path in self._frames:
            self.hits += 1
            df = self._frames.pop(path)
            self._frames[path] = df
        else:
            self.misses += 1
            df = pd.read_parquet(path, columns=columns)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            self._frames[path] = df
            while len(self._frames) > self.max_months:
                self._frames.popitem(last=False)
        return df.copy()

    def stats(self) -> str:
        return f"cache_months={len(self._frames)} hits={self.hits} misses={self.misses}"


def _purged_end(dates: pd.Series, boundary: pd.Timestamp, horizon: int) -> pd.Timestamp | None:
    """Return the last signal date whose forward label ends before boundary."""
    eligible = pd.Series(pd.to_datetime(dates, errors="coerce").dropna().unique())
    eligible = eligible[eligible < pd.Timestamp(boundary)].sort_values().reset_index(drop=True)
    if len(eligible) <= int(horizon):
        return None
    return pd.Timestamp(eligible.iloc[-int(horizon) - 1])


class DailyICCache:
    """Reuse date-local factor IC values across windows and daily runs."""

    def __init__(self, workers: int = 8, path: Path | None = None,
                 source_signatures: dict[str, str] | None = None):
        self.workers = max(int(workers), 1)
        self.path = path
        self.source_signatures = source_signatures or {}
        self._frames: dict[tuple[int, tuple[str, ...]], pd.DataFrame] = {}
        if path is not None and path.exists():
            try:
                saved = pd.read_parquet(path)
                if {"horizon", "factor_set", "date", "factor", "ic", "source_sig"}.issubset(saved.columns):
                    for (saved_horizon, factor_set), frame in saved.groupby(
                        ["horizon", "factor_set"], sort=False
                    ):
                        factors = tuple(str(f) for f in str(factor_set).split("|"))
                        self._frames[(int(saved_horizon), factors)] = frame.copy()
            except Exception:  # noqa: BLE001
                # A corrupt/stale auxiliary cache must never block training.
                self._frames = {}

    def get(self, train: pd.DataFrame, factors: list[str], horizon: int) -> tuple[pd.DataFrame, int, int]:
        use = tuple(factors)
        key = (int(horizon), use)
        desired = pd.to_datetime(train["date"], errors="coerce")
        desired_dates = pd.Index(desired.dropna().unique())
        cached = self._frames.get(
            key,
            pd.DataFrame(columns=["horizon", "factor_set", "date", "factor", "ic", "source_sig"]),
        )
        if not cached.empty:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            valid = cached["date"].dt.strftime("%Y-%m").map(self.source_signatures)
            cached = cached[valid.eq(cached["source_sig"])].copy()
        cached_dates = pd.Index(cached["date"].dropna().unique())
        missing_dates = desired_dates.difference(cached_dates)
        if len(missing_dates):
            missing = train[desired.isin(missing_dates)]
            calculated = factor_select.daily_ic(
                missing,
                list(use),
                horizon=horizon,
                workers=self.workers,
            )
            if not calculated.empty:
                calculated = calculated.copy()
                calculated["horizon"] = int(horizon)
                calculated["factor_set"] = "|".join(use)
                calculated["source_sig"] = pd.to_datetime(
                    calculated["date"], errors="coerce"
                ).dt.strftime("%Y-%m").map(self.source_signatures)
                cached = pd.concat([cached, calculated], ignore_index=True)
                cached = cached.drop_duplicates(["date", "factor"], keep="last")
                self._frames[key] = cached
        result = cached[pd.to_datetime(cached["date"], errors="coerce").isin(desired_dates)].copy()
        return result, int(len(desired_dates) - len(missing_dates)), int(len(missing_dates))

    def save(self) -> None:
        if self.path is None or not self._frames:
            return
        frames = [frame for frame in self._frames.values() if not frame.empty]
        if not frames:
            return
        payload = pd.concat(frames, ignore_index=True).drop_duplicates(
            ["horizon", "factor_set", "date", "factor"], keep="last"
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload.to_parquet(tmp, index=False)
        tmp.replace(self.path)


def _rolling_factor_selection(train: pd.DataFrame, factors: list[str], horizon: int,
                              top_n: int = 30, preselect_multiplier: int = 2,
                              max_ic_corr: float = 0.85,
                              daily_ic_cache: DailyICCache | None = None) -> tuple[list[str], pd.DataFrame]:
    """Select stable factors using only the purged training slice and remove IC-series clones."""
    use = [f for f in factors if f in train.columns]
    if not use or train.empty:
        return [], pd.DataFrame()
    if daily_ic_cache is None:
        ic = factor_select.daily_ic(train, use, horizon=horizon, workers=8)
    else:
        ic, cache_hits, cache_misses = daily_ic_cache.get(train, use, horizon)
        print(
            f"[train:factor-ic-cache] hit_dates={cache_hits} miss_dates={cache_misses}",
            flush=True,
        )
    if ic.empty:
        return [], pd.DataFrame()
    summary = factor_select.ic_summary_from_daily_ic(ic)
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary = summary[(summary["ic_count"] >= 20) & summary["ic_mean"].notna()].copy()
    if summary.empty:
        return [], pd.DataFrame()
    summary["stability_score"] = (
        summary["abs_ic_mean"].fillna(0.0)
        * np.sqrt(summary["ic_count"].clip(lower=1))
        * (0.5 + (summary["ic_win_rate"].fillna(0.5) - 0.5).abs())
    )
    pre_n = max(int(top_n), int(top_n) * max(int(preselect_multiplier), 1))
    candidates = summary.sort_values(
        ["stability_score", "abs_ic_mean", "icir"], ascending=[False, False, False]
    ).head(pre_n)
    wide = ic[ic["factor"].isin(candidates["factor"])].pivot(index="date", columns="factor", values="ic")
    selected: list[str] = []
    redundant: dict[str, tuple[str, float]] = {}
    for factor in candidates["factor"].astype(str):
        blocker = ""
        blocker_corr = np.nan
        for kept in selected:
            pair = wide[[factor, kept]].dropna() if factor in wide and kept in wide else pd.DataFrame()
            corr = pair[factor].corr(pair[kept]) if len(pair) >= 20 else np.nan
            if pd.notna(corr) and abs(float(corr)) >= float(max_ic_corr):
                blocker, blocker_corr = kept, float(corr)
                break
        if blocker:
            redundant[factor] = (blocker, blocker_corr)
            continue
        selected.append(factor)
        if len(selected) >= int(top_n):
            break
    audit = summary.copy()
    audit["selected"] = audit["factor"].astype(str).isin(selected)
    audit["redundant_with"] = audit["factor"].astype(str).map(lambda f: redundant.get(f, ("", np.nan))[0])
    audit["redundant_ic_corr"] = audit["factor"].astype(str).map(lambda f: redundant.get(f, ("", np.nan))[1])
    return selected, audit


def build_monthly_panel(name: str, horizon: int, batch_size: int,
                        min_price_rows: int, rebuild: bool = False,
                        limit_codes: int = 0, refresh_months: int = 0,
                        universe_file: str | None = None) -> Path:
    root, raw_dir, prepared_dir = _panel_dirs(name)
    if rebuild and root.exists():
        shutil.rmtree(root)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if universe_file:
        with open(universe_file, encoding="utf-8") as f:
            codes = sorted({token for token in f.read().split() if len(token) == 6 and token.isdigit()})
        if not codes:
            raise RuntimeError(f"training universe file is empty: {universe_file}")
    else:
        codes = engineering._read_codes(0)  # noqa: SLF001
    codes = engineering._filter_codes_by_price_rows(codes, min_price_rows)  # noqa: SLF001
    if limit_codes:
        codes = codes[:limit_codes]

    existing_prepared = _prepared_files(prepared_dir)
    has_trainable_history = _prepared_date_count(existing_prepared) >= 30
    refresh_keys = (
        _refresh_month_keys(codes, refresh_months, existing_prepared)
        if not rebuild and has_trainable_history
        else set()
    )
    if existing_prepared and not rebuild and not refresh_keys and has_trainable_history:
        print(f"[panel] reuse prepared monthly parts: {len(existing_prepared)}", flush=True)
        return prepared_dir
    if existing_prepared and not rebuild and not has_trainable_history:
        print(
            f"[panel] existing prepared history too short ({_prepared_date_count(existing_prepared)} dates); rebuilding full monthly panel",
            flush=True,
        )
        if root.exists():
            shutil.rmtree(root)
        prepared_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        existing_prepared = []

    if refresh_keys:
        for month in refresh_keys:
            month_dir = raw_dir / month
            if month_dir.exists():
                shutil.rmtree(month_dir)
            prepared_path = prepared_dir / f"{month}.parquet"
            if prepared_path.exists():
                prepared_path.unlink()
        print(f"[panel] refresh recent months: {sorted(refresh_keys)}", flush=True)

    refresh_start = (
        min(pd.Timestamp(f"{month}-01") for month in refresh_keys)
        if refresh_keys
        else None
    )
    if refresh_start is not None:
        print(
            f"[panel] incremental output_start={refresh_start.date()} warmup_rows=260",
            flush=True,
        )

    print(f"[panel] codes={len(codes)} batch_size={batch_size} min_price_rows={min_price_rows}", flush=True)
    shared_started = time.perf_counter()
    all_codes = set(codes)
    shared_factors = {
        "financial_yjbb": engineering._asof_report_factor("financial_yjbb", "yjbb", all_codes),  # noqa: SLF001
        "income": engineering._asof_report_factor("income", "income", all_codes),  # noqa: SLF001
        "cashflow": engineering._asof_report_factor("cashflow", "cashflow", all_codes),  # noqa: SLF001
        "balance": engineering._asof_report_factor("balance", "balance", all_codes),  # noqa: SLF001
        "performance_forecast": engineering._forecast_events(all_codes),  # noqa: SLF001
        "margin_underlying_szse": engineering._margin_underlying(all_codes),  # noqa: SLF001
        "block_trades": engineering._event_counts("block_trades", all_codes, "block_trade"),  # noqa: SLF001
        "lhb": engineering._event_counts("lhb", all_codes, "lhb"),  # noqa: SLF001
    }
    print(
        f"[panel] shared factor cache seconds={time.perf_counter() - shared_started:.2f}",
        flush=True,
    )

    for bi, batch in _chunks(codes, batch_size):
        batch_started = time.perf_counter()
        raw_write_seconds = 0.0
        raw = engineering.build_panel(
            codes=batch,
            horizon=horizon,
            min_price_rows=0,
            output_start=refresh_start,
            warmup_rows=260,
            shared_factors=shared_factors,
        )
        if raw.empty:
            print(f"[panel:raw] batch={bi} empty", flush=True)
            continue
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])
        for month, part in raw.groupby(_month_key(raw["date"]), sort=True):
            if not month or month == "NaT":
                continue
            if refresh_keys and str(month) not in refresh_keys:
                continue
            month_dir = raw_dir / str(month)
            month_dir.mkdir(parents=True, exist_ok=True)
            write_started = time.perf_counter()
            part.to_parquet(month_dir / f"part_{bi:05d}.parquet", index=False)
            raw_write_seconds += time.perf_counter() - write_started
        print(
            f"[panel:raw:timing] batch={bi} total={time.perf_counter() - batch_started:.2f}s "
            f"write={raw_write_seconds:.2f}s codes={len(batch)} rows={len(raw)}",
            flush=True,
        )
        print(f"[panel:raw] batch={bi} codes={len(batch)} rows={len(raw)}", flush=True)
        del raw
        gc.collect()

    month_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if refresh_keys:
        month_dirs = [p for p in month_dirs if p.name in refresh_keys]

    def prepare_month(month_dir: Path) -> tuple[str, int, int, pd.DataFrame]:
        month_started = time.perf_counter()
        out_path = prepared_dir / f"{month_dir.name}.parquet"
        part_files = sorted(month_dir.glob("*.parquet"))
        if not part_files:
            return month_dir.name, 0, 0, pd.DataFrame()
        read_started = time.perf_counter()
        raw = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
        read_seconds = time.perf_counter() - read_started
        feature_started = time.perf_counter()
        prepared, feats = engineering.prepare_features(
            raw,
            horizon=horizon,
            add_discrete=False,
            add_onehot=False,
            drop_target_na=False,
        )
        feature_seconds = time.perf_counter() - feature_started
        write_started = time.perf_counter()
        prepared.to_parquet(out_path, index=False)
        write_seconds = time.perf_counter() - write_started
        summary = engineering.summarize_features(feats)
        rows = len(prepared)
        cols = len(prepared.columns)
        print(
            f"[panel:prepared:timing] month={month_dir.name} read={read_seconds:.2f}s "
            f"features={feature_seconds:.2f}s write={write_seconds:.2f}s "
            f"total={time.perf_counter() - month_started:.2f}s rows={rows} cols={cols}",
            flush=True,
        )
        del raw, prepared
        gc.collect()
        return month_dir.name, rows, cols, summary

    feature_summary_saved = False
    with ThreadPoolExecutor(max_workers=12) as executor:
        for month, rows, cols, summary in executor.map(prepare_month, month_dirs):
            if not rows:
                continue
            if not feature_summary_saved and not summary.empty:
                warehouse.save(f"{name}_feature_summary", summary)
                feature_summary_saved = True
            print(f"[panel:prepared] month={month} rows={rows} cols={cols}", flush=True)

    return prepared_dir


def _walk_forward_window_count(dates: pd.Series, train_months: int,
                               validation_months: int, test_months: int) -> int:
    if dates.empty:
        return 0
    current = pd.Timestamp(dates.min()) + pd.DateOffset(months=train_months + validation_months)
    end = pd.Timestamp(dates.max()) + pd.Timedelta(days=1)
    count = 0
    while current < end:
        count += 1
        current = min(current + pd.DateOffset(months=test_months), end)
    return count


def _read_dates(prepared_dir: Path) -> pd.Series:
    vals = []
    for p in _prepared_files(prepared_dir):
        df = pd.read_parquet(p, columns=["date"])
        vals.append(pd.to_datetime(df["date"], errors="coerce"))
    if not vals:
        return pd.Series(dtype="datetime64[ns]")
    return pd.Series(sorted(pd.concat(vals, ignore_index=True).dropna().unique()))


def _load_window(prepared_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                 columns: list[str] | None = None,
                 cache: MonthFrameCache | None = None) -> pd.DataFrame:
    parts = []
    for p in _prepared_files(prepared_dir):
        month = pd.Timestamp(p.stem + "-01")
        month_end = month + pd.offsets.MonthEnd(1)
        if month_end < start or month >= end:
            continue
        df = cache.read(p, columns=columns) if cache else pd.read_parquet(p, columns=columns)
        if not cache:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[(df["date"] >= start) & (df["date"] < end)]
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# 跨窗口 per-code 可交易口径缓存：price_tradability 读的是整段 price 历史，
# 对同一 code+horizon 其产出与窗口无关（不变量）。walk-forward 36 个窗口本会
# 逐窗重复读 ~2000 个 price 文件 + 跑 rolled_sell_close 纯 Python 循环（36× 冗余，
# 全部计入 factor_select 计时段 → 变体腿虚高 ~188s）。这里按 code 记忆化：
# 首窗为缺失 code 调一次 price_tradability，其后各窗只从内存切片，命中即零 IO。
# 仅进程内、仅非 baseline 腿会用到；键含 horizon 以隔离不同 h 的口径。
_TRAD_CACHE: "dict[tuple[str, int], pd.DataFrame]" = {}


def _cached_price_tradability(codes: list[str], horizon: int) -> pd.DataFrame:
    """返回 codes 的可交易口径，仅对未缓存的 code 调用 price_tradability。

    与直接 tradability.price_tradability(codes, [horizon]) 等价（同一批 code 的并集），
    差别仅是把 per-code 结果记忆化，跨窗口复用。空结果的 code 也缓存（存空帧），
    避免对不存在 price 文件的 code 反复触发磁盘 stat。"""
    missing = [c for c in codes if (c, horizon) not in _TRAD_CACHE]
    if missing:
        fresh = tradability.price_tradability(missing, [horizon])
        if not fresh.empty:
            fresh["code"] = fresh["code"].astype(str).str.zfill(6)
            for code, grp in fresh.groupby("code", sort=False):
                _TRAD_CACHE[(code, horizon)] = grp.reset_index(drop=True)
        # 没产出行的 code 也落一个空帧占位，防下窗重复读盘。
        for code in missing:
            _TRAD_CACHE.setdefault((code, horizon), pd.DataFrame())
    parts = [_TRAD_CACHE[(c, horizon)] for c in codes
             if not _TRAD_CACHE[(c, horizon)].empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _join_tradability(window: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """为窗口 join 可交易口径列（tradable_ret_{h}d / buyable_close），供 A/B 变体使用。

    仅新增标签/掩码列，不新增特征列（buyable_close 只用当日 OHLC、因果安全；
    tradable_ret 含 shift(-h) 未来价，是标签本就该含未来，非泄漏）。
    join 键：code(str, zfill6) + date(datetime64[ns])，与 tradability.price_tradability 输出一致。
    join 后断言 tradable_ret 命中率，防 code/date 规范化不一致导致静默丢样本。"""
    codes = sorted(window["code"].astype(str).str.zfill(6).unique())
    trad = _cached_price_tradability(codes, horizon)
    keep = ["code", "date", f"tradable_ret_{horizon}d", "buyable_close"]
    keep = [c for c in keep if c in trad.columns]
    if trad.empty or len(keep) <= 2:
        raise RuntimeError(
            f"tradability join produced no usable columns for horizon={horizon}; "
            f"cannot run A/B variant without buyable_close/tradable_ret"
        )
    out = window.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    trad = trad[keep].drop_duplicates(["code", "date"])
    out = out.merge(trad, on=["code", "date"], how="left")
    # 命中率自检：tradable_ret 尾部丢尾会有正常 NaN，但整体 NaN 率异常高说明键没对上。
    tret = f"tradable_ret_{horizon}d"
    if tret in out.columns and len(out):
        na_rate = float(out[tret].isna().mean())
        # 尾部 horizon 天 + 停牌等正常缺失，给到 20% 冗余；超过即视为键不匹配。
        if na_rate > 0.20:
            print(f"[train:ab] WARN tradable_ret NaN rate={na_rate:.3f} (>0.20) — 检查 code/date join 键", flush=True)
    return out


def _selected_factors(selection_name: str, prepared_dir: Path, horizon: int) -> list[str]:
    sel = warehouse.load(selection_name)
    selected = sel["factor"].astype(str).tolist() if not sel.empty and "factor" in sel.columns else []
    files = _prepared_files(prepared_dir)
    if not files:
        return []
    sample = pd.read_parquet(files[0])
    available = set(engineering.feature_columns(sample, horizon))
    factors = [f for f in selected if f in available]
    if factors:
        return factors
    return [f for f in engineering.feature_columns(sample, horizon) if f in available]


def train_batched(name: str, output_prefix: str, selection_name: str, horizon: int,
                  train_months: int, validation_months: int, test_months: int,
                  top_n: int, ridge_quantile: float | None, lgbm_weight: float,
                  ic_weight: float, rank_vote_weight: float,
                  max_weight: float | None, positive_only: bool,
                  n_estimators: int, learning_rate: float | None,
                  early_stopping_rounds: int, decay_half_life_days: float | None,
                  min_weight: float, model_threads: int = 0,
                  max_windows: int = 0, skip_windows: int = 0,
                  month_cache_size: int = 30, rolling_factor_select: bool = False,
                  rolling_top_factors: int = 30, max_factor_ic_corr: float = 0.85,
                  purge_horizon: bool = False, recent_windows: int = 0,
                  expanding_train: bool = False,
                  elastic_net: bool = False, elastic_alpha: float = 0.001,
                  elastic_l1_ratio: float = 0.5, catboost_ranker: bool = False,
                  catboost_estimators: int = 200,
                  catboost_learning_rate: float = 0.03,
                  catboost_max_train_rows: int = 300_000,
                  extra_trees: bool = False, extra_trees_estimators: int = 120,
                  extra_trees_max_train_rows: int = 300_000,
                  window_cache_dir: str | None = None,
                  universe_file: str | None = None,
                  train_target_mode: str = "baseline") -> pd.DataFrame:
    _, _, prepared_dir = _panel_dirs(name)
    if train_target_mode not in ("baseline", "buyin-mask", "tradable-label"):
        raise ValueError(f"unknown train_target_mode: {train_target_mode}")
    # A/B 训练口径：baseline=现役（target_ret 标签、全样本）；
    #   buyin-mask=剔训练段封涨停买入日；tradable-label=用跌停顺延后的可实现收益作标签。
    ab_label_col = f"tradable_ret_{horizon}d" if train_target_mode == "tradable-label" else None
    ab_mask_col = "buyable_close" if train_target_mode == "buyin-mask" else None
    universe_codes: set[str] | None = None
    if universe_file:
        with open(universe_file, encoding="utf-8") as f:
            universe_codes = {token for token in f.read().split() if len(token) == 6 and token.isdigit()}
        if not universe_codes:
            raise RuntimeError(f"training universe file is empty: {universe_file}")
    factors = _selected_factors(selection_name, prepared_dir, horizon)
    if not factors:
        raise RuntimeError("no usable factors for full-A batched training")
    dates = _read_dates(prepared_dir)
    if len(dates) < 30:
        raise RuntimeError("prepared panel has too few dates")
    if recent_windows > 0:
        total_windows = _walk_forward_window_count(dates, train_months, validation_months, test_months)
        skip_windows = max(total_windows - int(recent_windows), 0)
        max_windows = min(int(recent_windows), total_windows)
        print(f"[train] recent_windows={recent_windows} total_windows={total_windows} skip_windows={skip_windows}", flush=True)

    train_delta = pd.DateOffset(months=train_months)
    valid_delta = pd.DateOffset(months=validation_months)
    test_delta = pd.DateOffset(months=test_months)
    current = dates.min() + train_delta + valid_delta
    latest_date = pd.Timestamp(dates.max())
    end = latest_date + pd.Timedelta(days=1)
    target = f"target_ret_{horizon}d"
    pred_parts = []
    audit_parts: list[pd.DataFrame] = []
    selected_counts: list[int] = []
    side_cols = ["volatility_10", "turnover", "rule_score"]
    windows = 0
    month_cache = MonthFrameCache(month_cache_size)
    daily_ic_cache: DailyICCache | None = None

    recipe_sig = None
    legacy_recipe_sig = None
    cache_dir = None
    cache_hits = 0
    cache_misses = 0
    if window_cache_dir:
        recipe_params = dict(
            factors=factors, horizon=horizon, train_months=train_months,
            validation_months=validation_months, test_months=test_months,
            decay_half_life_days=decay_half_life_days, min_weight=min_weight,
            n_estimators=n_estimators, learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds, lgbm_weight=lgbm_weight,
            model_threads=model_threads, ic_weight=ic_weight, rank_vote_weight=rank_vote_weight,
            elastic_net=elastic_net, elastic_alpha=elastic_alpha, elastic_l1_ratio=elastic_l1_ratio,
            catboost_ranker=catboost_ranker, catboost_estimators=catboost_estimators,
            catboost_learning_rate=catboost_learning_rate,
            catboost_max_train_rows=catboost_max_train_rows, extra_trees=extra_trees,
            extra_trees_estimators=extra_trees_estimators,
            extra_trees_max_train_rows=extra_trees_max_train_rows,
            purge_horizon=purge_horizon, expanding_train=expanding_train,
            rolling_factor_select=rolling_factor_select, rolling_top_factors=rolling_top_factors,
            max_factor_ic_corr=max_factor_ic_corr,
            universe_codes=sorted(universe_codes) if universe_codes is not None else [],
        )
        # 仅当非 baseline 时才注入 A/B 字段，保证 baseline 腿 recipe 哈希与现役完全一致（复用既有缓存）。
        if train_target_mode != "baseline":
            recipe_params["train_target_mode"] = train_target_mode
        recipe_sig = _recipe_signature(**recipe_params)
        legacy_recipe_sig = _legacy_recipe_signature(**recipe_params)
        cache_dir = Path(window_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        daily_ic_cache = DailyICCache(
            workers=min(model_threads or 8, 8),
            path=cache_dir / "daily_ic_cache.parquet",
            source_signatures=_prepared_source_signatures(prepared_dir),
        )
        print(f"[train:cache] window cache enabled dir={cache_dir} recipe={recipe_sig[:12]}", flush=True)

    if daily_ic_cache is None:
        daily_ic_cache = DailyICCache(workers=min(model_threads or 8, 8))

    print(
        f"[train] factors={len(factors)} date_range={dates.min().date()}..{dates.max().date()} "
        f"month_cache_size={month_cache_size}",
        flush=True,
    )
    while current < end:
        window_started = time.perf_counter()
        if max_windows and windows >= skip_windows + max_windows:
            break
        train_start = (
            pd.Timestamp(dates.min())
            if expanding_train
            else current - valid_delta - train_delta
        )
        valid_start = current - valid_delta
        test_end = min(current + test_delta, end)
        if windows < skip_windows:
            windows += 1
            current = test_end
            continue
        window_cache_key = None
        stage_started = time.perf_counter()
        if cache_dir is not None:
            window_cache_key = _window_cache_key(
                prepared_dir, recipe_sig, train_start, valid_start, current, test_end)
            cache_path = cache_dir / f"{window_cache_key}.parquet"
            legacy_cache_path = None
            if legacy_recipe_sig and legacy_recipe_sig != recipe_sig:
                legacy_key = _window_cache_key(
                    prepared_dir, legacy_recipe_sig, train_start, valid_start, current, test_end)
                legacy_cache_path = cache_dir / f"{legacy_key}.parquet"
            read_path = cache_path if cache_path.exists() else legacy_cache_path
            if read_path is not None and read_path.exists():
                try:
                    cached = pd.read_parquet(read_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[train:cache] window={windows} read failed, recomputing: {exc}", flush=True)
                    cached = None
                if cached is not None:
                    cache_valid, cache_reason = _cached_prediction_valid(
                        cached, current, test_end
                    )
                    if not cache_valid:
                        print(
                            f"[train:cache] window={windows} invalid reason={cache_reason}; recomputing",
                            flush=True,
                        )
                        cached = None
                if cached is not None:
                    if read_path != cache_path:
                        cached.to_parquet(cache_path, index=False)
                        print(
                            f"[train:cache] window={windows} promoted legacy key={read_path.stem[:12]}",
                            flush=True,
                        )
                    if universe_codes is not None:
                        cached = cached[cached["code"].astype(str).isin(universe_codes)].copy()
                    pred_parts.append(cached)
                    cache_hits += 1
                    print(f"[train] window={windows} cache-hit rows={len(cached)} "
                          f"test={current.date()}..{test_end.date()} key={window_cache_key[:12]} "
                          f"read_seconds={time.perf_counter() - stage_started:.2f} "
                          f"total_seconds={time.perf_counter() - window_started:.2f}", flush=True)
                    windows += 1
                    current = test_end
                    continue
            cache_misses += 1
            if cache_dir is not None:
                print(
                    f"[train:cache] window={windows} miss test={current.date()}..{test_end.date()} "
                    f"reason=signature-or-missing key={window_cache_key[:12]}",
                    flush=True,
                )
        need_cols = ["code", "date", target] + sorted(set(factors + side_cols))
        window = _load_window(
            prepared_dir,
            train_start,
            test_end,
            columns=[c for c in need_cols if c],
            cache=month_cache,
        )
        print(f"[train:timing] window={windows} stage=load_window "
              f"seconds={time.perf_counter() - stage_started:.2f} rows={len(window)} "
              f"cache={month_cache.stats()}", flush=True)
        stage_started = time.perf_counter()
        if universe_codes is not None and not window.empty:
            window = window[window["code"].astype(str).isin(universe_codes)].copy()
        # A/B：非 baseline 时，从 price 文件为本窗口 codes join 可交易列（tradable_ret_{h}d / buyable_close）。
        # prepared 面板本就没有这两列；仅新增标签/掩码列，不引入任何特征列（因果安全）。
        if train_target_mode != "baseline" and not window.empty:
            window = _join_tradability(window, horizon)
        factors_in_window = [f for f in factors if f in window.columns]
        train_end_ts = _purged_end(window["date"], valid_start, horizon) if purge_horizon else valid_start - pd.Timedelta(days=1)
        valid_end_ts = _purged_end(window["date"], current, horizon) if purge_horizon else current - pd.Timedelta(days=1)
        if train_end_ts is None or valid_end_ts is None or valid_end_ts < valid_start:
            print(f"[train] window={windows} skipped: purge leaves insufficient train/valid dates", flush=True)
            windows += 1
            current = test_end
            continue
        if rolling_factor_select:
            selection_train = window[(window["date"] >= train_start) & (window["date"] <= train_end_ts)]
            picked, audit = _rolling_factor_selection(
                selection_train,
                factors_in_window,
                horizon=horizon,
                top_n=rolling_top_factors,
                max_ic_corr=max_factor_ic_corr,
                daily_ic_cache=daily_ic_cache,
            )
            if picked:
                factors_in_window = picked
            if not audit.empty:
                audit = audit.copy()
                audit.insert(0, "window", windows)
                audit.insert(1, "train_start", train_start)
                audit.insert(2, "train_end", train_end_ts)
                audit.insert(3, "test_start", current)
                audit_parts.append(audit)
            print(f"[train:timing] window={windows} stage=factor_select "
                  f"seconds={time.perf_counter() - stage_started:.2f} factors={len(factors_in_window)}", flush=True)
            stage_started = time.perf_counter()
        selected_counts.append(len(factors_in_window))
        train_end = train_end_ts.strftime("%Y-%m-%d")
        valid_end = valid_end_ts.strftime("%Y-%m-%d")
        model_window = window[
            (window["date"] <= train_end_ts)
            | ((window["date"] >= valid_start) & (window["date"] <= valid_end_ts))
            | (window["date"] >= current)
        ].copy()
        print(f"[train] window={windows} train={train_start.date()}..{train_end} valid={valid_start.date()}..{valid_end} "
              f"test={current.date()}..{test_end.date()} rows={len(model_window)} factors={len(factors_in_window)}", flush=True)

        ridge_res = qmodel.train_ridge(
            model_window,
            factors_in_window,
            horizon=horizon,
            train_end=train_end,
            valid_end=valid_end,
            predict_start=current.strftime("%Y-%m-%d"),
            decay_half_life_days=decay_half_life_days,
            min_weight=min_weight,
            label_col=ab_label_col,
            train_mask_col=ab_mask_col,
        )
        print(f"[train:timing] window={windows} stage=ridge "
              f"seconds={time.perf_counter() - stage_started:.2f} ok={ridge_res.ok}", flush=True)
        stage_started = time.perf_counter()
        lgbm_res = qmodel.train_lightgbm_ranker(
            model_window,
            factors_in_window,
            horizon=horizon,
            train_end=train_end,
            valid_end=valid_end,
            predict_start=current.strftime("%Y-%m-%d"),
            decay_half_life_days=decay_half_life_days,
            min_weight=min_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=model_threads or None,
            label_col=ab_label_col,
            train_mask_col=ab_mask_col,
        )
        print(f"[train:timing] window={windows} stage=lightgbm_ranker "
              f"seconds={time.perf_counter() - stage_started:.2f} ok={lgbm_res.ok}", flush=True)
        stage_started = time.perf_counter()
        ic_res = None
        elastic_res = None
        model_pair_started = time.perf_counter()
        if ic_weight and elastic_net:
            with ThreadPoolExecutor(max_workers=2) as executor:
                ic_future = executor.submit(
                    qmodel.train_ic_weighted,
                    model_window,
                    factors_in_window,
                    horizon=horizon,
                    train_end=train_end,
                    valid_end=valid_end,
                    predict_start=current.strftime("%Y-%m-%d"),
                    decay_half_life_days=decay_half_life_days,
                    min_weight=min_weight,
                    label_col=ab_label_col,
                    train_mask_col=ab_mask_col,
                )
                elastic_future = executor.submit(
                    qmodel.train_elastic_net,
                    model_window,
                    factors_in_window,
                    horizon=horizon,
                    alpha=elastic_alpha,
                    l1_ratio=elastic_l1_ratio,
                    train_end=train_end,
                    valid_end=valid_end,
                    predict_start=current.strftime("%Y-%m-%d"),
                    decay_half_life_days=decay_half_life_days,
                    min_weight=min_weight,
                    label_col=ab_label_col,
                    train_mask_col=ab_mask_col,
                )
                ic_res = ic_future.result()
                elastic_res = elastic_future.result()
            print(f"[train:timing] window={windows} stage=ic_elastic_parallel "
                  f"seconds={time.perf_counter() - model_pair_started:.2f} "
                  f"ic_ok={ic_res.ok} elastic_ok={elastic_res.ok}", flush=True)
        else:
            if ic_weight:
                ic_res = qmodel.train_ic_weighted(
                    model_window,
                    factors_in_window,
                    horizon=horizon,
                    train_end=train_end,
                    valid_end=valid_end,
                    predict_start=current.strftime("%Y-%m-%d"),
                    decay_half_life_days=decay_half_life_days,
                    min_weight=min_weight,
                    label_col=ab_label_col,
                    train_mask_col=ab_mask_col,
                )
                print(f"[train:timing] window={windows} stage=ic_weighted "
                      f"seconds={time.perf_counter() - model_pair_started:.2f} ok={ic_res.ok}", flush=True)
            if elastic_net:
                elastic_res = qmodel.train_elastic_net(
                    model_window,
                    factors_in_window,
                    horizon=horizon,
                    alpha=elastic_alpha,
                    l1_ratio=elastic_l1_ratio,
                    train_end=train_end,
                    valid_end=valid_end,
                    predict_start=current.strftime("%Y-%m-%d"),
                    decay_half_life_days=decay_half_life_days,
                    min_weight=min_weight,
                    label_col=ab_label_col,
                    train_mask_col=ab_mask_col,
                )
                print(f"[train:timing] window={windows} stage=elastic_net "
                      f"seconds={time.perf_counter() - model_pair_started:.2f} ok={elastic_res.ok}", flush=True)
        stage_started = time.perf_counter()
        extra_trees_res = None
        if extra_trees:
            extra_trees_res = qmodel.train_extra_trees(
                model_window,
                factors_in_window,
                horizon=horizon,
                train_end=train_end,
                valid_end=valid_end,
                predict_start=current.strftime("%Y-%m-%d"),
                decay_half_life_days=decay_half_life_days,
                min_weight=min_weight,
                n_estimators=extra_trees_estimators,
                max_train_rows=extra_trees_max_train_rows,
                label_col=ab_label_col,
                train_mask_col=ab_mask_col,
            )
            print(f"[train:timing] window={windows} stage=extra_trees "
                  f"seconds={time.perf_counter() - stage_started:.2f} ok={extra_trees_res.ok}", flush=True)
            stage_started = time.perf_counter()
        catboost_res = None
        if catboost_ranker:
            catboost_res = qmodel.train_catboost_ranker(
                model_window,
                factors_in_window,
                horizon=horizon,
                train_end=train_end,
                valid_end=valid_end,
                predict_start=current.strftime("%Y-%m-%d"),
                decay_half_life_days=decay_half_life_days,
                min_weight=min_weight,
                n_estimators=catboost_estimators,
                learning_rate=catboost_learning_rate,
                early_stopping_rounds=early_stopping_rounds,
                n_jobs=model_threads or None,
                max_train_rows=catboost_max_train_rows,
            )
            print(f"[train:timing] window={windows} stage=catboost_ranker "
                  f"seconds={time.perf_counter() - stage_started:.2f} ok={catboost_res.ok}", flush=True)
            stage_started = time.perf_counter()
        rank_vote_res = None
        if rank_vote_weight:
            rank_vote_res = qmodel.train_rank_vote(
                model_window,
                factors_in_window,
                horizon=horizon,
                train_end=train_end,
                valid_end=valid_end,
                predict_start=current.strftime("%Y-%m-%d"),
                decay_half_life_days=decay_half_life_days,
                min_weight=min_weight,
                label_col=ab_label_col,
                train_mask_col=ab_mask_col,
            )
            print(f"[train:timing] window={windows} stage=rank_vote "
                  f"seconds={time.perf_counter() - stage_started:.2f} ok={rank_vote_res.ok}", flush=True)
            stage_started = time.perf_counter()
        merge_started = time.perf_counter()
        if ridge_res.ok and lgbm_res.ok and not ridge_res.predictions.empty and not lgbm_res.predictions.empty:
            rp = ridge_res.predictions[(ridge_res.predictions["date"] >= current) & (ridge_res.predictions["date"] < test_end)].copy()
            lp = lgbm_res.predictions[(lgbm_res.predictions["date"] >= current) & (lgbm_res.predictions["date"] < test_end)].copy()
            p = lp.rename(columns={"pred": "lgbm_pred"}).merge(
                rp[["code", "date", "pred"]].rename(columns={"pred": "ridge_pred"}),
                on=["code", "date"],
                how="inner",
            )
            if not p.empty:
                if ic_weight and ic_res and ic_res.ok and not ic_res.predictions.empty:
                    ip = ic_res.predictions[(ic_res.predictions["date"] >= current) & (ic_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        ip[["code", "date", "pred"]].rename(columns={"pred": "ic_pred"}),
                        on=["code", "date"],
                        how="left",
                    )
                if elastic_net and elastic_res and elastic_res.ok and not elastic_res.predictions.empty:
                    ep = elastic_res.predictions[(elastic_res.predictions["date"] >= current) & (elastic_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        ep[["code", "date", "pred"]].rename(columns={"pred": "elastic_pred"}),
                        on=["code", "date"],
                        how="left",
                    )
                if extra_trees and extra_trees_res and extra_trees_res.ok and not extra_trees_res.predictions.empty:
                    tp = extra_trees_res.predictions[(extra_trees_res.predictions["date"] >= current) & (extra_trees_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        tp[["code", "date", "pred"]].rename(columns={"pred": "extra_trees_pred"}),
                        on=["code", "date"],
                        how="left",
                    )
                if catboost_ranker and catboost_res and catboost_res.ok and not catboost_res.predictions.empty:
                    cp = catboost_res.predictions[(catboost_res.predictions["date"] >= current) & (catboost_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        cp[["code", "date", "pred"]].rename(columns={"pred": "catboost_pred"}),
                        on=["code", "date"],
                        how="left",
                    )
                if rank_vote_weight and rank_vote_res and rank_vote_res.ok and not rank_vote_res.predictions.empty:
                    vp = rank_vote_res.predictions[(rank_vote_res.predictions["date"] >= current) & (rank_vote_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        vp[["code", "date", "pred"]].rename(columns={"pred": "rank_vote_pred"}),
                        on=["code", "date"],
                        how="left",
                    )
                p["lgbm_z"] = p.groupby("date")["lgbm_pred"].transform(_daily_zscore)
                p["ridge_z"] = p.groupby("date")["ridge_pred"].transform(_daily_zscore)
                p["base_pred"] = float(lgbm_weight) * p["lgbm_z"] + (1 - float(lgbm_weight)) * p["ridge_z"]
                p["pred"] = p["base_pred"]
                if ic_weight and "ic_pred" in p.columns and p["ic_pred"].notna().any():
                    p["ic_z"] = p.groupby("date")["ic_pred"].transform(_daily_zscore)
                    p["pred"] = p["pred"] + float(ic_weight) * p["ic_z"].fillna(0.0)
                if elastic_net and "elastic_pred" in p.columns and p["elastic_pred"].notna().any():
                    p["elastic_z"] = p.groupby("date")["elastic_pred"].transform(_daily_zscore)
                if extra_trees and "extra_trees_pred" in p.columns and p["extra_trees_pred"].notna().any():
                    p["extra_trees_z"] = p.groupby("date")["extra_trees_pred"].transform(_daily_zscore)
                if catboost_ranker and "catboost_pred" in p.columns and p["catboost_pred"].notna().any():
                    p["catboost_z"] = p.groupby("date")["catboost_pred"].transform(_daily_zscore)
                if rank_vote_weight and "rank_vote_pred" in p.columns and p["rank_vote_pred"].notna().any():
                    p["rank_vote_z"] = p.groupby("date")["rank_vote_pred"].transform(_daily_zscore)
                    p["pred"] = p["pred"] + float(rank_vote_weight) * p["rank_vote_z"].fillna(0.0)
                p["model"] = MODEL_NAME
                side_present = [c for c in side_cols if c in model_window.columns]
                if side_present:
                    p = p.merge(model_window[["code", "date"] + side_present].drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
                # A/B：把可交易列并进预测输出，令训练内 portfolio 与统一评测同口径（baseline 无这些列，自动跳过）。
                # 关键：tradable-label 腿的标签列名就是 tradable_ret_{h}d，此时 p 已带该列（来自模型预测输出）。
                # 若不排除，merge 会因列名撞车加 _x/_y 后缀，导致下游 p[keep_cols] 找不到 tradable_ret_{h}d 而 KeyError。
                ab_cols = [c for c in (f"tradable_ret_{horizon}d", "buyable_close")
                           if c in model_window.columns and c not in p.columns]
                if ab_cols:
                    p = p.merge(model_window[["code", "date"] + ab_cols].drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
                # tradable-label 腿的标签列被改名为 tradable_ret_{h}d（model.py 用 label_col 命名输出列），
                # 此时预测输出里没有 target_ret_{h}d。这里用该腿实际的标签列名，避免 KeyError。
                label_out_col = ab_label_col or target
                keep_cols = ["code", "date", label_out_col, "pred", "model", "ridge_pred", "lgbm_pred"]
                keep_cols.extend([c for c in ("ic_pred", "base_pred", "ic_z", "elastic_pred", "elastic_z", "extra_trees_pred", "extra_trees_z", "catboost_pred", "catboost_z", "rank_vote_pred", "rank_vote_z") if c in p.columns])
                keep_cols.extend([c for c in side_cols if c in p.columns])
                keep_cols.extend([c for c in ab_cols if c in p.columns])
                # 去重保序：tradable-label 模式下 label_out_col 就是 tradable_ret_{h}d，与 ab_cols 重列时去重。
                keep_cols = list(dict.fromkeys(keep_cols))
                pred_out = p[keep_cols]
                pred_parts.append(pred_out)
                print(f"[train:timing] window={windows} stage=prediction_merge "
                      f"seconds={time.perf_counter() - merge_started:.2f} rows={len(p)}", flush=True)
                cache_started = time.perf_counter()
                if cache_dir is not None and window_cache_key:
                    try:
                        tmp_path = cache_dir / f".{window_cache_key}.tmp.parquet"
                        pred_out.to_parquet(tmp_path, index=False)
                        os.replace(tmp_path, cache_dir / f"{window_cache_key}.parquet")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[train:cache] window={windows} store failed: {exc}", flush=True)
                if cache_dir is not None and window_cache_key:
                    print(f"[train:timing] window={windows} stage=cache_write "
                          f"seconds={time.perf_counter() - cache_started:.2f}", flush=True)
                print(f"[train] window={windows} predictions={len(p)} "
                      f"total_seconds={time.perf_counter() - window_started:.2f}", flush=True)
        else:
            print(f"[train] window={windows} skipped ridge={ridge_res.message} lgbm={lgbm_res.message}", flush=True)
        windows += 1
        if month_cache.max_months > 0 and windows % 10 == 0:
            print(f"[train:cache] {month_cache.stats()}", flush=True)
        del window, model_window
        gc.collect()
        current = test_end

    if month_cache.max_months > 0:
        print(f"[train:cache] final {month_cache.stats()}", flush=True)
    if cache_dir is not None:
        print(f"[train:cache] window cache hits={cache_hits} misses={cache_misses}", flush=True)
    if daily_ic_cache is not None and cache_dir is not None:
        daily_ic_cache.save()
        print("[train:factor-ic-cache] persisted=true", flush=True)
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    returns, holdings = backtest.portfolio_from_predictions(
        pred,
        horizon=horizon,
        top_n=top_n,
        max_weight=max_weight if max_weight is not None else 1.0 / max(int(top_n), 1),
        positive_only=positive_only,
        ridge_quantile=ridge_quantile,
    )
    summary = backtest.evaluate_returns(returns["ret"] if not returns.empty else pd.Series(dtype=float), periods_per_year=max(1, 252 // horizon))
    if not returns.empty:
        summary["avg_turnover"] = float(returns["turnover"].mean())
        summary["avg_holdings"] = float(returns["n_holdings"].mean())
    summary["ridge_quantile"] = ridge_quantile
    summary["lgbm_weight"] = lgbm_weight
    summary["ic_weight"] = ic_weight
    summary["rank_vote_weight"] = rank_vote_weight
    summary["skip_windows"] = skip_windows
    summary["max_windows"] = max_windows
    summary["n_factors"] = len(factors)
    summary["avg_selected_factors"] = float(np.mean(selected_counts)) if selected_counts else float(len(factors))
    summary["min_selected_factors"] = int(min(selected_counts)) if selected_counts else int(len(factors))
    summary["max_selected_factors"] = int(max(selected_counts)) if selected_counts else int(len(factors))
    summary["rolling_factor_select"] = bool(rolling_factor_select)
    summary["purge_horizon"] = bool(purge_horizon)
    summary["recent_windows"] = int(recent_windows)
    summary["expanding_train"] = bool(expanding_train)
    summary["elastic_net"] = bool(elastic_net)
    summary["elastic_alpha"] = float(elastic_alpha) if elastic_net else None
    summary["elastic_l1_ratio"] = float(elastic_l1_ratio) if elastic_net else None
    summary["catboost_ranker"] = bool(catboost_ranker)
    summary["catboost_estimators"] = int(catboost_estimators) if catboost_ranker else None
    summary["catboost_learning_rate"] = float(catboost_learning_rate) if catboost_ranker else None
    summary["catboost_max_train_rows"] = int(catboost_max_train_rows) if catboost_ranker else None
    summary["extra_trees"] = bool(extra_trees)
    summary["extra_trees_estimators"] = int(extra_trees_estimators) if extra_trees else None
    summary["extra_trees_max_train_rows"] = int(extra_trees_max_train_rows) if extra_trees else None
    summary["n_predictions"] = len(pred)
    summary_df = pd.DataFrame([{"model": MODEL_NAME, **summary}])

    name_prefix = f"{output_prefix}_" if output_prefix else ""
    if audit_parts:
        warehouse.save(f"{name_prefix}factor_audit", pd.concat(audit_parts, ignore_index=True))
    warehouse.save(f"{name_prefix}bt_{MODEL_NAME}_predictions", pred)
    warehouse.save(f"{name_prefix}bt_{MODEL_NAME}_returns", returns)
    warehouse.save(f"{name_prefix}bt_{MODEL_NAME}_holdings", holdings)
    warehouse.save(f"{name_prefix}bt_{MODEL_NAME}_summary", summary_df)
    print(summary_df.to_string(index=False), flush=True)
    return summary_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory-conscious full-A batched training")
    ap.add_argument("--name", default="factor_panel_full_a_cont")
    ap.add_argument("--selection", default="factor_selection_lh1000_cont")
    ap.add_argument("--output-prefix", default="full_a_batched_rq04_default_ne200")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("PANEL_BATCH_SIZE", "200") or 200),
        help="panel stock batch size; larger batches reduce repeated factor setup",
    )
    ap.add_argument("--min-price-rows", type=int, default=1000)
    ap.add_argument("--limit-codes", type=int, default=0, help="debug only; 0 means all eligible codes")
    ap.add_argument("--universe-file", default="", help="optional six-digit code list restricting panel construction")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--refresh-months", type=int, default=0, help="rebuild only the latest N monthly raw/prepared partitions; 0 reuses existing panel unless --rebuild")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--validation-months", type=int, default=1)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--ridge-quantile", type=float, default=0.4)
    ap.add_argument("--lgbm-weight", type=float, default=1.0)
    ap.add_argument("--ic-weight", type=float, default=0.0, help="add an IC-weighted naive factor score to the Ridge+LightGBM z-score ensemble")
    ap.add_argument("--rank-vote-weight", type=float, default=0.0, help="add a cross-sectional rank-vote naive factor score to the ensemble")
    ap.add_argument("--max-weight", type=float, default=0.1)
    ap.add_argument("--positive-only", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--model-threads", type=int, default=0,
                    help="LightGBM worker threads; 0 keeps library auto-detection")
    ap.add_argument("--decay-half-life-days", type=float, default=90.0)
    ap.add_argument("--min-weight", type=float, default=0.05)
    ap.add_argument("--max-windows", type=int, default=0, help="debug only")
    ap.add_argument("--skip-windows", type=int, default=0, help="skip the first N walk-forward windows before training")
    ap.add_argument("--recent-windows", type=int, default=0,
                    help="automatically train only the latest N walk-forward windows; overrides skip/max windows")
    ap.add_argument("--expanding-train", action="store_true",
                    help="use all prepared history up to each train cutoff instead of a rolling train-month window")
    ap.add_argument("--month-cache-size", type=int, default=30, help="prepared monthly parquet files to keep in memory; 0 disables cache")
    ap.add_argument("--rolling-factor-select", action="store_true", help="select factors independently inside each purged training window")
    ap.add_argument("--rolling-top-factors", type=int, default=30)
    ap.add_argument("--max-factor-ic-corr", type=float, default=0.85)
    ap.add_argument("--purge-horizon", action="store_true", help="remove the final horizon signal dates at train/validation boundaries")
    ap.add_argument("--elastic-net", action="store_true", help="train an ElasticNet shadow leg without adding it to the live score")
    ap.add_argument("--elastic-alpha", type=float, default=0.001)
    ap.add_argument("--elastic-l1-ratio", type=float, default=0.5)
    ap.add_argument("--catboost-ranker", action="store_true", help="train a CatBoost ranking shadow leg")
    ap.add_argument("--catboost-estimators", type=int, default=200)
    ap.add_argument("--catboost-learning-rate", type=float, default=0.03)
    ap.add_argument("--catboost-max-train-rows", type=int, default=300000,
                    help="maximum recent complete query-group rows used by each CatBoost window; 0 disables")
    ap.add_argument("--extra-trees", action="store_true", help="train an ExtraTrees shadow leg")
    ap.add_argument("--extra-trees-estimators", type=int, default=120)
    ap.add_argument("--extra-trees-max-train-rows", type=int, default=300000)
    ap.add_argument("--window-cache-dir", default=None,
                    help="cache per-window predictions keyed by recipe + month-file signature; "
                         "unchanged historical windows are reused so only the refreshed tail is recomputed")
    ap.add_argument("--train-target-mode", default="baseline",
                    choices=["baseline", "buyin-mask", "tradable-label"],
                    help="A/B 训练目标口径：baseline=现役(target_ret 标签、全样本)；"
                         "buyin-mask=剔训练段封涨停买入日；tradable-label=用跌停顺延可实现收益作标签")
    args = ap.parse_args()

    config.ensure_dirs()
    build_monthly_panel(
        args.name,
        horizon=args.horizon,
        batch_size=args.batch_size,
        min_price_rows=args.min_price_rows,
        rebuild=args.rebuild,
        limit_codes=args.limit_codes,
        refresh_months=args.refresh_months,
        universe_file=args.universe_file or None,
    )
    if args.build_only:
        return
    decay = args.decay_half_life_days if args.decay_half_life_days > 0 else None
    train_batched(
        name=args.name,
        output_prefix=args.output_prefix,
        selection_name=args.selection,
        horizon=args.horizon,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        top_n=args.top_n,
        ridge_quantile=args.ridge_quantile,
        lgbm_weight=args.lgbm_weight,
        ic_weight=args.ic_weight,
        rank_vote_weight=args.rank_vote_weight,
        max_weight=args.max_weight,
        positive_only=args.positive_only,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        decay_half_life_days=decay,
        min_weight=args.min_weight,
        model_threads=args.model_threads,
        max_windows=args.max_windows,
        skip_windows=args.skip_windows,
        month_cache_size=args.month_cache_size,
        rolling_factor_select=args.rolling_factor_select,
        rolling_top_factors=args.rolling_top_factors,
        max_factor_ic_corr=args.max_factor_ic_corr,
        purge_horizon=args.purge_horizon,
        recent_windows=args.recent_windows,
        expanding_train=args.expanding_train,
        elastic_net=args.elastic_net,
        elastic_alpha=args.elastic_alpha,
        elastic_l1_ratio=args.elastic_l1_ratio,
        catboost_ranker=args.catboost_ranker,
        catboost_estimators=args.catboost_estimators,
        catboost_learning_rate=args.catboost_learning_rate,
        catboost_max_train_rows=args.catboost_max_train_rows,
        extra_trees=args.extra_trees,
        extra_trees_estimators=args.extra_trees_estimators,
        extra_trees_max_train_rows=args.extra_trees_max_train_rows,
        window_cache_dir=args.window_cache_dir,
        universe_file=args.universe_file or None,
        train_target_mode=args.train_target_mode,
    )


if __name__ == "__main__":
    main()
