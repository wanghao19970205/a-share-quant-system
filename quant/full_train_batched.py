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


MODEL_NAME = "ridge_lightgbm_ranker_ensemble"
# Bump when the per-window prediction computation changes in a way that invalidates
# previously cached window outputs (see _recipe_signature / window_cache_dir).
WINDOW_CACHE_VERSION = 2


def _canonical_sha256(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pit_window_universe(
    index_code: str,
    anchor_date: pd.Timestamp,
    security_master_path: str | Path | None = None,
    index_history_path: str | Path | None = None,
) -> dict:
    """Resolve a fixed index universe at a training-window start date."""
    master_path = Path(security_master_path or config.SECURITY_MASTER_FILE)
    history_path = Path(index_history_path or config.INDEX_CONSTITUENT_HISTORY_FILE)
    if not master_path.is_file() or not history_path.is_file():
        raise RuntimeError("PIT universe reference files are unavailable")
    master = pd.read_parquet(master_path)
    history = pd.read_parquet(history_path)
    master_required = {"code", "list_date", "delist_date"}
    history_required = {
        "index_code", "code", "is_standard_a_share", "in_date", "out_date",
    }
    if not master_required.issubset(master.columns):
        raise ValueError(f"security master missing columns: {sorted(master_required - set(master.columns))}")
    if not history_required.issubset(history.columns):
        raise ValueError(f"index history missing columns: {sorted(history_required - set(history.columns))}")

    anchor = pd.Timestamp(anchor_date).normalize()
    if pd.isna(anchor):
        raise ValueError("PIT universe anchor date is invalid")
    master = master.copy()
    history = history[history["index_code"].astype(str) == str(index_code)].copy()
    if history.empty:
        raise ValueError(f"index history has no rows for {index_code}")
    master["code"] = master["code"].astype(str).str.zfill(6)
    history["code"] = history["code"].astype("string").str.zfill(6)
    for frame, required_date, optional_date in (
        (master, "list_date", "delist_date"),
        (history, "in_date", "out_date"),
    ):
        frame[required_date] = pd.to_datetime(frame[required_date], errors="coerce")
        frame[optional_date] = pd.to_datetime(frame[optional_date], errors="coerce")
        if frame[required_date].isna().any():
            raise ValueError(f"PIT universe has invalid {required_date}")
        reversed_interval = frame[optional_date].notna() & (
            frame[optional_date] < frame[required_date]
        )
        if reversed_interval.any():
            raise ValueError(f"PIT universe has reversed {required_date}/{optional_date} interval")
    if master["code"].duplicated().any() or not master["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("security master has invalid or duplicate codes")

    standard = history["is_standard_a_share"].eq(True)
    valid_code = history["code"].str.fullmatch(r"\d{6}", na=False)
    active_member = (
        (history["in_date"] <= anchor)
        & (history["out_date"].isna() | (anchor < history["out_date"]))
    )
    member_codes = set(history.loc[standard & valid_code & active_member, "code"].astype(str))
    listed = (
        (master["list_date"] <= anchor)
        & (master["delist_date"].isna() | (anchor < master["delist_date"]))
    )
    listed_codes = set(master.loc[listed, "code"].astype(str))
    codes = sorted(member_codes & listed_codes)
    if not codes:
        raise RuntimeError(f"PIT universe is empty for {index_code} at {anchor.date()}")

    manifest = {
        "policy": "index_constituent_and_listing_at_train_start_v1",
        "index_code": str(index_code),
        "anchor_date": str(anchor.date()),
        "membership_interval": "[in_date,out_date)",
        "listing_interval": "[list_date,delist_date)",
        "resolved_codes": codes,
        "resolved_code_count": len(codes),
        "universe_hash": _canonical_sha256(codes),
        "sources": {
            "security_master": {
                "filename": master_path.name,
                "sha256": _file_sha256(master_path),
            },
            "index_constituent_history": {
                "filename": history_path.name,
                "sha256": _file_sha256(history_path),
            },
        },
    }
    manifest["manifest_hash"] = _canonical_sha256(manifest)
    return manifest


def _resolve_effective_window_universe(
    pit_index_code: str,
    train_start: pd.Timestamp,
    static_codes: set[str] | None = None,
) -> tuple[dict, set[str]]:
    manifest = resolve_pit_window_universe(pit_index_code, train_start)
    pit_codes = set(manifest["resolved_codes"])
    effective = pit_codes if static_codes is None else pit_codes & static_codes
    if not effective:
        raise RuntimeError(
            f"effective PIT universe is empty for {pit_index_code} at {pd.Timestamp(train_start).date()}"
        )
    return manifest, effective


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


def _price_source_signature(
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> str:
    """Hash price files that can contribute rows to the requested window.

    Without bounds this preserves the legacy whole-directory identity. With bounds,
    only files whose observed dates intersect ``[start, end)`` are included, so a
    daily rewrite of the latest price file does not invalidate historical windows.
    Read failures are included conservatively to avoid unsafe cache reuse.
    """
    price_dir = Path(config.QUANT_DIR) / "price"
    payload = []
    bounded = start is not None and end is not None
    start_ts = pd.Timestamp(start) if bounded else None
    end_ts = pd.Timestamp(end) if bounded else None
    if price_dir.is_dir():
        for path in sorted(price_dir.iterdir()):
            if path.suffix != ".parquet":
                continue
            stat = path.stat()
            entry = [path.name, int(stat.st_size), int(stat.st_mtime_ns)]
            if bounded:
                try:
                    dates = pd.to_datetime(
                        pd.read_parquet(path, columns=["date"])["date"],
                        errors="coerce",
                    ).dropna()
                    if dates.empty or not ((dates >= start_ts) & (dates < end_ts)).any():
                        continue
                except Exception:  # noqa: BLE001
                    entry.append("unreadable")
            payload.append(entry)
    return hashlib.sha1(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trading_status_source_signature() -> str:
    """Content identity for the status artifact required by strict labels."""
    path = Path(config.QUANT_DIR) / "trading_status_history.parquet"
    return _file_sha256(path) if path.is_file() else "missing"


def _legacy_window_cache_allowed(
    strict_execution_labels: bool,
    pit_index_code: str,
) -> bool:
    return not strict_execution_labels and not bool(pit_index_code)


def _rolling_cache_allowed(
    require_selection_provenance: bool,
    rolling_factor_select: bool,
) -> bool:
    return not (require_selection_provenance and rolling_factor_select)


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


def _train_lightgbm_window(ridge_only: bool, *args, **kwargs):
    if ridge_only:
        return None
    return qmodel.train_lightgbm_ranker(*args, **kwargs)


def _primary_window_predictions(
    ridge_res,
    lgbm_res,
    current: pd.Timestamp,
    test_end: pd.Timestamp,
    ridge_only: bool,
) -> pd.DataFrame:
    if not ridge_res.ok or ridge_res.predictions.empty:
        return pd.DataFrame()
    ridge = ridge_res.predictions[
        (ridge_res.predictions["date"] >= current)
        & (ridge_res.predictions["date"] < test_end)
    ].copy()
    if ridge.empty:
        return pd.DataFrame()
    if ridge_only:
        out = ridge.rename(columns={"pred": "ridge_pred"})
        out["lgbm_pred"] = np.nan
        return out
    if lgbm_res is None or not lgbm_res.ok or lgbm_res.predictions.empty:
        return pd.DataFrame()
    lgbm = lgbm_res.predictions[
        (lgbm_res.predictions["date"] >= current)
        & (lgbm_res.predictions["date"] < test_end)
    ].copy()
    if lgbm.empty:
        return pd.DataFrame()
    return lgbm.rename(columns={"pred": "lgbm_pred"}).merge(
        ridge[["code", "date", "pred"]].rename(columns={"pred": "ridge_pred"}),
        on=["code", "date"],
        how="inner",
    )


def _panel_dirs(name: str) -> tuple[Path, Path, Path]:
    root = Path(config.QUANT_DIR) / f"{name}_parts"
    return root, root / "raw_monthly", root / "prepared_monthly"


def _month_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m")


def _prepared_files(prepared_dir: Path) -> list[Path]:
    return sorted(prepared_dir.glob("*.parquet"))


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


def _purge_span(train_target_mode: str, horizon: int) -> int:
    if train_target_mode == "tradable-label":
        return int(horizon) + backtest.bt_sell_roll_max_days() - 1
    if train_target_mode in {"open-label", "open-buyin-mask"}:
        return int(horizon) + 1
    return int(horizon)


def _validate_execution_gate_params(
    train_target_mode: str,
    strict_execution_labels: bool,
    min_adv20: float | None,
) -> None:
    if strict_execution_labels and train_target_mode == "baseline":
        raise ValueError("strict execution labels require a non-baseline target mode")
    if min_adv20 is not None and not strict_execution_labels:
        raise ValueError("min_adv20 requires strict execution labels and authoritative calendar")


def _label_recipe_params(
    train_target_mode: str,
    strict_execution_labels: bool = False,
    enforce_c30_gates: bool = False,
    min_adv20: float | None = None,
    min_listing_sessions: int | None = None,
    price_source_signature: str | None = None,
) -> dict:
    if train_target_mode == "baseline" and not enforce_c30_gates:
        return {}
    params = {
        "train_target_mode": train_target_mode,
        "price_source_signature": price_source_signature or _price_source_signature(),
    }
    if train_target_mode != "baseline":
        params["sell_roll_max_days"] = backtest.bt_sell_roll_max_days()
    if strict_execution_labels:
        params["strict_execution_labels"] = True
        params["trading_status_source_signature"] = _trading_status_source_signature()
    if enforce_c30_gates:
        params["c30_gates"] = True
    if min_adv20 is not None:
        params["min_adv20"] = float(min_adv20)
    if min_listing_sessions is not None:
        params["min_listing_sessions"] = int(min_listing_sessions)
    return params


def _validate_c30_refresh_months(enforce_c30_gates: bool, refresh_months: int) -> None:
    if enforce_c30_gates and int(refresh_months) == 1:
        raise ValueError("C30 strict mode requires refresh_months=0 or refresh_months>=2")


def _c30_recipe_audit(
    train_target_mode: str,
    refresh_months: int,
    enforce_c30_gates: bool = False,
) -> dict:
    params = _label_recipe_params(
        train_target_mode, enforce_c30_gates=enforce_c30_gates,
    )
    return {
        "train_target_mode": str(train_target_mode),
        "refresh_months": int(refresh_months),
        "price_source_signature": _price_source_signature(),
        "c30_gates_enforced": bool(enforce_c30_gates),
        "baseline_price_signature_protected": (
            train_target_mode != "baseline" or "price_source_signature" in params
        ),
        "refresh_months_horizon_risk": int(refresh_months) == 1,
    }


def _purged_end(dates: pd.Series, boundary: pd.Timestamp, horizon: int) -> pd.Timestamp | None:
    """Return the last signal date whose forward label ends before boundary."""
    eligible = pd.Series(pd.to_datetime(dates, errors="coerce").dropna().unique())
    eligible = eligible[eligible < pd.Timestamp(boundary)].sort_values().reset_index(drop=True)
    if len(eligible) <= int(horizon):
        return None
    return pd.Timestamp(eligible.iloc[-int(horizon) - 1])


def _purged_end_by_calendar(
    boundary: pd.Timestamp,
    horizon: int,
) -> pd.Timestamp | None:
    calendar = warehouse.load("trading_calendar")
    if calendar.empty or list(calendar.columns) != ["date"]:
        raise RuntimeError("authoritative trading calendar unavailable or malformed")
    dates = pd.to_datetime(calendar["date"], errors="coerce").astype("datetime64[ns]")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("authoritative trading calendar must be unique and increasing")
    eligible = dates[dates < pd.Timestamp(boundary)]
    span = int(horizon)
    if span < 0:
        raise ValueError("purge horizon must be non-negative")
    if len(eligible) <= span:
        return None
    return pd.Timestamp(eligible.iloc[-span - 1])


def _purged_end_by_code(
    frame: pd.DataFrame,
    boundary: pd.Timestamp,
    horizon: int,
) -> pd.Timestamp | None:
    """Use the latest safe per-stock row, then choose the globally safest boundary."""
    required = {"code", "date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"purge frame missing {sorted(required - set(frame.columns))}")
    ends = []
    for _, group in frame.groupby("code", sort=False):
        safe = _purged_end(group["date"], boundary, horizon)
        if safe is None:
            return None
        ends.append(safe)
    return min(ends) if ends else None


class DailyICCache:
    """Reuse date-local factor IC values across windows and process restarts."""

    def __init__(
        self,
        workers: int = 8,
        cache_dir: Path | None = None,
        recipe_signature: str = "legacy-v1",
    ):
        self.workers = max(int(workers), 1)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.recipe_signature = str(recipe_signature)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frames: dict[tuple[str, int, str, tuple[str, ...]], pd.DataFrame] = {}

    def set_recipe_signature(self, recipe_signature: str) -> None:
        recipe_signature = str(recipe_signature)
        if recipe_signature != self.recipe_signature:
            self.recipe_signature = recipe_signature
            self._frames.clear()

    def _path(self, key: tuple[str, int, str, tuple[str, ...]]) -> Path | None:
        recipe_signature, horizon, target, factors = key
        payload = "|".join([recipe_signature, str(horizon), target, *factors]).encode("utf-8")
        return (
            self.cache_dir / f"{hashlib.sha256(payload).hexdigest()}.parquet"
            if self.cache_dir is not None else None
        )

    def get(
        self,
        train: pd.DataFrame,
        factors: list[str],
        horizon: int,
        label_col: str | None = None,
    ) -> tuple[pd.DataFrame, int, int]:
        use = tuple(factors)
        target = label_col or f"target_ret_{horizon}d"
        key = (self.recipe_signature, int(horizon), target, use)
        desired_dates = pd.Index(pd.to_datetime(train["date"], errors="coerce").dropna().unique())
        cached = self._frames.get(key)
        if cached is None:
            path = self._path(key)
            if path is not None and path.exists():
                try:
                    cached = pd.read_parquet(path)
                    cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
                    print(f"[train:factor-ic-cache] disk_hit={path.name}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[train:factor-ic-cache] disk cache ignored: {type(exc).__name__}: {exc}", flush=True)
            if cached is None:
                cached = pd.DataFrame(columns=["date", "factor", "ic"])
            self._frames[key] = cached
        cached_dates = pd.Index(pd.to_datetime(cached["date"], errors="coerce").dropna().unique())
        missing_dates = desired_dates.difference(cached_dates)
        if len(missing_dates):
            missing = train[pd.to_datetime(train["date"], errors="coerce").isin(missing_dates)]
            calculated = factor_select.daily_ic(
                missing,
                list(use),
                horizon=horizon,
                workers=self.workers,
                label_col=label_col,
            )
            if not calculated.empty:
                cached = (
                    calculated.copy()
                    if cached.empty
                    else pd.concat([cached, calculated], ignore_index=True)
                )
                cached = cached.drop_duplicates(["date", "factor"], keep="last")
                self._frames[key] = cached
                path = self._path(key)
                if path is not None:
                    tmp = path.with_suffix(".tmp.parquet")
                    cached.to_parquet(tmp, index=False)
                    tmp.replace(path)
        result = cached[pd.to_datetime(cached["date"], errors="coerce").isin(desired_dates)].copy()
        return result, int(len(desired_dates) - len(missing_dates)), int(len(missing_dates))


def _rolling_factor_selection(
    train: pd.DataFrame,
    factors: list[str],
    horizon: int,
    top_n: int = 30,
    preselect_multiplier: int = 2,
    max_ic_corr: float = 0.85,
    daily_ic_cache: DailyICCache | None = None,
    label_col: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select stable factors using only the purged training slice and remove IC-series clones."""
    use = [f for f in factors if f in train.columns]
    if not use or train.empty:
        return [], pd.DataFrame()
    if daily_ic_cache is None:
        ic = factor_select.daily_ic(
            train, use, horizon=horizon, workers=8, label_col=label_col
        )
    else:
        ic, cache_hits, cache_misses = daily_ic_cache.get(
            train, use, horizon, label_col=label_col
        )
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


def _rolling_selection_manifest_row(
    window: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    label_col: str,
    candidate_factors: list[str],
    selected_factors: list[str],
    purge_span: int,
) -> dict:
    candidates = list(map(str, candidate_factors))
    selected = list(map(str, selected_factors))
    row = {
        "window": int(window),
        "train_start": str(pd.Timestamp(train_start).date()),
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_start": str(pd.Timestamp(valid_start).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "test_end": str(pd.Timestamp(test_end).date()),
        "label_col": str(label_col),
        "purge_span": int(purge_span),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_pool_sha256": hashlib.sha256(
            "\n".join(sorted(candidates)).encode("utf-8")
        ).hexdigest(),
        "selected_sha256": hashlib.sha256(
            "\n".join(selected).encode("utf-8")
        ).hexdigest(),
        "generator_code_sha256": _file_sha256(Path(__file__)),
    }
    row["manifest_hash"] = _canonical_sha256(row)
    return row


def build_monthly_panel(name: str, horizon: int, batch_size: int,
                        min_price_rows: int, rebuild: bool = False,
                        limit_codes: int = 0, refresh_months: int = 0,
                        universe_file: str | None = None,
                        include_trading_gap_risk: bool = False,
                        strict_calendar_factors: bool = False,
                        strict_announcement_lag: bool = False,
                        strict_pit_min_price_rows: bool = False) -> Path:
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
    price_eligibility: dict[str, pd.Timestamp] = {}
    if strict_pit_min_price_rows:
        price_eligibility = engineering._price_row_eligibility_dates(  # noqa: SLF001
            codes, min_price_rows,
        )
        codes = sorted(price_eligibility)
    else:
        codes = engineering._filter_codes_by_price_rows(codes, min_price_rows)  # noqa: SLF001
    if limit_codes:
        codes = codes[:limit_codes]

    existing_prepared = _prepared_files(prepared_dir)
    panel_recipe = {
        "include_trading_gap_risk": bool(include_trading_gap_risk),
        "strict_calendar_factors": bool(strict_calendar_factors),
        "strict_announcement_lag": bool(strict_announcement_lag),
        "strict_pit_min_price_rows": bool(strict_pit_min_price_rows),
        "min_price_rows": int(min_price_rows) if strict_pit_min_price_rows else None,
    }
    panel_recipe_path = root / "panel_recipe.json"
    strict_panel = any(panel_recipe.values())
    if strict_panel and existing_prepared and not rebuild:
        existing_recipe = (
            json.loads(panel_recipe_path.read_text(encoding="utf-8"))
            if panel_recipe_path.exists()
            else None
        )
        if existing_recipe != panel_recipe:
            raise RuntimeError(
                "strict panel recipe does not match cached monthly panel; use --rebuild"
            )
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
        "financial_yjbb": engineering._asof_report_factor(  # noqa: SLF001
            "financial_yjbb", "yjbb", all_codes,
            strict_announcement_lag=strict_announcement_lag,
        ),
        "income": engineering._asof_report_factor(  # noqa: SLF001
            "income", "income", all_codes,
            strict_announcement_lag=strict_announcement_lag,
        ),
        "cashflow": engineering._asof_report_factor(  # noqa: SLF001
            "cashflow", "cashflow", all_codes,
            strict_announcement_lag=strict_announcement_lag,
        ),
        "balance": engineering._asof_report_factor(  # noqa: SLF001
            "balance", "balance", all_codes,
            strict_announcement_lag=strict_announcement_lag,
        ),
        "performance_forecast": engineering._forecast_events(  # noqa: SLF001
            all_codes, strict_announcement_lag=strict_announcement_lag,
        ),
        "margin_underlying_szse": engineering._margin_underlying(all_codes),  # noqa: SLF001
        "block_trades": engineering._event_counts("block_trades", all_codes, "block_trade"),  # noqa: SLF001
        "lhb": engineering._event_counts("lhb", all_codes, "lhb"),  # noqa: SLF001
    }
    print(
        f"[panel] shared factor cache seconds={time.perf_counter() - shared_started:.2f}",
        flush=True,
    )

    for bi, batch in _chunks(codes, batch_size):
        raw = engineering.build_panel(
            codes=batch,
            horizon=horizon,
            min_price_rows=0,
            output_start=refresh_start,
            warmup_rows=260,
            shared_factors=shared_factors,
            include_trading_gap_risk=include_trading_gap_risk,
            strict_calendar_factors=strict_calendar_factors,
            strict_announcement_lag=strict_announcement_lag,
        )
        if raw.empty:
            print(f"[panel:raw] batch={bi} empty", flush=True)
            continue
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])
        if strict_pit_min_price_rows:
            eligibility = raw["code"].astype(str).map(price_eligibility)
            raw = raw[eligibility.notna() & (raw["date"] >= eligibility)].copy()
        for month, part in raw.groupby(_month_key(raw["date"]), sort=True):
            if not month or month == "NaT":
                continue
            if refresh_keys and str(month) not in refresh_keys:
                continue
            month_dir = raw_dir / str(month)
            month_dir.mkdir(parents=True, exist_ok=True)
            part.to_parquet(month_dir / f"part_{bi:05d}.parquet", index=False)
        print(f"[panel:raw] batch={bi} codes={len(batch)} rows={len(raw)}", flush=True)
        del raw
        gc.collect()

    month_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if refresh_keys:
        month_dirs = [p for p in month_dirs if p.name in refresh_keys]

    def prepare_month(month_dir: Path) -> tuple[str, int, int, pd.DataFrame]:
        out_path = prepared_dir / f"{month_dir.name}.parquet"
        part_files = sorted(month_dir.glob("*.parquet"))
        if not part_files:
            return month_dir.name, 0, 0, pd.DataFrame()
        raw = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
        prepared, feats = engineering.prepare_features(
            raw,
            horizon=horizon,
            add_discrete=False,
            add_onehot=False,
            drop_target_na=False,
        )
        prepared.to_parquet(out_path, index=False)
        summary = engineering.summarize_features(feats)
        rows = len(prepared)
        cols = len(prepared.columns)
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

    if strict_panel:
        tmp_recipe_path = panel_recipe_path.with_suffix(".tmp")
        tmp_recipe_path.write_text(
            json.dumps(panel_recipe, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_recipe_path, panel_recipe_path)
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
_TRAD_CACHE: "dict[tuple[str, int, bool, float | None, int | None, str | None], pd.DataFrame]" = {}


def _cached_price_tradability(
    codes: list[str],
    horizon: int,
    strict_execution_labels: bool = False,
    min_adv20: float | None = None,
    min_listing_sessions: int | None = None,
    expected_status_signature: str | None = None,
) -> pd.DataFrame:
    """返回 codes 的可交易口径，仅对未缓存的 code 调用 price_tradability。

    与直接 tradability.price_tradability(codes, [horizon]) 等价（同一批 code 的并集），
    差别仅是把 per-code 结果记忆化，跨窗口复用。空结果的 code 也缓存（存空帧），
    避免对不存在 price 文件的 code 反复触发磁盘 stat。"""
    status_signature = (
        _trading_status_source_signature() if strict_execution_labels else None
    )
    if (
        strict_execution_labels
        and expected_status_signature is not None
        and status_signature != expected_status_signature
    ):
        raise RuntimeError("trading status artifact changed after recipe signature capture")
    if expected_status_signature is not None:
        status_signature = expected_status_signature

    def cache_key(code: str) -> tuple:
        return (
            code, horizon, strict_execution_labels, min_adv20,
            min_listing_sessions, status_signature,
        )

    def ensure_status_unchanged() -> None:
        if (
            strict_execution_labels
            and _trading_status_source_signature() != status_signature
        ):
            raise RuntimeError("trading status artifact changed during label loading")

    missing = [c for c in codes if cache_key(c) not in _TRAD_CACHE]
    if missing:
        fresh = tradability.price_tradability(
            missing,
            [horizon],
            require_status=strict_execution_labels,
            require_calendar=strict_execution_labels,
            min_adv20=min_adv20,
            min_listing_sessions=min_listing_sessions,
        )
        ensure_status_unchanged()
        if not fresh.empty:
            for code, grp in fresh.groupby("code", sort=False):
                _TRAD_CACHE[cache_key(code)] = grp.reset_index(drop=True)

        # 没产出行的 code 也落一个空帧占位，防下窗重复读盘。
        for code in missing:
            _TRAD_CACHE.setdefault(cache_key(code), pd.DataFrame())
    ensure_status_unchanged()
    parts = [
        _TRAD_CACHE[cache_key(c)] for c in codes
        if not _TRAD_CACHE[cache_key(c)].empty
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _join_tradability(
    df: pd.DataFrame,
    horizon: int,
    strict_execution_labels: bool = False,
    min_adv20: float | None = None,
    min_listing_sessions: int | None = None,
    expected_status_signature: str | None = None,
) -> pd.DataFrame:

    """为窗口 join 可交易口径列（tradable_ret_{h}d / buyable_close），供 A/B 变体使用。

    仅新增标签/掩码列，不新增特征列（buyable_close 只用当日 OHLC、因果安全；
    tradable_ret 含 shift(-h) 未来价，是标签本就该含未来，非泄漏）。
    join 键：code(str, zfill6) + date(datetime64[ns])，与 tradability.price_tradability 输出一致。
    join 后断言 tradable_ret 命中率，防 code/date 规范化不一致导致静默丢样本。"""
    if df.empty:
        return df
    codes = sorted(df["code"].astype(str).str.zfill(6).unique())
    trad = _cached_price_tradability(
        codes, horizon, strict_execution_labels=strict_execution_labels,
        min_adv20=min_adv20,
        min_listing_sessions=min_listing_sessions,
        expected_status_signature=expected_status_signature,
    )
    keep = [
        "code",
        "date",
        f"tradable_ret_{horizon}d",
        f"open_ret_{horizon}d",
        "buyable_close",
        "buyable_next",
    ]
    keep = [c for c in keep if c in trad.columns]
    if trad.empty or len(keep) <= 2:
        raise RuntimeError(
            f"tradability join produced no usable columns for horizon={horizon}; "
            f"cannot run A/B variant without buyable_close/tradable_ret"
        )
    out = df.copy()
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
            message = (
                f"tradable_ret NaN rate={na_rate:.3f} (>0.20) — 检查 code/date join 键"
            )
            if strict_execution_labels:
                raise RuntimeError(message)
            print(f"[train:ab] WARN {message}", flush=True)
    return out


def _reject_unsafe_factors(factors: list[str], context: str) -> None:
    unsafe = sorted({factor for factor in factors if engineering.is_forbidden_feature(factor)})
    if unsafe:
        raise RuntimeError(f"unsafe future-label factors in {context}: {unsafe}")


def _selected_factors(
    selection_name: str,
    prepared_dir: Path,
    horizon: int,
    require_selection_provenance: bool = False,
    rolling_factor_select: bool = False,
) -> list[str]:
    sel = warehouse.load(selection_name) if not rolling_factor_select else pd.DataFrame()
    selected = sel["factor"].astype(str).tolist() if not sel.empty and "factor" in sel.columns else []
    if require_selection_provenance and not rolling_factor_select and not selected:
        raise RuntimeError(f"selection {selection_name!r} is empty or missing factor column")
    _reject_unsafe_factors(selected, "selected factor manifest")
    files = _prepared_files(prepared_dir)
    if not files:
        return []
    sample = pd.read_parquet(files[0])
    available = set(engineering.feature_columns(sample, horizon))
    factors = [f for f in selected if f in available]
    if not factors:
        factors = [f for f in engineering.feature_columns(sample, horizon) if f in available]
    _reject_unsafe_factors(factors, "selected factor manifest")
    return factors


def _validate_selection_provenance(
    selection_name: str,
    selected: list[str],
    expected_label_col: str,
) -> str:
    manifest = warehouse.load(f"{selection_name}_manifest")
    required = {
        "selection_name", "label_col", "train_end", "valid_end", "predict_start",
        "candidate_count", "selected_count", "candidate_pool_sha256",
        "selected_sha256", "generator_code_sha256",
    }
    missing = sorted(required - set(manifest.columns))
    if manifest.empty or missing:
        raise RuntimeError(
            f"selection provenance unavailable for {selection_name!r}: "
            f"missing={missing or ['manifest row']}"
        )
    if len(manifest) != 1:
        raise RuntimeError(f"selection provenance must have one row: {selection_name!r}")
    row = manifest.iloc[0]
    if str(row["selection_name"]) != selection_name:
        raise RuntimeError("selection provenance name mismatch")
    if str(row["label_col"]) != expected_label_col:
        raise RuntimeError(
            f"selection label mismatch: expected={expected_label_col!r} "
            f"actual={row['label_col']!r}"
        )
    if int(row["selected_count"]) != len(selected):
        raise RuntimeError("selection provenance selected_count mismatch")
    selected_sha = hashlib.sha256(
        "\n".join(map(str, selected)).encode("utf-8")
    ).hexdigest()
    if str(row["selected_sha256"]) != selected_sha:
        raise RuntimeError("selection provenance selected factor hash mismatch")
    boundaries = [pd.Timestamp(row[col]) for col in ("train_end", "valid_end", "predict_start")]
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise RuntimeError("selection provenance boundaries are not strictly ordered")
    return _canonical_sha256(row.to_dict())


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
                  pit_index_code: str | None = None,
                  train_target_mode: str = "baseline",
                  strict_execution_labels: bool = False,
                  enforce_c30_gates: bool = False,
                  min_adv20: float | None = None,
                  min_listing_sessions: int | None = None,
                  extra_trees_weight: float = 0.0,
                  random_forest: bool = False, random_forest_estimators: int = 120,
                  random_forest_max_train_rows: int = 300_000,
                  random_forest_weight: float = 0.0,
                  rebalance_stride: int = 1,
                  hold_rank_buffer: int = 0,
                  min_window_completion_ratio: float = 0.90,
                  ridge_only: bool = False,
                  require_selection_provenance: bool = False) -> pd.DataFrame:
    _, _, prepared_dir = _panel_dirs(name)
    _validate_execution_gate_params(
        train_target_mode, strict_execution_labels, min_adv20,
    )
    if ridge_only:
        lgbm_weight = 0.0
    if ridge_only and (
        float(ic_weight) != 0.0
        or float(rank_vote_weight) != 0.0
        or elastic_net
        or catboost_ranker
        or extra_trees
        or random_forest
    ):
        raise ValueError(
            "ridge_only requires all non-Ridge model legs and weights to be disabled"
        )
    if train_target_mode not in (
        "baseline",
        "buyin-mask",
        "tradable-label",
        "open-label",
        "open-buyin-mask",
    ):
        raise ValueError(f"unknown train_target_mode: {train_target_mode}")
    # A/B 训练口径：baseline=现役（target_ret 标签、全样本）；
    #   buyin-mask=剔训练段封涨停买入日；tradable-label=用跌停顺延后的可实现收益作标签。
    ab_label_col = (
        f"tradable_ret_{horizon}d"
        if train_target_mode == "tradable-label"
        else f"open_ret_{horizon}d"
        if train_target_mode in {"open-label", "open-buyin-mask"}
        else None
    )
    ab_mask_col = (
        "buyable_close"
        if train_target_mode == "buyin-mask"
        else "buyable_next"
        if train_target_mode == "open-buyin-mask"
        else None
    )
    universe_codes: set[str] | None = None
    if universe_file:
        with open(universe_file, encoding="utf-8") as f:
            universe_codes = {token for token in f.read().split() if len(token) == 6 and token.isdigit()}
        if not universe_codes:
            raise RuntimeError(f"training universe file is empty: {universe_file}")
    factors = _selected_factors(
        selection_name, prepared_dir, horizon,
        require_selection_provenance=require_selection_provenance,
        rolling_factor_select=rolling_factor_select,
    )
    if not factors:
        raise RuntimeError("no usable factors for full-A batched training")
    selection_provenance_signature = None
    if require_selection_provenance and not rolling_factor_select:
        selection_provenance_signature = _validate_selection_provenance(
            selection_name, factors, ab_label_col or f"target_ret_{horizon}d",
        )
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
    rolling_selection_manifest_rows: list[dict] = []
    pit_manifest_rows: list[dict] = []
    selected_counts: list[int] = []
    requested_purge_windows = 0
    skipped_purge_windows = 0
    side_cols = ["volatility_10", "turnover", "rule_score"]
    windows = 0
    month_cache = MonthFrameCache(month_cache_size)
    recipe_sig = None
    legacy_recipe_sig = None
    cache_dir = None
    cache_hits = 0
    cache_misses = 0
    recipe_params = None
    label_recipe_params = _label_recipe_params(
        train_target_mode,
        strict_execution_labels=strict_execution_labels,
        enforce_c30_gates=enforce_c30_gates,
        min_adv20=min_adv20,
        min_listing_sessions=min_listing_sessions,
    )
    ic_recipe_signature = _canonical_sha256(label_recipe_params or {"recipe": "baseline"})
    daily_ic_cache = DailyICCache(
        workers=min(model_threads or 8, 8),
        cache_dir=(Path(window_cache_dir) / "factor_ic") if window_cache_dir else None,
        recipe_signature=ic_recipe_signature,
    )
    expected_status_signature = label_recipe_params.get(
        "trading_status_source_signature"
    )
    if window_cache_dir and _rolling_cache_allowed(
        require_selection_provenance, rolling_factor_select,
    ):
        recipe_params = dict(
            factors=factors, horizon=horizon, train_months=train_months,
            validation_months=validation_months, test_months=test_months,
            decay_half_life_days=decay_half_life_days, min_weight=min_weight,
            n_estimators=n_estimators, learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds, lgbm_weight=lgbm_weight,
            ridge_only=bool(ridge_only), model_threads=model_threads,
            ic_weight=ic_weight, rank_vote_weight=rank_vote_weight,
            elastic_net=elastic_net, elastic_alpha=elastic_alpha, elastic_l1_ratio=elastic_l1_ratio,
            catboost_ranker=catboost_ranker, catboost_estimators=catboost_estimators,
            catboost_learning_rate=catboost_learning_rate,
            catboost_max_train_rows=catboost_max_train_rows, extra_trees=extra_trees,
            extra_trees_estimators=extra_trees_estimators,
            extra_trees_max_train_rows=extra_trees_max_train_rows,
            extra_trees_weight=extra_trees_weight,
            random_forest=random_forest, random_forest_estimators=random_forest_estimators,
            random_forest_max_train_rows=random_forest_max_train_rows,
            random_forest_weight=random_forest_weight,
            purge_horizon=purge_horizon, expanding_train=expanding_train,
            rolling_factor_select=rolling_factor_select, rolling_top_factors=rolling_top_factors,
            max_factor_ic_corr=max_factor_ic_corr,
            universe_codes=sorted(universe_codes) if universe_codes is not None else [],
        )
        if selection_provenance_signature is not None:
            recipe_params["selection_provenance_signature"] = selection_provenance_signature
        # 仅当非 baseline 时才注入标签字段，保证 baseline 腿哈希与现役兼容。
        recipe_params.update(label_recipe_params)
        recipe_sig = _recipe_signature(**recipe_params)
        legacy_recipe_sig = _legacy_recipe_signature(**recipe_params)
        cache_dir = Path(window_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train:cache] window cache enabled dir={cache_dir} recipe={recipe_sig[:12]}", flush=True)

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
        pit_manifest = None
        effective_codes = universe_codes
        window_price_signature = _price_source_signature(train_start, test_end)
        window_recipe_sig = recipe_sig
        if recipe_params is not None:
            window_recipe_params = dict(recipe_params)
            window_recipe_params["price_source_signature"] = window_price_signature
            window_recipe_sig = _recipe_signature(**window_recipe_params)
            daily_ic_cache.set_recipe_signature(
                _canonical_sha256(window_recipe_params)
            )
        if pit_index_code:
            pit_manifest, effective_codes = _resolve_effective_window_universe(
                pit_index_code, train_start, universe_codes,
            )
            if recipe_params is not None:
                window_recipe_params["pit_universe_manifest_hash"] = pit_manifest["manifest_hash"]
                window_recipe_sig = _recipe_signature(**window_recipe_params)
            pit_manifest_rows.append({
                "window": int(windows),
                "train_start": pd.Timestamp(train_start),
                "valid_start": pd.Timestamp(valid_start),
                "test_start": pd.Timestamp(current),
                "test_end": pd.Timestamp(test_end),
                "index_code": pit_manifest["index_code"],
                "anchor_date": pit_manifest["anchor_date"],
                "resolved_code_count": pit_manifest["resolved_code_count"],
                "effective_code_count": len(effective_codes),
                "universe_hash": pit_manifest["universe_hash"],
                "manifest_hash": pit_manifest["manifest_hash"],
                "security_master_sha256": pit_manifest["sources"]["security_master"]["sha256"],
                "index_history_sha256": pit_manifest["sources"]["index_constituent_history"]["sha256"],
            })
        window_cache_key = None
        stage_started = time.perf_counter()
        if cache_dir is not None:
            window_cache_key = _window_cache_key(
                prepared_dir, window_recipe_sig, train_start, valid_start, current, test_end)
            cache_path = cache_dir / f"{window_cache_key}.parquet"
            legacy_cache_path = None
            if (
                _legacy_window_cache_allowed(strict_execution_labels, pit_index_code)
                and legacy_recipe_sig
                and legacy_recipe_sig != recipe_sig
            ):
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
                    if read_path != cache_path:
                        cached.to_parquet(cache_path, index=False)
                        print(
                            f"[train:cache] window={windows} promoted legacy key={read_path.stem[:12]}",
                            flush=True,
                        )
                    if effective_codes is not None:
                        cached = cached[cached["code"].astype(str).isin(effective_codes)].copy()
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
        if effective_codes is not None and not window.empty:
            window = window[window["code"].astype(str).isin(effective_codes)].copy()
        # A/B：非 baseline 时，从 price 文件为本窗口 codes join 可交易列（tradable_ret_{h}d / buyable_close）。
        # prepared 面板本就没有这两列；仅新增标签/掩码列，不引入任何特征列（因果安全）。
        if train_target_mode != "baseline" and not window.empty:
            window = _join_tradability(
                window,
                horizon,
                strict_execution_labels=strict_execution_labels,
                min_adv20=min_adv20,
                min_listing_sessions=min_listing_sessions,
                expected_status_signature=expected_status_signature,
            )
        factors_in_window = [f for f in factors if f in window.columns]
        candidate_factors = list(factors_in_window)
        _reject_unsafe_factors(candidate_factors, f"training window {windows}")
        purge_span = _purge_span(train_target_mode, horizon)
        if purge_horizon and strict_execution_labels:
            train_end_ts = _purged_end_by_calendar(valid_start, purge_span)
            valid_end_ts = _purged_end_by_calendar(current, purge_span)
        elif purge_horizon:
            train_end_ts = _purged_end_by_code(window, valid_start, purge_span)
            valid_end_ts = _purged_end_by_code(window, current, purge_span)
        else:
            train_end_ts = valid_start - pd.Timedelta(days=1)
            valid_end_ts = current - pd.Timedelta(days=1)
        if purge_horizon:
            requested_purge_windows += 1
        if train_end_ts is None or valid_end_ts is None or valid_end_ts < valid_start:
            if purge_horizon:
                skipped_purge_windows += 1
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
                label_col=ab_label_col or target,
            )
            if picked:
                factors_in_window = picked
            _reject_unsafe_factors(factors_in_window, f"rolling selection window {windows}")
            rolling_selection_manifest_rows.append(
                _rolling_selection_manifest_row(
                    windows, train_start, train_end_ts, valid_start, valid_end_ts,
                    current, test_end, ab_label_col or f"target_ret_{horizon}d",
                    candidate_factors, picked, purge_span,
                )
            )
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
        lgbm_res = None
        if ridge_only:
            print(f"[train:timing] window={windows} stage=lightgbm_ranker skipped=ridge_only", flush=True)
        else:
            lgbm_res = _train_lightgbm_window(
                ridge_only,
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
                model_threads=model_threads,
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
        random_forest_res = None
        if random_forest:
            random_forest_res = qmodel.train_random_forest(
                model_window, factors_in_window, horizon=horizon,
                train_end=train_end, valid_end=valid_end,
                predict_start=current.strftime("%Y-%m-%d"),
                decay_half_life_days=decay_half_life_days, min_weight=min_weight,
                n_estimators=random_forest_estimators,
                max_train_rows=random_forest_max_train_rows,
                label_col=ab_label_col, train_mask_col=ab_mask_col,
            )
            print(f"[train:timing] window={windows} stage=random_forest "
                  f"seconds={time.perf_counter() - stage_started:.2f} ok={random_forest_res.ok}", flush=True)
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
                label_col=ab_label_col,
                train_mask_col=ab_mask_col,
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
        p = _primary_window_predictions(
            ridge_res, lgbm_res, current, test_end, ridge_only,
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
                if random_forest and random_forest_res and random_forest_res.ok and not random_forest_res.predictions.empty:
                    fp = random_forest_res.predictions[(random_forest_res.predictions["date"] >= current) & (random_forest_res.predictions["date"] < test_end)].copy()
                    p = p.merge(
                        fp[["code", "date", "pred"]].rename(columns={"pred": "random_forest_pred"}),
                        on=["code", "date"], how="left",
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
                p["ridge_z"] = p.groupby("date")["ridge_pred"].transform(_daily_zscore)
                if ridge_only:
                    p["lgbm_z"] = np.nan
                    p["base_pred"] = p["ridge_z"]
                else:
                    p["lgbm_z"] = p.groupby("date")["lgbm_pred"].transform(_daily_zscore)
                    p["base_pred"] = float(lgbm_weight) * p["lgbm_z"] + (1 - float(lgbm_weight)) * p["ridge_z"]
                p["pred"] = p["base_pred"]
                if ic_weight and "ic_pred" in p.columns and p["ic_pred"].notna().any():
                    p["ic_z"] = p.groupby("date")["ic_pred"].transform(_daily_zscore)
                    p["pred"] = p["pred"] + float(ic_weight) * p["ic_z"].fillna(0.0)
                if elastic_net and "elastic_pred" in p.columns and p["elastic_pred"].notna().any():
                    p["elastic_z"] = p.groupby("date")["elastic_pred"].transform(_daily_zscore)
                if extra_trees and "extra_trees_pred" in p.columns and p["extra_trees_pred"].notna().any():
                    p["extra_trees_z"] = p.groupby("date")["extra_trees_pred"].transform(_daily_zscore)
                    p["pred"] = p["pred"] + float(extra_trees_weight) * p["extra_trees_z"].fillna(0.0)
                if random_forest and "random_forest_pred" in p.columns and p["random_forest_pred"].notna().any():
                    p["random_forest_z"] = p.groupby("date")["random_forest_pred"].transform(_daily_zscore)
                    p["pred"] = ((1.0 - float(random_forest_weight)) * p["ridge_z"]
                                 + float(random_forest_weight) * p["random_forest_z"].fillna(0.0))
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
                ab_cols = [
                    column
                    for column in (
                        f"tradable_ret_{horizon}d",
                        f"open_ret_{horizon}d",
                        "buyable_close",
                        "buyable_next",
                    )
                    if column in model_window.columns and column not in p.columns
                ]
                if ab_cols:
                    p = p.merge(model_window[["code", "date"] + ab_cols].drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
                # tradable-label 腿的标签列被改名为 tradable_ret_{h}d（model.py 用 label_col 命名输出列），
                # 此时预测输出里没有 target_ret_{h}d。这里用该腿实际的标签列名，避免 KeyError。
                label_out_col = ab_label_col or target
                keep_cols = ["code", "date", label_out_col, "pred", "model", "ridge_pred", "lgbm_pred"]
                keep_cols.extend([c for c in ("ic_pred", "base_pred", "ic_z", "elastic_pred", "elastic_z", "extra_trees_pred", "extra_trees_z", "random_forest_pred", "random_forest_z", "catboost_pred", "catboost_z", "rank_vote_pred", "rank_vote_z") if c in p.columns])
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
            lgbm_message = "skipped:ridge_only" if ridge_only else lgbm_res.message
            print(
                f"[train] window={windows} skipped "
                f"ridge={ridge_res.message} lgbm={lgbm_message}",
                flush=True,
            )
        windows += 1
        if month_cache.max_months > 0 and windows % 10 == 0:
            print(f"[train:cache] {month_cache.stats()}", flush=True)
        del window, model_window
        gc.collect()
        current = test_end

    if purge_horizon:
        completed_purge_windows = requested_purge_windows - skipped_purge_windows
        completion_ratio = (
            completed_purge_windows / requested_purge_windows
            if requested_purge_windows else 1.0
        )
        if completion_ratio < float(min_window_completion_ratio):
            raise RuntimeError(
                "purge window completion ratio below threshold: "
                f"completed={completed_purge_windows} "
                f"requested={requested_purge_windows} "
                f"ratio={completion_ratio:.3f} "
                f"threshold={float(min_window_completion_ratio):.3f}"
            )
    if month_cache.max_months > 0:
        print(f"[train:cache] final {month_cache.stats()}", flush=True)
    if cache_dir is not None:
        print(f"[train:cache] window cache hits={cache_hits} misses={cache_misses}", flush=True)
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    stride = max(int(rebalance_stride), 1)
    trading_calendar = warehouse.load("trading_calendar") if stride > 1 else None
    evaluated_pred = backtest._apply_rebalance_stride(
        pred, stride, trading_calendar=trading_calendar,
    )
    returns, holdings = backtest.portfolio_from_predictions(
        evaluated_pred,
        horizon=horizon,
        top_n=top_n,
        max_weight=max_weight if max_weight is not None else 1.0 / max(int(top_n), 1),
        positive_only=positive_only,
        ridge_quantile=ridge_quantile,
        filter_untradable=True,
        require_tradability=True,
        hold_rank_buffer=max(int(hold_rank_buffer), 0),
    )
    periods_per_year = max(1, int(round(252 / max(horizon, stride))))
    summary = backtest.evaluate_returns(
        returns["ret"] if not returns.empty else pd.Series(dtype=float),
        periods_per_year=periods_per_year,
    )
    if not returns.empty:
        summary["avg_turnover"] = float(returns["turnover"].mean())
        summary["avg_holdings"] = float(returns["n_holdings"].mean())
    summary["ridge_quantile"] = ridge_quantile
    summary["ridge_only"] = bool(ridge_only)
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
    summary["purge_windows_requested"] = int(requested_purge_windows)
    summary["purge_windows_skipped"] = int(skipped_purge_windows)
    summary["purge_window_completion_ratio"] = (
        (requested_purge_windows - skipped_purge_windows) / requested_purge_windows
        if requested_purge_windows else 1.0
    )
    summary["min_window_completion_ratio"] = float(min_window_completion_ratio)
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
    summary["extra_trees_weight"] = float(extra_trees_weight) if extra_trees else 0.0
    summary["random_forest"] = bool(random_forest)
    summary["random_forest_estimators"] = int(random_forest_estimators) if random_forest else None
    summary["random_forest_max_train_rows"] = int(random_forest_max_train_rows) if random_forest else None
    summary["random_forest_weight"] = float(random_forest_weight) if random_forest else 0.0
    summary["rebalance_stride"] = stride
    summary["hold_rank_buffer"] = max(int(hold_rank_buffer), 0)
    summary["authoritative_rebalance_calendar"] = trading_calendar is not None
    summary["n_predictions"] = len(pred)
    summary["n_evaluation_predictions"] = len(evaluated_pred)
    summary_df = pd.DataFrame([{"model": MODEL_NAME, **summary}])

    name_prefix = f"{output_prefix}_" if output_prefix else ""
    if require_selection_provenance and rolling_factor_select:
        if len(rolling_selection_manifest_rows) != len(selected_counts):
            raise RuntimeError(
                "rolling selection provenance is incomplete: "
                f"rows={len(rolling_selection_manifest_rows)} "
                f"completed_windows={len(selected_counts)}"
            )
        warehouse.save(
            f"{name_prefix}factor_selection_manifest",
            pd.DataFrame(rolling_selection_manifest_rows),
        )
    if audit_parts:
        warehouse.save(f"{name_prefix}factor_audit", pd.concat(audit_parts, ignore_index=True))
    if pit_manifest_rows:
        warehouse.save(
            f"{name_prefix}bt_{MODEL_NAME}_universe_manifest",
            pd.DataFrame(pit_manifest_rows),
        )
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
    ap.add_argument(
        "--require-selection-provenance", action="store_true",
        help="fail closed unless <selection>_manifest matches the selected factors and target label",
    )
    ap.add_argument("--output-prefix", default="full_a_batched_rq04_default_ne200")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument(
        "--rebalance-stride", type=int, default=1,
        help="evaluate every N authoritative exchange sessions; 1 keeps daily evaluation",
    )
    ap.add_argument(
        "--hold-rank-buffer", type=int, default=0,
        help="retain held names through top_n + buffer rank; 0 disables hysteresis",
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--min-price-rows", type=int, default=1000)
    ap.add_argument("--limit-codes", type=int, default=0, help="debug only; 0 means all eligible codes")
    ap.add_argument("--universe-file", default="", help="optional six-digit code list restricting panel construction")
    ap.add_argument("--pit-index-code", default="", help="optional PIT index code resolved at each train_start")
    ap.add_argument("--include-trading-gap-risk", action="store_true")
    ap.add_argument("--strict-calendar-factors", action="store_true")
    ap.add_argument("--strict-announcement-lag", action="store_true")
    ap.add_argument("--strict-pit-min-price-rows", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--refresh-months", type=int, default=0, help="rebuild only the latest N monthly raw/prepared partitions; 0 reuses existing panel unless --rebuild")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--validation-months", type=int, default=1)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--ridge-quantile", type=float, default=0.4)
    ap.add_argument("--lgbm-weight", type=float, default=1.0)
    ap.add_argument(
        "--ridge-only", action="store_true",
        help="skip LightGBM and all non-Ridge legs; emit the Ridge cohort directly",
    )
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
    ap.add_argument("--extra-trees-weight", type=float, default=0.0)
    ap.add_argument("--random-forest", action="store_true", help="train a RandomForest regression leg")
    ap.add_argument("--random-forest-estimators", type=int, default=120)
    ap.add_argument("--random-forest-max-train-rows", type=int, default=300000)
    ap.add_argument("--random-forest-weight", type=float, default=0.0)
    ap.add_argument("--window-cache-dir", default=None,
                    help="cache per-window predictions keyed by recipe + month-file signature; "
                         "unchanged historical windows are reused so only the refreshed tail is recomputed")
    ap.add_argument("--min-adv20", type=float, default=None, help="optional prior-20-session absolute ADV buy gate")
    ap.add_argument("--min-listing-sessions", type=int, default=None, help="optional minimum listed exchange sessions buy gate")
    ap.add_argument(
        "--enforce-c30-gates", action="store_true",
        help="protect baseline cache with price signature and reject refresh-months=1",
    )
    ap.add_argument(
        "--strict-execution-labels", action="store_true",
        help="require authoritative calendar and PIT status for non-baseline labels",
    )
    ap.add_argument("--train-target-mode", default="baseline",
                     choices=["baseline", "buyin-mask", "tradable-label", "open-label", "open-buyin-mask"],
                     help="训练目标口径：baseline=收盘到收盘；buyin-mask=收盘买入可买性掩码；"
                          "tradable-label=收盘可实现收益；open-label=次日开盘执行收益；"
                          "open-buyin-mask=次日开盘收益并剔除次日不可买样本")
    args = ap.parse_args()

    config.ensure_dirs()
    _validate_c30_refresh_months(args.enforce_c30_gates, args.refresh_months)
    print(
        "[c30:audit] " + json.dumps(
            _c30_recipe_audit(
                args.train_target_mode, args.refresh_months, args.enforce_c30_gates,
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    build_monthly_panel(
        args.name,
        horizon=args.horizon,
        batch_size=args.batch_size,
        min_price_rows=args.min_price_rows,
        rebuild=args.rebuild,
        limit_codes=args.limit_codes,
        refresh_months=args.refresh_months,
        universe_file=args.universe_file or None,
        include_trading_gap_risk=args.include_trading_gap_risk,
        strict_calendar_factors=args.strict_calendar_factors,
        strict_announcement_lag=args.strict_announcement_lag,
        strict_pit_min_price_rows=args.strict_pit_min_price_rows,
    )
    if args.build_only:
        return
    decay = args.decay_half_life_days if args.decay_half_life_days > 0 else None
    train_batched(
        name=args.name,
        output_prefix=args.output_prefix,
        selection_name=args.selection,
        require_selection_provenance=args.require_selection_provenance,
        horizon=args.horizon,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        top_n=args.top_n,
        ridge_quantile=args.ridge_quantile,
        lgbm_weight=args.lgbm_weight,
        ridge_only=args.ridge_only,
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
        extra_trees_weight=args.extra_trees_weight,
        random_forest=args.random_forest,
        random_forest_estimators=args.random_forest_estimators,
        random_forest_max_train_rows=args.random_forest_max_train_rows,
        random_forest_weight=args.random_forest_weight,
        window_cache_dir=args.window_cache_dir,
        universe_file=args.universe_file or None,
        pit_index_code=args.pit_index_code or None,
        train_target_mode=args.train_target_mode,
        strict_execution_labels=args.strict_execution_labels,
        enforce_c30_gates=args.enforce_c30_gates,
        min_adv20=args.min_adv20,
        min_listing_sessions=args.min_listing_sessions,
        rebalance_stride=args.rebalance_stride,
        hold_rank_buffer=args.hold_rank_buffer,
    )


if __name__ == "__main__":
    main()
