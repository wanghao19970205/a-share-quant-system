"""Persisted LLM ranking for the fixed Top10 views.

The scheduler runs this after a successful quant model publish. UIs only read the
published JSON; individual-stock analysis remains an on-demand live request.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

import pandas as pd

from stock_analyzer import all_a_meta, candidate_eval, data, quant_signal, screener, stock_meta

_GROUPS = ("白名单", "全A", "创新药")
_MAIN_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
_CACHE_VERSION = 3


def cache_path() -> str:
    return os.path.join(os.environ.get("SNAPSHOT_DIR", "snapshots"), "top10_llm_eval.json")


def _read_cache() -> dict:
    path = cache_path()
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_write(value: dict) -> None:
    path = cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".top10_llm_eval-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _innovation_codes() -> set[str]:
    path = all_a_meta.membership_path()
    if os.path.exists(path):
        membership = _safe(lambda: pd.read_parquet(path), pd.DataFrame())
        if membership is not None and not membership.empty:
            names = membership["board_name"].fillna("").astype(str)
            matched = membership[(membership["board_type"] == "concept") & names.str.contains("创新药", regex=False)]
            if not matched.empty:
                return {data._normalize_symbol(x) for x in matched["code"].astype(str)}
    return {data._normalize_symbol(x) for x in (_safe(lambda: screener.concept_constituents("创新药", limit=1000), []) or [])}


def ranking_frames() -> dict[str, pd.DataFrame]:
    frame = _safe(lambda: quant_signal.latest_frame(profile="short_stable", style="short_1_3"), pd.DataFrame())
    if frame is None or frame.empty:
        return {group: pd.DataFrame() for group in _GROUPS}
    whitelist = _safe(lambda: quant_signal.watchlist_frame(profile="short_stable", style="short_1_3"), pd.DataFrame())
    if whitelist is not None and not whitelist.empty:
        whitelist = whitelist.sort_values("watch_rank", na_position="last").head(10).copy()
    else:
        whitelist = pd.DataFrame()
    all_a = frame[frame["code"].astype(str).str.startswith(_MAIN_PREFIXES)].sort_values("rank").head(10).copy()
    innovation = frame[frame["code"].isin(_innovation_codes())].sort_values("rank").head(10).copy()
    if not all_a.empty:
        all_a = stock_meta.enrich_frame(all_a, remote=False, use_all_a_meta=True)
    if not innovation.empty:
        innovation = stock_meta.enrich_frame(innovation, remote=False, use_all_a_meta=True)
    return {"白名单": whitelist, "全A": all_a, "创新药": innovation}


def _fingerprint(frame: pd.DataFrame) -> str:
    rows = []
    quant_dir = os.environ.get("QUANT_DATA_DIR", "")
    for _, row in frame.iterrows():
        rank = row.get("watch_rank")
        if rank is None or pd.isna(rank):
            rank = row.get("rank")
        code = str(row.get("code"))
        quote_state = None
        if quant_dir:
            path = os.path.join(quant_dir, "price", f"{code}.parquet")
            try:
                stat = os.stat(path)
                quote_state = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
            except OSError:
                quote_state = None
        rows.append({
            "code": code,
            "rank": None if rank is None or pd.isna(rank) else int(rank),
            "pred": round(float(row.get("pred")), 8),
            "quote_state": quote_state,
        })
    raw = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _same_input(entry: dict, frame: pd.DataFrame) -> bool:
    if frame.empty or not isinstance(entry, dict):
        return False
    expected = frame["code"].astype(str).tolist()
    return (entry.get("cache_version") == _CACHE_VERSION
            and not entry.get("failed_codes")
            and str(entry.get("date")) == str(frame["date"].iloc[0])
            and entry.get("codes") == expected
            and entry.get("fingerprint") == _fingerprint(frame)
            and len(entry.get("rows") or []) == len(expected))


def refresh(key: str = "", model: str = "", base_url: str = "") -> dict:
    """Evaluate changed fixed Top10 groups and atomically publish their results."""
    cache = _read_cache()
    frames = ranking_frames()
    summary = {"updated": [], "skipped": [], "failed": []}
    changed = {}
    for group, frame in frames.items():
        if frame.empty:
            summary["failed"].append(group)
        elif _same_input(cache.get(group, {}), frame):
            summary["skipped"].append(group)
        else:
            changed[group] = frame

    unique_codes = list(dict.fromkeys(
        code
        for frame in changed.values()
        for code in frame["code"].astype(str).tolist()
    ))
    result = {}
    rows_by_code = {}
    if unique_codes:
        result = _safe(lambda: candidate_eval.evaluate_top(
            unique_codes, key=key, model=model, base_url=base_url, max_workers=2,
            profile="short_stable", style="short_1_3", broker_retry=1), {}) or {}
        rows_by_code = {
            str(row.get("code")): row
            for row in result.get("rows", [])
            if row.get("available")
        }

    for group, frame in changed.items():
        codes = frame["code"].astype(str).tolist()
        previous_rows = {
            str(row.get("code")): dict(row)
            for row in (cache.get(group, {}).get("rows") or [])
            if row.get("code")
        }
        failed_codes = [code for code in codes if code not in rows_by_code]
        rows = []
        for code in codes:
            if code in rows_by_code:
                row = dict(rows_by_code[code])
                row.pop("stale", None)
            elif code in previous_rows:
                row = dict(previous_rows[code])
                row["stale"] = True
            else:
                continue
            rows.append(row)
        if not rows:
            summary["failed"].append(group)
            continue
        rows.sort(key=lambda row: row.get("rank_score", -1e9), reverse=True)
        for rank, row in enumerate(rows, 1):
            row["llm_rank"] = rank
        cache[group] = {
            "cache_version": _CACHE_VERSION,
            "date": str(frame["date"].iloc[0]),
            "model": str(result.get("model") or "-"),
            "quant_model": str(frame.get("model", pd.Series(["active_quant"])).iloc[0]),
            "codes": codes,
            "fingerprint": _fingerprint(frame),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "failed_codes": failed_codes,
            "rows": rows,
        }
        summary["updated"].append(group)
        if failed_codes:
            summary["failed"].append(group)
    if summary["updated"] or not os.path.exists(cache_path()):
        _atomic_write(cache)
    summary["unique_evaluated"] = len(unique_codes)
    summary["path"] = cache_path()
    return summary


def load() -> dict:
    return _read_cache()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Refresh persisted LLM rankings for fixed Top10 groups")
    parser.add_argument("--key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()
    print(json.dumps(refresh(args.key, args.model, args.base_url), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
