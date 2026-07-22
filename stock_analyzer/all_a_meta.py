"""Build and load a persistent Eastmoney industry/concept map for all A shares."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import akshare as ak
import pandas as pd

from stock_analyzer import data, net

_MEMBERSHIP_COLUMNS = [
    "board_type", "board_code", "board_name", "member_count", "code", "name", "fetched_at",
]
_META_COLUMNS = [
    "code", "name", "market_board", "a_industry", "a_industries",
    "a_concepts", "meta_updated_at",
]
_INDUSTRY_HISTORY_COLUMNS = [
    "code", "industry", "valid_from", "valid_to", "available_from", "source_updated_at", "source",
]


def _snapshot_dir() -> str:
    return os.environ.get("SNAPSHOT_DIR", "snapshots")


def membership_path() -> str:
    return os.path.join(_snapshot_dir(), "all_a_board_membership.parquet")


def meta_path() -> str:
    return os.path.join(_snapshot_dir(), "all_a_stock_meta.parquet")


def industry_history_path() -> str:
    return os.path.join(_snapshot_dir(), "sw_industry_history_pit.parquet")


def build_sw_industry_history() -> pd.DataFrame:
    """Convert SW's classification-change file into a conservative PIT interval table."""
    raw = ak.stock_industry_clf_hist_sw()
    required = {"symbol", "start_date", "industry_code", "update_time"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"SW industry history columns missing: {sorted(missing)}")
    history = raw.rename(columns={
        "symbol": "code",
        "industry_code": "industry",
        "start_date": "valid_from",
        "update_time": "source_updated_at",
    }).copy()
    history["code"] = history["code"].astype(str).map(data._normalize_symbol)
    history["industry"] = history["industry"].astype(str).str.strip()
    history["valid_from"] = pd.to_datetime(history["valid_from"], errors="coerce").dt.normalize()
    history["source_updated_at"] = pd.to_datetime(
        history["source_updated_at"], errors="coerce"
    ).dt.normalize()
    history = history.dropna(subset=["code", "industry", "valid_from", "source_updated_at"])
    history = history.sort_values(["code", "valid_from"]).drop_duplicates(
        ["code", "valid_from"], keep="last"
    )
    history["valid_to"] = history.groupby("code")["valid_from"].shift(-1)
    # A classification is only usable after both its effective date and the source publication date.
    history["available_from"] = history[["valid_from", "source_updated_at"]].max(axis=1)
    history["source"] = "akshare.stock_industry_clf_hist_sw"
    return history[_INDUSTRY_HISTORY_COLUMNS].reset_index(drop=True)


def update_sw_industry_history() -> dict:
    history = build_sw_industry_history()
    if history.empty or history["code"].nunique() < 3000:
        raise RuntimeError("refusing to publish incomplete SW industry history")
    _atomic_parquet(history, industry_history_path())
    result = {
        "rows": len(history),
        "stocks": int(history["code"].nunique()),
        "industries": int(history["industry"].nunique()),
        "available_min": str(history["available_from"].min().date()),
        "available_max": str(history["available_from"].max().date()),
    }
    print(f"[sw-industry-history] done {result}", flush=True)
    return result


def market_board(code: str) -> str:
    code = data._normalize_symbol(code)
    if code.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    if code.startswith(("000", "001", "002", "003")):
        return "深市主板"
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("4", "8", "92")):
        return "北交所"
    if code.startswith("900"):
        return "沪市B股"
    if code.startswith("200"):
        return "深市B股"
    return "其他"


def _clean_text(value) -> str:
    value = str(value or "").strip()
    return "" if value.lower() == "nan" else value


def _join_unique(values, limit: int | None = None) -> str:
    out: list[str] = []
    for value in values:
        value = _clean_text(value)
        if value and value not in out:
            out.append(value)
    if limit is not None:
        out = out[:limit]
    return "、".join(out)


def _atomic_parquet(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _retry(call, retries: int, label: str):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with net.akshare_proxied():
                return call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(8.0, 1.5 * attempt) + random.uniform(0.2, 0.8))
    raise RuntimeError(f"{label}: {type(last_error).__name__}: {last_error}") from last_error


def _board_list(board_type: str, retries: int) -> pd.DataFrame:
    fn = ak.stock_board_industry_name_em if board_type == "industry" else ak.stock_board_concept_name_em
    frame = _retry(fn, retries, f"load {board_type} boards")
    if frame is None or frame.empty or not {"板块代码", "板块名称"}.issubset(frame.columns):
        raise RuntimeError(f"{board_type} board list is empty or malformed")
    return frame[["板块代码", "板块名称"]].dropna().drop_duplicates("板块代码")


def _fetch_board(
    board_type: str,
    board_code: str,
    board_name: str,
    retries: int,
    delay: float,
) -> pd.DataFrame:
    fn = ak.stock_board_industry_cons_em if board_type == "industry" else ak.stock_board_concept_cons_em
    try:
        members = _retry(lambda: fn(symbol=board_code), retries, f"{board_type} {board_code} {board_name}")
        if members is None or members.empty or "代码" not in members.columns:
            raise RuntimeError(f"{board_type} {board_code} {board_name}: empty constituents")
        fetched_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        names = members["名称"] if "名称" in members.columns else pd.Series("", index=members.index)
        result = pd.DataFrame({
            "board_type": board_type,
            "board_code": board_code,
            "board_name": board_name,
            "member_count": len(members),
            "code": members["代码"].astype(str).map(data._normalize_symbol),
            "name": names.map(_clean_text),
            "fetched_at": fetched_at,
        })
        return result[result["code"].str.len() == 6]
    finally:
        if delay > 0:
            time.sleep(delay + random.uniform(0.0, min(delay, 0.5)))


def _load_existing_membership() -> pd.DataFrame:
    path = membership_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)
    try:
        frame = pd.read_parquet(path)
        for column in _MEMBERSHIP_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[_MEMBERSHIP_COLUMNS].copy()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)


def _aggregate_meta(membership: pd.DataFrame) -> pd.DataFrame:
    name_map: dict[str, str] = {}
    try:
        listing = ak.stock_info_a_code_name()
        code_col = "code" if "code" in listing.columns else "代码"
        name_col = "name" if "name" in listing.columns else "名称"
        name_map = dict(zip(
            listing[code_col].astype(str).map(data._normalize_symbol),
            listing[name_col].map(_clean_text),
        ))
    except Exception:  # noqa: BLE001
        pass

    codes = set(name_map)
    grouped: dict[str, pd.DataFrame] = {}
    if not membership.empty:
        normalized = membership.copy()
        normalized["code"] = normalized["code"].astype(str).map(data._normalize_symbol)
        grouped = {code: items for code, items in normalized.groupby("code", sort=False)}
        codes.update(grouped)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict] = []
    empty_items = pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)
    for code in sorted(codes):
        items = grouped.get(code, empty_items)
        industry_items = items[items["board_type"] == "industry"].sort_values(
            ["member_count", "board_code"], na_position="last"
        )
        industries = industry_items["board_name"].tolist()
        concept_items = items[items["board_type"] == "concept"].sort_values("board_name")
        concepts = concept_items["board_name"].tolist()
        member_names = items["name"].tolist()
        rows.append({
            "code": code,
            "name": name_map.get(code) or next((_clean_text(x) for x in member_names if _clean_text(x)), code),
            "market_board": market_board(code),
            "a_industry": _join_unique(industries, limit=1),
            "a_industries": _join_unique(industries),
            "a_concepts": _join_unique(concepts),
            "meta_updated_at": now,
        })
    return pd.DataFrame(rows, columns=_META_COLUMNS)


def update_all_a_meta(workers: int = 1, retries: int = 3, delay: float = 0.4) -> dict:
    """Refresh Eastmoney board memberships, retaining old rows for failed boards."""
    existing = _load_existing_membership()
    fetched: list[pd.DataFrame] = []
    successful_keys: set[tuple[str, str]] = set()
    failures: list[str] = []
    board_total = 0
    board_totals: dict[str, int] = {}

    for board_type in ("industry", "concept"):
        boards = _board_list(board_type, retries)
        if board_type == "industry" and len(boards) < 300:
            raise RuntimeError(f"industry board list unexpectedly small: {len(boards)}")
        if board_type == "concept" and len(boards) < 300:
            raise RuntimeError(f"concept board list unexpectedly small: {len(boards)}")
        jobs = [
            (board_type, _clean_text(row["板块代码"]), _clean_text(row["板块名称"]))
            for _, row in boards.iterrows()
        ]
        board_total += len(jobs)
        board_totals[board_type] = len(jobs)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [
                (
                    pool.submit(_fetch_board, kind, code, name, retries, delay),
                    (kind, code, name),
                )
                for kind, code, name in jobs
            ]
            completed = as_completed([future for future, _ in futures]) if workers > 1 else (
                future for future, _ in futures
            )
            job_by_future = dict(futures)
            for index, future in enumerate(completed, start=1):
                kind, code, name = job_by_future[future]
                try:
                    fetched.append(future.result())
                    successful_keys.add((kind, code))
                except Exception as exc:  # noqa: BLE001
                    failures.append(str(exc))
                if index % 25 == 0 or index == len(jobs):
                    print(
                        f"[all-a-meta] {kind} {index}/{len(jobs)} "
                        f"ok={sum(1 for key in successful_keys if key[0] == kind)} failures={len(failures)}",
                        flush=True,
                    )

    if fetched:
        fresh = pd.concat(fetched, ignore_index=True)
    else:
        fresh = pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)
    if not existing.empty and successful_keys:
        old_keys = list(zip(existing["board_type"].astype(str), existing["board_code"].astype(str)))
        existing = existing[[key not in successful_keys for key in old_keys]]
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates(["board_type", "board_code", "code"], keep="last")

    first_snapshot = not os.path.exists(membership_path())
    success_by_type = {
        kind: sum(1 for key in successful_keys if key[0] == kind)
        for kind in board_totals
    }
    incomplete_type = any(
        success_by_type[kind] / total < 0.70
        for kind, total in board_totals.items()
        if total
    )
    if combined.empty or (first_snapshot and incomplete_type):
        raise RuntimeError(
            "refusing to publish incomplete first snapshot: "
            + ", ".join(
                f"{kind}={success_by_type[kind]}/{total}"
                for kind, total in board_totals.items()
            )
        )

    meta = _aggregate_meta(combined)
    _atomic_parquet(combined[_MEMBERSHIP_COLUMNS], membership_path())
    _atomic_parquet(meta, meta_path())
    result = {
        "boards_total": board_total,
        "boards_updated": len(successful_keys),
        "boards_failed": len(failures),
        "stocks": len(meta),
        "industry_covered": int(meta["a_industry"].ne("").sum()),
        "concept_covered": int(meta["a_concepts"].ne("").sum()),
    }
    print(f"[all-a-meta] done {result}", flush=True)
    if failures:
        print("[all-a-meta] failures:\n  - " + "\n  - ".join(failures[:30]), flush=True)
    return result


@lru_cache(maxsize=4)
def _load_meta_cached(path: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    try:
        frame = pd.read_parquet(path)
        if frame.empty or "code" not in frame.columns:
            return pd.DataFrame(columns=_META_COLUMNS)
        frame = frame.copy()
        frame["code"] = frame["code"].astype(str).map(data._normalize_symbol)
        return frame.drop_duplicates("code", keep="last")
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_META_COLUMNS)


def load_all_a_meta() -> pd.DataFrame:
    path = meta_path()
    try:
        mtime_ns = os.stat(path).st_mtime_ns
    except OSError:
        return pd.DataFrame(columns=_META_COLUMNS)
    return _load_meta_cached(path, mtime_ns).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh persistent all-A Eastmoney metadata")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--sw-history-only", action="store_true")
    args = parser.parse_args()
    if args.sw_history_only:
        update_sw_industry_history()
    else:
        update_all_a_meta(workers=args.workers, retries=args.retries, delay=args.delay)


if __name__ == "__main__":
    main()
