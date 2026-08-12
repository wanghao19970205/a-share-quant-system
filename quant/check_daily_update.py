"""Check whether daily quant data and snapshots have been updated."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from quant import config, datafeed, warehouse


def _max_date(df: pd.DataFrame, col: str = "date") -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    return s.max().strftime("%Y-%m-%d") if not s.empty else ""


def _read_watchlist(snapshot_dir: str) -> list[str]:
    path = os.path.join(snapshot_dir, "watchlist.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return sorted(set(re.findall(r"\d{6}", f.read())))


def _snapshot_date(snapshot_dir: str, code: str) -> str:
    path = os.path.join(snapshot_dir, f"{code}.csv")
    if not os.path.exists(path):
        return ""
    try:
        df = pd.read_csv(path, dtype={"date": str})
    except Exception:  # noqa: BLE001
        return ""
    if df.empty or "date" not in df.columns:
        return ""
    return str(df["date"].dropna().astype(str).max() or "")


def _market_bucket(code: str) -> str:
    normalized = str(code).zfill(6)
    if normalized.startswith(("6", "68")):
        return "shanghai"
    if normalized.startswith(("0", "3")):
        return "shenzhen"
    if normalized.startswith(("4", "8", "9")):
        return "beijing"
    return "other"


def _stratified_hash_sample(
    codes: list[str], sample_size: int, seed: str = "daily-health-v1"
) -> list[str]:
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    buckets: dict[str, list[str]] = {}
    for code in sorted(set(str(value).zfill(6) for value in codes)):
        buckets.setdefault(_market_bucket(code), []).append(code)
    for bucket_codes in buckets.values():
        bucket_codes.sort(
            key=lambda code: hashlib.sha256(
                f"{seed}:{code}".encode("ascii")
            ).hexdigest()
        )
    selected: list[str] = []
    bucket_names = sorted(buckets)
    while len(selected) < min(sample_size, sum(map(len, buckets.values()))):
        advanced = False
        for name in bucket_names:
            if buckets[name] and len(selected) < sample_size:
                selected.append(buckets[name].pop(0))
                advanced = True
        if not advanced:
            break
    return selected


def _expected_price_date(
    calendar: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    max_stale_sessions: int = 0,
) -> pd.Timestamp:
    if calendar.empty or list(calendar.columns) != ["date"]:
        raise ValueError("authoritative trading calendar must contain only date")
    dates = pd.to_datetime(calendar["date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("authoritative trading calendar must be unique and increasing")
    allowed = dates[dates <= pd.Timestamp(as_of_date).normalize()]
    lag = int(max_stale_sessions)
    if lag < 0:
        raise ValueError("max stale sessions must be non-negative")
    if len(allowed) <= lag:
        raise ValueError("authoritative trading calendar has no eligible session")
    return pd.Timestamp(allowed.iloc[-(lag + 1)])


def _stale_price_codes(df: pd.DataFrame, expected: pd.Timestamp) -> list[str]:
    if df.empty:
        return ["<empty-sample>"]
    observed = pd.to_datetime(df["price_last"], errors="coerce")
    return df.loc[observed.isna() | (observed < expected), "code"].astype(str).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="检查每日增量更新结果")
    ap.add_argument("--universe", default="full_a", choices=list(config.UNIVERSES))
    ap.add_argument("--sample", type=int, default=20, help="抽样检查股票数量")
    ap.add_argument("--sample-seed", default="daily-health-v1")
    ap.add_argument(
        "--fail-on-stale-price", action="store_true",
        help="按权威交易日历检查样本价格陈旧并以非零状态退出",
    )
    ap.add_argument("--max-stale-sessions", type=int, default=0)
    ap.add_argument("--as-of-date", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--snapshot-dir", default=os.environ.get("SNAPSHOT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots")))
    args = ap.parse_args()

    u = config.UNIVERSES[args.universe]
    codes = datafeed.universe(u["kind"], u["arg"])
    sample_codes = _stratified_hash_sample(codes, args.sample, args.sample_seed)

    rows = []
    for code in sample_codes:
        price = warehouse.load_price(code)
        valuation = warehouse.load_valuation(code)
        rows.append({
            "code": code,
            "price_rows": len(price),
            "price_last": _max_date(price),
            "valuation_rows": len(valuation),
            "valuation_last": _max_date(valuation),
        })
    df = pd.DataFrame(rows)
    print(f"[quant_dir] {config.QUANT_DIR}")
    print(f"[sample] {len(df)} / universe={len(codes)}")
    if not df.empty:
        print(df.to_string(index=False))
        print("[price_last_counts]")
        print(df["price_last"].value_counts(dropna=False).to_string())
        print("[valuation_last_counts]")
        print(df["valuation_last"].value_counts(dropna=False).to_string())

    for name in ("block_trades", "lhb", "margin_sse", "margin_szse", "financial_yjbb", "performance_forecast", "holder_num", "dividend"):
        table = warehouse.load(name)
        date_col = "date" if "date" in table.columns else ("ann_date" if "ann_date" in table.columns else ("report_date" if "report_date" in table.columns else ""))
        print(f"[table] {name}: rows={len(table)} last={_max_date(table, date_col) if date_col else ''}")

    watch = _read_watchlist(args.snapshot_dir)
    snap_rows = [{"code": c, "snapshot_last": _snapshot_date(args.snapshot_dir, c)} for c in watch]
    sdf = pd.DataFrame(snap_rows)
    print(f"[snapshots] dir={args.snapshot_dir} watchlist={len(watch)}")
    if not sdf.empty:
        print(sdf.to_string(index=False))
        print("[snapshot_last_counts]")
        print(sdf["snapshot_last"].value_counts(dropna=False).to_string())

    if args.fail_on_stale_price:
        expected = _expected_price_date(
            warehouse.load("trading_calendar"),
            args.as_of_date,
            args.max_stale_sessions,
        )
        stale_codes = _stale_price_codes(df, expected)
        print(
            f"[stale_gate] expected>={expected.date()} "
            f"stale={len(stale_codes)}/{len(df)}"
        )
        if stale_codes:
            raise SystemExit(
                "stale sampled price data: " + ",".join(stale_codes[:20])
            )


if __name__ == "__main__":
    main()
