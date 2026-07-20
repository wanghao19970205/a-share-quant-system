"""Check whether daily quant data and snapshots have been updated."""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description="检查每日增量更新结果")
    ap.add_argument("--universe", default="full_a", choices=list(config.UNIVERSES))
    ap.add_argument("--sample", type=int, default=20, help="抽样检查股票数量")
    ap.add_argument("--snapshot-dir", default=os.environ.get("SNAPSHOT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots")))
    args = ap.parse_args()

    u = config.UNIVERSES[args.universe]
    codes = datafeed.universe(u["kind"], u["arg"])
    sample_codes = codes[:args.sample]

    rows = []
    for code in sample_codes:
        rows.append({
            "code": code,
            "price_rows": len(warehouse.load_price(code)),
            "price_last": _max_date(warehouse.load_price(code)),
            "valuation_rows": len(warehouse.load_valuation(code)),
            "valuation_last": _max_date(warehouse.load_valuation(code)),
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


if __name__ == "__main__":
    main()
