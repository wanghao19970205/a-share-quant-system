"""量化选股 · 多数据仓结果对比。"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def load_summary(data_dir: str, name: str | None = None) -> pd.DataFrame:
    p = os.path.join(data_dir, "pipeline_summary.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.insert(0, "dataset", name or os.path.basename(os.path.normpath(data_dir)))
    return df


def compare(data_dirs: list[str]) -> pd.DataFrame:
    parts = [load_summary(p) for p in data_dirs]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="对比多个量化数据仓的训练/回测结果")
    ap.add_argument("data_dirs", nargs="+", help="例如 quant_data/hs300 quant_data/full_a_sample")
    ap.add_argument("--output", default="", help="可选：保存合并后的 parquet 路径")
    args = ap.parse_args()
    df = compare(args.data_dirs)
    if df.empty:
        raise SystemExit("未找到任何 pipeline_summary.parquet")
    if args.output:
        df.to_parquet(args.output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
