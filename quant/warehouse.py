"""量化选股 · 本地数据仓（parquet）。

- 全市场按报告期/日期的宽表：``save(name, df)`` / ``load(name)``（如 financial_yjbb、dzjy、lhb）。
- 每股时序（日线、估值）：``save_price/load_price``、``save_valuation/load_valuation``。
- ``upsert`` 支持按主键去重增量合并（日更时避免重复）。

存储用 parquet（列式、压缩、读取快），需 pyarrow（已确认可用）。
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from quant import config


def _p(name: str) -> str:
    return os.path.join(config.QUANT_DIR, f"{name}.parquet")


def _atomic_parquet(df: pd.DataFrame, path: str, row_group_size: int | None = None) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        df.to_parquet(temporary, index=False, row_group_size=row_group_size)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(name: str, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    _atomic_parquet(df, _p(name))


def load(name: str) -> pd.DataFrame:
    p = _p(name)
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


def upsert(name: str, df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """按 keys 去重合并到已有表（新数据覆盖旧），返回合并后的表并落盘。"""
    if df is None or df.empty:
        return load(name)
    old = load(name)
    merged = pd.concat([old, df], ignore_index=True) if not old.empty else df
    merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    save(name, merged)
    return merged


# ------------------------- 每股时序 -------------------------
def save_price(code: str, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    _atomic_parquet(df, os.path.join(config.PRICE_DIR, f"{code}.parquet"), row_group_size=256)


def load_price_tail(
    code: str,
    start_date: pd.Timestamp,
    warmup_rows: int,
) -> pd.DataFrame:
    """Read the exact warmup tail and current rows without decoding old price columns."""
    path = os.path.join(config.PRICE_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    dates = pd.read_parquet(path, columns=["date"])
    if dates.empty:
        return dates
    normalized = pd.to_datetime(dates["date"], errors="coerce")
    start = pd.Timestamp(start_date)
    before = normalized[normalized < start].dropna()
    warmup = max(int(warmup_rows), 0)
    if before.empty or warmup == 0:
        cutoff = start
    else:
        cutoff = pd.Timestamp(before.iloc[-min(len(before), warmup)])
    return pd.read_parquet(path, filters=[("date", ">=", cutoff)])


def load_price(code: str) -> pd.DataFrame:
    p = os.path.join(config.PRICE_DIR, f"{code}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


def save_valuation(code: str, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    _atomic_parquet(df, os.path.join(config.VALUATION_DIR, f"{code}.parquet"))


def load_valuation(code: str) -> pd.DataFrame:
    p = os.path.join(config.VALUATION_DIR, f"{code}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
