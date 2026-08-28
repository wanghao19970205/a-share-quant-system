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
import pyarrow.parquet as pq

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


def save_trading_calendar(df: pd.DataFrame) -> None:
    """原子保存规范化交易日历；只保留有效唯一 date 列。"""
    if df is None or "date" not in df.columns:
        raise ValueError("交易日历必须包含 date 列")
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce")})
    out = out.dropna()
    out["date"] = out["date"].dt.normalize()
    out = out.drop_duplicates().sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("交易日历为空")
    _atomic_parquet(out, config.TRADING_CALENDAR_FILE)


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
    # 统一 date 列精度为 ns：新版 pandas 解析日期可能产出 us，下游 merge_asof 要求一致。
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").astype("datetime64[ns]")
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


def latest_price_date(code: str):
    """Read a price file's latest date from Parquet footer statistics."""
    path = os.path.join(config.PRICE_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        parquet = pq.ParquetFile(path)
        date_index = parquet.schema_arrow.get_field_index("date")
        if date_index < 0:
            return None
        maxima = []
        for group_index in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(group_index).column(date_index).statistics
            if statistics is not None and statistics.has_min_max:
                maxima.append(statistics.max)
        if maxima:
            return pd.Timestamp(max(maxima)).date()
    except Exception:  # noqa: BLE001
        pass
    return _last_price_date_from_column(path)


def _last_price_date_from_column(path: str):
    try:
        frame = pd.read_parquet(path, columns=["date"])
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        return dates.max().date() if not dates.empty else None
    except Exception:  # noqa: BLE001
        return None


def save_valuation(code: str, df: pd.DataFrame) -> None:
    config.ensure_dirs()
    _atomic_parquet(df, os.path.join(config.VALUATION_DIR, f"{code}.parquet"))


def load_valuation(code: str) -> pd.DataFrame:
    p = os.path.join(config.VALUATION_DIR, f"{code}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
