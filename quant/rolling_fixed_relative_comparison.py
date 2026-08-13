"""Research-only rolling/fixed market and PIT-industry relative comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400.evaluation import _holm_adjust, _paired_block_bootstrap
from intraday_1400.storage import artifact_hash, atomic_json, atomic_parquet


LABEL = "tradable_ret_1d"
BUYABLE = "buyable_close"
KEYS = ["code", "date"]


def _normalize_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = set(KEYS + [LABEL, BUYABLE])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing strict columns: {missing}")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="coerce")
    frame[BUYABLE] = frame[BUYABLE].fillna(False).astype(bool)
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{path} has invalid key values")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path} has duplicate prediction keys")
    if not np.isfinite(frame[LABEL].dropna()).all():
        raise ValueError(f"{path} has non-finite labels")
    return frame


def _key_hash(frame: pd.DataFrame) -> str:
    keys = frame[KEYS].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    keys = keys.drop_duplicates().sort_values(["date", "code"])
    return hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _date_hash(frame: pd.DataFrame) -> str:
    dates = sorted(pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").unique())
    return hashlib.sha256("\n".join(dates).encode()).hexdigest()


def _pit_industry(frame: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    required = {"code", "industry", "valid_from", "available_from"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"industry history missing columns: {missing}")
    left = frame[KEYS].copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    hist = history.copy()
    hist["code"] = hist["code"].astype(str).str.zfill(6)
    for col in ("valid_from", "valid_to", "available_from"):
        if col not in hist:
            hist[col] = pd.NaT
        hist[col] = pd.to_datetime(hist[col], errors="coerce").dt.normalize()
    candidates = left.reset_index(names="row_id").merge(hist, on="code", how="left")
    visible = candidates[
        candidates["valid_from"].le(candidates["date"])
        & (candidates["valid_to"].isna() | candidates["date"].lt(candidates["valid_to"]))
        & candidates["available_from"].le(candidates["date"])
    ].sort_values(["row_id", "available_from", "valid_from"])
    visible = visible.drop_duplicates("row_id", keep="last")
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[visible["row_id"].to_numpy()] = visible["industry"].astype("string").to_numpy()
    return result


def _daily_benchmarks(frame: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = frame[["code", "date", LABEL]].copy()
    data["pit_industry"] = _pit_industry(data, history).to_numpy()
    market = data.groupby("date")[LABEL].mean().rename("market_equal_weight")
    industry = (
        data.dropna(subset=["pit_industry"])
        .groupby(["date", "pit_industry"])[LABEL].mean()
        .groupby("date").mean()
        .rename("pit_industry_equal_weight")
    )
    out = pd.concat([market, industry], axis=1).reset_index()
    out["industry_mapped_rows"] = data["pit_industry"].notna().groupby(data["date"]).sum()
    out["industry_total_rows"] = data.groupby("date")[LABEL].count()
    stats = {
        "prediction_rows": int(len(data)),
        "industry_mapped_rows": int(data["pit_industry"].notna().sum()),
        "industry_mapping_rate": float(data["pit_industry"].notna().mean()),
        "industry_count": int(data["pit_industry"].nunique()),
        "market_days": int(market.size),
        "industry_days": int(industry.size),
    }
    return out, stats


def compare(args: argparse.Namespace) -> None:
    rolling = _normalize_predictions(Path(args.rolling_predictions))
    fixed = _normalize_predictions(Path(args.fixed_predictions))
    r_keys, f_keys = _key_hash(rolling), _key_hash(fixed)
    if r_keys != f_keys:
        raise ValueError("rolling/fixed prediction key hashes differ")
    if not rolling[LABEL].equals(fixed[LABEL]):
        raise ValueError("rolling/fixed strict labels differ")
    if not rolling[BUYABLE].equals(fixed[BUYABLE]):
        raise ValueError("rolling/fixed buyability masks differ")
    returns = []
    for name, path in (("rolling", args.rolling_returns), ("fixed", args.fixed_returns)):
        frame = pd.read_parquet(path).copy()
        if not {"date", "ret"}.issubset(frame.columns):
            raise ValueError(f"{path} missing date/ret")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame = frame.drop_duplicates("date").set_index("date")["ret"].rename(f"{name}_absolute_return")
        returns.append(frame)
    history = pd.read_parquet(args.industry_history)
    bench, coverage = _daily_benchmarks(rolling, history)
    result = bench.set_index("date").join(returns, how="inner").reset_index()
    result["rolling_market_excess"] = result["rolling_absolute_return"] - result["market_equal_weight"]
    result["fixed_market_excess"] = result["fixed_absolute_return"] - result["market_equal_weight"]
    result["rolling_industry_excess"] = result["rolling_absolute_return"] - result["pit_industry_equal_weight"]
    result["fixed_industry_excess"] = result["fixed_absolute_return"] - result["pit_industry_equal_weight"]
    blocks = [int(x) for x in args.block_lengths]
    comparisons = {}
    for baseline in ("market_equal_weight", "pit_industry_equal_weight"):
        model_stats = {}
        for model in ("rolling", "fixed"):
            model_stats[model] = {
                str(block): _paired_block_bootstrap(
                    result[f"{model}_absolute_return"], result[baseline], args.bootstrap_samples, block, args.seed
                ) for block in blocks
            }
        p_values = [model_stats["rolling"][str(block)].get("p_value_one_sided") for block in blocks]
        valid_p = [p for p in p_values if p is not None]
        comparisons[baseline] = {"rolling": model_stats["rolling"], "fixed": model_stats["fixed"], "holm_p_values": _holm_adjust(valid_p)}
    comparisons["rolling_vs_fixed"] = {
        str(block): _paired_block_bootstrap(
            result["rolling_absolute_return"], result["fixed_absolute_return"],
            args.bootstrap_samples, block, args.seed,
        )
        for block in blocks
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_parquet(result, output / "relative_daily.parquet")
    evidence = {
        "schema_version": 1,
        "research_only": True,
        "production_publication": False,
        "label_col": LABEL,
        "prediction_key_hash": r_keys,
        "prediction_date_hash": _date_hash(rolling),
        "rolling_predictions_sha256": artifact_hash(Path(args.rolling_predictions)),
        "fixed_predictions_sha256": artifact_hash(Path(args.fixed_predictions)),
        "rolling_returns_sha256": artifact_hash(Path(args.rolling_returns)),
        "fixed_returns_sha256": artifact_hash(Path(args.fixed_returns)),
        "industry_history_sha256": artifact_hash(Path(args.industry_history)),
        "common_days": int(len(result)),
        "coverage": coverage,
        "comparisons": comparisons,
    }
    atomic_json(evidence, output / "relative_evidence.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-predictions", required=True)
    parser.add_argument("--fixed-predictions", required=True)
    parser.add_argument("--rolling-returns", required=True)
    parser.add_argument("--fixed-returns", required=True)
    parser.add_argument("--industry-history", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[5])
    parser.add_argument("--seed", type=int, default=42)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
