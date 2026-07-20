"""Build a CatBoost Ranker shadow leg from existing walk-forward factor windows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from quant import full_train_batched as batched
from quant import model


def build_shadow(
    quant_dir: Path,
    source_predictions: Path,
    factor_audit: Path,
    prepared_dir: Path,
    output_dir: Path,
    threads: int = 4,
    max_train_rows: int = 300_000,
    n_estimators: int = 200,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 40,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_dir = output_dir / "windows"
    window_dir.mkdir(exist_ok=True)
    recipe = {
        "source": str(source_predictions),
        "audit": str(factor_audit),
        "prepared_dir": str(prepared_dir),
        "threads": int(threads),
        "max_train_rows": int(max_train_rows),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "early_stopping_rounds": int(early_stopping_rounds),
    }
    recipe_path = output_dir / "recipe.json"
    if recipe_path.exists():
        existing = json.loads(recipe_path.read_text(encoding="utf-8"))
        if existing != recipe:
            raise RuntimeError("CatBoost shadow recipe changed; use a new output directory")
    else:
        recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    base = pd.read_parquet(source_predictions)
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base["code"] = base["code"].astype(str)
    audit = pd.read_parquet(factor_audit)
    for column in ("test_start", "train_start", "train_end"):
        audit[column] = pd.to_datetime(audit[column], errors="coerce").dt.normalize()
    starts = sorted(audit["test_start"].dropna().unique())
    parts = []
    windows = []
    for index, raw_start in enumerate(starts):
        current = pd.Timestamp(raw_start)
        next_start = pd.Timestamp(starts[index + 1]) if index + 1 < len(starts) else base["date"].max() + pd.Timedelta(days=1)
        window_audit = audit[audit["test_start"] == current]
        factors = window_audit.loc[window_audit["selected"].astype(bool), "factor"].astype(str).tolist()
        train_start = pd.Timestamp(window_audit["train_start"].iloc[0])
        train_end = pd.Timestamp(window_audit["train_end"].iloc[0])
        valid_start = current - pd.DateOffset(months=1)
        cached_prediction = window_dir / f"{current:%Y-%m-%d}.parquet"
        cached_report = window_dir / f"{current:%Y-%m-%d}.json"
        if cached_prediction.exists() and cached_report.exists():
            prediction = pd.read_parquet(cached_prediction)
            window = json.loads(cached_report.read_text(encoding="utf-8"))
            parts.append(prediction)
            windows.append(window)
            print(
                f"[catboost-shadow] window={index + 1}/{len(starts)} "
                f"test={current.date()} cache-hit rows={len(prediction)}",
                flush=True,
            )
            continue
        columns = ["code", "date", "target_ret_3d"] + factors
        frame = batched._load_window(prepared_dir, train_start, next_start, columns=columns, cache=None)  # noqa: SLF001
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        valid_dates = frame.loc[(frame["date"] >= valid_start) & (frame["date"] < current), "date"]
        valid_end = batched._purged_end(valid_dates, current, 3)  # noqa: SLF001
        if valid_end is None:
            raise RuntimeError(f"validation boundary is empty: {current.date()}")
        result = model.train_catboost_ranker(
            frame,
            factors,
            horizon=3,
            train_end=str(train_end.date()),
            valid_end=str(valid_end.date()),
            predict_start=str(current.date()),
            decay_half_life_days=60.0,
            min_weight=0.03,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=threads,
            max_train_rows=max_train_rows,
        )
        if not result.ok:
            raise RuntimeError(f"CatBoost window {current.date()} failed: {result.message}")
        prediction = result.predictions[
            (result.predictions["date"] >= current) & (result.predictions["date"] < next_start)
        ][["code", "date", "pred"]].rename(columns={"pred": "catboost_pred"})
        window = {
            "test_start": str(current.date()),
            "test_end": str((next_start - pd.Timedelta(days=1)).date()),
            "train_end": str(train_end.date()),
            "valid_end": str(valid_end.date()),
            "factors": len(factors),
            "rows": len(prediction),
            "metrics": result.metrics,
        }
        temporary_prediction = cached_prediction.with_suffix(".tmp.parquet")
        prediction.to_parquet(temporary_prediction, index=False)
        os.replace(temporary_prediction, cached_prediction)
        temporary_report = cached_report.with_suffix(".tmp.json")
        temporary_report.write_text(json.dumps(window, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_report, cached_report)
        parts.append(prediction)
        windows.append(window)
        print(f"[catboost-shadow] window={index + 1}/{len(starts)} test={current.date()} rows={len(prediction)}", flush=True)
    leg = pd.concat(parts, ignore_index=True)
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce").dt.normalize()
    leg["code"] = leg["code"].astype(str)
    merged = base.merge(leg, on=["code", "date"], how="left", validate="one_to_one")
    merged["catboost_z"] = merged.groupby("date")["catboost_pred"].transform(
        lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) else 0.0
    )
    temporary = output_dir / "predictions.tmp.parquet"
    merged.to_parquet(temporary, index=False)
    os.replace(temporary, output_dir / "predictions.parquet")
    report = {
        "source": source_predictions.name,
        "rows": len(merged),
        "date_min": str(merged["date"].min().date()),
        "date_max": str(merged["date"].max().date()),
        "coverage": float(merged["catboost_pred"].notna().mean()),
        "threads": int(threads),
        "max_train_rows": int(max_train_rows),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "windows": windows,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CatBoost shadow predictions")
    parser.add_argument("--quant-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    args = parser.parse_args()
    report = build_shadow(
        Path(args.quant_dir),
        Path(args.source),
        Path(args.audit),
        Path(args.prepared_dir),
        Path(args.output_dir),
        threads=args.threads,
        max_train_rows=args.max_train_rows,
    )
    print(json.dumps({key: report[key] for key in ("rows", "date_min", "date_max", "coverage")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
