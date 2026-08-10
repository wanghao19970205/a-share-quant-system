from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from intraday_1400 import config
from intraday_1400.storage import atomic_json, atomic_parquet


def publish_realtime_shadow(top_n: int = 30) -> dict:
    """Publish a realtime-compatible shadow snapshot without changing live config."""
    source = config.MODEL_DIR / "intraday_1400_shadow_predictions.parquet"
    if not source.exists():
        raise RuntimeError(f"missing shadow predictions: {source}")
    predictions = pd.read_parquet(source)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").astype("datetime64[ns]")
    latest = predictions[predictions["date"] == predictions["date"].max()].copy()
    latest = latest.sort_values(["rank", "code"]).head(max(int(top_n), 1)).reset_index(drop=True)
    latest["source"] = "intraday_1400_shadow"
    latest["mode"] = "shadow"
    parquet_path = config.MODEL_DIR / "realtime_shadow_predictions.parquet"
    atomic_parquet(latest, parquet_path)

    quant_dir = Path(os.environ.get("QUANT_DATA_DIR", "quant_data"))
    active_path = quant_dir / "active_quant_short_predictions.parquet"
    overlap: dict = {"available": False}
    if active_path.exists():
        active = pd.read_parquet(active_path)
        if {"code", "date"}.issubset(active.columns):
            active["date"] = pd.to_datetime(active["date"], errors="coerce")
            active_latest = active[active["date"] == active["date"].max()].copy()
            active_pred = next(
                (column for column in ("pred", "ensemble_pred", "score") if column in active_latest),
                None,
            )
            if active_pred:
                active_top = set(
                    active_latest.nlargest(max(int(top_n), 1), active_pred)["code"].astype(str).str[:6]
                )
                shadow_top = set(latest["code"].astype(str).str[:6])
                overlap = {
                    "available": True,
                    "count": len(active_top & shadow_top),
                    "ratio": len(active_top & shadow_top) / max(len(shadow_top), 1),
                }
    snapshot = {
        "schema_version": config.SCHEMA_VERSION,
        "mode": "shadow",
        "decision_enabled": False,
        "date": str(pd.Timestamp(latest["date"].max()).date()) if not latest.empty else None,
        "top_n": len(latest),
        "active_overlap": overlap,
        "predictions_file": str(parquet_path),
        "rows": [
            {
                "code": str(row["code"])[:6],
                "rank": int(row["rank"]),
                "pred": float(row["pred"]),
            }
            for _, row in latest.iterrows()
        ],
    }
    atomic_json(snapshot, config.MODEL_DIR / "realtime_shadow_candidates.json")
    print(
        f"[intraday1400:compat] date={snapshot['date']} top_n={len(latest)} "
        f"active_overlap={json.dumps(overlap, ensure_ascii=False)} decision_enabled=false",
        flush=True,
    )
    return snapshot
