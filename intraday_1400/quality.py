from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.storage import atomic_json, read_parts


def audit(fail_on_error: bool = False) -> dict:
    asof = read_parts(config.ASOF_PRICE_DIR)
    labels = read_parts(config.LABEL_DIR)
    if asof.empty:
        report = {"ok": False, "error": "no asof rows"}
    else:
        key_columns = ["code", "date", "asof_time"] + (["schema_version"] if "schema_version" in asof else [])
        duplicates = int(asof.duplicated(key_columns).sum())
        current = asof[
            asof["schema_version"] == config.SCHEMA_VERSION
        ].copy() if "schema_version" in asof else asof
        complete_ratio = float(current["is_complete"].fillna(False).mean())
        factor_ok_ratio = float((current["factor_status"] == "ok").mean()) if "factor_status" in current else 0.0
        factor_version_ratio = float(current["factor_version"].fillna("").astype(str).ne("").mean()) if "factor_version" in current else 0.0
        wrong_cutoff = int((current["cutoff_bar_time"].astype(str) != config.CUTOFF_TIME).sum())
        code_count_by_date = current.groupby("date")["code"].nunique()
        latest_code_coverage = (
            float(code_count_by_date.iloc[-1] / code_count_by_date.max())
            if not code_count_by_date.empty and code_count_by_date.max() > 0 else 0.0
        )
        latest_date = current["date"].max()
        latest_complete_ratio = float(
            current.loc[current["date"] == latest_date, "is_complete"].fillna(False).mean()
        ) if pd.notna(latest_date) else 0.0
        minute_columns = [column for column in current if column.startswith("m5_")]
        finite_ratio = float(np.isfinite(current[minute_columns].select_dtypes(include=[np.number])).mean().mean()) if minute_columns else 0.0
        current_labels = labels[
            labels["schema_version"] == config.SCHEMA_VERSION
        ].copy() if not labels.empty and "schema_version" in labels else labels
        label_coverage = (
            float(current_labels["label_entry_bar_present"].fillna(False).mean())
            if not current_labels.empty and "label_entry_bar_present" in current_labels else 0.0
        )
        rebuild_path = config.CHECKPOINT_DIR / "factor_rebuild_required.json"
        rebuild_required = []
        if rebuild_path.exists():
            try:
                import json
                rebuild_required = json.loads(rebuild_path.read_text(encoding="utf-8")).get("codes", [])
            except (OSError, json.JSONDecodeError):
                rebuild_required = ["UNKNOWN"]
        report = {
            "ok": (
                duplicates == 0
                and complete_ratio >= 0.98
                and latest_code_coverage >= 0.98
                and latest_complete_ratio >= 0.98
                and factor_ok_ratio == 1.0
                and factor_version_ratio == 1.0
                and finite_ratio >= 0.95
                and label_coverage >= 0.95
                and not rebuild_required
            ),
            "rows": len(current),
            "codes": int(current["code"].nunique()),
            "dates": int(current["date"].nunique()),
            "start": str(pd.Timestamp(current["date"].min()).date()),
            "end": str(pd.Timestamp(current["date"].max()).date()),
            "duplicates": duplicates,
            "complete_ratio": complete_ratio,
            "factor_ok_ratio": factor_ok_ratio,
            "factor_version_ratio": factor_version_ratio,
            "wrong_cutoff_rows": wrong_cutoff,
            "latest_code_coverage": latest_code_coverage,
            "latest_complete_ratio": latest_complete_ratio,
            "minute_feature_finite_ratio": finite_ratio,
            "label_entry_coverage": label_coverage,
            "factor_rebuild_required": rebuild_required,
        }
    atomic_json(report, config.REPORT_DIR / "quality.json")
    print(f"[intraday1400:quality] {report}", flush=True)
    if fail_on_error and not report.get("ok"):
        raise RuntimeError(f"intraday_1400 quality gate failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit intraday 14:00 partitions")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    audit(fail_on_error=args.fail_on_error)


if __name__ == "__main__":
    main()
