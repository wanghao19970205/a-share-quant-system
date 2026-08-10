from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_1400 import config
from intraday_1400.storage import atomic_json


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min()) if len(curve) else float("nan")


def _daily_turnover(selected: pd.DataFrame) -> float:
    sets = [set(part["code"].astype(str)) for _, part in selected.groupby("date", sort=True)]
    if len(sets) < 2:
        return float("nan")
    values = [1.0 - len(left & right) / max(len(left | right), 1) for left, right in zip(sets, sets[1:])]
    return float(np.mean(values))


def _bootstrap_ci(values: pd.Series, samples: int = 1000, seed: int = 42) -> list[float] | None:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 10:
        return None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(clean, size=(samples, len(clean)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _metrics(selected: pd.DataFrame, gross_column: str, cost: float) -> dict:
    evaluated = selected.copy()
    missing_targets = int(evaluated[gross_column].isna().sum())
    unsellable_return = float(
        os.environ.get("INTRADAY_1400_UNSELLABLE_RETURN", "-0.10") or -0.10
    )
    evaluated[gross_column] = evaluated[gross_column].fillna(unsellable_return + float(cost))
    daily = evaluated.groupby("date")[gross_column].mean() - float(cost)
    if daily.empty:
        return {"days": 0}
    std = float(daily.std())
    return {
        "days": int(len(daily)),
        "mean_return": float(daily.mean()),
        "median_return": float(daily.median()),
        "win_rate": float((daily > 0).mean()),
        "sharpe": float(daily.mean() / std * np.sqrt(252)) if std > 0 else None,
        "max_drawdown": _max_drawdown(daily),
        "compound_return": float((1.0 + daily).prod() - 1.0),
        "mean_return_ci95": _bootstrap_ci(daily),
        "turnover": _daily_turnover(evaluated),
        "mean_names": float(evaluated.groupby("date")["code"].nunique().mean()),
        "missing_targets": missing_targets,
        "unsellable_return": unsellable_return,
    }


def _load_labels() -> tuple[pd.DataFrame, dict]:
    paths = sorted(config.PREPARED_DIR.glob("????-??.parquet"))
    if not paths:
        raise RuntimeError("no prepared labels")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        target_columns = [column for column in frame.columns if column.startswith("target_")]
        frames.append(frame[["code", "date", "entry_buyable"] + target_columns])
    labels = pd.concat(frames, ignore_index=True)
    labels["code"] = labels["code"].astype(str).str[:6]
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    embedded_cost = float(os.environ.get("INTRADAY_1400_ROUNDTRIP_COST", "0.002") or 0.002)
    target_columns = [
        column for column in labels.columns
        if (column == "target_net_ret_t1" or column.endswith("_proxy_net"))
        and pd.api.types.is_numeric_dtype(labels[column])
    ]
    gross_columns: dict[str, str] = {}
    for target in target_columns:
        gross_column = f"gross__{target}"
        labels[gross_column] = labels[target] + embedded_cost
        gross_columns[target] = gross_column
    labels = labels[labels["entry_buyable"].fillna(False)].drop_duplicates(
        ["code", "date"], keep="last"
    )
    return labels, {"embedded_cost": embedded_cost, "gross_columns": gross_columns}


def _load_prediction(path: Path, score_candidates: tuple[str, ...] = ("pred", "ensemble_pred", "score")) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    score = next((column for column in score_candidates if column in frame.columns), None)
    if score is None:
        raise ValueError(f"no score column in {path}")
    frame = frame[["code", "date", score]].rename(columns={score: "score"})
    frame["code"] = frame["code"].astype(str).str[:6]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.dropna(subset=["code", "date", "score"]).drop_duplicates(["code", "date"], keep="last")


def _evaluate_source(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    causal: bool,
    top_values: tuple[int, ...],
    costs: tuple[float, ...],
    metadata: dict,
) -> dict:
    merged = labels.merge(predictions, on=["code", "date"], how="inner", validate="one_to_one")
    merged["daily_rank"] = merged.groupby("date")["score"].rank(method="first", ascending=False)
    source_report = {
        "available": True,
        "path": str(path),
        "causal_at_1400": causal,
        "overlap_rows": int(len(merged)),
        "overlap_days": int(merged["date"].nunique()),
        "top": {},
    }
    for top_n in top_values:
        selected = merged[merged["daily_rank"] <= int(top_n)].copy()
        source_report["top"][str(top_n)] = {
            target: {
                f"cost_{int(round(cost * 10000))}bp": _metrics(selected, gross_column, cost)
                for cost in costs
            }
            for target, gross_column in metadata["gross_columns"].items()
        }
    return source_report


def _common_labels(labels: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = labels[["code", "date"]].drop_duplicates()
    for frame in predictions.values():
        keys = keys.merge(frame[["code", "date"]], on=["code", "date"], how="inner")
    return labels.merge(keys.drop_duplicates(), on=["code", "date"], how="inner", validate="one_to_one")


def evaluate(
    top_values: tuple[int, ...] = (5, 10, 20, 30),
    costs: tuple[float, ...] = (0.0, 0.001, 0.002, 0.003),
) -> dict:
    labels, metadata = _load_labels()
    manifest_path = config.MODEL_DIR / "intraday_1400_shadow_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    train_end = pd.Timestamp(manifest.get("train_end", labels["date"].min()))
    valid_end = pd.Timestamp(manifest.get("valid_end", labels["date"].max()))
    labels = labels[(labels["date"] > train_end) & (labels["date"] <= valid_end)].copy()

    sources: dict[str, tuple[Path, bool]] = {
        "asof_base": (config.MODEL_DIR / "asof_base_predictions.parquet", True),
        "asof_plus_intraday": (config.MODEL_DIR / "asof_plus_intraday_predictions.parquet", True),
    }
    quant_dir = Path(os.environ.get("QUANT_DATA_DIR", "quant_data"))
    sources["close_active_reference"] = (
        quant_dir / "active_quant_short_predictions.parquet",
        False,
    )
    report = {
        "schema_version": config.SCHEMA_VERSION,
        "train_end": str(train_end.date()),
        "valid_end": str(valid_end.date()),
        "label_rows": int(len(labels)),
        "label_days": int(labels["date"].nunique()),
        "costs": list(costs),
        "top_values": list(top_values),
        "sources": {},
        "common_universe": {},
        **metadata,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, (path, causal) in sources.items():
        if not path.exists():
            report["sources"][name] = {"available": False, "path": str(path)}
            continue
        predictions = _load_prediction(path)
        loaded[name] = predictions
        report["sources"][name] = _evaluate_source(
            labels, predictions, path, causal, top_values, costs, metadata,
        )

    common_groups = {
        "causal_common": ["asof_base", "asof_plus_intraday"],
        "all_sources_common": list(sources),
    }
    for group_name, names in common_groups.items():
        if not all(name in loaded for name in names):
            report["common_universe"][group_name] = {"available": False, "sources": names}
            continue
        group_predictions = {name: loaded[name] for name in names}
        common = _common_labels(labels, group_predictions)
        group_report = {
            "available": True,
            "sources": names,
            "common_rows": int(len(common)),
            "common_days": int(common["date"].nunique()),
            "results": {},
        }
        for name in names:
            path, causal = sources[name]
            group_report["results"][name] = _evaluate_source(
                common, loaded[name], path, causal, top_values, costs, metadata,
            )
        report["common_universe"][group_name] = group_report
    atomic_json(report, config.REPORT_DIR / "fair_daily_topn_evaluation.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair daily TopN evaluation for intraday 14:00 models")
    parser.add_argument("--top", default="5,10,20,30")
    parser.add_argument("--costs-bp", default="0,10,20,30")
    args = parser.parse_args()
    top_values = tuple(int(value) for value in args.top.split(",") if int(value) > 0)
    costs = tuple(float(value) / 10000.0 for value in args.costs_bp.split(","))
    evaluate(top_values, costs)


if __name__ == "__main__":
    main()
