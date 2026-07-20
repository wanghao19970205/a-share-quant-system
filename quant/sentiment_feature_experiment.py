"""白名单舆情特征消融实验。

只读取现有量化月度面板和 news_data，并将全部产物写入一个全新的实验目录。
不会修改 factor selection、active predictions 或线上舆情模型。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest, model

_SENTIMENT_PREFIX = "news_sent_"
_DEFAULT_HALF_LIVES = (3, 7, 14)


def read_watchlist(path: Path) -> list[str]:
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() and not line.lstrip().startswith("#") else ""
        if len(token) >= 6 and token[:6].isdigit() and token[:6] not in codes:
            codes.append(token[:6])
    return codes


def parse_publish_time(values: pd.Series) -> pd.Series:
    """Parse mixed date/datetime strings without dropping valid rows on pandas 2.x."""
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return values.map(lambda value: pd.to_datetime(value, errors="coerce"))


def _available_date(frame: pd.DataFrame, cutoff_hour: int) -> pd.Series:
    raw = frame["publish_time"].fillna("").astype(str).str.strip()
    ts = parse_publish_time(raw)
    has_clock = raw.str.contains(r"\d{1,2}:\d{2}", regex=True)
    after_cutoff = has_clock & (ts.dt.hour >= int(cutoff_hour))
    # Date-only announcements/research have no reliable intraday time, so use next day.
    defer = (~has_clock) | after_cutoff
    return ts.dt.normalize() + pd.to_timedelta(defer.astype(int), unit="D")


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _daily_features(news: pd.DataFrame, dates: pd.DatetimeIndex, cutoff_hour: int,
                    half_lives: tuple[int, ...]) -> pd.DataFrame:
    if news.empty or dates.empty:
        return pd.DataFrame(index=dates)
    frame = news.copy()
    frame["available_date"] = _available_date(frame, cutoff_hour)
    frame = frame.dropna(subset=["available_date"])
    if frame.empty:
        return pd.DataFrame(index=dates)

    local = _numeric(frame, "sentiment")
    impact = pd.to_numeric(frame.get("llm_impact"), errors="coerce")
    relevance = _numeric(frame, "llm_relevance").clip(0, 1)
    certainty = _numeric(frame, "llm_certainty").clip(0, 1)
    novelty = _numeric(frame, "llm_novelty").clip(0, 1)
    structured = impact * relevance * certainty * novelty
    hybrid = structured.where(structured.notna(), local) * 0.8 + local * 0.2

    frame["local"] = local
    frame["structured"] = structured
    frame["hybrid"] = hybrid
    frame["positive"] = (local > 0).astype(float)
    frame["negative"] = (local < 0).astype(float)
    frame["annotated"] = impact.notna().astype(float)
    frame["relevance"] = relevance.where(impact.notna())
    frame["certainty"] = certainty.where(impact.notna())
    frame["novelty"] = novelty.where(impact.notna())
    category = frame.get("category", pd.Series("", index=frame.index)).fillna("").astype(str)
    for name in ("announcement", "research", "stock_news"):
        frame[f"category_{name}"] = category.eq(name).astype(float)

    grouped = frame.groupby("available_date", sort=True)
    daily = grouped.agg(
        local_sum=("local", "sum"), structured_sum=("structured", "sum"),
        hybrid_sum=("hybrid", "sum"), article_count=("local", "size"),
        positive_count=("positive", "sum"), negative_count=("negative", "sum"),
        annotated_count=("annotated", "sum"), relevance_sum=("relevance", "sum"),
        certainty_sum=("certainty", "sum"), novelty_sum=("novelty", "sum"),
        announcement_count=("category_announcement", "sum"),
        research_count=("category_research", "sum"),
        stock_news_count=("category_stock_news", "sum"),
    )
    start = min(dates.min(), daily.index.min())
    end = dates.max()
    calendar = pd.date_range(start, end, freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    result = pd.DataFrame(index=calendar)

    for half_life in half_lives:
        alpha = 1.0 - 0.5 ** (1.0 / max(float(half_life), 0.1))
        denominator = daily["article_count"].ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
        for source in ("local", "structured", "hybrid"):
            numerator = daily[f"{source}_sum"].ewm(alpha=alpha, adjust=False).mean()
            result[f"{_SENTIMENT_PREFIX}{source}_decay_{half_life}d"] = (numerator / denominator).fillna(0.0)

    rolling = daily.rolling("7D", min_periods=1).sum()
    count = rolling["article_count"].replace(0, np.nan)
    annotated = rolling["annotated_count"].replace(0, np.nan)
    result[f"{_SENTIMENT_PREFIX}article_count_1d"] = daily["article_count"]
    result[f"{_SENTIMENT_PREFIX}article_count_7d"] = rolling["article_count"]
    result[f"{_SENTIMENT_PREFIX}positive_ratio_7d"] = (rolling["positive_count"] / count).fillna(0.0)
    result[f"{_SENTIMENT_PREFIX}negative_ratio_7d"] = (rolling["negative_count"] / count).fillna(0.0)
    result[f"{_SENTIMENT_PREFIX}llm_coverage_7d"] = (rolling["annotated_count"] / count).fillna(0.0)
    result[f"{_SENTIMENT_PREFIX}relevance_7d"] = (rolling["relevance_sum"] / annotated).fillna(0.0)
    result[f"{_SENTIMENT_PREFIX}certainty_7d"] = (rolling["certainty_sum"] / annotated).fillna(0.0)
    result[f"{_SENTIMENT_PREFIX}novelty_7d"] = (rolling["novelty_sum"] / annotated).fillna(0.0)
    for name in ("announcement", "research", "stock_news"):
        result[f"{_SENTIMENT_PREFIX}{name}_count_7d"] = rolling[f"{name}_count"]
    return result.reindex(dates).fillna(0.0)


def build_feature_panel(panel: pd.DataFrame, news_dir: Path, codes: list[str], cutoff_hour: int = 15,
                        half_lives: tuple[int, ...] = _DEFAULT_HALF_LIVES) -> pd.DataFrame:
    base = panel.copy()
    base["code"] = base["code"].astype(str).str.zfill(6)
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base = base[base["code"].isin(set(codes))].dropna(subset=["code", "date"])
    parts: list[pd.DataFrame] = []
    for code, rows in base.groupby("code", sort=False):
        dates = pd.DatetimeIndex(sorted(rows["date"].unique()))
        path = news_dir / f"{code}.parquet"
        news = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        features = _daily_features(news, dates, cutoff_hour, half_lives)
        features.insert(0, "date", features.index)
        features.insert(0, "code", code)
        parts.append(features.reset_index(drop=True))
    if not parts:
        return base
    sentiment = pd.concat(parts, ignore_index=True)
    return base.merge(sentiment, on=["code", "date"], how="left")


def load_monthly_panel(prepared_dir: Path, codes: list[str], start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    code_set = set(codes)
    for path in sorted(prepared_dir.glob("*.parquet")):
        try:
            month = pd.Period(path.stem, freq="M")
        except ValueError:
            continue
        if month.end_time.normalize() < start or month.start_time.normalize() > end:
            continue
        frame = pd.read_parquet(path)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame[frame["code"].isin(code_set) & frame["date"].between(start, end)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise RuntimeError("指定区间内没有白名单月度面板数据")
    return pd.concat(parts, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def _base_features(panel: pd.DataFrame, horizon: int) -> list[str]:
    target = f"target_ret_{horizon}d"
    banned = {"code", "date", target, f"open_ret_{horizon}d", "buyable_next"}
    return [c for c in panel.columns if c not in banned and not c.startswith(_SENTIMENT_PREFIX)
            and pd.api.types.is_numeric_dtype(panel[c])]


def _standardize_sentiment(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for column in [c for c in out.columns if c.startswith(_SENTIMENT_PREFIX)]:
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        mean = values.groupby(out["date"]).transform("mean")
        std = values.groupby(out["date"]).transform("std").replace(0, np.nan)
        out[column] = ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _purge_boundary_labels(panel: pd.DataFrame, horizon: int, boundaries: list[pd.Timestamp]) -> pd.DataFrame:
    out = panel.copy()
    target = f"target_ret_{horizon}d"
    dates = pd.Index(sorted(out["date"].dropna().unique()))
    for boundary in boundaries:
        eligible = dates[dates <= boundary]
        purge_dates = set(eligible[-max(int(horizon), 1):])
        out.loc[out["date"].isin(purge_dates), target] = np.nan
    return out


def _prediction_metrics(predictions: pd.DataFrame, horizon: int) -> dict:
    target = f"target_ret_{horizon}d"
    if predictions.empty or target not in predictions or "pred" not in predictions:
        return {"n": 0, "rank_ic": None, "direction_accuracy": None}
    valid = predictions[[target, "pred"]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"n": 0, "rank_ic": None, "direction_accuracy": None}
    rank_ic = valid["pred"].corr(valid[target], method="spearman") if len(valid) > 2 else np.nan
    return {
        "n": int(len(valid)),
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else None,
        "direction_accuracy": float(((valid["pred"] > 0) == (valid[target] > 0)).mean()),
    }


def _portfolio_metrics(predictions: pd.DataFrame, horizon: int, top_n: int) -> tuple[dict, pd.DataFrame]:
    returns, _ = backtest.portfolio_from_predictions(
        predictions, horizon=horizon, top_n=top_n, max_weight=1.0 / max(top_n, 1),
        positive_only=False, use_open_fill=False, filter_untradable=False, cost_roundtrip=0.0,
    )
    if returns.empty:
        return {}, returns
    metrics = backtest.evaluate_returns(returns["ret"], periods_per_year=252 / max(int(horizon), 1))
    metrics["avg_turnover"] = float(returns["turnover"].mean())
    return metrics, returns


def _monthly_win_rate(candidate: pd.DataFrame, baseline: pd.DataFrame) -> float:
    if candidate.empty or baseline.empty:
        return 0.0
    merged = candidate[["date", "ret"]].merge(
        baseline[["date", "ret"]], on="date", suffixes=("_candidate", "_baseline"))
    if merged.empty:
        return 0.0
    monthly = merged.set_index("date")[["ret_candidate", "ret_baseline"]].resample("ME").apply(
        lambda values: (1.0 + values).prod() - 1.0)
    return float((monthly["ret_candidate"] > monthly["ret_baseline"]).mean()) if len(monthly) else 0.0


def run_experiment(panel: pd.DataFrame, output_dir: Path, horizon: int, train_end: pd.Timestamp,
                   valid_end: pd.Timestamp, predict_start: pd.Timestamp, top_n: int = 3,
                   models: tuple[str, ...] = ("ridge", "lightgbm_ranker")) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"实验输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    panel = _standardize_sentiment(panel)
    panel = _purge_boundary_labels(panel, horizon, [train_end, valid_end])
    base_features = _base_features(panel, horizon)
    sentiment_features = [c for c in panel.columns if c.startswith(_SENTIMENT_PREFIX)]
    if not sentiment_features:
        raise RuntimeError("面板没有舆情特征")

    panel.to_parquet(output_dir / "white_list_sentiment_panel.parquet", index=False)
    report: dict = {
        "horizon": horizon, "train_end": str(train_end.date()), "valid_end": str(valid_end.date()),
        "predict_start": str(predict_start.date()), "base_feature_count": len(base_features),
        "sentiment_features": sentiment_features, "models": {},
    }
    for model_name in models:
        variants = {"baseline": base_features, "augmented": base_features + sentiment_features}
        variant_results: dict = {}
        variant_returns: dict[str, pd.DataFrame] = {}
        for variant, features in variants.items():
            results = model.train_all(
                panel, horizon=horizon, models=[model_name], factors=features,
                train_end=str(train_end.date()), valid_end=str(valid_end.date()),
                predict_start=str(predict_start.date()), decay_half_life_days=90.0,
                min_weight=0.05, n_estimators=300, early_stopping_rounds=40,
            )
            result = results[0]
            if not result.ok:
                variant_results[variant] = {"ok": False, "message": result.message}
                continue
            predictions = result.predictions
            predictions.to_parquet(output_dir / f"{model_name}_{variant}_predictions.parquet", index=False)
            portfolio, returns = _portfolio_metrics(predictions, horizon, top_n)
            returns.to_parquet(output_dir / f"{model_name}_{variant}_returns.parquet", index=False)
            variant_returns[variant] = returns
            variant_results[variant] = {
                "ok": True, "training_metrics": result.metrics,
                "test_metrics": _prediction_metrics(predictions, horizon), "portfolio": portfolio,
            }
        base = variant_results.get("baseline", {})
        aug = variant_results.get("augmented", {})
        monthly_win = _monthly_win_rate(variant_returns.get("augmented", pd.DataFrame()),
                                        variant_returns.get("baseline", pd.DataFrame()))
        rank_gain = (aug.get("test_metrics", {}).get("rank_ic") or 0.0) - (base.get("test_metrics", {}).get("rank_ic") or 0.0)
        sharpe_gain = (aug.get("portfolio", {}).get("sharpe") or 0.0) - (base.get("portfolio", {}).get("sharpe") or 0.0)
        aug_dd = aug.get("portfolio", {}).get("max_drawdown")
        base_dd = base.get("portfolio", {}).get("max_drawdown")
        drawdown_change = float(aug_dd - base_dd) if aug_dd is not None and base_dd is not None else None
        promoted = bool(base.get("ok") and aug.get("ok") and rank_gain > 0 and sharpe_gain >= 0.10
                        and monthly_win >= 0.55 and drawdown_change is not None and drawdown_change >= -0.02)
        report["models"][model_name] = {
            **variant_results, "rank_ic_gain": rank_gain, "sharpe_gain": sharpe_gain,
            "monthly_win_rate": monthly_win, "drawdown_change": drawdown_change,
            "promoted": promoted,
        }
    comparable = [item for item in report["models"].values()
                  if item.get("baseline", {}).get("ok") and item.get("augmented", {}).get("ok")]
    report["promote_to_full_a_backfill"] = bool(comparable and all(item["promoted"] for item in comparable))
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行隔离的白名单舆情特征消融实验")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--news-dir", required=True)
    parser.add_argument("--watchlist", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-end", required=True)
    parser.add_argument("--predict-start", required=True)
    parser.add_argument("--models", default="ridge,lightgbm_ranker")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--cutoff-hour", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    codes = read_watchlist(Path(args.watchlist))
    if not codes:
        raise SystemExit("白名单为空")
    panel = load_monthly_panel(Path(args.prepared_dir), codes, pd.Timestamp(args.start), pd.Timestamp(args.end))
    panel = build_feature_panel(panel, Path(args.news_dir), codes, cutoff_hour=args.cutoff_hour)
    report = run_experiment(
        panel, output_dir, args.horizon, pd.Timestamp(args.train_end), pd.Timestamp(args.valid_end),
        pd.Timestamp(args.predict_start), top_n=args.top_n,
        models=tuple(item.strip() for item in args.models.split(",") if item.strip()),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
