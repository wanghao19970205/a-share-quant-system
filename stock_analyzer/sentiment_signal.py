"""白名单历史舆情信号：时点安全的日级聚合、留出期选型与运行时读取。

新闻库只保存文章发布时已知的信息。本模块按 publish_time 向后衰减聚合，任何日期的
信号都只使用该日及此前文章；模型选择按时间切分（前 75% 选择、后 25% 留出验证）。
它是白名单后模型腿，不进入全 A Ridge/LightGBM 冠军，避免样本选择偏差。
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_analyzer import data, news_store

_DEFAULT_CATEGORY_WEIGHTS = {
    "announcement": 1.0,
    "research": 1.0,
    "stock_news": 0.8,
    "flash": 0.4,
}
_MODEL_FILE = "sentiment_model.json"


@dataclass
class SentimentSignal:
    score: float = 0.0
    raw_score: float = 0.0
    article_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    lookback_days: int = 0
    half_life_days: float = 0.0
    blend_weight: float = 0.0
    model: str = ""
    asof: str = ""
    enabled: bool = False
    note: str = ""
    available: bool = False


def _model_path() -> str:
    return os.path.join(news_store._dir(), _MODEL_FILE)  # noqa: SLF001


def load_model() -> dict:
    try:
        with open(_model_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _strict_publish_timestamp(value) -> pd.Timestamp:
    """Return conservative Shanghai-local availability for a published article."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    text = str(value).strip()
    if isinstance(value, str):
        has_explicit_time = bool(re.search(r"(?:T|\s)\d{1,2}:\d{2}", text))
    else:
        has_explicit_time = timestamp != timestamp.normalize()
    if not has_explicit_time:
        return timestamp.normalize() + pd.Timedelta(hours=15)
    return timestamp


def _articles(code: str) -> pd.DataFrame:
    df = news_store.read_store(code)
    if df.empty:
        return df
    out = df.copy()
    out["publish_dt"] = pd.to_datetime(out["publish_time"], errors="coerce")
    out["strict_publish_dt"] = out["publish_time"].map(_strict_publish_timestamp)
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce").fillna(0.0)
    for name, default in (("llm_impact", np.nan), ("llm_relevance", 0.0),
                          ("llm_novelty", 0.0), ("llm_certainty", 0.0)):
        out[name] = pd.to_numeric(out.get(name, default), errors="coerce")
    # LLM 仅为文章语义结构化：影响方向乘相关度、事实确定性和新颖度。
    # 未标注文章绝不伪造为中性 LLM 样本，而是在 source=llm 时不贡献信号。
    out["llm_score"] = (out["llm_impact"] * out["llm_relevance"].clip(0, 1)
                        * out["llm_certainty"].clip(0, 1) * out["llm_novelty"].clip(0, 1))
    return out[
        out["publish_dt"].notna() | out["strict_publish_dt"].notna()
    ].copy()


def _article_sentiment(df: pd.DataFrame, source: str) -> pd.Series:
    """Return one point-in-time article score without replacing the raw lexicon label."""
    source = str(source or "lexicon")
    local = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)
    structured = pd.to_numeric(df["llm_score"], errors="coerce")
    if source == "llm_structured":
        return structured.fillna(0.0)
    if source == "hybrid":
        # LLM 覆盖时以语义标注为主；未覆盖历史仍沿用词典，使回填可渐进完成。
        return structured.where(structured.notna(), local) * 0.8 + local * 0.2
    return local


def score_at(code: str, asof=None, model: dict | None = None) -> SentimentSignal:
    """计算 asof 时点的衰减舆情；严格排除 asof 之后发布的文章。"""
    cfg = model if model is not None else load_model()
    if not cfg:
        return SentimentSignal(note="尚未训练舆情模型")
    strict_announcement_lag = bool(cfg.get("strict_announcement_lag", False))
    if strict_announcement_lag:
        end = (
            _strict_publish_timestamp(asof)
            if asof is not None
            else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        )
        if pd.isna(end):
            return SentimentSignal(
                model=str(cfg.get("name", "")), enabled=bool(cfg.get("enabled", False)),
                note="无效 asof",
            )
    else:
        end = pd.Timestamp(asof).normalize() + pd.Timedelta(days=1) if asof is not None else pd.Timestamp.now()
    half_life = float(cfg.get("half_life_days", 7.0))
    lookback = int(cfg.get("lookback_days", max(30, round(half_life * 5))))
    df = _articles(code)
    publish_col = "strict_publish_dt" if strict_announcement_lag else "publish_dt"
    if df.empty:
        return SentimentSignal(
            model=str(cfg.get("name", "")),
            asof=(end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d"),
            half_life_days=half_life, lookback_days=lookback,
            blend_weight=float(cfg.get("blend_weight", 0.0)),
            enabled=bool(cfg.get("enabled", False)), note="窗口内无舆情文章",
        )
    if strict_announcement_lag:
        df = df[df[publish_col] < end].copy()
        authoritative = _authoritative_calendar()
        end_date = end.normalize()
        if end_date > authoritative[-1]:
            raise RuntimeError(
                f"authoritative trading calendar ends at {authoritative[-1].date()}, before {end_date.date()}"
            )
        end_position = int(authoritative.searchsorted(end_date, side="right") - 1)
        positions = _effective_article_session_positions(
            pd.DatetimeIndex(df[publish_col]), authoritative,
        )
        age = end_position - positions
        available = (positions >= 0) & (age >= 0)
        df = df.loc[available].copy()
        age = age[available].astype(float)
    else:
        start = end - pd.Timedelta(days=lookback)
        df = df[(df[publish_col] >= start) & (df[publish_col] < end)].copy()
        age = (end - df[publish_col]).dt.total_seconds().clip(lower=0) / 86400.0
    if str(cfg.get("signal_source", "lexicon")) == "llm_structured":
        # 回填尚未完成时，缺失标注是未知而非中性，不能进入衰减分母。
        structured = df["llm_impact"].notna().to_numpy()
        df = df.loc[structured].copy()
        age = np.asarray(age)[structured]
    if df.empty:
        return SentimentSignal(model=str(cfg.get("name", "")), asof=(end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d"),
                               half_life_days=half_life, lookback_days=lookback,
                               blend_weight=float(cfg.get("blend_weight", 0.0)),
                               enabled=bool(cfg.get("enabled", False)), note="窗口内无舆情文章")
    cat_w = cfg.get("category_weights") or _DEFAULT_CATEGORY_WEIGHTS
    article_score = _article_sentiment(df, str(cfg.get("signal_source", "lexicon")))
    decay = np.power(0.5, np.asarray(age, dtype=float) / max(half_life, 0.1))
    cw = df["category"].map(cat_w).fillna(0.5).astype(float)
    weight = decay * cw
    denom = float(weight.sum())
    raw = float((article_score * weight).sum() / denom) if denom > 0 else 0.0
    # 分数压到 -2..2；严格模式的样本量窗口按交易会话计数。
    shrink_count = (
        int((np.asarray(age) < lookback).sum())
        if strict_announcement_lag else len(df)
    )
    shrink = 1.0 - math.exp(-shrink_count / 5.0)
    score = max(-2.0, min(2.0, raw * shrink))
    return SentimentSignal(
        score=round(score, 3), raw_score=round(raw, 3), article_count=len(df),
        positive_count=int((df["sentiment"] > 0).sum()),
        negative_count=int((df["sentiment"] < 0).sum()),
        lookback_days=lookback, half_life_days=half_life,
        blend_weight=float(cfg.get("blend_weight", 0.0)), model=str(cfg.get("name", "")),
        asof=(end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d"),
        enabled=bool(cfg.get("enabled", False)), available=True,
        note=str(cfg.get("validation_note", "")),
    )


def _price_forward_returns(code: str) -> pd.DataFrame:
    qd = os.environ.get("QUANT_DATA_DIR", "quant_data")
    p = Path(qd) / "price" / f"{data._normalize_symbol(code)}.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        px = pd.read_parquet(p, columns=["date", "close"])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna().sort_values("date")
    px["ret_1d"] = px["close"].shift(-1) / px["close"] - 1.0
    px["ret_3d"] = px["close"].shift(-3) / px["close"] - 1.0
    return px[["date", "ret_1d", "ret_3d"]]


def _authoritative_calendar() -> pd.DatetimeIndex:
    path = Path(os.environ.get("QUANT_DATA_DIR", "quant_data")) / "trading_calendar.parquet"
    if not path.exists():
        raise RuntimeError(f"authoritative trading calendar unavailable: {path}")
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["date"]:
        raise ValueError("authoritative trading calendar must contain only the date column")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce")).normalize()
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("authoritative trading calendar must be non-empty, unique, and increasing")
    return dates


def _effective_article_session_positions(
    article_times: pd.DatetimeIndex,
    authoritative: pd.DatetimeIndex,
) -> np.ndarray:
    """Map strict article availability to authoritative exchange sessions."""
    positions = np.full(len(article_times), -1, dtype=int)
    valid_time = ~article_times.isna()
    if not valid_time.any():
        return positions
    valid_indices = np.flatnonzero(valid_time)
    times = article_times[valid_time]
    dates = times.normalize()
    candidate_positions = authoritative.searchsorted(dates, side="left")
    in_range = candidate_positions < len(authoritative)
    same_session = np.zeros(len(candidate_positions), dtype=bool)
    same_session[in_range] = (
        authoritative.take(candidate_positions[in_range]).to_numpy()
        == dates[in_range].to_numpy()
    )
    at_or_after_close = times.hour * 60 + times.minute >= 15 * 60
    candidate_positions = candidate_positions + (
        same_session & at_or_after_close
    ).astype(int)
    resolved = candidate_positions < len(authoritative)
    positions[valid_indices[resolved]] = candidate_positions[resolved]
    return positions


def _candidate_daily(code: str, half_life: float, category_weights: dict,
                     start: pd.Timestamp, end: pd.Timestamp, signal_source: str = "lexicon",
                     strict_announcement_lag: bool = False,
                     articles_frame: pd.DataFrame | None = None,
                     prices_frame: pd.DataFrame | None = None,
                     authoritative_calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """向量化生成日级衰减信号；严格模式将收盘后/日期级文章顺延。"""
    articles = articles_frame.copy() if articles_frame is not None else _articles(code)
    prices = prices_frame.copy() if prices_frame is not None else _price_forward_returns(code)
    if articles.empty or prices.empty:
        return pd.DataFrame()
    publish_col = "strict_publish_dt" if strict_announcement_lag else "publish_dt"
    if strict_announcement_lag:
        articles = articles[
            articles[publish_col] < end + pd.Timedelta(days=1)
        ].copy()
    else:
        articles = articles[(articles[publish_col] >= start - pd.Timedelta(days=max(30, round(half_life * 5))))
                            & (articles[publish_col] < end + pd.Timedelta(days=1))].copy()
    if signal_source == "llm_structured":
        # 未回填的文章不是真正的 0 分，纯 LLM 候选中应完全排除。
        articles = articles[articles["llm_impact"].notna()].copy()
    if articles.empty:
        return pd.DataFrame()
    authoritative = (
        authoritative_calendar
        if authoritative_calendar is not None else _authoritative_calendar()
    )
    end_session = pd.Timestamp(end).normalize()
    if end_session > authoritative[-1]:
        raise RuntimeError(
            f"authoritative trading calendar ends at {authoritative[-1].date()}, before {end_session.date()}"
        )
    article_times = pd.DatetimeIndex(articles[publish_col])
    if strict_announcement_lag:
        positions = _effective_article_session_positions(
            article_times, authoritative,
        )
        valid = positions >= 0
    else:
        positions = authoritative.searchsorted(article_times.normalize(), side="left")
        valid = positions < len(authoritative)
    articles = articles.loc[valid].copy()
    positions = positions[valid]
    if articles.empty:
        return pd.DataFrame()
    articles["date"] = authoritative.take(positions).to_numpy(dtype="datetime64[ns]")
    articles = articles[articles["date"] <= end_session].copy()
    if articles.empty:
        return pd.DataFrame()
    articles["cat_weight"] = articles["category"].map(category_weights).fillna(0.5).astype(float)
    articles["signal_value"] = _article_sentiment(articles, signal_source)
    articles["weighted_sentiment"] = articles["signal_value"] * articles["cat_weight"]
    daily = articles.groupby("date").agg(
        numerator=("weighted_sentiment", "sum"), denominator=("cat_weight", "sum"),
        article_count=("signal_value", "size"),
    )
    sessions = authoritative[(authoritative >= daily.index.min()) & (authoritative <= end_session)]
    daily = daily.reindex(sessions, fill_value=0.0)
    # One EWM step now means one exchange session, including across holidays/weekends.
    alpha = 1.0 - 0.5 ** (1.0 / max(half_life, 0.1))
    num = daily["numerator"].ewm(
        alpha=alpha, adjust=strict_announcement_lag,
    ).mean()
    den = daily["denominator"].ewm(
        alpha=alpha, adjust=strict_announcement_lag,
    ).mean()
    score = (num / den.replace(0.0, np.nan)).fillna(0.0)
    # The article-count shrinkage window is also measured in exchange sessions.
    count = daily["article_count"].rolling(max(30, round(half_life * 5)), min_periods=1).sum()
    shrink = 1.0 - np.exp(-count / 5.0)
    signal = (score * shrink).clip(-2.0, 2.0)
    sig = pd.DataFrame({"date": sessions, "sentiment_score": signal.to_numpy()})
    out = prices[(prices["date"] >= start) & (prices["date"] <= end)].merge(sig, on="date", how="left")
    out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
    out["code"] = code
    return out


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "ic_1d": None, "ic_3d": None, "direction_1d": None, "score": -999.0}
    x = pd.to_numeric(df["sentiment_score"], errors="coerce")
    r1 = pd.to_numeric(df["ret_1d"], errors="coerce")
    r3 = pd.to_numeric(df["ret_3d"], errors="coerce")
    valid1 = x.notna() & r1.notna() & (x != 0)
    valid3 = x.notna() & r3.notna() & (x != 0)
    ic1 = x[valid1].corr(r1[valid1], method="spearman") if valid1.sum() >= 20 else np.nan
    ic3 = x[valid3].corr(r3[valid3], method="spearman") if valid3.sum() >= 20 else np.nan
    direction = ((x[valid1] > 0) == (r1[valid1] > 0)).mean() if valid1.any() else np.nan
    vals = [v for v in (ic1, ic3) if pd.notna(v)]
    score = float(np.mean(vals)) if vals else -999.0
    return {"n": int(valid1.sum()), "ic_1d": None if pd.isna(ic1) else float(ic1),
            "ic_3d": None if pd.isna(ic3) else float(ic3),
            "direction_1d": None if pd.isna(direction) else float(direction), "score": score}


def train(codes: list[str], start, end, output: str | None = None,
          strict_announcement_lag: bool = False,
          strict_label_purge: bool = False) -> dict:
    """时间切分选舆情衰减模型；留出不达标时产出 enabled=false（运行时不加分）。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    split = start + (end - start) * 0.75
    fit_end = split
    holdout_start = split
    purge_sessions = 0
    if strict_label_purge:
        authoritative = _authoritative_calendar()
        split_position = int(authoritative.searchsorted(split, side="left"))
        if split_position >= len(authoritative):
            raise RuntimeError("strict sentiment split exceeds authoritative trading calendar")
        purge_sessions = 3
        holdout_start = authoritative[split_position]
        fit_end = authoritative[max(split_position - purge_sessions, 0)]
    weight_sets = {
        "balanced": dict(_DEFAULT_CATEGORY_WEIGHTS),
        "official_heavy": {"announcement": 1.3, "research": 1.0, "stock_news": 0.5, "flash": 0.2},
        "media_heavy": {"announcement": 0.8, "research": 1.0, "stock_news": 1.2, "flash": 0.4},
    }
    candidates = []
    frames: dict[tuple, pd.DataFrame] = {}
    authoritative = _authoritative_calendar()
    article_cache = {code: _articles(code) for code in codes}
    price_cache = {code: _price_forward_returns(code) for code in codes}
    # LLM 标注回填是渐进的：训练会同时评估词典、纯结构化和混合特征，
    # 但纯 LLM 必须有足够非零样本，避免少数已标注文章误导选型。
    for signal_source in ("lexicon", "llm_structured", "hybrid"):
        for half_life in (3.0, 7.0, 14.0, 30.0):
            for name, weights in weight_sets.items():
                parts = [
                    _candidate_daily(
                        c, half_life, weights, start, end, signal_source,
                        strict_announcement_lag=strict_announcement_lag,
                        articles_frame=article_cache[c],
                        prices_frame=price_cache[c],
                        authoritative_calendar=authoritative,
                    )
                    for c in codes
                ]
                df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
                frames[(signal_source, half_life, name)] = df
                fit = _metrics(df[df["date"] < fit_end]) if not df.empty else _metrics(df)
                candidates.append({"signal_source": signal_source, "half_life_days": half_life,
                                   "weights_name": name, "fit": fit})
    best = max(candidates, key=lambda x: x["fit"]["score"])
    key = (best["signal_source"], best["half_life_days"], best["weights_name"])
    holdout_df = frames[key]
    holdout = _metrics(holdout_df[holdout_df["date"] >= holdout_start]) if not holdout_df.empty else _metrics(holdout_df)
    holdout_ic = np.mean([v for v in (holdout["ic_1d"], holdout["ic_3d"]) if v is not None]) if any(
        v is not None for v in (holdout["ic_1d"], holdout["ic_3d"])) else -1.0
    enabled = bool(holdout["n"] >= 100 and holdout_ic >= 0.01 and (holdout["direction_1d"] or 0) >= 0.50)
    # 验证强度映射到低权重后模型腿，绝不超过 0.15。
    blend_weight = min(0.15, max(0.0, float(holdout_ic) * 1.5)) if enabled else 0.0
    validation_note = (f"留出期 n={holdout['n']}，1日IC={holdout['ic_1d']}，3日IC={holdout['ic_3d']}，"
                       f"方向胜率={holdout['direction_1d']}；{'启用' if enabled else '未过门槛，不加分'}")
    result = {
        "version": 1, "name": f"decay_{best['weights_name']}_hl{int(best['half_life_days'])}",
        "trained_at": pd.Timestamp.now().isoformat(), "train_start": str(start.date()),
        "train_end": str(end.date()), "split_date": str(pd.Timestamp(split).date()),
        "fit_end_exclusive": str(pd.Timestamp(fit_end).date()),
        "holdout_start": str(pd.Timestamp(holdout_start).date()),
        "strict_label_purge": bool(strict_label_purge),
        "purge_sessions": int(purge_sessions),
        "signal_source": best["signal_source"],
        "half_life_days": best["half_life_days"],
        "lookback_days": max(30, round(best["half_life_days"] * 5)),
        "category_weights": weight_sets[best["weights_name"]],
        "fit": best["fit"], "holdout": holdout, "enabled": enabled,
        "strict_announcement_lag": bool(strict_announcement_lag),
        "blend_weight": round(blend_weight, 4), "validation_note": validation_note,
        "candidates": candidates,
    }
    path = output or _model_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _read_watchlist(path: str) -> list[str]:
    codes = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return codes
    for line in lines:
        token = line.strip().split()[0] if line.strip() and not line.lstrip().startswith("#") else ""
        if len(token) >= 6 and token[:6].isdigit() and token[:6] not in codes:
            codes.append(token[:6])
    return codes


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="训练白名单历史舆情衰减模型")
    ap.add_argument("--watchlist", default=os.path.join(os.environ.get("SNAPSHOT_DIR", "snapshots"), "watchlist.txt"))
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--end-date", default="", help="inclusive research end date; default is today")
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--strict-announcement-lag", action="store_true",
        help="lag post-close and date-only articles to the next authoritative session",
    )
    ap.add_argument(
        "--strict-label-purge", action="store_true",
        help="purge the last three fit sessions before the holdout split",
    )
    args = ap.parse_args()
    end = (
        pd.Timestamp(args.end_date).normalize()
        if args.end_date else pd.Timestamp.now().normalize()
    )
    start = end - pd.DateOffset(months=args.months)
    result = train(
        _read_watchlist(args.watchlist), start, end, args.output or None,
        strict_announcement_lag=args.strict_announcement_lag,
        strict_label_purge=args.strict_label_purge,
    )
    print(json.dumps({k: result[k] for k in ("name", "enabled", "blend_weight", "fit", "holdout", "validation_note")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
