"""白名单历史舆情信号：时点安全的日级聚合、留出期选型与运行时读取。

新闻库只保存文章发布时已知的信息。本模块按 publish_time 向后衰减聚合，任何日期的
信号都只使用该日及此前文章；模型选择按时间切分（前 75% 选择、后 25% 留出验证）。
它是白名单后模型腿，不进入全 A Ridge/LightGBM 冠军，避免样本选择偏差。
"""
from __future__ import annotations

import json
import math
import os
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


def _articles(code: str) -> pd.DataFrame:
    df = news_store.read_store(code)
    if df.empty:
        return df
    out = df.copy()
    out["publish_dt"] = pd.to_datetime(out["publish_time"], errors="coerce")
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce").fillna(0.0)
    for name, default in (("llm_impact", np.nan), ("llm_relevance", 0.0),
                          ("llm_novelty", 0.0), ("llm_certainty", 0.0)):
        out[name] = pd.to_numeric(out.get(name, default), errors="coerce")
    # LLM 仅为文章语义结构化：影响方向乘相关度、事实确定性和新颖度。
    # 未标注文章绝不伪造为中性 LLM 样本，而是在 source=llm 时不贡献信号。
    out["llm_score"] = (out["llm_impact"] * out["llm_relevance"].clip(0, 1)
                        * out["llm_certainty"].clip(0, 1) * out["llm_novelty"].clip(0, 1))
    return out.dropna(subset=["publish_dt"])


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
    end = pd.Timestamp(asof).normalize() + pd.Timedelta(days=1) if asof is not None else pd.Timestamp.now()
    half_life = float(cfg.get("half_life_days", 7.0))
    lookback = int(cfg.get("lookback_days", max(30, round(half_life * 5))))
    start = end - pd.Timedelta(days=lookback)
    df = _articles(code)
    if df.empty:
        return SentimentSignal(model=str(cfg.get("name", "")), asof=(end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d"),
                               half_life_days=half_life, lookback_days=lookback,
                               blend_weight=float(cfg.get("blend_weight", 0.0)),
                               enabled=bool(cfg.get("enabled", False)), note="窗口内无舆情文章")
    df = df[(df["publish_dt"] >= start) & (df["publish_dt"] < end)].copy()
    if str(cfg.get("signal_source", "lexicon")) == "llm_structured":
        # 回填尚未完成时，缺失标注是未知而非中性，不能进入衰减分母。
        df = df[df["llm_impact"].notna()].copy()
    if df.empty:
        return SentimentSignal(model=str(cfg.get("name", "")), asof=(end - pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d"),
                               half_life_days=half_life, lookback_days=lookback,
                               blend_weight=float(cfg.get("blend_weight", 0.0)),
                               enabled=bool(cfg.get("enabled", False)), note="窗口内无舆情文章")
    cat_w = cfg.get("category_weights") or _DEFAULT_CATEGORY_WEIGHTS
    article_score = _article_sentiment(df, str(cfg.get("signal_source", "lexicon")))
    age = (end - df["publish_dt"]).dt.total_seconds().clip(lower=0) / 86400.0
    decay = np.power(0.5, age / max(half_life, 0.1))
    cw = df["category"].map(cat_w).fillna(0.5).astype(float)
    weight = decay * cw
    denom = float(weight.sum())
    raw = float((article_score * weight).sum() / denom) if denom > 0 else 0.0
    # 分数压到 -2..2；少量文章按样本量收缩，避免单篇标题主导最终研判。
    shrink = 1.0 - math.exp(-len(df) / 5.0)
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


def _candidate_daily(code: str, half_life: float, category_weights: dict,
                     start: pd.Timestamp, end: pd.Timestamp, signal_source: str = "lexicon") -> pd.DataFrame:
    """向量化生成日级衰减信号；每个交易日只使用当日及此前发布文章。"""
    articles = _articles(code)
    prices = _price_forward_returns(code)
    if articles.empty or prices.empty:
        return pd.DataFrame()
    articles = articles[(articles["publish_dt"] >= start - pd.Timedelta(days=max(30, round(half_life * 5))))
                        & (articles["publish_dt"] < end + pd.Timedelta(days=1))].copy()
    if signal_source == "llm_structured":
        # 未回填的文章不是真正的 0 分，纯 LLM 候选中应完全排除。
        articles = articles[articles["llm_impact"].notna()].copy()
    if articles.empty:
        return pd.DataFrame()
    authoritative = _authoritative_calendar()
    end_session = pd.Timestamp(end).normalize()
    if end_session > authoritative[-1]:
        raise RuntimeError(
            f"authoritative trading calendar ends at {authoritative[-1].date()}, before {end_session.date()}"
        )
    article_dates = pd.DatetimeIndex(articles["publish_dt"].dt.normalize())
    positions = authoritative.searchsorted(article_dates, side="left")
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
    num = daily["numerator"].ewm(alpha=alpha, adjust=False).mean()
    den = daily["denominator"].ewm(alpha=alpha, adjust=False).mean()
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


def train(codes: list[str], start, end, output: str | None = None) -> dict:
    """时间切分选舆情衰减模型；留出不达标时产出 enabled=false（运行时不加分）。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    split = start + (end - start) * 0.75
    weight_sets = {
        "balanced": dict(_DEFAULT_CATEGORY_WEIGHTS),
        "official_heavy": {"announcement": 1.3, "research": 1.0, "stock_news": 0.5, "flash": 0.2},
        "media_heavy": {"announcement": 0.8, "research": 1.0, "stock_news": 1.2, "flash": 0.4},
    }
    candidates = []
    frames: dict[tuple, pd.DataFrame] = {}
    # LLM 标注回填是渐进的：训练会同时评估词典、纯结构化和混合特征，
    # 但纯 LLM 必须有足够非零样本，避免少数已标注文章误导选型。
    for signal_source in ("lexicon", "llm_structured", "hybrid"):
        for half_life in (3.0, 7.0, 14.0, 30.0):
            for name, weights in weight_sets.items():
                parts = [_candidate_daily(c, half_life, weights, start, end, signal_source) for c in codes]
                df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
                frames[(signal_source, half_life, name)] = df
                fit = _metrics(df[df["date"] < split]) if not df.empty else _metrics(df)
                candidates.append({"signal_source": signal_source, "half_life_days": half_life,
                                   "weights_name": name, "fit": fit})
    best = max(candidates, key=lambda x: x["fit"]["score"])
    key = (best["signal_source"], best["half_life_days"], best["weights_name"])
    holdout_df = frames[key]
    holdout = _metrics(holdout_df[holdout_df["date"] >= split]) if not holdout_df.empty else _metrics(holdout_df)
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
        "signal_source": best["signal_source"],
        "half_life_days": best["half_life_days"],
        "lookback_days": max(30, round(best["half_life_days"] * 5)),
        "category_weights": weight_sets[best["weights_name"]],
        "fit": best["fit"], "holdout": holdout, "enabled": enabled,
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
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(months=args.months)
    result = train(_read_watchlist(args.watchlist), start, end, args.output or None)
    print(json.dumps({k: result[k] for k in ("name", "enabled", "blend_weight", "fit", "holdout", "validation_note")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
