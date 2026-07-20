"""候选栏 Top8：白名单量化优选 + 大模型综合评估排序。

- top_candidates(): 从白名单量化打分表按池内排名取前 N（默认 8）。
- evaluate_top(): 对候选逐只做多维综合研判（技术/外围/板块/新闻/资金/基本面/量化 -> prediction.predict），
  再按大模型给出的方向+信心排序。大模型随机选择规则与快照一致（llm.get_random_model()）。
- 评估较慢（每只多次网络+Qwen 调用），通过 progress 回传完成度供 UI 展示。

只做倾向性研判，不构成投资建议。
"""
from __future__ import annotations

import datetime as dt
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from stock_analyzer import (advisor, broker_extra, data, indicators, llm, moneyflow,
                            news, overseas, prediction, quant_signal, sectors, sentiment_signal)


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


# 大模型综合分：方向×信心为主，规则综合分(-2~2)细分，量化分做末位兜底。
_CONF_WEIGHT = {"高": 1.0, "中": 0.6, "低": 0.3}


def _rank_score(direction: str, confidence: str, composite, quant_score) -> float:
    d = str(direction or "")
    sign = 1.0 if "多" in d else (-1.0 if "空" in d else 0.0)
    w = _CONF_WEIGHT.get(str(confidence or ""), 0.5)
    try:
        comp = float(composite)
    except Exception:  # noqa: BLE001
        comp = 0.0
    try:
        qs = float(quant_score)
    except Exception:  # noqa: BLE001
        qs = 0.0
    return sign * w * 10.0 + comp + qs * 0.01


def top_candidates(n: int = 8, profile: str | None = None, style: str = "short_1_3") -> list[str]:
    """白名单量化打分表按池内排名取前 N 个 6 位代码。"""
    frame = _safe(lambda: quant_signal.watchlist_frame(profile=profile, style=style))
    if frame is None or frame.empty or "code" not in frame.columns:
        return []
    sub = frame.copy()
    if "watch_rank" in sub.columns:
        sub["watch_rank"] = pd.to_numeric(sub["watch_rank"], errors="coerce")
        sub = sub.sort_values("watch_rank", ascending=True, na_position="last")
    elif "pred" in sub.columns:
        sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce")
        sub = sub.sort_values("pred", ascending=False)
    codes: list[str] = []
    for raw in sub["code"].dropna().astype(str):
        c = data._normalize_symbol(raw)
        if c and c not in codes:
            codes.append(c)
        if len(codes) >= n:
            break
    return codes


def _normalize_market_frame(frame: pd.DataFrame | None, code: str) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame or "close" not in frame:
        return pd.DataFrame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out[out["date"] <= pd.Timestamp(dt.date.today())]
    if "code" not in out:
        out.insert(0, "code", code)
    else:
        out["code"] = code
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _merged_market_frame(code: str, days: int = 250) -> tuple[pd.DataFrame, str]:
    """Merge network daily bars with the newer intraday bar from quant warehouse."""
    network = _normalize_market_frame(data.fetch_daily(code, days=days), code)
    quant_dir = os.environ.get("QUANT_DATA_DIR", "")
    local = pd.DataFrame()
    if quant_dir:
        path = os.path.join(quant_dir, "price", f"{code}.parquet")
        if os.path.exists(path):
            try:
                local = _normalize_market_frame(pd.read_parquet(path), code)
            except Exception:  # noqa: BLE001 损坏或正被原子替换时保留网络行情
                local = pd.DataFrame()
    if local.empty:
        return network, "行情接口"
    if network.empty:
        return local.tail(days).reset_index(drop=True), "午间/收盘日更仓"
    local_latest = local["date"].max()
    network_latest = network["date"].max()
    if local_latest < network_latest:
        return network, "行情接口"
    columns = list(dict.fromkeys(network.columns.tolist() + local.columns.tolist()))
    merged = pd.concat([network.reindex(columns=columns), local.reindex(columns=columns)], ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates("date", keep="last").tail(days).reset_index(drop=True)
    source = "午间/收盘日更仓" if local_latest >= network_latest else "行情接口"
    return merged, source


def _sanitize_limit_claims(text: str, name: str, latest: pd.Series) -> str:
    """Remove same-day limit claims unless the quote provides matching evidence."""
    value = str(text or "")
    pct = pd.to_numeric(pd.Series([latest.get("pct_change")]), errors="coerce").iloc[0]
    close = pd.to_numeric(pd.Series([latest.get("close")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([latest.get("high")]), errors="coerce").iloc[0]
    low = pd.to_numeric(pd.Series([latest.get("low")]), errors="coerce").iloc[0]
    threshold = 4.8 if "ST" in str(name).upper() else 9.5
    limit_up = (pd.notna(pct) and pd.notna(close) and pd.notna(high)
                and float(pct) >= threshold and abs(float(close) - float(high)) <= 0.005)
    limit_down = (pd.notna(pct) and pd.notna(close) and pd.notna(low)
                  and float(pct) <= -threshold and abs(float(close) - float(low)) <= 0.005)
    if not limit_up:
        value = value.replace("今日涨停价", "参考价").replace("当日涨停价", "参考价")
        value = value.replace("今日涨停", "今日上涨").replace("当日涨停", "当日上涨")
    if not limit_down:
        replacement = "当日下跌" if pd.notna(pct) and float(pct) < 0 else "当日波动"
        value = value.replace("今日跌停价", "参考价").replace("当日跌停价", "参考价")
        value = value.replace("今日跌停", replacement).replace("当日跌停", replacement)
    return value


def evaluate_candidate(symbol: str, key: str = "", model: str = "", base_url: str = "",
                       profile: str | None = None, style: str = "short_1_3",
                       broker_retry: int = 6) -> dict:
    """单只候选的多维综合研判，返回排序所需字段（不落盘）。"""
    code = data._normalize_symbol(symbol)
    name = _safe(lambda: data.get_stock_name(code)) or code
    try:
        market_frame, quote_source = _merged_market_frame(code, days=250)
        if len(market_frame) < 2:
            raise ValueError("有效行情不足两条")
        df = indicators.compute_all(market_frame)
    except Exception as e:  # noqa: BLE001
        return {"code": code, "name": name, "available": False,
                "note": f"行情失败：{type(e).__name__}", "rank_score": -1e9}
    advice = advisor.advise(df)
    latest = df.iloc[-1]
    prev = float(df["close"].iloc[-2])
    pct = (float(latest["close"]) / prev - 1) * 100

    sent = _safe(overseas.analyze)
    link = _safe(lambda: sectors.analyze_linkage(code, key=key, model=model, base_url=base_url))
    nws = _safe(lambda: news.analyze(code, key=key, model=model, base_url=base_url))
    mf = _safe(lambda: moneyflow.analyze(code))
    fund = _safe(lambda: broker_extra.analyze(code, retry=broker_retry))
    qsig = _safe(lambda: quant_signal.get(code, profile=profile, style=style))
    sentiment = _safe(lambda: sentiment_signal.score_at(code, latest["date"]))

    pred = prediction.predict(
        code, name, float(latest["close"]), pct,
        advice=advice, sent=sent, link=link, nws=nws, fund=fund, mf=mf, quant=qsig,
        tech_df=df, sentiment=sentiment, key=key, model=model, base_url=base_url)

    quant_score = round(qsig.score, 4) if qsig else None
    quote_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    return {
        "code": code, "name": name, "available": True,
        "close": round(float(latest["close"]), 2), "pct": round(pct, 2),
        "quote_date": quote_date, "quote_source": quote_source,
        "quote_asof": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "direction": pred.direction, "level": pred.level, "confidence": pred.confidence,
        "composite": pred.composite, "logic": _sanitize_limit_claims(pred.logic, name, latest),
        "action": _sanitize_limit_claims(pred.action, name, latest),
        "engine": pred.engine,
        "quant_score": quant_score,
        "sentiment_score": sentiment.score if sentiment and sentiment.available else None,
        "sentiment_model": sentiment.model if sentiment else "",
        "sentiment_enabled": bool(sentiment and sentiment.enabled),
        "watch_rank": getattr(qsig, "watch_rank", None) if qsig else None,
        "expected_return": qsig.expected_return if qsig else None,
        "expected_return_horizon": getattr(qsig, "expected_return_horizon", None) if qsig else None,
        "entry_price": getattr(qsig, "entry_price", None) if qsig else None,
        "stop_loss": getattr(qsig, "stop_loss", None) if qsig else None,
        "take_profit_1": getattr(qsig, "take_profit_1", None) if qsig else None,
        "take_profit_2": getattr(qsig, "take_profit_2", None) if qsig else None,
        "risk_reward_1": getattr(qsig, "risk_reward_1", None) if qsig else None,
        "risk_reward_2": getattr(qsig, "risk_reward_2", None) if qsig else None,
        "rank_score": _rank_score(pred.direction, pred.confidence, pred.composite, quant_score),
    }


def _quote_from_frame(frame, source: str) -> dict | None:
    if frame is None or len(frame) < 2:
        return None
    ordered = frame.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.dropna(subset=["date", "close"]).sort_values("date")
    if len(ordered) < 2:
        return None
    latest = ordered.iloc[-1]
    previous_close = float(ordered["close"].iloc[-2])
    close = float(latest["close"])
    return {
        "close": round(close, 2),
        "pct": round((close / previous_close - 1) * 100, 2),
        "quote_date": latest["date"].strftime("%Y-%m-%d"),
        "quote_source": source,
    }


def latest_quotes(codes, freshness_bucket: int, max_workers: int = 4) -> dict[str, dict]:
    """刷新候选行情；本地日更仓兜底，网络成功时使用更晚的数据覆盖。"""
    normalized = [c for c in (data._normalize_symbol(x) for x in (codes or [])) if c]
    quant_dir = os.environ.get("QUANT_DATA_DIR", "")

    def _one(code: str) -> tuple[str, dict | None]:
        local_quote = None
        if quant_dir:
            path = os.path.join(quant_dir, "price", f"{code}.parquet")
            try:
                if os.path.exists(path):
                    local_quote = _quote_from_frame(pd.read_parquet(path), "午间/收盘日更仓")
            except Exception:  # noqa: BLE001 损坏或正被原子替换时继续尝试网络源
                local_quote = None
        try:
            network_quote = _quote_from_frame(
                data.fetch_daily(code, days=10, freshness_bucket=freshness_bucket), "行情接口")
            if network_quote and (not local_quote or network_quote["quote_date"] >= local_quote["quote_date"]):
                return code, network_quote
        except Exception:  # noqa: BLE001 网络失败时使用本地日更仓
            pass
        return code, local_quote

    if not normalized:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(normalized)))) as ex:
        rows = list(ex.map(_one, normalized))
    return {code: quote for code, quote in rows if quote is not None}


def evaluate_top(codes, key: str = "", model: str = "", base_url: str = "",
                 progress: dict | None = None, max_workers: int = 2,
                 profile: str | None = None, style: str = "short_1_3",
                 broker_retry: int = 6) -> dict:
    """并发评估候选并按大模型综合排序。

    model 留空时随机选模型（与 snapshot_batch 一致：llm.get_random_model()）。
    progress: 传入可变 dict，会更新 {"model", "total", "done"} 供 UI 显示完成度。
    """
    codes = [c for c in (data._normalize_symbol(x) for x in (codes or [])) if c]
    key = llm.get_key(key)
    model = llm.get_model(model) if model else llm.get_random_model()
    if progress is not None:
        progress["model"] = model
        progress["total"] = len(codes)
        progress.setdefault("done", 0)
    if not codes:
        return {"model": model, "rows": []}

    # 预热市场级共享缓存（一次即可），避免各线程同时冷启动重复拉取
    _safe(overseas.analyze)
    _safe(sectors.analyze_sectors)
    _safe(lambda: news.analyze_sector_news(key, model, base_url))

    lock = threading.Lock()

    def _one(c):
        r = evaluate_candidate(c, key=key, model=model, base_url=base_url,
                               profile=profile, style=style, broker_retry=broker_retry)
        if progress is not None:
            with lock:
                progress["done"] = int(progress.get("done", 0)) + 1
        return r

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(codes)))) as ex:
        rows = list(ex.map(_one, codes))

    ok = [r for r in rows if r.get("available")]
    ok.sort(key=lambda r: r.get("rank_score", -1e9), reverse=True)
    for i, r in enumerate(ok, 1):
        r["llm_rank"] = i
    bad = [r for r in rows if not r.get("available")]
    return {"model": model, "rows": ok + bad}
