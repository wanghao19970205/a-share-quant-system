"""批量记录多股票的每日多维快照（供定时任务积累数据，用于将来完整回测）。

用法（容器内）：
    WATCHLIST="600707,600519,000001" python -m stock_analyzer.snapshot_batch
或传参：
    python -m stock_analyzer.snapshot_batch 600707 600519

也可在 UI「选股/回测」页一键批量记录当前股票池。
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

from stock_analyzer import (advisor, broker_extra, data, indicators, llm, moneyflow,
                            news, overseas, prediction, quant_signal, sectors, sentiment_signal, snapshot)


_EVENT_KEYWORDS = {
    "业绩": ["预增", "预减", "预盈", "预亏", "扭亏", "业绩", "利润", "营收"],
    "订单": ["中标", "合同", "订单", "签约", "采购"],
    "产能": ["扩产", "投产", "量产", "产能", "项目建设"],
    "并购": ["并购", "收购", "重组", "注入", "股权转让"],
    "融资": ["定增", "融资", "发债", "可转债", "募资"],
    "回购增持": ["回购", "增持", "员工持股"],
    "减持": ["减持", "清仓", "套现"],
    "监管风险": ["处罚", "立案", "问询", "调查", "违规", "诉讼"],
    "政策热点": ["政策", "补贴", "规划", "试点", "支持"],
    "价格变化": ["涨价", "提价", "降价", "价格", "供需"],
}


def _join(items) -> str:
    return "|".join(str(x) for x in items if str(x))


def _snapshot_news_fields(nws, link) -> dict:
    stock_items = list(getattr(nws, "stock_items", []) or []) if nws else []
    matched_sector_news = list(getattr(nws, "matched_sector_news", []) or []) if nws else []
    matched = getattr(link, "matched", {}) if link else {}
    if not matched and nws:
        matched = {s: [] for s in (getattr(nws, "matched_sectors", []) or [])}

    texts = []
    for item in stock_items:
        texts.append(f"{getattr(item, 'title', '')} {getattr(item, 'summary', '')}")
    for sec in matched_sector_news:
        texts.extend(getattr(sec, "samples", []) or [])
    text = " ".join(texts)

    hot_keywords = []
    if isinstance(matched, dict):
        for kws in matched.values():
            for kw in kws or []:
                if kw not in hot_keywords:
                    hot_keywords.append(kw)
    for sec in matched_sector_news[:5]:
        name = getattr(sec, "name", "")
        if name and name not in hot_keywords:
            hot_keywords.append(name)

    tags = []
    for tag, words in _EVENT_KEYWORDS.items():
        if any(w in text for w in words):
            tags.append(tag)

    return {
        "news_count": len(stock_items),
        "news_pos_count": sum(1 for i in stock_items if getattr(i, "sentiment", 0) > 0),
        "news_neg_count": sum(1 for i in stock_items if getattr(i, "sentiment", 0) < 0),
        "sector_matched_count": len(matched or {}),
        "hot_keywords": _join(hot_keywords[:12]),
        "qwen_event_tags": _join(tags[:12]),
    }


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def snapshot_symbol(symbol: str, key: str = "", model: str = "", base_url: str = "", force: bool = False) -> int:
    """计算单只股票的多维信号并落盘一条快照，返回累计天数（失败返回 -1）。

    force=False 时：若当日快照已存在，直接跳过，**不再发起任何大模型/网络请求**，
    避免重跑时对已落盘的日期重复消耗 token。
    """
    try:
        df = indicators.compute_all(data.fetch_daily(symbol, days=250))
    except Exception:  # noqa: BLE001
        return -1
    latest = df.iloc[-1]
    date_str = latest["date"].strftime("%Y-%m-%d")
    if not force:
        hist = snapshot.history(symbol)
        if not hist.empty and (hist["date"].astype(str) == date_str).any():
            return len(hist)  # 当日已落盘，跳过（省去 Qwen/券商等调用）

    advice = advisor.advise(df)
    prev = float(df["close"].iloc[-2])
    pct = (float(latest["close"]) / prev - 1) * 100

    sent = _safe(overseas.analyze)
    link = _safe(lambda: sectors.analyze_linkage(symbol, key=key, model=model, base_url=base_url))
    nws = _safe(lambda: news.analyze(symbol, key=key, model=model, base_url=base_url))
    mf = _safe(lambda: moneyflow.analyze(symbol))
    fund = _safe(lambda: broker_extra.analyze(symbol))

    qsig = _safe(lambda: quant_signal.get(symbol))
    sentiment = _safe(lambda: sentiment_signal.score_at(symbol, latest["date"]))
    pred = prediction.predict(
        symbol, data.get_stock_name(symbol), float(latest["close"]), pct,
        advice=advice, sent=sent, link=link, nws=nws, fund=fund, mf=mf, quant=qsig,
        tech_df=df, sentiment=sentiment, key=key, model=model, base_url=base_url)

    record = {
        "date": date_str,
        "close": round(float(latest["close"]), 2),
        "tech": advice.total_score,
        "overseas": round(sent.weighted_score, 2) if sent else None,
        "sector": round(link.link_score, 2) if link else None,
        "news": round(nws.overall_score, 2) if nws else None,
        "moneyflow": round(mf.score, 2) if (mf and getattr(mf, "available", False)) else None,
        "fund": round(fund.score, 2) if (fund and getattr(fund, "available", False)) else None,
        "quant_score": round(qsig.score, 4) if qsig else None,
        "quant_rank_pct": round(qsig.rank_pct, 4) if qsig else None,
        "quant_model": qsig.model if qsig else None,
        "sentiment_score": sentiment.score if (sentiment and sentiment.available) else None,
        "sentiment_model": sentiment.model if sentiment else None,
        "sentiment_count": sentiment.article_count if sentiment else 0,
        "pred_composite": pred.composite,
        "pred_level": pred.level,
        "engine": pred.engine,
    }
    record.update(_snapshot_news_fields(nws, link))
    return snapshot.save(symbol, record)


def run(codes, key: str = "", model: str = "", base_url: str = "", max_workers: int = 4, force: bool = False) -> dict:
    """并发记录多股票快照。每只股票写各自的 snapshots/{code}.csv，无文件写竞争；
    含 Qwen 调用（新闻/板块/预估），并发度默认 4 以平衡速度与限流。

    force=False（默认）：当日已落盘的股票直接跳过，不重复消耗 token；重跑安全。
    并发前先在主线程预热『市场级』共享缓存（外围美股 / 外围板块 / 市场新闻，均为
    lru_cache 且对所有股票相同）。否则多个 worker 会同时冷启动这些重网络调用，
    造成缓存踩踏——同一份数据被拉取多次（analyze_sectors 每次还要拉 13 只美股 ETF）。
    """
    key = llm.get_key(key)
    model = llm.get_model(model) if model else llm.get_random_model()
    base_url = llm.get_base_url(base_url)
    print(f"[snapshots] qwen_model_preferred={model} force={force}")
    seen, uniq = set(), []
    for c in codes:
        c = data._normalize_symbol(c)
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return {"ok": [], "fail": [], "count": 0, "skipped": 0}

    # 预热共享缓存（一次即可，之后各线程命中缓存瞬时返回）
    _safe(overseas.analyze)
    _safe(sectors.analyze_sectors)
    _safe(lambda: news.analyze_sector_news(key, model, base_url))

    # 预取“当日已落盘”的代码，用于统计跳过数（snapshot_symbol 内也会再判一次）
    def _one(c):
        pre = snapshot.count(c)
        try:
            n = snapshot_symbol(c, key=key, model=model, base_url=base_url, force=force)
        except Exception:  # noqa: BLE001
            n = -1
        return c, n, pre

    workers = min(max_workers, len(uniq))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, uniq))
    ok = [c for c, n, _pre in results if n >= 0]
    fail = [c for c, n, _pre in results if n < 0]
    # 跳过=成功但天数未增加（当日已存在，未新写）
    skipped = sum(1 for c, n, pre in results if n >= 0 and n == pre and pre > 0)
    return {"ok": ok, "fail": fail, "count": len(ok), "skipped": skipped}


def _compute_context(symbol: str, key: str, model: str, base_url: str, force: bool):
    """算好某只票除“最终预估LLM”外的一切，返回 batch 所需材料；当日已落盘则返回 None。"""
    try:
        df = indicators.compute_all(data.fetch_daily(symbol, days=250))
    except Exception:  # noqa: BLE001
        return {"code": symbol, "skip": False, "fail": True}
    latest = df.iloc[-1]
    date_str = latest["date"].strftime("%Y-%m-%d")
    if not force:
        hist = snapshot.history(symbol)
        if not hist.empty and (hist["date"].astype(str) == date_str).any():
            return {"code": symbol, "skip": True, "days": len(hist)}
    advice = advisor.advise(df)
    prev = float(df["close"].iloc[-2])
    pct = (float(latest["close"]) / prev - 1) * 100
    name = data.get_stock_name(symbol)
    sent = _safe(overseas.analyze)
    link = _safe(lambda: sectors.analyze_linkage(symbol, key=key, model=model, base_url=base_url))
    nws = _safe(lambda: news.analyze(symbol, key=key, model=model, base_url=base_url))
    mf = _safe(lambda: moneyflow.analyze(symbol))
    fund = _safe(lambda: broker_extra.analyze(symbol))
    qsig = _safe(lambda: quant_signal.get(symbol))
    sentiment = _safe(lambda: sentiment_signal.score_at(symbol, latest["date"]))
    summary = prediction.build_summary(symbol, name, float(latest["close"]), pct,
                                       advice, sent, link, nws, fund, mf, qsig, df, sentiment)
    comp, r_dir, r_level, r_conf = prediction._rule_based(advice, sent, link, nws, fund, mf, qsig, sentiment)  # noqa: SLF001
    record = {
        "date": date_str,
        "close": round(float(latest["close"]), 2),
        "tech": advice.total_score,
        "overseas": round(sent.weighted_score, 2) if sent else None,
        "sector": round(link.link_score, 2) if link else None,
        "news": round(nws.overall_score, 2) if nws else None,
        "moneyflow": round(mf.score, 2) if (mf and getattr(mf, "available", False)) else None,
        "fund": round(fund.score, 2) if (fund and getattr(fund, "available", False)) else None,
        "quant_score": round(qsig.score, 4) if qsig else None,
        "quant_rank_pct": round(qsig.rank_pct, 4) if qsig else None,
        "quant_model": qsig.model if qsig else None,
        "sentiment_score": sentiment.score if (sentiment and sentiment.available) else None,
        "sentiment_model": sentiment.model if sentiment else None,
        "sentiment_count": sentiment.article_count if sentiment else 0,
    }
    record.update(_snapshot_news_fields(nws, link))
    return {"code": symbol, "skip": False, "fail": False, "record": record,
            "summary": summary, "rule": (comp, r_dir, r_level)}


def run_batched(codes, key: str = "", model: str = "", base_url: str = "", max_workers: int = 4,
                force: bool = False, poll_interval: float = 60.0, max_wait: float = 90000.0) -> dict:
    """离线批量版：本地信号并发算好后，把“次日预估”的 Qwen 调用合并成一个 DashScope Batch 提交，
    等待完成后统一落盘。省 token（batch 约半价）、抗限流；异步、耗时更长，适合夜间定时。

    说明：新闻/板块联动的 Qwen 仍走实时（有磁盘缓存，重复成本低）；本阶段先把最大且未缓存的
    “次日预估”调用改为 batch。无 key 或 batch 失败时，自动回退到规则法预估（不阻塞落盘）。
    """
    key = llm.get_key(key)
    model = llm.get_model(model) if model else llm.get_random_model()
    base_url = llm.get_base_url(base_url)
    batch_base_url = llm.get_batch_base_url()
    print(f"[snapshots:batch] qwen_model={model} force={force} batch_base={batch_base_url}", flush=True)
    seen, uniq = set(), []
    for c in codes:
        c = data._normalize_symbol(c)
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    if not uniq:
        return {"ok": [], "fail": [], "count": 0, "skipped": 0}

    _safe(overseas.analyze)
    _safe(sectors.analyze_sectors)
    _safe(lambda: news.analyze_sector_news(key, model, base_url))

    workers = min(max_workers, len(uniq))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ctxs = list(ex.map(lambda c: _compute_context(c, key, model, base_url, force), uniq))

    skipped = sum(1 for x in ctxs if x.get("skip"))
    fail = [x["code"] for x in ctxs if x.get("fail")]
    todo = [x for x in ctxs if not x.get("skip") and not x.get("fail")]
    print(f"[snapshots:batch] 本地信号完成：待预估 {len(todo)}，当日已存在跳过 {skipped}，失败 {len(fail)}", flush=True)

    # 组一个 batch：每只票一条“次日预估”请求
    items = [{"custom_id": x["code"], "system": prediction._SYSTEM, "user": x["summary"]}  # noqa: SLF001
             for x in todo]
    contents: dict[str, str] = {}
    if items:
        print(f"[snapshots:batch] 提交 DashScope Batch，共 {len(items)} 条，轮询中…", flush=True)
        contents = llm.run_chat_batch(
            items, key=key, model=model, base_url=batch_base_url,
            metadata={"ds_name": "daily_snapshot_pred", "ds_description": f"{len(items)} preds"},
            poll_interval=poll_interval, max_wait=max_wait,
            on_progress=lambda b: print(f"[snapshots:batch] status={b.get('status')}", flush=True))
        print(f"[snapshots:batch] Batch 返回 {len(contents)}/{len(items)} 条", flush=True)

    ok = []
    for x in todo:
        code = x["code"]
        comp, r_dir, r_level = x["rule"]
        content = contents.get(code, "")
        data_json = llm._extract_json(content) if content else None  # noqa: SLF001
        if data_json and data_json.get("direction"):
            d = str(data_json["direction"])
            level = "bullish" if "多" in d else ("bearish" if "空" in d else "neutral")
            engine = "Qwen大模型(batch)"
        else:
            level, engine = r_level, "本地规则"  # batch 缺失/解析失败 → 规则法兜底
        rec = dict(x["record"])
        rec["pred_composite"] = comp
        rec["pred_level"] = level
        rec["engine"] = engine
        try:
            snapshot.save(code, rec)
            ok.append(code)
        except Exception:  # noqa: BLE001
            fail.append(code)
    return {"ok": ok, "fail": fail, "count": len(ok), "skipped": skipped,
            "batched": len(items), "batch_returned": len(contents)}


def _watchlist() -> list:
    """自选池来源：环境变量 WATCHLIST > snapshots/watchlist.txt。
    从文本中提取所有 6 位代码（文件里可带名称/注释，只抓代码）。"""
    import re
    raw = os.environ.get("WATCHLIST", "")
    if not raw.strip():
        path = os.path.join(os.environ.get("SNAPSHOT_DIR", "snapshots"), "watchlist.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = f.read()
    return re.findall(r"\d{6}", raw)
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--force", "--batch")]
    force = ("--force" in sys.argv) or os.environ.get("SNAPSHOT_FORCE", "").strip().lower() in ("1", "true", "yes", "on")
    use_batch = ("--batch" in sys.argv) or os.environ.get("SNAPSHOT_BATCH", "").strip().lower() in ("1", "true", "yes", "on")
    codes = args or _watchlist()
    if not codes:
        print("未指定股票：设置环境变量 WATCHLIST、传入代码参数，或写入 snapshots/watchlist.txt")
        sys.exit(1)
    res = run_batched(codes, force=force) if use_batch else run(codes, force=force)
    print(f"快照完成：成功 {res['count']} 只（其中当日已存在跳过 {res.get('skipped', 0)} 只）"
          + (f"；失败 {res['fail']}" if res["fail"] else ""))
    # 券商 tgw 原生库退出时可能段错误(SIGSEGV)，此时快照已落盘。os._exit(0) 干净退出避免误报失败。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
