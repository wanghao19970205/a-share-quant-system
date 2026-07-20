"""Qwen 批量结构化标注白名单新闻，结果与原文同库持久化。

该模块只解释新闻语义，绝不要求模型预测股价。日级舆情特征和是否入分由
sentiment_signal 的时间切分验证决定。Batch 状态保存于 NEWS_DIR，进程重启后可继续
collect；内容哈希保证同一篇文本最多提交一次。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_analyzer import llm, news_store

_STATE_FILE = "news_annotation_batches.json"
_MAX_TITLE = 240
_MAX_SUMMARY = 900

_SYSTEM = """你是A股新闻结构化标注器。只根据新闻文本判断可验证的事件含义，不能根据后续股价推断，不能给出买卖建议。\n
严格只输出一个 JSON 对象，字段必须齐全：\n{
  \"impact\": -2到2的整数,
  \"relevance\": 0到1的小数,
  \"horizon\": \"intraday|1_3d|1_4w|unknown\",
  \"event_types\": [\"业绩|订单|产能|并购|融资|回购增持|减持|监管风险|政策热点|价格变化|行业景气|其他\"],
  \"novelty\": 0到1的小数,
  \"certainty\": 0到1的小数,
  \"reason\": \"不超过40字的事实摘要\"
}\n
impact 只表示对该公司未来基本面/预期的方向：2重大利好、1利好、0中性或信息不足、-1利空、-2重大利空。relevance 表示与给定股票的直接相关程度；市场泛资讯通常低于0.4。novelty 表示相对常见转载是否包含新事实。certainty 表示文本本身的事实确定性。"""


def _state_path() -> str:
    return os.path.join(news_store._dir(), _STATE_FILE)  # noqa: SLF001


def _load_state() -> dict:
    try:
        return json.loads(Path(_state_path()).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"version": 1, "batches": {}}


def _save_state(state: dict) -> None:
    tmp = _state_path() + ".tmp"
    Path(tmp).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _state_path())


def _content_hash(row: pd.Series) -> str:
    text = "\n".join(str(row.get(k) or "").strip() for k in ("code", "category", "title", "summary"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _clean_float(value, low: float, high: float, default: float = 0.0) -> float:
    try:
        return round(max(low, min(high, float(value))), 3)
    except (TypeError, ValueError):
        return default


_TEXT_FIELDS = {"llm_horizon", "llm_event_types", "llm_reason", "llm_model", "llm_annotated_at", "llm_content_hash"}


def _set_annotation(df: pd.DataFrame, index: int, annotation: dict, model: str,
                    annotated_at: str, content_hash: str) -> None:
    """Write schema-migrated annotations safely when old all-null Parquet columns infer float64."""
    values = dict(annotation)
    values.update({"llm_model": model, "llm_annotated_at": annotated_at, "llm_content_hash": content_hash})
    for name, value in values.items():
        if name in _TEXT_FIELDS and str(df[name].dtype) != "object":
            df[name] = df[name].astype(object)
        df.loc[index, name] = value


def _annotation(content: str) -> dict | None:
    data = llm._extract_json(content)  # noqa: SLF001 - shared tolerant JSON parser
    if not isinstance(data, dict):
        return None
    impact = data.get("impact")
    try:
        impact = int(round(float(impact)))
    except (TypeError, ValueError):
        return None
    if impact < -2 or impact > 2:
        return None
    events = data.get("event_types", [])
    if not isinstance(events, list):
        events = []
    horizon = str(data.get("horizon") or "unknown")
    if horizon not in {"intraday", "1_3d", "1_4w", "unknown"}:
        horizon = "unknown"
    return {
        "llm_impact": impact,
        "llm_relevance": _clean_float(data.get("relevance"), 0.0, 1.0),
        "llm_horizon": horizon,
        "llm_event_types": "|".join(str(x)[:24] for x in events[:4]),
        "llm_novelty": _clean_float(data.get("novelty"), 0.0, 1.0),
        "llm_certainty": _clean_float(data.get("certainty"), 0.0, 1.0),
        "llm_reason": str(data.get("reason") or "").strip()[:160],
    }


def _iter_pending() -> tuple[dict[str, list[tuple[str, int]]], dict[str, dict]]:
    """Return unique unannotated content hashes and all rows they should update."""
    locations: dict[str, list[tuple[str, int]]] = defaultdict(list)
    payloads: dict[str, dict] = {}
    for filename in os.listdir(news_store._dir()):  # noqa: SLF001
        if not filename.endswith(".parquet") or filename == "_market.parquet":
            continue
        code = filename[:-8]
        df = news_store.read_store(code)
        for index, row in df.iterrows():
            if pd.notna(row.get("llm_impact")):
                continue
            content_hash = _content_hash(row)
            locations[content_hash].append((code, int(index)))
            if content_hash not in payloads:
                payloads[content_hash] = {
                    "code": str(row.get("code") or code),
                    "category": str(row.get("category") or ""),
                    "source": str(row.get("source") or ""),
                    "title": str(row.get("title") or "")[:_MAX_TITLE],
                    "summary": str(row.get("summary") or "")[:_MAX_SUMMARY],
                }
    return locations, payloads


def pending_stats() -> dict:
    locations, payloads = _iter_pending()
    return {"unique_pending": len(payloads), "rows_pending": sum(map(len, locations.values()))}


def submit(limit: int = 500, key: str = "", model: str = "", base_url: str = "") -> dict:
    """Submit one asynchronous Batch request and persist every row mapping for later collection."""
    key = llm.get_key(key)
    if not llm.available(key):
        raise RuntimeError("DASHSCOPE_API_KEY unavailable")
    state = _load_state()
    active = [b for b in state.get("batches", {}).values() if b.get("status") in {"submitted", "processing", "validating", "finalizing"}]
    max_active = max(1, int(os.environ.get("NEWS_ANNOTATION_MAX_ACTIVE", "4")))
    if len(active) >= max_active:
        return {"status": "blocked", "reason": "active batch limit reached", "active": len(active), "max_active": max_active}
    locations, payloads = _iter_pending()
    take = list(payloads)[:max(1, int(limit))]
    if not take:
        return {"status": "empty", **pending_stats()}
    items = []
    mapping = {}
    for content_hash in take:
        p = payloads[content_hash]
        user = (f"股票代码：{p['code']}\n新闻类别：{p['category']}\n来源：{p['source']}\n"
                f"标题：{p['title']}\n摘要：{p['summary']}")
        items.append({"custom_id": content_hash, "system": _SYSTEM, "user": user, "max_tokens": 320})
        mapping[content_hash] = locations[content_hash]
    selected_model = llm.get_model(model) if model else llm.get_random_model()
    jsonl = llm.build_batch_jsonl(items, selected_model, max_tokens=320)
    file_id = llm.upload_batch_file(jsonl, key, base_url=base_url)
    batch_id = llm.create_batch(file_id, key, base_url=base_url, metadata={"job": "news_annotation", "count": str(len(items))})
    state.setdefault("batches", {})[batch_id] = {
        "status": "submitted", "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": selected_model, "count": len(items), "mapping": mapping,
    }
    _save_state(state)
    return {"status": "submitted", "batch_id": batch_id, "count": len(items), **pending_stats()}


def purge_rule_annotations() -> dict:
    """Remove the retired offline_history_v1 pseudo-labels; preserve actual Qwen outputs."""
    cleared = 0
    annotation_fields = [name for name in news_store.NEWS_FIELDS if name.startswith("llm_")]
    for filename in os.listdir(news_store._dir()):  # noqa: SLF001
        if not filename.endswith(".parquet") or filename == "_market.parquet":
            continue
        code = filename[:-8]
        df = news_store.read_store(code)
        mask = df["llm_model"].fillna("").astype(str).eq("offline_history_v1")
        if mask.any():
            for name in annotation_fields:
                df.loc[mask, name] = None
            cleared += int(mask.sum())
            news_store.save_store(code, df)
    return {"mode": "purge_rule_annotations", "cleared_rows": cleared, **pending_stats()}


def _write_back(by_code: dict) -> int:
    """把累积的标注按 code 写回 parquet，返回写入行数并清空缓冲。

    分块落盘用：每处理一小批就调用一次，避免长任务中途崩溃丢失全部进度。
    重复内容 hash 命中多条新闻时同步写回，已标注的行跳过。
    """
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    for code, updates in by_code.items():
        df = news_store.read_store(code)
        hashes = df.apply(_content_hash, axis=1) if not df.empty else pd.Series(dtype=str)
        for content_hash, annotation, actual_model in updates:
            for current_index in hashes[hashes == content_hash].index.tolist():
                if pd.notna(df.loc[current_index, "llm_impact"]):
                    continue
                _set_annotation(df, current_index, annotation, actual_model, now, content_hash)
                written += 1
        news_store.save_store(code, df)
    by_code.clear()
    return written


def annotate_realtime(limit: int = 20, key: str = "", model: str = "", base_url: str = "",
                      newest: bool = False, flush_every: int = 50,
                      max_consecutive_failures: int = 20) -> dict:
    """Use realtime Qwen for newly ingested news; newest selects recent records before old backlog.

    flush_every: 每累积这么多条成功标注就写回一次并打印进度，使长回溯任务分块落盘、可断点续跑。
    max_consecutive_failures: 连续失败达到该阈值即中止（多为额度耗尽/权限失效），避免默默空转。
    """
    key = llm.get_key(key)
    if not llm.available(key):
        raise RuntimeError("DASHSCOPE_API_KEY unavailable")
    locations, payloads = _iter_pending()
    if newest:
        # 日更只消化本次/近期新闻，绝不从一年前的存量排队头部开始。
        ordered = []
        for filename in os.listdir(news_store._dir()):  # noqa: SLF001
            if not filename.endswith(".parquet") or filename == "_market.parquet":
                continue
            for _, row in news_store.read_store(filename[:-8]).sort_values("publish_time", ascending=False).iterrows():
                if pd.notna(row.get("llm_impact")):
                    continue
                h = _content_hash(row)
                if h in payloads and h not in ordered:
                    ordered.append(h)
        pending_hashes = ordered
    else:
        pending_hashes = list(payloads)
    # 每篇请求随机选择当前可用候选模型，分散各模型 token 消耗；
    # 某模型失败时 llm._chat 会自动冷却并依次尝试候补模型。
    requested_model = llm.get_model(model) if model else ""
    by_code: dict[str, list[tuple[str, dict, str]]] = defaultdict(list)
    attempted = annotated = failed = 0
    since_flush = 0
    consecutive_failures = 0
    aborted = ""
    flush_every = max(1, int(flush_every))
    max_consecutive_failures = max(1, int(max_consecutive_failures))
    used_models: dict[str, int] = defaultdict(int)
    for content_hash in pending_hashes[:max(1, int(limit))]:
        p = payloads[content_hash]
        selected_model = requested_model or llm.get_random_model()
        user = (f"股票代码：{p['code']}\n新闻类别：{p['category']}\n来源：{p['source']}\n"
                f"标题：{p['title']}\n摘要：{p['summary']}")
        result = llm.chat_json(_SYSTEM, user, key, model=selected_model, base_url=base_url)
        attempted += 1
        parsed = _annotation(json.dumps(result, ensure_ascii=False)) if result else None
        if not parsed:
            failed += 1
            consecutive_failures += 1
            # 连续失败通常意味着额度耗尽/权限失效/全模型限流；立即中止并落盘，避免默默空转。
            if consecutive_failures >= max_consecutive_failures:
                aborted = (f"连续 {consecutive_failures} 条调用失败，疑似 Qwen 额度耗尽/权限失效，已中止。"
                           f"（已成功 attempted={attempted} failed={failed}）")
                print(f"[realtime] ABORT {aborted}", flush=True)
                break
            time.sleep(0.5)
            continue
        used_models[selected_model] += 1
        consecutive_failures = 0
        for code, _index in locations[content_hash]:
            by_code[code].append((content_hash, parsed, selected_model))
        since_flush += 1
        # 分块落盘：每 flush_every 条成功标注写回一次，长任务中途崩溃最多只丢当前这一小块。
        if since_flush >= flush_every:
            annotated += _write_back(by_code)
            since_flush = 0
            print(f"[realtime] progress attempted={attempted} annotated={annotated} failed={failed}",
                  flush=True)
        time.sleep(0.5)
    annotated += _write_back(by_code)
    result = {"mode": "realtime", "attempted": attempted, "annotated_rows": annotated,
              "failed": failed, "models": dict(used_models), **pending_stats()}
    if aborted:
        result["aborted"] = aborted
    return result


def collect(key: str = "", base_url: str = "") -> dict:
    """Collect completed batches and atomically write annotations into per-code news Parquet files."""
    key = llm.get_key(key)
    if not llm.available(key):
        raise RuntimeError("DASHSCOPE_API_KEY unavailable")
    state = _load_state()
    changed = 0
    reports = []
    for batch_id, batch in state.get("batches", {}).items():
        if batch.get("status") in {"completed", "failed", "expired", "cancelled"}:
            continue
        info = llm.get_batch(batch_id, key, base_url=base_url)
        status = str(info.get("status") or "unknown")
        batch["status"] = status
        batch["last_checked_at"] = datetime.now(timezone.utc).isoformat()
        if status != "completed":
            reports.append({"batch_id": batch_id, "status": status})
            continue
        output_id = info.get("output_file_id")
        if not output_id:
            batch["status"] = "failed"
            batch["error"] = "completed without output_file_id"
            continue
        results = llm.parse_batch_output(llm.download_file_content(output_id, key, base_url=base_url))
        by_code: dict[str, list[tuple[int, dict, str]]] = defaultdict(list)
        for content_hash, rows in batch.get("mapping", {}).items():
            parsed = _annotation(results.get(content_hash, ""))
            if not parsed:
                continue
            for code, index in rows:
                by_code[code].append((int(index), parsed, content_hash))
        now = datetime.now(timezone.utc).isoformat()
        for code, updates in by_code.items():
            df = news_store.read_store(code)
            hashes = df.apply(_content_hash, axis=1) if not df.empty else pd.Series(dtype=str)
            for index, annotation, content_hash in updates:
                # 日更落库会按发布时间重排，所以不能信任提交时的行号；内容哈希才是稳定主键。
                matches = hashes[hashes == content_hash].index.tolist()
                for current_index in matches:
                    if pd.notna(df.loc[current_index, "llm_impact"]):
                        continue
                    _set_annotation(df, current_index, annotation, batch.get("model", ""), now, content_hash)
                    changed += 1
            news_store.save_store(code, df)
        batch["status"] = "completed"
        batch["completed_at"] = now
        batch["result_count"] = len(results)
        batch["annotated_rows"] = sum(len(v) for v in by_code.values())
        reports.append({"batch_id": batch_id, "status": "completed", "annotated": batch["annotated_rows"]})
    _save_state(state)
    return {"updated_rows": changed, "batches": reports, **pending_stats()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen 离线结构化标注白名单新闻")
    ap.add_argument("--mode", choices=("stats", "submit", "collect", "realtime", "purge-rule"), default="stats")
    ap.add_argument("--limit", type=int, default=20, help="实时标注条数，或单个 DashScope Batch 的唯一新闻数")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--newest", action="store_true", help="实时模式优先标注最新入库新闻")
    ap.add_argument("--flush-every", type=int, default=50, help="实时模式每标注多少条写回一次并打印进度")
    ap.add_argument("--max-consecutive-failures", type=int, default=20,
                    help="实时模式连续失败达到该阈值即中止（多为额度耗尽/权限失效）")
    args = ap.parse_args()
    if args.mode == "stats":
        result = pending_stats()
    elif args.mode == "submit":
        result = submit(args.limit, args.key, args.model, args.base_url)
    elif args.mode == "realtime":
        result = annotate_realtime(args.limit, args.key, args.model, args.base_url, args.newest,
                                   flush_every=args.flush_every,
                                   max_consecutive_failures=args.max_consecutive_failures)
    elif args.mode == "purge-rule":
        result = purge_rule_annotations()
    else:
        result = collect(args.key, args.base_url)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
