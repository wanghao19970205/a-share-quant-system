"""通义千问(Qwen / DashScope) 客户端，用于新闻情绪的 LLM 辅助分析。

- 通过 DashScope 的 OpenAI 兼容接口调用，模型默认 qwen-plus。
- API key 来源：函数入参 > 环境变量 DASHSCOPE_API_KEY。
- 无 key 或调用失败时返回 None，由上层优雅降级到词典打分。
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from functools import lru_cache
from pathlib import Path

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_BATCH_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
_FALLBACK_MODEL = "deepseek-v4-flash"
_MODEL_COOLDOWN_SECONDS = 30 * 60
_MODEL_STATUS_LOCK = threading.Lock()
_MODEL_UNAVAILABLE_UNTIL: dict[str, float] = {}


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    path = root / ".env"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        return {}
    return values


def _setting(name: str, default: str = "") -> str:
    return (os.environ.get(name, "") or _env_file_values().get(name, "") or default).strip()


def get_key(override: str = "") -> str:
    return (override or _setting("DASHSCOPE_API_KEY")).strip()


def get_base_url(override: str = "") -> str:
    base = (override or _setting("DASHSCOPE_BASE_URL", _DEFAULT_BASE)).strip()
    return base.rstrip("/")


def get_batch_base_url(override: str = "") -> str:
    """离线 Batch 专用节点；默认固定到支持 /files、/batches 的官方 DashScope 地址。"""
    base = (override or _setting("DASHSCOPE_BATCH_BASE_URL", _BATCH_BASE)).strip()
    return base.rstrip("/")


def _normalize_model_name(model: str) -> str:
    return (model or "").strip().lower()


def _split_model_list(raw: str) -> list[str]:
    items = []
    for part in raw.replace("\n", ",").replace(";", ",").split(","):
        model = _normalize_model_name(part)
        if model and model not in items:
            items.append(model)
    return items


def get_model_list() -> list[str]:
    raw = _setting("DASHSCOPE_MODELS") or _setting("DASHSCOPE_MODEL_LIST")
    models = _split_model_list(raw)
    for fallback in (get_model(), get_fallback_model()):
        if fallback and fallback not in models:
            models.append(fallback)
    return models or [get_fallback_model() or DEFAULT_MODEL]


def get_model(override: str = "") -> str:
    return _normalize_model_name(override or _setting("DASHSCOPE_MODEL", DEFAULT_MODEL))


def get_fallback_model(override: str = "") -> str:
    return _normalize_model_name(override or _setting("DASHSCOPE_FALLBACK_MODEL", _FALLBACK_MODEL))


def _model_available_now(model: str) -> bool:
    model = _normalize_model_name(model)
    if not model:
        return False
    now = time.time()
    with _MODEL_STATUS_LOCK:
        unavailable_until = _MODEL_UNAVAILABLE_UNTIL.get(model, 0)
        if unavailable_until and unavailable_until <= now:
            _MODEL_UNAVAILABLE_UNTIL.pop(model, None)
            return True
        return unavailable_until <= now


def _mark_model_available(model: str) -> None:
    model = _normalize_model_name(model)
    if not model:
        return
    with _MODEL_STATUS_LOCK:
        _MODEL_UNAVAILABLE_UNTIL.pop(model, None)


def _mark_model_unavailable(model: str) -> None:
    model = _normalize_model_name(model)
    if not model:
        return
    with _MODEL_STATUS_LOCK:
        _MODEL_UNAVAILABLE_UNTIL[model] = time.time() + _MODEL_COOLDOWN_SECONDS


def get_random_model(override: str = "") -> str:
    if override:
        return get_model(override)
    fallback = get_fallback_model()
    models = [m for m in get_model_list() if m != fallback and _model_available_now(m)]
    return random.choice(models) if models else fallback


def _model_candidates(preferred: str = "") -> list[str]:
    preferred = get_model(preferred) if preferred else ""
    fallback = get_fallback_model()
    pool = [m for m in get_model_list() if m not in {preferred, fallback} and _model_available_now(m)]
    random.shuffle(pool)
    candidates = []
    if preferred and _model_available_now(preferred):
        candidates.append(preferred)
    candidates.extend(pool)
    if fallback:
        candidates.append(fallback)
    return candidates or [fallback]


def available(override: str = "") -> bool:
    return bool(get_key(override)) and requests is not None


def _quota_or_model_error(resp) -> bool:
    if resp is None:
        return True
    if resp.status_code in {400, 404, 429, 500, 502, 503, 504}:
        return True
    text = (getattr(resp, "text", "") or "").lower()
    needles = [
        "quota", "insufficient", "rate limit", "throttl", "too many requests",
        "model", "not exist", "not found", "unavailable", "限流", "额度", "余额不足",
        "模型不存在", "不可用",
    ]
    return any(n in text for n in needles)


def _chat_once(system: str, user: str, key: str, model: str,
               base_url: str = "", timeout: int = 60) -> str | None:
    if requests is None or not key or not model:
        return None
    url = get_base_url(base_url) + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
            # Qwen3 系列为混合推理模型，关闭思考链，直接产出可解析的 JSON
            "enable_thinking": False,
        },
        timeout=timeout,
    )
    if not resp.ok:
        if _quota_or_model_error(resp):
            raise RuntimeError(f"model unavailable: {model} status={resp.status_code}")
        resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content:  # 兜底：极少数情况下答案落在 reasoning_content
        content = msg.get("reasoning_content") or ""
    return content or None


def _chat(system: str, user: str, key: str, model: str,
          base_url: str = "", timeout: int = 60) -> str | None:
    if requests is None or not key:
        return None
    for candidate in _model_candidates(model):
        try:
            content = _chat_once(system, user, key, candidate, base_url=base_url, timeout=timeout)
            if content:
                _mark_model_available(candidate)
                return content
        except Exception:  # noqa: BLE001 网络/额度/鉴权异常统一尝试下一个模型
            _mark_model_unavailable(candidate)
            continue
    return None


def _extract_json(text: str) -> dict | None:
    """从模型输出里提取 JSON（容忍 ```json 包裹或前后多余文字）。"""
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        # 取第一个代码块内容
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    l, r = s.find("{"), s.rfind("}")
    if l == -1 or r == -1 or r < l:
        return None
    try:
        return json.loads(s[l:r + 1])
    except Exception:  # noqa: BLE001
        return None


def chat_json(system: str, user: str, key: str, model: str = DEFAULT_MODEL,
              base_url: str = "") -> dict | None:
    """通用「输入 system/user，返回解析后 JSON dict」。失败返回 None。"""
    if not available(key):
        return None
    raw = _chat(system, user, key, model, base_url=base_url)
    return _extract_json(raw)


def map_sectors(business: str, products: str, sector_names: list, key: str,
                model: str = DEFAULT_MODEL, base_url: str = "") -> dict | None:
    """用大模型把个股主营/产品映射到给定外围板块。

    会考虑上下游、应用场景与市场概念（如金刚石可用于半导体散热→关联半导体）。
    返回 {"sectors": [板块名...], "reason": "一句话"}，失败返回 None。
    """
    if not available(key) or not sector_names:
        return None
    system = (
        "你是A股行业研究助手。给定公司的主营业务与产品，从【固定板块列表】中选出"
        "与其最相关的板块（1-3个）。要结合上下游、应用场景与当前市场概念判断，"
        "例如：金刚石/超硬材料可用于半导体散热则关联“半导体”；锂电材料关联“太阳能光伏”。"
        "只能从给定列表里选，严格只输出JSON："
        '{"sectors":["板块名",...],"reason":"一句话理由"}。'
    )
    user = f"固定板块列表：{list(sector_names)}\n主营业务：{business}\n产品：{products}"
    raw = _chat(system, user, key, model, base_url=base_url)
    data = _extract_json(raw)
    if not data or "sectors" not in data:
        return None
    secs = [s for s in data.get("sectors", []) if s in sector_names]
    return {"sectors": secs, "reason": str(data.get("reason", "")).strip()}


def score_news(titles: list[str], key: str, model: str = DEFAULT_MODEL,
               context: str = "", base_url: str = "") -> dict | None:
    """对一批新闻标题做情绪打分。

    返回 {"summary": str, "scores": [int...]}，scores 与 titles 等长
    （利好为正、利空为负，取值 -2~2）。失败返回 None。
    """
    if not titles or not available(key):
        return None
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    system = (
        "你是A股金融新闻情绪分析助手。请判断每条新闻对相关个股/板块是利好还是利空，"
        "利好为正、利空为负、中性为0，用-2到2的整数打分（2=重大利好，-2=重大利空）。"
        "再用一句话给出整体情绪总结。严格只输出JSON，不要多余文字，格式："
        '{"summary":"一句话总结","scores":[整数,...]}，scores长度必须与输入条数一致。'
    )
    user = (f"{context}\n\n共{len(titles)}条新闻：\n{numbered}" if context
            else f"共{len(titles)}条新闻：\n{numbered}")
    raw = _chat(system, user, key, model, base_url=base_url)
    data = _extract_json(raw)
    if not data or "scores" not in data:
        return None
    scores = data.get("scores", [])
    # 对齐长度：不足补0，超出截断
    scores = [int(x) if isinstance(x, (int, float)) else 0 for x in scores]
    if len(scores) < len(titles):
        scores += [0] * (len(titles) - len(scores))
    scores = scores[:len(titles)]
    return {"summary": str(data.get("summary", "")).strip(), "scores": scores}


# ------------------------- DashScope Batch API（OpenAI 兼容 /v1/files + /v1/batches）-------------------------
# 用于离线批量任务（如夜间快照），价格约为实时的一半、抗限流；异步（SLA 最长24h）。
# 流程：构造 JSONL -> 上传 /files(purpose=batch) -> 建 /batches -> 轮询 -> 下载结果 JSONL。
def build_batch_jsonl(items: list[dict], model: str, *, max_tokens: int = 2048,
                      temperature: float = 0.2, url: str = "/v1/chat/completions") -> str:
    """items: [{custom_id, system, user, [model], [max_tokens], [temperature]}] -> JSONL 文本。"""
    lines = []
    for it in items:
        body = {
            "model": it.get("model") or model,
            "messages": [
                {"role": "system", "content": it.get("system", "")},
                {"role": "user", "content": it.get("user", "")},
            ],
            "temperature": it.get("temperature", temperature),
            "max_tokens": it.get("max_tokens", max_tokens),
            "enable_thinking": False,
        }
        lines.append(json.dumps({
            "custom_id": str(it["custom_id"]),
            "method": "POST",
            "url": url,
            "body": body,
        }, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def upload_batch_file(jsonl_text: str, key: str, base_url: str = "") -> str:
    # Batch API 与实时 chat 端点可不同；专属推理节点通常不开放 /files、/batches。
    resp = requests.post(
        get_batch_base_url(base_url) + "/files",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": ("batch_input.jsonl", jsonl_text.encode("utf-8"), "application/json")},
        data={"purpose": "batch"},
        timeout=180,
    )
    if not resp.ok:
        raise RuntimeError(f"Batch file upload failed status={resp.status_code}: {resp.text[:800]}")
    return resp.json()["id"]


def create_batch(input_file_id: str, key: str, base_url: str = "",
                 endpoint: str = "/v1/chat/completions", completion_window: str = "24h",
                 metadata: dict | None = None) -> str:
    payload = {"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": completion_window}
    if metadata:
        payload["metadata"] = metadata
    resp = requests.post(
        get_batch_base_url(base_url) + "/batches",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def get_batch(batch_id: str, key: str, base_url: str = "") -> dict:
    resp = requests.get(
        get_batch_base_url(base_url) + f"/batches/{batch_id}",
        headers={"Authorization": f"Bearer {key}"}, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def download_file_content(file_id: str, key: str, base_url: str = "") -> str:
    resp = requests.get(
        get_batch_base_url(base_url) + f"/files/{file_id}/content",
        headers={"Authorization": f"Bearer {key}"}, timeout=300,
    )
    resp.raise_for_status()
    return resp.text


def wait_batch(batch_id: str, key: str, base_url: str = "",
               poll_interval: float = 15.0, max_wait: float = 3600.0,
               on_progress=None) -> dict:
    deadline = time.time() + max_wait
    last = {}
    while True:
        last = get_batch(batch_id, key, base_url=base_url)
        status = last.get("status")
        if on_progress:
            try:
                on_progress(last)
            except Exception:  # noqa: BLE001
                pass
        if status in ("completed", "failed", "expired", "cancelled"):
            return last
        if time.time() > deadline:
            return last
        time.sleep(max(1.0, float(poll_interval)))


def parse_batch_output(output_text: str) -> dict[str, str]:
    """解析结果 JSONL -> {custom_id: content 文本}。"""
    out: dict[str, str] = {}
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        cid = obj.get("custom_id")
        body = (obj.get("response") or {}).get("body") or {}
        try:
            msg = body["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
        except Exception:  # noqa: BLE001
            content = ""
        if cid is not None:
            out[str(cid)] = content
    return out


def run_chat_batch(items: list[dict], key: str = "", model: str = "", base_url: str = "",
                   metadata: dict | None = None, poll_interval: float = 15.0,
                   max_wait: float = 3600.0, on_progress=None) -> dict[str, str]:
    """高层封装：提交一批 chat 请求，等待完成，返回 {custom_id: content}。失败/超时返回 {}。"""
    key = get_key(key)
    if requests is None or not key or not items:
        return {}
    model = get_model(model) if model else get_random_model()
    # base_url 在 Batch 场景是 Batch 专用覆盖；默认走 DASHSCOPE_BATCH_BASE_URL。
    base_url = get_batch_base_url(base_url)
    try:
        jsonl = build_batch_jsonl(items, model)
        file_id = upload_batch_file(jsonl, key, base_url=base_url)
        batch_id = create_batch(file_id, key, base_url=base_url, metadata=metadata)
        info = wait_batch(batch_id, key, base_url=base_url,
                          poll_interval=poll_interval, max_wait=max_wait, on_progress=on_progress)
        if info.get("status") != "completed" or not info.get("output_file_id"):
            return {}
        text = download_file_content(info["output_file_id"], key, base_url=base_url)
        return parse_batch_output(text)
    except Exception:  # noqa: BLE001
        return {}
