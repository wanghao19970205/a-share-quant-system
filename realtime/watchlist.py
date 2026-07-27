"""订阅清单加载：Top10 名单(手机UI) ∪ 持仓，去重、规范 6 位、按上限夹紧。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import RealtimeConfig

# mobile_snapshot.json 里的三个固定分组（与 stock_analyzer.top10_eval._GROUPS 一致）。
_TOP10_GROUPS = ("白名单", "全A", "创新药")


def _norm(code: str) -> str:
    return str(code).strip().zfill(6)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            s = ln.split("#", 1)[0].strip()   # 支持行内注释
            if not s:
                continue
            # 持仓文件可能是 "代码 买入日期" / "代码,买入日期"，只取第一段作代码。
            token = s.replace(",", " ").split()[0]
            out.append(_norm(token))
    return out


def _read_top10_groups(path: Path) -> list[str]:
    """读手机 UI 快照 mobile_snapshot.json，取三名单(白名单/全A/创新药)的股票代码。

    结构：{"groups": {"<组名>": {"rows": [{"code": "600519", ...}, ...]}}}。
    按组序、组内 rows 序去重收集代码。文件缺失/损坏/结构异常都返回 []（交给上层兜底）。
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 快照损坏不拦启动，退回预测兜底
        return []
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        return []
    out: list[str] = []
    for name in _TOP10_GROUPS:
        entry = groups.get(name) or {}
        rows = entry.get("rows") if isinstance(entry, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            code = row.get("code") if isinstance(row, dict) else None
            if code is None:
                continue
            c = _norm(code)
            if len(c) == 6 and c.isdigit():
                out.append(c)
    return out


def _read_predictions(path: Path) -> list[str]:
    """读最新一期短线预测，按预测分 pred 降序返回代码列表（模型看多最强在前）。

    仅作 Top10 快照缺失时的兜底：预测文件是全 A 打分表（每日数千只），必须按分排序，
    才能在 max_subscribe 夹紧时保住"最强的 N 只"，而不是代码号最小的 N 只。
    """
    if not path.exists():
        return []
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return []
    try:
        df = pd.read_parquet(path, columns=["code", "date", "pred"])
    except Exception:  # noqa: BLE001
        # 列名不确定时退回全读
        df = pd.read_parquet(path)
        if "code" not in df.columns:
            return []
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df[d == d.max()]
    # 打分列：优先 pred，缺失则退回原始行序。
    score_col = next((c for c in ("pred", "score", "blended_score", "ridge_pred")
                      if c in df.columns), None)
    if score_col is not None:
        df = df.sort_values(score_col, ascending=False)
    return [_norm(c) for c in df["code"].tolist()]


def load_codes(cfg: RealtimeConfig) -> list[str]:
    """返回去重后的订阅代码列表（6 位）。

    优先级（保持"名单在前、持仓在后"的稳定顺序，便于上限夹紧时优先保住名单）：
      1) Top10 名单(mobile_snapshot) ∪ 持仓   —— 主路径
      2) 预测 top(active_quant_short_predictions) ∪ 持仓 —— 主路径为空时兜底(保旧行为)
      3) 兜底股票池(universe_file)             —— 都空时最后兜底
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def _extend(codes: Iterable[str]) -> None:
        for c in codes:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)

    _extend(_read_top10_groups(cfg.mobile_snapshot_file))
    holdings = _read_lines(cfg.holdings_file)
    _extend(holdings)

    # Top10 快照缺失/为空：退回按预测分取 top（旧行为），仍并入持仓。
    if not ordered:
        _extend(_read_predictions(cfg.predictions_file))
        _extend(holdings)

    if not ordered:
        _extend(_read_lines(cfg.universe_file))

    if cfg.max_subscribe and len(ordered) > cfg.max_subscribe:
        ordered = ordered[: cfg.max_subscribe]
    return ordered
