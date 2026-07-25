"""订阅清单加载：选股清单 ∪ 持仓，去重、规范 6 位、按上限夹紧。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import RealtimeConfig


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


def _read_predictions(path: Path) -> list[str]:
    """读最新一期短线预测，按预测分 pred 降序返回代码列表（模型看多最强在前）。

    预测文件是全 A 打分表（每日数千只），必须按分排序，才能在 max_subscribe
    夹紧时保住"最强的 N 只"，而不是代码号最小的 N 只。
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

    优先级：选股清单 ∪ 持仓；都空则退回兜底股票池。
    保持"清单在前、持仓在后"的稳定顺序，便于上限夹紧时优先保住选股清单。
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def _extend(codes: Iterable[str]) -> None:
        for c in codes:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)

    _extend(_read_predictions(cfg.predictions_file))
    _extend(_read_lines(cfg.holdings_file))

    if not ordered:
        _extend(_read_lines(cfg.universe_file))

    if cfg.max_subscribe and len(ordered) > cfg.max_subscribe:
        ordered = ordered[: cfg.max_subscribe]
    return ordered
