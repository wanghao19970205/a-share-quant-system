"""启动期股票名称映射：code -> 中文简称，供推送标题展示（如 600519 贵州茅台）。

只在引擎启动时构建一次（离线读本地 meta parquet），盘中完全不碰网络/IO，
与 reference.build 同期一次性预取的设计一致。任何来源缺失都优雅降级为空 map
→ 推送退回只显示代码（当前行为），绝不因缺名称崩溃。

名称来源优先级（全离线，不依赖 akshare/券商网络）：
1) snapshots/all_a_stock_meta.parquet（全 A code+name，daily 由 all_a_meta 维护）
2) snapshots/watchlist.txt（自选股名称表，watchlist_info_map 同源格式）
"""
from __future__ import annotations

import os
from pathlib import Path


def _snapshot_dir() -> Path:
    # 与 stock_analyzer.all_a_meta._snapshot_dir 同约定：env SNAPSHOT_DIR，默认 ./snapshots。
    return Path(os.environ.get("SNAPSHOT_DIR", "snapshots"))


def _from_meta_parquet(codes: set[str]) -> dict[str, str]:
    path = _snapshot_dir() / "all_a_stock_meta.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}
    try:
        df = pd.read_parquet(path, columns=["code", "name"])
    except Exception:  # noqa: BLE001 - 列缺失/文件损坏都降级
        return {}
    out: dict[str, str] = {}
    for code, name in zip(df["code"].astype(str).str.zfill(6), df["name"]):
        nm = str(name or "").strip()
        if nm and nm.lower() != "nan" and code in codes:
            out[code] = nm
    return out


def _from_watchlist_txt(codes: set[str]) -> dict[str, str]:
    path = _snapshot_dir() / "watchlist.txt"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            body = line.split("#", 1)[0].strip()
            if not body:
                continue
            parts = body.split(maxsplit=1)
            code = str(parts[0]).strip().zfill(6)
            if len(code) != 6 or not code.isdigit() or code not in codes:
                continue
            name = parts[1].strip() if len(parts) > 1 else ""
            if name:
                out[code] = name
    except Exception:  # noqa: BLE001
        return out
    return out


def load_name_map(codes: list[str]) -> dict[str, str]:
    """为订阅代码集合构建 code->name 映射（缺失的票不进 map，展示时退回代码）。"""
    want = {str(c).strip().zfill(6) for c in codes if str(c).strip()}
    if not want:
        return {}
    names = _from_meta_parquet(want)
    # 补齐 meta 未覆盖的票（自选股名称表兜底）。
    for code, nm in _from_watchlist_txt(want).items():
        names.setdefault(code, nm)
    return names
