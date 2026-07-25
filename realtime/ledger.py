"""独立账本：把信号 + 快照关键值追加写 JSONL，与 quant_data 业务数据完全隔离。

- 每交易日一份文件：<ledger_dir>/signals_YYYYMMDD.jsonl
- 追加写，一行一条 JSON，永不覆盖既有数据。
- 只记录结构化数值，不写任何券商 Token/凭证。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import RealtimeConfig
from .strategy import Signal


class Ledger:
    def __init__(self, cfg: RealtimeConfig):
        self._cfg = cfg
        cfg.ensure_dirs()

    def _path(self) -> Path:
        day = time.strftime("%Y%m%d")
        return self._cfg.ledger_dir / f"signals_{day}.jsonl"

    def record(self, sig: Signal, notified: bool) -> None:
        rec = {
            "ts": sig.ts,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sig.ts)),
            "code": sig.code,
            "kind": sig.kind,
            "level": sig.level,
            "reason": sig.reason,
            "metrics": sig.metrics,
            "notified": notified,
        }
        try:
            with open(self._path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 - 记账失败不影响主流程
            print(f"[ledger] 写入失败: {type(e).__name__}", flush=True)
