"""清单来源诊断：不订阅、不登录，只打印各来源解析情况，定位清单为何为空/退兜底。

跑法（容器内）：
    python3 -m realtime.diag
输出：
    - 三路来源文件的解析路径 + 是否存在 + 各读到几只
    - load_codes 最终结果（受 max_subscribe 夹紧后）
    - 若走了兜底池会显式标红提示
纯读，不写任何文件，不碰券商。
"""
from __future__ import annotations

from pathlib import Path

from .config import load
from . import watchlist as wl


def _info(path: Path, n: int, label: str) -> None:
    exists = "存在" if path.exists() else "缺失"
    print(f"  [{label}] {path}  ({exists}, 读到 {n} 只)", flush=True)


def main() -> int:
    cfg = load()
    print("[diag] 清单来源解析：", flush=True)

    preds = wl._read_predictions(cfg.predictions_file)  # noqa: SLF001
    _info(cfg.predictions_file, len(preds), "选股清单")
    if preds:
        print(f"          前5 {preds[:5]}", flush=True)

    holds = wl._read_lines(cfg.holdings_file)            # noqa: SLF001
    _info(cfg.holdings_file, len(holds), "持仓")

    uni = wl._read_lines(cfg.universe_file)              # noqa: SLF001
    _info(cfg.universe_file, len(uni), "兜底池")

    final = wl.load_codes(cfg)
    used_fallback = (not preds and not holds) and bool(uni)
    print(f"[diag] load_codes 最终 {len(final)} 只 "
          f"(max_subscribe={cfg.max_subscribe})", flush=True)
    if used_fallback:
        print("[diag][WARN] 选股清单+持仓都为空 → 退回了全市场兜底池！"
              "需校准 predictions_file 路径或补 holdings 文件。", flush=True)
    else:
        print(f"[diag][OK] 走选股清单/持仓，前5 {final[:5]}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
