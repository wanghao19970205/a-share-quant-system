"""前复权(qfq)口径一致性维护：全量重建 / 月度除权票整条重拉。

背景（为什么需要这个模块）：
qfq 归一化到「因子表最新交易日」，每次新除权 base 前移，整条历史的 qfq 值都应重算。
但例行日更是增量合并（``--lookback-days 5`` 只重写最近几行），除权票的老行仍用除权前
的 base、新行用除权后的 base → 同一只票历史 qfq 口径分裂。这与数据源无关：免费源
akshare 每次也是全量 qfq，同一天除权前后两次拉取的 qfq 值不同，只要「存储 + 增量合并」
就会分裂。

两个模式：
  full    ：一次性全量重建 universe 内所有票 2018→今 的 qfq 价格（覆盖写）。
            部署本次复权修复后跑一次，抹平历史遗留的混合口径。
  ex-div  ：默认。用券商后复权因子检测最近 --window-days 内发生除权（因子跳变）的票，
            只对这些票整条重拉覆盖。供月度 cron 常态维护，成本低。

除权检测信号 = 后复权因子在窗口内发生跳变，精确等价于「该票 qfq base 变了」，
且天然覆盖分红/送转/配股/拆股所有情形。券商不可用时 ex-div 模式无法检测，直接退出
（不静默跳过，避免误以为已维护）。

容器内跑（务必挂后台 + 关行情磁盘缓存拿最新价）：
  docker exec -d -e CACHE_TTL_KLINE=0 a-scheduler-1 \\
      python3 -m quant.refresh_qfq --mode full   # 一次性全量
  docker exec -d -e CACHE_TTL_KLINE=0 a-scheduler-1 \\
      python3 -m quant.refresh_qfq --mode ex-div  # 月度增量（默认）
看日志：docker exec a-scheduler-1 tail -f /app/logs/refresh-qfq.out.log
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from quant import config, datafeed, warehouse
from stock_analyzer import amazingdata_source

_FACTOR_BATCH = max(int(os.environ.get("AMAZINGDATA_KLINE_BATCH_SIZE", "200") or 200), 1)


def _factor_changed(series: "pd.Series | None", cutoff: pd.Timestamp) -> bool:
    """因子序列在 cutoff 之后是否发生跳变（= 该票 qfq base 变了，需整条重拉）。

    baseline = cutoff 之前最后一个因子值；与序列最新值比较，不同即判定除权。
    baseline 缺失（上市不足一个窗口/因子起点晚于 cutoff）也判为需重建（历史短，成本低）。
    """
    if series is None or series.empty:
        return False
    before = series[series.index < cutoff]
    if before.empty:
        return True
    baseline = float(before.iloc[-1])
    last = float(series.iloc[-1])
    return abs(last - baseline) > 1e-9 * max(1.0, abs(baseline))


def _changed_codes(codes: list[str], window_days: int) -> list[str]:
    """批量拉后复权因子，返回最近 window_days 内因子跳变（除权）的 6 位代码。"""
    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=window_days))
    mapping = {code: amazingdata_source._to_broker_code(code) for code in codes}
    items = list(mapping.items())
    changed: list[str] = []
    for offset in range(0, len(items), _FACTOR_BATCH):
        chunk = items[offset:offset + _FACTOR_BATCH]
        frame = amazingdata_source._get_factor_frame(tuple(bc for _, bc in chunk))
        if frame is None:
            print(f"[ex-div] factor batch {offset}-{offset + len(chunk)} 取因子失败: "
                  f"{amazingdata_source._last_error}", flush=True)
            continue
        for code, broker_code in chunk:
            series = amazingdata_source._factor_series(frame, broker_code)
            if _factor_changed(series, cutoff):
                changed.append(code)
    print(f"[ex-div] window_days={window_days} scanned={len(codes)} changed={len(changed)}",
          flush=True)
    return changed


def _rebuild_one(code: str) -> tuple[str, str, int]:
    """整条重拉 2018→今 的 qfq 价格并覆盖写（不 merge，避免残留旧口径行）。"""
    try:
        new = datafeed.daily_price(code, "20180101")  # 多源 qfq，券商优先
        if new is None or new.empty:
            return code, "EmptyData", 0
        warehouse.save_price(code, new.reset_index(drop=True))
        return code, "ok", len(new)
    except Exception as e:  # noqa: BLE001
        return code, type(e).__name__, 0


def _run_batch(codes: list[str], workers: int, label: str) -> dict:
    ok = fail = rows = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, status, n in ex.map(_rebuild_one, codes):
            if status == "ok":
                ok += 1
                rows += int(n or 0)
            else:
                fail += 1
                if len(failures) < 20:
                    failures.append(f"{code}:{status}")
    print(f"[{label}] rebuilt_ok={ok} fail={fail} rows={rows}"
          + (f" failures={failures}" if failures else ""), flush=True)
    return {"ok": ok, "fail": fail, "rows": rows, "failures": failures}


def _resolve_codes(universe: str, limit: int, codes_file: str | None) -> list[str]:
    if codes_file:
        with open(codes_file, encoding="utf-8") as fh:
            codes = sorted({line.strip() for line in fh
                            if re.fullmatch(r"\d{6}", line.strip())})
    else:
        u = config.UNIVERSES[universe]
        if u["kind"] == "mainboard_active":
            datafeed.refresh_mainboard_universe()
        codes = datafeed.universe(u["kind"], u["arg"])
    return codes[:limit] if limit else codes


def run(mode: str = "ex-div", universe: str = "mainboard_active", workers: int = 12,
        window_days: int = 35, limit: int = 0, codes_file: str | None = None) -> dict:
    config.ensure_dirs()
    codes = _resolve_codes(universe, limit, codes_file)
    print(f"[refresh-qfq] mode={mode} universe={universe} codes={len(codes)} "
          f"quant_dir={config.QUANT_DIR}", flush=True)

    if mode == "ex-div":
        if not datafeed.broker_available():
            print("[ex-div] AmazingData 不可用，无法检测除权（跳过；请检查券商配置）", flush=True)
            return {"mode": mode, "targets": 0, "reason": "broker-unavailable"}
        targets = _changed_codes(codes, window_days)
    elif mode == "full":
        targets = codes
    else:
        raise SystemExit(f"未知 mode={mode!r}，应为 full 或 ex-div")

    if not targets:
        print("[refresh-qfq] 无需重拉的票", flush=True)
        return {"mode": mode, "targets": 0}
    summary = _run_batch(targets, workers, f"rebuild-{mode}")
    summary["mode"] = mode
    summary["targets"] = len(targets)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="qfq 口径一致性维护：全量重建 / 月度除权票重拉")
    ap.add_argument("--mode", choices=["full", "ex-div"], default="ex-div")
    ap.add_argument("--universe", default="mainboard_active", choices=list(config.UNIVERSES))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--window-days", type=int, default=35,
                    help="ex-div 模式：检测最近 N 天内因子跳变（除权）的票")
    ap.add_argument("--limit", type=int, default=0, help="调试用，仅处理前 N 只")
    ap.add_argument("--codes-file", default="", help="仅处理文件中列出的 6 位代码")
    args = ap.parse_args()
    res = run(mode=args.mode, universe=args.universe, workers=args.workers,
              window_days=args.window_days, limit=args.limit,
              codes_file=args.codes_file or None)
    print("[done]", res, flush=True)
    # 券商 tgw 原生库退出析构可能 SIGSEGV，此时数据已落盘；跳过 native 析构干净退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
