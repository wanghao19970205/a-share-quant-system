"""A/B 训练目标口径对齐实验编排器（task #5）。

串行训练四条腿（baseline / buyin-mask / tradable-label / tradable-mask），唯一变量 = 训练目标口径，
其余超参全部对齐现役冠军。训完后，用**同一把可交易尺子**（cost=0.002、
filter_untradable=1、top_n=2）统一评测，输出对比表，看哪条腿在真实成本+可交易口径下最优。

四条腿分别补足乐观标签（close(T)→close(T+h)）的不同短板：
- baseline：控制组，假设买得进也卖得掉；
- buyin-mask：只补买入端（剔除封涨停买不进的训练样本）；
- tradable-label：只补卖出端（跌停封板顺延后的可实现收益作标签）；
- tradable-mask：买卖两端同时补足。

**只读实验，不改冠军 / manifest / 每日发布名单。** 详见 OPS_AB_TARGET_ALIGN_2026-07-25.md。

结论（2026-08-31 训四腿、2026-09-03 用 --eval-only 统一尺子重算，cost=0.002、
filter_untradable=1、top_n=2、231 期）：

    mode            sharpe  annual   maxDD   月胜率
    baseline        -1.346  -0.841  -0.829   0.083
    buyin-mask      -1.460  -0.581  -0.637   0.167
    tradable-label  -1.218  -0.727  -0.759   0.250
    tradable-mask   -0.981  -0.443  -0.538   0.333

口径对齐的方向是对的：两端都补的 tradable-mask 比 baseline 好 +39.8pp，月胜率从
8.3% 升到 33.3%，四条腿的排序也符合预期（补得越全越好）。但**没有一条腿可投**：
top_n=2 的日均换手 0.86-0.98，0.002 的往返成本按 252 期就是 -43% 到 -49%，
tradable-mask 的 -44.3% 基本等于纯成本，也就是口径对齐之后 gross alpha ≈ 0。
baseline 多亏的那 40pp 说明乐观标签不只是"高估"，是**主动有害**——它把模型推向
买不进的封板票，评测时被 filter 剔掉，剩下的是残渣。

所以这条线不再推进：不是"该不该修标签"，而是修完就没有信号了。V1/V2 的模型腿在
top_n=4 上判负 alpha 与此一致，收益改进的方向已转到 V7/V8 的无模型分位带。

用法（容器内，模块方式）：
    export QUANT_BT_COST_ROUNDTRIP=0.002
    python -m quant.target_ab_experiment                 # 全量 36 窗
    python -m quant.target_ab_experiment --recent-windows 2   # 小窗 dry-run

前置自检：遍历 /proc 发现 scheduled_workflow / full_train_batched 在跑则中止（抢 CPU/券商连接）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from quant import backtest, config, tradability, warehouse

MODEL_NAME = "ridge_lightgbm_ranker_ensemble"
MODES = ["baseline", "buyin-mask", "tradable-label", "tradable-mask"]
HORIZON = 1


def _quant_dir() -> Path:
    return Path(config.QUANT_DIR)


def _preflight_no_training_running() -> None:
    """遍历 /proc 自检：发现 scheduled_workflow / full_train_batched 在跑则中止。"""
    busy = []
    proc = Path("/proc")
    if not proc.exists():
        return
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            cl = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            continue
        if "scheduled_workflow" in cl or "full_train_batched" in cl:
            busy.append(f"{d.name}: {cl.strip()}")
    if busy:
        print("!! 检测到训练/月度在跑，先等它结束再跑本实验（抢 CPU/券商连接）：", flush=True)
        for b in busy:
            print(f"   {b}", flush=True)
        sys.exit(3)
    print("  无训练/月度在跑，可继续。", flush=True)


def _train_cmd(mode: str, prefix: str, cache_dir: Path, recent_windows: int,
               max_windows: int, train_months: int) -> list[str]:
    panel_name = f"factor_panel_mainboard_active_h{HORIZON}"
    cmd = [
        sys.executable, "-m", "quant.full_train_batched",
        "--name", panel_name,
        "--output-prefix", prefix,
        "--horizon", str(HORIZON),
        "--refresh-months", "0",          # 复用既有 prepared 面板，不拉数不重建
        "--universe-file", config.MAINBOARD_UNIVERSE_FILE,
        "--top-n", "2",
        "--ridge-quantile", "0.7",
        "--lgbm-weight", "0.85",
        "--ic-weight", "0.15",
        "--rank-vote-weight", "0.0",
        "--n-estimators", "200",
        "--learning-rate", "0.015",
        "--early-stopping-rounds", "40",
        "--model-threads", "12",
        "--decay-half-life-days", "60",
        "--min-weight", "0.03",
        "--rolling-factor-select",
        "--rolling-top-factors", "30",
        "--max-factor-ic-corr", "0.85",
        "--purge-horizon",
        "--train-months", str(train_months),
        "--train-target-mode", mode,
        "--window-cache-dir", str(cache_dir),
    ]
    if recent_windows > 0:
        cmd.extend(["--recent-windows", str(recent_windows)])
    if max_windows > 0:
        cmd.extend(["--max-windows", str(max_windows)])
    return cmd


def _pred_path_name(prefix: str) -> str:
    return f"{prefix}_bt_{MODEL_NAME}_predictions"


def _evaluate_leg(prefix: str, cost: float, top_n: int) -> dict:
    """用同一把可交易尺子评测某条腿：重 join 可交易口径列 → portfolio_from_predictions。

    baseline 腿的 predictions 没有 tradable_ret/buyable_close，这里为**三腿统一**从 price 重 join，
    保证评测口径完全一致（差异只来自训练目标口径，不来自评测口径）。"""
    pred = warehouse.load(_pred_path_name(prefix))
    if pred.empty:
        return {"prefix": prefix, "error": "predictions empty/missing"}
    pred = pred.copy()
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    codes = sorted(pred["code"].dropna().unique())
    trad = tradability.price_tradability(codes, [HORIZON])
    tret = f"tradable_ret_{HORIZON}d"
    keep = [c for c in ["code", "date", tret, "buyable_close", f"target_ret_{HORIZON}d"] if c in trad.columns]
    trad = trad[keep].drop_duplicates(["code", "date"])
    # 用重 join 的可交易列覆盖（避免训练期 join 的列与评测口径分裂）
    drop = [c for c in (tret, "buyable_close") if c in pred.columns]
    if drop:
        pred = pred.drop(columns=drop)
    pred = pred.merge(trad, on=["code", "date"], how="left")
    returns, _ = backtest.portfolio_from_predictions(
        pred, horizon=HORIZON, top_n=top_n,
        max_weight=1.0 / max(top_n, 1),
        filter_untradable=True, cost_roundtrip=cost,
    )
    summary = backtest.evaluate_returns(
        returns["ret"] if not returns.empty else pd.Series(dtype=float),
        periods_per_year=max(1, 252 // HORIZON),
    )
    # 逐月胜率（与 stability 软闸同口径的直观参考）
    monthly_win = None
    if not returns.empty and "ret" in returns.columns:
        r = returns.copy()
        r["date"] = pd.to_datetime(r["date"], errors="coerce")
        r = r.dropna(subset=["date"])
        if not r.empty:
            # 按月分组用 Period 别名，避免 pandas 2.2 起 resample("M") 被废弃报错。
            m = r.set_index("date")["ret"].groupby(
                lambda ts: ts.to_period("M")
            ).apply(lambda s: (1 + s).prod() - 1 if len(s) else float("nan"))
            m = m.dropna()
            if len(m):
                monthly_win = float((m > 0).mean())
    return {
        "prefix": prefix,
        "sharpe": summary.get("sharpe"),
        "annual_return": summary.get("annual_return"),
        "max_drawdown": summary.get("max_drawdown"),
        "win_rate": summary.get("win_rate"),
        "monthly_win_rate": monthly_win,
        "periods": summary.get("periods"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B 训练目标口径对齐实验")
    ap.add_argument("--recent-windows", type=int, default=36)
    ap.add_argument("--max-windows", type=int, default=0, help="dry-run 用，限制窗口数")
    ap.add_argument("--train-months", type=int, default=36)
    ap.add_argument("--top-n", type=int, default=2, help="统一评测的 top_n")
    ap.add_argument("--skip-preflight", action="store_true", help="跳过 /proc 自检（仅调试）")
    ap.add_argument("--eval-only", action="store_true", help="跳过训练，直接对已有产物评测")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="只跑指定腿，逗号分隔；默认四腿全跑")
    ap.add_argument("--tag", default="",
                    help="产物前缀后缀，用于与已有短样本产物并存，如 full")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"未知腿 {unknown}，可选 {MODES}")

    def _prefix(mode: str) -> str:
        base = f"ab_{mode.replace('-', '_')}"
        return f"{base}_{args.tag}" if args.tag else base

    cost = backtest.bt_cost_roundtrip()
    print(f"== A/B 实验 · cost_roundtrip={cost} · recent_windows={args.recent_windows} "
          f"max_windows={args.max_windows or 'all'} ==", flush=True)

    if not args.skip_preflight and not args.eval_only:
        _preflight_no_training_running()

    cache_root = _quant_dir() / "window_cache"
    if not args.eval_only:
        for mode in modes:
            prefix = _prefix(mode)
            cache_dir = cache_root / prefix
            pred_file = _quant_dir() / f"{_pred_path_name(prefix)}.parquet"
            if pred_file.exists():
                print(f"[skip] {mode}: 已存在 {pred_file.name}，幂等跳过训练（删文件可重跑）", flush=True)
                continue
            cmd = _train_cmd(mode, prefix, cache_dir, args.recent_windows, args.max_windows, args.train_months)
            print(f"[run ] {mode}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd, env=os.environ.copy())
            if rc != 0:
                print(f"!! {mode} 训练失败 rc={rc}，中止（前面成功的腿产物保留）", flush=True)
                sys.exit(rc)

    print("\n== 统一可交易评测（四腿同尺：filter_untradable=1, cost, top_n）==", flush=True)
    rows = []
    for mode in modes:
        prefix = _prefix(mode)
        res = _evaluate_leg(prefix, cost, args.top_n)
        res["mode"] = mode
        rows.append(res)
    table = pd.DataFrame(rows)
    cols = ["mode", "prefix", "sharpe", "annual_return", "max_drawdown", "win_rate",
            "monthly_win_rate", "periods"]
    cols = [c for c in cols if c in table.columns] + [c for c in table.columns if c not in cols]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[cols].to_string(index=False), flush=True)
    print("\n判读：baseline 是控制组；某变体 sharpe/月胜率显著高于 baseline 才值得考虑升级冠军"
          "（另走正规发布，本实验不落地）。", flush=True)


if __name__ == "__main__":
    main()
