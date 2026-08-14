"""Scheduled quant workflow: daily update -> full-A training -> publish active model.

This module is a conservative orchestration entry for cron/launchd. It keeps the
training artifact under its experiment prefix, then publishes a stable active
prediction file for stock_analyzer to read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant import config
from quant import datafeed
from quant import watchlist_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ridge_lightgbm_ranker_ensemble"
DEFAULT_OUTPUT_PREFIX = "scheduled_dual_short_swing_ne200_es40"
ACTIVE_STYLE_FILES = {
    "short_1_3": "active_quant_short_predictions.parquet",
    "swing_7_15": "active_quant_swing_predictions.parquet",
}
LEGACY_ACTIVE_FILE = "active_quant_predictions.parquet"
_ACTIVE_MANIFEST = "active_quant_model.json"

FINAL_TRADE_STYLE_ORDER = ["short_1_3", "swing_7_15"]
FINAL_TRADE_STYLES = {
    "short_1_3": {
        "label": "短线",
        "holding_days": "1天",
        "profile": "short_stable",
        "horizons": [1, 2],
        "score_params": {
            "ic_weight": 0.03,
            "top_n": 2,
            "gross_exposure": 0.30,
            "ridge_quantile": 0.55,
            "pred_quantile": None,
        },
        "target_price": "take_profit_1",
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_short_h1_watchlist_grid.parquet",
        "note": "短线最终口径：训练目标1日(尾盘T买、T+1卖)，白名单top2优先，按1/2日网格选参(稳健性缓冲)。",
    },
    "swing_7_15": {
        "label": "波段",
        "holding_days": "7-15天",
        "profile": "v2_attack",
        "horizons": [7, 10, 15],
        "score_params": {
            "ic_weight": 0.15,
            "top_n": 3,
            "max_weight": 0.07,
            "ridge_quantile": 0.50,
            "pred_quantile": None,
        },
        "target_price": "take_profit_2",
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_swing_h10_watchlist_grid.parquet",
        "note": "波段口径：7-15天持仓、top3、take_profit_2；打分来源改为复用短线模型预测（不再单独训练h10）。",
    },
}
TRAINING_PROFILE = {
    "label": "short_target_with_derived_trade_styles",
    "source": "short_model_predictions_reused_by_derived_styles",
    "note": "定时任务只训练短线目标；波段保留独立持有期评估，但打分和排名复用短线模型预测。",
}

# 波段不再单独训练：直接复用短线模型的预测作为波段信号来源（省去 h10 训练，约减半训练耗时）。
# 波段仍保留自己的持仓周期/止盈口径展示，只是打分/排名来自短线模型。
DERIVE_FROM = {"swing_7_15": "short_1_3"}
TRAIN_STYLE_ORDER = [s for s in FINAL_TRADE_STYLE_ORDER if s not in DERIVE_FROM]


def _published_prediction_horizons(horizons: dict[str, int]) -> dict[str, int]:
    """Report each style's actual model horizon, including derived styles."""
    return {
        style: int(horizons[DERIVE_FROM.get(style, style)])
        for style in FINAL_TRADE_STYLE_ORDER
    }

# 派生风格（波段）在短线预测之上做白名单网格搜索，挑选最优参数以提升效果。
SWING_GRID_HORIZONS = [7, 10, 15]
# 短线自身也在短线预测上做白名单网格搜索（1/2日），在当前评估口径下重选最优参数。
SHORT_GRID_HORIZONS = [1, 2]


def _watchlist_file(args: argparse.Namespace) -> Path:
    return Path(config.MAINBOARD_UNIVERSE_FILE)


def _optimization_universe(args: argparse.Namespace) -> set[str]:
    path = _watchlist_file(args)
    codes = watchlist_grid._read_watchlist(path)  # noqa: SLF001
    if not codes:
        raise RuntimeError(f"mainboard optimization universe is missing or empty: {path}")
    return codes


def _training_panel_name(horizon: int) -> str:
    return f"factor_panel_mainboard_active_h{horizon}"


def _daily_short_neighbor(best: pd.DataFrame, champion: dict) -> pd.DataFrame:
    """Restrict daily adaptation to a small, non-drifting neighborhood of the monthly champion."""
    if best.empty:
        return best
    mask = pd.Series(True, index=best.index)
    limits = {
        "ic_weight": 0.03,
        "top_n": 1.0,
        "ridge_quantile": 0.05,
        "pred_quantile": 0.05,
        "naive_weight": 0.10,
    }
    for key, limit in limits.items():
        anchor = watchlist_grid._empty_to_none(champion.get(key))  # noqa: SLF001
        if anchor is None or key not in best.columns:
            continue
        values = pd.to_numeric(best[key], errors="coerce")
        mask &= values.notna() & ((values - float(anchor)).abs() <= limit + 1e-12)
    if champion.get("gross_exposure") is not None and "gross_exposure" in best.columns:
        gross = pd.to_numeric(best["gross_exposure"], errors="coerce")
        mask &= (gross - float(champion["gross_exposure"])).abs() <= 1e-12
    return best[mask].copy()


def run_short_grid(output_prefixes: dict[str, str], args: argparse.Namespace,
                   champion_params: dict | None = None) -> dict | None:
    """Run the short grid, optionally selecting only near the monthly champion."""
    quant_dir = _quant_dir()
    src = quant_dir / f"{output_prefixes['short_1_3']}_bt_{MODEL_NAME}_predictions.parquet"
    if not src.exists():
        print(f"[short_grid] 跳过：短线预测文件不存在 {src.name}", flush=True)
        return None
    eval_file = quant_dir / FINAL_TRADE_STYLES["short_1_3"]["evaluation_file"]
    best_file = eval_file.with_name(eval_file.stem + "_best.parquet")
    watchlist = _optimization_universe(args)
    print(f"[short_grid] predictions={src.name} horizons={SHORT_GRID_HORIZONS} "
          f"mainboard_universe={len(watchlist)} -> {eval_file.name}", flush=True)
    best = None
    try:
        _grid, best = watchlist_grid.run_grid(
            predictions=src,
            template=eval_file,   # 已存在则作为参数模板；不存在则用内置网格
            output=eval_file,
            best_output=best_file,
            horizons=SHORT_GRID_HORIZONS,
            kind="short",
            watchlist=watchlist,
            positive_only=True,
            neighborhood=champion_params,
            fixed_params={
                "rebalance_stride": max(
                    int(getattr(args, "short_rebalance_stride", 1)), 1
                ),
                **(
                    {"hold_rank_buffer": max(int(args.short_hold_rank_buffer), 0)}
                    if bool(getattr(args, "short_hold_rank_buffer_explicit", False))
                    else {}
                ),
            },
        )

    except Exception as e:  # noqa: BLE001
        print(f"[short_grid] 失败（保留原参数）：{type(e).__name__}: {e}", flush=True)
        return None
    if best is None or best.empty:
        return None
    row = best.iloc[0]
    params: dict = {}
    for k in ("ic_weight", "top_n", "gross_exposure", "slot_weight",
              "ridge_quantile", "pred_quantile", "naive_weight", "rebalance_stride",
              "hold_rank_buffer"):
        if k not in best.columns:
            continue
        v = watchlist_grid._empty_to_none(row.get(k))  # noqa: SLF001
        if v is None:
            params[k] = None
        elif k == "top_n":
            params[k] = int(v)
        else:
            params[k] = float(v)
    print(f"[short_grid] best score={row.get('selection_score'):.4f} "
          f"avg_sharpe={row.get('avg_sharpe')} params={params}", flush=True)
    return params


def _run(cmd: list[str], env: dict[str, str], dry_run: bool = False) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _quant_dir() -> Path:
    return Path(config.QUANT_DIR)


def _configure_quant_data_dir(path: str, env: dict[str, str]) -> str:
    resolved = config.configure_quant_dir(path)
    env["QUANT_DATA_DIR"] = resolved
    os.environ["QUANT_DATA_DIR"] = resolved
    return resolved


def _atomic_copy(src: Path, dst: Path) -> None:
    """Publish an independent file atomically; active artifacts must never hard-link training output."""
    if not src.exists():
        raise FileNotFoundError(src)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def merge_active_predictions(
    active_path: Path,
    fresh_path: Path,
    dst: Path,
    fresh_frame: pd.DataFrame | None = None,
) -> None:
    """Merge fresh predictions into active history without mutating shared input."""
    if fresh_frame is None:
        if not fresh_path.exists():
            raise FileNotFoundError(fresh_path)
        fresh = pd.read_parquet(fresh_path)
        fresh["code"] = fresh["code"].astype(str)
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.normalize()
        fresh = fresh.dropna(subset=["date"])
    else:
        fresh = fresh_frame.copy()

    if active_path.exists():
        old = pd.read_parquet(active_path)
        old["code"] = old["code"].astype(str)
        old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
        old = old.dropna(subset=["date"])
        # Preserve the union of columns so schema drift never drops data.
        for col in fresh.columns:
            if col not in old.columns:
                old[col] = pd.NA
        for col in old.columns:
            if col not in fresh.columns:
                fresh[col] = pd.NA
        fresh = fresh[old.columns]
        fresh_keys = set(zip(fresh["code"], fresh["date"]))
        keep = old[~old.set_index(["code", "date"]).index.isin(fresh_keys)]
        merged = pd.concat([keep, fresh], ignore_index=True)
    else:
        merged = fresh

    merged = (
        merged.sort_values(["date", "code"])
        .drop_duplicates(subset=["code", "date"], keep="last")
        .reset_index(drop=True)
    )
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    merged.to_parquet(tmp)
    os.replace(tmp, dst)


def _files_identical(left: Path, right: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """Compare active artifacts without loading either full file into memory."""
    try:
        if not left.exists() or not right.exists() or left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_file, right.open("rb") as right_file:
            while True:
                left_chunk = left_file.read(chunk_size)
                right_chunk = right_file.read(chunk_size)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


def _merge_short_and_legacy(
    short_active: Path,
    legacy_active: Path,
    source_predictions: Path,
    fresh: pd.DataFrame,
) -> str:
    """Merge once when short and legacy histories are byte-identical."""
    if _files_identical(short_active, legacy_active):
        merge_active_predictions(
            short_active, source_predictions, short_active, fresh_frame=fresh)
        _atomic_copy(short_active, legacy_active)
        return "shared-history"
    merge_active_predictions(
        short_active, source_predictions, short_active, fresh_frame=fresh)
    merge_active_predictions(
        legacy_active, source_predictions, legacy_active, fresh_frame=fresh)
    return "independent-history"


def _parquet_max_date(path: Path) -> pd.Timestamp | None:
    try:
        parquet = pq.ParquetFile(path)
        date_index = parquet.schema_arrow.get_field_index("date")
        if date_index >= 0:
            maxima = []
            for group_index in range(parquet.metadata.num_row_groups):
                statistics = parquet.metadata.row_group(group_index).column(date_index).statistics
                if statistics is not None and statistics.has_min_max:
                    maxima.append(statistics.max)
            if maxima:
                value = pd.to_datetime(pd.Series(maxima), errors="coerce").max()
                if pd.notna(value):
                    return pd.Timestamp(value).normalize()
    except Exception:  # noqa: BLE001
        pass

    try:
        dates = pd.read_parquet(path, columns=["date"])
    except Exception:  # noqa: BLE001
        return None
    if dates.empty:
        return None
    value = pd.to_datetime(dates["date"], errors="coerce").max()
    return pd.Timestamp(value).normalize() if pd.notna(value) else None


def _latest_price_date() -> pd.Timestamp | None:
    latest: pd.Timestamp | None = None
    for path in Path(config.PRICE_DIR).glob("*.parquet"):
        value = _parquet_max_date(path)
        if value is not None and (latest is None or value > latest):
            latest = value
    return latest


def _price_freshness(codes: list[str] | None = None) -> tuple[pd.Timestamp | None, float, int]:
    """Return latest date and coverage, restricted to the active update universe."""
    paths = {
        code: Path(config.PRICE_DIR) / f"{code}.parquet"
        for code in (codes or [])
    } if codes else {
        path.stem: path for path in Path(config.PRICE_DIR).glob("*.parquet")
    }
    latests: list[pd.Timestamp | None] = []
    for path in paths.values():
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:  # noqa: BLE001
            latests.append(None)
            continue
        if df.empty:
            latests.append(None)
            continue
        d = pd.to_datetime(df["date"], errors="coerce").max()
        latests.append(pd.Timestamp(d).normalize() if pd.notna(d) else None)
    valid = [value for value in latests if value is not None]
    if not valid:
        return None, 0.0, len(latests)
    overall = max(valid)
    coverage = float(sum(value == overall for value in latests) / max(len(latests), 1))
    return overall, coverage, len(latests)


def _write_recovery_codes(codes: list[str]) -> Path:
    fd, name = tempfile.mkstemp(prefix="daily-update-recovery-", suffix=".txt")
    os.close(fd)
    path = Path(name)
    path.write_text("".join(f"{code}\n" for code in codes), encoding="utf-8")
    return path


def _prediction_max_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").max()
    return pd.Timestamp(d).normalize() if pd.notna(d) else None


def assert_active_is_latest(
    active_path: Path,
    strict: bool = True,
    price_latest: pd.Timestamp | None = None,
) -> None:
    if price_latest is None:
        price_latest = _latest_price_date()
    pred_latest = _prediction_max_date(active_path)
    print(f"[publish:check] price_latest={price_latest.date() if price_latest is not None else ''} "
          f"prediction_latest={pred_latest.date() if pred_latest is not None else ''}", flush=True)
    if not strict:
        return
    if price_latest is None or pred_latest is None:
        raise RuntimeError("无法校验 active 预测日期：行情或预测文件日期为空")
    if pred_latest < price_latest:
        raise RuntimeError(
            f"active 预测日期落后于最新行情：prediction={pred_latest.date()} price={price_latest.date()}。"
            "请确认训练后实盘推理已生成最新交易日预测。"
        )


def _style_artifact(output_prefix: str, active_file: str, horizon: int) -> dict:
    quant_dir = _quant_dir()
    src = quant_dir / f"{output_prefix}_bt_{MODEL_NAME}_predictions.parquet"
    active = quant_dir / active_file
    return {
        "horizon": horizon,
        "active_path": active,
        "predictions_file": active.name,
        "source_predictions_file": src.name,
        "summary_file": f"{output_prefix}_bt_{MODEL_NAME}_summary.parquet",
        "returns_file": f"{output_prefix}_bt_{MODEL_NAME}_returns.parquet",
        "holdings_file": f"{output_prefix}_bt_{MODEL_NAME}_holdings.parquet",
    }


def _select_promotion_trial(trials: list[dict]) -> tuple[dict, int]:
    """Choose the first selection-ranked pass; use holdout scores only for failure diagnostics."""
    passed = [trial for trial in trials if trial.get("promote")]
    if passed:
        return dict(passed[0]), len(passed)
    if trials:
        diagnostic = sorted(trials, key=lambda trial: (
            float(trial.get("avg_sharpe_gain") if trial.get("avg_sharpe_gain") is not None else -1e9),
            float((trial.get("stability_gate") or {}).get("monthly_win_rate") or 0.0),
            float(trial.get("worst_drawdown_change") if trial.get("worst_drawdown_change") is not None else -1e9),
        ), reverse=True)
        return dict(diagnostic[0]), 0
    return {"promote": False, "reason": "candidate_evaluation_empty"}, 0


def _promotion_gate(output_prefixes: dict[str, str], args: argparse.Namespace,
                    styles_to_evaluate: list[str] | None = None) -> dict:
    """Compare candidate strategies with active champions on a common recent holdout."""
    quant_dir = _quant_dir()
    manifest_path = quant_dir / _ACTIVE_MANIFEST
    if not manifest_path.exists():
        return {"promote": False, "reason": "active_manifest_missing"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"promote": False, "reason": f"active_manifest_invalid: {type(exc).__name__}"}
    watchlist = _optimization_universe(args)
    styles = manifest.get("final_trade_styles") or {}
    reports: dict[str, dict] = {}
    evaluation_styles = styles_to_evaluate or FINAL_TRADE_STYLE_ORDER
    for style in evaluation_styles:
        cfg = styles.get(style) or {}
        active_file = cfg.get("predictions_file") or ACTIVE_STYLE_FILES[style]
        source_style = DERIVE_FROM.get(style, style)
        candidate_file = f"{output_prefixes[source_style]}_bt_{MODEL_NAME}_predictions.parquet"
        active_path = quant_dir / str(active_file)
        candidate_path = quant_dir / candidate_file
        if not active_path.exists() or not candidate_path.exists():
            reports[style] = {"promote": False, "reason": "prediction_file_missing"}
            continue
        try:
            active_dates = pd.to_datetime(pd.read_parquet(active_path, columns=["date"])["date"], errors="coerce")
            candidate_dates = pd.to_datetime(pd.read_parquet(candidate_path, columns=["date"])["date"], errors="coerce")
        except Exception as exc:  # noqa: BLE001
            reports[style] = {"promote": False, "reason": f"prediction_read_failed: {type(exc).__name__}"}
            continue
        common_end = min(active_dates.max(), candidate_dates.max()).normalize() + pd.Timedelta(days=1)
        candidate_start = max(active_dates.min(), candidate_dates.min())
        holdout_start = common_end - pd.DateOffset(months=args.promotion_holdout_months)
        if candidate_start >= holdout_start:
            reports[style] = {"promote": False, "reason": "insufficient_selection_period"}
            continue
        # horizons 以代码 FINAL_TRADE_STYLES 为真源，避免读到 manifest 里的陈旧口径([1,2,3])。
        horizons = [int(x) for x in (FINAL_TRADE_STYLES.get(style, {}).get("horizons")
                                     or ([1, 2] if style == "short_1_3" else [7, 10, 15]))]
        kind = "short" if style == "short_1_3" else "swing"
        incumbent_params = dict(cfg.get("champion_score_params") or cfg.get("score_params") or {})
        incumbent_params.setdefault("rebalance_stride", int(cfg.get("rebalance_stride", 1)))
        incumbent_params.setdefault("hold_rank_buffer", int(cfg.get("hold_rank_buffer", 0)))

        # Select candidate parameters strictly before the untouched holdout period.
        selection_file = quant_dir / f"{output_prefixes[source_style]}_{style}_selection.parquet"
        selection_best_file = selection_file.with_name(selection_file.stem + "_best.parquet")
        template = quant_dir / str(cfg.get("evaluation_file") or selection_file.name)
        try:
            fixed_model_params = {
                key: incumbent_params[key]
                for key in ("lgbm_weight", "elastic_weight", "catboost_weight", "extra_trees_weight")
                if incumbent_params.get(key) is not None
            }
            fixed_model_params["rebalance_stride"] = max(int(
                args.short_rebalance_stride if style == "short_1_3"
                else args.swing_rebalance_stride
            ), 1)
            requested_buffer = (
                getattr(args, "short_hold_rank_buffer", 0)
                if style == "short_1_3"
                else getattr(args, "swing_hold_rank_buffer", 0)
            )
            buffer_explicit = (
                bool(getattr(args, "short_hold_rank_buffer_explicit", False))
                if style == "short_1_3"
                else bool(getattr(args, "swing_hold_rank_buffer_explicit", False))
            )
            if buffer_explicit:
                fixed_model_params["hold_rank_buffer"] = max(int(requested_buffer), 0)
            _selection_grid, selection_best = watchlist_grid.run_grid(
                predictions=candidate_path,
                template=template,
                output=selection_file,
                best_output=selection_best_file,
                horizons=horizons,
                kind=kind,
                watchlist=watchlist,
                positive_only=True,
                start_date=candidate_start,
                end_date=holdout_start,
                fixed_params=fixed_model_params,
            )
        except Exception as exc:  # noqa: BLE001
            reports[style] = {"promote": False, "reason": f"parameter_selection_failed: {type(exc).__name__}"}
            continue
        if selection_best.empty:
            reports[style] = {"promote": False, "reason": "parameter_selection_empty"}
            continue
        param_keys = ("lgbm_weight", "ic_weight", "elastic_weight", "catboost_weight", "extra_trees_weight", "top_n", "gross_exposure",
                      "slot_weight", "max_weight", "ridge_quantile", "pred_quantile", "naive_weight",
                      "rebalance_stride", "hold_rank_buffer")

        candidate_pred, candidate_prepared = watchlist_grid.prepare_fixed_context(
            candidate_path, horizons, watchlist, holdout_start, common_end)
        active_pred, active_prepared = watchlist_grid.prepare_fixed_context(
            active_path, horizons, watchlist, holdout_start, common_end)
        baseline_eval = watchlist_grid.evaluate_prepared_params(
            active_prepared, incumbent_params, horizons, kind, True)
        baseline_returns = watchlist_grid.evaluate_prepared_returns(
            active_pred, incumbent_params, horizons, kind, True)

        trials: list[dict] = []
        for selection_rank, (_, candidate_row) in enumerate(selection_best.head(20).iterrows(), start=1):
            candidate_params: dict = {}
            for key in param_keys:
                if key not in selection_best.columns:
                    continue
                value = watchlist_grid._empty_to_none(candidate_row.get(key))  # noqa: SLF001
                if value is None:
                    candidate_params[key] = None
                elif key in ("top_n", "rebalance_stride", "hold_rank_buffer"):
                    candidate_params[key] = int(value)
                else:
                    candidate_params[key] = float(value)
            candidate_eval = watchlist_grid.evaluate_prepared_params(
                candidate_prepared, candidate_params, horizons, kind, True)
            decision = watchlist_grid.promotion_decision(
                candidate_eval, baseline_eval,
                min_sharpe_gain=args.promotion_min_sharpe_gain,
                max_drawdown_worsening=args.promotion_max_drawdown_worsening,
                min_improved_horizons=1,
            )
            candidate_returns = watchlist_grid.evaluate_prepared_returns(
                candidate_pred, candidate_params, horizons, kind, True)
            stability = watchlist_grid.stability_decision(
                candidate_returns, baseline_returns,
                min_monthly_win_rate=args.promotion_min_monthly_win_rate)
            decision["daily_gate_passed"] = bool(decision.get("promote"))
            decision["stability_gate"] = stability
            decision["promote"] = bool(decision.get("promote") and stability.get("passed"))
            if not decision["promote"] and decision.get("reason") == "passed":
                decision["reason"] = "stability_threshold_not_met"
            decision["selection_rank"] = selection_rank
            decision["selection_score"] = float(candidate_row.get("selection_score", 0.0))
            decision["candidate_params"] = candidate_params
            trials.append(decision)
            print(f"[promotion] style={style} candidate={selection_rank}/{min(20, len(selection_best))} "
                  f"passed={decision['promote']} gain={decision.get('avg_sharpe_gain')}", flush=True)

        decision, passed_count = _select_promotion_trial(trials)
        decision["promote"] = passed_count > 0
        if not passed_count and decision.get("reason") == "passed":
            decision["reason"] = "no_top_candidate_passed"
        decision["selection_range"] = [str(candidate_start.date()), str((holdout_start - pd.Timedelta(days=1)).date())]
        decision["holdout_range"] = [str(holdout_start.date()), str((common_end - pd.Timedelta(days=1)).date())]
        decision["incumbent_params"] = incumbent_params
        decision["candidates_evaluated"] = len(trials)
        decision["candidates_passed"] = passed_count
        decision["candidate_trials"] = trials
        reports[style] = decision
    promote = bool(reports) and all(bool(r.get("promote")) for r in reports.values())
    report = {"promote": promote, "mode": "candidate-upgrade", "styles": reports,
              "evaluated_at": dt.datetime.now().isoformat(timespec="seconds")}
    report_path = quant_dir / "candidate_promotion_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, report_path)
    return report


def publish_short_champion(source_predictions: Path, source_prefix: str,
                           short_score_params: dict, training_params: dict,
                           merge_history: bool = False,
                           refresh_swing: bool = False,
                           publish_predictions: bool = True,
                           champion_score_params: dict | None = None) -> list[Path]:
    """Atomically upgrade only the short champion; preserve the swing artifact and parameters verbatim.

    merge_history=True (daily incumbent refresh) merges the fresh single-window
    predictions into existing active history instead of replacing it wholesale.

    refresh_swing=True additionally re-derives the swing predictions from the SAME
    fresh short predictions (swing derives from short), keeping the swing champion's
    approved score_params / holding period frozen — only the prediction data advances
    to the latest trading day so swing signals do not go stale.
    """
    quant_dir = _quant_dir()
    manifest_path = quant_dir / _ACTIVE_MANIFEST
    if not source_predictions.exists() or not manifest_path.exists():
        raise FileNotFoundError(source_predictions if not source_predictions.exists() else manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    short_active = quant_dir / ACTIVE_STYLE_FILES["short_1_3"]
    legacy_active = quant_dir / LEGACY_ACTIVE_FILE
    fresh: pd.DataFrame | None = None
    if publish_predictions:
        assert_active_is_latest(source_predictions, strict=True)
        if merge_history:
            fresh_started = time.perf_counter()
            fresh = pd.read_parquet(source_predictions)
            fresh["code"] = fresh["code"].astype(str)
            fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.normalize()
            fresh = fresh.dropna(subset=["date"])
            print(
                f"[publish:timing] stage=fresh_read seconds={time.perf_counter() - fresh_started:.2f} "
                f"rows={len(fresh)}",
                flush=True,
            )
            merge_started = time.perf_counter()
            merge_mode = _merge_short_and_legacy(
                short_active, legacy_active, source_predictions, fresh)
            print(
                f"[publish:timing] stage=short_legacy_merge mode={merge_mode} "
                f"seconds={time.perf_counter() - merge_started:.2f}",
                flush=True,
            )
        else:
            _atomic_copy(source_predictions, short_active)
            _atomic_copy(source_predictions, legacy_active)
    latest = _prediction_max_date(short_active)
    styles = deepcopy(manifest.get("final_trade_styles") or FINAL_TRADE_STYLES)
    short_cfg = deepcopy(styles.get("short_1_3") or FINAL_TRADE_STYLES["short_1_3"])
    short_cfg["score_params"] = dict(short_score_params)
    short_cfg["champion_score_params"] = dict(
        champion_score_params if champion_score_params is not None
        else short_cfg.get("champion_score_params") or short_score_params
    )
    short_cfg["predictions_file"] = short_active.name
    short_cfg["prediction_horizon"] = int(training_params.get("short_horizon", 1))
    short_cfg["rebalance_stride"] = int(
        training_params.get("short_rebalance_stride", 1)
    )
    short_cfg["hold_rank_buffer"] = int(
        training_params.get("short_hold_rank_buffer", 0)
    )
    short_cfg["horizons"] = list(FINAL_TRADE_STYLES["short_1_3"]["horizons"])
    styles["short_1_3"] = short_cfg
    artifacts = deepcopy(manifest.get("style_artifacts") or {})
    short_artifact = deepcopy(artifacts.get("short_1_3") or {})
    short_artifact.update({
        "horizon": int(training_params.get("short_horizon", 1)),
        "rebalance_stride": int(training_params.get("short_rebalance_stride", 1)),
        "hold_rank_buffer": int(training_params.get("short_hold_rank_buffer", 0)),
        "predictions_file": short_active.name,
        "source_predictions_file": source_predictions.name,
        "summary_file": f"{source_prefix}_bt_{MODEL_NAME}_summary.parquet",
        "returns_file": f"{source_prefix}_bt_{MODEL_NAME}_returns.parquet",
        "holdings_file": f"{source_prefix}_bt_{MODEL_NAME}_holdings.parquet",
        "prediction_latest_date": latest.strftime("%Y-%m-%d") if latest is not None else "",
    })
    artifacts["short_1_3"] = short_artifact
    if refresh_swing and publish_predictions:
        # 波段派生自短线：其 score_params / 持仓周期等审定参数保持冻结，只用当天新鲜的短线
        # 预测刷新波段预测文件，避免波段信号停在旧交易日。
        swing_active = quant_dir / ACTIVE_STYLE_FILES["swing_7_15"]
        if merge_history:
            swing_started = time.perf_counter()
            merge_active_predictions(
                swing_active, source_predictions, swing_active, fresh_frame=fresh)
            print(
                f"[publish:timing] stage=swing_merge "
                f"seconds={time.perf_counter() - swing_started:.2f}",
                flush=True,
            )
        else:
            _atomic_copy(source_predictions, swing_active)
        swing_latest = _prediction_max_date(swing_active)
        swing_cfg = deepcopy(styles.get("swing_7_15") or FINAL_TRADE_STYLES["swing_7_15"])
        swing_cfg["predictions_file"] = swing_active.name
        swing_cfg["prediction_horizon"] = short_artifact["horizon"]
        swing_cfg["rebalance_stride"] = int(
            training_params.get("swing_rebalance_stride", 1)
        )
        swing_cfg["hold_rank_buffer"] = int(
            training_params.get("swing_hold_rank_buffer", 0)
        )
        styles["swing_7_15"] = swing_cfg
        swing_artifact = deepcopy(artifacts.get("swing_7_15") or {})
        swing_artifact.update({
            "horizon": short_artifact["horizon"],
            "rebalance_stride": int(training_params.get("swing_rebalance_stride", 1)),
            "hold_rank_buffer": int(training_params.get("swing_hold_rank_buffer", 0)),
            "predictions_file": swing_active.name,
            "source_predictions_file": source_predictions.name,
            "summary_file": short_artifact["summary_file"],
            "returns_file": short_artifact["returns_file"],
            "holdings_file": short_artifact["holdings_file"],
            "prediction_latest_date": (
                swing_latest.strftime("%Y-%m-%d") if swing_latest is not None else ""),
        })
        artifacts["swing_7_15"] = swing_artifact
    prediction_horizons = {
        style: int(artifact["horizon"])
        for style, artifact in artifacts.items()
        if artifact.get("horizon") is not None
    }
    params = deepcopy(manifest.get("params") or {})
    params.update(training_params)
    payload = deepcopy(manifest)
    payload.update({
        "output_prefix": source_prefix,
        "predictions_file": LEGACY_ACTIVE_FILE,
        "style_artifacts": artifacts,
        "published_at": dt.datetime.now().isoformat(timespec="seconds"),
        "prediction_horizon": short_artifact["horizon"],
        "prediction_horizons": prediction_horizons,
        "prediction_latest_date": latest.strftime("%Y-%m-%d") if latest is not None else manifest.get("prediction_latest_date", ""),
        "final_trade_styles": styles,
        "params": params,
    })
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    published = [short_active, legacy_active]
    if refresh_swing:
        published.append(quant_dir / ACTIVE_STYLE_FILES["swing_7_15"])
    return published


def validate_publish_dry_run(
    output_prefixes: dict[str, str],
    horizons: dict[str, int],
) -> dict[str, str]:
    """Validate existing publication sources without copying or writing artifacts."""
    price_latest = _latest_price_date()
    published_horizons = _published_prediction_horizons(horizons)
    statuses: dict[str, str] = {}
    for style in FINAL_TRADE_STYLE_ORDER:
        src_style = DERIVE_FROM.get(style, style)
        artifact = _style_artifact(
            output_prefixes[src_style], ACTIVE_STYLE_FILES[style], published_horizons[style]
        )
        src = _quant_dir() / artifact["source_predictions_file"]
        if not src.exists():
            statuses[style] = "source-missing-not-validated"
            print(f"[publish:dry-run] {style}: source missing, freshness not validated", flush=True)
            continue
        assert_active_is_latest(src, strict=True, price_latest=price_latest)
        statuses[style] = "fresh"
        print(f"[publish:dry-run] {style}: freshness passed", flush=True)
    return statuses


def publish_active_models(output_prefixes: dict[str, str], horizons: dict[str, int], params: dict,
                          swing_score_params: dict | None = None,
                          short_score_params: dict | None = None,
                          dry_run: bool = False) -> list[Path]:
    quant_dir = _quant_dir()
    manifest = quant_dir / "active_quant_model.json"
    style_artifacts: dict[str, dict] = {}
    active_paths: list[Path] = []
    if dry_run:
        validate_publish_dry_run(output_prefixes, horizons)
    price_latest = None if dry_run else _latest_price_date()
    published_horizons = _published_prediction_horizons(horizons)

    for style in FINAL_TRADE_STYLE_ORDER:
        active_file = ACTIVE_STYLE_FILES[style]
        # 波段等派生风格的预测来源指向其基准风格（短线）的产物文件。
        src_style = DERIVE_FROM.get(style, style)
        artifact = _style_artifact(
            output_prefixes[src_style], active_file, published_horizons[style]
        )
        src = quant_dir / artifact["source_predictions_file"]
        active = artifact["active_path"]
        print(f"[publish:{style}] {src.name} -> {active.name}", flush=True)
        if not dry_run:
            assert_active_is_latest(src, strict=True, price_latest=price_latest)

            _atomic_copy(src, active)
        pred_latest = None if dry_run else _prediction_max_date(active)
        style_artifacts[style] = {
            k: v for k, v in artifact.items()
            if k != "active_path"
        }
        style_artifacts[style]["prediction_latest_date"] = pred_latest.strftime("%Y-%m-%d") if pred_latest is not None else ""
        active_paths.append(active)

    if not dry_run:
        # Keep the legacy single-file entry pointing at the short model for older callers.
        _atomic_copy(active_paths[0], quant_dir / LEGACY_ACTIVE_FILE)
        styles = deepcopy(FINAL_TRADE_STYLES)
        # 短线/波段均采用网格搜索得到的最优参数（若有），使实盘打分/排名自动吃到调优结果。
        if short_score_params:
            base_sp = dict(styles["short_1_3"].get("score_params") or {})
            base_sp.update({k: v for k, v in short_score_params.items() if v is not None})
            styles["short_1_3"]["score_params"] = base_sp
            styles["short_1_3"]["champion_score_params"] = dict(base_sp)
        if swing_score_params:
            base_sp = dict(styles["swing_7_15"].get("score_params") or {})
            base_sp.update({k: v for k, v in swing_score_params.items() if v is not None})
            styles["swing_7_15"]["score_params"] = base_sp
        for style, artifact in style_artifacts.items():
            styles[style]["predictions_file"] = artifact["predictions_file"]
            styles[style]["prediction_horizon"] = artifact["horizon"]
        latest_dates = [d for d in (a.get("prediction_latest_date") for a in style_artifacts.values()) if d]
        payload = {
            "model": "active_quant",
            "source_model": MODEL_NAME,
            "output_prefix": output_prefixes.get("short_1_3", ""),
            "output_prefixes": output_prefixes,
            "predictions_file": LEGACY_ACTIVE_FILE,
            "style_artifacts": style_artifacts,
            "published_at": dt.datetime.now().isoformat(timespec="seconds"),
            "prediction_horizon": published_horizons["short_1_3"],
            "prediction_horizons": published_horizons,
            "prediction_latest_date": min(latest_dates) if latest_dates else "",
            "price_latest_date": price_latest.strftime("%Y-%m-%d") if price_latest is not None else "",
            "training_profile": TRAINING_PROFILE,
            "default_trade_styles": FINAL_TRADE_STYLE_ORDER,
            "final_trade_styles": styles,
            "params": params,
        }
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, manifest)
    return active_paths


def run_daily_update(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.skip_daily_update:
        print("[daily_update] skipped", flush=True)
        return
    cmd = [
        sys.executable, "-m", "quant.daily_update",
        "--universe", args.universe,
        "--workers", str(args.update_workers),
        "--lookback-days", str(args.lookback_days),
        "--event-window-days", str(args.event_window_days),
        "--snapshot-dir", args.snapshot_dir,
    ]
    if args.skip_valuation:
        cmd.append("--skip-valuation")
    if args.skip_events:
        cmd.append("--skip-events")
    if args.skip_fundamentals:
        cmd.append("--skip-fundamentals")
    if args.skip_snapshots:
        cmd.append("--skip-snapshots")
    # 收盘后（A股 15:00 收盘）强制重抓最新交易日K线：盘中/午间跑批会先写入当日盘中价，
    # 而 daily_update 仅按日期判新鲜度、同日即跳过，导致收盘后的运行沿用盘中价。
    # 容器时区为 Asia/Shanghai，此处用本地时间判断是否已收盘。
    force_latest = bool(getattr(args, "force_latest_price", False))
    if not force_latest and dt.datetime.now().time() >= dt.time(15, 0):
        force_latest = True
    if force_latest:
        cmd.append("--force-latest")
    if getattr(args, "intraday_spot", False):
        cmd.append("--intraday-spot")
    print("[run] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    before = _latest_price_date()
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
    if rc == 0:
        return
    # 券商 tgw 原生库常在解释器退出(atexit/析构)时段错误(SIGSEGV, rc=-11 或 128+11=139)，
    # 此时行情/事件数据其实已全部落盘。但段错误也可能发生在拉取中途（大量个股仍停在旧交易日），
    # 因此不能只看全库最新日期，必须叠加「最新交易日覆盖率」判断：达标才算完备、可安全训练；
    # 否则视为残缺、中止，避免用残缺数据训练。
    if rc in (-signal.SIGSEGV, 128 + int(signal.SIGSEGV)):
        u = config.UNIVERSES[args.universe]
        if u["kind"] == "mainboard_active":
            datafeed.refresh_mainboard_universe()
        codes = datafeed.universe(u["kind"], u["arg"])
        after, coverage, n = _price_freshness(codes)
        today = pd.Timestamp.now().normalize()
        advanced = after is not None and before is not None and after > before
        fresh = after is not None and after >= today - pd.Timedelta(days=5)
        MIN_COVERAGE = 0.85
        if (advanced or fresh) and coverage < MIN_COVERAGE and after is not None:
            missing: list[str] = []
            for code in codes:
                path = Path(config.PRICE_DIR) / f"{code}.parquet"
                try:
                    dates = pd.read_parquet(path, columns=["date"])
                    latest = pd.to_datetime(dates["date"], errors="coerce").max()
                except Exception:  # noqa: BLE001
                    latest = None
                if latest is None or pd.isna(latest) or pd.Timestamp(latest).normalize() < after:
                    missing.append(code)
            if missing:
                recovery_file = _write_recovery_codes(missing)
                recovery_cmd = [
                    sys.executable, "-m", "quant.daily_update",
                    "--universe", args.universe,
                    "--workers", str(min(int(args.update_workers), 4)),
                    "--lookback-days", str(args.lookback_days),
                    "--snapshot-dir", args.snapshot_dir,
                    "--skip-valuation", "--skip-events", "--skip-fundamentals", "--skip-snapshots",
                    "--force-latest", "--codes-file", str(recovery_file),
                ]
                recovery_env = dict(env)
                recovery_env["AMAZINGDATA_AUTO_LOGIN"] = "0"
                print(
                    f"[daily_update] tgw SIGSEGV rc={rc}; recovering missing="
                    f"{len(missing)}/{len(codes)} via free sources workers="
                    f"{min(int(args.update_workers), 4)}",
                    flush=True,
                )
                try:
                    recovery_rc = subprocess.run(
                        recovery_cmd, cwd=PROJECT_ROOT, env=recovery_env,
                    ).returncode
                finally:
                    recovery_file.unlink(missing_ok=True)
                if recovery_rc == 0:
                    after, coverage, n = _price_freshness(codes)
        if (advanced or fresh) and coverage >= MIN_COVERAGE:
            print(
                f"[daily_update] tgw 退出段错误(rc={rc})，价格仓库已落盘至 "
                f"{after.date() if after is not None else '?'}、本次股票池覆盖率 "
                f"{coverage:.1%}（{n} 只）≥ {MIN_COVERAGE:.0%}；继续训练。",
                flush=True,
            )
            return
        raise RuntimeError(
            f"daily_update 段错误(rc={rc}) 后数据不完备：latest="
            f"{after.date() if after is not None else None}、本次股票池覆盖率={coverage:.1%}"
            f"（{n} 只，阈值 {MIN_COVERAGE:.0%}，advanced={advanced} fresh={fresh}）；"
            f"拒绝在残缺数据上训练，请补齐后再训。"
        )
    raise subprocess.CalledProcessError(rc, cmd)


def _output_prefixes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "short_1_3": f"{args.output_prefix}_short_h{args.short_horizon}",
        "swing_7_15": f"{args.output_prefix}_swing_h{args.swing_horizon}",
    }


def _horizons(args: argparse.Namespace) -> dict[str, int]:
    return {"short_1_3": int(args.short_horizon), "swing_7_15": int(args.swing_horizon)}


def _uses_rolling_training(args: argparse.Namespace) -> bool:
    return args.strategy_mode == "candidate-upgrade" or bool(getattr(args, "incumbent_rolling_factor_select", False))


def _uses_purge_training(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "purge_horizon", False)
        or args.strategy_mode == "candidate-upgrade"
        or getattr(args, "incumbent_purge_horizon", False)
    )


def _minimum_validation_months(horizon: int) -> int:
    if int(horizon) >= 10:
        return 3
    if int(horizon) >= 3:
        return 2
    return 1


def _uses_elastic_training(args: argparse.Namespace) -> bool:
    return args.strategy_mode == "candidate-upgrade" or bool(getattr(args, "incumbent_elastic_net", False))


def _uses_catboost_training(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "catboost_ranker", False) or getattr(args, "incumbent_catboost_ranker", False))


def _uses_extra_trees_training(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "extra_trees", False) or getattr(args, "incumbent_extra_trees", False))


def _restore_explicit_portfolio_params(
    args: argparse.Namespace,
    cli_values: dict[str, int],
    argv: list[str],
) -> None:
    options = {
        "short_rebalance_stride": "--short-rebalance-stride",
        "swing_rebalance_stride": "--swing-rebalance-stride",
        "short_hold_rank_buffer": "--short-hold-rank-buffer",
        "swing_hold_rank_buffer": "--swing-hold-rank-buffer",
    }
    for attribute, option in options.items():
        if option not in argv:
            continue
        lower = 1 if attribute.endswith("rebalance_stride") else 0
        setattr(args, attribute, max(int(cli_values[attribute]), lower))


def _apply_incumbent_training_config(args: argparse.Namespace) -> None:
    """Keep daily refresh on the last approved training method without invoking the promotion gate."""
    args.incumbent_rolling_factor_select = False
    args.incumbent_purge_horizon = False
    args.incumbent_elastic_net = False
    args.incumbent_catboost_ranker = False
    args.incumbent_extra_trees = False
    args.incumbent_style_configs = {}
    manifest_path = _quant_dir() / _ACTIVE_MANIFEST
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        params = manifest.get("params") or {}
        args.incumbent_style_configs = manifest.get("final_trade_styles") or {}
        if params.get("model_threads") is not None:
            args.model_threads = int(params["model_threads"])
    except Exception:  # noqa: BLE001
        return
    if params.get("short_rebalance_stride") is not None:
        args.short_rebalance_stride = max(int(params["short_rebalance_stride"]), 1)
    if params.get("swing_rebalance_stride") is not None:
        args.swing_rebalance_stride = max(int(params["swing_rebalance_stride"]), 1)
    if params.get("short_hold_rank_buffer") is not None:
        args.short_hold_rank_buffer = max(int(params["short_hold_rank_buffer"]), 0)
    if params.get("swing_hold_rank_buffer") is not None:
        args.swing_hold_rank_buffer = max(int(params["swing_hold_rank_buffer"]), 0)
    if bool(params.get("elastic_net")):
        args.incumbent_elastic_net = True
        args.elastic_alpha = float(params.get("elastic_alpha") or args.elastic_alpha)
        args.elastic_l1_ratio = float(params.get("elastic_l1_ratio") or args.elastic_l1_ratio)
    if bool(params.get("catboost_ranker")):
        args.incumbent_catboost_ranker = True
        args.catboost_estimators = int(params.get("catboost_estimators") or args.catboost_estimators)
        args.catboost_learning_rate = float(params.get("catboost_learning_rate") or args.catboost_learning_rate)
        args.catboost_max_train_rows = int(
            params.get("catboost_max_train_rows") or args.catboost_max_train_rows)
    if bool(params.get("extra_trees")):
        args.incumbent_extra_trees = True
        args.extra_trees_estimators = int(params.get("extra_trees_estimators") or args.extra_trees_estimators)
        args.extra_trees_max_train_rows = int(params.get("extra_trees_max_train_rows") or args.extra_trees_max_train_rows)
    args.incumbent_purge_horizon = bool(params.get("purge_horizon"))
    if bool(params.get("rolling_factor_select")):
        args.incumbent_rolling_factor_select = True
        args.rolling_top_factors = int(params.get("rolling_top_factors") or args.rolling_top_factors)
        args.max_factor_ic_corr = float(params.get("max_factor_ic_corr") or args.max_factor_ic_corr)
        args.recent_windows = int(params.get("recent_windows") or args.recent_windows or 24)
        print(f"[train] approved rolling strategy: recent_windows={args.recent_windows} "
              f"top_factors={args.rolling_top_factors} max_ic_corr={args.max_factor_ic_corr}", flush=True)


def _append_research_training_args(cmd: list[str], args: argparse.Namespace) -> None:
    flag_options = {
        "include_trading_gap_risk": "--include-trading-gap-risk",
        "strict_calendar_factors": "--strict-calendar-factors",
        "strict_announcement_lag": "--strict-announcement-lag",
        "strict_pit_min_price_rows": "--strict-pit-min-price-rows",
        "strict_execution_labels": "--strict-execution-labels",
        "require_selection_provenance": "--require-selection-provenance",
        "pit_index_code": "--pit-index-code",
        "enforce_c30_gates": "--enforce-c30-gates",
    }
    for attribute, option in flag_options.items():
        value = getattr(args, attribute, None)
        if attribute == "pit_index_code":
            if value:
                cmd.extend([option, str(value)])
        elif bool(value):
            cmd.append(option)
    train_target_mode = str(getattr(args, "train_target_mode", "baseline"))
    if train_target_mode != "baseline":
        cmd.extend(["--train-target-mode", train_target_mode])
    for attribute, option in (
        ("min_adv20", "--min-adv20"),
        ("min_listing_sessions", "--min-listing-sessions"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            cmd.extend([option, str(value)])


def run_training(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.skip_train:
        print("[train] skipped", flush=True)
        return
    output_prefixes = _output_prefixes(args)
    horizons = _horizons(args)
    for style in TRAIN_STYLE_ORDER:
        horizon = horizons[style]
        rebalance_stride = (
            args.short_rebalance_stride if style == "short_1_3"
            else args.swing_rebalance_stride
        )
        hold_rank_buffer = (
            getattr(args, "short_hold_rank_buffer", 0) if style == "short_1_3"
            else getattr(args, "swing_hold_rank_buffer", 0)
        )
        prefix = output_prefixes[style]
        panel_name = _training_panel_name(horizon)
        score_params = FINAL_TRADE_STYLES[style].get("score_params", {})
        incumbent_cfg = (getattr(args, "incumbent_style_configs", {}) or {}).get(style) or {}
        if args.strategy_mode == "incumbent-refresh":
            approved = incumbent_cfg.get("champion_score_params") or incumbent_cfg.get("score_params")
            if approved:
                score_params = approved
        top_n = int(score_params.get("top_n", 3))
        ridge_quantile = score_params.get("ridge_quantile", args.ridge_quantile)
        ic_weight = score_params.get("ic_weight", args.ic_weight)
        lgbm_weight = score_params.get("lgbm_weight", args.lgbm_weight)
        max_weight = score_params.get("max_weight")
        if max_weight is None and score_params.get("gross_exposure") is not None:
            max_weight = float(score_params["gross_exposure"]) / max(top_n, 1)
        if max_weight is None:
            max_weight = args.max_weight
        validation_months = int(getattr(args, "validation_months", 1))
        required_validation_months = _minimum_validation_months(horizon)
        if validation_months < required_validation_months:
            raise ValueError(
                f"horizon={horizon} requires validation_months>="
                f"{required_validation_months}; got {validation_months}"
            )
        print(
            f"[train:{style}] horizon={horizon} output_prefix={prefix} panel={panel_name} "
            f"top_n={top_n} max_weight={max_weight} ridge_quantile={ridge_quantile} "
            f"lgbm_weight={lgbm_weight} ic_weight={ic_weight} "
            f"validation_months={validation_months} purge_horizon={_uses_purge_training(args)}",
            flush=True,
        )
        cmd = [
            sys.executable, "-m", "quant.full_train_batched",
            "--name", panel_name,
            "--output-prefix", prefix,
            "--horizon", str(horizon),
            "--rebalance-stride", str(max(int(rebalance_stride), 1)),
            "--hold-rank-buffer", str(max(int(hold_rank_buffer), 0)),
            "--refresh-months", str(args.refresh_months),
            "--universe-file", config.MAINBOARD_UNIVERSE_FILE,
            "--top-n", str(top_n),
            "--n-estimators", str(args.n_estimators),
            "--learning-rate", str(args.learning_rate),
            "--early-stopping-rounds", str(args.early_stopping_rounds),
            "--model-threads", str(args.model_threads),
            "--ridge-quantile", str(ridge_quantile),
            "--lgbm-weight", str(lgbm_weight),
            "--ic-weight", str(ic_weight),
            "--rank-vote-weight", str(args.rank_vote_weight),
            "--max-weight", str(max_weight),
            "--decay-half-life-days", str(args.decay_half_life_days),
            "--min-weight", str(args.min_weight),
            "--month-cache-size", str(args.month_cache_size),
            "--train-months", str(args.train_months),
            "--validation-months", str(validation_months),
        ]
        if args.positive_only:
            cmd.append("--positive-only")
        if args.skip_windows:
            cmd.extend(["--skip-windows", str(args.skip_windows)])
        if args.max_windows:
            cmd.extend(["--max-windows", str(args.max_windows)])
        if args.recent_windows:
            cmd.extend(["--recent-windows", str(args.recent_windows)])
        if getattr(args, "expanding_train", False):
            cmd.append("--expanding-train")
        if _uses_rolling_training(args):
            cmd.extend([
                "--rolling-factor-select",
                "--rolling-top-factors", str(args.rolling_top_factors),
                "--max-factor-ic-corr", str(args.max_factor_ic_corr),
            ])
        if _uses_purge_training(args):
            cmd.append("--purge-horizon")
        _append_research_training_args(cmd, args)
        if _uses_elastic_training(args):
            cmd.extend([
                "--elastic-net",
                "--elastic-alpha", str(args.elastic_alpha),
                "--elastic-l1-ratio", str(args.elastic_l1_ratio),
            ])
        if _uses_catboost_training(args):
            cmd.extend([
                "--catboost-ranker",
                "--catboost-estimators", str(args.catboost_estimators),
                "--catboost-learning-rate", str(args.catboost_learning_rate),
                "--catboost-max-train-rows", str(args.catboost_max_train_rows),
            ])
        if _uses_extra_trees_training(args):
            cmd.extend([
                "--extra-trees",
                "--extra-trees-estimators", str(args.extra_trees_estimators),
                "--extra-trees-max-train-rows", str(args.extra_trees_max_train_rows),
            ])
        if getattr(args, "window_cache", True):
            cache_dir = _quant_dir() / "window_cache" / prefix
            cmd.extend(["--window-cache-dir", str(cache_dir)])
        _run(cmd, env, dry_run=args.dry_run)


def build_params(args: argparse.Namespace) -> dict:
    return {
        "short_horizon": args.short_horizon,
        "swing_horizon": args.swing_horizon,
        "short_rebalance_stride": max(int(args.short_rebalance_stride), 1),
        "swing_rebalance_stride": max(int(args.swing_rebalance_stride), 1),
        "short_hold_rank_buffer": max(int(getattr(args, "short_hold_rank_buffer", 0)), 0),
        "swing_hold_rank_buffer": max(int(getattr(args, "swing_hold_rank_buffer", 0)), 0),
        "horizons": _horizons(args),
        "refresh_months": args.refresh_months,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "early_stopping_rounds": args.early_stopping_rounds,
        "ridge_quantile": args.ridge_quantile,
        "lgbm_weight": args.lgbm_weight,
        "ic_weight": args.ic_weight,
        "rank_vote_weight": args.rank_vote_weight,
        "max_weight": args.max_weight,
        "positive_only": args.positive_only,
        "decay_half_life_days": args.decay_half_life_days,
        "min_weight": args.min_weight,
        "month_cache_size": args.month_cache_size,
        "model_threads": args.model_threads,
        "skip_windows": args.skip_windows,
        "max_windows": args.max_windows,
        "recent_windows": args.recent_windows,
        "strategy_mode": args.strategy_mode,
        "validation_months": int(getattr(args, "validation_months", 1)),
        "rolling_factor_select": _uses_rolling_training(args),
        "purge_horizon": _uses_purge_training(args),
        "include_trading_gap_risk": bool(getattr(args, "include_trading_gap_risk", False)),
        "strict_calendar_factors": bool(getattr(args, "strict_calendar_factors", False)),
        "strict_announcement_lag": bool(getattr(args, "strict_announcement_lag", False)),
        "strict_execution_labels": bool(getattr(args, "strict_execution_labels", False)),
        "pit_index_code": str(getattr(args, "pit_index_code", "") or ""),
        "enforce_c30_gates": bool(getattr(args, "enforce_c30_gates", False)),
        "train_target_mode": str(getattr(args, "train_target_mode", "baseline")),
        "min_adv20": getattr(args, "min_adv20", None),
        "min_listing_sessions": getattr(args, "min_listing_sessions", None),
        "rolling_top_factors": args.rolling_top_factors,
        "max_factor_ic_corr": args.max_factor_ic_corr,
        "elastic_net": _uses_elastic_training(args),
        "elastic_alpha": args.elastic_alpha,
        "elastic_l1_ratio": args.elastic_l1_ratio,
        "catboost_ranker": _uses_catboost_training(args),
        "catboost_estimators": args.catboost_estimators,
        "catboost_learning_rate": args.catboost_learning_rate,
        "catboost_max_train_rows": args.catboost_max_train_rows,
        "extra_trees": _uses_extra_trees_training(args),
        "extra_trees_estimators": args.extra_trees_estimators,
        "extra_trees_max_train_rows": args.extra_trees_max_train_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run daily update, train full-A model, and publish active quant signal")
    ap.add_argument("--quant-data-dir", default=os.environ.get("QUANT_DATA_DIR", str(PROJECT_ROOT / "quant_data" / "full_a_2018_wide")))
    ap.add_argument("--snapshot-dir", default=os.environ.get("SNAPSHOT_DIR", str(PROJECT_ROOT / "snapshots")))
    ap.add_argument("--universe", default="mainboard_active", choices=list(config.UNIVERSES))
    ap.add_argument("--update-workers", type=int, default=12)
    ap.add_argument("--lookback-days", type=int, default=5)
    ap.add_argument("--event-window-days", type=int, default=30)
    ap.add_argument("--skip-valuation", action="store_true")
    ap.add_argument("--skip-events", action="store_true", help="skip slow event APIs during scheduled light runs")
    ap.add_argument("--skip-fundamentals", action="store_true")
    ap.add_argument("--skip-snapshots", action="store_true")
    ap.add_argument("--skip-daily-update", action="store_true")
    ap.add_argument("--force-latest-price", action="store_true",
                    help="强制重抓最新交易日K线（覆盖盘中/午间写入的临时价）；默认收盘后自动开启")
    ap.add_argument("--intraday-spot", action="store_true",
                    help="仅盘中轻量任务使用全市场快照覆盖当日临时K线")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-publish", action="store_true")
    ap.add_argument("--strategy-mode", choices=["incumbent-refresh", "candidate-upgrade"],
                    default="incumbent-refresh",
                    help="incumbent-refresh 每日发布最新数据；candidate-upgrade 仅在增益闸门通过后升级策略")
    ap.add_argument("--short-only-upgrade", action="store_true",
                    help="candidate mode: evaluate and publish only the short champion; preserve swing unchanged")
    ap.add_argument("--rolling-top-factors", type=int, default=30)
    ap.add_argument("--max-factor-ic-corr", type=float, default=0.85)
    ap.add_argument("--elastic-alpha", type=float, default=0.001)
    ap.add_argument("--elastic-l1-ratio", type=float, default=0.5)
    ap.add_argument("--catboost-ranker", action="store_true", help="train a CatBoost ranker shadow leg")
    ap.add_argument("--catboost-estimators", type=int, default=200)
    ap.add_argument("--catboost-learning-rate", type=float, default=0.03)
    ap.add_argument("--catboost-max-train-rows", type=int, default=300000,
                    help="maximum recent complete query-group rows used by each CatBoost window")
    ap.add_argument("--extra-trees", action="store_true", help="train an ExtraTrees shadow leg")
    ap.add_argument("--extra-trees-estimators", type=int, default=120)
    ap.add_argument("--extra-trees-max-train-rows", type=int, default=300000)
    ap.add_argument("--promotion-holdout-months", type=int, default=6)
    ap.add_argument("--promotion-min-sharpe-gain", type=float, default=0.10)
    ap.add_argument("--promotion-max-drawdown-worsening", type=float, default=0.02)
    ap.add_argument("--promotion-min-monthly-win-rate", type=float, default=0.40,
                    help="stability soft-gate monthly win-rate floor")
    ap.add_argument("--skip-swing-grid", action="store_true", help="跳过波段白名单网格搜索（默认在发布前执行）")
    ap.add_argument("--skip-short-grid", action="store_true", help="跳过短线白名单网格搜索（默认在发布前执行）")

    ap.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    ap.add_argument("--horizon", type=int, default=1, help="legacy alias for --short-horizon")
    ap.add_argument("--short-horizon", type=int, default=None, help="短线训练目标天数；默认沿用 --horizon")
    ap.add_argument("--swing-horizon", type=int, default=10, help="波段训练目标天数")
    ap.add_argument(
        "--short-rebalance-stride", type=int, default=1,
        help="短线评估每 N 个权威交易会话调仓；1 保持每日评估",
    )
    ap.add_argument(
        "--swing-rebalance-stride", type=int, default=1,
        help="波段评估每 N 个权威交易会话调仓；1 保持每日评估",
    )
    ap.add_argument(
        "--short-hold-rank-buffer", type=int, default=0,
        help="固定短线 Top-N + buffer；省略时训练使用 0、网格搜索已注册维度",
    )
    ap.add_argument(
        "--swing-hold-rank-buffer", type=int, default=0,
        help="固定波段 Top-N + buffer；省略时训练使用 0、网格搜索已注册维度",
    )
    ap.add_argument("--refresh-months", type=int, default=1)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.015)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--model-threads", type=int, default=0,
                    help="LightGBM worker threads; 0 keeps library auto-detection")
    ap.add_argument("--ridge-quantile", type=float, default=0.35)
    ap.add_argument("--lgbm-weight", type=float, default=0.85)
    ap.add_argument("--ic-weight", type=float, default=0.10)
    ap.add_argument("--rank-vote-weight", type=float, default=0.0)
    ap.add_argument("--max-weight", type=float, default=0.09)
    ap.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decay-half-life-days", type=float, default=60.0)
    ap.add_argument("--min-weight", type=float, default=0.03)
    ap.add_argument("--month-cache-size", type=int, default=30)
    ap.add_argument("--train-months", type=int, default=24,
                    help="length of each rolling walk-forward train window in months; larger = each "
                         "independent model sees more history (peak memory is bounded by ONE window)")
    ap.add_argument("--validation-months", type=int, default=1,
                    help="validation window months; horizons >=3 require 2 and horizons >=10 require 3")
    ap.add_argument("--purge-horizon", action="store_true",
                    help="enable label-horizon purge independently of rolling factor selection")
    ap.add_argument("--include-trading-gap-risk", action="store_true")
    ap.add_argument("--strict-calendar-factors", action="store_true")
    ap.add_argument("--strict-announcement-lag", action="store_true")
    ap.add_argument("--strict-pit-min-price-rows", action="store_true")
    ap.add_argument("--strict-execution-labels", action="store_true")
    ap.add_argument("--pit-index-code", default="",
                    help="resolve a PIT index universe at each training window anchor")
    ap.add_argument("--require-selection-provenance", action="store_true")
    ap.add_argument("--enforce-c30-gates", action="store_true")
    ap.add_argument(
        "--train-target-mode", default="baseline",
        choices=["baseline", "buyin-mask", "tradable-label", "open-label", "open-buyin-mask"],
    )
    ap.add_argument("--min-adv20", type=float, default=None)
    ap.add_argument("--min-listing-sessions", type=int, default=None)
    ap.add_argument("--skip-windows", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--recent-windows", type=int, default=0,
                    help="candidate mode: automatically use the latest N walk-forward windows")
    ap.add_argument("--expanding-train", action="store_true",
                    help="train the selected window(s) on all prepared history from 2018 instead of a rolling train-month window")
    ap.add_argument("--allow-stale-active", action="store_true", help="do not fail if published active predictions lag latest local price date")
    ap.add_argument("--no-window-cache", dest="window_cache", action="store_false",
                    help="disable per-window prediction caching (recompute every walk-forward window)")
    ap.set_defaults(window_cache=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.short_hold_rank_buffer_explicit = "--short-hold-rank-buffer" in sys.argv[1:]
    args.swing_hold_rank_buffer_explicit = "--swing-hold-rank-buffer" in sys.argv[1:]
    if args.short_horizon is None:
        args.short_horizon = args.horizon

    env = os.environ.copy()
    args.quant_data_dir = _configure_quant_data_dir(args.quant_data_dir, env)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SNAPSHOT_DIR"] = args.snapshot_dir
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    os.environ["QUANT_DATA_DIR"] = args.quant_data_dir

    print(f"[workflow] quant_data_dir={args.quant_data_dir}", flush=True)
    print(f"[workflow] snapshot_dir={args.snapshot_dir}", flush=True)
    cli_recent_windows = int(args.recent_windows or 0)  # explicit CLI intent, captured before manifest override
    cli_portfolio_params = {
        "short_rebalance_stride": int(args.short_rebalance_stride),
        "swing_rebalance_stride": int(args.swing_rebalance_stride),
        "short_hold_rank_buffer": int(args.short_hold_rank_buffer),
        "swing_hold_rank_buffer": int(args.swing_hold_rank_buffer),
    }
    _apply_incumbent_training_config(args)
    _restore_explicit_portfolio_params(args, cli_portfolio_params, sys.argv[1:])
    if args.strategy_mode == "incumbent-refresh":
        # Daily/weekly refresh: rolling walk-forward. Default is the latest 24 windows (each a
        # 24-month train slice); peak memory is bounded by ONE window, so this is memory-safe on
        # a 12GB box regardless of window count (windows are independent fits over the month cache).
        # An explicit --recent-windows on the CLI wins over the incumbent manifest value; pass a
        # value >= total windows (e.g. 999) to run the full 2018..now history from window 0.
        # (A single expanding 2018..now fit concatenates ~8.5M rows and OOMs a 12GB box.)
        if cli_recent_windows:
            args.recent_windows = cli_recent_windows
        elif not args.recent_windows:
            args.recent_windows = 24
        args.expanding_train = False
        print(f"[train] incumbent-refresh: recent_windows={args.recent_windows} rolling (expanding_train=False)", flush=True)
    run_daily_update(args, env)
    run_training(args, env)
    if args.skip_publish:
        print("[publish] skipped", flush=True)
        return
    swing_score_params = None
    short_score_params = None
    if args.strategy_mode == "candidate-upgrade":
        if args.dry_run:
            print("[promotion] dry-run: candidate gate skipped", flush=True)
            return
        evaluation_styles = ["short_1_3"] if args.short_only_upgrade else None
        report = _promotion_gate(_output_prefixes(args), args, styles_to_evaluate=evaluation_styles)
        print(f"[promotion] promote={report.get('promote')} report={json.dumps(report, ensure_ascii=False)}", flush=True)
        if not report.get("promote"):
            print("[promotion] candidate rejected; current active strategy remains unchanged", flush=True)
            return
        # Candidate parameters were selected before the holdout and are published only after all gates pass.
        style_reports = report.get("styles") or {}
        short_score_params = dict((style_reports.get("short_1_3") or {}).get("candidate_params") or {})
        if args.short_only_upgrade:
            if not short_score_params:
                print("[promotion] short candidate parameters missing; active strategy remains unchanged", flush=True)
                return
            short_prefix = _output_prefixes(args)["short_1_3"]
            source = _quant_dir() / f"{short_prefix}_bt_{MODEL_NAME}_predictions.parquet"
            publish_short_champion(
                source, short_prefix, short_score_params, build_params(args),
                publish_predictions=False,
                champion_score_params=short_score_params,
            )
            print("[workflow] short champion configuration upgraded; active predictions preserved "
                  "until the next incumbent refresh", flush=True)
            return
        swing_score_params = dict((style_reports.get("swing_7_15") or {}).get("candidate_params") or {})
        if not short_score_params or not swing_score_params:
            print("[promotion] candidate parameters missing; active strategy remains unchanged", flush=True)
            return
    else:
        # Daily incumbent refresh always publishes fresh predictions, while keeping approved trade parameters frozen.
        print("[publish] mode=incumbent-refresh: latest successful predictions will be published", flush=True)
        if not args.dry_run:
            try:
                incumbent = json.loads((_quant_dir() / _ACTIVE_MANIFEST).read_text(encoding="utf-8"))
                incumbent_styles = incumbent.get("final_trade_styles") or {}
                short_cfg = incumbent_styles.get("short_1_3") or {}
                champion_params = dict(short_cfg.get("champion_score_params") or short_cfg.get("score_params") or {})
                short_score_params = dict(champion_params)
                swing_score_params = dict((incumbent_styles.get("swing_7_15") or {}).get("score_params") or {})
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"无法读取 incumbent 参数，拒绝发布以避免策略漂移: {exc}"
                ) from exc
            if not args.skip_short_grid:
                try:
                    daily_candidate = run_short_grid(
                        _output_prefixes(args), args, champion_params
                    )
                except Exception as exc:  # noqa: BLE001
                    daily_candidate = None
                    print(
                        f"[short_grid] 异常（保留 incumbent）："
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if daily_candidate:
                    short_score_params.update(daily_candidate)
                    print(
                        f"[publish] daily short adjustment anchored to champion: "
                        f"{daily_candidate}",
                        flush=True,
                    )
            if not args.skip_swing_grid:
                print(
                    "[swing_grid] 当前研究副本无可调用实现，跳过监控网格；"
                    "active swing params 保持 incumbent",
                    flush=True,
                )
            print(
                "[publish] monthly champion retained as anchor; "
                "daily bounded adjustment applied",
                flush=True,
            )
    if args.strategy_mode == "incumbent-refresh" and args.dry_run:
        validate_publish_dry_run(_output_prefixes(args), _horizons(args))
        print("[publish] dry-run: incumbent short would refresh; swing champion would remain unchanged", flush=True)
        return
    if args.strategy_mode == "incumbent-refresh":
        short_prefix = _output_prefixes(args)["short_1_3"]
        source = _quant_dir() / f"{short_prefix}_bt_{MODEL_NAME}_predictions.parquet"
        # Daily refresh trains only the newest window on expanding history; record that as
        # separate runtime metadata so the champion's monthly research config is not overwritten.
        daily_params = {
            "daily_recent_windows": int(args.recent_windows),
            "daily_expanding_train": bool(getattr(args, "expanding_train", False)),
            "daily_refreshed_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        active_paths = publish_short_champion(
            source, short_prefix, short_score_params or {}, daily_params,
            merge_history=True, refresh_swing=True,
            champion_score_params=champion_params,
        )
        check_started = time.perf_counter()
        price_latest = _latest_price_date()
        for active_path in active_paths:
            assert_active_is_latest(
                active_path,
                strict=not args.allow_stale_active,
                price_latest=price_latest,
            )
        print(
            f"[publish:timing] stage=latest_checks "
            f"seconds={time.perf_counter() - check_started:.2f}",
            flush=True,
        )
        print("[workflow] incumbent short refreshed; swing predictions re-derived from fresh short "
              "(swing params frozen); history merged", flush=True)
        return
    active_paths = publish_active_models(_output_prefixes(args), _horizons(args), build_params(args),
                                         swing_score_params=swing_score_params,
                                         short_score_params=short_score_params, dry_run=args.dry_run)
    if not args.dry_run:
        check_started = time.perf_counter()
        price_latest = _latest_price_date()
        for active_path in active_paths:
            assert_active_is_latest(
                active_path,
                strict=not args.allow_stale_active,
                price_latest=price_latest,
            )
        print(
            f"[publish:timing] stage=latest_checks "
            f"seconds={time.perf_counter() - check_started:.2f}",
            flush=True,
        )
    print("[workflow] done", flush=True)


if __name__ == "__main__":
    main()
