"""Read latest quant model scores as an optional stock_analyzer signal.

The analyzer app may run in Docker, while quant training usually writes parquet files on
the host. Mount the quant data directory and set QUANT_DATA_DIR when needed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from stock_analyzer import data, stock_meta


DEFAULT_MODEL = os.environ.get("QUANT_MODEL", "active_quant")
_ACTIVE_MANIFEST = "active_quant_model.json"
_PREDICTION_FILES = {
    "active_quant": "active_quant_predictions.parquet",
    "ridge_lightgbm_ranker_ensemble": "bt_ridge_lightgbm_ranker_ensemble_predictions.parquet",
    "lightgbm_ranker": "bt_lightgbm_ranker_predictions.parquet",
    "ridge": "bt_ridge_predictions.parquet",
}

_QUANT_PROFILES = {
    "short_stable": {
        "label": "稳健短线",
        "note": "top2，总仓位30%，单票15%；按1/2/3日白名单网格选择",
        "ic_weight": 0.03,
        "top_n": 2,
        "gross_exposure": 0.30,
        "ridge_quantile": 0.55,
        "pred_quantile": None,
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_short_h3_watchlist_grid.parquet",
    },
    "short_attack": {
        "label": "进攻短线",
        "note": "top3，总仓位24%，单票8%；同一批票更重仓",
        "ic_weight": 0.06,
        "top_n": 3,
        "gross_exposure": 0.24,
        "ridge_quantile": 0.55,
        "pred_quantile": 0.60,
        "evaluation_file": "short_horizon_watchlist_grid_v2.parquet",
    },
    "short_low_dd": {
        "label": "低回撤短线",
        "note": "top3，总仓位18%，ridge门槛更高；牺牲收益压回撤",
        "ic_weight": 0.06,
        "top_n": 3,
        "gross_exposure": 0.18,
        "ridge_quantile": 0.65,
        "pred_quantile": 0.65,
        "evaluation_file": "short_horizon_watchlist_grid_v2.parquet",
    },
    "v2_attack": {
        "label": "V2波段口径",
        "note": "top3，单票7%；按7/10/15日白名单网格选择",
        "ic_weight": 0.15,
        "top_n": 3,
        "max_weight": 0.07,
        "ridge_quantile": 0.50,
        "pred_quantile": None,
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_swing_h10_watchlist_grid.parquet",
    },
}

_TRADE_STYLES = {
    "short_1_3": {
        "label": "短线",
        "holding_days": "1天",
        "profile": "short_stable",
        "horizons": (1, 2),
        "ic_weight": 0.03,
        "top_n": 2,
        "gross_exposure": 0.30,
        "ridge_quantile": 0.55,
        "pred_quantile": None,
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_short_h1_watchlist_grid.parquet",
        "target_col": "take_profit_1",
        "top_rank": 2,
        "watch_rank_cutoff": 10,
        "rank_pct_cutoff": 0.80,
        "note": "看1日弹性(尾盘T买、T+1卖)，优先取白名单前排和较近止盈位。",
    },
    "swing_7_15": {
        "label": "波段",
        "holding_days": "7-15天",
        "profile": "v2_attack",
        "horizons": (7, 10, 15),
        "ic_weight": 0.15,
        "top_n": 3,
        "max_weight": 0.07,
        "ridge_quantile": 0.50,
        "pred_quantile": None,
        "evaluation_file": "scheduled_dual_short_swing_ne200_es40_swing_h10_watchlist_grid.parquet",
        "target_col": "take_profit_2",
        "top_rank": 3,
        "watch_rank_cutoff": 15,
        "rank_pct_cutoff": 0.70,
        "note": "看7-15日趋势延续，验证以7/10/15日为主。",
    },
}

_DEFAULT_TRADE_STYLES = ("short_1_3", "swing_7_15")


@dataclass(frozen=True)
class QuantSignal:
    code: str
    score: float
    rank_pct: float
    rank: int
    universe_size: int
    date: str
    model: str
    top3: bool = False
    available: bool = True
    # 白名单口径排名（仅在自选池内计算；不在白名单时为 None）
    watch_rank: int | None = None
    watch_rank_pct: float | None = None
    watch_universe_size: int | None = None
    watch_top3: bool = False
    # 预估收益与方向；由当前模型 manifest 中的 prediction_horizon 决定
    expected_return: float | None = None
    expected_return_horizon: int = 5
    direction: str = "中性"
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    risk_reward_1: float | None = None
    risk_reward_2: float | None = None
    risk_note: str = ""


@dataclass(frozen=True)
class MissingQuantSignal:
    code: str
    available: bool = False
    note: str = "量化预测文件不可用"


def _quant_dir() -> str:
    return os.environ.get(
        "QUANT_DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "quant_data", "full_a_2018_wide"),
    )


def _watchlist_path() -> str:
    return os.environ.get(
        "QUANT_WATCHLIST",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots", "watchlist.txt"),
    )


def _file_version(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0


def _manifest_version() -> int:
    return _file_version(os.path.join(_quant_dir(), _ACTIVE_MANIFEST))


def _watchlist_version() -> int:
    return _file_version(_watchlist_path())


def _prediction_cache_version(model: str = DEFAULT_MODEL, style: str | None = None) -> tuple[int, int]:
    return (_manifest_version(), _file_version(_prediction_path(model, style=style)))


def _evaluation_cache_version(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> tuple:
    files = _candidate_evaluation_files(model, profile=profile, style=style)
    return (_manifest_version(), tuple((path, _file_version(path)) for path in files))


@lru_cache(maxsize=1)
def profile_options() -> dict[str, str]:
    return {key: str(cfg["label"]) for key, cfg in _QUANT_PROFILES.items()}


def profile_config(profile: str | None = None) -> dict:
    key = profile or os.environ.get("QUANT_PROFILE", "short_stable")
    cfg = _QUANT_PROFILES.get(key, _QUANT_PROFILES["short_stable"]).copy()
    cfg["key"] = key if key in _QUANT_PROFILES else "short_stable"
    return cfg


def profile_label(profile: str | None = None) -> str:
    return str(profile_config(profile).get("label") or profile or "量化档位")


@lru_cache(maxsize=1)
def trade_style_options() -> dict[str, str]:
    return {key: f"{cfg['label']}（{cfg['holding_days']}）" for key, cfg in _TRADE_STYLES.items()}


def trade_style_config(style: str | None = None) -> dict:
    key = style or _DEFAULT_TRADE_STYLES[0]
    base_key = key if key in _TRADE_STYLES else _DEFAULT_TRADE_STYLES[0]
    cfg = _TRADE_STYLES[base_key].copy()
    try:
        manifest_styles = _active_manifest().get("final_trade_styles") or {}
    except Exception:  # noqa: BLE001
        manifest_styles = {}
    if isinstance(manifest_styles, dict) and key in manifest_styles and isinstance(manifest_styles[key], dict):
        raw = manifest_styles[key].copy()
        score_params = raw.pop("score_params", {})
        if isinstance(score_params, dict):
            raw.update(score_params)
        if raw.get("target_price") and not raw.get("target_col"):
            raw["target_col"] = raw["target_price"]
        cfg.update(raw)
        base_key = key
    cfg["key"] = base_key
    return cfg


def _read_watchlist() -> tuple[str, ...]:
    """读取自选池，仅提取 6 位股票代码，保持文件顺序。"""
    return _read_watchlist_cached(_watchlist_version())


@lru_cache(maxsize=8)
def _read_watchlist_cached(cache_version: int) -> tuple[str, ...]:
    path = _watchlist_path()
    if not os.path.exists(path):
        return tuple()
    codes: list[str] = []
    seen: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split()[0].strip()
                if len(token) >= 6 and token[:6].isdigit():
                    code = token[:6]
                    if code not in seen:
                        seen.add(code)
                        codes.append(code)
    except Exception:  # noqa: BLE001
        return tuple()
    return tuple(codes)


def _direction(score: float) -> str:
    if score > 0:
        return "看多"
    if score < 0:
        return "看空"
    return "中性"


def _price_path(code: str) -> str:
    return os.path.join(_quant_dir(), "price", f"{code}.parquet")


def _risk_levels(code: str, expected_return: float | None = None) -> dict:
    """基于日线 ATR 的交易风控层，不参与模型训练/排序。"""
    return _risk_levels_cached(code, expected_return, _file_version(_price_path(code)))


@lru_cache(maxsize=512)
def _risk_levels_cached(code: str, expected_return: float | None = None, cache_version: int = 0) -> dict:
    path = _price_path(code)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path, columns=["date", "high", "low", "close"])
    except Exception:  # noqa: BLE001
        return {}
    if df.empty:
        return {}
    for c in ("high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "high", "low", "close"]).sort_values("date")
    if len(df) < 20:
        return {}
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    entry = float(df["close"].iloc[-1])
    if not entry or not pd.notna(entry) or not pd.notna(atr) or atr <= 0:
        return {}
    atr_pct = atr / entry
    est = abs(float(expected_return)) if expected_return is not None and pd.notna(expected_return) else 0.0
    # 止损以 ATR 为核心；止盈用 ATR 和模型预估收益共同约束，避免小预估给出过窄目标。
    stop_gap = max(1.2 * atr, entry * 0.035)
    tp1_gap = max(1.5 * atr, entry * min(max(est, 0.04), 0.10))
    tp2_gap = max(2.4 * atr, entry * min(max(est * 1.6, 0.07), 0.16))
    stop = max(entry - stop_gap, 0.01)
    tp1 = entry + tp1_gap
    tp2 = entry + tp2_gap
    return {
        "entry_price": round(entry, 2),
        "stop_loss": round(stop, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "atr_14": round(atr, 4),
        "atr_pct": round(atr_pct, 4),
        "risk_reward_1": round((tp1 - entry) / max(entry - stop, 1e-9), 2),
        "risk_reward_2": round((tp2 - entry) / max(entry - stop, 1e-9), 2),
        "risk_note": "ATR14风控：止损=max(1.2ATR,3.5%)；止盈=max(ATR倍数, Ridge预估收益约束)",
    }


def _active_manifest() -> dict:
    path = os.path.join(_quant_dir(), _ACTIVE_MANIFEST)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _style_artifact(style: str | None = None) -> dict:
    if not style:
        return {}
    manifest = _active_manifest()
    artifacts = manifest.get("style_artifacts") or {}
    if isinstance(artifacts, dict) and isinstance(artifacts.get(style), dict):
        return artifacts[style]
    styles = manifest.get("final_trade_styles") or {}
    if isinstance(styles, dict) and isinstance(styles.get(style), dict):
        raw = styles[style]
        return {k: raw.get(k) for k in ("predictions_file", "prediction_horizon") if raw.get(k)}
    return {}


def _prediction_path(model: str, style: str | None = None) -> str:
    if model == "active_quant":
        manifest = _active_manifest()
        artifact = _style_artifact(style)
        filename = artifact.get("predictions_file")
        if filename:
            path = os.path.join(_quant_dir(), str(filename))
            if os.path.exists(path):
                return path
        filename = manifest.get("predictions_file") or _PREDICTION_FILES["active_quant"]
    else:
        filename = _PREDICTION_FILES.get(model, f"bt_{model}_predictions.parquet")
    return os.path.join(_quant_dir(), str(filename))


def _display_model_name(model: str, style: str | None = None) -> str:
    if model != "active_quant":
        return model
    manifest = _active_manifest()
    prefixes = manifest.get("output_prefixes") or {}
    if style and isinstance(prefixes, dict) and prefixes.get(style):
        return str(prefixes[style])
    return str(manifest.get("output_prefix") or model)


def prediction_horizon(model: str = DEFAULT_MODEL, style: str | None = None) -> int:
    manifest = _active_manifest() if model == "active_quant" else {}
    artifact = _style_artifact(style) if model == "active_quant" else {}
    horizons = manifest.get("prediction_horizons") or {}
    try:
        return int(
            artifact.get("prediction_horizon")
            or artifact.get("horizon")
            or (horizons.get(style) if isinstance(horizons, dict) and style else None)
            or manifest.get("prediction_horizon")
            or manifest.get("horizon")
            or 5
        )
    except Exception:  # noqa: BLE001
        return 5


def _candidate_evaluation_files(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> list[str]:
    manifest = _active_manifest() if model == "active_quant" else {}
    names: list[str] = []
    if style:
        scfg = trade_style_config(style)
        if scfg.get("evaluation_file"):
            names.append(str(scfg["evaluation_file"]))
    pcfg = profile_config(profile)
    if pcfg.get("evaluation_file"):
        names.append(str(pcfg["evaluation_file"]))
    for key in ("evaluation_file", "multihorizon_evaluation_file"):
        v = manifest.get(key)
        if isinstance(v, str) and v:
            names.append(v)
    prefixes = manifest.get("output_prefixes") or {}
    out_prefix = ""
    if model == "active_quant" and style and isinstance(prefixes, dict):
        out_prefix = str(prefixes.get(style) or "")
    if not out_prefix:
        out_prefix = str(manifest.get("output_prefix") or "") if model == "active_quant" else str(model or "")
    if out_prefix:
        names.extend([
            f"{out_prefix}_watchlist_multihorizon.parquet",
            f"{out_prefix}_watchlist_eval_multihorizon.parquet",
            f"{out_prefix}_multihorizon.parquet",
        ])
    names.extend([
        "v2_watchlist_fine_grid_multihorizon.parquet",
        "v2_watchlist_fine_grid_ic_ridge_weight.parquet",
    ])
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(os.path.join(_quant_dir(), name))
    return out


def evaluation_frame(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> pd.DataFrame:
    """读取量化模型白名单历史验证结果。

    只要求文件有 horizon 列；UI 会按实际 horizon 动态展开，所以后续新增
    10/15/30 日等周期时不需要改展示逻辑。
    """
    return _evaluation_frame_cached(model, profile, style, _evaluation_cache_version(model, profile=profile, style=style))


@lru_cache(maxsize=32)
def _evaluation_frame_cached(model: str, profile: str | None, style: str | None, cache_version: tuple) -> pd.DataFrame:
    for path in _candidate_evaluation_files(model, profile=profile, style=style):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "horizon" not in df.columns:
            continue
        df = df.copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df = df.dropna(subset=["horizon"])
        if df.empty:
            continue
        df["horizon"] = df["horizon"].astype(int)
        df["evaluation_file"] = os.path.basename(path)
        return df
    return pd.DataFrame()


def evaluation_horizons(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> tuple[int, ...]:
    frame = evaluation_frame(model, profile=profile, style=style)
    if frame.empty or "horizon" not in frame.columns:
        return tuple()
    return tuple(sorted(int(h) for h in frame["horizon"].dropna().unique()))


def selected_evaluation(model: str = DEFAULT_MODEL, prefer: dict | None = None, profile: str | None = None, style: str | None = None) -> pd.DataFrame:
    """返回一组代表性参数在各 horizon 上的验证指标。

    prefer 可传入 ic_weight/max_weight/ridge_quantile/pred_quantile 等参数。
    如果未显式传 prefer，则使用 profile 对应参数。
    如果找不到完全匹配，就退化为每个 horizon 下 Sharpe 最高的一组。
    """
    frame = evaluation_frame(model, profile=profile, style=style)
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if prefer is None and style is not None:
        scfg = trade_style_config(style)
        prefer = {k: scfg.get(k) for k in ("ic_weight", "top_n", "gross_exposure", "max_weight", "ridge_quantile", "pred_quantile") if k in out.columns and k in scfg}
    if prefer is None and profile is not None:
        pcfg = profile_config(profile)
        prefer = {k: pcfg.get(k) for k in ("ic_weight", "top_n", "gross_exposure", "max_weight", "ridge_quantile", "pred_quantile") if k in out.columns and k in pcfg}
    if prefer:
        filt = pd.Series(True, index=out.index)
        for key, val in prefer.items():
            if key not in out.columns:
                continue
            col = out[key]
            if val is None:
                filt &= col.isna()
            elif pd.api.types.is_numeric_dtype(col):
                filt &= (pd.to_numeric(col, errors="coerce") - float(val)).abs() < 1e-9
            else:
                filt &= col.astype(str) == str(val)
        matched = out[filt]
        if not matched.empty:
            return matched.sort_values("horizon").reset_index(drop=True)
    rank_col = "sharpe" if "sharpe" in out.columns else ("annual_return" if "annual_return" in out.columns else "horizon")
    return (out.sort_values(["horizon", rank_col], ascending=[True, False])
               .groupby("horizon", as_index=False)
               .head(1)
               .sort_values("horizon")
               .reset_index(drop=True))


def recommended_horizon(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> dict:
    """从白名单历史验证中，给出该口径下最优的持有天数(h)及其指标。

    选取规则：优先按 Sharpe，其次单票方向胜率。返回 {} 表示暂无验证数据。
    """
    ev = selected_evaluation(model=model, profile=profile, style=style)
    if ev is None or ev.empty or "horizon" not in ev.columns:
        return {}
    d = ev.copy()
    d = d.dropna(subset=["horizon"])
    if d.empty:
        return {}
    d["_sharpe"] = pd.to_numeric(d.get("sharpe"), errors="coerce")
    d["_dwr"] = pd.to_numeric(d.get("direction_win_rate"), errors="coerce") if "direction_win_rate" in d.columns else 0.0
    d = d.sort_values(["_sharpe", "_dwr"], ascending=False, na_position="last")
    best = d.iloc[0]

    def _f(col):
        v = pd.to_numeric(best.get(col), errors="coerce") if col in best.index else None
        return float(v) if v is not None and pd.notna(v) else None

    return {
        "horizon": int(best["horizon"]),
        "sharpe": _f("sharpe"),
        "annual_return": _f("annual_return"),
        "win_rate": _f("win_rate"),
        "direction_win_rate": _f("direction_win_rate"),
        "max_drawdown": _f("max_drawdown"),
    }



def _latest_complete_date(dates: pd.Series, min_ratio: float = 0.8, floor: int = 100) -> pd.Timestamp | None:
    """选取"覆盖完整"的最新交易日。

    免费行情源当日 EOD 数据未发布时，最新交易日可能只有零星几只（如北交所或
    个别标的抢先出了当日 K线），导致预测最新日样本数远低于正常规模。此时自动
    回退到上一完整交易日，避免 UI/选股只拿到极少数候选。
    """
    counts = dates.value_counts().sort_index()
    if counts.empty:
        return None
    recent = counts.tail(20)
    reference = float(recent.median()) if not recent.empty else 0.0
    threshold = max(floor, reference * min_ratio)
    for d in reversed(list(counts.index)):
        if counts.loc[d] >= threshold:
            return d
    return counts.index[-1]


def latest_frame(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> pd.DataFrame:
    return _latest_frame_cached(model, profile, style, _prediction_cache_version(model, style=style))


@lru_cache(maxsize=64)
def _latest_frame_cached(model: str, profile: str | None, style: str | None, cache_version: tuple[int, int]) -> pd.DataFrame:
    path = _prediction_path(model, style=style)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001
        cols = set()
    want = ["code", "date", "pred"]
    # ridge_pred 是 Ridge 对未来 5 日收益率的回归预测，作为可读的“预估收益”
    for c in ("ridge_pred", "lgbm_pred", "base_pred", "ic_z", "elastic_z", "catboost_z", "extra_trees_z", "rule_score"):
        if c in cols:
            want.append(c)
    try:
        df = pd.read_parquet(path, columns=want)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["code", "date", "pred"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()
    latest = _latest_complete_date(df["date"])
    if latest is None:
        return pd.DataFrame()
    day = df[df["date"] == latest].copy()
    if day.empty:
        return pd.DataFrame()
    day["code"] = day["code"].astype(str).str.zfill(6)
    day["pred"] = pd.to_numeric(day["pred"], errors="coerce")
    pcfg = trade_style_config(style) if style else profile_config(profile)
    base = pd.to_numeric(day.get("base_pred", day["pred"]), errors="coerce")
    if pcfg.get("lgbm_weight") is not None and "lgbm_pred" in day.columns and "ridge_pred" in day.columns:
        lgbm = pd.to_numeric(day["lgbm_pred"], errors="coerce")
        ridge = pd.to_numeric(day["ridge_pred"], errors="coerce")
        lgbm_std = lgbm.std(ddof=0)
        ridge_std = ridge.std(ddof=0)
        lgbm_z = (lgbm - lgbm.mean()) / lgbm_std if lgbm_std else 0.0
        ridge_z = (ridge - ridge.mean()) / ridge_std if ridge_std else 0.0
        weight = float(pcfg["lgbm_weight"])
        base = weight * lgbm_z + (1.0 - weight) * ridge_z
    day["pred"] = base
    if "ic_z" in day.columns and pcfg.get("ic_weight") is not None:
        day["pred"] = day["pred"] + float(pcfg["ic_weight"]) * pd.to_numeric(day["ic_z"], errors="coerce").fillna(0.0)
    if "elastic_z" in day.columns and pcfg.get("elastic_weight") is not None:
        day["pred"] = day["pred"] + float(pcfg["elastic_weight"]) * pd.to_numeric(day["elastic_z"], errors="coerce").fillna(0.0)
    if "catboost_z" in day.columns and pcfg.get("catboost_weight") is not None:
        day["pred"] = day["pred"] + float(pcfg["catboost_weight"]) * pd.to_numeric(day["catboost_z"], errors="coerce").fillna(0.0)
    if "extra_trees_z" in day.columns and pcfg.get("extra_trees_weight") is not None:
        day["pred"] = day["pred"] + float(pcfg["extra_trees_weight"]) * pd.to_numeric(day["extra_trees_z"], errors="coerce").fillna(0.0)
    if "rule_score" in day.columns and pcfg.get("naive_weight") is not None:
        rule = pd.to_numeric(day["rule_score"], errors="coerce")
        rule_std = rule.std(ddof=0)
        rule_z = (rule - rule.mean()) / rule_std if rule_std else 0.0
        day["pred"] = day["pred"] + float(pcfg["naive_weight"]) * rule_z
    day = day.dropna(subset=["pred"]).sort_values("pred", ascending=False).reset_index(drop=True)
    if day.empty:
        return pd.DataFrame()
    day["rank"] = day.index + 1
    n = len(day)
    day["rank_pct"] = 1.0 - ((day["rank"] - 1) / max(n - 1, 1))
    day["date"] = latest.strftime("%Y-%m-%d")
    day["model"] = f"{_display_model_name(model, style=style)} / {profile_label(profile)}"
    if "ridge_pred" not in day.columns:
        day["ridge_pred"] = pd.NA
    return day[["code", "date", "pred", "ridge_pred", "rank", "rank_pct", "model"]]


def watchlist_frame(model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> pd.DataFrame:
    """白名单口径的量化打分表：仅保留自选池股票，并在池内重算排名/分位。

    返回列: code, date, pred, expected_return, direction,
            rank(全A), rank_pct(全A), watch_rank, watch_rank_pct, model
    """
    return _watchlist_frame_cached(
        model,
        profile,
        style,
        _prediction_cache_version(model, style=style),
        _watchlist_version(),
    )


@lru_cache(maxsize=64)
def _watchlist_frame_cached(
    model: str,
    profile: str | None,
    style: str | None,
    prediction_cache_version: tuple[int, int],
    watchlist_cache_version: int,
) -> pd.DataFrame:
    frame = latest_frame(model, profile=profile, style=style)
    if frame.empty:
        return pd.DataFrame()
    watch = _read_watchlist()
    if not watch:
        return pd.DataFrame()
    sub = frame[frame["code"].isin(set(watch))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("pred", ascending=False).reset_index(drop=True)
    sub["watch_rank"] = sub.index + 1
    m = len(sub)
    sub["watch_rank_pct"] = 1.0 - ((sub["watch_rank"] - 1) / max(m - 1, 1))
    sub["expected_return"] = pd.to_numeric(sub.get("ridge_pred"), errors="coerce")
    sub["expected_return_horizon"] = prediction_horizon(model, style=style)
    sub["direction"] = sub["pred"].apply(_direction)
    risks = [_risk_levels(str(row.code), row.expected_return if pd.notna(row.expected_return) else None)
             for row in sub[["code", "expected_return"]].itertuples(index=False)]
    risk_df = pd.DataFrame(risks)
    for c in ["entry_price", "stop_loss", "take_profit_1", "take_profit_2", "risk_reward_1", "risk_reward_2"]:
        sub[c] = risk_df[c].to_numpy() if c in risk_df.columns else pd.NA
    out = sub[["code", "date", "pred", "expected_return", "expected_return_horizon", "direction",
               "entry_price", "stop_loss", "take_profit_1", "take_profit_2", "risk_reward_1", "risk_reward_2",
               "rank", "rank_pct", "watch_rank", "watch_rank_pct", "model"]]
    return stock_meta.enrich_frame(out, remote=False)


def get(symbol: str, model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None):
    code = data._normalize_symbol(symbol)
    frame = latest_frame(model, profile=profile, style=style)
    if frame.empty:
        return None
    row = frame[frame["code"] == code]
    if row.empty:
        return None
    r = row.iloc[0]
    exp_ret = pd.to_numeric(r.get("ridge_pred"), errors="coerce")
    exp_ret = float(exp_ret) if pd.notna(exp_ret) else None
    risk = _risk_levels(code, exp_ret)
    watch = watchlist_frame(model, profile=profile, style=style)
    watch_rank = watch_rank_pct = watch_universe = None
    watch_top3 = False
    if not watch.empty:
        wrow = watch[watch["code"] == code]
        if not wrow.empty:
            wr = wrow.iloc[0]
            watch_rank = int(wr["watch_rank"])
            watch_rank_pct = float(wr["watch_rank_pct"])
            watch_universe = int(len(watch))
            watch_top3 = watch_rank <= 3
    return QuantSignal(
        code=code,
        score=float(r["pred"]),
        rank_pct=float(r["rank_pct"]),
        rank=int(r["rank"]),
        universe_size=int(len(frame)),
        date=str(r["date"]),
        model=str(r["model"]),
        top3=int(r["rank"]) <= 3,
        watch_rank=watch_rank,
        watch_rank_pct=watch_rank_pct,
        watch_universe_size=watch_universe,
        watch_top3=watch_top3,
        expected_return=exp_ret,
        expected_return_horizon=prediction_horizon(model, style=style),
        direction=_direction(float(r["pred"])),
        entry_price=risk.get("entry_price"),
        stop_loss=risk.get("stop_loss"),
        take_profit_1=risk.get("take_profit_1"),
        take_profit_2=risk.get("take_profit_2"),
        atr_14=risk.get("atr_14"),
        atr_pct=risk.get("atr_pct"),
        risk_reward_1=risk.get("risk_reward_1"),
        risk_reward_2=risk.get("risk_reward_2"),
        risk_note=str(risk.get("risk_note") or ""),
    )


def quant_for_codes(symbols, model: str = DEFAULT_MODEL, profile: str | None = None, style: str | None = None) -> pd.DataFrame:
    """为一组股票返回量化结果（供 UI 三只股票并排展示 / 回测栏使用）。

    列: code, quant_score, expected_return, expected_return_horizon,
        direction, entry_price, stop_loss, take_profit_1, take_profit_2,
        watch_rank, watch_rank_pct, rank(全A), rank_pct(全A), date, model, available
    """
    rows = []
    for sym in symbols:
        sig = get(sym, model=model, profile=profile, style=style)
        if sig is None:
            rows.append({"code": data._normalize_symbol(sym), "available": False})
            continue
        rows.append({
            "code": sig.code,
            "quant_score": round(sig.score, 4),
            "expected_return": round(sig.expected_return, 4) if sig.expected_return is not None else None,
            "expected_return_horizon": sig.expected_return_horizon,
            "direction": sig.direction,
            "entry_price": sig.entry_price,
            "stop_loss": sig.stop_loss,
            "take_profit_1": sig.take_profit_1,
            "take_profit_2": sig.take_profit_2,
            "atr_14": sig.atr_14,
            "atr_pct": sig.atr_pct,
            "risk_reward_1": sig.risk_reward_1,
            "risk_reward_2": sig.risk_reward_2,
            "risk_note": sig.risk_note,
            "watch_rank": sig.watch_rank,
            "watch_rank_pct": round(sig.watch_rank_pct, 4) if sig.watch_rank_pct is not None else None,
            "rank": sig.rank,
            "rank_pct": round(sig.rank_pct, 4),
            "date": sig.date,
            "model": sig.model,
            "available": True,
        })
    return stock_meta.enrich_frame(pd.DataFrame(rows))


def _style_eval_label(style: str, model: str = DEFAULT_MODEL) -> str:
    cfg = trade_style_config(style)
    ev = selected_evaluation(model=model, profile=str(cfg["profile"]), style=style)
    if ev.empty or "horizon" not in ev.columns:
        return "暂无历史验证"
    horizons = set(int(h) for h in cfg.get("horizons", ()))
    got = sorted(int(h) for h in ev["horizon"].dropna().astype(int).unique() if int(h) in horizons)
    return "/".join(f"{h}日" for h in got) if got else "暂无匹配周期验证"


def _style_suggestion(row: pd.Series, cfg: dict) -> str:
    if str(row.get("direction") or "") != "看多":
        return "观望"
    rank = row.get("watch_rank")
    rank_pct = row.get("watch_rank_pct")
    if pd.notna(rank) and int(rank) <= int(cfg.get("top_rank", 3)):
        return "优先"
    if pd.notna(rank) and int(rank) <= int(cfg.get("watch_rank_cutoff", 10)):
        return "可选"
    if pd.notna(rank_pct) and float(rank_pct) >= float(cfg.get("rank_pct_cutoff", 0.8)):
        return "关注"
    return "备选"


def _style_reason(row: pd.Series, cfg: dict, eval_label: str) -> str:
    parts = [f"{cfg['holding_days']}口径"]
    if pd.notna(row.get("watch_rank")):
        parts.append(f"白名单第{int(row['watch_rank'])}")
    if pd.notna(row.get("target_price")):
        parts.append(f"目标{float(row['target_price']):.2f}")
    if pd.notna(row.get("stop_loss")):
        parts.append(f"止损{float(row['stop_loss']):.2f}")
    if eval_label:
        parts.append(f"验证{eval_label}")
    return "，".join(parts)


def _apply_trade_style(frame: pd.DataFrame, style: str, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    cfg = trade_style_config(style)
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["style_key"] = cfg["key"]
    out["style_label"] = cfg["label"]
    out["holding_days"] = cfg["holding_days"]
    out["profile_key"] = cfg["profile"]
    out["profile_label"] = profile_label(str(cfg["profile"]))
    target_col = str(cfg.get("target_col") or "take_profit_1")
    rr_col = "risk_reward_2" if target_col.endswith("2") else "risk_reward_1"
    out["target_price"] = out[target_col] if target_col in out.columns else pd.NA
    out["target_label"] = "止盈2" if target_col.endswith("2") else "止盈1"
    out["risk_reward"] = out[rr_col] if rr_col in out.columns else pd.NA
    eval_label = _style_eval_label(style, model=model)
    out["validation_horizons"] = eval_label
    out["suggestion"] = out.apply(lambda r: _style_suggestion(r, cfg), axis=1)
    out["reason"] = out.apply(lambda r: _style_reason(r, cfg, eval_label), axis=1)
    return out


def watchlist_trade_advice(model: str = DEFAULT_MODEL, styles: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows = []
    for style in styles or _DEFAULT_TRADE_STYLES:
        cfg = trade_style_config(style)
        frame = watchlist_frame(model=model, profile=str(cfg["profile"]), style=style)
        if frame.empty:
            continue
        base = frame.rename(columns={"pred": "quant_score"}).copy()
        base["available"] = True
        rows.append(_apply_trade_style(base, style, model=model))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def trade_advice_for_codes(symbols, model: str = DEFAULT_MODEL, styles: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows = []
    for style in styles or _DEFAULT_TRADE_STYLES:
        cfg = trade_style_config(style)
        frame = quant_for_codes(symbols, model=model, profile=str(cfg["profile"]), style=style)
        if frame.empty:
            continue
        available = frame[frame.get("available", False) == True].copy()  # noqa: E712
        if available.empty:
            missing = frame.copy()
            missing["style_key"] = cfg["key"]
            missing["style_label"] = cfg["label"]
            missing["holding_days"] = cfg["holding_days"]
            missing["suggestion"] = "无数据"
            rows.append(missing)
            continue
        rows.append(_apply_trade_style(available, style, model=model))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def score_for_screener(symbol: str, model: str = DEFAULT_MODEL, profile: str | None = None) -> dict:
    sig = get(symbol, model=model, profile=profile)
    if not sig:
        return {"quant_available": False}
    # 白名单场景优先使用池内分位；否则回退全 A 分位
    eff_rank_pct = sig.watch_rank_pct if sig.watch_rank_pct is not None else sig.rank_pct
    return {
        "quant_available": True,
        "quant_score": round(sig.score, 6),
        "quant_rank_pct": round(sig.rank_pct, 4),
        "quant_rank": sig.rank,
        "quant_date": sig.date,
        "quant_model": sig.model,
        "quant_top3": sig.top3,
        "quant_direction": sig.direction,
        "quant_expected_return": round(sig.expected_return, 6) if sig.expected_return is not None else None,
        "quant_expected_return_horizon": sig.expected_return_horizon,
        "quant_entry_price": sig.entry_price,
        "quant_stop_loss": sig.stop_loss,
        "quant_take_profit_1": sig.take_profit_1,
        "quant_take_profit_2": sig.take_profit_2,
        "quant_atr_14": sig.atr_14,
        "quant_atr_pct": sig.atr_pct,
        "quant_risk_reward_1": sig.risk_reward_1,
        "quant_risk_reward_2": sig.risk_reward_2,
        "quant_risk_note": sig.risk_note,
        "quant_watch_rank": sig.watch_rank,
        "quant_watch_rank_pct": round(sig.watch_rank_pct, 4) if sig.watch_rank_pct is not None else None,
        "quant_watch_universe": sig.watch_universe_size,
        "quant_watch_top3": sig.watch_top3,
        "quant_effective_rank_pct": round(eff_rank_pct, 4),
    }
