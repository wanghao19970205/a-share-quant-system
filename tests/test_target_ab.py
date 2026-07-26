"""A/B 训练目标口径对齐 · 单元测试（task #5）。

在容器内运行（模块方式，勿直接 python3 文件路径以免 import select 被遮蔽）：
    python -m pytest tests/test_target_ab.py -q
或无 pytest 时：
    python tests/test_target_ab.py

覆盖：
1. baseline 回归：label_col=None / train_mask_col=None 时 train_ridge 走原路径（预测逐字节等价）。
2. mask 语义：剔除「买入日 T 当天封涨停(buyable_close=False)」样本，保留「T 买、T+1 涨停」正样本。
3. rolled_sell_close：跌停顺延卖出实现价与丢尾语义。
4. _price_targets 重构后 == tradability.price_tradability（保护生产选参路径）。
"""
import numpy as np
import pandas as pd

from quant import model as qmodel
from quant import tradability


def _toy_panel(n_dates=40, n_codes=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for d in dates:
        for c in range(n_codes):
            rows.append({
                "code": f"{600000 + c:06d}",
                "date": d,
                "f1": rng.normal(),
                "f2": rng.normal(),
                "target_ret_1d": rng.normal() * 0.02,
                "buyable_close": True,
            })
    return pd.DataFrame(rows)


def test_baseline_regression():
    """label_col=None / train_mask_col=None 必须与不传两参完全一致。"""
    panel = _toy_panel()
    feats = ["f1", "f2"]
    kw = dict(horizon=1, train_end="2022-02-04", valid_end="2022-02-11",
              predict_start="2022-02-14", decay_half_life_days=60.0, min_weight=0.03)
    a = qmodel.train_ridge(panel, feats, **kw)
    b = qmodel.train_ridge(panel, feats, label_col=None, train_mask_col=None, **kw)
    assert a.ok and b.ok
    pd.testing.assert_frame_equal(
        a.predictions.reset_index(drop=True), b.predictions.reset_index(drop=True))


def test_mask_drops_only_buyday_sealed():
    """mask 只看买入日 T 当天 buyable_close：
    - 「T 当天封涨停」样本被剔除；
    - 「T 买入(未封)、T+1 涨停」正样本保留（mask 不牵连 T+1）。"""
    panel = _toy_panel(seed=1)
    # 人为把某一天某只票标为买入日封涨停
    mask_false_idx = panel.index[(panel["date"] == panel["date"].unique()[5])
                                 & (panel["code"] == "600000")]
    panel.loc[mask_false_idx, "buyable_close"] = False
    train = panel[panel["date"] <= panel["date"].unique()[20]].copy()
    kept = qmodel._apply_train_mask(train, "buyable_close")
    # 被标 False 的样本不在训练集
    assert kept[(kept["date"] == panel["date"].unique()[5]) & (kept["code"] == "600000")].empty
    # 其它样本全部保留（只掉 1 行）
    assert len(kept) == len(train) - 1
    # NaN 的 buyable_close 视为可买（保留）——先转 object 承载 NaN
    # （pandas≥2 的 bool 列不允许直接写入 NaN，会 TypeError；生产中该列 left-join 后本就是 object/float）
    train2 = train.copy()
    train2["buyable_close"] = train2["buyable_close"].astype("object")
    train2.loc[mask_false_idx, "buyable_close"] = np.nan
    kept2 = qmodel._apply_train_mask(train2, "buyable_close")
    assert len(kept2) == len(train2)


def test_apply_train_mask_none_is_noop():
    panel = _toy_panel(seed=2)
    assert len(qmodel._apply_train_mask(panel, None)) == len(panel)
    assert len(qmodel._apply_train_mask(panel, "nonexistent_col")) == len(panel)


def test_rolled_sell_close_semantics():
    """跌停顺延：卖出日被封则顺延到下一可卖日收盘；末尾丢尾置 NaN。"""
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)
    # 无封板：T 买、T+1 收盘卖 → 等于 close.shift(-1)
    blocked = np.zeros(6, dtype=bool)
    out = tradability.rolled_sell_close(close, blocked, horizon=1, cap=3)
    assert out[0] == 11.0 and out[4] == 15.0
    assert np.isnan(out[5])  # 丢尾
    # T+1 卖出日(index1)被封 → 顺延到 index2 收盘
    blocked2 = np.array([False, True, False, False, False, False])
    out2 = tradability.rolled_sell_close(close, blocked2, horizon=1, cap=3)
    assert out2[0] == 12.0  # index0 买，预定 index1 卖被封，顺延 index2


def test_price_targets_refactor_equivalence(tmp_path=None):
    """_price_targets 重构后应等价于 tradability.price_tradability（保护生产选参路径）。
    需要真实 price 文件，无数据时跳过。"""
    from quant import watchlist_grid, config
    from pathlib import Path
    price_dir = Path(config.QUANT_DIR) / "price"
    if not price_dir.exists():
        print("[skip] no price dir; skipping equivalence test")
        return
    codes = sorted(p.stem for p in list(price_dir.glob("*.parquet"))[:5])
    if not codes:
        print("[skip] no price files")
        return
    a = watchlist_grid._price_targets(codes, [1])
    b = tradability.price_tradability(codes, [1], quant_dir=Path(config.QUANT_DIR))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


if __name__ == "__main__":
    test_baseline_regression()
    test_mask_drops_only_buyday_sealed()
    test_apply_train_mask_none_is_noop()
    test_rolled_sell_close_semantics()
    test_price_targets_refactor_equivalence()
    print("all target_ab tests passed")
