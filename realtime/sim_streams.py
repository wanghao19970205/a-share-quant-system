"""虚拟数据流验证：喂脚本化快照序列驱动 RerankScorer / PaperTrader，断言实时层新逻辑。

不依赖券商连接、不触网、不碰 quant_data——纯内存构造 Snapshot 喂进 StrategyContext，
覆盖用户关心的 6 类行为：
  A. 盘中动态重排随盘中量变化（同池两票模型分接近，便宜+买盘强的爬升到追高票之前）
  B. 重排有界（模型分差距大时，盘中极端量也翻不动次序——守「模型锚定」）
  C. 候选池封闭（重排/买入绝不引入模型池外的票）
  D. A股 T+1（当日买入 held<1，暴跌也不出场）
  E. 次日出场（held>=1 时 stop_loss/take_profit/vwap_break/time_cap 各自按优先级触发）
  F. 入场过滤（追高/偏贵/卖盘强候选被跳过，顺延买下一名）

跑法（容器内，py3.12，/app 在 PYTHONPATH）：python /tmp/sim_streams.py
退出码 0=全通过，非 0=有断言失败（详见输出 [FAIL] 行）。
"""
from __future__ import annotations

import datetime as _dt
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from realtime import paper_trader as paper_mod
from realtime.snapshot import Snapshot
from realtime.strategy import StrategyContext, default_strategies
from realtime.rerank import RerankScorer
from realtime.paper_trader import PaperTrader
from realtime.v2 import V2PaperTrader
from realtime.v3 import V3PaperTrader
from realtime.v4 import V4PaperTrader
from realtime.sector_etf import SectorETFContext
from realtime.feed import subscription_code
from realtime.reference import RefRow, _load_expected_return, expected_return_text
from realtime.watchlist import load_codes

# ---- 轻量断言框架 ----------------------------------------------------------
_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {msg}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {msg}")


# ---- 假配置：属性齐全，阈值取生产默认 --------------------------------------
class Cfg:
    paper_trade_enabled = True
    paper_v2_enabled = True
    paper_v3_enabled = True
    paper_v4_enabled = True
    sector_etf_enabled = True
    sector_etf_benchmark = "510300.SH"
    sector_etf_specs = (
        "半导体=512480.SH:数字芯片设计;"
        "银行=512800.SH:股份制银行Ⅲ;"
        "军工=512660.SH:军工电子Ⅱ")
    sector_etf_quote_max_age_sec = 90.0
    paper_v4_sector_weak_excess = -0.003
    paper_v4_sector_strong_excess = 0.003
    paper_v4_sector_mapping_min_confidence = 0.8
    sector_meta_file = ""
    paper_buy_n = 2
    paper_buy_start = 1450
    paper_buy_end = 1455
    paper_time_cap_start = 1450
    paper_max_positions = 4
    paper_breakeven_arm = 0.03
    paper_breakeven_margin = 0.005
    paper_take_profit_tighten = 0.03
    paper_limit_down_roll_max = 3
    paper_v3_quote_max_age_sec = 90.0
    paper_v3_atr_k = 2.0
    paper_start_equity = 100000.0
    paper_cost = 0.002
    sell_horizon = 1
    paper_stop_loss = 0.05
    paper_take_profit = 0.09
    paper_trail_k = 3.0
    paper_vwap_break = 0.02
    paper_entry_gap_eaten = 0.0
    paper_entry_rich = 0.01
    paper_entry_ask_strong = 0.2
    ledger_dir = "/tmp"
    paper_state_file = ""  # 每场景覆盖
    rerank_enabled = True
    rank_pool_n = 30
    rerank_cap = 0.30
    rerank_w_vwap = 0.35
    rerank_w_imb = 0.30
    rerank_w_gap = 0.0
    rerank_w_spread = 0.15
    rerank_intraday_rank_scale = 0.10
    rank_raw_safety_margin = 0.001
    rank_min_raw_return = 0.0
    rank_min_net_return = 0.0  # 兼容旧审计字段，不再作为硬门
    ensemble_return_enabled = True
    ensemble_ridge_weight = 0.30
    ensemble_elastic_weight = 0.20
    ensemble_extra_trees_weight = 0.50


class FakeNotifier:
    """收集推送而不真发；push(title, body) 兼容 PaperTrader 调用。"""

    def __init__(self):
        self.msgs = []

    def push(self, title, body=""):
        self.msgs.append((title, body))


# ---- 快照构造器：一条 Level-1 快照，缺省中性 ------------------------------
def mk_snap(code, last, *, pre_close=None, open_px=None,
            vwap=None, imb=0.0, spread=0.001, high_limit=None):
    """构造快照。vwap 通过 amount/volume 反推；imb 通过 bid/ask_volume1；spread 通过 bid/ask_price1。"""
    pre_close = pre_close if pre_close is not None else last
    open_px = open_px if open_px is not None else pre_close
    volume = 100000.0
    amount = (vwap if vwap is not None else last) * volume
    # 失衡 imb=(b-a)/(b+a) → 取 b+a=2000 反解 b,a
    total = 2000.0
    b = total * (1 + imb) / 2.0
    a = total - b
    # 价差：以 last 为中间价，(ask-bid)/mid = spread
    half = last * spread / 2.0
    bid1 = round(last - half, 4)
    ask1 = round(last + half, 4)
    hl = high_limit if high_limit is not None else round(pre_close * 1.1, 2)
    return Snapshot(
        code=code, trade_time="14:55:00", last=last, pre_close=pre_close,
        open=open_px, high_limited=hl, low_limited=round(pre_close * 0.9, 2),
        volume=volume, amount=amount,
        bid_price1=bid1, ask_price1=ask1, bid_volume1=b, ask_volume1=a,
        raw={"sim": True})


def build_ctx(refs, snaps):
    """构造已喂过快照的 StrategyContext：refs={code:RefRow}，snaps=[Snapshot]。"""
    ctx = StrategyContext(ref=refs)
    for s in snaps:
        ctx.update(s)
    return ctx


def new_cfg(**over):
    """一份带独立临时目录的配置：state 与 trades 流水各场景隔离，互不污染。"""
    c = Cfg()
    d = tempfile.mkdtemp(prefix="sim_paper_")  # 每 cfg 独占目录 → state/trades 完全隔离
    c.ledger_dir = d
    c.paper_state_file = str(Path(d) / "paper_state.json")  # 不预创建 → 起干净空仓
    for k, v in over.items():
        setattr(c, k, v)

    def _paper_state_files():
        base = Path(c.paper_state_file)
        files = [base]
        if getattr(c, "paper_v2_enabled", False):
            files.append(base.parent / f"{base.stem}_v2{base.suffix}")
        if getattr(c, "paper_v3_enabled", False):
            files.append(base.parent / f"{base.stem}_v3{base.suffix}")
        if getattr(c, "paper_v4_enabled", False):
            files.append(base.parent / f"{base.stem}_v4{base.suffix}")
        return tuple(files)

    c.paper_state_files = _paper_state_files
    return c


# ============================================================================
# A. 盘中动态重排：同池两票模型分接近，便宜+买盘强的爬到追高+偏贵票之前
# ============================================================================
def scenario_A():
    print("\n== A. 盘中动态重排（模型分接近时，盘中量决定次序）==")
    refs = {
        "000001": RefRow(expected_return=0.030),  # 模型分略低
        "000002": RefRow(expected_return=0.032),  # 模型分略高，但盘中偏贵+追高
    }
    snaps = [
        # 000001：现价 9.7 低于 VWAP 10.0（便宜）+ 买盘强（imb=+0.5），无跳空
        mk_snap("000001", 9.7, pre_close=10.0, open_px=10.0, vwap=10.0, imb=0.5, spread=0.001),
        # 000002：现价 10.3 高于 VWAP 10.0（偏贵）+ 高开 2% 吃预期 + 卖盘强（imb=-0.5）
        mk_snap("000002", 10.3, pre_close=10.0, open_px=10.2, vwap=10.0, imb=-0.5, spread=0.004),
    ]
    ctx = build_ctx(refs, snaps)
    rows = RerankScorer(new_cfg(), ctx).ranked()
    order = [r.code for r in rows]
    print(f"     重排后次序={order}  " + " | ".join(
        f"{r.code} exp={r.exp:.3f} adj={r.adj:+.3f} score={r.score:.4f}" for r in rows))
    check(order[0] == "000001",
          "便宜+买盘强票(000001)重排到 追高+偏贵+模型分更高票(000002)之前")
    check(rows[[r.code for r in rows].index("000001")].adj > 0, "000001 盘中 adj>0（上移）")
    check(rows[[r.code for r in rows].index("000002")].adj < 0, "000002 盘中 adj<0（下移）")


# ============================================================================
# B. 重排有界：模型分差距大时，盘中极端量也翻不动次序（模型锚定）
# ============================================================================
def scenario_B():
    print("\n== B. 重排有界（cap 保证模型分差大时不翻序）==")
    refs = {
        "000010": RefRow(expected_return=0.100),  # 模型分远高
        "000020": RefRow(expected_return=0.030),  # 模型分低
    }
    snaps = [
        # 000010：极端偏贵+追高+卖盘强（本该被狠压），但 adj 被 cap 到 -0.30
        mk_snap("000010", 11.0, pre_close=10.0, open_px=10.9, vwap=10.0, imb=-1.0, spread=0.01),
        # 000020：极端便宜+买盘强（本该被狠抬），adj 被 cap 到 +0.30
        mk_snap("000020", 9.0, pre_close=10.0, open_px=10.0, vwap=10.0, imb=1.0, spread=0.0),
    ]
    ctx = build_ctx(refs, snaps)
    cfg = new_cfg()
    rows = RerankScorer(cfg, ctx).ranked()
    order = [r.code for r in rows]
    by = {r.code: r for r in rows}
    print(f"     次序={order}  " + " | ".join(
        f"{r.code} exp={r.exp:.3f} adj={r.adj:+.3f} score={r.score:.4f}" for r in rows))
    check(order[0] == "000010", "模型分远高的票(000010)即便盘中极端不利仍居首（模型锚定）")
    check(abs(by["000010"].adj) <= cfg.rerank_cap + 1e-9, "000010 adj 被 clamp 到 ±cap 内")
    check(abs(by["000020"].adj) <= cfg.rerank_cap + 1e-9, "000020 adj 被 clamp 到 ±cap 内")


# ============================================================================
# C. 候选池封闭：expected_return<=0 / 无模型分的票，绝不进重排榜
# ============================================================================
def scenario_C():
    print("\n== C. 候选池封闭（池外票绝不入榜/被买）==")
    refs = {
        "000100": RefRow(expected_return=0.05),   # 池内
        "000200": RefRow(expected_return=-0.01),  # 模型看空，池外
        "000300": RefRow(expected_return=None),   # 无模型分，池外
    }
    snaps = [
        mk_snap("000100", 10.0, vwap=10.2, imb=0.3),
        # 池外两票即便盘中极其漂亮也不该入榜
        mk_snap("000200", 9.0, pre_close=10.0, vwap=10.0, imb=1.0),
        mk_snap("000300", 9.0, pre_close=10.0, vwap=10.0, imb=1.0),
    ]
    ctx = build_ctx(refs, snaps)
    rows = RerankScorer(new_cfg(), ctx).ranked()
    codes = {r.code for r in rows}
    print(f"     入榜={sorted(codes)}")
    check(codes == {"000100"}, "仅 expected_return>0 的票入榜（看空/无分票被挡在池外）")


# ============================================================================
# D. A股 T+1：当日买入的持仓，即便暴跌也不出场
# ============================================================================
def scenario_D():
    print("\n== D. A股 T+1（当日买入 held<1，暴跌不卖）==")
    refs = {"000001": RefRow(expected_return=0.05, atr=0.1)}
    snaps = [mk_snap("000001", 10.0, pre_close=10.0, vwap=10.0)]
    ctx = build_ctx(refs, snaps)
    trader = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    trader.maybe_trade(1455)  # 收盘前买入窗
    check(len(trader._state["positions"]) == 1, "买入窗成功建仓 1 只")
    # 同日价格暴跌 -20%，再评估出场
    ctx.update(mk_snap("000001", 8.0, pre_close=10.0, vwap=10.0))
    trader._run_sells()
    check(len(trader._state["positions"]) == 1, "当日买入暴跌 -20% 仍不卖（T+1 守住，无 T+0 违规）")


# ============================================================================
# E. 次日出场：held>=1 时 5 级出场按优先级各自触发
# ============================================================================
def _exit_trader(px, *, buy_price=10.0, peak=None, vwap=None, atr=None,
                 stop_loss=0.05, take_profit=0.09, vwap_break=0.02, trail_k=3.0):
    """构造一个含【历史持仓】(buy_date 远在过去→held>=1) 的 trader，喂当前价后评估出场。"""
    refs = {"000001": RefRow(expected_return=0.05, atr=atr)}
    ctx = build_ctx(refs, [mk_snap("000001", px, pre_close=10.0,
                                   vwap=(vwap if vwap is not None else px))])
    cfg = new_cfg(paper_stop_loss=stop_loss, paper_take_profit=take_profit,
                  paper_vwap_break=vwap_break, paper_trail_k=trail_k)
    trader = PaperTrader(cfg, ctx, FakeNotifier())
    trader._state["positions"] = [{
        "code": "000001", "name": "", "buy_date": "2020-01-02",
        "buy_time": "14:50:00", "buy_price": buy_price, "shares": 1000,
        "peak": peak if peak is not None else buy_price,
        "cost_basis": buy_price * 1000, "exp": 0.05}]
    return trader


def _last_exit(trader):
    """读该 trader 最近一笔平仓流水的 exit_reason（无则 None）。"""
    tf = Path(trader._trades_file)
    if not tf.exists():
        return None
    import json as _j
    lines = [l for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
    return _j.loads(lines[-1])["exit_reason"] if lines else None


def scenario_E():
    print("\n== E. 次日出场（5 级按优先级触发）==")
    # 1) 硬止损：ret=-6% <= -5%
    t = _exit_trader(9.4, vwap=9.4)  # vwap=px 避免 vwap_break 抢先
    t._run_sells()
    check(len(t._state["positions"]) == 0 and _last_exit(t) == "stop_loss",
          "ret -6% → stop_loss（硬止损）")
    # 2) 止盈：ret=+10% >= +9%
    t = _exit_trader(11.0, vwap=11.0)
    t._run_sells()
    check(len(t._state["positions"]) == 0 and _last_exit(t) == "take_profit",
          "ret +10% → take_profit（止盈）")
    # 3) 移动止盈：peak=10.5, atr=0.1,k=3 → 止损线10.2；px=10.1<=10.2；ret 仅 +1% 不触前两级
    t = _exit_trader(10.1, peak=10.5, atr=0.1, vwap=10.1)
    t._run_sells()
    check(len(t._state["positions"]) == 0 and _last_exit(t) == "trailing_stop",
          "px 回撤破吊灯线(peak-3*ATR) → trailing_stop（移动止盈）")
    # 4) 破位：px=9.8 高于 vwap*0.98=9.996？ 9.8<9.996 触发；ret -2% 不触止损；无 atr 不触吊灯
    t = _exit_trader(9.8, vwap=10.2, atr=None)
    t._run_sells()
    check(len(t._state["positions"]) == 0 and _last_exit(t) == "vwap_break",
          "px 跌破 VWAP*(1-2%) → vwap_break（破位）")
    # 5) 时间上限：ret≈0、无 atr、vwap≈px（不破位）→ 落到 T+N 兜底
    t = _exit_trader(10.0, vwap=10.0, atr=None)
    t._run_sells()
    check(len(t._state["positions"]) == 0 and _last_exit(t) == "time_cap",
          "持有达 T+N 且无其它触发 → time_cap（到期兜底）")


# ============================================================================
# F. 入场过滤：追高候选被跳过，买入顺延到干净的下一名
# ============================================================================
def scenario_F():
    print("\n== F. 入场过滤（追高/偏贵/卖盘强被跳过，顺延买下一名）==")
    refs = {
        "000700": RefRow(expected_return=0.05),  # 模型分更高但高开吃预期（追高）
        "000800": RefRow(expected_return=0.04),  # 干净：不追高、不偏贵、买盘中性
    }
    snaps = [
        # 000700：高开 3%，(open/pre-1)/exp=0.03/0.05=0.6 >= paper_entry_gap_eaten → 追高，应跳过
        mk_snap("000700", 10.3, pre_close=10.0, open_px=10.3, vwap=10.3, imb=0.0, spread=0.001),
        # 000800：无跳空、现价=VWAP、买盘中性 → 不触发任何入场过滤
        mk_snap("000800", 10.0, pre_close=10.0, open_px=10.0, vwap=10.0, imb=0.0, spread=0.001),
    ]
    ctx = build_ctx(refs, snaps)
    trader = PaperTrader(new_cfg(paper_buy_n=1, paper_entry_gap_eaten=0.6),
                         ctx, FakeNotifier())
    trader.maybe_trade(1455)
    held = [p["code"] for p in trader._state["positions"]]
    print(f"     持仓={held}")
    check("000700" not in held, "追高票(000700 高开吃预期60%)被入场过滤跳过")
    check(held == ["000800"], "买入顺延到干净候选(000800)")


# ============================================================================
# G. 优雅降级：缺 VWAP/盘口量时退化为纯模型序，不崩
# ============================================================================
def scenario_G():
    print("\n== G. 优雅降级（缺盘中量→纯模型序，不崩）==")
    refs = {
        "000900": RefRow(expected_return=0.06),
        "000901": RefRow(expected_return=0.03),
    }
    # 只有裸价，无 volume/amount(→无VWAP)、买卖量相等(imb=0)、价差中性
    snaps = [
        Snapshot(code="000900", last=10.0, pre_close=10.0, open=10.0,
                 high_limited=11.0, low_limited=9.0, raw={}),
        Snapshot(code="000901", last=20.0, pre_close=20.0, open=20.0,
                 high_limited=22.0, low_limited=18.0, raw={}),
    ]
    ctx = build_ctx(refs, snaps)
    rows = RerankScorer(new_cfg(), ctx).ranked()
    order = [r.code for r in rows]
    print(f"     次序={order}  " + " | ".join(
        f"{r.code} adj={r.adj:+.3f}" for r in rows))
    check(order == ["000900", "000901"], "缺盘中量时退化为纯模型分降序")
    check(all(abs(r.adj) < 1e-9 for r in rows), "缺量各票 adj=0（无盘中微调，不误判）")


# ============================================================================
# H. 模拟盘持仓保护性订阅：即使跌出 Top10 也必须保留报价
# ============================================================================
def scenario_H():
    """验证模拟盘持仓在订阅上限下仍受保护。"""
    print("\n== H. 模拟盘持仓保护性订阅（不被 Top10 / 上限挤掉）==")
    cfg = new_cfg()
    root = Path(cfg.ledger_dir)
    cfg.mobile_snapshot_file = root / "mobile_snapshot.json"
    cfg.holdings_file = root / "realtime_holdings.txt"
    cfg.predictions_file = root / "missing_predictions.parquet"
    cfg.universe_file = root / "missing_universe.txt"
    cfg.max_subscribe = 1
    Path(cfg.paper_state_file).write_text(json.dumps({
        "cash": 50000,
        "positions": [{"code": "600941.SH"}, {"code": "000766"}],
    }), encoding="utf-8")
    cfg.mobile_snapshot_file.write_text(json.dumps({
        "groups": {"全A": {"rows": [{"code": "000001"}]}}
    }, ensure_ascii=False), encoding="utf-8")
    codes = load_codes(cfg)
    print(f"     订阅={codes}")
    check(codes == ["600941", "000766"],
          "全部模拟盘持仓优先订阅，max_subscribe 截断也不丢报价")

    cfg30 = new_cfg()
    root30 = Path(cfg30.ledger_dir)
    cfg30.mobile_snapshot_file = root30 / "mobile_snapshot.json"
    cfg30.holdings_file = root30 / "missing_holdings.txt"
    cfg30.predictions_file = root30 / "missing_predictions.parquet"
    cfg30.universe_file = root30 / "missing_universe.txt"
    cfg30.max_subscribe = 100
    all_a_rows = [{"code": f"600{index:03d}"} for index in range(30)]
    cfg30.mobile_snapshot_file.write_text(json.dumps({
        "groups": {
            "白名单": {"rows": all_a_rows[:10]},
            "全A": {"rows": all_a_rows},
            "创新药": {"rows": all_a_rows[20:30]},
        }
    }, ensure_ascii=False), encoding="utf-8")
    top30_codes = load_codes(cfg30)
    check(len(top30_codes) == 30 and top30_codes[-1] == "600029",
          "模拟盘订阅完整读取全A Top30，跨组重复代码只保留一次")


# ============================================================================
# I. 成本后净预期门：毛收益不能覆盖 round-trip 成本的票不入池
# ============================================================================
def scenario_I():
    """验证原始预期覆盖成本+安全边际，校准值不再硬过滤。"""
    print("\n== I. 原始预期收益门（成本 + 安全边际）==")
    refs = {
        "000910": RefRow(expected_return=0.0025, calibrated_return=0.00156,
                          calibrated_net_return=-0.00044, win_rate=0.49),
        "000911": RefRow(expected_return=0.0050, calibrated_return=0.00156,
                          calibrated_net_return=-0.00044, win_rate=0.52),
    }
    snaps = [mk_snap("000910", 10.0), mk_snap("000911", 10.0)]
    rows = RerankScorer(new_cfg(), build_ctx(refs, snaps)).ranked()
    check([r.code for r in rows] == ["000911"], "原始预期覆盖 0.20% 成本 + 0.10% 安全边际")
    text = expected_return_text(refs["000910"], 0.0025, 0.002)
    print(f"     展示={text}")
    check("模型净+0.05%" in text and "历史校准+0.16%" in text,
          "展示同时给出原始模型净收益与历史校准收益")


# ============================================================================
# J. 高开惩罚默认关闭：隔夜动量不再被错误减分或拦买
# ============================================================================
def scenario_J():
    """验证高开惩罚默认关闭且不再生成误导信号。"""
    print("\n== J. 高开惩罚默认关闭（保留隔夜动量）==")
    refs = {"000920": RefRow(expected_return=0.05, calibrated_net_return=0.01)}
    snap = mk_snap("000920", 10.3, pre_close=10.0, open_px=10.3,
                   vwap=10.3, imb=0.0, spread=0.0025)
    ctx = build_ctx(refs, [snap])
    cfg = new_cfg(rerank_w_vwap=0.0, rerank_w_imb=0.0, rerank_w_spread=0.0)
    row = RerankScorer(cfg, ctx).ranked()[0]
    trader = PaperTrader(cfg, ctx, FakeNotifier())
    check(all(reason[0] != "追高" for reason in row.reasons), "高开不再触发错误方向的重排减分")
    check(trader._entry_skip("000920", 0.05, 10.3) is None, "高开不再触发错误方向的入场阻断")
    check("gap_calibrate" not in {s.name for s in default_strategies()},
          "高开吃预期不再进入默认信号策略")


# ============================================================================
# K. 到期卖出口径：T+1 风险退出全天有效，纯 time_cap 只在收盘前执行
# ============================================================================
def scenario_K():
    """验证到期仓不会在 T+1 开盘卖飞，而在收盘前按 close 口径退出。"""
    print("\n== K. 到期卖出口径（风险全天、到期收盘前）==")
    trader = _exit_trader(10.0, vwap=10.0, atr=None)
    trader._run_sells(930)
    check(len(trader._state["positions"]) == 1,
          "T+1 早盘无风险信号时继续持有，不把 close→close 做成 next-open")
    trader._run_sells(1450)
    check(len(trader._state["positions"]) == 0 and _last_exit(trader) == "time_cap",
          "T+1 到 14:50 后按 time_cap 收盘前退出")

    refs = {"000930": RefRow(expected_return=0.05, calibrated_net_return=0.01)}
    ctx = build_ctx(refs, [mk_snap("000930", 10.0, vwap=10.0)])
    before = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    before.maybe_trade(1449)
    check(not before._state["positions"], "14:49 尚未进入买入窗口")
    at_start = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    at_start.maybe_trade(1450)
    check(len(at_start._state["positions"]) == 1, "14:50 先执行卖出腿，再允许按收盘口径建仓")
    at_end = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    at_end.maybe_trade(1455)
    check(len(at_end._state["positions"]) == 1, "14:55 仍在买入窗口闭区间内")
    after = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    after.maybe_trade(1456)
    check(not after._state["positions"], "14:56 已超过买入窗口，不再建仓")


# ============================================================================
# L. 决策审计：买入全候选理由 + 卖出后反事实幂等补齐
# ============================================================================
def scenario_L():
    """验证决策快照完整记录过滤/成交，反事实重复补齐不产生重复行。"""
    print("\n== L. 决策审计（买入 trace + 卖出反事实）==")
    refs = {
        "000940": RefRow(expected_return=0.06, calibrated_net_return=0.01),
        "000941": RefRow(expected_return=0.05, calibrated_net_return=0.01),
        "000942": RefRow(expected_return=0.0025, calibrated_net_return=-0.001),
    }
    snaps = [
        mk_snap("000940", 10.2, vwap=10.0),
        mk_snap("000941", 10.0, vwap=10.0),
        mk_snap("000942", 10.0, vwap=10.0),
    ]
    trader = PaperTrader(new_cfg(paper_buy_n=1), build_ctx(refs, snaps), FakeNotifier())
    trader.maybe_trade(1455)
    decisions = [json.loads(x) for x in trader._decisions_file.read_text().splitlines()]
    by_code = {r["code"]: r for r in decisions[0]["candidates"]}
    check(by_code["000940"]["entry_decision"] == "filtered",
          "买入快照记录高于 VWAP 的过滤原因")
    check(by_code["000941"]["entry_decision"] == "bought",
          "买入快照记录最终成交、股数和成本")
    check(by_code["000942"]["status"] == "excluded_raw_return_gate" and
          by_code["000942"]["raw_min_return"] == 0.003,
          "买入快照记录原始预期成本安全边际门槛")

    trade = {
        "action": "sell", "time": "2026-08-04 14:50:00", "code": "000941",
        "buy_date": "2026-08-03", "buy_time": "14:55:00", "buy_price": 10.0,
        "sell_date": "2026-08-04", "sell_price": 10.1, "shares": 1000,
        "pnl": 79.9, "return": 0.008, "exit_reason": "time_cap",
    }
    trader._trades_file.write_text(json.dumps(trade) + "\n", encoding="utf-8")
    original = paper_mod.warehouse.load_price_tail
    paper_mod.warehouse.load_price_tail = lambda *_args, **_kwargs: pd.DataFrame({
        "date": pd.to_datetime(["2026-08-04", "2026-08-05"]),
        "close": [10.3, 10.5], "high": [10.4, 10.6],
    })
    try:
        trader._refresh_counterfactuals()
        trader._refresh_counterfactuals()
    finally:
        paper_mod.warehouse.load_price_tail = original
    counterfactuals = [
        json.loads(x) for x in trader._counterfactuals_file.read_text().splitlines()]
    check(len(counterfactuals) == 1, "同一卖出 trade_id 重复补齐仍只有一条记录")
    check(len(counterfactuals[0]["markouts"]) == 2 and
          counterfactuals[0]["markouts"][0]["opportunity_pnl"] > 0,
          "反事实记录卖出日和后续日的收盘/最高价及机会损益")


# ============================================================================
# M. V2 赛马对照：动态分配 + 保护止盈 + 统一买窗
# ============================================================================
def scenario_M():
    """验证 V2 端到端：资金利用率提升 / 保护性止盈不误触发 / 统一买窗。"""
    print("\n== M. V2 赛马（动态分配 + 保护止盈 + 买窗 1450-1455）==")
    refs = {
        "000945": RefRow(expected_return=0.05, calibrated_net_return=0.01),
        "000946": RefRow(expected_return=0.04, calibrated_net_return=0.01),
    }
    snaps = [mk_snap("000945", 450.0, vwap=450.0), mk_snap("000946", 10.0, vwap=10.0)]
    trader = V2PaperTrader(
        new_cfg(paper_buy_n=2), build_ctx(refs, snaps), FakeNotifier())
    trader.maybe_trade(1455)
    held = [p["code"] for p in trader._state["positions"]]
    print(f"     V2持仓={held}")
    check("000946" in held, "V2 高价股资金不足名额顺延到下一只")
    # 名义等额分配但高价股只买得起 100 股(¥45,045)，余钱集中到第二只
    total_shares = sum(p["shares"] for p in trader._state["positions"])
    check(total_shares >= 5500,
          f"V2 动态分配总股数 ≥ 5500（实际 {total_shares}）")

    # 场景 2：V2 继承四版公共买窗，不再保留独立的 14:57 上限。
    trader2 = V2PaperTrader(
        new_cfg(paper_buy_n=1), build_ctx(refs, snaps), FakeNotifier())
    check(trader2._buy_start == 1450 and trader2._buy_end == 1455,
          f"V2 继承公共买窗 1450-1455（got {trader2._buy_start}-{trader2._buy_end}）")

    # 场景 3：保护性止盈完整校验
    cfg_v2 = new_cfg()
    trader3 = V2PaperTrader(cfg_v2, build_ctx(
        {"000948": RefRow(expected_return=0.05)},
        [mk_snap("000948", 10.05, vwap=10.05)]), FakeNotifier())
    print(f"     breakeven_arm={trader3._breakeven_arm} breakeven_margin={trader3._breakeven_margin}")
    pos = {"code": "000948", "buy_price": 10.0, "peak": 10.4, "shares": 1000,
           "cost_basis": 10000.0}
    reason = trader3._exit_decision(pos, 10.05, 1, True)
    check(reason == "breakeven_stop",
          f"V2 浮盈 4% 回撤到 +0.5% 触发保本退出（got {reason}）")
    # 情景 b：peak=10.1 (< buy_price+3%=10.3)，不触发保护，落到 time_cap
    reason2 = trader3._exit_decision(
        {**pos, "peak": 10.1}, 10.09, 1, True)
    check(reason2 != "breakeven_stop",
          "V2 peak 仅 +1% 不误触发保本退出")

    # 场景 4：V2 审计文件名必须保留原后缀（后缀派生不得吃掉 .jsonl 的 l）
    names = {n: V2PaperTrader._suffixed(Path("/x") / n).name for n in (
        "paper_state.json", "paper_trades.jsonl",
        "paper_buy_decisions.jsonl", "paper_sell_counterfactuals.jsonl")}
    print(f"     V2文件名={list(names.values())}")
    check(names == {
        "paper_state.json": "paper_state_v2.json",
        "paper_trades.jsonl": "paper_trades_v2.jsonl",
        "paper_buy_decisions.jsonl": "paper_buy_decisions_v2.jsonl",
        "paper_sell_counterfactuals.jsonl": "paper_sell_counterfactuals_v2.jsonl",
    }, "V2 四个审计文件名均为 _v2 且后缀完整")

    cfg_audit = new_cfg(paper_buy_n=1, paper_entry_ask_strong=0.2)
    ctx_audit = build_ctx(
        {"000947": RefRow(expected_return=0.05)},
        [mk_snap("000947", 10.0, pre_close=10.0, vwap=10.0,
                 imb=-1.0, spread=0.001)])
    v2_audit = V2PaperTrader(cfg_audit, ctx_audit, FakeNotifier())
    v2_audit._now_hhmm = 1455
    v2_audit._run_buys()
    zero_decision = json.loads(v2_audit._decisions_file.read_text().splitlines()[-1])
    check(zero_decision["event_type"] == "paper_buy_decision_v2" and
          zero_decision["decision_status"] == "all_candidates_filtered" and
          zero_decision["account_after"]["bought_count"] == 0,
          "V2 全部候选过滤时仍落独立决策审计，零成交也是有效样本")

    # 场景 5：炸板判定走公开接口，不再访问私有状态；T+1 持仓评估不抛异常
    ctx5 = build_ctx({"000949": RefRow(expected_return=0.05, atr=0.1)},
                     [mk_snap("000949", 10.5, pre_close=10.0, vwap=10.5)])
    trader5 = V2PaperTrader(new_cfg(), ctx5, FakeNotifier())
    yesterday = (_dt.date.today() - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    trader5._state["positions"] = [{
        "code": "000949", "buy_price": 10.0, "peak": 10.5, "shares": 1000,
        "cost_basis": 10010.0, "buy_date": yesterday, "buy_time": "14:56:00"}]
    trader5._run_sells(1030)
    check(len(trader5._state["positions"]) == 1,
          "V2 T+1 持仓早盘无风险信号时不抛异常且继续持有")

    # 场景 6：任一出场原因成交后，磁盘状态必须与内存一致（防重启重复卖出）
    ctx6 = build_ctx({"000950": RefRow(expected_return=0.05, atr=None)},
                     [mk_snap("000950", 10.0, pre_close=10.0, vwap=10.0)])
    trader6 = V2PaperTrader(new_cfg(), ctx6, FakeNotifier())
    trader6._state["positions"] = [{
        "code": "000950", "buy_price": 10.0, "peak": 10.0, "shares": 1000,
        "cost_basis": 10010.0, "buy_date": yesterday, "buy_time": "14:56:00"}]
    trader6._save_state()
    trader6._run_sells(1450)
    disk = json.loads(trader6._state_file.read_text())
    check(not disk["positions"] and abs(disk["cash"] - trader6._state["cash"]) < 0.01,
          "V2 到期卖出后磁盘状态与内存一致（现金已入账、持仓已移除）")

    # 场景 7：V2 构造绝不触碰 V1 的四个账户文件
    cfg7 = new_cfg()
    v1_state = Path(cfg7.paper_state_file)
    v1_state.parent.mkdir(parents=True, exist_ok=True)
    v1_state.with_name("paper_trades.jsonl").write_text(json.dumps({
        "action": "sell", "code": "000951", "buy_date": "2026-08-03",
        "buy_price": 10.0, "sell_date": "2026-08-04", "sell_price": 10.2,
        "shares": 1000, "pnl": 180.0, "return": 0.018,
        "exit_reason": "time_cap"}) + "\n", encoding="utf-8")
    before = {p.name for p in v1_state.parent.iterdir()}
    original = paper_mod.warehouse.load_price_tail
    paper_mod.warehouse.load_price_tail = lambda *_a, **_k: pd.DataFrame({
        "date": pd.to_datetime(["2026-08-04", "2026-08-05"]),
        "close": [10.5, 10.7], "high": [10.6, 10.8]})
    try:
        V2PaperTrader(cfg7, build_ctx({"000951": RefRow(expected_return=0.05)},
                                      [mk_snap("000951", 10.2)]), FakeNotifier())
    finally:
        paper_mod.warehouse.load_price_tail = original
    created = {p.name for p in v1_state.parent.iterdir()} - before
    v1_cf = v1_state.with_name("paper_sell_counterfactuals.jsonl")
    print(f"     V2构造新建={sorted(created)} V1反事实存在={v1_cf.exists()}")
    check(all("_v2" in n for n in created) and not v1_cf.exists(),
          "V2 构造只写 _v2 文件，不读不写 V1 状态/流水/审计")

    # 场景 8：跌停顺延按交易日计数，同日多次心跳不重复累加
    ld_snap = mk_snap("000952", 9.0, pre_close=10.0, vwap=9.0)
    object.__setattr__(ld_snap, "low_limited", 9.0)
    ctx8 = build_ctx({"000952": RefRow(expected_return=0.05)}, [ld_snap])
    trader8 = V2PaperTrader(new_cfg(), ctx8, FakeNotifier())
    trader8._state["positions"] = [{
        "code": "000952", "buy_price": 10.0, "peak": 10.0, "shares": 1000,
        "cost_basis": 10010.0, "buy_date": yesterday, "buy_time": "14:56:00"}]
    for _ in range(5):
        trader8._run_sells(1030)
    still_held = trader8._state["positions"]
    rolls = still_held[0].get("_ld_rolls") if still_held else None
    print(f"     跌停5次心跳后 _ld_rolls={rolls} 持仓={len(still_held)}")
    check(len(still_held) == 1 and rolls == 1,
          "V2 跌停同日多次心跳只累加 1 次，不误触发强制平仓")

    # 场景 9：持仓上限收缩目标只数，避免接近上限时资金闲置
    refs9 = {"000953": RefRow(expected_return=0.05, calibrated_net_return=0.01),
             "000954": RefRow(expected_return=0.04, calibrated_net_return=0.01)}
    ctx9 = build_ctx(refs9, [mk_snap("000953", 10.0, vwap=10.0),
                             mk_snap("000954", 10.0, vwap=10.0)])
    trader9 = V2PaperTrader(
        new_cfg(paper_buy_n=2, paper_max_positions=2), ctx9, FakeNotifier())
    trader9._state["positions"] = [{
        "code": "000955", "buy_price": 10.0, "peak": 10.0, "shares": 100,
        "cost_basis": 1001.0, "buy_date": yesterday, "buy_time": "14:56:00"}]
    trader9.maybe_trade(1455)
    new_pos = [p for p in trader9._state["positions"] if p["code"] != "000955"]
    print(f"     上限2已持1只 → 新建{len(new_pos)}只 股数={[p['shares'] for p in new_pos]}")
    check(len(new_pos) == 1 and new_pos[0]["shares"] >= 9000,
          "V2 剩余 1 个名额时全额投入，不按 buy_n 均分闲置现金")

    # 场景 10：订阅保护覆盖 V2 持仓
    cfg10 = new_cfg()
    v2_state = Path(cfg10.paper_state_file)
    v2_state.parent.mkdir(parents=True, exist_ok=True)
    (v2_state.parent / "paper_state_v2.json").write_text(
        json.dumps({"cash": 0, "positions": [{"code": "600941"}]}), encoding="utf-8")
    cfg10.mobile_snapshot_file = v2_state.parent / "mobile_snapshot.json"
    cfg10.holdings_file = v2_state.parent / "realtime_holdings.txt"
    cfg10.predictions_file = v2_state.parent / "missing.parquet"
    cfg10.universe_file = v2_state.parent / "missing.txt"
    cfg10.max_subscribe = 1
    cfg10.mobile_snapshot_file.write_text(
        json.dumps({"groups": {"全A": {"rows": [{"code": "000001"}]}}}), encoding="utf-8")
    codes10 = load_codes(cfg10)
    print(f"     订阅={codes10}")
    check("600941" in codes10,
          "V2 持仓同样进入保护性订阅，max_subscribe 截断也不丢报价")


# ============================================================================
# N. V3 赛马：盘口确认 + 可成交价 + ATR 出场 + 跌停阻塞
# ============================================================================
def scenario_N():
    print("\n== N. V3 赛马（ask1/bid1 成交 + 当日预测 + ATR 自适应出场）==")
    today = _dt.date.today().strftime("%Y-%m-%d")
    refs = {
        "000960": RefRow(expected_return=0.06, calibrated_net_return=0.01,
                          atr=0.2, prediction_date="2020-01-02"),
        "000961": RefRow(expected_return=0.05, calibrated_net_return=0.01,
                          atr=0.2, prediction_date=today),
    }
    snaps = [
        mk_snap("000960", 10.0, vwap=10.0, imb=0.5, spread=0.001),
        mk_snap("000961", 10.0, vwap=10.0, imb=0.5, spread=0.002),
    ]
    cfg = new_cfg(paper_buy_n=1)
    trader = V3PaperTrader(cfg, build_ctx(refs, snaps), FakeNotifier())
    trader.maybe_trade(1455)
    held = trader._state["positions"]
    check([p["code"] for p in held] == ["000961"],
          "V3 过滤非当日预测并顺延买入下一名")
    expected_ask = trader._ctx.snapshot_of("000961").ask_price1
    check(held and held[0]["buy_price"] == round(expected_ask, 3) and
          held[0]["buy_fill_source"] == "ask1",
          "V3 按卖一 ask1 买入，不再按 last 虚拟成交")
    decision = json.loads(trader._decisions_file.read_text().splitlines()[0])
    by_code = {r["code"]: r for r in decision["candidates"]}
    check(by_code["000960"]["entry_decision"] == "filtered" and
          "预测非当日" in by_code["000960"]["entry_filter_reason"],
          "V3 审计记录预测日期过滤原因")
    check(decision["schema_version"] == 3 and
          decision["event_type"] == "paper_buy_decision_v3",
          "V3 抽取版本钩子后仍保持原审计 schema 和事件类型")
    trader._ctx.state_of("000961").cur_ts -= 1000
    stale_reason, _ = trader._entry_quote("000961", 0.05)
    check(stale_reason is not None and "行情过期" in stale_reason,
          "V3 拒绝使用超过新鲜度门槛的旧快照买入")

    names = {n: V3PaperTrader._suffixed(Path("/x") / n).name for n in (
        "paper_state.json", "paper_trades.jsonl",
        "paper_buy_decisions.jsonl", "paper_sell_counterfactuals.jsonl")}
    check(names == {
        "paper_state.json": "paper_state_v3.json",
        "paper_trades.jsonl": "paper_trades_v3.jsonl",
        "paper_buy_decisions.jsonl": "paper_buy_decisions_v3.jsonl",
        "paper_sell_counterfactuals.jsonl": "paper_sell_counterfactuals_v3.jsonl",
    }, "V3 四个账户文件均以 _v3 隔离")

    yesterday = (_dt.date.today() - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    ctx_exit = build_ctx(
        {"000962": RefRow(expected_return=0.05, atr=0.2, prediction_date=today)},
        [mk_snap("000962", 10.82, pre_close=10.0, vwap=10.7,
                 imb=0.2, spread=0.001)])
    exit_trader = V3PaperTrader(new_cfg(), ctx_exit, FakeNotifier())
    exit_trader._state["positions"] = [{
        "code": "000962", "buy_price": 10.0, "peak": 10.8, "peak_bid": 10.8,
        "shares": 1000, "cost_basis": 10010.0, "buy_date": yesterday,
        "buy_time": "14:56:00", "prediction_date": today,
    }]
    sell_bid = ctx_exit.snapshot_of("000962").bid_price1
    exit_trader._run_sells(1030)
    trade = json.loads(exit_trader._trades_file.read_text().splitlines()[-1])
    check(trade["exit_reason"] == "atr_take_profit",
          "V3 达到 4ATR 后触发 ATR 止盈")
    check(trade["sell_price"] == round(sell_bid, 3) and
          trade["sell_fill_source"] == "bid1",
          "V3 按买一 bid1 卖出，不再按 last 虚拟成交")

    ctx_trail = build_ctx(
        {"000963": RefRow(expected_return=0.05, atr=0.2, prediction_date=today)},
        [mk_snap("000963", 10.29, pre_close=10.0, vwap=10.3,
                 imb=0.0, spread=0.001)])
    trail_trader = V3PaperTrader(new_cfg(), ctx_trail, FakeNotifier())
    pos_trail = {
        "code": "000963", "buy_price": 10.0, "peak": 10.7, "peak_bid": 10.7,
    }
    reason = trail_trader._exit_decision(
        pos_trail, ctx_trail.snapshot_of("000963").bid_price1, 1, False)
    check(reason == "atr_trailing", "V3 盈利达到 2ATR 后回撤 2ATR 触发移动止盈")
    stop_reason = trail_trader._exit_decision(
        {**pos_trail, "peak": 10.0, "peak_bid": 10.0}, 9.59, 1, False)
    check(stop_reason == "atr_stop", "V3 买一价跌破入场价 2ATR 时触发硬止损")

    ld_snap = mk_snap("000964", 9.0, pre_close=10.0, vwap=9.0, imb=-1.0)
    ld_snap.low_limited = 9.0
    ctx_ld = build_ctx(
        {"000964": RefRow(expected_return=0.05, atr=0.2, prediction_date=today)},
        [ld_snap])
    ld_trader = V3PaperTrader(new_cfg(paper_limit_down_roll_max=0), ctx_ld, FakeNotifier())
    ld_trader._state["positions"] = [{
        "code": "000964", "buy_price": 10.0, "peak": 10.0, "peak_bid": 10.0,
        "shares": 1000, "cost_basis": 10010.0, "buy_date": yesterday,
        "buy_time": "14:56:00", "prediction_date": today,
    }]
    ld_trader._run_sells(1450)
    check(len(ld_trader._state["positions"]) == 1 and
          ld_trader._state["positions"][0].get("v3_sell_blocked", {}).get("reason") ==
          "跌停无买盘承接",
          "V3 跌停且买一量为零时阻塞卖出，不按 last 强制平仓")

    liquid_ld = mk_snap("000965", 9.0, pre_close=10.0, vwap=9.0, imb=1.0)
    liquid_ld.low_limited = 9.0
    ctx_liquid_ld = build_ctx(
        {"000965": RefRow(expected_return=0.05, atr=0.2, prediction_date=today)},
        [liquid_ld])
    liquid_trader = V3PaperTrader(new_cfg(), ctx_liquid_ld, FakeNotifier())
    liquid_trader._state["positions"] = [{
        "code": "000965", "buy_price": 10.0, "peak": 10.0, "peak_bid": 10.0,
        "shares": 1000, "cost_basis": 10010.0, "buy_date": yesterday,
        "buy_time": "14:56:00", "prediction_date": today,
    }]
    liquid_trader._run_sells(1450)
    check(not liquid_trader._state["positions"],
          "V3 跌停价仍有有效买一承接时允许按 bid1 卖出")

    cfg_sub = new_cfg(paper_v2_enabled=False, paper_v3_enabled=True)
    root = Path(cfg_sub.ledger_dir)
    (root / "paper_state_v3.json").write_text(
        json.dumps({"cash": 0, "positions": [{"code": "600941"}]}), encoding="utf-8")
    cfg_sub.mobile_snapshot_file = root / "mobile_snapshot.json"
    cfg_sub.holdings_file = root / "holdings.txt"
    cfg_sub.predictions_file = root / "missing.parquet"
    cfg_sub.universe_file = root / "missing.txt"
    cfg_sub.max_subscribe = 1
    cfg_sub.mobile_snapshot_file.write_text(
        json.dumps({"groups": {"全A": {"rows": [{"code": "000001"}]}}}), encoding="utf-8")
    check(load_codes(cfg_sub) == ["600941"],
          "V3 持仓进入保护性订阅并优先于候选名单")


# ============================================================================
# O. V4：ETF 板块相对弱势过滤（V3 规则保持不变）
# ============================================================================
def scenario_O():
    print("\n== O. V4 赛马（V3 + 行业ETF相对弱势回避）==")
    cfg = new_cfg(paper_buy_n=1, paper_v2_enabled=False, paper_v3_enabled=False,
                  paper_v4_enabled=True)
    root = Path(cfg.ledger_dir)
    cfg.sector_meta_file = root / "all_a_stock_meta.parquet"
    pd.DataFrame([
        {"code": "000970", "a_industry": "数字芯片设计",
         "a_industries": "数字芯片设计、集成电路、电子"},
        {"code": "000971", "a_industry": "股份制银行Ⅲ",
         "a_industries": "股份制银行Ⅲ、银行Ⅱ、银行"},
        {"code": "000972", "a_industry": "印制电路板",
         "a_industries": "印制电路板、军工电子Ⅱ、电子"},
    ]).to_parquet(cfg.sector_meta_file, index=False)
    sector = SectorETFContext(cfg)
    now = time.time()
    sector.update(Snapshot(code="510300.SH", last=10.0, pre_close=10.0), now=now)
    weak_signals = sector.update(
        Snapshot(code="512480.SH", last=9.95, pre_close=10.0), now=now)
    sector.update(Snapshot(code="512800.SH", last=10.01, pre_close=10.0), now=now)
    check(bool(weak_signals) and weak_signals[0].kind == "sector_etf_weak",
          "ETF 相对沪深300跌 0.5% 时产生 sector_etf_weak 状态信号")
    check(subscription_code("159928.SZ", lambda _: "159928.SH") == "159928.SZ" and
          subscription_code("000001", lambda _: "000001.SZ") == "000001.SZ",
          "订阅层保留 ETF 完整后缀，仅对旧六位股票代码推断交易所")

    today = _dt.date.today().strftime("%Y-%m-%d")
    ctx = build_ctx({
        "000970": RefRow(expected_return=0.06, atr=0.2, prediction_date=today),
        "000971": RefRow(expected_return=0.05, atr=0.2, prediction_date=today),
        "000972": RefRow(expected_return=0.04, atr=0.2, prediction_date=today),
    }, [
        mk_snap("000970", 10.0, pre_close=10.0, vwap=10.0, imb=0.2, spread=0.001),
        mk_snap("000971", 10.0, pre_close=10.0, vwap=10.0, imb=0.2, spread=0.001),
        mk_snap("000972", 10.0, pre_close=10.0, vwap=10.0, imb=0.2, spread=0.001),
    ])
    trader = V4PaperTrader(cfg, ctx, FakeNotifier(), sector)
    trader._now_hhmm = 1455
    trader._run_buys()
    held = [p["code"] for p in trader._state["positions"]]
    check(held == ["000971"], "V4 过滤弱势半导体候选并顺延买入中性银行候选")
    position = trader._state["positions"][0]
    sector_entry = position.get("sector_etf", {})
    check(sector_entry.get("etf_code") == "512800.SH",
          "V4 持仓锁定入场行业ETF上下文，供平仓归因")
    check(sector_entry.get("mapping_version") == "industry_etf_exact_v2" and
          sector_entry.get("mapping_source") == "exact_primary" and
          sector_entry.get("mapping_confidence") == 1.0 and
          sector_entry.get("stock_industry") == "股份制银行Ⅲ",
          "V4 审计记录映射版本、来源、置信度和精确主行业")
    audit = json.loads(trader._decisions_file.read_text().splitlines()[-1])
    weak_row = next(r for r in audit["candidates"] if r["code"] == "000970")
    check(audit["schema_version"] == 4 and audit["event_type"] == "paper_buy_decision_v4" and
          "行业ETF相对弱势" in weak_row.get("entry_filter_reason", ""),
          "V4 独立审计记录版本、ETF弱势过滤原因和候选快照")
    check(all("_v4" in p.name for p in (
        trader._state_file, trader._trades_file, trader._decisions_file,
        trader._counterfactuals_file)),
          "V4 四个账户文件均以 _v4 隔离")
    missing_skip, missing_quote = trader._entry_quote_detail("000972", 0.04)
    missing_assessment = sector.assessment_for_stock("000972")
    check(missing_skip is None and missing_quote is not None and
          missing_assessment["mapping_source"] == "unmapped" and
          missing_assessment["stock_industry"] == "印制电路板",
          "V4 只按主行业精确映射，父级文本命中军工也不误过滤")

    cfg_sub = new_cfg(paper_v2_enabled=False, paper_v3_enabled=False, paper_v4_enabled=True)
    sub_root = Path(cfg_sub.ledger_dir)
    (sub_root / "paper_state_v4.json").write_text(
        json.dumps({"cash": 0, "positions": [{"code": "600941"}]}), encoding="utf-8")
    cfg_sub.mobile_snapshot_file = sub_root / "mobile_snapshot.json"
    cfg_sub.holdings_file = sub_root / "holdings.txt"
    cfg_sub.predictions_file = sub_root / "missing.parquet"
    cfg_sub.universe_file = sub_root / "missing.txt"
    cfg_sub.max_subscribe = 1
    cfg_sub.mobile_snapshot_file.write_text(
        json.dumps({"groups": {"全A": {"rows": [{"code": "000001"}]}}}), encoding="utf-8")
    check(load_codes(cfg_sub) == ["600941"],
          "V4 持仓进入保护性订阅并优先于候选名单")


# ============================================================================
# P. 四版联合生命周期：同场买入、持久化重启、跨日卖出
# ============================================================================
def scenario_P():
    print("\n== P. V1-V4 联合买卖生命周期（隔离账户 + 重启恢复 + 跨日平仓）==")
    cfg = new_cfg(paper_buy_n=1)
    root = Path(cfg.ledger_dir)
    cfg.sector_meta_file = root / "all_a_stock_meta.parquet"
    pd.DataFrame([{
        "code": "000980", "a_industry": "股份制银行Ⅲ",
        "a_industries": "股份制银行Ⅲ、银行Ⅱ、银行",
    }]).to_parquet(cfg.sector_meta_file, index=False)

    sector = SectorETFContext(cfg)
    now = time.time()
    sector.update(Snapshot(code="510300.SH", last=10.0, pre_close=10.0), now=now)
    sector.update(Snapshot(code="512800.SH", last=10.01, pre_close=10.0), now=now)

    today = _dt.date.today().strftime("%Y-%m-%d")
    refs = {"000980": RefRow(
        expected_return=0.05, calibrated_net_return=0.01,
        atr=0.2, prediction_date=today,
    )}
    ctx = build_ctx(refs, [
        mk_snap("000980", 10.0, pre_close=10.0, vwap=10.0,
                imb=0.2, spread=0.001),
    ])
    notifier = FakeNotifier()
    traders = {
        "V1": PaperTrader(cfg, ctx, notifier),
        "V2": V2PaperTrader(cfg, ctx, notifier),
        "V3": V3PaperTrader(cfg, ctx, notifier),
        "V4": V4PaperTrader(cfg, ctx, notifier, sector),
    }
    check(all(t._buy_start == 1450 and t._buy_end == 1455
              for t in traders.values()),
          "V1-V4 统一使用 14:50-14:55 买入窗口")

    for trader in traders.values():
        trader.maybe_trade(1455)

    check(all(len(t._state["positions"]) == 1 for t in traders.values()),
          "V1-V4 在各自买入窗口均成功建立 1 只独立持仓")
    buy_prices = {version: t._state["positions"][0]["buy_price"]
                  for version, t in traders.items()}
    check(buy_prices["V1"] == 10.0 and buy_prices["V2"] == 10.0 and
          buy_prices["V3"] > 10.0 and buy_prices["V4"] > 10.0,
          "V1/V2 按 last 买入，V3/V4 按 ask1 买入")

    expected_decisions = {
        "V1": (1, "paper_buy_decision"),
        "V2": (2, "paper_buy_decision_v2"),
        "V3": (3, "paper_buy_decision_v3"),
        "V4": (4, "paper_buy_decision_v4"),
    }
    decision_ok = True
    for version, trader in traders.items():
        decision = json.loads(trader._decisions_file.read_text().splitlines()[-1])
        expected_schema, expected_event = expected_decisions[version]
        decision_ok = decision_ok and (
            decision["schema_version"] == expected_schema and
            decision["event_type"] == expected_event and
            decision["decision_status"] == "bought" and
            decision["account_after"]["bought_count"] == 1
        )
    check(decision_ok, "四版买入决策分别写入独立 schema、事件类型和成交结果")
    check(len({str(t._state_file) for t in traders.values()}) == 4 and
          len({str(t._decisions_file) for t in traders.values()}) == 4 and
          all(t._state_file.exists() for t in traders.values()),
          "四版状态与买入审计文件互不串写且均已持久化")

    # 将同一批已落盘持仓推进到历史交易日，再重建对象模拟盘中进程重启。
    for trader in traders.values():
        trader._state["positions"][0]["buy_date"] = "2020-01-02"
        trader._save_state()
    ctx.update(mk_snap("000980", 9.4, pre_close=10.0, vwap=9.6,
                       imb=0.2, spread=0.001))
    restarted = {
        "V1": PaperTrader(cfg, ctx, notifier),
        "V2": V2PaperTrader(cfg, ctx, notifier),
        "V3": V3PaperTrader(cfg, ctx, notifier),
        "V4": V4PaperTrader(cfg, ctx, notifier, sector),
    }
    check(all(len(t._state["positions"]) == 1 for t in restarted.values()),
          "四版重启后均从各自状态文件恢复原持仓")

    for trader in restarted.values():
        trader.maybe_trade(1030)

    check(all(not t._state["positions"] for t in restarted.values()),
          "V1-V4 在 T+1 风险退出时均完成平仓，无持仓卡死")
    check(all(not json.loads(t._state_file.read_text())["positions"]
              for t in restarted.values()),
          "四版卖出后磁盘状态与内存一致，重启不会重复卖出")

    trades = {
        version: json.loads(trader._trades_file.read_text().splitlines()[-1])
        for version, trader in restarted.items()
    }
    check(trades["V1"]["exit_reason"] == "stop_loss" and
          trades["V2"]["exit_reason"] == "stop_loss" and
          trades["V3"]["exit_reason"] == "atr_stop" and
          trades["V4"]["exit_reason"] == "atr_stop",
          "V1/V2 触发百分比止损，V3/V4 触发 ATR 止损")
    check("sell_fill_source" not in trades["V1"] and
          "sell_fill_source" not in trades["V2"] and
          trades["V3"].get("sell_fill_source") == "bid1" and
          trades["V4"].get("sell_fill_source") == "bid1",
          "V1/V2 按 last 卖出，V3/V4 按 bid1 卖出")
    v4_sector = trades["V4"].get("sector_etf_at_entry") or {}
    check(v4_sector.get("etf_code") == "512800.SH" and
          v4_sector.get("mapping_source") == "exact_primary",
          "V4 平仓流水保留入场行业 ETF 与精确映射归因")
    check(len({trades[v]["trade_id"] for v in trades}) == 4,
          "四版 position/trade ID 独立，卖出流水不会互相覆盖")


# ============================================================================
# Q. 双融合实时主序：三模型收益门 + 融合 pred 百分位排序
# ============================================================================
def scenario_Q():
    print("\n== Q. 双融合实时主序（三模型收益门 + pred 百分位排序）==")
    cfg = new_cfg()
    cfg.predictions_file = Path(cfg.ledger_dir) / "predictions.parquet"
    pd.DataFrame([
        {"code": "000981", "date": "2026-08-04", "ridge_pred": 0.02,
         "elastic_pred": None, "extra_trees_pred": None, "pred": 99.0},
        {"code": "000981", "date": "2026-08-05", "ridge_pred": 0.004,
         "elastic_pred": 0.005, "extra_trees_pred": 0.006, "pred": 2.1},
        {"code": "000982", "date": "2026-08-05", "ridge_pred": 0.009,
         "elastic_pred": 0.008, "extra_trees_pred": 0.007, "pred": 1.3},
        {"code": "000983", "date": "2026-08-05", "ridge_pred": 0.0029,
         "elastic_pred": 0.0029, "extra_trees_pred": 0.0029, "pred": 2.5},
        {"code": "000984", "date": "2026-08-05", "ridge_pred": 0.004,
         "elastic_pred": None, "extra_trees_pred": None, "pred": 0.5},
    ]).to_parquet(cfg.predictions_file, index=False)
    returns, date, components, scores, percentiles = _load_expected_return(cfg)
    check(date == "2026-08-05" and abs(returns["000982"] - 0.0078) < 1e-12,
          "参考层按 30% Ridge + 20% ElasticNet + 50% ExtraTrees 融合收益")
    check(returns["000984"] == 0.004 and
          components["000984"]["source"] == "single_model_fallback" and
          components["000984"]["weights"] == {"ridge_pred": 1.0},
          "收益模型缺值时按可用权重归一化并安全退回 Ridge")
    check(scores["000983"] == 2.5 and percentiles["000983"] == 1.0 and
          percentiles["000981"] == 0.75,
          "参考层把无量纲 pred 转为当日全A百分位，不冒充收益率")

    refs = {
        "000981": RefRow(expected_return=0.0053,
                         return_components=components["000981"],
                         model_score=2.1, model_rank_pct=0.99),
        "000982": RefRow(expected_return=0.0078,
                         return_components=components["000982"],
                         model_score=1.3, model_rank_pct=0.95),
        "000983": RefRow(expected_return=0.0029,
                         return_components=components["000983"],
                         model_score=2.5, model_rank_pct=1.00),
    }
    snaps = [
        mk_snap("000981", 10.0, imb=0.0, spread=0.005),
        mk_snap("000982", 10.0, imb=0.0, spread=0.005),
        mk_snap("000983", 10.0, imb=0.0, spread=0.005),
    ]
    trace: list[dict] = []
    rows = RerankScorer(new_cfg(), build_ctx(refs, snaps)).ranked(trace=trace)
    check([row.code for row in rows] == ["000981", "000982"],
          "无量纲融合 pred 百分位主序优先于融合收益率高低")
    by_code = {item["code"]: item for item in trace}
    check(by_code["000983"]["status"] == "excluded_raw_return_gate",
          "排序融合分最高但融合收益不足 0.30% 的候选仍被阻断")
    check(by_code["000981"]["model_score"] == 2.1 and
          by_code["000981"]["model_rank_pct"] == 0.99 and
          by_code["000981"]["return_components"] == components["000981"],
          "买入审计同时保留收益融合各腿和无量纲融合排序字段")


def main() -> int:
    print("=" * 68)
    print(" 虚拟数据流验证：实时层重排/模拟盘/出场逻辑")
    print("=" * 68)
    for fn in (scenario_A, scenario_B, scenario_C, scenario_D,
               scenario_E, scenario_F, scenario_G, scenario_H,
               scenario_I, scenario_J, scenario_K, scenario_L,
               scenario_M, scenario_N, scenario_O, scenario_P, scenario_Q):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            global _FAIL
            _FAIL += 1
            import traceback
            print(f"  [FAIL] {fn.__name__} 抛异常：{type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 68)
    print(f" 小结：PASS={_PASS}  FAIL={_FAIL}")
    print("=" * 68)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
