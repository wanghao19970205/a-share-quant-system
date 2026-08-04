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
from pathlib import Path

import pandas as pd

from realtime import paper_trader as paper_mod
from realtime.snapshot import Snapshot
from realtime.strategy import StrategyContext, default_strategies
from realtime.rerank import RerankScorer
from realtime.paper_trader import PaperTrader
from realtime.reference import RefRow, expected_return_text
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
    paper_buy_n = 2
    paper_buy_start = 1455
    paper_time_cap_start = 1450
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
    rank_min_net_return = 0.0


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


# ============================================================================
# I. 成本后净预期门：毛收益不能覆盖 round-trip 成本的票不入池
# ============================================================================
def scenario_I():
    """验证成本后净收益门过滤毛正净负候选。"""
    print("\n== I. 成本后净预期门（过滤毛正净负候选）==")
    refs = {
        "000910": RefRow(expected_return=0.0030, calibrated_return=0.00156,
                          calibrated_net_return=-0.00044, win_rate=0.49),
        "000911": RefRow(expected_return=0.0050, calibrated_return=0.00578,
                          calibrated_net_return=0.00377, win_rate=0.52),
    }
    snaps = [mk_snap("000910", 10.0), mk_snap("000911", 10.0)]
    rows = RerankScorer(new_cfg(), build_ctx(refs, snaps)).ranked()
    check([r.code for r in rows] == ["000911"], "仅校准净收益覆盖成本的候选入榜")
    text = expected_return_text(refs["000910"], 0.0030, 0.002)
    print(f"     展示={text}")
    check("历史净-0.04%" in text and "毛+0.16%" in text,
          "展示同时给出扣费净收益与历史毛收益，不再四舍五入成模糊 +0.1%")


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
    buyer = PaperTrader(new_cfg(paper_buy_n=1), ctx, FakeNotifier())
    buyer.maybe_trade(1450)
    check(not buyer._state["positions"], "14:50 先完成旧仓到期腿，不提前建立新仓")
    buyer.maybe_trade(1455)
    check(len(buyer._state["positions"]) == 1, "14:55 后才按收盘口径建立新仓")


# ============================================================================
# L. 决策审计：买入全候选理由 + 卖出后反事实幂等补齐
# ============================================================================
def scenario_L():
    """验证决策快照完整记录过滤/成交，反事实重复补齐不产生重复行。"""
    print("\n== L. 决策审计（买入 trace + 卖出反事实）==")
    refs = {
        "000940": RefRow(expected_return=0.06, calibrated_net_return=0.01),
        "000941": RefRow(expected_return=0.05, calibrated_net_return=0.01),
        "000942": RefRow(expected_return=0.04, calibrated_net_return=-0.001),
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
    check(by_code["000942"]["status"] == "excluded_net_return_gate",
          "买入快照记录模型池前置净收益门槛")

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


def main() -> int:
    print("=" * 68)
    print(" 虚拟数据流验证：实时层重排/模拟盘/出场逻辑")
    print("=" * 68)
    for fn in (scenario_A, scenario_B, scenario_C, scenario_D,
               scenario_E, scenario_F, scenario_G, scenario_H,
               scenario_I, scenario_J, scenario_K, scenario_L):
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
