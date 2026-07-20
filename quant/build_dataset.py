"""量化选股 · 一键构建本地数据集（并发拉取 + 落盘 parquet）。

用法：
    # 沪深300，价格/估值从2018年起，财报按季度
    python -m quant.build_dataset --universe hs300 --start 20180101
    # 全A，先抽样20只跑通
    python -m quant.build_dataset --universe full_a --limit 20

产出（默认 quant_data/）：
    price/{code}.parquet        每股日线
    valuation/{code}.parquet    每股估值时序
    financial_yjbb.parquet      业绩报表（含 ann_date 公告日，供防前视对齐）
    balance/income/cashflow.parquet   三大报表（按报告期）
    performance_forecast.parquet         业绩预告（含 ann_date 公告日）
    holder_num / dividend / block_trades / lhb.parquet   事件类
    margin_sse / margin_szse / margin_underlying_szse.parquet   两融汇总/标的

防数据穿越（look-ahead）设计：
- 财报表保留 ann_date（公告日）；做因子时按『交易日 >= ann_date』才可用，禁用报告期直接对齐。
- 指数成分随时间变化，训练时点成分需从现在起按月落盘积累（历史无法回补，避免幸存者偏差）。
- 标签（未来收益）只用 t 之后的价格，训练/回测按时间切分，不打乱。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from quant import config, datafeed, warehouse


def _quarter_ends(start: str) -> list[str]:
    """从 start 年到今年，生成各季度末 YYYYMMDD。"""
    y0 = int(start[:4])
    y1 = _dt.date.today().year
    ends = []
    for y in range(y0, y1 + 1):
        for md in ("0331", "0630", "0930", "1231"):
            d = f"{y}{md}"
            if d <= _dt.date.today().strftime("%Y%m%d"):
                ends.append(d)
    return ends


def _batch(codes, fn, save_fn, workers, label, existing_dir: str | None = None, skip_existing: bool = False):
    ok = fail = skipped = 0
    if skip_existing and existing_dir:
        before = len(codes)
        codes = [c for c in codes if not os.path.exists(os.path.join(existing_dir, f"{c}.parquet"))]
        skipped = before - len(codes)
    lock = threading.Lock()

    def one(c):
        nonlocal ok, fail
        try:
            df = fn(c)
            if df is not None and not df.empty:
                save_fn(c, df)
                with lock:
                    ok += 1
            else:
                with lock:
                    fail += 1
        except Exception:  # noqa: BLE001
            with lock:
                fail += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, codes))
    extra = f" / 跳过已存在 {skipped}" if skipped else ""
    print(f"  [{label}] 成功 {ok} / 失败 {fail}{extra}")


def _periodic_upsert(name, fn, qdates, keys_fn, workers):
    def one(d):
        try:
            return d, fn(d)
        except Exception:  # noqa: BLE001
            return d, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, df in ex.map(one, qdates):
            if df is None or df.empty:
                continue
            warehouse.upsert(name, df, keys=keys_fn(df))
    print(f"  [{name}] 累计 {len(warehouse.load(name))} 行")


def run(universe: str = "hs300", start: str = "20180101",
        limit: int = 0, workers: int = 12, with_events: bool = True,
        skip_price: bool = False, skip_valuation: bool = False,
        skip_fundamentals: bool = False, skip_existing: bool = False) -> dict:
    config.ensure_dirs()
    u = config.UNIVERSES.get(universe)
    if not u:
        raise ValueError(f"未知股票池 {universe}，可选：{list(config.UNIVERSES)}")
    codes = datafeed.universe(u["kind"], u["arg"])
    if limit:
        codes = codes[:limit]
    print(f"股票池 {universe}（{u['desc']}）：{len(codes)} 只")

    # 1) 每股：日线 + 估值（并发）
    if not skip_price:
        print("拉取日线…")
        _batch(codes, lambda c: datafeed.daily_price(c, start), warehouse.save_price, workers, "price",
               existing_dir=config.PRICE_DIR, skip_existing=skip_existing)
    else:
        print("跳过日线拉取")
    if not skip_valuation:
        print("拉取估值时序…")
        _batch(codes, datafeed.valuation, warehouse.save_valuation, workers, "valuation",
               existing_dir=config.VALUATION_DIR, skip_existing=skip_existing)
    else:
        print("跳过估值拉取")

    # 2) 全市场按报告期财报（每季一次整表调用，并发）
    qdates = _quarter_ends(start)
    if skip_fundamentals:
        print("跳过财报/业绩预告/股东户数/分红宽表拉取")
    else:
        print(f"拉取财报（{len(qdates)} 个报告期，并发 {workers}）…")
        for name, fn in (("financial_yjbb", datafeed.financial_yjbb),
                         ("balance", datafeed.balance_sheet),
                         ("income", datafeed.income),
                         ("cashflow", datafeed.cashflow)):
            _periodic_upsert(name, fn, qdates, lambda df: ["code", "report_date"], workers)

        # 3) 业绩预告 / 股东户数 / 分红（按报告期，并发）
        for name, fn in (("performance_forecast", datafeed.performance_forecast),
                         ("holder_num", datafeed.holder_num),
                         ("dividend", datafeed.dividend)):
            def keys_for(df, table=name):
                keys = ["code", "report_date"]
                if table == "performance_forecast" and "预测指标" in df.columns:
                    keys.append("预测指标")
                return keys
            _periodic_upsert(name, fn, qdates, keys_for, workers)

    # 4) 事件类 / 两融（按区间或当前日）
    if with_events:
        end = _dt.date.today().strftime("%Y%m%d")
        for name, fn in (("block_trades", datafeed.block_trades), ("lhb", datafeed.lhb)):
            try:
                warehouse.save(name, fn(start, end))
            except Exception:  # noqa: BLE001
                pass
            print(f"  [{name}] {len(warehouse.load(name))} 行")

        try:
            warehouse.upsert("margin_sse", datafeed.margin_sse(start, end), keys=["date"])
        except Exception:  # noqa: BLE001
            pass
        print(f"  [margin_sse] {len(warehouse.load('margin_sse'))} 行")

        for name, fn in (("margin_szse", datafeed.margin_szse),
                         ("margin_underlying_szse", datafeed.margin_underlying_szse)):
            try:
                df = fn(end)
                keys = ["code", "date"] if "code" in df.columns else ["date"]
                warehouse.upsert(name, df, keys=keys)
            except Exception:  # noqa: BLE001
                pass
            print(f"  [{name}] {len(warehouse.load(name))} 行")

    return {"universe": universe, "n_codes": len(codes)}


def main():
    ap = argparse.ArgumentParser(description="构建量化本地数据集")
    ap.add_argument("--universe", default="hs300", choices=list(config.UNIVERSES))
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--limit", type=int, default=0, help="仅取前N只（抽样跑通用）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--skip-price", action="store_true", help="跳过日线，续跑宽表/估值时使用")
    ap.add_argument("--skip-valuation", action="store_true", help="跳过估值，续跑宽表时使用")
    ap.add_argument("--skip-fundamentals", action="store_true", help="跳过财报/业绩预告/股东户数/分红宽表，续跑价格/估值时使用")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已有 price/valuation parquet，适合中断后续跑补缺")
    a = ap.parse_args()
    res = run(a.universe, a.start, a.limit, a.workers, not a.no_events, a.skip_price, a.skip_valuation, a.skip_fundamentals, a.skip_existing)
    print("完成：", res)


if __name__ == "__main__":
    main()
