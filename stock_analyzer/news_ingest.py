"""白名单新闻入库 CLI：回填最近 N 个月 + 每日增量积累。

用法：
  # 回填最近 6 个月（公告 + 研报，带真实发布日期）：
  python -m stock_analyzer.news_ingest --mode backfill --months 6

  # 每日增量（近 lookback 天公告/研报 + 当日市场要闻快讯）：
  python -m stock_analyzer.news_ingest --mode daily --lookback-days 7

数据源见 news_store 模块说明。仅对白名单（snapshots/watchlist.txt）生效。
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from stock_analyzer import news, news_store

_DEFAULT_WATCHLIST = str(Path(__file__).resolve().parents[1] / "snapshots" / "watchlist.txt")


def read_watchlist(path: str) -> list:
    p = Path(path)
    codes: list = []
    seen = set()
    if not p.exists():
        return codes
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].split(",")[0].strip()
        if len(token) >= 6 and token[:6].isdigit() and token[:6] not in seen:
            seen.add(token[:6])
            codes.append(token[:6])
    return codes


# ------------------------- 抓取器（akshare） -------------------------
def fetch_announcements(code: str, start_date: str, end_date: str) -> list:
    """个股公告（巨潮 cninfo，支持日期区间，可回溯历史）。"""
    import akshare as ak
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京", start_date=start_date, end_date=end_date)
    except Exception:  # noqa: BLE001
        return []
    items = []
    for _, r in df.iterrows():
        items.append({
            "code": code,
            "publish_time": str(r.get("公告时间", "")),
            "category": "announcement",
            "source": "巨潮cninfo",
            "title": str(r.get("公告标题", "")),
            "summary": "",
            "url": str(r.get("公告链接", "")),
        })
    return items


def fetch_research(code: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list:
    """个股研报（东财，含发布日期，历史较深）；按窗口过滤。"""
    import akshare as ak
    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception:  # noqa: BLE001
        return []
    items = []
    for _, r in df.iterrows():
        d = pd.to_datetime(r.get("日期"), errors="coerce")
        if pd.isna(d) or d < start_ts or d > end_ts:
            continue
        org = str(r.get("机构", "")).strip()
        rating = str(r.get("东财评级", "")).strip()
        items.append({
            "code": code,
            "publish_time": d.strftime("%Y-%m-%d"),
            "category": "research",
            "source": "东财研报",
            "title": str(r.get("报告名称", "")),
            "summary": " ".join(x for x in (org, rating) if x),
            "url": str(r.get("报告PDF链接", "")),
        })
    return items


def fetch_stock_news(code: str) -> list:
    """个股新闻（东财 stock_news_em，仅近端约 10 条；尽力而为，个别 code 偶发失败会被吞掉）。"""
    items = []
    try:
        for it in news.fetch_stock_news(code, limit=30):
            items.append({
                "code": code,
                "publish_time": it.time,
                "category": "stock_news",
                "source": it.source or "东财",
                "title": it.title,
                "summary": it.summary,
                "url": it.url,
            })
    except Exception:  # noqa: BLE001
        pass
    return items


def fetch_market_flash(limit: int = 80) -> list:
    """市场级要闻/快讯（新浪快讯 + 财经早餐，仅近端）。"""
    items = []
    try:
        for it in news.fetch_market_news(limit=limit):
            items.append({
                "code": "",
                "publish_time": it.time,
                "category": "flash",
                "source": it.source,
                "title": it.title,
                "summary": it.summary,
                "url": it.url,
            })
    except Exception:  # noqa: BLE001
        pass
    return items


# ------------------------- 编排 -------------------------
def _ingest_code(code: str, start_date: str, end_date: str,
                 start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                 with_news: bool) -> dict:
    items: list = []
    items += fetch_announcements(code, start_date, end_date)
    items += fetch_research(code, start_ts, end_ts)
    if with_news:
        items += fetch_stock_news(code)
    added, total = news_store.save_items(code, items)
    return {"code": code, "fetched": len(items), "added": added, "total": total}


def run(watchlist: str, months: int, lookback_days: int, mode: str,
        workers: int, limit: int, with_news: bool) -> None:
    codes = read_watchlist(watchlist)
    if limit > 0:
        codes = codes[:limit]
    if not codes:
        print(f"[news] 白名单为空：{watchlist}", flush=True)
        return

    end_ts = pd.Timestamp.now().normalize()
    if mode == "backfill":
        start_ts = end_ts - pd.DateOffset(months=months)
    else:
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
    start_date, end_date = start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d")

    print(f"[news] mode={mode} codes={len(codes)} window={start_ts.date()}..{end_ts.date()} "
          f"workers={workers} with_news={with_news}", flush=True)

    t0 = time.time()
    done = tot_added = tot_fetched = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        futs = {ex.submit(_ingest_code, c, start_date, end_date, start_ts, end_ts, with_news): c
                for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                res = fut.result()
                tot_added += res["added"]
                tot_fetched += res["fetched"]
            except Exception as e:  # noqa: BLE001
                print(f"[news] {code} 失败: {e}", flush=True)
            done += 1
            if done == 1 or done % 25 == 0 or done == len(codes):
                print(f"[news] {done}/{len(codes)} 累计新增={tot_added} "
                      f"抓取={tot_fetched} 用时={time.time()-t0:.0f}s", flush=True)

    # 市场要闻快讯（无历史，仅当前批次）。
    fadd, ftot = news_store.save_items("", fetch_market_flash())
    print(f"[news] 市场要闻快讯 新增={fadd} 总={ftot}", flush=True)

    s = news_store.stats()
    print(f"[news] 完成：库内个股={s['codes']} 总条数={s['total']} "
          f"时间跨度={s['date_min']}..{s['date_max']} 分类={s['by_category']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="白名单新闻入库：回填 + 每日增量")
    ap.add_argument("--watchlist", default=_DEFAULT_WATCHLIST)
    ap.add_argument("--mode", choices=["backfill", "daily"], default="backfill")
    ap.add_argument("--months", type=int, default=6, help="回填窗口（月），mode=backfill 生效")
    ap.add_argument("--lookback-days", type=int, default=7, help="增量回看天数，mode=daily 生效")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 只（调试用，0=全部）")
    ap.add_argument("--no-stock-news", action="store_true",
                    help="跳过东财个股新闻抓取（默认抓取；该接口偶发不稳定，异常已吞掉不影响整体）")
    args = ap.parse_args()
    run(watchlist=args.watchlist, months=args.months, lookback_days=args.lookback_days,
        mode=args.mode, workers=args.workers, limit=args.limit,
        with_news=not args.no_stock_news)


if __name__ == "__main__":
    main()
