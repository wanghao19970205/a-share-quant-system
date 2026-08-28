"""量化选股 · 配置：数据仓路径与股票池定义。

数据仓默认在项目根目录 ``quant_data/``，可用环境变量 QUANT_DATA_DIR 覆盖。
"""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_DIR = os.environ.get("QUANT_DATA_DIR", os.path.join(_ROOT, "quant_data"))
TRADING_CALENDAR_FILE = os.environ.get(
    "TRADING_CALENDAR_FILE", os.path.join(QUANT_DIR, "trading_calendar.parquet"))
PRICE_DIR = os.path.join(QUANT_DIR, "price")          # 每股一份日线 parquet
VALUATION_DIR = os.path.join(QUANT_DIR, "valuation")  # 每股一份估值时序 parquet
MAINBOARD_UNIVERSE_FILE = os.path.join(QUANT_DIR, "mainboard_active_universe.txt")

# 股票池定义：kind 决定取成分的方式
#   "all"          全市场 A 股（stock_info_a_code_name）
#   "csindex"      中证指数成分（index_stock_cons_csindex），arg 为指数代码
UNIVERSES = {
    "full_a": {"kind": "all", "arg": None, "desc": "全部A股"},
    "mainboard_active": {"kind": "mainboard_active", "arg": None, "desc": "全A沪深主板有效股票"},
    "hs300":  {"kind": "csindex", "arg": "000300", "desc": "沪深300"},
    "zz500":  {"kind": "csindex", "arg": "000905", "desc": "中证500"},
    "zz1000": {"kind": "csindex", "arg": "000852", "desc": "中证1000"},
}


def ensure_dirs() -> None:
    for d in (QUANT_DIR, PRICE_DIR, VALUATION_DIR):
        os.makedirs(d, exist_ok=True)
