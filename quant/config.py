"""量化选股 · 配置：数据仓路径与股票池定义。

数据仓默认在项目根目录 ``quant_data/``，可用环境变量 QUANT_DATA_DIR 覆盖。
"""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configure_quant_dir(path: str | os.PathLike[str]) -> str:
    """Rebind all data paths for a long-lived parent process."""
    global QUANT_DIR, PRICE_DIR, VALUATION_DIR
    global MAINBOARD_UNIVERSE_FILE, TRADING_CALENDAR_FILE, SECURITY_MASTER_FILE
    global INDEX_CONSTITUENT_HISTORY_FILE, TRADING_STATUS_HISTORY_FILE

    quant_dir = os.path.realpath(os.path.expanduser(os.fspath(path)))
    QUANT_DIR = quant_dir
    PRICE_DIR = os.path.join(quant_dir, "price")
    VALUATION_DIR = os.path.join(quant_dir, "valuation")
    MAINBOARD_UNIVERSE_FILE = os.path.join(
        quant_dir, "mainboard_active_universe.txt"
    )
    TRADING_CALENDAR_FILE = os.path.join(quant_dir, "trading_calendar.parquet")
    SECURITY_MASTER_FILE = os.path.join(quant_dir, "security_master.parquet")
    INDEX_CONSTITUENT_HISTORY_FILE = os.path.join(
        quant_dir, "index_constituent_history.parquet"
    )
    TRADING_STATUS_HISTORY_FILE = os.path.join(
        quant_dir, "trading_status_history.parquet"
    )
    return quant_dir


configure_quant_dir(
    os.environ.get("QUANT_DATA_DIR", os.path.join(_ROOT, "quant_data"))
)

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
