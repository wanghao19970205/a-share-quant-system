"""V8 模拟盘：等权低成交额带账户。

机制与 V7 逐字相同（分位带进出 + 等权 + 无其它出场规则），只把指标从 60 日波动率换成
20 日均成交额（``volume × close``），进 ``(0.10, 0.20]``、跌出 ``(0.05, 0.50]`` 才卖。

开这个账户的唯一理由是它和 V7 几乎不重合：每个调仓日候选前 20 的 Jaccard 均值 0.0032
（中位 0），日度超额相关系数 0.174，五五等权合并后 IR 从 1.86/1.78 升到 2.37。研究口径
（2089 个交易日、往返成本 0.004、n=20）年化超额 +17.97%，IS +17.85%（t 4.32）、
OOS +17.23%（t 2.87）。

两点必须记住：

1. **"低成交额"不等于"买不进去"。** 实测带内 20 日均成交额中位 4237 万元、最小
   3455 万元，单笔 5000 元只占 0.012%，99.1% 的标的买得起 1 手。主板活跃池本身已经
   筛掉了真正的僵尸股，所以这一档仍然是能成交的。真正要盯的是实盘买卖价差——这也是
   V8 相对 V7 最可能失效的地方，`buy_last/bid1/ask1` 会把它记下来。
2. **参数稳健性比 V7 弱。** 进带扫描里最强的是最低那一档 `(0.10,0.20]`，而不是中间
   某档（`0.25-0.35` 的留出期 t 只有 1.94），越靠边缘越要担心是选出来的。所以 V8 的
   定位是前向验证，不是已经确认的结论。
"""
from __future__ import annotations

from .v7 import V7PaperTrader


class V8PaperTrader(V7PaperTrader):
    _FILE_SUFFIX = "_v8"
    _VERSION = 8
    _EVENT_TYPE = "paper_buy_decision_v8"
    _EVENT_ID_PREFIX = "paper-buy-decision-v8"
    _POSITION_ID_PREFIX = "paper-pos-v8"
    _PAPER_TITLE = "模拟盘V8"
    _METRIC = "dollar_vol20"
    _CFG_PREFIX = "paper_v8"
    _BAND_DEFAULTS = (0.10, 0.20, 0.05, 0.50)
    _EXIT_TEXT = "跌出成交额分位带"
