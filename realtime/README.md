# Realtime Strategy

本文档说明 `realtime/` 当前生产策略、模拟盘执行口径、持久化数据和运维方式。代码与运行参数冲突时，以 `realtime/config.py` 和远端 `logs/realtime/notify.env` 的实际加载结果为准。

## 1. 目标与边界

实时层负责四件事：

1. 订阅模型候选和持仓的 Level-1 快照。
2. 在模型候选池内进行有界盘中重排，并推送带个股主行业和精简概念标签的 Top5 候选榜。
3. 用纸上账户执行收盘前买入、A 股 T+1 风控和卖出。
4. 记录信号、候选过滤、成交和卖出后反事实，供前向评估。

实时层不重新训练模型，不修改 `quant_data`，也不会把盘中信号当成新的选股 alpha。模型决定候选池，盘中数据只做有限择序、过滤和风险退出。

## 2. 主链路

```text
09:20 cron / watchdog
  -> python -m realtime.engine
  -> watchlist.load_codes
  -> reference.build
  -> feed Level-1 snapshots
  -> StrategyContext
       -> default_strategies -> signals ledger / notifier
       -> RerankScorer       -> RankBoard Top5
                            -> PaperTrader Top2
```

引擎自管交易时段：

- 09:25 前等待；
- 11:32-13:00 静默；
- 15:05 后退出；
- 交易时段每 10 分钟由 cron watchdog 检查，进程存在则跳过，进程消失则重拉。

## 3. 订阅清单

订阅优先级如下：

1. 全部模拟盘账户持仓（`paper_state.json` 与启用时的 `_v2` 到 `_v5` 独立状态）；
2. `realtime_holdings.txt` 中的人工持仓；
3. `mobile_snapshot.json` 中固定候选组（白名单 Top10、全A Top30、创新药 Top10）；
4. 固定候选快照缺失时使用最新预测文件；
5. 全部缺失时使用兜底股票池。

模拟盘和人工持仓属于保护性订阅，优先于候选清单。即使保护性持仓数量超过 `REALTIME_MAX_SUBSCRIBE`，也不能因截断而失去报价和卖出能力。

引擎监听固定候选快照、人工持仓和模拟盘状态文件的 mtime。代码集合发生变化时使用 `execv` 自我重启，重新建立订阅。

## 4. 预期收益口径

- `expected_return`：`30% Ridge + 20% ElasticNet + 50% ExtraTrees` 的同量纲融合收益率，用于成本安全边际门和收益展示；缺模型腿时按可用权重归一化。
- `return_components`：三条收益模型的原始值、实际归一化权重和降级来源，写入买入审计。
- `calibrated_return`：历史同融合收益分档的实际毛收益均值，只用于展示和审计，不再直接作为买入硬门。
- `calibrated_net_return`：按模拟盘买卖各半成本模型扣除 round-trip 成本后的净收益，仅作历史兑现参考。
- `pred`：包含 LightGBM、IC、ElasticNet、ExtraTrees 的现役无量纲融合分，转换为当日全A百分位后作为实时排序主序；不得当作收益率展示或用于成本判断。

通知展示格式为：

```text
模型净+0.30%(原始毛+0.50%;历史校准+0.58% 胜率52%)
```

买入硬门使用三模型融合收益覆盖成本和安全边际；缺少历史校准时只是不显示历史校准参考。

## 5. 候选池与盘中重排

### 5.1 模型门槛

候选必须依次满足：

1. 不在排除集合中，例如当前已持仓；
2. 融合 `expected_return >= max(REALTIME_RANK_MIN_RAW_RETURN, paper_cost + REALTIME_RANK_RAW_SAFETY_MARGIN)`；
3. 历史 `calibrated_net_return` 只保留用于展示和审计；
4. 按融合 `pred` 的当日全A百分位进入模型 Top `REALTIME_RANK_POOL_N`；缺 `pred` 时退回 Ridge 主序；
5. 买入腿要求有实时价且未封涨停。

默认值：

```text
REALTIME_ENSEMBLE_RETURN=true
REALTIME_ENSEMBLE_RIDGE_WEIGHT=0.30
REALTIME_ENSEMBLE_ELASTIC_WEIGHT=0.20
REALTIME_ENSEMBLE_EXTRA_TREES_WEIGHT=0.50
REALTIME_RANK_POOL_N=30
REALTIME_RANK_MIN_RAW_RETURN=0
REALTIME_RANK_RAW_SAFETY_MARGIN=0.001
REALTIME_PAPER_COST=0.002
```

### 5.2 重排公式

```text
model_anchor = percentile_rank(pred)       # 当日全A，范围 0-1
score = model_anchor + clamp(adj, -cap, +cap) * rank_scale
```

旧预测文件缺少 `pred` 时降级为：

```text
score = expected_return * (1 + clamp(adj, -cap, +cap))
```

默认参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `REALTIME_RERANK_CAP` | 0.30 | 盘中调整值限制为正负 0.30 |
| `REALTIME_RERANK_INTRADAY_RANK_SCALE` | 0.10 | 将盘中调整换算成最多约 3 个百分位点的主序位移 |
| `REALTIME_RERANK_W_VWAP` | 0.35 | 低于 VWAP 加分，高于 VWAP 减分 |
| `REALTIME_RERANK_W_IMB` | 0.30 | 买盘强加分，卖盘强减分 |
| `REALTIME_RERANK_W_SPREAD` | 0.15 | 盘口窄加分，盘口宽减分 |
| `REALTIME_RERANK_W_GAP` | 0.00 | 高开惩罚关闭 |

高开惩罚默认关闭。离线数据表明高开是正向隔夜动量，不能再按“高开吃掉预期”做负向惩罚。

重排只在模型池内改变相近候选的顺序，不能把池外股票拉进来。

## 6. 模拟盘买入

V1-V5 统一买入窗口为 14:50-14:55（闭区间）：14:50 执行 primary，只有未成交账户在 14:53 执行 retry，每个阶段最多执行一次。

执行顺序：

1. 14:50 后先处理旧仓风险和到期卖出；
2. 同一轮卖出腿完成后，在 primary/retry 阶段读取当日模型候选；
3. 按重排后的 `score` 降序检查候选；
4. 买入通过过滤的前 2 只；
5. 预算取剩余名额均分、ATR 风险额度和单票净值上限的最小值，股数向下取整到 100 股；
6. 买入成本按 round-trip 成本的一半计入。

默认买前过滤：

| 条件 | 默认行为 |
|---|---|
| 高开吃预期 | 关闭 |
| 当前价高于 VWAP 1% | 跳过 |
| `bid_ask_imbalance <= -0.2` | 跳过 |
| `spread_pct >= 0.6%` | 跳过 |
| 封涨停 | 跳过 |

如果候选被过滤，会继续检查下一名。若旧仓在 14:50 卖出后，同一股票仍是当轮有效 Top2，可以重新买入，相当于用新预测决定是否续持。

## 7. 模拟盘卖出

A 股 T+1 是总前提：买入当日不可卖出。到 T+1 后按以下优先级评估，每次只采用第一个命中的原因：

1. 硬止损：`ret <= -5%`；
2. 硬止盈：`ret >= +9%`；
3. ATR 移动止盈：`last <= peak - 3 * ATR`，且持仓曾有浮盈；
4. VWAP 破位：`last < VWAP * (1 - 2%)`；
5. 时间上限：持有达到 `T+sell_horizon`。

前四项风险退出在 T+1 全天有效。纯 `time_cap` 只在 14:50 后执行，使模拟盘与模型的 close-to-close 训练口径一致，避免在 T+1 开盘把策略错误执行成 close-to-next-open。

默认参数：

```text
REALTIME_SELL_HORIZON=1
REALTIME_PAPER_TIME_CAP_START=1450
REALTIME_PAPER_BUY_START=1450
REALTIME_PAPER_BUY_END=1455
REALTIME_PAPER_STOP_LOSS=0.05
REALTIME_PAPER_TAKE_PROFIT=0.09
REALTIME_PAPER_TRAIL_K=3.0
REALTIME_PAPER_VWAP_BREAK=0.02
```

## 8. 默认信号策略

`default_strategies()` 当前装配 6 条信号策略：

- `limit_move_watch`：涨跌停和开板；
- `surge_watch`：快速拉升或跳水；
- `volume_surge`：成交量异动；
- `vwap_deviation`：相对 VWAP 的偏离；
- `chandelier_stop`：ATR 吊灯止损信号；
- `holding_expiry`：持有到期提示。

`GapCalibrate` 类仍保留用于显式实验，但不进入默认装配。

信号策略不直接替代 `PaperTrader` 的成交判断。生产默认 `REALTIME_SIGNAL_PUSH=false`：六类通用信号和行业 ETF 状态只写账本、不发 Push；仅保留 RankBoard 的 Top 榜和 V1-V5 模拟盘买卖 Push。需要临时恢复通用信号提醒时，显式设置 `REALTIME_SIGNAL_PUSH=true`，再由 `REALTIME_NOTIFY_KINDS` 控制白名单。

### V1-V5 赛马总览

| 版本 | 买卖成交价 | 入场增强 | 出场与风险 | 独立文件 |
|---|---|---|---|---|
| **V1** | `last / last` | 模型池 + 盘中重排 + 风险预算 | 固定止损止盈、ATR吊灯、VWAP破位、T+N | 无后缀 |
| **V2** | `last / last` | V1 + 持仓上限 | 保护止盈、炸板、跌停顺延、时间加权止盈 | `_v2` |
| **V3** | `ask1 / bid1` | 当日预测、新鲜度、盘口确认 + 风险预算 | 入场锁定ATR、硬止损/止盈/移动止盈、T+N | `_v3` |
| **V4** | `ask1 / bid1` | V3 + 精确主行业ETF弱势与集中度控制 | 完整继承V3，暂不叠加板块卖出 | `_v4` |
| **V5** | `ask1 / bid1` | V4 + 行业相对强度动态仓位 | 完整继承V4，只改变风险预算 | `_v5` |
| **V7** | `ask1 / bid1` | 不看模型：60日波动率分位带 `(0.30,0.40]` + 等权 | 只有跌出 `(0.20,0.70]` 才卖，无止损/止盈/ATR/T+N | `_v7` |

五个账户共享模型候选池、交易成本和二阶段买入：14:50 执行 `primary`，仅未成交账户在 14:53 执行 `retry`，14:55 后关闭；阶段状态跨重启持久化且分别形成审计。仓位取剩余名额均分、ATR 单笔风险额度和单票净值上限的最小值，融合预期收益只做 0.75-1.25 倍有限调整，缺 ATR 时退回固定止损口径。任一版本装配或交易异常均独立降级，不影响其他版本。

## 8.1 V2 赛马账户

`realtime/v2.py` 的 `V2PaperTrader` 与 V1 在同一引擎内并行，共享 ctx、模型候选池和通知通道，但账户状态、流水和审计写入 `_v2` 独立副本。V1 现役策略不受任何影响。

V2 与 V1 的差异仅在执行端：

- 与其他版本统一使用 `14:50 primary / 14:53 retry / 14:55 close` 二阶段买入；
- 资金按 ATR 风险、融合预期收益、单票上限和剩余名额动态分配，高价股买不起时顺延下一名；
- 目标只数受 `paper_max_positions` 收缩，风险额度之外的现金允许保留；
- 浮盈达 `paper_breakeven_arm` 后回落至成本附近触发 `breakeven_stop`；
- 持有多日后按 `paper_take_profit_tighten` 收窄止盈阈值；
- 曾封涨停后开板且有浮盈触发 `limit_open`；
- 封跌停时按交易日顺延，最多 `paper_limit_down_roll_max` 日后强制平仓。

模型口径、`close→close` 到期腿、`±30%` 重排上限和成本假设与 V1 完全一致。

关键参数：

```text
REALTIME_PAPER_V2=true
REALTIME_PAPER_BUY_RETRY_START=1453
REALTIME_PAPER_BUY_END=1455
REALTIME_PAPER_MAX_POSITIONS=4
REALTIME_PAPER_RISK_PER_TRADE=0.015
REALTIME_PAPER_MAX_POSITION_WEIGHT=0.40
REALTIME_PAPER_ALLOCATION_ATR_K=2.0
REALTIME_PAPER_ALLOCATION_TARGET_RETURN=0.02
REALTIME_PAPER_SECTOR_MAX_POSITIONS=2
REALTIME_PAPER_BREAKEVEN_ARM=0.03
REALTIME_PAPER_BREAKEVEN_MARGIN=0.005
REALTIME_PAPER_TAKE_PROFIT_TIGHTEN=0.03
REALTIME_PAPER_LIMIT_DOWN_ROLL_MAX=3
```

V1/V2 的持仓都进入保护性订阅。V2 装配异常会降级跳过并保留 V1 运行。

## 8.2 V3 赛马账户

V3 与 V1/V2 并行运行，使用独立的 `_v3` 状态、流水、买入决策和卖出反事实文件。候选池、初始资金、交易成本、动态资金分配、买入窗口、持仓上限和 T+1 口径与 V2 一致，只测试两组可归因变化。

买入规则：

1. 最新预测日期必须是当日，旧预测只允许管理已有持仓，不建立新仓。
2. 本进程收到行情快照的时间不超过 `paper_v3_quote_max_age_sec`。
3. 买一、卖一和一档挂单量必须有效，买一挂单量不低于卖一挂单量。
4. 保留 V2 的 VWAP 偏贵和宽价差过滤，命中后向下顺延候选。
5. 按卖一价 `ask1` 买入，不再按最新成交价 `last` 假设成交。

卖出规则：

1. T+0 始终禁止卖出。
2. 无有效买一、买一量为零或行情过期时保留持仓并记录阻塞；跌停价仍有买一承接时允许按 `bid1` 卖出。
3. 按买入时锁定的 ATR 风险单位退出：跌破入场价 `2ATR` 止损；上涨 `4ATR` 止盈；盈利达到 `2ATR` 后从最高有效买一价回撤 `2ATR` 移动止盈。
4. ATR 缺失时只保留 T+N 到期退出，不混入固定百分比阈值。
5. 所有卖出按买一价 `bid1` 结算。

关键参数：

```text
REALTIME_PAPER_V3=true
REALTIME_PAPER_V3_QUOTE_MAX_AGE_SEC=300
REALTIME_PAPER_V3_ATR_K=2.0
```

V3 使用更保守的可成交报价，绝对净值可能低于按 `last` 结算的 V1/V2。赛马应联合比较成本后收益、最大回撤、可成交率、卖出阻塞、滑点和退出后反事实，不能只按净值高低晋级。

## 8.3 V4 赛马账户

V4 完整继承 V3 的候选池、当日预测、盘口确认、`ask1/bid1` 成交、T+1 和 ATR 出场，使用独立 `_v4` 文件。第一版只增加一个策略变量：读取本地 `all_a_stock_meta.parquet` 的申万三级主行业 `a_industry`，按版本化精确表映射行业 ETF，并用其相对沪深300 ETF 的当日收益确认入场环境。`a_industries` 只进审计，不参与匹配；概念和模糊关键词均不参与。

- 映射必须为 `exact_primary` 且置信度不低于 `0.8`，同时 `行业ETF收益 - 沪深300ETF收益 <= -0.3%` 时才判为弱势并拒绝候选；
- 强势和中性均按 V3 原规则处理，不加仓、不改排序、不改卖点；
- 个股未映射、ETF或基准行情缺失/超过90秒时标记 `unavailable` 并放行，数据故障不当作利空；
- ETF 与个股使用同一个 `SubscribeData` 登录会话，完整保留 `.SH/.SZ` 后缀；ETF回调只进入板块上下文，不进入个股策略；
- ETF强弱状态切换写入 `signals_YYYYMMDD.jsonl`，V4买入审计记录映射版本、来源、置信度、主行业、完整行业层级、ETF、基准、超额收益、行情年龄和过滤原因。

默认精确覆盖半导体、证券、银行、医药、军工、锂电/新能源车、光伏、有色、食品饮料、计算机和通信的指定申万三级行业。可用 `REALTIME_SECTOR_ETFS` 按 `名称=完整ETF代码:精确主行业1|精确主行业2;...` 覆盖；同一主行业出现多次时冲突项会被拒绝，防止配置顺序决定归属。关键参数：

```text
REALTIME_PAPER_V4=true
REALTIME_SECTOR_ETF=true
REALTIME_SECTOR_ETF_BENCHMARK=510300.SH
REALTIME_SECTOR_ETF_QUOTE_MAX_AGE_SEC=90
REALTIME_PAPER_V4_SECTOR_WEAK_EXCESS=-0.003
REALTIME_PAPER_V4_SECTOR_STRONG_EXCESS=0.003
REALTIME_PAPER_V4_SECTOR_MAPPING_MIN_CONFIDENCE=0.8
```

V1/V2/V3/V4 的持仓都由 `config.paper_state_files()` 纳入保护性订阅和热重载。V3 对 V4 的主要差异为“是否回避相对弱势行业”，赛马时应重点比较 V4 过滤候选的后续收益反事实、交易覆盖率和回撤。

## 8.4 V5 赛马账户

V5 完整继承 V4 的候选、弱势拒绝、同行业上限、二阶段买入、`ask1/bid1` 成交和 ATR 退出，只改变一个变量：行业 ETF 相对沪深300的强度对风险额度的缩放。

```text
相对收益 <= -0.30%：继承 V4，拒绝买入
-0.30% < 相对收益 < 0%：风险额度 x 0.85
0% <= 相对收益 < +0.30%：风险额度 x 1.00
相对收益 >= +0.30%：风险额度 x 1.15
低置信度、无映射或 ETF 行情不可用：风险额度 x 1.00
```

行业系数只作用于 ATR 风险额度，最终预算仍重新受现金、剩余名额均分和单票净值上限约束，因此 `1.15x` 不会突破 40% 单票硬上限。买入持仓和决策审计记录 `sector_allocation`，卖出流水保留 `sector_allocation_at_entry`，用于与 V4 做同候选、同买卖点的仓位归因。

```text
REALTIME_PAPER_V5=true
REALTIME_PAPER_V5_SECTOR_LAGGING_FACTOR=0.85
REALTIME_PAPER_V5_SECTOR_NEUTRAL_FACTOR=1.00
REALTIME_PAPER_V5_SECTOR_STRONG_FACTOR=1.15
```

V1-V5 的持仓都由 `config.paper_state_files()` 纳入保护性订阅和热重载。V4/V5 赛马应重点比较成本后收益、最大回撤，以及 strong/neutral/lagging 三组实际兑现收益，不能仅按绝对净值晋级。

## 8.5 V7 赛马账户（等权低波动带，不用模型）

`realtime/v7.py` 的 `V7PaperTrader` 是与 V1-V6 取向相反的对照组：不读模型预测，只按 `realtime/vol_band.py` 算出的 60 日波动率日度截面分位进出，等权持有 20 只，除分位带外没有任何出场规则。研究口径（2057 个交易日、往返成本 0.004）对全可买等权基准年化超额 +16.07%（t=10.86），留出期 +15.08%（t=6.50），日均换手 0.033。

```text
进场：分位落在 (0.30, 0.40]，按分位升序取，等权（净值/20 每只，整手向下取整）
出场：分位跌出 (0.20, 0.70] → vol_band_exit；分位取不到时继续持有
不叠加：硬止损、止盈、ATR 出场、VWAP 破位、T+N 到期——任何额外出场都会把换手推高，
        那就不是被验证过的那条策略了
```

```text
REALTIME_PAPER_V7=true
REALTIME_PAPER_V7_POSITIONS=20
REALTIME_PAPER_V7_ENTRY_LO=0.30
REALTIME_PAPER_V7_ENTRY_HI=0.40
REALTIME_PAPER_V7_EXIT_LO=0.20
REALTIME_PAPER_V7_EXIT_HI=0.70
REALTIME_PAPER_V7_SUBSCRIBE_N=40
```

订阅是 V7 唯一影响其他账户的地方：带内约占全池 10%（实测 2026-09-03 全池 3194 只、有效截面 3191 只、带内 319 只），不可能全部订阅。`watchlist._read_vol_band()` 只取带内分位最低的 `REALTIME_PAPER_V7_SUBSCRIBE_N` 只（与研究的选股规则一致，不引入额外选择偏差，实测 40 只覆盖分位 0.3002-0.3124），并使用 `max_subscribe` 之外的独立配额，因此 V1-V6 的候选名额一只不减，代价是行情订阅总数增加约 40 只——2 MB 带宽上限下需要在首日核对 `recv` 与报价新鲜度。持仓照旧由 `config.paper_state_files()` 纳入保护性订阅。

全池分位要读 3000 多个 parquet 尾部，实测 29 秒。`vol_band.rank_pct(cache_dir=...)` 在 `logs/realtime/vol_rank_YYYYMMDD_60.json` 落一份当日快照并只保留最近 5 天：进程内缓存救不了盘中 `execv` 重启，而重启可能正好落在 14:50-14:55 买窗里，把所有账户的建仓机会一起拖过去。

口径上有一天的滞后：实盘 14:50 建仓时价格文件只到 T-1 收盘，研究算到 T 收盘。实测带内成员 319 vs 319、交集 284、Jaccard 0.8023，全池分位 Spearman 0.998652，但最大绝对差 0.1377——分位在带下沿很密，"带内最低 40 只"两种口径只重合 16/40。这不改变预期收益：同一条带下 `target_n` 从 20 扫到 319，年化超额 15.89% / 17.52% / 15.33% / 16.07% / 14.53%，留出期 t 全在 3.6 以上，说明超额来自"进了这条带"而不是带内的细微排序。选 20 只的代价是风险调整而非边缘：IR 1.86（n=50 为 2.86），最大回撤 -31.9%。

## 9. 持久化与决策审计

所有文件位于 `logs/realtime/` 挂载盘，容器重建后保留：

| 文件 | 内容 |
|---|---|
| `paper_state.json` | 现金、持仓、持仓峰值、最近买入日 |
| `paper_position_snapshots.jsonl` | 按心跳记录每只持仓的成本、估值价、行情年龄、浮动盈亏和收益率 |
| `paper_trades.jsonl` | 每笔平仓的不可变流水，包含成本、卖出净额和盈亏 |
| `paper_buy_decisions.jsonl` | 每日模型池、重排、过滤和成交快照 |
| `paper_sell_counterfactuals.jsonl` | 卖出日及后续 3 个交易日反事实 |
| `paper_*_v2.json(l)` | V2 赛马账户的同名独立副本 |
| `paper_*_v3.json(l)` | V3 可成交报价与 ATR 规则账户的同名独立副本 |
| `paper_*_v4.json(l)` | V4 行业ETF弱势回避账户的同名独立副本 |
| `paper_*_v5.json(l)` | V5 行业相对强度动态仓位账户的同名独立副本 |
| `paper_*_v7.json(l)` | V7 等权低波动带账户的同名独立副本 |
| `signals_YYYYMMDD.jsonl` | 实时信号账本（含ETF强弱状态切换） |
| `engine.YYYYMMDD.log` | 引擎运行日志 |

买入决策审计记录：

- 模型池排除原因；
- 三模型融合收益、各腿原始值、归一化权重、校准毛收益和校准净收益；
- 无量纲融合 `pred` 原始分、当日全A百分位；
- rank、score、adj 和分项贡献；
- 实时价、开盘价、昨收、VWAP、盘口失衡、价差和涨停状态；
- 入场过滤、未进入 Top2、资金不足或实际成交；
- `primary/retry` 阶段、ATR 风险距离、信号系数、预算上限与集中度原因；
- 成交股数、价格、成本和账户前后状态。

### 零成交也是有效样本

买入阶段正常执行但没有成交时，各版本仍会按 `trade_date + attempt_stage` 写入唯一决策事件，因此同日最多有 primary/retry 两条。`decision_status` 用于区分：

| 状态 | 含义 |
|---|---|
| `bought` / `partial_fill` | 完成全部或部分目标建仓 |
| `all_candidates_filtered` | 有可排名候选，但全部被盘口、预测或板块规则过滤 |
| `no_ranked_candidate` | 模型池、涨停或成本后净收益门之后没有候选 |
| `insufficient_cash_or_lot` | 候选有效，但现金或整手约束无法成交 |

因此赛马的有效样本包括“成交结果”和“过滤反事实”两部分。若零成交日没有对应版本的 `paper_buy_decisions*.jsonl` 事件，才应判断为运行或审计链路异常。

卖出反事实按稳定 `trade_id` 幂等更新，记录 sell-day、D+1、D+2、D+3 的收盘价、最高价、继续持有收益和机会损益。日线尚未到达时保留空 markout，后续引擎启动自动补齐。

审计写入使用文件锁、临时文件、`fsync` 和原子替换。审计失败只写日志，不阻断交易。

策略信号的 HTTP 通知和 `signals_YYYYMMDD.jsonl` 记账由单消费者按 FIFO 顺序异步执行，行情回调只做状态更新、策略计算和非阻塞入队。默认队列容量为 1000，退出时最多等待 3 秒排空：

```text
REALTIME_EFFECT_QUEUE_SIZE=1000
REALTIME_EFFECT_SHUTDOWN_GRACE_SEC=3
```

心跳中的 `effects_pending`、`effects_done`、`effects_dropped` 分别表示待处理、已处理和因队列满而丢弃的副作用任务。`effects_dropped` 非零通常表示通知服务持续超时，需要先检查推送通道，再决定是否调整队列容量。

### Realtime 权重自动例行任务

工作日 18:30 由独立 `realtime-weight-shadow` 任务读取包含 `target_ret_1d` 的历史预测，先在非负且和为1的单纯形上做投影梯度约束堆叠，再用24个候选、24代精英保留/交叉/变异做确定性进化搜索；不是固定网格枚举。互相独立的滚动调仓窗口默认由4个 worker 并发。只有存在有限兑现标签和至少一条有效模型预测的日期才进入训练/调仓日历；某日只有在其权重完全由此前训练窗口产生时才计为样本外交易日，一天数千只股票仍只算一个日期。至少 40 个样本外交易日才允许进入晋级判断。

当前优化目标是全截面截断 MSE，自动晋级指标是样本外 Top10 等权组合收益、Sharpe、胜率和回撤，两者口径差异会写入 manifest 诊断。该诊断用于识别全押单模型等目标错位，不会放宽晋级门槛；改为组合/排序目标前必须作为独立 Challenger 积累 OOS 证据。

Challenger 必须同时通过成本后平均收益、Sharpe、胜率、最大回撤、单次权重跳变和晋级冷静期门槛。通过后原子写入 `logs/realtime/weight_shadow/active_weights.json`，V1-V5 下一次引擎启动共同读取；缺失、损坏、负数或非有限权重会安全回退 `30%/20%/50%` 环境配置。晋级后新增至少 10 个前向交易日持续跑输上一 Champion 时自动回滚。

评估版本、当前报告和晋级历史分别写入 `versions/<version>.json`、`manifest.json` 和 `promotion_history.jsonl`。并发数可用 `REALTIME_WEIGHT_SHADOW_WORKERS` 调整，生产夹紧在1-4，默认4。该任务不写 `active_quant_model.json`，不修改 `notify.env`，不调用模型训练/发布，也不被 daily/intraday/weekly/monthly 模型任务读取。

## 10. 运行与验证

在远端宿主机 `/www/A` 执行：

```bash
# 查看服务和实时进程
docker compose ps scheduler
docker top a-scheduler-1 -eo pid,lstart,cmd

# 幂等拉起实时引擎
docker compose exec -T scheduler /app/docker/scheduler_jobs.sh realtime

# 策略自检
docker compose exec -T scheduler python -m realtime.selftest

# 虚拟数据流回归
bash run_sim_streams.sh

# 手工补跑自动评估/晋级任务（生产已由工作日 18:30 cron 例行触发）
docker compose exec -T scheduler /app/docker/scheduler_jobs.sh realtime-weight-shadow
# 仅评估、不晋级
docker compose exec -T scheduler python -m realtime.weight_shadow --evaluate-only

# 查看运行时关键参数（只打印非密钥字段）
docker compose exec -T scheduler python -c "from realtime.config import load; c=load(); print(c.paper_time_cap_start, c.paper_buy_start, c.sell_horizon, c.paper_cost)"
```

### 每日盘后有效性核对

| 检查项 | 正常标准 | 异常时的含义 |
|---|---|---|
| 当日预测 | 14:50前预测最大日期等于交易日 | V3-V5会记录 `预测非当日` 并拒绝新仓 |
| 实时行情 | 14:50附近心跳 `active=True` 且 `recv` 持续增长 | 买入窗口可能未获得有效盘口 |
| 五版决策 | 每版有 primary；未成交版本另有 retry 事件 | 缺失阶段表示买入腿未执行或审计写入失败 |
| ETF行情 | `sector_recv > 0`，ETF强/中/弱分布可见 | V4放行不可用信号，V5按1.00x安全降级 |
| 异步副作用 | `effects_dropped=0` | 非零表示通知/账本队列持续拥塞 |

`paper_state_v3.json`、`paper_state_v4.json` 和 `paper_state_v5.json` 在首次买入决策保存时才创建；仅在部署后、买入窗口前不存在不代表故障。排查零成交优先运行 `./restart.sh diag`，再查看对应版本决策事件，不应只检查成交流水。

当前虚拟数据流回归为 **141 PASS / 0 FAIL**，覆盖模型池封闭、重排有界、要求有效实时价且排除封涨停票的行业/概念 Top5 Push、绕过行情 SDK stop 的安全热重载、A 股 T+1、五级卖出、保护性订阅、融合预期成本安全边际门、二阶段买入及跨重启恢复、决策 trace 和反事实幂等补齐；V1-V5 的 ATR 风险预算、融合收益有限加权和整手约束；V4 行业集中度；V5 strong/neutral/lagging 动态仓位、缺行情降级、独立账户和保护性订阅；四版基线同场买卖完整生命周期；NaN/Inf、全模型异常、无效盘口、损坏状态文件故障注入；以及 realtime-only 滚动约束、未兑现日期排除、进化权重目标诊断、40 日 OOS、4 worker 确定性、自动晋级、损坏回退、幂等执行和 10 日前向回滚；V7 的带内升序建仓、等权整手、跌出出带才卖（含"分位取不到即继续持有"）、T+1、独立账户文件和独立订阅配额。

## 11. 部署纪律

1. 先修改本地源文件并完成测试；
2. 使用保留相对路径的增量同步将源文件送到远端 `/www/A`；
3. 在远端执行 `git diff --check`，确认只包含本次改动；
4. 使用 `docker compose build scheduler`；
5. 使用 `docker compose up -d --no-deps scheduler` 替换服务；
6. 手动幂等拉起 realtime；
7. 运行 selftest、虚拟数据流和源码哈希核对。

不要只修改容器内 `/app`，不要只把代码放进部署脚本 heredoc，也不要用旧版全量内嵌文件覆盖现有源文件。

## 12. 当前限制

- 模拟盘历史样本仍少，不能据此精确拟合止损、止盈、ATR 或 VWAP 阈值；
- VWAP 位置已有离线正向证据，L1 imbalance 和 spread 仍需依靠买入决策审计做前向评估；
- 模拟盘使用实时快照近似成交，未建模完整盘口冲击、排队和实际成交概率；V3 只确认一档报价与挂单方向，SDK 挂单量单位核实前不把一档容量当作完整成交深度；
- V3-V5 新鲜度表示本进程最近收到回调的时间，`trade_time` 格式和时区经真实回调确认前，不能识别行情源重放旧时间戳的情况；
- V4/V5 的 ETF 是行业价格代理，不等于完整成分股广度；ETF折溢价、资金申赎和行业映射覆盖会产生跟踪误差，必须结合过滤与仓位反事实前向评估；
- 日线反事实只能评估收盘和日内最高价，不能重建分钟级最优退出路径；
- `notify.env` 含推送和行情凭证，只能存放在挂载盘，禁止提交到 git。
