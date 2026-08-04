# Realtime Strategy

本文档说明 `realtime/` 当前生产策略、模拟盘执行口径、持久化数据和运维方式。代码与运行参数冲突时，以 `realtime/config.py` 和远端 `logs/realtime/notify.env` 的实际加载结果为准。

## 1. 目标与边界

实时层负责四件事：

1. 订阅模型候选和持仓的 Level-1 快照。
2. 在模型候选池内进行有界盘中重排，并推送 Top5 候选榜。
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

1. 全部模拟盘账户持仓（`paper_state.json` 与启用时的 `paper_state_v2.json`）；
2. `realtime_holdings.txt` 中的人工持仓；
3. `mobile_snapshot.json` 中手机端 Top10 组；
4. Top10 缺失时使用最新预测文件；
5. 全部缺失时使用兜底股票池。

模拟盘和人工持仓属于保护性订阅，优先于候选清单。即使保护性持仓数量超过 `REALTIME_MAX_SUBSCRIBE`，也不能因截断而失去报价和卖出能力。

引擎监听 Top10、人工持仓和模拟盘状态文件的 mtime。代码集合发生变化时使用 `execv` 自我重启，重新建立订阅。

## 4. 预期收益口径

- `expected_return`：Ridge 的原始 `ridge_pred`，单位为小数收益率，用于候选池和排序。
- `calibrated_return`：历史同预测分档的实际毛收益均值，只用于解释和交易经济性校验。
- `calibrated_net_return`：按模拟盘买卖各半成本模型扣除 round-trip 成本后的净收益。
- `pred`：融合后的无量纲排序分，不得当作收益率展示或用于成本判断。

通知展示格式为：

```text
历史净+0.38%(毛+0.58% 胜率52%)
```

缺少历史校准时回退为模型原始收益扣成本后的净收益。

## 5. 候选池与盘中重排

### 5.1 模型门槛

候选必须依次满足：

1. 不在排除集合中，例如当前已持仓；
2. `expected_return > 0`；
3. `calibrated_net_return > REALTIME_RANK_MIN_NET_RETURN`；
4. 按原始 `expected_return` 进入模型 Top `REALTIME_RANK_POOL_N`；
5. 买入腿要求有实时价且未封涨停。

默认值：

```text
REALTIME_RANK_POOL_N=30
REALTIME_RANK_MIN_NET_RETURN=0
REALTIME_PAPER_COST=0.002
```

### 5.2 重排公式

```text
score = expected_return * (1 + clamp(adj, -cap, +cap))
```

默认参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `REALTIME_RERANK_CAP` | 0.30 | 总调整限制为正负 30% |
| `REALTIME_RERANK_W_VWAP` | 0.35 | 低于 VWAP 加分，高于 VWAP 减分 |
| `REALTIME_RERANK_W_IMB` | 0.30 | 买盘强加分，卖盘强减分 |
| `REALTIME_RERANK_W_SPREAD` | 0.15 | 盘口窄加分，盘口宽减分 |
| `REALTIME_RERANK_W_GAP` | 0.00 | 高开惩罚关闭 |

高开惩罚默认关闭。离线数据表明高开是正向隔夜动量，不能再按“高开吃掉预期”做负向惩罚。

重排只在模型池内改变相近候选的顺序，不能把池外股票拉进来。

## 6. 模拟盘买入

买入窗口为 14:55-15:00，每个交易日只执行一次。

执行顺序：

1. 14:50 后先处理旧仓风险和到期卖出；
2. 14:55 后重新读取当日模型候选；
3. 按重排后的 `score` 降序检查候选；
4. 买入通过过滤的前 2 只；
5. 现金按目标只数等额分配，股数向下取整到 100 股；
6. 买入成本按 round-trip 成本的一半计入。

默认买前过滤：

| 条件 | 默认行为 |
|---|---|
| 高开吃预期 | 关闭 |
| 当前价高于 VWAP 1% | 跳过 |
| `bid_ask_imbalance <= -0.2` | 跳过 |
| `spread_pct >= 0.6%` | 跳过 |
| 封涨停 | 跳过 |

如果候选被过滤，会继续检查下一名。若旧仓在 14:50 卖出后，同一股票仍是 14:55 的有效 Top2，可以重新买入，相当于用新预测决定是否续持。

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
REALTIME_PAPER_BUY_START=1455
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

信号策略负责账本和通知，不直接替代 `PaperTrader` 的成交判断。

## 8.1 V2 赛马账户

`realtime/v2.py` 的 `V2PaperTrader` 与 V1 在同一引擎内并行，共享 ctx、模型候选池和通知通道，但账户状态、流水和审计写入 `_v2` 独立副本。V1 现役策略不受任何影响。

V2 与 V1 的差异仅在执行端：

- 买入窗收窄到 `14:55-14:57`，规避收盘集合竞价成交价偏差；
- 资金按剩余名额动态分配，高价股买不起时余钱顺延给下一名；
- 目标只数受 `paper_max_positions` 收缩，避免接近上限时闲置现金；
- 浮盈达 `paper_breakeven_arm` 后回落至成本附近触发 `breakeven_stop`；
- 持有多日后按 `paper_take_profit_tighten` 收窄止盈阈值；
- 曾封涨停后开板且有浮盈触发 `limit_open`；
- 封跌停时按交易日顺延，最多 `paper_limit_down_roll_max` 日后强制平仓。

模型口径、`close→close` 到期腿、`±30%` 重排上限和成本假设与 V1 完全一致。

关键参数：

```text
REALTIME_PAPER_V2=true
REALTIME_PAPER_BUY_END=1457
REALTIME_PAPER_MAX_POSITIONS=4
REALTIME_PAPER_BREAKEVEN_ARM=0.03
REALTIME_PAPER_BREAKEVEN_MARGIN=0.005
REALTIME_PAPER_TAKE_PROFIT_TIGHTEN=0.03
REALTIME_PAPER_LIMIT_DOWN_ROLL_MAX=3
```

两个账户的持仓都进入保护性订阅，`config.paper_state_files()` 是订阅清单与热重载的统一口径。V2 装配异常会降级跳过并保留 V1 运行。

## 9. 持久化与决策审计

所有文件位于 `logs/realtime/` 挂载盘，容器重建后保留：

| 文件 | 内容 |
|---|---|
| `paper_state.json` | 现金、持仓、持仓峰值、最近买入日 |
| `paper_trades.jsonl` | 每笔平仓的不可变流水 |
| `paper_buy_decisions.jsonl` | 每日模型池、重排、过滤和成交快照 |
| `paper_sell_counterfactuals.jsonl` | 卖出日及后续 3 个交易日反事实 |
| `paper_*_v2.json(l)` | V2 赛马账户的同名独立副本 |
| `signals_YYYYMMDD.jsonl` | 实时信号账本 |
| `engine.YYYYMMDD.log` | 引擎运行日志 |

买入决策审计记录：

- 模型池排除原因；
- 原始、校准毛收益和校准净收益；
- rank、score、adj 和分项贡献；
- 实时价、开盘价、昨收、VWAP、盘口失衡、价差和涨停状态；
- 入场过滤、未进入 Top2、资金不足或实际成交；
- 成交股数、价格、成本和账户前后状态。

卖出反事实按稳定 `trade_id` 幂等更新，记录 sell-day、D+1、D+2、D+3 的收盘价、最高价、继续持有收益和机会损益。日线尚未到达时保留空 markout，后续引擎启动自动补齐。

审计写入使用文件锁、临时文件、`fsync` 和原子替换。审计失败只写日志，不阻断交易。

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

# 查看运行时关键参数（只打印非密钥字段）
docker compose exec -T scheduler python -c "from realtime.config import load; c=load(); print(c.paper_time_cap_start, c.paper_buy_start, c.sell_horizon, c.paper_cost)"
```

当前虚拟数据流覆盖：模型池封闭、重排有界、A 股 T+1、五级卖出、保护性订阅、成本后净收益门、高开惩罚关闭、14:50 到期卖出、14:55 买入、决策 trace、反事实幂等补齐，以及 V2 的账户隔离、文件命名、跌停按日顺延、名额收缩和保护性止盈。

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
- 模拟盘使用实时快照近似成交，未建模完整盘口冲击、排队和实际成交概率；
- 日线反事实只能评估收盘和日内最高价，不能重建分钟级最优退出路径；
- `notify.env` 含推送和行情凭证，只能存放在挂载盘，禁止提交到 git。
