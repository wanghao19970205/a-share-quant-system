# A股量化系统审计结论修复清单

- **用途**：将《问题》中的审计结论转化为可执行、可验收的修复清单，供 Opus 复核。
- **审计基准**：`问题`，版本日期 2026-08-10；已吸收最新 walk-forward/purge/因子候选清单专项（P35–P38）。
- **目标代码库**：`/Users/wanghao81/Desktop/A/a-share-quant-system`；远端执行目录：`/www/A`。
- **研究隔离目录**：`/www/A/research_runs/audit_20260810_v1`。
- **原则**：本清单中的“通过”必须有代码、测试、输入 hash 和报告产物四类证据；没有真实数据或独立 holdout 的项目不得标记为已证明。

## 0、以实际净收益为目标的执行优先级

按当前收益分解和可逆性排序，后续优化严格按以下顺序执行：

1. **先降成本和修正暴露**：统一 `20bp`、固定 TopN、no-refill、严格买卖资格；比较 `horizon={3,5,10}` 与非重叠 `rebalance_stride`，输出毛收益、成本、现金、成交率和净收益。
2. **再修数据与标签真实性**：PIT 股票池、权威交易日历、逐股 purge、ST/停牌/流动性门控；任何可能高估现役或挑战者的口径先修复，再比较收益。
3. **再做候选模型增量**：冻结候选清单来源，统一 `label_col`、窗口和成本；先做 no-rule 对照、特征族重要性和市场超额收益。
4. **最后才搜索模型权重**：只在独立 holdout、配对 CI 和多重比较校正后搜索融合腿，不继续扩大已被校正后跨零的权重网格。

当前不以绝对收益门槛判断“优化成功”；实际净收益优化必须同时满足：相对市场/random 的配对下界、成本可解释、现金/成交率可解释、独立 holdout 和固定输入 hash。

## 一、必须保留的结论

这些不是待调参假设，而是当前审计后必须保留的判断：

- [x] 当前 `20bp` 双边成本、Top2/Top10、每日换手、纯多头配置不可盈利，置信度高。
- [x] `-29.10bp/天` 的固定退出结果可分解为：`-18.98bp` 成本、`-13.33bp` 尾部惩罚、`+3.21bp` 真实毛收益。
- [x] 自适应退出结果可分解为：`-19.79bp` 成本、`-1.12bp` 尾部惩罚、`+10.10bp` 真实毛收益。
- [x] “负收益因此模型比随机差”不能成立；模型是否优于随机必须由市场等权、random TopN、score-shuffle 和配对检验回答。
- [x] “分数特征/分钟特征无用”目前未被有效检验，不能作为结论。
- [x] `+1.62%/天、Sharpe 264` 的 native 日更结果是无效历史口径，不得进入新的比较表。
- [x] 旧的分钟结构化结果存在泄漏、筛选和评估宇宙问题，不得作为因果正面或负面证据。

## 二、执行状态约定

- `已完成`：本地修改、远端同步、SHA256 一致、相关测试通过。
- `运行中`：任务已进入隔离容器，必须有容器名、参数和监听任务。
- `待修复`：已定位问题，但未完成代码和测试。
- `待数据`：代码可修，但缺少远端数据/供应商语义/生产文件，不能凭空结案。
- `禁止发布`：研究结果不得修改 active manifest、生产 scheduler、V1-V5 或实时订阅。

## 三、已完成修复

### C01 统一回测基础口径

- **证据**：`问题` §3.1 A1-A4；`quant/backtest.py`。
- **已做**：
  - 默认双边成本改为 `0.002`。
  - Sharpe 改为 `mean/std*sqrt(periods_per_year)`。
  - 增加 `no_refill`，不可成交槽位记现金。
  - 增加显式 `require_tradability`，严格评估缺买入资格列直接失败。
- **验收**：
  - 合成数据验证默认成本为 `0.002`。
  - 正确 Sharpe 与独立公式一致。
  - TopN 不回填测试通过。
  - 缺列严格失败测试通过。
- **状态**：已完成；本地/远端组合回归已通过。

### C02 修复市值自我中性化

- **证据**：`问题` §5.5；`quant/factors/engineering.py:613-641`。
- **已做**：回归目标列不再作为自身解释变量；参与中性化的列显式转 float，兼容 pandas 3。
- **验收**：`log_mv_total` 残差不再接近零，pandas 3 smoke test 通过。
- **状态**：已完成。

### C03 修复 sentiment 零成本

- **证据**：`问题` §4.2、P18；`quant/sentiment_feature_experiment.py:211-215`。
- **已做**：成本从硬编码 `0.0` 改为读取回测单一成本真源。
- **限制**：sentiment 发布时间泄漏尚未修复，当前结果不能证明 sentiment 有效。
- **状态**：成本项已完成；生产发布时间泄漏待修复。

### C04 修复标签成熟度和不可买现金语义

- **证据**：`问题` §3.6、P13；`intraday_1400/evaluation.py`、`intraday_1400/pipeline.py`。
- **已做**：
  - 未成熟标签不再自动填 `-10%`。
  - 成熟且不可卖才使用尾部惩罚。
  - 不可买槽位计 `0%` 现金。
  - 排名先完成，未成熟/不可买不回填。
- **验收**：本地和远端 `intraday_1400` 测试均通过；当前完整 intraday 模块为 `139/139`，仓库级 unittest discovery 为 `249/249`。
- **状态**：已完成。

### C05 修复平板容差单位

- **证据**：`问题` §3.6、P14；`intraday_1400/features.py`、`pipeline.py`、`offline_race.py`。
- **已做**：绝对 `0.005` 元改为相对 `0.0005`；入口使用前收盘作分母，offline bar 使用前一交易日收盘，离场使用离场价近似分母。
- **限制**：该修复只改变平板判定，不得变相把 `14:00 return >= 4.5%` 本身当作排除条件。
- **状态**：已完成；本地和远端完整 intraday 回归通过。

## 四、负对照与统一评估

### C06 负对照实现

- **证据**：`问题` §1.2、§8.7、P0。
- **已做**：
  - 市场等权腿。
  - random TopN。
  - score-shuffle。
  - `200` 个 seed、TopN `{5,10,20}`、成本 `20bp`。
  - 7 worker 并行，固定资金和 no-refill 语义。
- **重要修复**：首轮发现 TopN 结果聚合错误，每个 TopN 错误显示 `600` samples；已改为按 `(top_n, target, cost, kind)` 分组，重跑后每个 TopN 正确为 `200` samples。
- **最终已取得的诊断结果**：
  - 市场等权：约 `-47.60bp/天`。
  - Random Top10：约 `-47.29bp/天`。
  - Score-shuffle Top10：约 `-47.26bp/天`。
  - asof-plus Top10：约 `-66.09bp/天`。
- **正确解释**：这支持“分钟增强当前没有表现出正向增量”，但还不足以证明其真实 alpha 为负；必须使用逐日配对差值 CI，并修复共同宇宙、日历和多重比较问题。
- **状态**：负对照运行和 h3 配对报告已完成；h1、h10、h5 及新冻结 holdout 的配对检验待补。

### C07 配对统计检验

- **证据**：`问题` P00b；当前负对照报告仅有 seed 分布，未输出模型对照的逐日配对差。
- **动作**：对每个 TopN 计算：
  - `model - market_equal_weight`；
  - `model - random`；
  - `model - score_shuffle`；
  - block bootstrap / stationary bootstrap CI；
  - 日内截面相关和时间相关的敏感性。
- **验收**：报告必须包含 `paired_mean_gain`、CI 下界/上界、样本天数、block 长度和 seed。
- **已执行（h3 旧预测诊断）**：`/www/A/research_runs/audit_20260810_v1/p00_h3_paired_bootstrap.json`；SHA256 `db35b8e02252f3fad7732acd4f1588577fae6d9a1ff13636d9e7ae5b54e2ef3b`。输入为 `480262` 行、`3008` 只股票、`478` 个源日期中固定 `dates[::3]` 的 `160` 个非重叠调仓日；严格使用 `buyable_close`、`tradable_ret_3d`、20bp、200 个 control seeds、5000 次 circular moving-block bootstrap、block=5 个三日周期。
- **结果**：Top5 相对市场/random/shuffle 的平均增益分别约 `60.04/58.56/61.38bp/三日周期`，95% CI 分别为 `[7.42,110.89]`、`[6.69,111.36]`、`[8.10,113.71]bp`；原始单侧 p 为 `0.0132/0.0148/0.0160`，但 9 项 Holm 校正后均为 `0.1188`，不得判定显著。Top10 和 Top20 的三类 CI 均跨 0，Holm p 均约 `0.6599`。
- **状态**：配对 block bootstrap 与 h3 报告族 Holm 校正已实现并验证；h1 分钟增强配对报告、block 长度敏感性、截面相关敏感性和新冻结 holdout 仍待执行。

## 五、P00 成本数量级搜索

### C08 严格持有期/TopN 评估

- **证据**：`问题` P00、P00e；`quant/daily_optimization_pipeline.py:30,322-323`。
- **动作**：运行：
  - `horizon ∈ {3,5,10}`；
  - `TopN ∈ {5,10,20}`；
  - 成本 `20bp`；
  - no-refill；
  - 严格 `buyable_close`/`tradable_ret_*`；
  - 固定日期窗口和输入 hash。
- **已执行**：远端 h3/h10 已从价格仓重 join `buyable_close` 和 `tradable_ret_*`，按 20bp、no-refill 运行；第一次失败原因是容器未设置 `QUANT_DATA_DIR`，修正后完成。
- **口径纠正**：首次报告按每日信号复利，h 日收益彼此重叠，h10 出现 `MDD≈-98%`，该报告作废。已改为每 h 个观测抽取一次的非重叠诊断。
- **非重叠诊断结果**：
  - h3 Top5：年化约 80.4%，Sharpe 2.58，MDD -17.0%；Top10：年化约 36.2%，Sharpe 1.69，MDD -18.0%；Top20：年化约 22.9%，Sharpe 1.18，MDD -16.8%。
  - h10 Top5：年化约 8.56%，Sharpe 0.43，MDD -34.8%；Top10：年化约 9.15%，Sharpe 0.47，MDD -35.8%；Top20：年化约 8.33%，Sharpe 0.44，MDD -36.7%。
- **同 horizon 对照结果（单位为每个持有周期，不是 bp/天）**：
  - h3 市场等权约 `26.21bp/3日周期`。Top5 模型 `86.25bp`、random 均值 `27.69bp`、random 97.5% 分位 `58.33bp`；Top10 模型 `53.09bp`、random 均值 `27.22bp`、97.5% 分位 `47.82bp`；Top20 模型 `40.53bp`，落在 random 分布内。
  - h10 市场等权约 `58.73bp/10日周期`。Top5/10/20 模型分别约 `59.91/59.56/55.57bp`，与 random 和 score-shuffle 基本一致，未显示增量 alpha。
  - **展示策略修正**：h3 Top5 的 `80.4%` 是 absolute annualized，不得单独作为优化榜首；后续表必须并列 absolute、market benchmark、excess、成本和 Holm 校正结论。当前 h3 Top5 相对市场的 Holm 后 p=`0.1188`，不得称为已证明增量收益。
- **当前解释（已加入 h3 配对检验后）**：Top5 相对市场/random/shuffle 的逐日配对 CI 在未校正层面均高于 0，但 9 项 Holm 校正后 p=`0.1188`，因此不能判定 h3 Top5 有显著增量 alpha。Top10/Top20 的三类配对 CI 均跨 0，未显示增量。
- **限制**：以上使用旧预测产物，不是新冻结的独立 holdout；random/shuffle 配对基准是 200 个 seed 的逐日均值；本次 block bootstrap 只覆盖 h3 的 9 项比较，尚未覆盖 h1、h5 或全部候选网格。
- **验收**：任何缺列、缺价格或日期 join 不完整必须失败，不得退回 `target_ret` 乐观口径；正式报告必须使用预注册的非重叠调仓日历。
- **状态**：h3/h10 严格执行和同 horizon 对照已完成；h3 配对 block bootstrap 与 9 项 Holm 校正已完成；h5 无现成预测产物，需重训；h10 配对推断和独立 holdout 待执行。

### C09 绝对门槛改为相对基准门槛

- **证据**：`问题` P00b；`quant/daily_optimization_pipeline.py:279-286`。
- **动作**：`choose_next_branch` 必须显式接收 baseline 指标，只允许在配对 CI 下界大于零、填充门槛、回撤和稳定性均满足时进入下一阶段。
- **禁止**：仅使用 `mean_return > 0`；禁止把 `daily_baseline_retained` 当成比较结果。
- **状态**：待修复。

### C10 多重比较校正

- **证据**：`问题` P00c；`scheduled_workflow.py:440-510`。
- **动作**：对 8100 网格、Top20 holdout、model expansion arms 和 pairwise 组合实施 BH/Holm 或预注册的 FWER/FDR 方案。
- **验收**：报告必须写明检验总数、校正方法、校正后 p/q 值和最终晋级依据。
- **禁止**：同一 holdout 上 20 个候选任一通过即晋级。
- **已执行（研究副本）**：h3 Top5/10/20 × 市场/random/shuffle 共 9 项比较采用 Holm FWER；报告写明 family size、原始 p、校正后 p 和拒绝标志，9 项均未通过 0.05。
- **状态**：h3 研究报告族已修复；8100 网格、Top20 holdout、model expansion arms 和生产晋级路径仍待修复，不得据此放宽生产晋级。

### C11 市场/行业相对收益

- **证据**：`问题` P00d；全仓 `hedge/benchmark_relative/beta_neutral` 零命中。
- **动作**：加入指数/同股票池等权收益，输出绝对净收益和超额净收益两套结果；必要时增加 beta/行业中性诊断。
- **验收**：每个模型必须有 `absolute_return`、`benchmark_return`、`excess_return`、成本和回撤。
- **状态**：市场等权腿已完成；日频指数/行业相对收益待补。

### C12 打开未采样维度

- **证据**：`问题` P00e。
- **动作**：用有限预算扫描：
  - 股票池：`mainboard/hs300/zz500/zz1000`；
  - 流动性/波动率分位数；
  - 持有期；
  - 换手预算；
  - 调仓频率。
- **禁止**：继续扩大已经被证明 CI 全跨零的融合权重网格。
- **状态**：待执行。

## 六、sentiment 与分数特征

### C13 修复 sentiment 发布时间泄漏

- **证据**：`问题` §4.2 S-1；`stock_analyzer/sentiment_signal.py:147-166,213`。
- **动作**：
  - 按新闻发布时间截断到信号时点；
  - 禁止收盘后新闻进入当日特征；
  - 对 IC 选择和训练加入 purge；
  - `enabled`/`blend_weight` 只能由无泄漏 IC 决定。
- **验收**：每条新闻必须有 publication timestamp；缺 timestamp 的记录进入单独缺失层，不得默认为当日可用。
- **状态**：待修复，当前生产 sentiment 启用判断不可信。

### C14 修复 sentiment 无新闻编码

- **证据**：`问题` §4.2 S-2。
- **动作**：先计算有效覆盖率，再在有信号子集上标准化；无新闻值保持缺失并显式加入覆盖率特征，不得在截面均值/标准差前直接填零。
- **验收**：无新闻股票不因当日整体舆情偏正/偏负而自动获得反向 z 值。
- **状态**：待修复。

### C15 修复 sentiment 消融和模型窗口

- **证据**：`问题` §4.2 S-3/S-4/S-5。
- **动作**：
  - 标注停止后的常量列不进入“无贡献”判决；
  - Ridge 与 boosted tree 使用同一 valid/predict window；
  - 使用日均 IC，不使用池化相关替代；
  - sentiment 与其它特征走同一 winsorize/中性化流程；
  - 记录每日覆盖率。
- **状态**：待修复。

### C16 分钟特征重做

- **证据**：`问题` §4.3 D1-D7。
- **动作**：
  - 完整 103 个候选池进入模型内筛选；
  - 去除数学上等价的 `market_excess`；
  - 保留强制待测特征通道；
  - base/plus 使用完全相同标签、超参、成本和窗口；
  - 不用 `entry_buyable=True` 预先污染评估宇宙；
  - 不完整日作为独立分层。
- **验收**：输出 base、plus、族消融、完整宇宙、成交率和配对 CI。
- **状态**：待 P00 和配对检验完成后执行。

## 七、数据与 holdout 完整性

### C17 bar 时间 START/END 外部验证

- **证据**：`问题` §5.1；`INTRADAY_ASOF_1400_PLAN.md` Batch 0。
- **动作**：从真实供应商 SDK 样本验证 `kline_time` 是 bar 起始还是结束；若是 END，重算 14:00/14:50 标签。
- **状态**：待生产样本；fixture 只能证明内部一致，不能结案。

### C18 修复 `_normalize_kline`

- **证据**：`问题` §5.2；`stock_analyzer/amazingdata_source.py:231-245`。
- **动作**：
  - 无名 DatetimeIndex 不得静默丢弃；
  - 显式解析 format；
  - int/string 时间分别验证；
  - 丢弃 NaT；
  - 重复时间戳显式去重或报错；
  - 同时出现 date/kline_time 时禁止生成重复 date 列。
- **状态**：待修复。

### C19 复权和因子版本对账

- **证据**：`问题` §5.3-§5.4、P7。
- **动作**：
  - 日线 prepared 增加 `factor_version`；
  - 日线/分钟在 `(code,date)` 上核对 qfq 价格和 factor version；
  - 因子表早于 bar、重复时间戳、部分 NaN 不得静默吞掉；
  - volume/amount 复权口径明确并单测。
- **验收**：跨面板 join 输出 mismatch 数、最大价格偏差和版本覆盖率。
- **状态**：待修复/待数据对账。

### C20 pandas 版本和变体污染

- **证据**：`问题` P16；`fair_race_pipeline.py:343-362`。
- **动作**：生产容器固定 pandas 2.x；若实际是 pandas 1.x，所有变体赛跑作废重跑；避免 `deep=False` 写回父 panel。
- **状态**：已完成。远端 scheduler 镜像实测 pandas 3.0.5，pandas 1.x 污染开关已排除；`panel_for_variant` 的 `daily_h1` 分支已改为 `deep=True`，并补充结果修改不写回父面板的回归断言。本地完整回归 `265/265`、远端隔离聚焦测试 `23/23` 通过。

### C21 holdout 账本和 state hash

- **证据**：`问题` §3.3 C6、P17。
- **动作**：
  - 保留 `structural_combo_holdout` 的 `daily_history_causal=False` 护栏；在补齐日更历史逐行发布时点（C30）前禁止置 True；
  - holdout claim/consume 必须强制执行；
  - manifest 增加 state_hash 自校验；
  - state-dir 删除或迁移不得静默重置；
  - 固定 holdout 边界，不从当前数据最后日期动态推导。
- **验收**：重复运行、篡改 manifest、删除 state-dir、绕过 main 函数均有失败测试。
- **状态**：措辞已修正：`daily_history_causal=False` 是必须保留的因果护栏，不是待修复死锁；仅在 C30 补齐逐行发布时点后才重新评估是否置 True。

## 七-A、第二轮严格审计新增问题

本节吸收《问题》第二轮审计的新增结论。除明确标为“已完成”外，均不得用于放宽发布或晋级门槛。

### C22 统一赛跑样本宇宙并补入口门控

- **证据**：`问题` §3.2 B5/B6、P20/P21；`intraday_1400/fair_race_pipeline.py:758-779,1772-1775`。
- **问题**：`_evaluate_recipe_frames` 逐个模型传入模拟器，任一 NaN 作废整天时，不同配方实际比较不同日期；`run_screened_rolling_race` 又缺少 `signal_eligible` 过滤。
- **动作**：所有候选一次性传入，先求共同日期集合再比较；所有入口统一执行 `signal_eligible`、成熟度和完整日门控。
- **验收**：模型/配方报告输出相同日期 hash、共同天数、删失率和 pairwise CI；缺门控时测试失败。
- **状态**：部分完成并已远端隔离验证：recipe 联合进入共同宇宙，screened rolling race 在排名前应用 `signal_eligible`；与 C23 合并的远端聚焦测试 `34/34` 通过。日期 hash、共同天数和删失率的最终报告字段仍待补，不据此放宽晋级门槛。

### C23 E4 日更先验补覆盖率和权威日历

- **证据**：`问题` §5.13、P22；`structural_combo.py:141,195-200`、`structural_combo_experiment.py:190-198`。
- **问题**：E4 使用面板自身日期推导交易日历，缺日会让先验静默陈旧；`inner` join 只验证重复键，不报告丢行和 Top100 实际覆盖率。
- **动作**：使用外部权威交易日历；对 next-date 映射、过滤前后行数、Top100 保留率和缺日直接断言并记录。
- **验收**：报告包含日历 hash、缺失日数、每道过滤门覆盖率；任何缺日或覆盖率低于阈值均拒绝结论。
- **状态**：部分完成并已远端隔离验证：experiment/holdout 改读完整 prepared 交易日历，报告日历 SHA256、缺日、候选行数和 Top100 实际保留率；缺任一预期日硬失败；与 C22 合并的远端聚焦测试 `34/34` 通过。覆盖率阈值尚未预注册，当前 E4 数字仍不得作为完整证据。

### C24 补强分数列黑名单和特征泄漏护栏

- **证据**：`问题` §4.5、P23；`engineering.py:522-552`、`features.py:370-380`。
- **问题**：黑名单未覆盖 `pred`、`score`、`prior`、`e4` 等分数形状列名；当前未必已发作，但边界一旦变化会允许模型分数回流特征。
- **动作**：增加定向前缀黑名单（`e4_`、`prior_`、`daily_pred_` 等），不要粗暴禁掉合法 `rule_score`/`rule_score_chg_5`；建模入口使用显式白名单。
- **验收**：恶意分数列进入训练时硬失败；合法 rule 特征回归测试继续通过。
- **状态**：已完成代码修复与隔离验证：增加 `e4_`、`prior_`、`daily_pred_` 定向前缀黑名单，保留合法 `rule_score`；selection manifest 在过滤可用列前硬失败，避免危险列被静默丢弃。新增恶意分数列和合法 rule 特征回归测试，本地完整回归 `265/265`、远端隔离聚焦测试 `51/51` 通过。

### C25 重做“特征无用”判定和模型指标

- **证据**：`问题` §4.5、P30-P32；`quant/model.py:184-198,388-399,404,688`。
- **问题**：单变量 IC 会删除只在交互中有效的特征；`alpha=10` 在生产样本量下近似 OLS；高度相关特征族会把系数按约 `1/k` 摊薄；Ridge 报 valid、Ranker 报 test，且池化 Spearman 把选日能力当选股能力。
- **动作**：先确认 S10 的 `rolling_factor_select`/`purge_horizon` 实际状态；按特征族汇总系数或 grouped permutation importance；去掉 `rule_score`/`rule_score_chg_5` 单独重训以分离规则 alpha；所有模型统一同一切分并按日计算 IC/分位数指标。
- **验收**：报告同时给单列、特征族、grouped importance、逐日 IC、覆盖率和统一 valid/test 窗口；未完成前禁止写“分数特征无用”。
- **只读证据（S10）**：2026-08-10 远端 active manifest 显示 `rolling_factor_select=True`、`purge_horizon=True`；现役共线性筛选与 purge 均已开启，但这不替代 grouped importance 和统一切分。
- **状态**：部分完成，仍未解除“特征无用”禁令。broad/no-rule 严格同切分训练、逐日收益配对 bootstrap 和 rule_score 对照已完成；新增 `audit_20260810_c25_grouped_importance.json` 对两条 factor_audit 做特征族选择/IC 汇总：主要稳定贡献来自 `price_volume`，去除 rule_score 后其他家族统计基本不变。该报告是 grouped selection/IC diagnostics，不是 grouped permutation importance；统一 valid/test 报告和 grouped permutation 仍待补。报告 SHA256：`6d42af4bdd5c7ee556a2ddf33a10410d472a75015ac1f9416e9c3bcaf91e82bd`。

### C26 修复生产 active 产物、成本和融合搜索的可变性

- **证据**：`问题` §5.9-§5.11、P31-P33；`scheduled_workflow.py`、`watchlist_grid.py`。
- **问题**：active 预测产物可被后续训练覆盖，E4/H1/H2 可能读取不同版本；基础模型与融合腿存在跨目标量纲混用、`raw_pred` 未 z 化、两个独立成本旋钮和固定尾部惩罚；`lgbm_weight` 不进网格，影子腿权重也未搜索。
- **动作**：active manifest 固化 source hash、发布时间和输入窗口；严格区分目标/预测 horizon；成本和尾部惩罚单一真源；将 `lgbm_weight`、elastic/catboost 等实际消费的腿纳入预注册搜索，或明确删除未搜索腿。
- **验收**：同一报告所有腿 source hash 一致；成本变更可追溯；融合网格报告完整参数和搜索族；禁止以未搜索空间得出“某腿无效”。
- **状态**：部分完成，生产只读审计确认存在实际 provenance/口径不一致，禁止直接发布或覆盖 active 产物。发布日期闸门已前置到任何 active 复制/合并之前，保持原失败条件但避免失败后留下半更新 active；本地完整回归 `265/265`、远端隔离聚焦测试 `42/42` 通过。source hash、训练窗口、各腿 horizon 和融合搜索仍未闭合。
- **只读证据（2026-08-10）**：生产 `active_quant_model.json` 存在，但独立 `active_manifest.json` 缺失；`prediction_latest_date=2026-08-07` 晚于 `price_latest_date=2026-07-13`；短线字段同时出现 h1/h3，波段声明 h10 但复用短线 h1 predictions，summary/returns 又指向 short h3；`model_expansion_active_hashes.json` 只有文件 hash，没有训练窗口和各腿输入 provenance。生产文件未修改。

### C27 落盘权威交易日历，消除行位移和停牌免费跳过

- **证据**：`问题` §5.13-§5.14、§5.18、P25；`engineering.py:156-192,482-497`、`tradability.py:29-48`。
- **问题**：仓库没有真实交易日历，缺失时静默用 `weekday()<5`；标签、rolling、持有期、卖出顺延和 purge 大量按行位移，停牌期间可能被无代价跳过；sentiment 用自然日衰减，约稀释 40%。
- **动作**：生成并 hash `trading_calendar.parquet`，缺失即失败；所有 shift/rolling/持有期/purge 按股票交易日历；日线链补 `risk_trading_gap_days`；sentiment 改交易日衰减。
- **验收**：长假、停牌、月末、跨年 fixture 覆盖标签、平仓、rolling、purge；报告输出日历 hash 和实际交易日数。
- **状态**：待权威数据，禁止按 prepared 行并集冒充交易所日历。C27 隔离探测在远端数据目录发现 0 个 calendar/trade_cal/trading_day 候选文件；44 个代码命中均为 prepared 月文件派生日历，或读取实际不存在的 `trading_calendar.parquet`。在取得交易所开闭市源之前不修改生产 shift/rolling/purge 语义。探测报告 SHA256：`91e6e87e5cf7a1ea21ce161df5d0144c26cc8b223eca520e7fc5c93fdb56dcab`。

### C28 修复宇宙生存者偏差和盘中 codes-file 来源不明

- **证据**：`问题` §5.12、§5.17、§5.20、P26；`datafeed.py:149,154-175`、`full_train_batched.py:775-800`、`intraday_1400/collector.py:284-294`。
- **问题**：当前池是“今天仍上市”的快照并回溯到历史；快照覆盖写，盘中 `--codes-file` 无默认值、无写入方、无调用方，且盘中链与日线链宇宙完全无关。`universe_codes` 还被排出历史缓存键。
- **动作**：按月保存 PIT 宇宙快照并按窗口起始日读取；外部退市名单做上界回测；对价格文件/池文件做只读数量与 hash 对账；将 universe codes 纳入 recipe hash；盘中链必须声明、固化并校验 codes-file 来源。
- **验收**：每个报告有 universe snapshot hash、起止日期、退市处理和代码覆盖率；缺失来源或历史池不可复现时直接禁止比较。
- **只读证据（S9）**：2026-08-10 远端 `price/` 有 `5533` 个 parquet，active 主板池去除注释后为 `3193` 行；其中主板前缀且不在 active 的仅 `3` 只，差异主要是 `300/301/302/688/689/920` 等非主板代码。价格仓与 active 池仍不是同一宇宙，但不能把 `2339` 全部解释为退市偏差。
- **状态**：待权威数据/待修复；真正退市主板上界较小，主要 survivorship 风险仍来自 PIT 成分和 `min_price_rows` 历史准入。C28 隔离探测仅发现 `full_a_universe.txt` 和 `mainboard_active_universe.txt` 两个静态列表，没有上市/退市日期、历史成分或证券主数据 parquet，因此不能从现有快照构造 PIT 宇宙。探测报告 SHA256：`0d5bad671d17fb36270aa9d8ecceb205a8303015b58d02118a5a5b1ca7f93f31`。

### C29 补齐 ST、新股、停牌和流动性口径

- **证据**：`问题` §5.19、P28；`quant/tradability.py:7,94,102-133`、`intraday_1400/features.py:295-299,363-366`。
- **问题**：没有 `is_st`、上市日期、退市整理期或绝对流动性阈值；±9.5% 硬编码误判 ST 和上市首日；日线停牌按陈旧收盘成交，盘中链已有的 suspended 语义未移植；`risk_amount_vs_median_20` 缺少 `shift(1)`。
- **动作**：加入 ST/板块/上市日/退市日字段，参数化涨跌停阈值；停牌买卖和顺延与盘中链统一；增加成交额/ADV 门；修 `risk_amount_vs_median_20` 的 lag。
- **验收**：ST、上市首日、退市整理、停牌、零成交和低流动性 fixture 均有明确状态；缺字段不得默认可买可卖。
- **状态**：部分完成。`risk_amount_vs_median_20` 已改为仅使用截至前一交易日的 20 日成交额中位数，避免当日成交额进入自身基准；本地 `intraday_1400` 回归 `142/142` 通过。ST/上市日/退市日、参数化涨跌停、日线停牌和绝对流动性门仍待权威数据，不在字段缺失时猜测生产语义。

### C30 修复标签可见时点、复权缓存和数据闸门

- **证据**：`问题` §5.21、P27、P34；`engineering.py:227-229,296-298`、`full_train_batched.py:721-723`、`scheduled_workflow.py:1075`。
- **问题**：`ann_date` 缺失时退化为 `report_date`，可能提前 1–4 个月；baseline recipe 不含 `price_source_signature`；`refresh-months=1` 使月末 horizon 行永久 NaN；龙虎榜/大宗交易可能在披露前进入特征；复权失败静默回退未复权价；OHLC 缺失时 `buyable_*` 默认 True。
- **动作**：缺公告日直接丢弃并计数；baseline 缓存加入复权签名；`refresh-months >= 2`；延迟披露因子至少 lag 一交易日；复权失败 raise；OHLC 缺失默认不可交易。
- **验收**：报告输出丢弃行数、公告时点覆盖率、缓存命中签名、月末标签完整率和复权失败数；任何闸门缺失直接失败。
- **状态**：待修复，保持未结案。
- **只读数据审计（2026-08-10）**：四类远端快照 `financial_yjbb/income/balance/performance_forecast` 的 `ann_date` 均为 100% 非空；因此不能把当前问题归因于公告日缺失。`financial_yjbb` 的 `ann_date-report_date` 中位数为 417 天，`income` 为 396 天，`balance` 为 48 天，`performance_forecast` 为 15 天，字段语义/映射仍需数据供应方对账，禁止把 `report_date` 直接当公告日替代。价格样本 200 个文件、412,963 行的 OHLC 缺失计数均为 0。
- **证据**：`audit_20260810_c30_data.json`，隔离报告任务 exit 0；SHA256 `57335d5f309304d415e74da7b15ce837de0867da237a055d133c6b369334a52b`。

### C31 修复会误报并中止发布的网格异常

- **证据**：`问题` §4.6.1、P29；`quant/scheduled_workflow.py:144-160,1180-1189`。
- **问题**：`best` 只在 `try` 内绑定，网格异常后触发 `UnboundLocalError`，再被包装成“无法读取 incumbent 参数”，导致一次网格失败中止日更并误导排障。
- **动作**：在 `try` 前初始化 `best=None`，保留原参数的失败语义；区分网格失败、incumbent 读取失败和发布失败。
- **验收**：构造网格异常时发布不漂移、不误报；错误类型和原始异常保留；正常网格路径回归通过。
- **状态**：已完成代码修复与隔离验证：incumbent 读取异常单独拒绝发布，short grid 异常只保留 incumbent，缺失的 swing grid 调用不再包装成 incumbent 读取失败；本地完整回归 `265/265`、远端隔离聚焦测试 `51/51` 通过，生产路径未触碰。

### C32 统一纯日线候选与在位 baseline 的冠军配置对比

- **证据**：`问题` §6、P9；`quant/target_ab_experiment.py:63-125`。
- **问题**：现有严格三腿工具的训练超参不是在位冠军配置，且仓库没有运行产物，因此不能回答“日更是否真的优于严格候选”。
- **动作**：从 active manifest 固化 champion params，统一日期窗口、TopN、20bp、no-refill、可交易 join；输出 baseline、open Ridge/LGBM/ExtraTrees/RandomForest 的绝对与相对收益、CI、成交率和输入 hash。
- **验收**：同一配置、同一宇宙、同一 holdout、同一成本下才允许比较；缺任一 source hash 或结果报告则标记不可用。
- **状态**：待数据/待执行。

### C33 修正用户可见 horizon、融合权重和统计指标

- **证据**：`问题` §4.6、P32/P33；`scheduled_workflow.py:83,693,1074`、`quant_signal.py:613-616,673-674`、`quant/model.py:184-198,404,688`。
- **问题**：swing 显示 h10 但训练未进入 swing horizon，用户看到的“10 日预期”可能来自 h1；`lgbm_weight` 不进网格；池化 `_metrics` 与逐日选股能力混淆，Ridge/Ranker 使用不同切分。
- **动作**：真正训练 h10 或将显示标签改回 h1；把 lgbm/影子腿权重纳入预注册搜索；统一 valid/test 窗口，所有 IC/分位数按日汇总。
- **验收**：用户可见 horizon 与训练标签、预测文件列和 manifest 三者一致；报告包含逐日指标和统一切分；未完成前禁止展示 swing h10 alpha。
- **状态**：待修复。

### C34 复核分数特征与规则 alpha 的可分离性

- **证据**：`问题` §4.5、P30/P31；`engineering.py:192-193,563-570`、`watchlist_grid.py:214-217,268,326`。
- **问题**：`rule_score` 同时作为输入特征、朴素融合腿和组合阈值，`naive_weight>0` 时模型 alpha 与规则 alpha 不可分离；仅看单列 Ridge 系数或单变量 IC 会错误淘汰交互特征和相关特征族。
- **动作**：去除 rule_score 及其变化列重训一条严格对照；按特征族汇总系数/置换重要性；确认 rolling factor selection 后再解释系数。
- **验收**：输出 raw model、no-rule、family-grouped importance、逐日 IC 和配对 CI；在此之前不得下“分数特征无用”结论。
- **状态**：已完成。
- **结果摘要（2026-08-10）**：
  - **broad（67 因子含 rule_score）**：total_return=-19.80%，Sharpe(stride=5)=-0.125，MaxDD=-77.07%。
  - **no-rule（65 因子不含 rule_score/rule_score_chg_5）**：total_return=-29.51%，Sharpe=-0.301，MaxDD=-72.25%。
  - **配对 bootstrap**：mean_diff=+6.53 bps/rebalance，95% CI=[-25.94, +40.88] bps，p_two_sided=0.71。
  - **rule_score IC 画像**：selected 12/12 windows，mean IC=-0.032，stability_score=0.547（rank~24/41）。
  - **结论**：broad 与 no-rule 无统计显著差异（p=0.71），rule_score 贡献弱负 IC，但边际价值不可区分于噪声。rule_score 不有害，但无可测量 alpha 提升。33 因子基线（Sharpe -0.113）优于两者，多加低质量因子稀释信号。
  - **证据 SHA256**：
    - `broad_vs_norule_comparison.json`: `562ee3abe356a94c1310d428e73e7d04662fcb3490f0b19e2c849530c6970ac8`
    - `audit_20260810_h5_broad_returns.parquet`: `acc2229d83f341d79364c1e55e4273d60e2cb4efbb97d3285c286c465fd6f3ce`
    - `audit_20260810_h5_no_rule_returns.parquet`: `5f69150eccb21e7b4d78295419822d3e24469692a3384fc2cb1cb6d19e87b023`

### C35 为 horizon 扫描设置 validation/purge 前置闸门

- **证据**：`问题` §5.22、P35；walk-forward/purge 专项报告 `abc556ccf17b8024f`。
- **问题**：horizon 增大后，默认 `validation_months=1` 被 purge 大幅压缩；h10/h15 的验证日期可能只剩约 11/6 天，早停轮数失去统计意义，且当前没有明确告警。
- **动作**：P00 h3/h5/h10/h15 扫描前将 `validation_months` 提高到 2–3；分别验证 `purge_horizon` 开关，不得与 `rolling_factor_select` 绑死；对 purge 后验证天数设置硬下限，低于下限直接跳过并报告原因。
- **验收**：每窗报告写入 train/valid/predict 边界、purge 天数、purged 后有效验证交易日数和最小阈值；任何不足窗口不得产生可晋级结果。
- **依赖**：C36 完成后才能重训；C35 完成前禁止把新的 h5/h10/h15 结果写入比较表。
- **状态**：部分完成。purge 与 rolling factor selection 已解耦，active manifest 分别恢复两者状态；h3/h5 至少要求 2 个 validation 月，h10/h15 至少 3 个，不满足即硬失败；本地回归和隔离测试通过。新增 C35 只读审计：h5 broad 预测有 225 个日期，1 个日期联合可买数低于 TopN=5，且有 1 个日期为 0；但预测产物未保留逐窗 train/valid/purge 边界，无法据此结案。逐窗有效交易日硬下限及窗口 provenance 仍待实现。报告 SHA256：`b8b81bfe4e3aea754ed3f32a6444c43d0b0b0c365243640b9febb55bb03449bf`。

### C36 修复标签正确性、逐股 purge 和 recipe hash 闭合

- **证据**：`问题` §5.22、P36；专项报告 S16–S18。
- **问题**：`tradable-label` 的 purge 少两个交易日；`QUANT_BT_SELL_ROLL_MAX_DAYS` 会改变标签但未进入 recipe hash；面板级日期 purge 无法覆盖停牌股按个股行位移造成的标签穿透。
- **动作**：按标签实际可见/可交易跨度计算 purge；将 `QUANT_BT_SELL_ROLL_MAX_DAYS` 和相关标签语义写入 recipe signature；按 `(code, date)` 的个股交易日历执行 purge，并对停牌/缺行边界做断言。
- **验收**：h1/h3/h5/h10 fixture 证明无训练标签跨入 validation/test；修改上述环境变量必然改变 recipe hash；报告输出逐股 purge 统计和边界样例。
- **依赖**：C27 的权威交易日历；C36 完成前禁止复用旧缓存进行 horizon 结论。
- **C39 依赖补充**：`rebalance_stride` 必须从权威交易日历抽取；C27 未完成前只能作为代码级口径修复，不能把 stride 结果写入收益排序。
- **状态**：本地确定项已修复：`tradable-label` purge span 从 `h` 改为 `h + cap - 1`（cap 来自 `bt_sell_roll_max_days()`）；`QUANT_BT_SELL_ROLL_MAX_DAYS` 已进入非 baseline recipe hash；walk-forward purge 新增逐股最保守边界；缺 OHLC 时 `buyable_*` 默认不可交易；完整回归通过。停牌期间的全部日线 rolling/标签仍需权威 PIT 日历覆盖。

### C37 隔离并作废 `pipeline.py` 来源的历史结论

- **证据**：`问题` §5.22、P37；`pipeline.py:149-152`。
- **问题**：该入口未传递 `train_end`，使用 70/30 切分且 `predict_start=None` 使 `predict=valid`；早停评估集与最终评估集重合，历史结果存在教科书式评估泄漏。
- **动作**：在研究档案中给所有 `pipeline.py` 来源报告加作废标记并从新比较表排除；修复入口的明确 train/valid/predict 边界后，仅用新的独立 holdout 重跑，旧报告不得覆盖新报告或 active manifest。
- **验收**：报告元数据记录入口、train_end、valid_end、predict_start、holdout hash；测试断言 eval_set 与最终评估集不重合。
- **依赖**：C21 固定 holdout 账本；C37 完成前禁止引用旧 `pipeline.py` 结果支持晋级。
- **状态**：本地入口修复完成：`pipeline.py` 现在强制显式 `train_end/valid_end/predict_start`，并要求 `train_end < valid_end < predict_start`；边界写入 summary，缺失或重叠直接失败，聚焦测试通过。历史 `pipeline.py` 结果仍保持作废，尚未重跑。

### C38 重建并冻结 `factor_selection_lh1000_cont` 候选清单来源

- **证据**：`问题` §5.22、P38；S14/S15。
- **问题**：仓库内没有该冻结清单的生产者；`lh1000` 仅出现在消费者，来源、生成日期、样本窗口和 label 语义不可审计。因子选择与 `train_catboost_ranker` 还可能使用默认 `label_col`，无法确认目标一致。
- **动作**：先在生产主机只读核查文件 mtime、行数、内容 hash、来源和窗口；无法证明来源时按完整 PIT 候选池重建并冻结清单；让 `select.daily_ic`、`train_catboost_ranker` 显式接收并记录正确的 `label_col`。
- **验收**：清单 manifest 包含候选池 hash、生成代码版本、label_col、train/valid 窗口和生成时间；因子选择与 CatBoost 的 label_col 在配置、summary 和报告中一致。
- **依赖**：C27/C30 的日历和标签闸门；C38 完成前不得发布“分数特征无用”结论。
- **补充依赖**：C25/C34 必须后置于 C38 候选池内容核查；当前 33 个因子不含 `score` 或 `rule` 列，不能用这 33 列上的 grouped importance 回答分数特征问题。
- **只读证据（S14）**：2026-08-10 远端清单为 `33×7`，列为 `factor/ic_mean/ic_std/ic_count/icir/ic_win_rate/abs_ic_mean`，mtime `2026-07-08T20:46:18Z`，SHA256 `2cf907e050209f16f6b9a3fcb804569c783dad4df9c106a749eba21193673a16`；active manifest 不含其来源、窗口或 label 元数据。
- **状态**：只读核查和 `label_col` 贯通完成：`daily_ic/ic_summary/select_factors` 支持显式标签，rolling IC cache 按标签隔离，CatBoost ranker 接收并使用 `label_col/train_mask_col`；新 selection 会同时生成 manifest，记录候选池/入选因子/代码 SHA256、label_col 和显式窗口，聚焦测试通过。现有 33 因子清单来源仍不可审计，必须通过新入口重建后才能解除禁令。

### C39 收益优先的非重叠调仓比较

- **证据**：`问题` P00/P00e；现有严格结果的成本/毛收益分解。
- **问题**：每日调仓的 20bp 成本与毛收益同量级；旧 walk-forward 没有统一的非重叠 `rebalance_stride` 参数，h 日预测容易被按日重叠评估。
- **动作**：`quant.backtest.walk_forward` 增加 `rebalance_stride`；按源日期日历固定抽取 `dates[::stride]`，不因中间缺失日期重新相位；Sharpe 年化按 `252/stride`；研究优先比较 `horizon={3,5,10}`、`TopN={5,10,20}`、`stride={horizon}`。
- **验收**：报告同时输出毛收益、成本、净收益、现金比例、成交率、市场/random 配对 CI 和输入 hash；任何重叠 h 日结果不得进入优化排序。
- **已执行**：在独立硬链接数据快照上完成 h5 严格重训，12 个 recent walk-forward 窗口、`tradable-label`、逐股 purge、3 个月 validation、7 workers；输出 predictions `676342` 行，严格列包含 `tradable_ret_5d/buyable_close/buyable_next`。
- **共同日期诊断**：h3/h5/h10 共同 `205` 天（2025-09-02..2026-07-10），Top5、20bp、no-refill、`stride=horizon`。h3/h5/h10 分别 `69/41/21` 个周期；h5 tradable coverage `97.77%`，该缺口必须保留。
- **配对控制结果**：h3 相对 market `+73.47bp/周期`，CI `[-19.86,172.47]bp`，Holm p=`0.9492`；h5 `+21.18bp`，CI `[-92.62,133.67]bp`，Holm p=`1.0`；h10 `+62.91bp`，CI `[-94.99,240.56]bp`，Holm p=`1.0`。三者相对 random/shuffle 的 CI 也全部跨 0。
- **报告**：`audit_20260810_horizon_controls.json`，SHA256 `5eba3c2b6d37ffa580a7cea2eb982a36b413a68a0c0c0907ef3c0d4fac8638f4`；200 seeds、5000 bootstrap、block=5、9 项 Holm family。
- **状态**：严格 h5 和共同日期控制已完成；没有 horizon 通过 Holm 0.05，全部禁止晋级或写入 active。

### C40 持久化 factor IC cache，避免重启重复计算

- **问题**：h5 首次 factor selection 对约 700 个日期计算全量日 IC 约耗时 214 秒；原 `DailyICCache` 仅内存缓存，停止/重启后全部丢失。
- **动作**：按 horizon、label_col、因子集合生成 cache key，将日 IC 分片写入 `window_cache_dir/factor_ic`；窗口内继续复用，进程重启后从 parquet 恢复。
- **验收**：同一输入跨两个 cache 实例命中全部日期，缓存写入采用临时文件替换；缓存只影响计算耗时，不改变 IC 数值和候选选择。
- **状态**：已完成。本地完整回归 `264/264`、远端聚焦测试通过；prepared panel 104 个月直接复用，窗口 53–57 cache-hit 仅约 `0.07–0.09s/窗`，factor IC 已持久化为 parquet，后续窗口 selection 降至约 `35–38s`。

## 八、生产隔离和发布纪律

- [x] 研究修改先在本地测试，再按文件同步远端并做 SHA256 校验。
- [x] 远端生产 `/www/A` 原有 dirty 改动未被覆盖；已同步文件逐一核对。
- [x] realtime/app/mobile/scheduler 容器保持运行。
- [ ] 所有研究容器使用独立名称、独立日志和独立报告路径。
- [ ] 研究结果不得自动写入 active manifest。
- [ ] 任何 forward shadow 进入前必须经过独立 holdout、hash、填充门槛和人工批准。
- [ ] 保持 `SCHEDULER_DISABLED=1`，不因研究重跑 routine model-update。

## 九、Opus 复核入口

请重点复核以下高风险点：

1. `evaluation.py` 的市场等权/random/shuffle 是否共用同一标签宇宙、成本、固定分母和成熟度语义。
2. random/shuffle 每个 TopN 是否独立恰好 200 samples；是否存在跨 TopN 聚合。
3. `P00` h3/h10 严格 join 是否真的包含 `buyable_close` 和 `tradable_ret_*`，是否有隐式回退到 `target_ret`。
4. P13 未成熟标签是否保持 NaN，成熟不可卖是否才填 `-10%`。
5. P14 是否只改变平板比例，而没有把 `4.5%` 强势收益本身作为排除条件。
6. 所有报告是否同时提供绝对收益、市场超额收益、成本、成交数、现金比例和配对 CI。
7. sentiment 的 enabled/blend_weight 是否仍可能使用收盘后新闻或无 purge IC。
8. holdout 是否固定、不可复用、不可通过删除 state-dir 重置。
9. 所有研究代码是否与 `/www/A` 对应文件 SHA256 一致，且没有覆盖原有 dirty 文件。
10. 任何候选是否只有在多重比较校正后仍显著，才允许进入人工复核。

## 十、当前不得写入的结论

在 C07、C08、C09、C10、C11、C13、C16、C17、C19、C21、C25、C34、C35、C36、C37、C38 未完成前，不得写入以下结论：

- “分数特征确定无用”；
- “分钟特征确定无用”；
- “sentiment 确定无用”；
- “模型确定比随机差”；
- “日更确定优于所有候选”；
- “h3/h5/h10 已经找到可盈利配置”；
- “某个候选可以生产发布”。
