<div align="center">

# 📈 A 股量化选股与实时交易辅助系统

**从数据采集 → 滚动训练 → 候选晋级 → 盘中实时重排与模拟交易的一体化研究与生产系统**

集 Streamlit 桌面/移动分析界面、行情与新闻多维快照、walk-forward 因子训练、
多模型融合、候选晋级评估、盘中实时策略层与容器化定时调度于一体。

</div>

> ⚠️ **免责声明**：本项目仅用于数据分析与量化研究，**不构成任何投资建议**。
> 历史回测与模型评分不代表未来收益，据此产生的任何盈亏由使用者自行承担。

---

## ✨ 特性一览

| 维度 | 能力 |
|---|---|
| 📥 **数据采集** | A 股日线 / 估值 / 财务 / 资金流 / 事件 / 新闻；AmazingData 券商源优先，AKShare 兜底 |
| 🧠 **模型训练** | 主板活跃池 walk-forward 滚动训练；Ridge + LightGBM Ranker + ElasticNet + ExtraTrees 融合 |
| 🔬 **晋级评估** | CatBoost 等候选模型先跑影子实验（RankIC / 相关性 / 留出期收益 / 稳定性），过门槛才升级 |
| 📊 **回测发布** | 短线 / 波段预测发布、固定候选组跟踪评估、成本口径回测 |
| ⚡ **盘中实时层** | 单会话订阅个股与行业 ETF Level-1 快照；V1/V2/V3/V4 独立账户并行赛马、动态重排与异动推送 |
| 📰 **新闻情绪** | Qwen 结构化标注（无 Key 时降级本地规则）；舆情因子经 ablation 评估决定是否入模 |
| 🖥️ **可视化** | Streamlit 桌面端（8501）+ 移动端（8502）双界面 |
| 🐳 **生产部署** | Docker Compose 三服务 + 容器内 cron 定时调度 + watchdog 自愈 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          数据源 (Data Sources)                        │
│   AmazingData 券商源(优先)  ·  AKShare 公开接口(兜底)  ·  新闻/舆情     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  app (8501)   │     │   scheduler       │     │  mobile (8502)   │
│  桌面分析 UI  │     │  定时调度/训练     │     │   移动端 UI      │
│  stock_app    │     │  cron + jobs      │     │  只读快照        │
└───────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
        │                      │                        │
        │         ┌────────────┴────────────┐           │
        │         ▼                          ▼           │
        │  ┌──────────────┐         ┌─────────────────┐  │
        └─▶│ quant_data/  │◀────────│  quant/ 训练管线 │  │
   (只读)  │ 行情·预测·   │  (读写)  │  walk-forward   │  (只读)
        ┌─▶│ 训练缓存     │         │  多模型融合      │◀─┘
        │  └──────────────┘         └─────────────────┘
        │                                   │
        │                                   ▼ 发布 active 预测
        │                          active_quant_short_predictions.parquet
        │                                   │
        ▼                                   ▼
┌──────────────────────────────────────────────────────────┐
│              realtime/ 盘中实时层 (交易时段常驻)             │
│  engine → 单会话订阅个股/ETF L1 → rerank 动态重排 →         │
│  V1/V2/V3/V4 独立模拟盘赛马 + RankBoard 候选榜 →           │
│  notifier 推送 + decisions/trades/counterfactuals 审计     │
└──────────────────────────────────────────────────────────┘
```

**数据流主线**：`scheduler` 拉数训练 → 发布预测到 `quant_data/`（共享卷）→ `app`/`mobile` 只读展示、`realtime` 引擎读预测作为选股候选池 → 盘中动态重排 + 模拟交易 + 推送。

### 实时策略赛马

四个版本共享模型候选池、交易成本和实时上下文，但账户及审计完全隔离，便于前向归因：

| 版本 | 成交口径 | 主要差异 | 验证目标 |
|---|---|---|---|
| **V1** | `last` | 基准资金分配与固定比例/ATR/VWAP 出场 | 提供现役基线 |
| **V2** | `last` | 动态资金、持仓上限、保护止盈、炸板和跌停顺延 | 验证账户与执行管理 |
| **V3** | `ask1` 买 / `bid1` 卖 | 当日预测、行情新鲜度、盘口确认、ATR 风险单位 | 验证可成交性与自适应退出 |
| **V4** | `ask1` 买 / `bid1` 卖 | V3 + 申万主行业精确映射 + 行业 ETF 弱势回避 | 验证板块确认的增量价值 |

> **零成交不等于零样本。** 每个版本都会记录模型池、排名、过滤原因、盘口、账户前后状态和卖出反事实；没有买入时仍可评估过滤规则是否有效。

---

## 📁 项目结构

```text
.
├── app.py                     # Streamlit 桌面端入口
├── mobile_app.py              # Streamlit 移动端入口
├── restart.sh                 # 🛠️ Docker 一站式管理脚本（详见下方）
├── docker-compose.yml         # 三服务编排：app / mobile / scheduler
├── Dockerfile                 # Web 镜像 (app / mobile)
├── Dockerfile.scheduler       # 训练与定时任务镜像
│
├── stock_analyzer/            # 📊 行情/指标/新闻/快照/预测展示层
│   ├── amazingdata_source.py  #   券商数据源（批量拉行情+复权因子）
│   ├── news_*.py              #   新闻采集/入库/标注/情绪
│   ├── top10_eval.py          #   固定候选组评估（白10/全A30/创新药10）
│   └── sectors.py all_a_meta.py  #  行业/概念板块映射
│
├── quant/                     # 🧠 因子/训练/回测/融合/晋级评估
│   ├── scheduled_workflow.py  #   ⭐ 生产总入口（拉数→训练→发布）
│   ├── full_train_batched.py  #   walk-forward 分批训练核心
│   ├── daily_update.py        #   行情增量更新
│   ├── backtest.py            #   组合回测与收益评估
│   ├── select.py              #   选股/组合构建
│   ├── tradability.py         #   可交易性过滤（涨跌停/停牌）
│   └── refresh_qfq.py         #   qfq 复权口径维护（除权票重拉）
│
├── realtime/                  # ⚡ 盘中实时策略层（交易时段常驻）
│   ├── README.md              #   现役策略、参数、审计与部署说明
│   ├── engine.py              #   主循环+生命周期（自管交易时段）
│   ├── feed.py                #   订阅流封装（L1 快照回调）
│   ├── rerank.py              #   ⭐ 盘中动态重排打分器（共享入口）
│   ├── rankboard.py           #   实时买入候选榜（推送 digest）
│   ├── paper_trader.py        #   实时模拟盘（持久化纸上账户+5级出场）
│   ├── v2.py                  #   V2 赛马账户（执行端优化，状态与 V1 隔离）
│   ├── v3.py                  #   V3 赛马账户（盘口确认+可成交价+ATR 出场）
│   ├── v4.py                  #   V4 赛马账户（V3 + 行业 ETF 弱势回避）
│   ├── sector_etf.py          #   申万主行业精确 ETF 映射、相对基准信号与快照状态
│   ├── strategy.py            #   信号策略骨架
│   ├── snapshot.py            #   Snapshot 数据模型+盘口微结构派生量
│   ├── reference.py           #   启动期静态基准（校准用）
│   ├── watchlist.py           #   订阅清单（固定候选组 ∪ 持仓）
│   ├── notifier.py            #   推送通道（Bark/Server酱/PushDeer）
│   └── ledger.py              #   独立账本（JSONL，与业务数据隔离）
│
└── docker/
    ├── scheduler_jobs.sh      #   ⭐ 所有 cron 任务的总入口
    ├── scheduler.crontab      #   定时表（COPY 烤进镜像，改后需 rebuild）
    └── scheduler-entrypoint.sh #  容器启动脚本
```

> 以下目录为**运行时生成/挂载的数据，不进 Git**：
> `quant_data/`（行情面板·训练缓存·模型·预测）、`snapshots/`（快照）、`logs/`（日志）、`.cache/`（请求缓存）、`sdk/`（私有券商 SDK wheel）。

---

## 🚀 快速开始

### 前置要求

- **Python 3.12**、**Docker Engine + Docker Compose**
- **Linux x86_64**（或 Apple Silicon 上用 `linux/amd64` 容器）
- 建议 **≥ 8 逻辑 CPU / 16GB 内存**（训练峰值内存较高）

### 1️⃣ 准备运行目录与配置

```bash
mkdir -p snapshots quant_data/full_a_2018_wide logs
cp .env.example .env   # 若无示例则手动创建，见下方「配置」
```

### 2️⃣ 一键构建并启动

```bash
./restart.sh build      # 构建 app+scheduler 镜像并后台启动
./restart.sh status     # 查看容器状态
```

### 3️⃣ 访问界面

| 服务 | 地址 | 说明 |
|---|---|---|
| 🖥️ 桌面端 | <http://localhost:8501> | 完整分析界面 |
| 📱 移动端 | <http://localhost:8502> | 只读快照精简版 |
| ⏰ scheduler | 无 HTTP 端口 | 容器内 cron 调度 |

> 远端 server 上经 SSH 隧道或 `服务器IP:8501` 访问。

---

## 🛠️ restart.sh — 一站式管理脚本

所有日常运维都收敛到 `./restart.sh`，运行 `./restart.sh help` 查看带示例的完整手册。

<details>
<summary><b>点击展开常用命令速查</b></summary>

```bash
# —— 生命周期 ——
./restart.sh up                  # 启动全部服务
./restart.sh restart scheduler   # 只重启 scheduler（改 .env 后）
./restart.sh stop / down         # 停止 / 停止并删容器（数据不丢）
./restart.sh status              # 容器状态

# —— 日志 ——
./restart.sh logs scheduler      # 跟踪定时任务/训练日志
./restart.sh rtlog               # 跟踪今天实时层引擎日志（自动脱敏）

# —— 盘中实时层 ——
./restart.sh realtime            # 幂等拉起引擎（已在跑则 SKIP，绝不起双引擎）
./restart.sh diag                # 只读诊断「今天模拟盘为什么没买入」

# —— 手动补跑定时任务 ——
./restart.sh run daily-light     # 快速日更（拉数+训练最近24窗+发布）
./restart.sh run weekly-full     # 完整周更（估值/财报/180天事件窗口）
./restart.sh run intraday-light  # 盘中轻量刷新
./restart.sh run snapshots       # 多维快照

# —— 改代码/依赖/crontab 后重建 ——
./restart.sh build scheduler     # 只重建 scheduler
```

</details>

> 💡 脚本自动探测 `docker compose`(v2) / `docker-compose`(v1)，本地 mac 与远端 Linux 通用。

---

## ⚙️ 配置

在项目根目录创建 `.env`（**已被 Git 忽略，切勿提交真实凭证**）：

```dotenv
# 可选：DeepSeek 新闻标注（无 Key 时降级为本地规则）
DASHSCOPE_API_KEY=
DASHSCOPE_MODEL=deepseek-v4-flash

# 可选：AmazingData 券商数据源（不用则走 AKShare）
AMAZINGDATA_USER=
AMAZINGDATA_PASSWORD=
AMAZINGDATA_HOST=
AMAZINGDATA_PORT=
AMAZINGDATA_AUTO_LOGIN=1

# 可选：实时层推送通道（任填其一）
BARK_KEY=              # iOS Bark
SERVERCHAN_KEY=        # Server 酱
PUSHDEER_KEY=          # PushDeer

# 容器内通常由 docker-compose 注入
SNAPSHOT_DIR=/app/snapshots
QUANT_DATA_DIR=/app/quant_data/full_a_2018_wide
QUANT_MODEL=active_quant
```

### 私有 SDK（可选）

使用 AmazingData 券商源需将以下 wheel 放入 `./sdk`（不随仓库分发）：

```text
sdk/tgw-1.0.8.7-py3-none-any.whl
sdk/AmazingData-1.1.7-cp312-none-any.whl
```

不使用时，需先在 Dockerfile 中移除 SDK 的 `COPY` 与 `pip install` 步骤；数据接入层会在 AmazingData 不可用时自动回退到其他数据源。

---

## 🧠 量化工作流

> ⚠️ **务必在容器内执行**：pandas / lightgbm 等量化依赖只装在 `stock-scheduler` 镜像里（Python 3.12）。
> 宿主机的系统 Python 没有这些包，直接跑会报 `ModuleNotFoundError: No module named 'pandas'`。

**生产总入口**（查看全部参数）：

```bash
# ✅ 容器内执行（-it 交互查看帮助）
docker exec -it a-scheduler-1 sh -lc 'cd /app && PYTHONPATH=/app python -m quant.scheduled_workflow --help'

# ❌ 不要在宿主机直接跑 python -m quant.scheduled_workflow —— 宿主机无 pandas
```

**日常无需直接调它**——`docker/scheduler_jobs.sh` 已把各任务的参数组合封装好，用 `restart.sh` 触发即可：

```bash
./restart.sh run daily-light      # = scheduled_workflow incumbent-refresh（快速日更）
./restart.sh run weekly-full      # 完整周更
./restart.sh run intraday-light   # 盘中轻量刷新
```

**两种策略模式**：

| 模式 | 用途 |
|---|---|
| `incumbent-refresh` | 用当前生产参数刷新最新滚动窗口、发布 active 预测（日常，`daily-light`/`weekly-full` 走它） |
| `candidate-upgrade` | 选择期选参 + 留出期验证，**过晋级门槛才允许升级**（审计，`monthly-factor` 走它） |

**手动跑一次完整日更**（等价 `./restart.sh run daily-light`，展开为容器内命令）：

```bash
docker exec -d a-scheduler-1 /app/docker/scheduler_jobs.sh daily-light
# 看进度：./restart.sh logs scheduler   或   docker exec a-scheduler-1 tail -f /app/logs/daily-light.out.log
```

**训练特性**：walk-forward 分窗训练，带 PIT 边界、标签 purge、窗口缓存、原子发布。生产模型默认融合 Ridge + LightGBM Ranker + ElasticNet + ExtraTrees。CatBoost 等候选模型须先作为**影子模型**通过独立 RankIC / 相关性 / 留出期收益 / 稳定性评估，才考虑接入生产。

---

## ⏰ 定时任务

容器时区 `Asia/Shanghai`，定义在 `docker/scheduler.crontab`：

| 时间 | 任务 | 说明 |
|---|---|---|
| 交易日 09:20 | `realtime` | 拉起盘中实时层引擎（常驻至收盘自动退出） |
| 交易日 09-14 每 10min | `realtime` | 🐕 **watchdog 自愈**：引擎被重训 OOM 挤死则自动重拉，保障 14:50 到期卖出与 14:50-14:55 买入 |
| 交易日 10:30 / 13:30 | `intraday-light` | 盘中轻量刷新（训练最近 6 窗） |
| 交易日 11:40 | `daily-light` | 盘中快速日更 + 短线/波段训练 + 发布 |
| 周一~周四 15:05 | `daily-light` | 收盘后快速日更 |
| 周五 15:05 | `weekly-full` | 完整周更（估值/财报/180天事件窗口） |
| 每日 16:40 / 17:00 | `news-daily` / `snapshots` | 新闻入库 / 多维快照 |
| 每月首周六 | `monthly-factor` / `refresh-qfq` | 滚动因子审计 / qfq 口径维护 |

> 所有量化重任务通过文件锁 (`/tmp/stock-quant-workflow.lock`) 避免并发。任务起止与耗时写入 `logs/cron.log`。

---

## 🧪 测试

> 同样**需在容器内跑**（依赖 pandas 等）。宿主机直接跑会 `ModuleNotFoundError`。

```bash
# 单元测试（按包名运行，避免 quant/select.py 与标准库 select 冲突）
docker exec -it a-scheduler-1 sh -lc 'cd /app && PYTHONPATH=/app python -m unittest \
  quant.test_model_expansion_experiment \
  quant.test_sentiment_ablation_experiment \
  quant.test_sentiment_feature_experiment \
  stock_analyzer.test_candidate_eval \
  stock_analyzer.test_top10_eval'

# 核心模块编译检查
docker exec -it a-scheduler-1 sh -lc 'cd /app && python -m py_compile \
  quant/model.py quant/full_train_batched.py \
  quant/scheduled_workflow.py stock_analyzer/candidate_eval.py'
```

> 若在**宿主机**做纯本地开发（非容器），需先 `pip install -r requirements-scheduler.txt` 装齐依赖，且宿主机 Python 版本需与镜像一致（3.12）方能完全对齐。

---

## 🔧 稳定性优化记录（生产踩坑归档）

<details>
<summary><b>券商批量接口超时与「进程中毒」</b></summary>

- **现象**：日更价格拉取卡在首批 `TimeoutError; 重登录后重试`，随后挂死不自愈。
- **根因**：`query_kline` 首批（~200 只）冷启约 24s，超过默认 `_SDK_TIMEOUT=15s`；超时后底层原生线程仍在后台崩溃，**污染整个进程**，同进程内重试必然继续挂死。
- **修复**：批量接口单独放宽超时——`_KLINE_TIMEOUT=90`、`_BROKER_TIMEOUT` 25→90、`_FACTOR_TIMEOUT` 40→120，走「足够长超时→失败→逐股兜底」正常路径。均可用环境变量覆盖。
</details>

<details>
<summary><b>独立入口脚本退出段错误 (rc=139)</b></summary>

- **现象**：`top10-eval` 长期 `rc=139`（Segmentation fault）被记为失败。
- **根因**：TGW 原生库在解释器退出/析构阶段段错误——实际评估已跑完、结果已落盘，属「数据已写、仅退出崩」的假失败。
- **修复**：`main()` 打印结果后 `flush` 并 `os._exit(0)`，跳过 native 析构干净退出。凡碰 TGW SDK 的独立入口均应如此收尾。
</details>

<details>
<summary><b>复权因子 HDF5 落地隔离</b></summary>

- **现象**：多次 `get_backward_factor` 共用同一 `.h5`，反复覆写导致文件损坏（`already opened` / `block0_items` 不一致）。
- **修复**：每次因子拉取使用独立临时子目录落地，避免并发/连续调用互相覆写。
</details>

<details>
<summary><b>实时层引擎被重训 OOM 连坐（watchdog 自愈）</b></summary>

- **现象**：引擎（内存很轻）被 10:30/13:30 盘中重训（峰值 ~9.5GB 撞 16GB 物理墙）连坐 OOM kill，导致当日 14:50 到期卖出与 14:50-14:55 买入腿缺失。
- **修复**：cron 交易时段每 10 分钟扫 `/proc`——引擎活着零副作用 SKIP，死了 10 分钟内自动重拉。分钟错开 :20 避免与 09:20 首拉起双引擎。
</details>

---

## 🔒 数据与安全边界

仓库**不包含**：行情/财务/新闻/快照/训练面板/预测产物、`.env` 与一切凭证、私有券商 SDK wheel、外部手册资料、生产日志与本地缓存。

- 实时层日志含券商凭证明文 → 查看时务必脱敏（`rtlog` 命令已内置 `grep -aviE "logon|token"`）。
- 首次部署需自行准备数据源访问条件与运行目录。
- 公开仓库前应再次执行凭证与漏洞扫描，并确认数据源、SDK、模型依赖符合各自授权条款。

---

## 📜 License

当前仓库尚未声明开源许可证，代码默认**保留全部权利**。若计划公开分发，请先选择合适许可证并确认第三方数据源与 SDK 授权兼容性。


