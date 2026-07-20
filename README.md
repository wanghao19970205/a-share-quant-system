# A股分析与量化选股系统

一个面向A股的完整研究与生产运行项目，包含Streamlit分析界面、移动端界面、行情与新闻快照、滚动因子训练、模型融合、候选晋级评估和容器内定时调度。

> 本项目仅用于数据分析和量化研究，不构成投资建议。历史回测和模型评分不代表未来收益。

## 功能概览

- A股日线、估值、财务、资金流、事件和新闻数据采集
- 技术指标、外围市场、板块关联和个股新闻分析
- Qwen新闻结构化标注；未配置API Key时降级为本地规则
- 主板活跃股票池的滚动walk-forward训练
- Ridge、LightGBM Ranker、ElasticNet和ExtraTrees模型融合
- 可选CatBoost Ranker影子模型与统一晋级评估
- 短线和波段预测发布、Top10跟踪评估
- Streamlit桌面端和移动端界面
- Docker Compose生产部署和cron定时任务

## 项目结构

```text
.
├── app.py                         # Streamlit桌面端入口
├── mobile_app.py                  # Streamlit移动端入口
├── stock_analyzer/                # 行情、指标、新闻、快照和预测展示
├── quant/                         # 因子、训练、回测、融合和晋级评估
├── docker/                        # scheduler入口、任务脚本和crontab
├── Dockerfile                     # Web镜像
├── Dockerfile.scheduler           # 训练与定时任务镜像
├── docker-compose.yml
├── requirements-web.txt           # Web侧依赖
├── requirements-scheduler.txt     # scheduler依赖
├── requirements.txt               # 本地完整研究环境依赖
└── restart.sh                     # Docker管理脚本
```

以下目录是运行时生成或挂载的数据，不进入Git：

```text
quant_data/                        # 行情面板、训练缓存、模型和预测
snapshots/                         # 股票、新闻、行业和评估快照
logs/                              # cron与工作流日志
.cache/                            # 本地请求缓存
```

## 环境要求

- Python 3.12
- Docker Engine和Docker Compose
- Linux x86_64，或在Apple Silicon上使用`linux/amd64`容器
- 建议至少8个逻辑CPU、16GB内存

数据源以AKShare公开接口为基础。项目也支持可选的AmazingData券商数据源，但相关SDK wheel不包含在仓库中。

## 私有SDK

`Dockerfile`和`Dockerfile.scheduler`默认从本地`./sdk`安装以下文件：

```text
sdk/tgw-1.0.8.7-py3-none-any.whl
sdk/AmazingData-1.1.7-cp312-none-any.whl
```

这些文件属于外部二进制依赖，不随仓库分发。使用AmazingData时，请从合法来源取得对应wheel并放入`./sdk`。如果不使用该数据源，需要先调整Dockerfile，移除SDK的`COPY`和`pip install`步骤；系统的数据接入代码会在AmazingData不可用时使用其他数据源。

## 配置

在项目根目录创建`.env`。该文件已被Git忽略，不要提交真实账号、密码或API Key。

常用环境变量：

```dotenv
# 可选：Qwen新闻标注
DASHSCOPE_API_KEY=
DASHSCOPE_MODEL=qwen-plus

# 可选：AmazingData券商数据源
AMAZINGDATA_USER=
AMAZINGDATA_PASSWORD=
AMAZINGDATA_HOST=
AMAZINGDATA_PORT=
AMAZINGDATA_AUTO_LOGIN=1

# 可选：代理
A_PROXIES=

# 容器内通常由docker-compose设置
SNAPSHOT_DIR=/app/snapshots
QUANT_DATA_DIR=/app/quant_data/full_a_2018_wide
QUANT_MODEL=active_quant
```

## Docker运行

准备运行目录：

```bash
mkdir -p snapshots quant_data/full_a_2018_wide logs
```

构建并启动全部服务：

```bash
docker compose build app scheduler
docker compose up -d
```

默认服务：

- 桌面端：<http://localhost:8501>
- 移动端：<http://localhost:8502>
- scheduler：容器内cron调度，无HTTP端口

常用命令：

```bash
./restart.sh status
./restart.sh logs app
./restart.sh logs scheduler
./restart.sh run daily-light
./restart.sh run weekly-full
./restart.sh run snapshots
./restart.sh build
./restart.sh down
```

也可以直接使用Docker Compose：

```bash
docker compose up -d app mobile scheduler
docker compose logs -f scheduler
docker compose exec scheduler /app/docker/scheduler_jobs.sh daily-light
```

## 本地运行

仅运行Web界面：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
streamlit run app.py
```

移动端：

```bash
streamlit run mobile_app.py --server.port 8502
```

完整量化研究环境：

```bash
pip install -r requirements.txt
```

scheduler容器使用独立依赖集：

```bash
pip install -r requirements-scheduler.txt
```

## 量化工作流

生产入口为：

```bash
python -m quant.scheduled_workflow --help
```

主要策略模式：

- `incumbent-refresh`：使用当前生产参数刷新最新滚动窗口并发布active预测
- `candidate-upgrade`：在选择期选参、留出期验证，通过晋级门槛后才允许升级

手工触发日更：

```bash
docker compose exec scheduler /app/docker/scheduler_jobs.sh daily-light
```

完整周更：

```bash
docker compose exec scheduler /app/docker/scheduler_jobs.sh weekly-full
```

模型训练按walk-forward窗口执行，并使用PIT边界、标签purge、窗口缓存和原子发布。生产模型默认包含Ridge、LightGBM Ranker、ElasticNet和ExtraTrees。CatBoost及其他候选模型应先作为影子模型完成独立RankIC、相关性、留出期收益和稳定性评估，再考虑接入生产。

## 定时任务

容器内时区为`Asia/Shanghai`。主要任务定义在`docker/scheduler.crontab`：

- 交易日10:30、13:30：盘中轻量刷新
- 交易日11:40：日内生产刷新
- 周一至周四15:40：收盘日更
- 周五15:40：完整周更
- 每日16:40：新闻增量入库和最新新闻标注
- 每日17:00：多维快照
- 每10分钟：历史新闻标注回填
- 每月：因子候选晋级与舆情模型选择

所有量化重任务通过文件锁避免相互并发。任务开始、结束和耗时写入`logs/cron.log`及对应任务日志。

## 测试

按包名运行当前测试，避免`quant/select.py`与Python标准库`select`在部分环境中的模块名冲突：

```bash
python3 -m unittest \
  quant.test_model_expansion_experiment \
  quant.test_sentiment_ablation_experiment \
  quant.test_sentiment_feature_experiment \
  stock_analyzer.test_candidate_eval \
  stock_analyzer.test_top10_eval
```

执行核心模块编译检查：

```bash
python3 -m py_compile \
  quant/model.py \
  quant/full_train_batched.py \
  quant/scheduled_workflow.py \
  stock_analyzer/candidate_eval.py
```

## 数据与安全边界

仓库不包含：

- 行情、财务、新闻、快照、训练面板和预测产物
- `.env`、账号密码、API Key、私钥和证书
- 私有券商SDK及其wheel
- 外部ZIP、PDF、HTML手册和供应商资料
- 生产日志和本地缓存

首次部署需要自行准备数据源访问条件和运行目录。公开仓库前应再次执行凭证与漏洞扫描，并确认所使用的数据源、SDK和模型依赖符合其授权条款。

## License

当前仓库尚未声明开源许可证。在添加明确许可证前，代码默认保留全部权利。若计划公开分发，请先选择适合的许可证并确认第三方数据源和SDK授权兼容性。
