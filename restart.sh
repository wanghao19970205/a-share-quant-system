#!/bin/zsh
# 股票分析系统 Docker 管理脚本（本地 mac / 远端 Linux server 通用）
# 用法: ./restart.sh [命令]
#   (无参数) / up [服务]  后台启动（默认 app+mobile+scheduler；可指定单个服务名）
#   down                停止并删除容器（挂载卷数据不丢）
#   stop  [服务]        仅停止（保留容器）
#   restart [服务]      重启服务（默认 app scheduler）
#   status              查看容器状态
#   logs [服务]         跟踪日志：app / mobile / scheduler（默认 app）
#   rtlog               跟踪今天的实时层引擎日志（logs/realtime/engine.YYYYMMDD.log，自动脱敏）
#   run [任务]          容器内手动补跑 scheduler_jobs.sh 的任务（见下方 JOBS）
#   realtime            幂等拉起盘中实时层引擎（内部扫 /proc 防重复，绝不起双引擎）
#   diag                只读诊断：模拟盘今天为什么没买入（若存在 check_paper.sh）
#   build [服务]        重建镜像并后台启动（改了代码/依赖/crontab 后用；默认 app scheduler）
set -euo pipefail

cd "$(dirname "$0")"

# scheduler_jobs.sh 支持的全部任务（run 子命令校验用）
JOBS="daily-light weekly-full intraday-light snapshots news-daily refresh-qfq monthly-factor all-a-meta sentiment-model rotate realtime-weight-shadow realtime"

# 探测 compose 命令：优先 docker compose(v2，远端)，回退 docker-compose(v1，本地旧环境)
# 用数组存放，避免 zsh 默认不对未加引号变量做单词分割导致 "docker compose" 被当成单个命令名。
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose)
else
  echo "❌ 未找到 docker compose(v2) 或 docker-compose(v1)。"; exit 1
fi

# 确认 Docker 守护进程可用（本地 Docker Desktop/OrbStack 未启动，或远端 dockerd 未运行时提示）
_check_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker 未就绪：请先确认守护进程在运行（本地启动 OrbStack/Docker Desktop；远端 systemctl start docker）。"
    exit 1
  fi
}

cmd="${1:-up}"

case "$cmd" in
  up|"")
    _check_docker
    "${DC[@]}" up -d ${2:+"$2"}
    "${DC[@]}" ps
    echo "✅ 已后台启动。UI: http://localhost:8501 （远端经 SSH 隧道或服务器 IP:8501 访问）"
    ;;
  down)
    _check_docker
    "${DC[@]}" down
    echo "✅ 已停止并删除容器（数据保留在挂载卷）。"
    ;;
  stop)
    _check_docker
    "${DC[@]}" stop ${2:+"$2"}
    echo "✅ 已停止（容器保留）。"
    ;;
  restart)
    _check_docker
    "${DC[@]}" restart ${2:-app scheduler}
    "${DC[@]}" ps
    echo "✅ 已重启。"
    ;;
  status)
    _check_docker
    "${DC[@]}" ps
    ;;
  logs)
    _check_docker
    svc="${2:-app}"
    "${DC[@]}" logs -f "$svc"
    ;;
  rtlog)
    # 跟踪今天的实时层引擎日志；脱敏(隐去 logon/token 明文凭证)。容器名固定 a-scheduler-1。
    _check_docker
    CT="${CT:-a-scheduler-1}"
    RTLOG="/app/logs/realtime/engine.$(date '+%Y%m%d').log"
    echo "== 跟踪 $CT:$RTLOG （Ctrl-C 退出，已脱敏） =="
    docker exec "$CT" sh -lc "[ -f '$RTLOG' ] && tail -f '$RTLOG' || echo '今天还没有引擎日志(引擎未起/非交易时段)'" \
      | grep -aviE "logon|token" || true
    ;;
  run)
    _check_docker
    job="${2:-daily-light}"
    case " $JOBS " in
      *" $job "*) ;;
      *) echo "❌ 未知任务: $job"; echo "   可选: $JOBS"; exit 2 ;;
    esac
    "${DC[@]}" exec -d scheduler /app/docker/scheduler_jobs.sh "$job"
    echo "✅ 已在容器内后台触发: $job"
    echo "   看进度: ./restart.sh logs scheduler   或   tail -f logs/${job}.out.log"
    ;;
  realtime)
    # 幂等拉起盘中实时层引擎：scheduler_jobs.sh realtime 内部扫 /proc 防重复(引擎活着则 SKIP)，
    # 绝不起第二个 engine(双引擎会抢券商连接+双份推送)。交易日 cron 09:20 已自动拉起，此处供手动补拉。
    _check_docker
    "${DC[@]}" exec -d scheduler /app/docker/scheduler_jobs.sh realtime
    echo "✅ 已触发实时层拉起(幂等：引擎已在跑则容器内自动 SKIP)。"
    echo "   看引擎日志: ./restart.sh rtlog"
    ;;
  diag)
    # 只读诊断：今天模拟盘为什么没买入(逐层取证)。依赖根目录 check_paper.sh。
    _check_docker
    if [ -x ./check_paper.sh ]; then
      bash ./check_paper.sh
    else
      echo "❌ 未找到可执行的 check_paper.sh（只读诊断脚本）。"; exit 2
    fi
    ;;
  build)
    _check_docker
    "${DC[@]}" build ${2:-app scheduler}
    "${DC[@]}" up -d
    "${DC[@]}" ps
    echo "✅ 已重建并后台启动。"
    echo "   ⚠️ 若改了 docker/scheduler.crontab：crontab 经 Dockerfile COPY 烤进镜像，build 已使其生效。"
    echo "   ⚠️ rebuild 会刷新 prepared_monthly mtime → 当晚日更可能一次性全窗重训(一次性代价)。"
    ;;
  help|-h|--help)
    cat <<'HELP'
====================================================================
 restart.sh — A 股量化系统 Docker 管理（本地 mac / 远端 server 通用）
 远端实操目录: /home/work/wanghao81/A/a-share-quant-system-main
 远端容器: a-scheduler-1(scheduler) / stock-app(app,mobile)
====================================================================

【生命周期】
  ./restart.sh up               后台启动全部服务(app+mobile+scheduler)
  ./restart.sh up scheduler     只启 scheduler
  ./restart.sh status           查看容器状态(等价 docker compose ps)
  ./restart.sh restart          重启 app+scheduler(默认)
  ./restart.sh restart scheduler  只重启 scheduler(改 .env/环境变量后用)
  ./restart.sh stop             停止全部(保留容器)
  ./restart.sh stop scheduler   只停 scheduler
  ./restart.sh down             停止并删除容器(挂载卷数据不丢)

【日志】
  ./restart.sh logs             跟踪 app 日志(默认)
  ./restart.sh logs scheduler   跟踪 scheduler(定时任务/训练)日志
  ./restart.sh logs mobile      跟踪 mobile(8502)日志
  ./restart.sh rtlog            跟踪今天实时层引擎日志(自动脱敏隐去凭证)
                                等价 docker exec a-scheduler-1 tail -f \
                                  /app/logs/realtime/engine.YYYYMMDD.log
  CT=其它容器名 ./restart.sh rtlog   本地容器名不同时覆盖(默认 a-scheduler-1)

【盘中实时层】
  ./restart.sh realtime         幂等拉起引擎(内部扫 /proc,已在跑则 SKIP,绝不起双引擎)
                                cron 交易日 09:20 已自动拉起+每10min watchdog 自愈,
                                此命令供手动补拉(如确认引擎意外死亡后)
  ./restart.sh diag             只读诊断"今天模拟盘为什么没买入"(逐层取证,调 check_paper.sh)

【手动补跑定时任务】(容器内后台执行,不阻塞)
  ./restart.sh run daily-light      快速日更(拉数+训练最近24窗+发布)  ← 默认
  ./restart.sh run weekly-full      完整日更(估值/财报/180天事件窗口)
  ./restart.sh run intraday-light   盘中轻量刷新(只拉快数据+训练最近6窗)
  ./restart.sh run snapshots        多维快照(新闻/情绪)
  ./restart.sh run news-daily       白名单新闻增量入库
  ./restart.sh run refresh-qfq      qfq 口径维护(除权票整条重拉)
  ./restart.sh run monthly-factor   滚动因子审计(36窗)
  ./restart.sh run all-a-meta       刷新全A行业/概念板块映射
  ./restart.sh run sentiment-model  重选舆情衰减/类别权重
  ./restart.sh run rotate           日志清理(超阈值 .log 截尾)
     进度: ./restart.sh logs scheduler  或  tail -f logs/<任务>.out.log

【改代码/依赖/crontab 后重建】
  ./restart.sh build            重建 app+scheduler 镜像并启动(默认)
  ./restart.sh build scheduler  只重建 scheduler
     ⚠️ docker/scheduler.crontab 经 Dockerfile COPY 烤进镜像 → 改 crontab 必须 build 才生效
     ⚠️ build 刷新 prepared_monthly mtime → 当晚日更可能一次性全窗重训(一次性代价)
     ⚠️ build 前先确认无训练在跑,否则会腰斩持锁的 full_train_batched/scheduled_workflow

【常用组合】
  改了 realtime/*.py 想立即生效:
     ./restart.sh build scheduler && ./restart.sh rtlog
  收盘后看今天有没有买入:
     ./restart.sh diag
  引擎被重训 OOM 挤死了手动补拉(平时靠 cron watchdog 自愈):
     ./restart.sh realtime && ./restart.sh rtlog
====================================================================
HELP
    ;;
  *)
    echo "用法: ./restart.sh [up|down|stop|restart|status|logs [服务]|rtlog|run [任务]|realtime|diag|build [服务]|help]"
    echo "  任务(run): $JOBS"
    echo "  详细用法+示例: ./restart.sh help"
    exit 2
    ;;
esac

