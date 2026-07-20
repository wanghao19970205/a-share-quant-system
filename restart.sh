#!/bin/zsh
# 股票分析系统 Docker 管理脚本
# 用法: ./restart.sh [命令]
#   (无参数) / up   后台启动 app + scheduler
#   down           停止并删除容器（挂载卷数据不丢）
#   stop           仅停止（保留容器）
#   restart        重启两个服务
#   status         查看容器状态
#   logs [服务]    跟踪日志，服务可选 app / scheduler（默认 app）
#   run [任务]     容器内手动补跑，任务可选 daily-light / weekly-full / snapshots（默认 daily-light）
#   build          重建镜像并后台启动（改了代码/依赖后用）
set -euo pipefail

cd "$(dirname "$0")"

# 确认 Docker 守护进程可用（Docker Desktop 未启动时给出提示）
_check_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker 未就绪：请先启动 OrbStack（或对应的 Docker 守护进程）后重试。"
    exit 1
  fi
}

cmd="${1:-up}"

case "$cmd" in
  up|"")
    _check_docker
    docker-compose up -d
    docker-compose ps
    echo "✅ 已后台启动。UI: http://localhost:8501"
    ;;
  down)
    _check_docker
    docker-compose down
    echo "✅ 已停止并删除容器（数据保留在挂载卷）。"
    ;;
  stop)
    _check_docker
    docker-compose stop
    echo "✅ 已停止（容器保留）。"
    ;;
  restart)
    _check_docker
    docker-compose restart app scheduler
    docker-compose ps
    echo "✅ 已重启。"
    ;;
  status)
    _check_docker
    docker-compose ps
    ;;
  logs)
    _check_docker
    svc="${2:-app}"
    docker-compose logs -f "$svc"
    ;;
  run)
    _check_docker
    job="${2:-daily-light}"
    case "$job" in
      daily-light|weekly-full|snapshots) ;;
      *) echo "❌ 未知任务: $job（可选 daily-light / weekly-full / snapshots）"; exit 2 ;;
    esac
    docker-compose exec -d scheduler /app/docker/scheduler_jobs.sh "$job"
    echo "✅ 已在容器内后台触发: $job"
    echo "   看进度: ./restart.sh logs scheduler  或  tail -f logs/${job}.out.log"
    ;;
  build)
    _check_docker
    docker-compose build app scheduler
    docker-compose up -d
    docker-compose ps
    echo "✅ 已重建并后台启动。"
    ;;
  *)
    echo "用法: ./restart.sh [up|down|stop|restart|status|logs [服务]|run [任务]|build]"
    exit 2
    ;;
esac
