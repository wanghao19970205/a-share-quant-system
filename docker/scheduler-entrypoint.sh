#!/bin/sh
# 定时任务容器入口：物化 cron 运行环境、设置时区，然后前台启动 cron。
set -e

mkdir -p /app/logs

# cron 执行时环境极简，这里把运行所需变量固化到 /app/cron.env（含券商账号，权限 600）。
umask 077
{
  for k in QUANT_DATA_DIR SNAPSHOT_DIR SNAPSHOT_BATCH QUANT_MODEL PYTHONPATH LD_LIBRARY_PATH TZ LOG_DIR \
           SCHEDULER_DISABLED INTRADAY_1400_DATA_DIR INTRADAY_1400_MINUTE_BATCH_SIZE \
           INTRADAY_1400_PARTITION_SIZE INTRADAY_1400_MINUTE_TIMEOUT INTRADAY_1400_FEATURE_WORKERS \
           INTRADAY_1400_PIPELINE_WORKERS \
           DASHSCOPE_API_KEY DASHSCOPE_BASE_URL DASHSCOPE_BATCH_BASE_URL DASHSCOPE_MODEL \
           DASHSCOPE_MODELS DASHSCOPE_MODEL_LIST DASHSCOPE_FALLBACK_MODEL \
           AMAZINGDATA_USER AMAZINGDATA_PASSWORD AMAZINGDATA_HOST AMAZINGDATA_PORT AMAZINGDATA_AUTO_LOGIN; do
    eval v="\${$k:-}"
    printf "export %s='%s'\n" "$k" "$v"
  done
} > /app/cron.env

# 时区：让 cron 的调度时刻按 Asia/Shanghai 生效
if [ -n "${TZ:-}" ] && [ -f "/usr/share/zoneinfo/$TZ" ]; then
  ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime
  echo "$TZ" > /etc/timezone
fi

# 可选：启动即补跑一次快速日更（默认关闭，避免每次重启都触发训练）
if [ "${RUN_ON_START:-0}" = "1" ]; then
  echo "[scheduler] RUN_ON_START=1, kick daily-light once"
  /app/docker/scheduler_jobs.sh daily-light || true
fi

echo "[scheduler] cron starting, TZ=${TZ:-UTC}"

# 启动足迹写入中心日志：记录容器/cron 启动时刻与生效的任务条目，
# 便于事后核对“cron 是否在跑、加载了哪些任务”。
CRON_LOG="${LOG_DIR:-/app/logs}/cron.log"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== scheduler container start, TZ=${TZ:-UTC} ====="
  echo "---- effective /etc/cron.d/stock-scheduler ----"
  grep -vE '^[[:space:]]*(#|$)' /etc/cron.d/stock-scheduler 2>/dev/null || true
  echo "------------------------------------------------"
} >> "$CRON_LOG" 2>&1

# -L 15 提高日志级别；无 syslog 时任务触发以 cron.d 心跳 + 各任务 clog 为准。
exec cron -f -L 15
