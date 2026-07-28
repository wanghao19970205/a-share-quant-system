#!/bin/sh
# 容器内定时任务的统一入口：拉数 -> 训练 -> 发布 / 快照。
# cron 环境变量极简，这里从 /app/cron.env 载入运行所需变量。
#
# 关键：**不使用 set -e**。任务进程非零退出（含 AmazingData SDK 在退出 teardown
# 时的 SIGSEGV）也必须继续记录 rc，否则 set -e 会在 python 崩溃时提前中止脚本，
# 连 "done rc=" 都写不出来，导致“到底跑没跑、成功没成功”无从查证。
set -u

if [ -f /app/cron.env ]; then
    . /app/cron.env
fi

# 真实交易成本（roundtrip）：佣金双边 万5×2=0.001 + 印花税卖出 千1=0.001 = 0.002。
# 接入选参/晋级门/训练内回测，使指标反映真实可交易成本（此前默认 0=乐观口径）。
# 2026-07-25 实证：成本进来后月度候选逐月胜率 0.4286→~0.30，晋级判定 True→False，确认成本影响结果。
# 用 ${VAR:-0.002} 形式：默认 0.002，运行时可用 -e QUANT_BT_COST_ROUNDTRIP=0 复现乐观基线、
# 或 =0.0025 跑含滑点的悲观口径。一处生效，覆盖 monthly-factor/daily-light/intraday-light/weekly-full。
export QUANT_BT_COST_ROUNDTRIP="${QUANT_BT_COST_ROUNDTRIP:-0.002}"
cd /app

LOG_DIR="${LOG_DIR:-/app/logs}"
mkdir -p "$LOG_DIR"
CRON_LOG="$LOG_DIR/cron.log"          # 中心日志：所有任务的触发/结束都汇总到这里
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/app/snapshots}"
# 新闻库：复用已挂载的 snapshots 卷（宿主机 ./snapshots/news_data），无需新增挂载即可持久化。
NEWS_DIR="${NEWS_DIR:-/app/snapshots/news_data}"
export NEWS_DIR

ts() { date '+%Y-%m-%d %H:%M:%S'; }
clog() { echo "[$(ts)] $*" >> "$CRON_LOG"; }
format_duration() {
    total="${1:-0}"
    printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

# 月度面板缓存：内存越大可设越大（缓存命中率高、训练更快）。
# 12GB 内存 + 36 月训练窗口下：一个窗口本身常驻 36 个月，cache 只需再留几个月供
# 相邻窗口复用即可；设过大（如 52）会白占 ~280MB 峰值内存。故降到 40。
MONTH_CACHE_SIZE="${MONTH_CACHE_SIZE:-64}"

# 滚动训练窗口长度（月）：每个独立模型看多少历史。峰值内存只由单个窗口决定，
# 相邻窗口串行且共享 month cache，故加长窗口在 12GB 下仍安全（全量扩张才会 OOM）。
# 如遇 OOM(SIGKILL) 可用环境变量调小。
TRAIN_MONTHS="${TRAIN_MONTHS:-36}"

job="${1:-}"
job_started_epoch=$(date +%s)
job_started_at=$(ts)
job_end_logged=0
monthly_lock_dir=""
clog "FIRE   job=$job pid=$$ started_at=$job_started_at"

log_job_end() {
    rc="$1"
    [ "$job_end_logged" -eq 0 ] || return 0
    job_end_logged=1
    ended_epoch=$(date +%s)
    duration_sec=$((ended_epoch - job_started_epoch))
    duration=$(format_duration "$duration_sec")
    clog "END    job=$job rc=$rc duration_sec=$duration_sec duration=$duration"
}

on_exit() {
    rc=$?
    trap - EXIT INT TERM
    if [ -n "$monthly_lock_dir" ]; then
        rmdir "$monthly_lock_dir" 2>/dev/null || true
    fi
    log_job_end "$rc"
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# run_job <label> <out.log> <err.log> <命令...>
run_job() {
    label="$1"; out="$2"; err="$3"; shift 3
    stage_started_epoch=$(date +%s)
    echo "[$(ts)] start $label (month_cache=$MONTH_CACHE_SIZE)" >> "$out"
    "$@" >> "$out" 2>> "$err"
    rc=$?
    stage_ended_epoch=$(date +%s)
    stage_duration_sec=$((stage_ended_epoch - stage_started_epoch))
    stage_duration=$(format_duration "$stage_duration_sec")
    echo "[$(ts)] done $label rc=$rc duration_sec=$stage_duration_sec duration=$stage_duration" >> "$out"
    clog "STAGE  job=$job stage=$label rc=$rc duration_sec=$stage_duration_sec duration=$stage_duration"
    return "$rc"
}

acquire_quant_lock() {
    exec 6>/tmp/stock-quant-workflow.lock
    if ! flock -n 6; then
        clog "SKIP   job=$job reason=quant-workflow-running"
        exit 0
    fi
}

case "$job" in
  intraday-light)
    # 10:30/13:30 轻量盘中刷新：只回看当天、跳过慢数据源，并重训最近 6 个预测窗口。
    acquire_quant_lock
    run_job intraday-light "$LOG_DIR/intraday-light.out.log" "$LOG_DIR/intraday-light.err.log" \
        python -m quant.scheduled_workflow \
            --universe mainboard_active --update-workers 12 \
            --lookback-days 1 --event-window-days 1 \
            --skip-events --skip-valuation --skip-fundamentals --skip-snapshots \
            --intraday-spot \
            --strategy-mode incumbent-refresh \
            --skip-swing-grid \
            --model-threads 12 \
            --recent-windows 6 \
            --train-months "$TRAIN_MONTHS" \
            --month-cache-size "$MONTH_CACHE_SIZE" \
            --snapshot-dir "$SNAPSHOT_DIR"
    workflow_rc=$?
    if [ "$workflow_rc" -eq 0 ]; then
        TOP10_SOURCE_JOB=intraday-light "$0" top10-eval
    else
        exit "$workflow_rc"
    fi
    ;;
  daily-light)
    acquire_quant_lock
    # 盘中/收盘快速日更：滚动重训「最新 24 个预测窗口」（每个窗口 TRAIN_MONTHS 个月训练集，逐窗独立拟合）。
    # 峰值内存受单窗口约束，12GB 环境安全；相邻窗口共享 month cache 避免重复读盘。
    # 新窗口预测按 code+date 合并进历史 active，波段冠军保持冻结。
    run_job daily-light "$LOG_DIR/daily-light.out.log" "$LOG_DIR/daily-light.err.log" \
        python -m quant.scheduled_workflow \
            --universe mainboard_active --update-workers 12 \
            --lookback-days 5 --event-window-days 30 \
            --skip-valuation --skip-fundamentals --skip-snapshots \
            --force-latest-price \
            --strategy-mode incumbent-refresh \
            --skip-swing-grid \
            --model-threads 12 \
            --recent-windows 24 \
            --train-months "$TRAIN_MONTHS" \
            --month-cache-size "$MONTH_CACHE_SIZE" \
            --snapshot-dir "$SNAPSHOT_DIR"
    workflow_rc=$?
    if [ "$workflow_rc" -eq 0 ]; then
        TOP10_SOURCE_JOB=daily-light "$0" top10-eval
    else
        exit "$workflow_rc"
    fi
    ;;
  weekly-full)
    acquire_quant_lock
    # 每周五完整日更（含估值/财报/180天事件窗口），训练/发布语义同 daily-light。
    run_job weekly-full "$LOG_DIR/weekly-full.out.log" "$LOG_DIR/weekly-full.err.log" \
        python -m quant.scheduled_workflow \
            --universe mainboard_active --update-workers 12 \
            --lookback-days 5 --event-window-days 180 \
            --skip-snapshots \
            --strategy-mode incumbent-refresh \
            --skip-swing-grid \
            --model-threads 12 \
            --recent-windows 24 \
            --train-months "$TRAIN_MONTHS" \
            --month-cache-size "$MONTH_CACHE_SIZE" \
            --snapshot-dir "$SNAPSHOT_DIR"
    workflow_rc=$?
    if [ "$workflow_rc" -eq 0 ]; then
        "$0" top10-eval
    else
        exit "$workflow_rc"
    fi
    ;;
  monthly-factor)
    acquire_quant_lock
    LOCK_DIR="/tmp/stock-monthly-factor.lock"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        clog "SKIP   job=monthly-factor reason=already-running"
        exit 0
    fi
    monthly_lock_dir="$LOCK_DIR"
    run_job monthly-factor "$LOG_DIR/monthly-factor.out.log" "$LOG_DIR/monthly-factor.err.log" \
        python -m quant.scheduled_workflow \
            --skip-daily-update \
            --strategy-mode candidate-upgrade --short-only-upgrade \
            --output-prefix monthly_factor_candidate \
            --refresh-months 1 --recent-windows 36 \
            --train-months "$TRAIN_MONTHS" \
            --n-estimators 200 --learning-rate 0.015 --early-stopping-rounds 40 \
            --model-threads 12 \
            --lgbm-weight 0.85 --rank-vote-weight 0.0 \
            --decay-half-life-days 60 --min-weight 0.03 \
            --rolling-top-factors 30 --max-factor-ic-corr 0.85 \
            --promotion-holdout-months 6 \
            --promotion-min-sharpe-gain 0.10 \
            --promotion-max-drawdown-worsening 0.02 \
            --month-cache-size "$MONTH_CACHE_SIZE" \
            --snapshot-dir "$SNAPSHOT_DIR"
    ;;
  refresh-qfq)
    # 月度 qfq 口径维护：检测最近 window-days 内除权(后复权因子跳变)的票，整条重拉覆盖。
    # 增量日更只重写最近几行，除权票历史 qfq base 会与新行分裂——这里把这些票整条刷新。
    # CACHE_TTL_KLINE=0 关行情磁盘缓存，确保拿到最新价而非缓存的旧口径。
    acquire_quant_lock
    QFQ_WINDOW_DAYS="${QFQ_WINDOW_DAYS:-35}"
    run_job refresh-qfq "$LOG_DIR/refresh-qfq.out.log" "$LOG_DIR/refresh-qfq.err.log" \
        env CACHE_TTL_KLINE=0 \
        python -m quant.refresh_qfq \
            --mode ex-div --universe mainboard_active \
            --workers 4 --window-days "$QFQ_WINDOW_DAYS"
    ;;
  snapshots)
    run_job snapshots "$LOG_DIR/snapshots-daily.out.log" "$LOG_DIR/snapshots-daily.err.log" \
        python -m stock_analyzer.snapshot_batch --batch
    ;;
  news-daily)
    # 仅做新闻增量入库（纯 akshare 抓取，不费 token）。
    # [停用 Qwen 标注 2026-07-28] 舆情/新闻因子经 ablation 判负(sentiment-factor-ablation-negative)，
    #   不进综合分、不改选股，故日更后的 news_annotation --newest（Qwen 标注最新新闻）取消，省 LLM token。
    #   入库仍保留：建库/UI 即时 news.analyze 用户触发时可读原始新闻，不受影响。
    run_job news-daily "$LOG_DIR/news-daily.out.log" "$LOG_DIR/news-daily.err.log" \
        python -m stock_analyzer.news_ingest \
            --mode daily --lookback-days 7 --workers 12
    ;;
  news-annotation)
    # 历史回填必须单实例运行：手工触发、cron 与日更不能并发消耗 Qwen 额度或抢同一批记录。
    # 与手工大批量回溯共用同一个 flock 文件锁；进程退出时内核自动释放，无需删除锁文件。
    LOCK_FILE="$SNAPSHOT_DIR/.stock-news-annotation.lock"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        clog "SKIP   job=news-annotation reason=already-running"
        exit 0
    fi
    # 当前凭据不开放 DashScope Files/Batch API（403），使用已验证的实时兼容接口小批回填。
    run_job news-annotation "$LOG_DIR/news-annotation.out.log" "$LOG_DIR/news-annotation.err.log" \
        python -m stock_analyzer.news_annotation --mode realtime --limit 20
    ;;
  all-a-meta)
    # 东财全 A 行业/概念板块映射：低并发抓取，失败板块沿用上次成功数据。
    LOCK_FILE="/tmp/stock-all-a-meta.lock"
    exec 8>"$LOCK_FILE"
    if ! flock -n 8; then
        clog "SKIP   job=all-a-meta reason=already-running"
        exit 0
    fi
    run_job all-a-meta "$LOG_DIR/all-a-meta.out.log" "$LOG_DIR/all-a-meta.err.log" \
        python -m stock_analyzer.all_a_meta --workers 1 --retries 3 --delay 0.5
    ;;
  top10-eval)
    # 固定 Top10 在量化发布后统一做一次大模型评估；输入指纹不变时幂等跳过。
    LOCK_FILE="/tmp/stock-top10-eval.lock"
    exec 7>"$LOCK_FILE"
    if ! flock -n 7; then
        clog "SKIP   job=top10-eval reason=already-running"
        exit 0
    fi
    run_job top10-eval "$LOG_DIR/top10-eval.out.log" "$LOG_DIR/top10-eval.err.log" \
        python -m stock_analyzer.top10_eval
    ;;
  sentiment-model)
    # 月度只重估舆情衰减/类别权重，不重新回填新闻；留出验证不过门槛则自动保持展示但不加分。
    run_job sentiment-model "$LOG_DIR/sentiment-model.out.log" "$LOG_DIR/sentiment-model.err.log" \
        python -m stock_analyzer.sentiment_signal --months 12
    ;;
  realtime)
    # 交易日盘中实时层：常驻 python -m realtime.engine，订阅选股清单∪持仓的 Level-1 快照，
    # 异动/买卖点/持仓到期 → 推送(Bark/Server酱/PushDeer) + 独立账本(logs/realtime)。
    # 引擎自管时段：<session_start 休眠、午休静默、>session_end(15:05) 自动退出，故每交易日
    # 一拉起、收盘自然结束。这里【只负责幂等拉起 + 后台运行】，不阻塞 cron。
    #
    # 防重复：engine 是长驻进程，cron 若补跑重复触发绝不能起第二个（会双份推送 + 抢券商连接）。
    # 扫 /proc 找已在跑的 realtime.engine（先按 comm=python* 过滤，再匹配 cmdline），有则跳过。
    RT_LOG="$LOG_DIR/realtime/engine.$(date '+%Y%m%d').log"
    mkdir -p "$LOG_DIR/realtime"
    # 账本 + 推送凭证 env 文件都落到已挂载的 /app/logs/realtime（宿主机 ./logs/realtime，持久化）。
    # 否则 config 默认推导出 /app/quant_data/logs/realtime —— 该路径不在任何挂载卷，docker rm 即丢，
    # 且与已测通的推送配置(notify.env)所在处不一致。显式 export 覆盖，指向挂载盘。
    REALTIME_LEDGER_DIR="${REALTIME_LEDGER_DIR:-$LOG_DIR/realtime}"
    REALTIME_ENV_FILE="${REALTIME_ENV_FILE:-$LOG_DIR/realtime/notify.env}"
    export REALTIME_LEDGER_DIR REALTIME_ENV_FILE
    RT_BUSY=""
    for d in /proc/[0-9]*; do
        comm=$(cat "$d/comm" 2>/dev/null) || continue
        case "$comm" in python*) ;; *) continue ;; esac
        cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        case "$cl" in
          *realtime.engine*) RT_BUSY="PID=$(basename "$d") $RT_BUSY" ;;
        esac
    done
    if [ -n "$RT_BUSY" ]; then
        clog "SKIP   job=realtime reason=already-running $RT_BUSY"
    else
        clog "START  job=realtime -> nohup python -m realtime.engine (log=$RT_LOG)"
        # 后台常驻；日志按天落 logs/realtime/。SDK 退出期偶发 SIGSEGV 不影响当日已完成的推送。
        nohup python -m realtime.engine >> "$RT_LOG" 2>&1 &
        clog "STAGE  job=realtime stage=launch pid=$! rc=0"
    fi
    ;;
  rotate)
    # 日志自动清理：超过阈值大小的 .log 只保留末尾 N 行，避免无限增长。
    # 阈值可用环境变量覆盖：LOG_MAX_BYTES(默认2MB)、LOG_KEEP_LINES(默认3000)。
    KEEP_LINES="${LOG_KEEP_LINES:-3000}"
    MAX_BYTES="${LOG_MAX_BYTES:-2097152}"
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        sz=$(wc -c < "$f" 2>/dev/null || echo 0)
        if [ "$sz" -gt "$MAX_BYTES" ]; then
            tmp="$f.rot"
            if tail -n "$KEEP_LINES" "$f" > "$tmp" 2>/dev/null; then
                mv "$tmp" "$f"
                clog "ROTATE $f ${sz}B -> tail ${KEEP_LINES} 行"
            else
                rm -f "$tmp"
            fi
        fi
    done
    ;;
  *)
    echo "usage: scheduler_jobs.sh {intraday-light|daily-light|weekly-full|monthly-factor|refresh-qfq|snapshots|news-daily|news-annotation|all-a-meta|top10-eval|sentiment-model|realtime|rotate}" >&2
    clog "ERROR  unknown job=$job"
    exit 2
    ;;
esac
