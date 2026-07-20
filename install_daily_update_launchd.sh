#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/wanghao81/Desktop/A"
AGENT_DIR="$HOME/Library/LaunchAgents"
USER_DOMAIN="gui/$(id -u)"

mkdir -p "$PROJECT_DIR/logs" "$AGENT_DIR"

# Remove legacy or stale tasks before installing the production schedule.
for old in \
  "$AGENT_DIR/com.wanghao81.stock-daily-update.plist" \
  "$AGENT_DIR/com.wanghao81.stock-daily-light.plist" \
  "$AGENT_DIR/com.wanghao81.stock-weekly-full.plist" \
  "$AGENT_DIR/com.wanghao81.stock-snapshots-daily.plist"; do
  launchctl bootout "$USER_DOMAIN" "$old" >/dev/null 2>&1 || true
  rm -f "$old"
done

install_one() {
  local plist_name="$1"
  local label="$2"
  local src="$PROJECT_DIR/$plist_name"
  local dst="$AGENT_DIR/$plist_name"
  cp "$src" "$dst"
  launchctl bootout "$USER_DOMAIN" "$dst" >/dev/null 2>&1 || true
  launchctl bootstrap "$USER_DOMAIN" "$dst"
  launchctl enable "$USER_DOMAIN/$label"
}

install_one "com.wanghao81.stock-daily-light.plist" "com.wanghao81.stock-daily-light"
install_one "com.wanghao81.stock-weekly-full.plist" "com.wanghao81.stock-weekly-full"
install_one "com.wanghao81.stock-snapshots-daily.plist" "com.wanghao81.stock-snapshots-daily"

cat <<EOF
已安装三个生产定时任务：

1. com.wanghao81.stock-daily-light
   执行时间：每天 11:40（周五除外，周五由 weekly-full 覆盖）
   内容：快速日更 + 短线h3训练 + 波段h10训练 + active发布 + 新鲜度校验
   说明：跳过 valuation/fundamentals/snapshots，避免快照阻塞实战训练。
   日志：
     $PROJECT_DIR/logs/stock-daily-light.out.log
     $PROJECT_DIR/logs/stock-daily-light.err.log

2. com.wanghao81.stock-weekly-full
   执行时间：每周五 11:40
   内容：更完整日更 + 短线h3训练 + 波段h10训练 + active发布 + 新鲜度校验
   说明：保留 valuation/fundamentals 和 180 天事件窗口；跳过 snapshots。
   日志：
     $PROJECT_DIR/logs/stock-weekly-full.out.log
     $PROJECT_DIR/logs/stock-weekly-full.err.log

3. com.wanghao81.stock-snapshots-daily
   执行时间：每天 17:00
   内容：读取 snapshots/watchlist.txt，单独积累白名单多维快照
   日志：
     $PROJECT_DIR/logs/stock-snapshots-daily.out.log
     $PROJECT_DIR/logs/stock-snapshots-daily.err.log

查看任务：
  launchctl print $USER_DOMAIN/com.wanghao81.stock-daily-light
  launchctl print $USER_DOMAIN/com.wanghao81.stock-weekly-full
  launchctl print $USER_DOMAIN/com.wanghao81.stock-snapshots-daily

手动触发：
  launchctl kickstart -k $USER_DOMAIN/com.wanghao81.stock-daily-light
  launchctl kickstart -k $USER_DOMAIN/com.wanghao81.stock-weekly-full
  launchctl kickstart -k $USER_DOMAIN/com.wanghao81.stock-snapshots-daily

检查更新结果：
  cd $PROJECT_DIR && ./run_daily_update_once.sh check --sample 50

卸载：
  launchctl bootout $USER_DOMAIN "$AGENT_DIR/com.wanghao81.stock-daily-light.plist"
  launchctl bootout $USER_DOMAIN "$AGENT_DIR/com.wanghao81.stock-weekly-full.plist"
  launchctl bootout $USER_DOMAIN "$AGENT_DIR/com.wanghao81.stock-snapshots-daily.plist"
EOF
