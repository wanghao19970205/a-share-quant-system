#!/bin/zsh
set -euo pipefail

cd /Users/wanghao81/Desktop/A
export PYTHONPATH="/Users/wanghao81/Desktop/A${PYTHONPATH:+:$PYTHONPATH}"
export QUANT_DATA_DIR="/Users/wanghao81/Desktop/A/quant_data/full_a_2018_wide"
export SNAPSHOT_DIR="/Users/wanghao81/Desktop/A/snapshots"
PY="/Library/Developer/CommandLineTools/usr/bin/python3"

case "${1:-daily}" in
  check)
    shift
    "$PY" -m quant.check_daily_update "$@"
    ;;
  daily)
    shift || true
    "$PY" -m quant.scheduled_workflow --universe mainboard_active --update-workers 12 --lookback-days 5 --event-window-days 30 --skip-valuation --skip-fundamentals --skip-snapshots --snapshot-dir "$SNAPSHOT_DIR" "$@"
    ;;
  weekly)
    shift || true
    "$PY" -m quant.scheduled_workflow --universe mainboard_active --update-workers 12 --lookback-days 5 --event-window-days 180 --skip-snapshots --snapshot-dir "$SNAPSHOT_DIR" "$@"
    ;;
  snapshots)
    shift || true
    "$PY" -m stock_analyzer.snapshot_batch "$@"
    ;;
  data-only)
    shift || true
    "$PY" -m quant.daily_update --universe mainboard_active --workers 12 --lookback-days 5 --event-window-days 30 --skip-snapshots --snapshot-dir "$SNAPSHOT_DIR" "$@"
    ;;
  weekly-data-only)
    shift || true
    "$PY" -m quant.daily_update --universe mainboard_active --workers 12 --lookback-days 5 --event-window-days 180 --skip-snapshots --snapshot-dir "$SNAPSHOT_DIR" "$@"
    ;;
  *)
    "$PY" -m quant.scheduled_workflow "$@"
    ;;
esac
