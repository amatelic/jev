#!/usr/bin/env bash
# schedule-drift.sh — install/uninstall/status a nightly launchd job that runs
# scripts/drift-eval.sh (gate accuracy for kev-0.6b + packaged fine-tunes,
# appends to data/eval-history.log, fails on threshold breach).
#
# Usage:
#   bash scripts/schedule-drift.sh install     # nightly at 08:00 (override: JEV_DRIFT_HOUR)
#   bash scripts/schedule-drift.sh uninstall
#   bash scripts/schedule-drift.sh status      # job state + last log lines
#
# launchd keeps it across reboots; the script self-heals path/arch issues via
# the repo's own resolvers. Log: ~/Library/Logs/jev-drift-eval.log
set -euo pipefail

LABEL="com.amatelic.jef.drift-eval"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/jev-drift-eval.log"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOUR="${JEV_DRIFT_HOUR:-8}"

cmd="${1:-status}"

case "$cmd" in
  install)
    mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/drift-eval.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
EOF
    # refresh if already loaded, then (re)load
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed: nightly drift-eval at $HOUR:00 (log: $LOG)"
    echo "first run: launchctl start $LABEL"
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled"
    ;;
  status)
    if [ -f "$PLIST" ]; then
      echo "plist: $PLIST (hour $HOUR configured at install time)"
      launchctl list | grep "$LABEL" || echo "not loaded"
      [ -f "$LOG" ] && { echo "── last log lines:"; tail -6 "$LOG"; } || echo "no log yet (has not run)"
    else
      echo "not installed — run: bash scripts/schedule-drift.sh install"
    fi
    ;;
  *) echo "usage: $0 install|uninstall|status" >&2; exit 2;;
esac
