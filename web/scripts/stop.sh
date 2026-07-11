#!/usr/bin/env bash
# Stop the background dev server started by restart.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PID_FILE="data/uvicorn.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Not running (no $PID_FILE)."
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  echo "Stopping uvicorn (pid $pid)..."
  kill "$pid"
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.2
  done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
else
  echo "Stale pid file (process $pid not running)."
fi

rm -f "$PID_FILE"
echo "Stopped."
