#!/usr/bin/env bash
# Stop any running dev server and start a fresh one in the background.
# Logs: data/uvicorn.log · PID file: data/uvicorn.pid
#
# Usage: scripts/restart.sh [port]   (default port 8080)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PORT="${1:-8080}"
PID_FILE="data/uvicorn.pid"
LOG_FILE="data/uvicorn.log"

mkdir -p data

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping uvicorn (pid $old_pid)..."
    kill "$old_pid"
    for _ in $(seq 1 20); do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$old_pid" 2>/dev/null && kill -9 "$old_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "error: .venv/bin/uvicorn not found — run 'uv venv && uv pip install -r requirements.txt' first" >&2
  exit 1
fi

echo "Starting uvicorn on :$PORT..."
nohup .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port "$PORT" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 1

if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "error: server failed to start — see $LOG_FILE" >&2
  tail -n 20 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "Running (pid $(cat "$PID_FILE")). Logs: $LOG_FILE"
echo "  http://localhost:$PORT/"
