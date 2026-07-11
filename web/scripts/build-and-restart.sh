#!/usr/bin/env bash
# Rebuild the Svelte frontend, then restart the backend so it picks up the
# new dist/ bundle. Use this after editing anything under frontend/src.
#
# Usage: scripts/build-and-restart.sh [port]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -d "frontend/node_modules" ]]; then
  echo "error: frontend/node_modules not found — run 'npm install' in frontend/ first" >&2
  exit 1
fi

echo "Building frontend..."
(cd frontend && npm run build)

exec "$(dirname "${BASH_SOURCE[0]}")/restart.sh" "$@"
