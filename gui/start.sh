#!/bin/bash
# Start EconSimulacra GUI demo server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"

echo "Building frontend..."
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
(cd "$SCRIPT_DIR/frontend" && npm run build)

echo ""
echo "Starting backend server at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""
"$VENV/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --app-dir "$SCRIPT_DIR/backend"
