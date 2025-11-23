#!/usr/bin/env bash
set -euo pipefail

# Send a round of test queries to Node 0 using client.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
NODE_0_IP=${NODE_0_IP:-"127.0.0.1:8000"}

echo "Sending test queries to http://${NODE_0_IP}..."
NODE_0_IP=$NODE_0_IP "$PYTHON_BIN" client.py
