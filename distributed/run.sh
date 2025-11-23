#!/usr/bin/env bash
set -euo pipefail

# Start Nodes one at a time to see logs
Start them in the foreground one at a time so the HF download progress shows:
# NODE_NUMBER=0 python -m distributed.run_node
# NODE_NUMBER=1 python -m distributed.run_node
# NODE_NUMBER=2 python -m distributed.run_node

# Convenience launcher for the distributed pipeline.
# MODE=all spins up nodes 0,1,2 locally. Otherwise, run a single node (NODE_NUMBER required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODE=${MODE:-all}
PYTHON_BIN=${PYTHON_BIN:-python3}

NODE_0_IP=${NODE_0_IP:-"127.0.0.1:8000"}
NODE_1_IP=${NODE_1_IP:-"127.0.0.1:8001"}
NODE_2_IP=${NODE_2_IP:-"127.0.0.1:8002"}

start_node() {
  local node_num=$1
  NODE_NUMBER=$node_num NODE_0_IP=$NODE_0_IP NODE_1_IP=$NODE_1_IP NODE_2_IP=$NODE_2_IP \
    "$PYTHON_BIN" -m distributed.run_node &
  echo $!
}

if [[ "$MODE" == "all" ]]; then
  echo "Starting all nodes locally (0,1,2)..."
  PID0=$(start_node 0)
  PID1=$(start_node 1)
  PID2=$(start_node 2)

  trap 'kill $PID0 $PID1 $PID2 2>/dev/null || true' INT TERM
  wait $PID0 $PID1 $PID2
else
  NODE_NUMBER=${NODE_NUMBER:-0}
  echo "Starting node ${NODE_NUMBER}..."
  exec NODE_NUMBER=$NODE_NUMBER NODE_0_IP=$NODE_0_IP NODE_1_IP=$NODE_1_IP NODE_2_IP=$NODE_2_IP \
    "$PYTHON_BIN" -m distributed.run_node
fi
