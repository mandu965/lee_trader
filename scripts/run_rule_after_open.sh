#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${RULE_TRADING_RUN_MODE:-paper}" == "paper" ]]; then
  python python/rule_execution_simulator.py
else
  python python/rule_order_fill_sync.py
fi
