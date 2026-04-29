#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python python/rule_signal_builder.py
python python/rule_backtest.py
python python/rule_portfolio_manager.py
python python/rule_order_preview_builder.py
python python/rule_daily_report.py
