from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from python.us.trade_orchestration.conflict_guard import check_buy_conflict
from python.us.trade_orchestration.config import load_trade_orchestration_config


class TradeConflictGuardTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=False)
    def test_open_position_blocks_buy(self) -> None:
        cfg = load_trade_orchestration_config()
        result = check_buy_conflict(
            {"symbol": "AAPL"},
            {
                "status": "OK",
                "open_position_map": {"AAPL": {"paper_position_id": "POS1"}},
                "sell_signal_map": {},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": [],
            },
            cfg,
        )
        self.assertFalse(result["buy_allowed_after_conflict_check"])
        self.assertIn("OPEN_POSITION_EXISTS", result["conflict_reasons"])

    @patch.dict(os.environ, {}, clear=False)
    def test_sell_signal_blocks_buy(self) -> None:
        cfg = load_trade_orchestration_config()
        result = check_buy_conflict(
            {"symbol": "NVDA"},
            {
                "status": "OK",
                "open_position_map": {},
                "sell_signal_map": {"NVDA": {"decision": "SELL", "exit_reason": "STOP_LOSS"}},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": [],
            },
            cfg,
        )
        self.assertFalse(result["buy_allowed_after_conflict_check"])
        self.assertIn("SELL_SIGNAL_EXISTS", result["conflict_reasons"])
        self.assertIn("SELL_PRIORITY_OVER_BUY", result["conflict_reasons"])

    @patch.dict(os.environ, {}, clear=False)
    def test_inconsistent_state_blocks_buy_failsafe(self) -> None:
        cfg = load_trade_orchestration_config()
        result = check_buy_conflict(
            {"symbol": "TSLA"},
            {
                "status": "PORTFOLIO_STATE_INCONSISTENT",
                "open_position_map": {},
                "sell_signal_map": {},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": [],
            },
            cfg,
        )
        self.assertFalse(result["buy_allowed_after_conflict_check"])
        self.assertIn("PORTFOLIO_STATE_INCONSISTENT", result["conflict_reasons"])


if __name__ == "__main__":
    unittest.main()
