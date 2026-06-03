from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

from python.us.trade_orchestration.daily_trade_orchestrator import run_trade_orchestration


class DailyTradeOrchestratorTests(unittest.TestCase):
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.persist_integrated_logs")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.build_integrated_report")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.write_integrated_report_markdown")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.write_integrated_report_json")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.run_buy_automation")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.load_portfolio_state")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.run_sell_automation")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_REPORT_ENABLED": "1",
        },
        clear=False,
    )
    def test_sell_runs_before_buy_and_conflict_summary_is_built(
        self,
        sell_mock,
        portfolio_mock,
        buy_mock,
        write_json_mock,
        write_md_mock,
        build_report_mock,
        persist_mock,
    ) -> None:
        call_order: list[str] = []

        def sell_side_effect(*args, **kwargs):
            call_order.append("sell")
            return {
                "trade_date": "2026-05-14",
                "loaded_positions": 1,
                "hold_positions": 0,
                "sell_signals": 1,
                "partial_sell_signals": 0,
                "review_required": 0,
                "paper_sell_orders": [],
                "positions": [{"symbol": "AAPL", "paper_position_id": "POS1", "status": "OPEN"}],
                "decisions": [{"symbol": "AAPL", "decision": "SELL", "exit_reason": "STOP_LOSS"}],
            }

        def portfolio_side_effect(*args, **kwargs):
            call_order.append("portfolio")
            return {
                "trade_date": "2026-05-14",
                "status": "OK",
                "open_positions": [{"symbol": "AAPL", "paper_position_id": "POS1"}],
                "open_position_map": {"AAPL": {"paper_position_id": "POS1"}},
                "sell_signal_map": {"AAPL": {"decision": "SELL", "exit_reason": "STOP_LOSS"}},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": [],
            }

        def buy_side_effect(*args, **kwargs):
            call_order.append("buy")
            return {
                "loaded_candidates": 1,
                "allowed_before_conflict": 1,
                "allowed_after_conflict": 0,
                "conflict_blocked_candidates": 1,
                "allowed_candidates": 0,
                "blocked_candidates": 1,
                "paper_orders": [],
                "candidates": [
                    {
                        "symbol": "AAPL",
                        "conflict_checked": True,
                        "buy_allowed_after_conflict_check": False,
                        "conflict_reasons": ["OPEN_POSITION_EXISTS", "SELL_SIGNAL_EXISTS"],
                        "related_position_id": "POS1",
                        "related_sell_signal": {"decision": "SELL", "exit_reason": "STOP_LOSS"},
                        "allowed": False,
                        "block_reasons": ["OPEN_POSITION_EXISTS", "SELL_SIGNAL_EXISTS"],
                    }
                ],
                "block_summary": {"OPEN_POSITION_EXISTS": 1, "SELL_SIGNAL_EXISTS": 1},
            }

        sell_mock.side_effect = sell_side_effect
        portfolio_mock.side_effect = portfolio_side_effect
        buy_mock.side_effect = buy_side_effect
        build_report_mock.return_value = {"trade_date": "2026-05-14", "mode": "SHADOW", "orchestration_enabled": True, "success": True, "sell_summary": {}, "buy_summary": {}, "conflict_summary": {}}
        persist_mock.return_value = {}

        result = run_trade_orchestration(trade_date="2026-05-14", account_id="US_TEST", persist_logs=True)

        self.assertTrue(result["success"])
        self.assertEqual(call_order[:3], ["sell", "portfolio", "buy"])
        self.assertEqual(result["conflict_summary"]["TOTAL_CONFLICT_BLOCKED"], 1)
        self.assertEqual(result["buy_summary"]["conflict_blocked"], 1)
        write_json_mock.assert_called_once()
        write_md_mock.assert_called_once()

    @patch("python.us.trade_orchestration.daily_trade_orchestrator.persist_integrated_logs")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.build_integrated_report")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.write_integrated_report_markdown")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.write_integrated_report_json")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.persist_position_snapshots")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.load_paper_positions")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.load_sell_automation_config")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.run_buy_automation")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.load_portfolio_state")
    @patch("python.us.trade_orchestration.daily_trade_orchestrator.run_sell_automation")
    @patch.dict(
        os.environ,
        {
            "US_TRADE_ORCHESTRATION_ENABLED": "1",
            "US_TRADE_REPORT_ENABLED": "1",
            "US_TRADE_ORCHESTRATION_MODE": "PAPER",
        },
        clear=False,
    )
    def test_paper_mode_refreshes_snapshots_after_buy(
        self,
        sell_mock,
        portfolio_mock,
        buy_mock,
        sell_cfg_mock,
        refresh_positions_mock,
        persist_snapshots_mock,
        write_json_mock,
        write_md_mock,
        build_report_mock,
        persist_mock,
    ) -> None:
        sell_mock.return_value = {
            "trade_date": "2026-05-15",
            "loaded_positions": 0,
            "hold_positions": 0,
            "sell_signals": 0,
            "partial_sell_signals": 0,
            "review_required": 0,
            "paper_sell_orders": [],
            "positions": [],
            "decisions": [],
        }
        portfolio_mock.side_effect = [
            {
                "trade_date": "2026-05-15",
                "status": "OK",
                "open_positions": [],
                "open_position_map": {},
                "sell_signal_map": {},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": [],
            },
            {
                "trade_date": "2026-05-15",
                "status": "OK",
                "open_positions": [{"symbol": "NVDA", "paper_position_id": "POS_NVDA"}],
                "open_position_map": {"NVDA": {"paper_position_id": "POS_NVDA"}},
                "sell_signal_map": {},
                "review_required_map": {},
                "cooldown_map": {},
                "paper_buy_symbols_today": ["NVDA"],
            },
        ]
        buy_mock.return_value = {
            "loaded_candidates": 1,
            "allowed_before_conflict": 1,
            "allowed_after_conflict": 1,
            "conflict_blocked_candidates": 0,
            "allowed_candidates": 1,
            "blocked_candidates": 0,
            "paper_orders": [{"symbol": "NVDA", "trade_date": "2026-05-15"}],
            "candidates": [],
            "block_summary": {},
        }
        sell_cfg_mock.return_value = object()
        refresh_positions_mock.return_value = {
            "trade_date": date(2026, 5, 15),
            "positions": [{"symbol": "NVDA", "paper_position_id": "POS_NVDA", "status": "OPEN"}],
        }
        persist_snapshots_mock.return_value = 1
        build_report_mock.return_value = {
            "trade_date": "2026-05-15",
            "mode": "PAPER",
            "orchestration_enabled": True,
            "success": True,
            "sell_summary": {},
            "buy_summary": {},
            "conflict_summary": {},
        }
        persist_mock.return_value = {}

        result = run_trade_orchestration(trade_date="2026-05-15", account_id="US_TEST", persist_logs=True)

        self.assertTrue(result["success"])
        refresh_positions_mock.assert_called_once()
        persist_snapshots_mock.assert_called_once_with(
            trade_date="2026-05-15",
            positions=[{"symbol": "NVDA", "paper_position_id": "POS_NVDA", "status": "OPEN"}],
        )
        self.assertEqual(result["paper_snapshot_persistence"]["positions"], 1)
        self.assertEqual(result["portfolio_state"]["paper_buy_symbols_today"], ["NVDA"])
        self.assertEqual(len(result["refreshed_positions"]), 1)
        write_json_mock.assert_called_once()
        write_md_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
