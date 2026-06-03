from __future__ import annotations

import unittest

from python.us.buy_automation.performance_metrics import (
    calculate_excess_return,
    calculate_max_drawdown,
    calculate_return_pct,
    calculate_win_rate,
    summarize_returns,
)


class BuyPerformanceMetricsTests(unittest.TestCase):
    def test_calculate_return_pct(self) -> None:
        self.assertEqual(calculate_return_pct(100, 110), 0.1)
        self.assertIsNone(calculate_return_pct(0, 110))

    def test_calculate_win_rate(self) -> None:
        self.assertEqual(calculate_win_rate([0.1, -0.05, 0.02]), 2 / 3)
        self.assertIsNone(calculate_win_rate([]))

    def test_calculate_max_drawdown(self) -> None:
        self.assertAlmostEqual(calculate_max_drawdown([1.0, 1.1, 0.9, 1.2]), (1.1 - 0.9) / 1.1)

    def test_calculate_excess_return(self) -> None:
        self.assertEqual(calculate_excess_return(0.12, 0.05), 0.07)
        self.assertIsNone(calculate_excess_return(None, 0.05))

    def test_summarize_returns(self) -> None:
        summary = summarize_returns([0.1, -0.1, 0.05])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["best_trade_return_pct"], 0.1)
        self.assertEqual(summary["worst_trade_return_pct"], -0.1)


if __name__ == "__main__":
    unittest.main()
