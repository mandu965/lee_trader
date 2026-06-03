from __future__ import annotations

from datetime import date
import unittest

from python.us.init_us_stock_universe import LEVERAGED_OR_INVERSE, UniverseSeedRow, _merge_seed_rows


class USMetaUniverseTests(unittest.TestCase):
    def test_merge_preserves_multiple_universe_groups(self) -> None:
        rows = [
            UniverseSeedRow("AAPL", "Apple Inc.", "Technology", "Hardware", "SP500", False),
            UniverseSeedRow("AAPL", "Apple Inc.", "Technology", "Hardware", "NASDAQ100", False),
        ]
        merged = _merge_seed_rows(rows, include_etf=True, check_date=date(2026, 5, 12))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["universe_group"], "NASDAQ100,SP500")

    def test_leveraged_and_inverse_etfs_are_inactive(self) -> None:
        rows = [
            UniverseSeedRow(symbol, symbol, "ETF", "ETF", "ETF", True)
            for symbol in LEVERAGED_OR_INVERSE.keys()
        ]
        merged = _merge_seed_rows(rows, include_etf=True, check_date=date(2026, 5, 12))
        self.assertTrue(all(not row["is_active"] for row in merged))
        self.assertTrue(any(row["is_leveraged"] for row in merged))
        self.assertTrue(any(row["is_inverse"] for row in merged))

    def test_standard_etf_can_be_disabled_by_option(self) -> None:
        rows = [UniverseSeedRow("SPY", "SPY", "ETF", "ETF", "ETF", True)]
        merged = _merge_seed_rows(rows, include_etf=False, check_date=date(2026, 5, 12))
        self.assertFalse(merged[0]["is_active"])
        self.assertEqual(merged[0]["exclude_reason"], "ETF excluded by init option.")


if __name__ == "__main__":
    unittest.main()
