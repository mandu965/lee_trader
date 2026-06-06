from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from market_guard.detector import MarketSnapshot, evaluate_data_quality


class MarketGuardDataQualityTests(unittest.TestCase):
    def _market_status_csv(self, directory: Path, rows: str) -> Path:
        path = directory / "market_status.csv"
        path.write_text("date,kospi_close,market_up\n" + rows, encoding="utf-8")
        return path

    def test_blocks_activation_when_source_is_older_than_internal_market_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-05,8160.59,false\n")
            snapshot = MarketSnapshot(
                trade_date=date(2026, 6, 2),
                kospi_close=8801.49,
                ret_1d=-0.04,
                ret_5d=0.01,
                ret_10d=0.05,
                row_count=20,
            )

            result = evaluate_data_quality(snapshot, market_status_csv=market_csv)

        self.assertEqual(result.status, "BLOCK")
        self.assertFalse(result.can_activate)
        self.assertIn("source_stale:2026-06-02<internal:2026-06-05", result.warnings)

    def test_allows_activation_when_source_matches_internal_market_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-05,8160.59,false\n")
            snapshot = MarketSnapshot(
                trade_date=date(2026, 6, 5),
                kospi_close=8195.38,
                ret_1d=-0.0689,
                ret_5d=-0.004,
                ret_10d=0.127,
                row_count=20,
            )

            result = evaluate_data_quality(snapshot, market_status_csv=market_csv)

        self.assertEqual(result.status, "OK")
        self.assertTrue(result.can_activate)
        self.assertEqual(result.warnings, [])

    def test_blocks_activation_on_extreme_daily_return_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            market_csv = self._market_status_csv(Path(tmp), "2026-06-05,8160.59,false\n")
            snapshot = MarketSnapshot(
                trade_date=date(2026, 6, 5),
                kospi_close=8160.59,
                ret_1d=-0.35,
                ret_5d=-0.01,
                ret_10d=0.02,
                row_count=20,
            )

            result = evaluate_data_quality(snapshot, market_status_csv=market_csv)

        self.assertEqual(result.status, "BLOCK")
        self.assertFalse(result.can_activate)
        self.assertIn("daily_return_outlier:-35.00%", result.warnings)


if __name__ == "__main__":
    unittest.main()
