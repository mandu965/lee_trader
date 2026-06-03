from __future__ import annotations

from datetime import date
import json
import unittest

from python.us.us_rank_design import (
    US_RANKING_GRADE_GUIDE,
    US_RANKING_SCORE_WEIGHTS,
    build_sample_rank_rows,
    build_score_detail_json,
)


class USRankDesignTests(unittest.TestCase):
    def test_risk_score_convention_is_negative_penalty(self) -> None:
        self.assertEqual(US_RANKING_SCORE_WEIGHTS["risk_score_floor"], -10.0)
        self.assertEqual(US_RANKING_SCORE_WEIGHTS["risk_score_ceiling"], 0.0)

    def test_grade_guide_includes_all_expected_grades(self) -> None:
        self.assertEqual(
            list(US_RANKING_GRADE_GUIDE.keys()),
            ["STRONG_BUY", "BUY", "WATCH", "HOLD", "EXCLUDE"],
        )

    def test_sample_rows_follow_rank_and_grade_contract(self) -> None:
        rows = build_sample_rank_rows(trade_date=date(2099, 12, 31), source="phase3_2_dryrun")
        self.assertEqual([row["rank_no"] for row in rows], [1, 2, 3, 4, 5])
        self.assertTrue(all(row["risk_score"] <= 0 for row in rows))
        self.assertTrue(all(isinstance(row["score_detail_json"], str) for row in rows))

    def test_score_detail_json_has_expected_sections(self) -> None:
        payload = json.loads(build_score_detail_json("AAPL"))
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertIn("momentum", payload)
        self.assertIn("relative_strength", payload)
        self.assertIn("fundamental", payload)
        self.assertIn("growth", payload)
        self.assertIn("valuation", payload)
        self.assertIn("risk", payload)


if __name__ == "__main__":
    unittest.main()
