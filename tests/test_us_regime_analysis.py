from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from python.us.analyze_us_stock_backtest_by_regime import (
    _build_interpretation_lines,
    _build_quality_summary,
    aggregate_regime_rows,
    build_console_report,
)
from python.us.build_us_market_regime_daily import compute_market_regime_rows
from python.us.us_config import USMarketRegimeConfig


def make_cfg() -> USMarketRegimeConfig:
    return USMarketRegimeConfig(
        spy_vol20_high_threshold=0.025,
        qqq_vol20_high_threshold=0.030,
        min_test_days_warning=20,
        report_output_dir=Path("outputs/us_stock_backtest"),
        report_default_format="console",
        log_level="INFO",
    )


class MarketRegimeBuildTests(unittest.TestCase):
    def test_compute_market_regime_rows_builds_bull_low_vol(self) -> None:
        import python.us.build_us_market_regime_daily as regime_mod

        original_loader = regime_mod._load_price_frame
        try:
            def fake_loader(*, start_date, end_date):  # noqa: ANN001
                import pandas as pd

                dates = pd.date_range("2026-01-01", periods=90, freq="B")
                spy_prices = [100.0 + i * 0.4 for i in range(len(dates))]
                qqq_prices = [200.0 + i * 0.5 for i in range(len(dates))]
                rows = []
                for idx, day in enumerate(dates):
                    rows.append({"trade_date": day, "ticker": "SPY", "price": spy_prices[idx]})
                    rows.append({"trade_date": day, "ticker": "QQQ", "price": qqq_prices[idx]})
                return pd.DataFrame(rows)

            regime_mod._load_price_frame = fake_loader
            rows = compute_market_regime_rows(start_date=date(2026, 4, 1), end_date=date(2026, 4, 30), cfg=make_cfg())
        finally:
            regime_mod._load_price_frame = original_loader

        self.assertTrue(rows)
        self.assertTrue(any(row["market_regime"] == "BULL_LOW_VOL" for row in rows))
        self.assertTrue(all(row["spy_regime"] in {"BULL", "UNKNOWN"} for row in rows))


class RegimeAnalysisTests(unittest.TestCase):
    def make_joined_row(self, **overrides) -> dict[str, object]:
        row = {
            "backtest_id": "US_RANK_RULE_V1_TEST",
            "trade_date": date(2026, 5, 11),
            "strategy_name": "US_RANK_TOP20",
            "selection_rule": "rank_no <= 20",
            "holding_days": 20,
            "selected_count": 20,
            "avg_return_pct": 0.03,
            "median_return_pct": 0.028,
            "win_rate": 0.58,
            "avg_excess_return_vs_spy": 0.01,
            "avg_excess_return_vs_qqq": 0.006,
            "avg_excess_return_vs_universe": 0.012,
            "win_rate_vs_spy": 0.55,
            "win_rate_vs_qqq": 0.53,
            "win_rate_vs_universe": 0.57,
            "market_regime": "BULL_LOW_VOL",
            "spy_regime": "BULL",
            "qqq_regime": "QQQ_BULL",
            "vol_regime": "LOW_VOL",
            "regime_data_status": "OK",
        }
        row.update(overrides)
        return row

    def test_aggregate_regime_rows_builds_month_and_market_groups(self) -> None:
        rows = [
            self.make_joined_row(trade_date=date(2026, 5, 11), avg_return_pct=0.03),
            self.make_joined_row(trade_date=date(2026, 5, 12), avg_return_pct=0.05),
        ]
        aggregate = aggregate_regime_rows(rows)
        market_rows = [row for row in aggregate if row["regime_type"] == "MARKET_REGIME"]
        month_rows = [row for row in aggregate if row["regime_type"] == "MONTH"]
        self.assertTrue(market_rows)
        self.assertTrue(month_rows)
        target = next(row for row in market_rows if row["regime_value"] == "BULL_LOW_VOL")
        self.assertEqual(target["test_days"], 2)
        self.assertAlmostEqual(float(target["avg_return_pct"]), 0.04, places=6)

    def test_console_report_contains_best_and_worst_regime_sections(self) -> None:
        joined_rows = [
            self.make_joined_row(trade_date=date(2026, 5, 11), avg_return_pct=0.05, avg_excess_return_vs_spy=0.02),
            self.make_joined_row(trade_date=date(2026, 5, 12), market_regime="BEAR_HIGH_VOL", spy_regime="BEAR", vol_regime="HIGH_VOL", avg_return_pct=-0.02, avg_excess_return_vs_spy=-0.01, win_rate_vs_spy=0.40),
        ]
        aggregate = aggregate_regime_rows(joined_rows)
        quality = _build_quality_summary(joined_rows, make_cfg())
        interpretation = _build_interpretation_lines(aggregate, make_cfg())
        rendered = build_console_report(
            backtest_id="US_RANK_RULE_V1_TEST",
            joined_rows=joined_rows,
            aggregate_rows=aggregate,
            quality=quality,
            interpretation=interpretation,
            improvement_candidates=["candidate"],
            strategy="US_RANK_TOP20",
            holding_days=20,
            cfg=make_cfg(),
        )
        self.assertIn("[Best Regime]", rendered)
        self.assertIn("[Worst Regime]", rendered)
        self.assertIn("BULL_LOW_VOL", rendered)


if __name__ == "__main__":
    unittest.main()
