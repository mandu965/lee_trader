"""
run_us_macro_overlay_scheduler.py

Standalone runner for US Macro Overlay Phase 1~2.
Executes data collection → feature computation in sequence.

⚠ SHADOW MODE (Phase 1~2):
   This script does NOT modify any Korean auto-trading logic.
   It only reads US market data and writes to market/signal schemas.
   No orders, rankings, or scores are affected.

Usage (manual):
    python run_us_macro_overlay_scheduler.py
    python run_us_macro_overlay_scheduler.py --date 2026-05-07
    python run_us_macro_overlay_scheduler.py --lookback-days 30
    python run_us_macro_overlay_scheduler.py --dry-run

Docker (one-off):
    docker compose run --rm python-pipeline python python/run_us_macro_overlay_scheduler.py

Scheduled (cron after US market close, KST ~07:30):
    30 7 * * 2-6 docker compose run --rm python-pipeline python python/run_us_macro_overlay_scheduler.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from db import get_engine

LOGGER = logging.getLogger("run_us_macro_overlay_scheduler")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="US Macro Overlay Scheduler — Phase 1 (collect) + Phase 2 (features)"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="US trade date to process (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=10,
        help="Calendar days of history to collect. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and compute but do not write to DB.",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip Phase 1 (collection). Useful if price data is already loaded.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip Phase 2 (feature computation).",
    )
    return parser.parse_args()


def _guard_enabled() -> bool:
    val = os.environ.get("US_MACRO_ENABLED", "1").strip().lower()
    return val in ("1", "true", "yes", "y")


def _guard_shadow_mode() -> bool:
    val = os.environ.get("US_MACRO_SHADOW_MODE", "1").strip().lower()
    return val in ("1", "true", "yes", "y")


def _guard_allow_real_apply() -> bool:
    val = os.environ.get("US_MACRO_ALLOW_REAL_APPLY", "0").strip().lower()
    return val in ("1", "true", "yes", "y")


def run_phase1(
    target_date: date,
    lookback_days: int,
    dry_run: bool,
    source: str,
    engine: object,
) -> dict:
    from collect_us_macro_etf_daily import collect, _get_adapter, _resolve_tickers

    tickers = _resolve_tickers()
    adapter = _get_adapter(source)
    end_date = target_date
    start_date = end_date - timedelta(days=lookback_days)

    LOGGER.info("[Phase 1] Collecting %d tickers | %s → %s", len(tickers), start_date, end_date)
    summary = collect(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        adapter=adapter,
        engine=engine,
        dry_run=dry_run,
        data_source=source,
    )
    return summary


def run_phase2(target_date: date, dry_run: bool, engine: object) -> dict:
    from compute_us_macro_feature_daily import run as compute_run

    LOGGER.info("[Phase 2] Computing macro features for %s", target_date)
    result = compute_run(target_date=target_date, engine=engine, dry_run=dry_run)
    return result


def main() -> None:
    setup_logging()
    args = parse_args()

    if not _guard_enabled():
        LOGGER.info("US_MACRO_ENABLED=0 — skipping US macro overlay run.")
        sys.exit(0)

    shadow = _guard_shadow_mode()
    allow_real = _guard_allow_real_apply()

    if not shadow and not allow_real:
        LOGGER.error(
            "US_MACRO_SHADOW_MODE=0 but US_MACRO_ALLOW_REAL_APPLY=0. "
            "Set US_MACRO_ALLOW_REAL_APPLY=1 to enable real apply (Phase 3+). "
            "Exiting."
        )
        sys.exit(1)

    LOGGER.info("=" * 60)
    LOGGER.info("US Macro Overlay Scheduler — Phase 1~2 (Shadow Mode=%s)", shadow)
    LOGGER.info("⚠ No Korean auto-trading logic will be affected.")
    LOGGER.info("=" * 60)

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    source = os.environ.get("US_MACRO_DATA_SOURCE", "yfinance")
    engine = get_engine()

    phase1_ok = True
    phase2_ok = True

    # Phase 1: Collect
    if not args.skip_collect:
        try:
            p1 = run_phase1(
                target_date=target_date,
                lookback_days=args.lookback_days,
                dry_run=args.dry_run,
                source=source,
                engine=engine,
            )
            if p1["failed"]:
                LOGGER.warning(
                    "[Phase 1] Completed with failures: %s / %d tickers failed.",
                    p1["failed"], len(p1["failed"]),
                )
            else:
                LOGGER.info(
                    "[Phase 1] All %d tickers collected. Rows upserted: %d.",
                    len(p1["success"]), p1["rows_upserted"],
                )
        except Exception as exc:
            LOGGER.error("[Phase 1] Fatal error: %s", exc, exc_info=True)
            phase1_ok = False
    else:
        LOGGER.info("[Phase 1] Skipped (--skip-collect).")

    # Phase 2: Compute features
    if not args.skip_features:
        try:
            p2 = run_phase2(
                target_date=target_date,
                dry_run=args.dry_run,
                engine=engine,
            )
            LOGGER.info(
                "[Phase 2] Done | us_trade_date=%s | kr_apply_date=%s | status=%s | missing=%s",
                p2.get("us_trade_date"),
                p2.get("kr_apply_date"),
                p2.get("macro_status"),
                p2.get("missing_tickers"),
            )
        except Exception as exc:
            LOGGER.error("[Phase 2] Fatal error: %s", exc, exc_info=True)
            phase2_ok = False
    else:
        LOGGER.info("[Phase 2] Skipped (--skip-features).")

    LOGGER.info("=" * 60)
    if phase1_ok and phase2_ok:
        LOGGER.info("US Macro Overlay run completed successfully.")
    else:
        LOGGER.warning("US Macro Overlay run completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
