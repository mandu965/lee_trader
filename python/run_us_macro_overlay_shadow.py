"""
run_us_macro_overlay_shadow.py

Phase 3 standalone runner — Shadow Mode overlay log generation.

Reads US macro features + Korean buy candidates, applies overlay rules,
and writes hypothetical results to signal.kr_macro_overlay_log.

⚠ SHADOW MODE ONLY:
   - Does NOT modify any order, ranking, score, or existing table.
   - Writes ONLY to signal.kr_macro_overlay_log.
   - US_MACRO_ALLOW_REAL_APPLY must remain 0 for this script to run.

Usage:
    python run_us_macro_overlay_shadow.py
    python run_us_macro_overlay_shadow.py --kr-apply-date 2026-05-08
    python run_us_macro_overlay_shadow.py --dry-run
    python run_us_macro_overlay_shadow.py --rule-signals-csv /path/to/rule_signals.csv

Docker:
    docker compose run --rm python-pipeline python python/run_us_macro_overlay_shadow.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from db import get_engine

LOGGER = logging.getLogger("run_us_macro_overlay_shadow")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: US Macro Overlay Shadow Mode — writes to signal.kr_macro_overlay_log only"
    )
    parser.add_argument(
        "--kr-apply-date",
        default=None,
        help="Korean apply date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute overlay but do not write to DB.",
    )
    parser.add_argument(
        "--rule-signals-csv",
        type=Path,
        default=None,
        help="Path to rule_signals.csv. Default: data/rule_signals.csv.",
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


def main() -> None:
    setup_logging()
    args = parse_args()

    # ── Safety guards ────────────────────────────────────────────────────────
    if not _guard_enabled():
        LOGGER.info("US_MACRO_ENABLED=0 - skipping Phase 3 shadow overlay.")
        sys.exit(0)

    if _guard_allow_real_apply():
        LOGGER.error(
            "US_MACRO_ALLOW_REAL_APPLY=1 is set. "
            "Phase 3 is shadow-only. This script does not support real apply. "
            "Set US_MACRO_ALLOW_REAL_APPLY=0 and re-run."
        )
        sys.exit(1)

    if not _guard_shadow_mode():
        LOGGER.warning(
            "US_MACRO_SHADOW_MODE=0 detected. "
            "Phase 3 is shadow-only regardless - proceeding in shadow mode."
        )

    LOGGER.info("=" * 60)
    LOGGER.info("Phase 3: US Macro Overlay - SHADOW MODE")
    LOGGER.info("[SHADOW] 실제 주문/랭킹/점수에는 영향을 주지 않습니다.")
    LOGGER.info("=" * 60)

    from compute_us_macro_overlay_shadow import (
        DEFAULT_RULE_SIGNALS_CSV,
        fetch_latest_macro_apply_date,
        run,
    )

    rule_signals_csv = args.rule_signals_csv or DEFAULT_RULE_SIGNALS_CSV
    engine = get_engine()
    requested_kr_apply_date = (
        date.fromisoformat(args.kr_apply_date)
        if args.kr_apply_date
        else fetch_latest_macro_apply_date(engine)
    )
    if requested_kr_apply_date is None:
        LOGGER.error("signal.us_macro_feature_daily has no rows; run Phase 1~2 first.")
        sys.exit(1)
    LOGGER.info("Resolved kr_apply_date=%s", requested_kr_apply_date.isoformat())

    try:
        result = run(
            kr_apply_date=requested_kr_apply_date,
            engine=engine,
            rule_signals_csv=rule_signals_csv,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        LOGGER.error("Phase 3 failed: %s", exc, exc_info=True)
        sys.exit(1)

    LOGGER.info("=" * 60)
    LOGGER.info(
        "Phase 3 완료 | kr_apply_date=%s | macro_status=%s"
        " | rule_rows=%d | ai_rows=%d | blocked=%d",
        result.get("kr_apply_date"),
        result.get("macro_status"),
        result.get("rule_rows", 0),
        result.get("ai_rows", 0),
        result.get("blocked_count", 0),
    )
    LOGGER.info("[SHADOW MODE] 이 결과는 실제 자동매매에 영향을 주지 않습니다.")
    LOGGER.info("결과 확인: signal.kr_macro_overlay_log 테이블 조회")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
