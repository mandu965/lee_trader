"""
US 랭킹 백필 스크립트

과거 날짜의 rule_v1 랭킹을 재계산해 us_stock_rank_daily 테이블에 채운다.
가격/피처 데이터가 있는 날짜를 기준으로 순서대로 처리한다.

사용법:
    python python/us/run_us_rank_backfill.py --start-date 2021-08-01 --end-date 2026-05-09
    python python/us/run_us_rank_backfill.py --start-date 2021-08-01 --end-date 2026-05-09 --dry-run
    python python/us/run_us_rank_backfill.py --start-date 2021-08-01 --end-date 2026-05-09 --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_config import load_us_rule_ranking_config, parse_iso_date
from python.us.us_db import get_us_engine
from python.us.calculate_us_stock_rule_scores import calculate_rule_scores

LOGGER = logging.getLogger("us_rank_backfill")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US 랭킹 백필 — 과거 날짜 rule_v1 재계산")
    p.add_argument("--start-date", required=True, help="백필 시작일 YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="백필 종료일 YYYY-MM-DD (포함)")
    p.add_argument("--dry-run", action="store_true", help="DB 저장 없이 계산만 실행")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="이미 랭킹이 있는 날짜 스킵 (기본값: True)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="이미 있는 날짜도 덮어씀")
    p.add_argument("--source", default=None, help="랭킹 소스 태그 (기본값: rule_v1)")
    p.add_argument("--delay", type=float, default=0.3,
                   help="날짜 간 대기 시간(초), DB 부하 방지 (기본값: 0.3)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def fetch_trading_dates(engine, start: date, end: date) -> list[date]:
    """가격 데이터가 있는 거래일 목록을 오름차순으로 반환."""
    sql = text("""
        SELECT DISTINCT trade_date::date AS d
        FROM market.us_stock_daily_price
        WHERE trade_date >= :start AND trade_date <= :end
        ORDER BY d
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"start": start.isoformat(), "end": end.isoformat()}).fetchall()
    return [row[0] for row in rows]


def fetch_existing_rank_dates(engine, start: date, end: date, source: str) -> set[date]:
    """이미 랭킹이 있는 날짜 집합 반환."""
    sql = text("""
        SELECT DISTINCT trade_date::date AS d
        FROM recommend.us_stock_rank_daily
        WHERE trade_date >= :start AND trade_date <= :end
          AND source = :source
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": source,
        }).fetchall()
    return {row[0] for row in rows}


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_us_rule_ranking_config()
    source = str(args.source or cfg.source).strip() or "rule_v1"

    start = parse_iso_date(args.start_date, field_name="start-date")
    end = parse_iso_date(args.end_date, field_name="end-date")
    if start is None or end is None:
        raise SystemExit("start-date / end-date 파싱 실패")
    if start > end:
        raise SystemExit("start-date가 end-date보다 늦습니다")

    engine = get_us_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"DB 연결 실패: {exc}") from exc

    LOGGER.info("[backfill] start=%s end=%s source=%s dry_run=%s skip_existing=%s",
                start, end, source, args.dry_run, args.skip_existing)

    trading_dates = fetch_trading_dates(engine, start, end)
    LOGGER.info("[backfill] 거래일 %d개 발견 (%s ~ %s)", len(trading_dates),
                trading_dates[0] if trading_dates else "-",
                trading_dates[-1] if trading_dates else "-")

    if not trading_dates:
        LOGGER.warning("[backfill] 처리할 날짜가 없습니다. 종료.")
        return

    skip_dates: set[date] = set()
    if args.skip_existing and not args.dry_run:
        skip_dates = fetch_existing_rank_dates(engine, start, end, source)
        LOGGER.info("[backfill] 기존 랭킹 있는 날짜 %d개 스킵 예정", len(skip_dates))

    target_dates = [d for d in trading_dates if d not in skip_dates]
    LOGGER.info("[backfill] 실제 처리할 날짜: %d개", len(target_dates))

    if not target_dates:
        LOGGER.info("[backfill] 모든 날짜에 이미 랭킹이 있습니다. 종료.")
        return

    total = len(target_dates)
    ok_count = 0
    fail_count = 0
    t_start = time.time()

    for idx, trade_date in enumerate(target_dates, 1):
        elapsed = time.time() - t_start
        avg_per = elapsed / max(idx - 1, 1)
        remain = avg_per * (total - idx + 1)
        LOGGER.info(
            "[backfill] [%d/%d] %s 처리 중 | 경과 %.0fs | 잔여 예상 %.0fs",
            idx, total, trade_date.isoformat(), elapsed, remain,
        )

        try:
            result = calculate_rule_scores(
                trade_date=trade_date,
                symbols=None,
                dry_run=args.dry_run,
                top_n=5,
                source=source,
                cfg=cfg,
            )
            LOGGER.info(
                "[backfill]   → evaluated=%d written=%d dry_run=%s",
                result.evaluated_symbol_count,
                result.written_row_count,
                result.dry_run,
            )
            ok_count += 1
        except Exception as exc:
            LOGGER.error("[backfill]   → 실패 %s: %s", trade_date.isoformat(), exc)
            fail_count += 1

        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    total_elapsed = time.time() - t_start
    LOGGER.info(
        "[backfill] 완료 — 성공 %d / 실패 %d / 전체 %d | 총 %.1fs",
        ok_count, fail_count, total, total_elapsed,
    )
    if args.dry_run:
        LOGGER.info("[backfill] dry-run 모드: DB에 저장하지 않았습니다.")


if __name__ == "__main__":
    main()
