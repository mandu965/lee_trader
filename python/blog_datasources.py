"""
TYPE_B / TYPE_C 데이터 소스 계층.

운영 PC(DATABASE_URL 설정)에서는 PostgreSQL을 조회하고,
개인 PC(DB 없음)에서는 fixtures/blog/*.json 으로 폴백한다.
어떤 소스를 썼는지 로그로 남긴다.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "blog"
log = logging.getLogger("blog_gen")


def _db_available() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _load_fixture(name: str) -> dict:
    with open(FIXTURE_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# TYPE_B: 랭킹 급상승
# ---------------------------------------------------------------------------
def get_ranking_change(date: str | None, top_n: int = 5) -> dict:
    """5영업일 전 대비 순위 상승폭 상위 종목.

    반환: {"today": d, "prev": d, "risers": [{code,name,sector,rank,prev_rank,rank_gain,final_score}, ...]}
    """
    if _db_available():
        try:
            return _get_ranking_change_db(date, top_n)
        except Exception as exc:  # noqa: BLE001
            log.warning("DB 랭킹 조회 실패(%s). fixture로 폴백합니다.", exc)
    else:
        log.info("DATABASE_URL 미설정 → 랭킹 fixture 사용(개인 PC 모드).")
    return _get_ranking_change_fixture(top_n)


def _compute_risers(rows_today: list[dict], prev_map: dict[str, int], top_n: int) -> list[dict]:
    risers = []
    for r in rows_today:
        code = str(r["code"])
        if code not in prev_map:
            continue  # 신규 진입은 별도 취급(여기서는 제외)
        prev_rank = prev_map[code]
        rank_today = int(r["rank_final"])
        gain = prev_rank - rank_today
        if rank_today > 50 or gain < 10:
            continue
        risers.append({
            "code": code,
            "name": r.get("name", "-"),
            "sector": r.get("sector", "-"),
            "rank": rank_today,
            "prev_rank": prev_rank,
            "rank_gain": gain,
            "final_score": round(float(r.get("final_score", 0)), 1),
        })
    risers.sort(key=lambda x: x["rank_gain"], reverse=True)
    return risers[:top_n]


def _get_ranking_change_fixture(top_n: int) -> dict:
    fx = _load_fixture("daily_ranking_sample.json")
    prev_map = {str(r["code"]): int(r["rank_final"]) for r in fx["rows_prev"]}
    risers = _compute_risers(fx["rows_today"], prev_map, top_n)
    return {"today": fx["today"], "prev": fx["prev"], "risers": risers, "source": "fixture"}


def _get_ranking_change_db(date: str | None, top_n: int) -> dict:
    import db  # python/db.py
    conn = db.raw_psycopg2_conn()
    try:
        with conn.cursor() as cur:
            # 대상일(미지정 시 최신), 비교일은 화면과 동일하게 최근 5개 거래일 중 5번째 날짜 사용
            if date:
                cur.execute("SELECT max(date) FROM public.daily_ranking WHERE date <= %s", (date,))
            else:
                cur.execute("SELECT max(date) FROM public.daily_ranking")
            today = cur.fetchone()[0]
            cur.execute(
                """
                SELECT DISTINCT date
                FROM public.daily_ranking
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT 10
                """,
                (today,),
            )
            recent_dates = [row[0] for row in cur.fetchall()]
            prev = recent_dates[4] if len(recent_dates) >= 5 else (recent_dates[-1] if recent_dates else today)
            cur.execute(
                """
                WITH latest AS (
                  SELECT code, name, sector, COALESCE(live_rank, rank_final) AS rank_final, final_score
                  FROM public.daily_ranking WHERE date = %s
                ),
                pv AS (
                  SELECT code, COALESCE(live_rank, rank_final) AS prev_rank
                  FROM public.daily_ranking WHERE date = %s
                )
                SELECT l.code, l.name, l.sector, l.rank_final, p.prev_rank, l.final_score
                FROM latest l JOIN pv p USING (code)
                """,
                (today, prev),
            )
            rows_today, prev_map = [], {}
            for code, name, sector, rank_final, prev_rank, final_score in cur.fetchall():
                rows_today.append({"code": code, "name": name, "sector": sector,
                                   "rank_final": rank_final, "final_score": final_score})
                prev_map[str(code)] = int(prev_rank)
        risers = _compute_risers(rows_today, prev_map, top_n)
        return {"today": str(today), "prev": str(prev), "risers": risers, "source": "db"}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TYPE_C: 시장 상태
# ---------------------------------------------------------------------------
def get_market_status(date: str | None) -> dict:
    """시장 상태: DB → 종가 산출물 파일(market_status_validation_report.json) → fixture 순."""
    if _db_available():
        try:
            return _get_market_status_db(date)
        except Exception as exc:  # noqa: BLE001
            log.warning("DB 시장상태 조회 실패(%s). 파일로 폴백합니다.", exc)
    # 파일 우선: 종가 배치가 생성하는 시장 검증 리포트(실데이터)
    ms = _get_market_status_file()
    if ms is not None:
        log.info("시장상태: market_status_validation_report.json 사용(실데이터 파일).")
        return ms
    log.info("시장상태: 실파일 없음 → fixture 사용.")
    fx = _load_fixture("market_status_sample.json")
    fx["source"] = "fixture"
    return fx


def _get_market_status_file() -> dict | None:
    """outputs/market_status_validation_report.json 의 latest 필드에서 시장 상태 구성."""
    path = ROOT / "outputs" / "market_status_validation_report.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        latest = report.get("latest")
        if not latest:
            return None
        close_gt_ma20 = latest.get("close_gt_ma20")
        return {
            "date": str(latest.get("date") or report.get("latest_date") or ""),
            "kospi_close": latest.get("kospi_close"),
            "kospi_ma20": latest.get("kospi_ma20"),
            "volatility_5d": latest.get("volatility_5d"),
            "foreign_net_5d": latest.get("foreign_net_5d"),
            # market_up 컬럼이 없으므로 close_gt_ma20으로 파생
            "market_up": bool(close_gt_ma20) if close_gt_ma20 is not None else None,
            "source": "file",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("시장상태 파일 파싱 실패(%s).", exc)
        return None


def _get_market_status_db(date: str | None) -> dict:
    import db
    conn = db.raw_psycopg2_conn()
    try:
        with conn.cursor() as cur:
            if date:
                cur.execute(
                    "SELECT date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up "
                    "FROM public.market_status WHERE date <= %s ORDER BY date DESC LIMIT 1", (date,))
            else:
                cur.execute(
                    "SELECT date, kospi_close, kospi_ma20, volatility_5d, foreign_net_5d, market_up "
                    "FROM public.market_status ORDER BY date DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            raise RuntimeError("market_status 데이터 없음")
        d, close, ma20, vol, fnet, up = row
        return {"date": str(d), "kospi_close": float(close) if close else None,
                "kospi_ma20": float(ma20) if ma20 else None,
                "volatility_5d": float(vol) if vol else None,
                "foreign_net_5d": float(fnet) if fnet else None,
                "market_up": bool(up), "source": "db"}
    finally:
        conn.close()
