"""
fetch_flow_pykrx.py

pykrx + KRX 데이터포털 자격증명으로 투자자별 순매수 데이터를 수집하고
flow_daily 테이블에 upsert한다.

KIS API(최대 90일)와 달리 수년치 과거 데이터 조회 가능.
날짜 1개당 전 종목 일괄 조회 → API 부담 최소.

선행 조건:
    .env 에 KRX_ID, KRX_PW 설정 필요 (data.krx.co.kr 무료 회원가입)

사용법:
    python python/fetch_flow_pykrx.py --start 20250101 --end 20260513
    python python/fetch_flow_pykrx.py --start 20240101 --end 20260513 --delay 2.0
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

SOURCE_ENDPOINT = "pykrx/get_market_trading_value_by_investor"
FETCH_STATUS_SUCCESS = "success"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            if value.startswith('"'):
                end = value.find('"', 1)
                value = value[1:end] if end != -1 else value[1:]
            elif value.startswith("'"):
                end = value.find("'", 1)
                value = value[1:end] if end != -1 else value[1:]
            else:
                value = re.sub(r"\s+#.*$", "", value).strip()
            if key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KRX 투자자별 수급 데이터 수집 (pykrx)")
    parser.add_argument("--start", required=True, help="시작일 YYYYMMDD")
    parser.add_argument("--end", required=True, help="종료일 YYYYMMDD")
    parser.add_argument(
        "--codes",
        default="",
        help="수집 대상 종목코드 콤마 구분 (생략 시 universe.csv 또는 DB 유니버스 사용)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="날짜 간 요청 딜레이(초) (기본값: 1.5, KRX 부하 방지)",
    )
    return parser.parse_args()


def iter_business_dates(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    dates = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def load_target_codes(raw_codes: str) -> Optional[List[str]]:
    if raw_codes.strip():
        codes = [c.strip().zfill(6) for c in raw_codes.split(",") if c.strip()]
        logging.info("CLI 지정 종목: %d개", len(codes))
        return codes

    # 1순위: flow_daily에 이미 수집된 종목 (KIS 백필 기준 전체 유니버스)
    try:
        from db import get_engine
        import sqlalchemy as sa
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sa.text(
                "SELECT DISTINCT code FROM flow_daily WHERE fetch_status='success' ORDER BY code"
            ))
            codes = [str(row[0]).zfill(6) for row in result]
            if codes:
                logging.info("flow_daily 기존 종목 %d개 로드 (KIS 유니버스 기준)", len(codes))
                return codes
    except Exception as exc:
        logging.warning("flow_daily 종목 로드 실패: %s", exc)

    # 2순위: stocks 테이블
    try:
        from db import get_engine
        import sqlalchemy as sa
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sa.text(
                "SELECT code FROM stocks WHERE delisted_at IS NULL ORDER BY code"
            ))
            codes = [str(row[0]).zfill(6) for row in result]
            if codes:
                logging.info("DB stocks에서 %d개 종목 로드", len(codes))
                return codes
    except Exception as exc:
        logging.warning("DB stocks 로드 실패: %s", exc)

    # 3순위: universe.csv (일별 재생성으로 종목수 변동 가능)
    universe_path = ROOT / "data" / "universe.csv"
    if universe_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(universe_path, dtype={"code": str})
            if "code" in df.columns:
                codes = [str(c).zfill(6) for c in df["code"].dropna() if str(c).strip()]
                if codes:
                    logging.info("universe.csv에서 %d개 종목 로드", len(codes))
                    return sorted(set(codes))
        except Exception as exc:
            logging.warning("universe.csv 로드 실패: %s", exc)

    logging.info("유니버스 없음 → 전 종목 저장")
    return None


def _hash(raw: Dict[str, Any]) -> str:
    s = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def fetch_one_day(date_str: str) -> Dict[str, Dict[str, Any]]:
    """
    date_str(YYYYMMDD) 하루치 전 종목 외국인+기관 순매수 반환.
    날짜 1개 × 시장 2개 × 투자자 2종 = 4회 HTTP 요청.

    반환: {code6: {"foreign_amount": float|None, "foreign_volume": float|None,
                   "inst_amount": float|None,    "inst_volume": float|None}}
    """
    from pykrx import stock as pykrx_stock

    # 실제 컬럼명 (pykrx 1.2.7 확인)
    AMOUNT_COL = "순매수거래대금"
    VOLUME_COL = "순매수거래량"

    result: Dict[str, Dict[str, Any]] = {}

    for market in ("KOSPI", "KOSDAQ"):
        for investor, key_prefix in (("외국인", "foreign"), ("기관합계", "inst")):
            try:
                df = pykrx_stock.get_market_net_purchases_of_equities_by_ticker(
                    date_str, date_str, market=market, investor=investor
                )
            except Exception as exc:
                logging.warning(
                    "[pykrx] date=%s market=%s investor=%s 실패: %s",
                    date_str, market, investor, exc,
                )
                continue

            if df is None or df.empty:
                continue

            for ticker in df.index:
                code = str(ticker).zfill(6)
                if code not in result:
                    result[code] = {
                        "foreign_amount": None,
                        "foreign_volume": None,
                        "inst_amount": None,
                        "inst_volume": None,
                    }
                try:
                    result[code][f"{key_prefix}_amount"] = float(df.at[ticker, AMOUNT_COL])
                except Exception:
                    pass
                try:
                    result[code][f"{key_prefix}_volume"] = float(df.at[ticker, VOLUME_COL])
                except Exception:
                    pass

    return result


def build_rows(
    date_str: str,
    day_data: Dict[str, Dict[str, Any]],
    target_codes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    collected_at = datetime.utcnow().isoformat()
    target_set = set(target_codes) if target_codes else None
    business_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    rows = []

    for code, vals in day_data.items():
        if target_set and code not in target_set:
            continue

        raw_payload = {"date": date_str, "code": code, "source": SOURCE_ENDPOINT, **vals}

        for investor_type, amount_key, volume_key in (
            ("foreign", "foreign_amount", "foreign_volume"),
            ("institution", "inst_amount", "inst_volume"),
        ):
            rows.append({
                "date": business_date,
                "code": code,
                "investor_type": investor_type,
                "net_buy_amount": vals.get(amount_key),
                "net_buy_volume": vals.get(volume_key),
                "raw_payload_hash": _hash(raw_payload),
                "collected_at": collected_at,
                "source_endpoint": SOURCE_ENDPOINT,
                "market_div_code": "KRX",
                "input_date": date_str,
                "tr_id": "pykrx",
                "raw_payload_json": json.dumps(raw_payload, ensure_ascii=False),
                "created_run_id": None,
                "fetch_status": FETCH_STATUS_SUCCESS,
                "error_code": None,
                "error_message": None,
                "is_partial_page": False,
            })
    return rows


def upsert_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    try:
        from db import get_engine
        import sqlalchemy as sa
        engine = get_engine()
        sql = sa.text("""
            INSERT INTO flow_daily (
                date, code, investor_type,
                net_buy_amount, net_buy_volume, raw_payload_hash,
                collected_at, source_endpoint, market_div_code,
                input_date, tr_id, raw_payload_json,
                created_run_id, fetch_status, error_code, error_message, is_partial_page
            ) VALUES (
                :date, :code, :investor_type,
                :net_buy_amount, :net_buy_volume, :raw_payload_hash,
                :collected_at, :source_endpoint, :market_div_code,
                :input_date, :tr_id, CAST(:raw_payload_json AS JSONB),
                :created_run_id, :fetch_status, :error_code, :error_message, :is_partial_page
            )
            ON CONFLICT (date, code, investor_type) DO UPDATE SET
                net_buy_amount   = EXCLUDED.net_buy_amount,
                net_buy_volume   = EXCLUDED.net_buy_volume,
                raw_payload_hash = EXCLUDED.raw_payload_hash,
                collected_at     = EXCLUDED.collected_at,
                source_endpoint  = EXCLUDED.source_endpoint,
                fetch_status     = EXCLUDED.fetch_status,
                raw_payload_json = EXCLUDED.raw_payload_json
        """)
        with engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)
    except Exception as exc:
        logging.error("upsert 실패: %s", exc)
        return 0


def main() -> int:
    _load_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # KRX 자격증명 확인
    krx_id = os.environ.get("KRX_ID", "").strip()
    krx_pw = os.environ.get("KRX_PW", "").strip()
    if not krx_id or not krx_pw:
        logging.error(
            "KRX_ID / KRX_PW 환경변수가 설정되지 않았습니다.\n"
            "data.krx.co.kr 무료 회원가입 후 .env에 추가하세요:\n"
            "  KRX_ID=your_id\n"
            "  KRX_PW=your_password"
        )
        return 1

    # pykrx가 KRX_ID/KRX_PW를 환경변수로 읽음
    os.environ["KRX_ID"] = krx_id
    os.environ["KRX_PW"] = krx_pw

    args = parse_args()
    target_codes = load_target_codes(args.codes)

    dates = iter_business_dates(args.start, args.end)
    logging.info("수집 기간: %s ~ %s (%d 영업일)", args.start, args.end, len(dates))

    total_upserted = 0
    failed_dates: List[str] = []

    for i, date_str in enumerate(dates):
        try:
            day_data = fetch_one_day(date_str)
            if not day_data:
                logging.info("[%d/%d] %s 데이터 없음 (휴장일 또는 미제공)", i + 1, len(dates), date_str)
                time.sleep(args.delay)
                continue

            rows = build_rows(date_str, day_data, target_codes)
            n = upsert_rows(rows)
            total_upserted += n
            logging.info(
                "[%d/%d] %s 완료 — 종목=%d, upsert=%d",
                i + 1, len(dates), date_str, len(day_data), n,
            )
        except Exception as exc:
            logging.error("[%d/%d] %s 실패: %s", i + 1, len(dates), date_str, exc)
            failed_dates.append(date_str)

        time.sleep(args.delay)

    logging.info(
        "완료: 총 upsert=%d, 실패=%d일 %s",
        total_upserted, len(failed_dates),
        failed_dates if failed_dates else "",
    )
    return 0 if not failed_dates else 1


if __name__ == "__main__":
    raise SystemExit(main())
