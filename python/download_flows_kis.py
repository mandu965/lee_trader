import argparse
import csv
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from kis_client import KISAuthError, KISBusinessError, KISClient, KISHTTPError, KISResponse

try:
    from db import get_engine
except Exception:
    get_engine = None


API_PATH = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
TR_ID = "FHPTJ04160001"
RAW_DIR = Path("output") / "kis_flow_raw"
FAILURE_CSV = Path("output") / "flow_ingestion_failures.csv"
FETCH_STATUS_SUCCESS = "success"
FETCH_STATUS_SKIPPED = "skipped"
FETCH_STATUS_FAILED = "failed"
FLOW_DAILY_COLUMNS = [
    "date",
    "code",
    "investor_type",
    "net_buy_amount",
    "net_buy_volume",
    "raw_payload_hash",
    "collected_at",
    "source_endpoint",
    "market_div_code",
    "input_date",
    "tr_id",
    "raw_payload_json",
    "created_run_id",
    "fetch_status",
    "error_code",
    "error_message",
    "is_partial_page",
]


@dataclass
class FailureRecord:
    code: str
    error_type: str
    error_message: str
    failed_at: str
    stage: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KIS investor flow raw payloads.")
    parser.add_argument("--codes", required=True, help="Comma-separated 6-digit stock codes")
    parser.add_argument("--start-date", dest="start_date", help="YYYYMMDD")
    parser.add_argument("--end-date", dest="end_date", help="YYYYMMDD")
    parser.add_argument("--dry-run", action="store_true", help="Sample run for 3-5 recent trading days")
    parser.add_argument("--save-raw", action="store_true", help="Save raw payload JSON files")
    return parser.parse_args()


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_CSV.parent.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_codes(raw_codes: str) -> List[str]:
    codes = [part.strip().zfill(6) for part in raw_codes.split(",") if part.strip()]
    if not codes:
        raise ValueError("No valid codes supplied")
    return codes


def infer_dates(args: argparse.Namespace) -> tuple[str, str]:
    if args.start_date and args.end_date:
        return args.start_date, args.end_date
    if args.dry_run:
        end = datetime.today()
        start = end - timedelta(days=7)
        return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    raise ValueError("start-date and end-date are required unless --dry-run is set")


def iter_business_dates(start_date: str, end_date: str) -> List[str]:
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    if end_dt < start_dt:
        raise ValueError("end-date must be >= start-date")

    dates: List[str] = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def get_tr_cont(response: KISResponse) -> str:
    headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return headers.get("tr_cont", "").strip().upper()


def save_raw_payload(*, code: str, page_no: int, params: Dict[str, Any], response: KISResponse) -> Path:
    ts = now_utc().strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{code}_{params['FID_INPUT_DATE_1']}_page{page_no}_{ts}.json"
    payload = {
        "saved_at_utc": ts,
        "request": {
            "path": API_PATH,
            "tr_id": TR_ID,
            "params": params,
        },
        "response": {
            "status_code": response.status_code,
            "headers": response.headers,
            "body": response.data,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_date_string(value: Any) -> Optional[str]:
    text_value = str(value or "").strip()
    if len(text_value) != 8 or not text_value.isdigit():
        return None
    return f"{text_value[:4]}-{text_value[4:6]}-{text_value[6:]}"


def _hash_raw_row(raw_row: Dict[str, Any]) -> str:
    serialized = json.dumps(raw_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_flow_daily_row(
    *,
    code: str,
    investor_type: str,
    business_date: str,
    input_date: str,
    market_div_code: str,
    net_buy_amount: Optional[float],
    net_buy_volume: Optional[float],
    raw_row: Dict[str, Any],
    is_partial_page: bool,
    fetch_status: str,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    collected_at = now_utc().isoformat()
    run_id = os.getenv("RUN_ID")
    created_run_id = int(run_id) if run_id and run_id.isdigit() else None
    raw_payload_json = {
        "response_row": raw_row,
        "mapping_scope": "foreign_institution_v1",
    }
    return {
        "date": business_date,
        "code": code,
        "investor_type": investor_type,
        "net_buy_amount": net_buy_amount,
        "net_buy_volume": net_buy_volume,
        "raw_payload_hash": _hash_raw_row(raw_row),
        "collected_at": collected_at,
        "source_endpoint": API_PATH,
        "market_div_code": market_div_code,
        "input_date": input_date,
        "tr_id": TR_ID,
        "raw_payload_json": json.dumps(raw_payload_json, ensure_ascii=False),
        "created_run_id": created_run_id,
        "fetch_status": fetch_status,
        "error_code": error_code,
        "error_message": error_message,
        "is_partial_page": is_partial_page,
    }


def normalize_flow_rows(
    *,
    code: str,
    payload: Dict[str, Any],
    input_date: str,
    market_div_code: str,
    is_partial_page: bool,
) -> List[Dict[str, Any]]:
    body_rows = payload.get("output2") or payload.get("output1") or []
    if isinstance(body_rows, dict):
        body_rows = [body_rows]
    if not isinstance(body_rows, list):
        logging.warning("[normalize] code=%s input_date=%s response body is not a list", code, input_date)
        return []

    normalized: List[Dict[str, Any]] = []
    for raw_row in body_rows:
        business_date = _normalize_date_string(raw_row.get("stck_bsop_date"))
        if not business_date:
            logging.warning("[normalize] code=%s input_date=%s missing stck_bsop_date; row skipped", code, input_date)
            continue

        foreign_amount = _to_float(raw_row.get("frgn_ntby_tr_pbmn"))
        foreign_volume = _to_float(raw_row.get("frgn_ntby_qty"))
        institution_amount = _to_float(raw_row.get("orgn_ntby_tr_pbmn"))
        institution_volume = _to_float(raw_row.get("orgn_ntby_qty"))

        if (
            foreign_amount is None
            and foreign_volume is None
            and institution_amount is None
            and institution_volume is None
        ):
            logging.warning(
                "[normalize] code=%s date=%s foreign/institution mapping unavailable; raw kept, normalized rows skipped",
                code,
                business_date,
            )
            continue

        normalized.append(
            build_flow_daily_row(
                code=code,
                investor_type="foreign",
                business_date=business_date,
                input_date=input_date,
                market_div_code=market_div_code,
                net_buy_amount=foreign_amount,
                net_buy_volume=foreign_volume,
                raw_row=raw_row,
                is_partial_page=is_partial_page,
                fetch_status=FETCH_STATUS_SUCCESS,
            )
        )
        normalized.append(
            build_flow_daily_row(
                code=code,
                investor_type="institution",
                business_date=business_date,
                input_date=input_date,
                market_div_code=market_div_code,
                net_buy_amount=institution_amount,
                net_buy_volume=institution_volume,
                raw_row=raw_row,
                is_partial_page=is_partial_page,
                fetch_status=FETCH_STATUS_SUCCESS,
            )
        )
    return normalized


def fetch_code_flow(
    client: KISClient,
    *,
    code: str,
    start_date: str,
    end_date: str,
    save_raw: bool,
) -> tuple[List[Dict[str, Any]], List[Path]]:
    all_rows: List[Dict[str, Any]] = []
    saved_files: List[Path] = []
    market_div_code = "J"

    for business_date in iter_business_dates(start_date, end_date):
        params = {
            "FID_COND_MRKT_DIV_CODE": market_div_code,
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": business_date,
            "FID_ORG_ADJ_PRC": "",
            "FID_ETC_CLS_CODE": "",
        }
        tr_cont = ""
        page_no = 1

        logging.info("[fetch] code=%s input_date=%s page=%d start", code, business_date, page_no)
        while True:
            response = client.get_with_meta(
                API_PATH,
                tr_id=TR_ID,
                params=params,
                tr_cont=tr_cont,
            )
            if save_raw:
                saved_files.append(save_raw_payload(code=code, page_no=page_no, params=params, response=response))

            current_tr_cont = get_tr_cont(response)
            is_partial_page = current_tr_cont in {"M", "F"}
            normalized_rows = normalize_flow_rows(
                code=code,
                payload=response.data,
                input_date=business_date,
                market_div_code=market_div_code,
                is_partial_page=is_partial_page,
            )
            all_rows.extend(normalized_rows)
            logging.info(
                "[normalize] code=%s input_date=%s page=%d normalized_rows=%d partial=%s",
                code,
                business_date,
                page_no,
                len(normalized_rows),
                is_partial_page,
            )

            if not is_partial_page:
                break
            tr_cont = "N"
            page_no += 1
            logging.info("[fetch] code=%s input_date=%s paging continue page=%d", code, business_date, page_no)

    return all_rows, saved_files


def write_failures_csv(failures: List[FailureRecord]) -> None:
    if not failures:
        if FAILURE_CSV.exists():
            FAILURE_CSV.unlink()
        return

    with FAILURE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "error_type", "error_message", "failed_at", "stage"],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure.__dict__)


def _flow_upsert_sql() -> str:
    return """
        INSERT INTO flow_daily (
            date,
            code,
            investor_type,
            net_buy_amount,
            net_buy_volume,
            raw_payload_hash,
            collected_at,
            source_endpoint,
            market_div_code,
            input_date,
            tr_id,
            raw_payload_json,
            created_run_id,
            fetch_status,
            error_code,
            error_message,
            is_partial_page
        ) VALUES (
            :date,
            :code,
            :investor_type,
            :net_buy_amount,
            :net_buy_volume,
            :raw_payload_hash,
            :collected_at,
            :source_endpoint,
            :market_div_code,
            :input_date,
            :tr_id,
            CAST(:raw_payload_json AS JSONB),
            :created_run_id,
            :fetch_status,
            :error_code,
            :error_message,
            :is_partial_page
        )
        ON CONFLICT (date, code, investor_type)
        DO UPDATE SET
            net_buy_amount = EXCLUDED.net_buy_amount,
            net_buy_volume = EXCLUDED.net_buy_volume,
            raw_payload_hash = EXCLUDED.raw_payload_hash,
            collected_at = EXCLUDED.collected_at,
            source_endpoint = EXCLUDED.source_endpoint,
            market_div_code = EXCLUDED.market_div_code,
            input_date = EXCLUDED.input_date,
            tr_id = EXCLUDED.tr_id,
            raw_payload_json = EXCLUDED.raw_payload_json,
            created_run_id = EXCLUDED.created_run_id,
            fetch_status = EXCLUDED.fetch_status,
            error_code = EXCLUDED.error_code,
            error_message = EXCLUDED.error_message,
            is_partial_page = EXCLUDED.is_partial_page
    """


def upsert_flow_rows(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        logging.info("[load] no normalized rows to upsert")
        return 0
    if not get_engine:
        logging.warning("[load] DB engine unavailable; skipping flow_daily upsert")
        return 0

    try:
        engine = get_engine()
        stmt = text(_flow_upsert_sql())
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        logging.info("[load] upserted flow_daily rows=%d", len(rows))
        return len(rows)
    except Exception as exc:
        logging.warning("[load] flow_daily upsert skipped: %s", exc)
        return 0


def count_flow_rows(*, codes: List[str], start_date: str, end_date: str) -> int:
    if not get_engine:
        return 0
    try:
        engine = get_engine()
        stmt = text(
            """
            SELECT COUNT(*)
            FROM flow_daily
            WHERE code = ANY(:codes)
              AND date BETWEEN :start_date AND :end_date
            """
        )
        with engine.begin() as conn:
            result = conn.execute(
                stmt,
                {
                    "codes": codes,
                    "start_date": _normalize_date_string(start_date),
                    "end_date": _normalize_date_string(end_date),
                },
            )
            return int(result.scalar_one())
    except Exception as exc:
        logging.warning("[check] flow_daily row count unavailable: %s", exc)
        return 0


def sample_flow_row_structure(*, codes: List[str], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    if not get_engine:
        return []
    try:
        engine = get_engine()
        stmt = text(
            """
            SELECT date, code, COUNT(*) AS row_count, STRING_AGG(investor_type, ',' ORDER BY investor_type) AS investor_types
            FROM flow_daily
            WHERE code = ANY(:codes)
              AND date BETWEEN :start_date AND :end_date
            GROUP BY date, code
            ORDER BY date DESC, code
            LIMIT 10
            """
        )
        with engine.begin() as conn:
            rows = conn.execute(
                stmt,
                {
                    "codes": codes,
                    "start_date": _normalize_date_string(start_date),
                    "end_date": _normalize_date_string(end_date),
                },
            ).mappings().all()
            return [dict(row) for row in rows]
    except Exception as exc:
        logging.warning("[check] flow_daily sample structure unavailable: %s", exc)
        return []


def main() -> int:
    setup_logging()
    ensure_output_dirs()
    args = parse_args()
    codes = parse_codes(args.codes)
    start_date, end_date = infer_dates(args)
    failures: List[FailureRecord] = []

    logging.info(
        "Starting KIS flow download: codes=%s start=%s end=%s dry_run=%s save_raw=%s",
        ",".join(codes),
        start_date,
        end_date,
        args.dry_run,
        args.save_raw,
    )

    try:
        client = KISClient.from_env()
        client.issue_access_token()
    except KISAuthError as exc:
        failures.append(
            FailureRecord(
                code="__GLOBAL__",
                error_type="KISAuthError",
                error_message=str(exc),
                failed_at=now_utc().isoformat(),
                stage="init_client",
            )
        )
        write_failures_csv(failures)
        logging.error("Authentication failed: %s", exc)
        raise
    except Exception as exc:
        failures.append(
            FailureRecord(
                code="__GLOBAL__",
                error_type=type(exc).__name__,
                error_message=str(exc),
                failed_at=now_utc().isoformat(),
                stage="init_client",
            )
        )
        write_failures_csv(failures)
        logging.error("Unable to initialize KIS client: %s", exc)
        raise

    total_normalized_rows = 0
    total_raw_files = 0
    total_upserted_rows = 0

    for code in codes:
        try:
            normalized_rows, raw_files = fetch_code_flow(
                client,
                code=code,
                start_date=start_date,
                end_date=end_date,
                save_raw=args.save_raw,
            )
            upserted_rows = upsert_flow_rows(normalized_rows)
            total_normalized_rows += len(normalized_rows)
            total_raw_files += len(raw_files)
            total_upserted_rows += upserted_rows
            logging.info(
                "[summary] code=%s normalized_rows=%d upserted_rows=%d raw_files=%d",
                code,
                len(normalized_rows),
                upserted_rows,
                len(raw_files),
            )
        except KISBusinessError as exc:
            failures.append(
                FailureRecord(
                    code=code,
                    error_type="KISBusinessError",
                    error_message=str(exc),
                    failed_at=now_utc().isoformat(),
                    stage="fetch_code_flow",
                )
            )
            logging.warning("[fetch] per-code business failure: code=%s error=%s", code, exc)
            continue
        except KISHTTPError as exc:
            failures.append(
                FailureRecord(
                    code=code,
                    error_type="KISHTTPError",
                    error_message=str(exc),
                    failed_at=now_utc().isoformat(),
                    stage="fetch_code_flow",
                )
            )
            logging.error("[fetch] global HTTP failure, aborting batch: code=%s error=%s", code, exc)
            write_failures_csv(failures)
            raise
        except Exception as exc:
            failures.append(
                FailureRecord(
                    code=code,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    failed_at=now_utc().isoformat(),
                    stage="fetch_code_flow",
                )
            )
            logging.error("[fetch] global unexpected failure, aborting batch: code=%s error=%s", code, exc)
            write_failures_csv(failures)
            raise

    write_failures_csv(failures)

    db_row_count = count_flow_rows(codes=codes, start_date=start_date, end_date=end_date)
    logging.info(
        "KIS flow download finished: codes=%d total_normalized_rows=%d total_upserted_rows=%d total_raw_files=%d failures=%d db_row_count=%d",
        len(codes),
        total_normalized_rows,
        total_upserted_rows,
        total_raw_files,
        len(failures),
        db_row_count,
    )

    sample_rows = sample_flow_row_structure(codes=codes, start_date=start_date, end_date=end_date)
    if sample_rows:
        for row in sample_rows:
            logging.info(
                "[check] date=%s code=%s row_count=%s investor_types=%s",
                row["date"],
                row["code"],
                row["row_count"],
                row["investor_types"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
