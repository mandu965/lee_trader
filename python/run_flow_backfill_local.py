"""
run_flow_backfill_local.py

로컬 환경에서 KIS 수급 데이터를 과거 날짜 범위로 백필하는 스크립트.
.env 파일을 자동으로 로드한다.

사용법:
    python python/run_flow_backfill_local.py --start 20250101 --end 20260513
    python python/run_flow_backfill_local.py --start 20260101 --end 20260513  # 최근 5개월
"""
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """
    .env 파일을 파싱해 os.environ에 주입한다.
    python-dotenv가 인라인 주석을 제거하지 않는 환경을 대비해 직접 파싱.
    이미 설정된 환경변수는 덮어쓰지 않는다 (override=False 동작).
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            # 인라인 주석 제거: 공백 + # 이후는 주석으로 처리
            # 단, 따옴표로 감싼 값은 그대로 유지
            if value.startswith('"'):
                end = value.find('"', 1)
                value = value[1:end] if end != -1 else value[1:]
            elif value.startswith("'"):
                end = value.find("'", 1)
                value = value[1:end] if end != -1 else value[1:]
            else:
                # 공백 + # 패턴을 주석 시작으로 간주
                import re
                value = re.sub(r"\s+#.*$", "", value)
                value = value.strip()
            if key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KIS 수급 데이터 로컬 백필")
    parser.add_argument("--start", required=True, help="시작일 (YYYYMMDD, 예: 20250101)")
    parser.add_argument("--end", required=True, help="종료일 (YYYYMMDD, 예: 20260513)")
    return parser.parse_args()


def main() -> int:
    _load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    os.environ["ENABLE_FLOW_INGESTION"] = "1"
    os.environ["FLOW_START_DATE"] = args.start
    os.environ["FLOW_END_DATE"] = args.end

    logging.info("Flow backfill 시작: start=%s end=%s", args.start, args.end)

    from run_pipeline import create_model_run_id, maybe_run_flow_ingestion
    run_id = create_model_run_id()
    maybe_run_flow_ingestion(run_id)

    logging.info("Flow backfill 완료")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "python"))
    raise SystemExit(main())
