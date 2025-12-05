import ast
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



# ===== 설정값 =====
# 언제부터 데이터를 가져올지 (YYYYMMDD)
START_DATE = "20240101"  # 필요하면 더 과거로 늘려도 됨
# 끝 날짜는 오늘
END_DATE = datetime.today().strftime("%Y%m%d")

# 네이버 비공식 API (siseJson.naver)
NAVER_URL = "https://api.finance.naver.com/siseJson.naver"


# ===== 경로 설정 =====
BASE_DIR = Path(__file__).resolve().parent  # lee_trader 루트
DATA_DIR = BASE_DIR / "data"
UNIVERSE_PATH = DATA_DIR / "universe.csv"
PRICES_DIR = DATA_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)
PRICES_PATH = PRICES_DIR / "prices_daily.csv"


def fetch_naver_daily_prices(code: str, start: str, end: str) -> pd.DataFrame:
    """
    네이버 금융에서 특정 종목(code)의 일별 시세 데이터를 가져온다.
    반환 컬럼: date(YYYY-MM-DD), code, close
    """
    params = {
        "symbol": code,
        "requestType": 1,
        "startTime": start,
        "endTime": end,
        "timeframe": "day",
    }

    try:
        resp = requests.get(NAVER_URL, params=params, timeout=10, verify=False)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] code={code} 요청 실패: {e}")
        return pd.DataFrame(columns=["date", "code", "close"])

    text = resp.text.strip()

    if not text or text == "[]":
        print(f"[WARN] code={code} 응답이 비어 있음")
        return pd.DataFrame(columns=["date", "code", "close"])

    try:
        # 응답 형태: [['날짜','시가','고가','저가','종가','거래량','외국인소진율'], [...], ...]
        data = ast.literal_eval(text)
    except Exception as e:
        print(f"[WARN] code={code} 응답 파싱 실패: {e}")
        return pd.DataFrame(columns=["date", "code", "close"])

    if not isinstance(data, list) or len(data) <= 1:
        print(f"[WARN] code={code} 유효한 데이터 없음")
        return pd.DataFrame(columns=["date", "code", "close"])

    header = data[0]
    rows = data[1:]

    # 날짜/종가 컬럼 인덱스 찾기 (보통 0:날짜, 4:종가)
    try:
        date_idx = header.index("날짜")
    except ValueError:
        date_idx = 0
    try:
        close_idx = header.index("종가")
    except ValueError:
        close_idx = 4

    out_rows = []
    for row in rows:
        try:
            date_raw = str(row[date_idx])
            close = float(row[close_idx])
        except Exception:
            continue

        # YYYYMMDD -> YYYY-MM-DD
        if len(date_raw) == 8 and date_raw.isdigit():
            date_str = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        else:
            # 혹시 다른 포맷이면 그대로 둔다
            date_str = date_raw

        out_rows.append(
            {
                "date": date_str,
                "code": code,
                "close": close,
            }
        )

    return pd.DataFrame(out_rows)


def load_universe_codes() -> List[str]:
    """
    node/data/universe.csv 에서 종목코드 목록만 추출.
    """
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"universe.csv not found: {UNIVERSE_PATH}")

    df = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    if "code" not in df.columns:
        raise ValueError("universe.csv 에 code 컬럼이 없습니다.")

    codes = (
        df["code"]
        .astype(str)
        .str.zfill(6)  # 6자리 코드 형식 맞추기
        .drop_duplicates()
        .tolist()
    )

    print(f"Universe 종목 수: {len(codes)}")
    return codes


def main():
    print("=== Naver 가격 수집 시작 ===")
    print(f"DATA_DIR      : {DATA_DIR}")
    print(f"UNIVERSE_PATH : {UNIVERSE_PATH}")
    print(f"PRICES_PATH   : {PRICES_PATH}")
    print(f"기간          : {START_DATE} ~ {END_DATE}")

    codes = load_universe_codes()
    all_dfs: List[pd.DataFrame] = []

    for i, code in enumerate(codes, start=1):
        print(f"[{i}/{len(codes)}] 코드 {code} 수집 중...")
        df_code = fetch_naver_daily_prices(code, START_DATE, END_DATE)
        if not df_code.empty:
            all_dfs.append(df_code)

        # 네이버 쪽에 너무 부담 안 주려고 약간 쉬어주자
        time.sleep(0.3)

    if not all_dfs:
        print("[ERROR] 수집된 데이터가 없습니다.")
        return

    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices.drop_duplicates(subset=["date", "code"])
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICES_PATH, index=False, encoding="utf-8-sig")

    print("=== 수집 완료 ===")
    print(f"총 행수: {len(prices)}")
    print(f"저장 위치: {PRICES_PATH}")


if __name__ == "__main__":
    main()
