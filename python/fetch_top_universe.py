import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import sqlite3

DATA_DIR = Path("data")
UNIVERSE_CSV = DATA_DIR / "universe.csv"
SECTORS_CSV = DATA_DIR / "sectors.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
try:
    from db import get_engine
except Exception:
    get_engine = None

# pykrx is required (added in requirements.txt)
try:
    from pykrx import stock
except Exception as e:
    stock = None

# FinanceDataReader for sector/industry metadata
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def last_trading_date(max_back_days: int = 10) -> str:
    """
    pykrx가 비영업일이거나 장 시작 전에는 '시가총액=0'인 DF를 돌려줄 수 있어서,
    실제로 시가총액 값이 존재하는 날짜만 거래일로 인정한다.
    반환 형식: YYYYMMDD
    """
    base = datetime.today()
    for i in range(max_back_days):
        dt = base - timedelta(days=i)
        ymd = dt.strftime("%Y%m%d")
        try:
            df = stock.get_market_cap_by_ticker(ymd, market="KOSPI")
            if df is None or df.empty:
                continue

            # 시가총액 컬럼 탐색 (top_by_market와 동일한 로직 일부 재사용)
            mcap_col = None
            cols = list(df.columns)

            if "시가총액" in cols:
                mcap_col = "시가총액"
            else:
                for c in cols:
                    if "시가총" in str(c):
                        mcap_col = c
                        break

            if mcap_col is None:
                # numeric 컬럼 중 '상장주식수'는 제외하고 사용
                num_cols = df.select_dtypes(include="number")
                num_cols = num_cols[[c for c in num_cols.columns if "상장주" not in str(c)]]
                if not num_cols.empty:
                    mcap_col = num_cols.sum().idxmax()

            # 시가총액 후보 컬럼이 없거나, 해당 컬럼이 전부 0이면 이 날짜는 패스
            if mcap_col is None:
                continue

            mcap_series = df[mcap_col]
            # 상장주식수만 살아있고, 다른 값이 전부 0인 케이스를 필터링
            if (mcap_series.fillna(0) == 0).all():
                # logging.debug(f"{ymd}: mcap_col '{mcap_col}' is all zero, skip")
                continue

            # 여기까지 왔으면 유효한 거래일
            return ymd

        except Exception:
            continue

    # 실패 시 오늘 날짜라도 반환 (fallback)
    return base.strftime("%Y%m%d")



def top_by_market(ymd: str, market: str, top_n: int) -> pd.DataFrame:
    """
    market: 'KOSPI' or 'KOSDAQ'
    반환: DataFrame with columns ['code', 'name', 'market']
    실제 '시가총액' 기준으로 상위 종목만 뽑도록 컬럼 탐색 로직을 강화했다.
    """
    df = stock.get_market_cap_by_ticker(ymd, market=market)

    # ------------ DEBUG LOGS ------------
    if df is None or df.empty:
        logging.error(f"[DEBUG] {market} DataFrame is EMPTY")
    else:
        logging.info(f"[DEBUG] MARKET={market} RAW COLUMNS = {list(df.columns)}")
        logging.info(f"[DEBUG] MARKET={market} HEAD = \n{df.head(3)}")
    # ------------------------------------


    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "market"])

    mcap_col = None
    cols = list(df.columns)

    # 1) 정확히 '시가총액' 이면 최우선
    if "시가총액" in cols:
        mcap_col = "시가총액"
    else:
        # 2) '시가총' 이라는 문자열이 들어간 컬럼 (예: '시가총액(보통주)')
        for c in cols:
            if "시가총" in str(c):
                mcap_col = c
                break

        # 3) 영문 cap 관련 컬럼 탐색 (mktcap, market_cap 등)
        if mcap_col is None:
            for c in cols:
                cl = str(c).lower()
                # 'mkt'와 'cap' 둘 다 들어가는 경우
                if "mkt" in cl and "cap" in cl:
                    mcap_col = c
                    break

        if mcap_col is None:
            for c in cols:
                cl = str(c).lower()
                # 단순히 cap 으로 끝나는 숫자형 컬럼 (free cap 등은 제외)
                if cl.endswith("cap") and "free" not in cl:
                    mcap_col = c
                    break

        # 4) 최후의 수단: 숫자형 컬럼 중 합계가 가장 큰 컬럼을 시총으로 추정
        if mcap_col is None:
            num_cols = df.select_dtypes(include="number")
            if not num_cols.empty:
                mcap_col = num_cols.sum().idxmax()

    # 🔥 바로 여기 넣어라! (가장 중요)
    logging.info(f"[DEBUG] MARKET={market} > USING MCAP COLUMN = {mcap_col}")
    
    if mcap_col is None:
        # 진짜로 시가총액 컬럼을 못 찾는 경우 → 인덱스 순서로 fallback
        logging.warning(
            "[top_by_market] %s: 시가총액 컬럼을 찾지 못했습니다. index 순서 기준으로 상위 %d개 사용.",
            market,
            top_n,
        )
        codes = df.index.astype(str).str.zfill(6).tolist()[:top_n]
    else:
        # 찾은 시가총액 컬럼 기준으로 내림차순 정렬
        logging.info(
            "[top_by_market] %s: 시가총액 컬럼 '%s' 사용, 상위 %d개 추출",
            market,
            mcap_col,
            top_n,
        )
        df_sorted = df.sort_values(mcap_col, ascending=False)
        codes = df_sorted.index.astype(str).str.zfill(6).tolist()[:top_n]

    names = [stock.get_market_ticker_name(c) for c in codes]
    out = pd.DataFrame({"code": codes, "name": names})
    out["market"] = market
    return out



def main():
    setup_logging()
    ensure_data_dir()

    if stock is None:
        logging.error("pykrx is not installed. Please add 'pykrx' to requirements and rebuild the image.")
        return

    ymd = last_trading_date()
    logging.info(f"Using trading date: {ymd}")

    try:
        kospi_top = top_by_market(ymd, "KOSPI", top_n=100)
        logging.info(f"KOSPI top fetched: {len(kospi_top)}")
    except Exception as e:
        logging.exception(f"Failed to fetch KOSPI top: {e}")
        kospi_top = pd.DataFrame(columns=["code", "name"])

    try:
        kosdaq_top = top_by_market(ymd, "KOSDAQ", top_n=100)
        logging.info(f"KOSDAQ top fetched: {len(kosdaq_top)}")
    except Exception as e:
        logging.exception(f"Failed to fetch KOSDAQ top: {e}")
        kosdaq_top = pd.DataFrame(columns=["code", "name"])

    uni = pd.concat([kospi_top, kosdaq_top], ignore_index=True)
    # 중복 제거, 코드 기준 우선 유지
    uni = uni.drop_duplicates(subset=["code"]).reset_index(drop=True)
    # 컬럼 정리 및 대문자화
    if "market" in uni.columns:
        uni["market"] = uni["market"].astype(str).str.upper().str.strip()
    else:
        uni["market"] = ""
    # 이전 universe.csv에서 sector 보존을 위한 맵 구성
    old_sector_map = {}
    try:
        if UNIVERSE_CSV.exists():
            old = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
            old["code"] = old["code"].astype(str).str.zfill(6)
            if "sector" in old.columns:
                old_sector_map = dict(zip(old["code"], old["sector"].fillna("").astype(str)))
    except Exception:
        pass

    # sector 병합: data/sectors.csv(code, sector) 존재 시 left-merge
    if SECTORS_CSV.exists():
        try:
            s = pd.read_csv(SECTORS_CSV, dtype={"code": str})
            s["code"] = s["code"].astype(str).str.zfill(6)
            if "sector" in s.columns:
                uni = uni.merge(s[["code", "sector"]], on="code", how="left")
            else:
                uni["sector"] = ""
        except Exception:
            uni["sector"] = ""
    else:
        # sector 미제공 시 기본값
        if "sector" not in uni.columns:
            uni["sector"] = ""
        else:
            uni["sector"] = uni["sector"].fillna("").astype(str)

    # FDR 메타 병합: 비어있는 sector를 FDR(KRX/KOSPI/KOSDAQ) 메타로 보강
    if fdr is not None:
        try:
            metas = []
            for market_id in ["KRX", "KOSPI", "KOSDAQ"]:
                try:
                    m = fdr.StockListing(market_id)
                    if m is not None and not m.empty:
                        m["__market_id__"] = market_id
                        metas.append(m)
                except Exception:
                    continue
            if metas:
                meta = pd.concat(metas, ignore_index=True)
                # 코드 컬럼 탐지
                code_col = None
                for c in ["Code", "Symbol", "종목코드", "Ticker"]:
                    if c in meta.columns:
                        code_col = c
                        break
                # 섹터/산업 컬럼 후보 탐지
                sector_col = None
                # 우선순위: Sector, Industry, 업종, 섹터
                for c in meta.columns:
                    cl = str(c).lower()
                    if "sector" in cl or c in ["Sector"]:
                        sector_col = c
                        break
                if sector_col is None:
                    for c in meta.columns:
                        cl = str(c).lower()
                        if "industry" in cl or c in ["Industry", "업종", "섹터", "산업"]:
                            sector_col = c
                            break
                if code_col and sector_col:
                    meta = meta[[code_col, sector_col]].rename(columns={code_col: "code", sector_col: "sector_fdr"})
                    meta["code"] = meta["code"].astype(str).str.zfill(6)
                    # merge 후 빈 sector만 FDR 값으로 채움
                    uni = uni.merge(meta, on="code", how="left")
                    if "sector" not in uni.columns:
                        uni["sector"] = ""
                    mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
                    uni.loc[mask, "sector"] = uni.loc[mask, "sector_fdr"].fillna("").astype(str)
                    if "sector_fdr" in uni.columns:
                        uni.drop(columns=["sector_fdr"], inplace=True)
        except Exception:
            pass

    # Naver Finance crawl fallback: fill remaining blank sectors (limited batch)
    try:
        mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
        missing_codes = uni.loc[mask, "code"].astype(str).str.zfill(6).unique().tolist()[:200]
        if missing_codes:
            found = {}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "ko-KR,ko;q=0.9"
            }
            for code in missing_codes:
                url = f"https://finance.naver.com/item/main.naver?code={code}"
                try:
                    resp = requests.get(url, headers=headers, timeout=8)
                    if resp.status_code != 200 or not resp.text:
                        continue
                    soup = BeautifulSoup(resp.text, "html.parser")
                    a = None
                    for cand in soup.find_all("a", href=True):
                        href = cand.get("href", "")
                        if "sise_group" in href and "type=upjong" in href:
                            a = cand
                            break
                    if a:
                        sec = a.get_text(strip=True)
                        if sec:
                            found[code] = sec
                    time.sleep(0.15)
                except Exception:
                    continue
            if found:
                mdf = pd.DataFrame(list(found.items()), columns=["code", "sector_nav"])
                uni = uni.merge(mdf, on="code", how="left")
                mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
                uni.loc[mask, "sector"] = uni.loc[mask, "sector_nav"].fillna("")
                if "sector_nav" in uni.columns:
                    uni.drop(columns=["sector_nav"], inplace=True)
                # persist to sectors.csv
                try:
                    if len(found):
                        if SECTORS_CSV.exists():
                            s = pd.read_csv(SECTORS_CSV, dtype={"code": str})
                            s["code"] = s["code"].astype(str).str.zfill(6)
                            # update or append
                            s_map = dict(zip(s["code"], s.get("sector", "").fillna("").astype(str)))
                            s_map.update(found)
                            s_out = pd.DataFrame(list(s_map.items()), columns=["code", "sector"])
                            s_out.to_csv(SECTORS_CSV, index=False, encoding="utf-8")
                        else:
                            out = pd.DataFrame(list(found.items()), columns=["code", "sector"])
                            out.to_csv(SECTORS_CSV, index=False, encoding="utf-8")
                except Exception:
                    pass
    except Exception:
        pass

    # 키워드 기반 폴백: 종목명으로 대분류 추정(남은 빈 칸만)
    try:
        def _classify_sector(name: str) -> str:
            n = (name or "").lower()
            # 반도체/전자부품
            if any(k in n for k in ["반도체", "하이닉스", "리노공", "테크윙", "하나마이크론", "tck", "tcky", "동진쎄미", "유진테크", "파두", "isc", "psk", "코미코", "솔브레인", "솔브레인홀딩스", "주성엔지니어링"]):
                return "반도체"
            if any(k in n for k in ["전자", "elec", "lg이노텍", "삼성전기"]):
                return "전자/부품"
            # 2차전지/소재
            if any(k in n for k in ["배터리", "에너지솔루션", "sdi", "엘앤에프", "포스코퓨처엠", "엔켐", "레이크머티리얼즈"]):
                return "2차전지"
            # 인터넷/플랫폼/게임/콘텐츠
            if any(k in n for k in ["naver", "카카오", "카카오페이", "카카오뱅크", "cj enm", "jyp", "와이지엔터", "스튜디오드래곤", "넥슨게임즈", "위메이드", "카카오게임즈"]):
                return "인터넷/플랫폼·콘텐츠"
            # 바이오/제약/헬스케어
            if any(k in n for k in ["바이오", "제약", "셀트리온", "씨젠", "휴젤", "메지온", "에스티팜", "엘앤씨바이오", "큐리언트", "알테오젠", "네이처셀"]):
                return "바이오/제약"
            # 정유/화학/소재
            if any(k in n for k in ["s-oil", "s-oil", "정유", "화학", "lg화학", "포스코인터내셔널", "현대오일", "이노베이션"]):
                return "정유/화학"
            # 자동차/부품·모빌리티
            if any(k in n for k in ["현대차", "기아", "모비스", "오토에버", "글로비스", "한진칼"]):
                return "자동차/모빌리티"
            # 조선/해양/해운
            if any(k in n for k in ["조선", "현대미포", "마린솔루션", "ocean", "hmm"]):
                return "조선/해양·해운"
            # 기계/중공업·방산
            if any(k in n for k in ["두산", "hd현대중공업", "한화오션", "한화에어로", "lignex1", "한국항공우주", "로보틱스"]):
                return "기계/중공업·방산"
            # 금융(은행/증권/보험/지주)
            if any(k in n for k in ["금융", "은행", "증권", "보험", "지주", "kb", "신한", "하나금융", "bnk", "키움증권", "nh투자"]):
                return "금융"
            # 통신/미디어/유통
            if any(k in n for k in ["통신", "sk텔레콤", "kt", "lg유플러스"]):
                return "통신"
            # 유통/소비재
            if any(k in n for k in ["아모레", "gs", "cj", "코웨이", "삼양식품", "맥쿼리인프라"]):
                return "소비재/유통"
            return ""

        mask_kw = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
        if mask_kw.any():
            uni.loc[mask_kw, "sector"] = uni.loc[mask_kw, "name"].apply(_classify_sector).fillna("")
    except Exception:
        pass

    # sectors.csv에 없던 종목은 이전 universe.csv의 sector를 보존
    if "sector" not in uni.columns:
        uni["sector"] = ""
    try:
        if old_sector_map:
            mask = (uni["sector"].isna()) | (uni["sector"].astype(str).str.strip() == "")
            uni.loc[mask, "sector"] = uni.loc[mask, "code"].map(old_sector_map).fillna("")
        uni["sector"] = uni["sector"].fillna("").astype(str)
    except Exception:
        pass

    # 저장 컬럼 순서 고정
    cols = [c for c in ["code", "name", "market", "sector"] if c in uni.columns]
    uni = uni[cols]

    # sector 템플릿 생성(sectors.csv 미존재 시 1회성 가이드 파일 생성)
    try:
        template_path = DATA_DIR / "sectors_template.csv"
        if not SECTORS_CSV.exists():
            tmp = uni[["code"]].copy()
            tmp["sector"] = ""
            tmp.to_csv(template_path, index=False, encoding="utf-8")
    except Exception:
        pass

    # 저장
    uni.to_csv(UNIVERSE_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved universe: {UNIVERSE_CSV.resolve()} (rows={len(uni)})")

    # DB upsert
    # Save to DB (prefer Postgres via SQLAlchemy engine)
    try:
        if get_engine:
            eng = get_engine()
            uni.to_sql("stocks", eng, if_exists="replace", index=False)
            logging.info("Saved universe to Postgres via SQLAlchemy (rows=%d)", len(uni))
            return
    except Exception:
        logging.exception("SQLAlchemy save failed, fallback to sqlite")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                code        TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                market      TEXT,
                sector      TEXT,
                listed_at   DATE,
                delisted_at DATE
            );
            """
        )
        uni.to_sql("stocks", conn, if_exists="replace", index=False)
        conn.commit()
        logging.info("Saved universe to sqlite DB: %s (rows=%d)", DB_PATH.resolve(), len(uni))
    except Exception:
        logging.exception("Failed to save universe to sqlite DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
