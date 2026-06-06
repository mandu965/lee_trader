"""
midcap_phase0_liquidity.py  —  중형주(100~200위) 확장 연구 / Phase 0: 유동성 정찰

목적:
  코스피/코스닥 시총 1~100위(현 운영 유니버스)와 101~200위(연구 후보) 의
  거래대금(ADV proxy) 분포를 비교하여 "우리 포지션 크기에서 체결이 현실적인가"
  를 백필 없이 먼저 판정한다.

원칙 (CLAUDE.md 1번):
  - 운영 데이터/모듈을 일절 건드리지 않는다. (Naver 직접 파싱 + DB read-only SELECT)
  - 산출물은 data/research_midcap/ 격리 디렉터리에만 쓴다.
  - KIS/실주문 경로 미사용.

거래대금은 시총 순위 페이지의 (현재가 × 거래량) 1일 스냅샷이다. 정밀 ADV가 아니라
분포 비교용 1차 근사다(주석으로 한계 명시).
"""
from __future__ import annotations

import os
import re
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("data/research_midcap")
NAVER_SOSOK = {"KOSPI": 0, "KOSDAQ": 1}
HEADERS = {"User-Agent": "Mozilla/5.0 (research; midcap-phase0)"}
REQUEST_DELAY = 0.4
MAX_PAGES = 6  # 50/page → 300종목까지. 200위 확보 + 필터 여유.

# Phase 0 판정용 포지션 크기 시나리오 (1종목당 주문금액, KRW)
POSITION_SIZES = [1_000_000, 5_000_000, 10_000_000]


def setup_logging() -> None:
    # Windows 콘솔(cp949)에서 한글/em-dash 출력 시 UnicodeEncodeError 방지
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _to_num(text: str) -> float:
    """'55,989' / '1,234' → float. 실패 시 NaN."""
    if text is None:
        return np.nan
    s = re.sub(r"[^0-9.\-]", "", str(text))
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_page(soup: BeautifulSoup, market: str) -> list[dict]:
    table = soup.select_one("table.type_2")
    if table is None:
        return []
    out: list[dict] = []
    for row in table.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 11:
            continue
        name_tag = row.select_one("td a")
        if not name_tag:
            continue
        href = name_tag.get("href", "")
        code = href.split("code=")[-1] if "code=" in href else ""
        texts = [c.text.strip() for c in cols]
        out.append({
            "market": market,
            "code": str(code).zfill(6),
            "name": name_tag.text.strip(),
            "price": _to_num(texts[2]),        # 현재가
            "mcap_eok": _to_num(texts[6]),     # 시가총액(억원)
            "volume": _to_num(texts[9]),       # 거래량(주)
        })
    return out


def fetch_market(market: str) -> pd.DataFrame:
    sosok = NAVER_SOSOK[market]
    records: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = (
            "https://finance.naver.com/sise/sise_market_sum.naver"
            f"?sosok={sosok}&page={page}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logging.warning("[%s] page %d 요청 실패: %s", market, page, e)
            break
        recs = _parse_page(BeautifulSoup(resp.text, "html.parser"), market)
        if not recs:
            break
        records.extend(recs)
        logging.info("[%s] page %d: +%d (누적 %d)", market, page, len(recs), len(records))
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # 보통주만: 코드 끝자리 '0' 이 아닌 우선주(5/7/K 등) 제거. ETF/ETN 이름 키워드 제거.
    etf_kw = ("ETF", "ETN", "KODEX", "TIGER", "PLUS", "ACE", "SOL", "RISE", "KBSTAR", "ARIRANG", "HANARO")
    name_up = df["name"].str.upper()
    is_etf = name_up.apply(lambda n: any(k in n for k in etf_kw))
    is_pref = ~df["code"].str.endswith("0")
    df = df.loc[~is_etf & ~is_pref].copy()
    df = df.dropna(subset=["mcap_eok"]).drop_duplicates(subset=["code"])
    df = df.sort_values("mcap_eok", ascending=False).reset_index(drop=True)
    df["mcap_rank"] = np.arange(1, len(df) + 1)
    # 거래대금(원) = 현재가 × 거래량  (1일 스냅샷 proxy)
    df["turnover_krw"] = df["price"] * df["volume"]
    return df


def load_existing_feature_codes() -> set[str]:
    """research: features 테이블에 이미 이력이 있는 코드(=공짜 백필분). read-only."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        logging.warning("DATABASE_URL 미설정 → overlap 계산 생략")
        return set()
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(url)
        with eng.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT code FROM public.features")).fetchall()
        return {str(r[0]).zfill(6) for r in rows}
    except Exception as e:
        logging.warning("기존 feature 코드 조회 실패: %s", e)
        return set()


def bucket_stats(df: pd.DataFrame, label: str) -> dict:
    t = df["turnover_krw"].dropna()
    t = t[t > 0]
    return {
        "bucket": label,
        "n": int(len(t)),
        "turnover_median_eok": round(t.median() / 1e8, 1),
        "turnover_p25_eok": round(t.quantile(0.25) / 1e8, 1),
        "turnover_p10_eok": round(t.quantile(0.10) / 1e8, 1),
        "turnover_min_eok": round(t.min() / 1e8, 1),
    }


def participation_table(midcap: pd.DataFrame) -> pd.DataFrame:
    """포지션 크기별 참여율(주문금액/거래대금) — 체결 현실성 핵심 지표."""
    t = midcap["turnover_krw"].dropna()
    t = t[t > 0]
    med, p10 = t.median(), t.quantile(0.10)
    rows = []
    for size in POSITION_SIZES:
        rows.append({
            "position_krw": f"{size//1_000_000}M",
            "vs_median_ADV_%": round(size / med * 100, 2),
            "vs_p10_ADV_%": round(size / p10 * 100, 2),
        })
    return pd.DataFrame(rows)


def main() -> int:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        df = fetch_market(market)
        if df.empty:
            logging.error("[%s] 수집 실패", market)
            continue
        frames.append(df)
    if not frames:
        logging.error("전 시장 수집 실패 — 종료")
        return 1

    alldf = pd.concat(frames, ignore_index=True)
    alldf["tier"] = np.where(alldf["mcap_rank"] <= 100, "top100",
                      np.where(alldf["mcap_rank"] <= 200, "midcap_101_200", "below_200"))

    top = alldf[alldf["tier"] == "top100"]
    mid = alldf[alldf["tier"] == "midcap_101_200"]

    # 분포 비교
    stats = pd.DataFrame([
        bucket_stats(top[top.market == "KOSPI"], "KOSPI top100"),
        bucket_stats(mid[mid.market == "KOSPI"], "KOSPI 101-200"),
        bucket_stats(top[top.market == "KOSDAQ"], "KOSDAQ top100"),
        bucket_stats(mid[mid.market == "KOSDAQ"], "KOSDAQ 101-200"),
        bucket_stats(mid, "ALL midcap 101-200"),
    ])

    # 참여율
    part = participation_table(mid)

    # 기존 feature 이력과의 overlap (공짜 백필분)
    existing = load_existing_feature_codes()
    mid_codes = set(mid["code"])
    overlap = mid_codes & existing
    overlap_info = {
        "midcap_count": len(mid_codes),
        "already_in_features": len(overlap),
        "need_backfill": len(mid_codes - existing),
    }

    # 저장
    alldf.to_csv(OUT_DIR / "phase0_universe_snapshot.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(OUT_DIR / "phase0_turnover_stats.csv", index=False, encoding="utf-8-sig")
    part.to_csv(OUT_DIR / "phase0_participation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([overlap_info]).to_csv(OUT_DIR / "phase0_overlap.csv", index=False, encoding="utf-8-sig")

    # 콘솔 요약
    print("\n" + "=" * 64)
    print(" Phase 0 — 유동성 정찰 결과 (거래대금=현재가×거래량, 1일 스냅샷)")
    print("=" * 64)
    print("\n[거래대금 분포 — 억원]")
    print(stats.to_string(index=False))
    print("\n[포지션 크기별 참여율 = 주문금액 / 거래대금]  (midcap 101-200)")
    print(part.to_string(index=False))
    print("\n[백필 필요량]")
    print(f"  midcap 101-200 종목수      : {overlap_info['midcap_count']}")
    print(f"  이미 features에 이력 존재   : {overlap_info['already_in_features']}")
    print(f"  신규 백필 필요             : {overlap_info['need_backfill']}")
    print(f"\n산출물: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
