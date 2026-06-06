"""
midcap_pit_universe.py  —  Phase 3 사전: point-in-time 유니버스 멤버십 복원

생존편향 제거의 토대. 과거 각 리밸런스 시점의 시총 101~200위를 복원한다.
get_market_cap_by_ticker(date, market) (by_ticker per date) 사용 — 저렴(수십 콜).

출력: data/research_midcap/pit_universe_membership.csv (date, code, market, mcap_rank)
요약: 과거 한 번이라도 101~200위였던 종목 수 / 현재 200과의 차이(=신규 백필 필요 dropout 수).
원칙: 읽기 전용 KRX 공개데이터, 운영 무접촉.
"""
from __future__ import annotations

import os
import re
import socket
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests as _rq

socket.setdefaulttimeout(30)
_orig = _rq.sessions.Session.request
def _to(self, *a, **k):
    if k.get("timeout") is None:
        k["timeout"] = (10, 15)
    return _orig(self, *a, **k)
_rq.sessions.Session.request = _to

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "data" / "research_midcap"
OUT = RESEARCH_DIR / "pit_universe_membership.csv"
CUR_UNI = RESEARCH_DIR / "universe_midcap.csv"

RANK_LO, RANK_HI = 101, 200  # 중형주 밴드


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_env() -> None:
    """KRX_ID/KRX_PW 등 .env 로딩 (get_market_cap_by_ticker가 KRX 로그인 필요)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if v.startswith(('"', "'")):
            q = v[0]; end = v.find(q, 1); v = v[1:end] if end != -1 else v[1:]
        else:
            v = re.sub(r"\s+#.*$", "", v).strip()
        if k and k not in os.environ:
            os.environ[k] = v


def quarter_dates(start="20220630", end="20260331") -> list[str]:
    """분기말 근처 영업일(리밸런스 그리드). 약 16개."""
    s = datetime.strptime(start, "%Y%m%d"); e = datetime.strptime(end, "%Y%m%d")
    out, cur = [], s
    while cur <= e:
        # 분기말 달(3,6,9,12)의 말일 근처 평일로 보정
        d = cur
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        out.append(d.strftime("%Y%m%d"))
        # 다음 분기로
        m = cur.month + 3
        y = cur.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        import calendar
        last = calendar.monthrange(y, m)[1]
        cur = datetime(y, m, last)
    return out


def fetch_rank_band(date_str: str) -> pd.DataFrame:
    from pykrx import stock
    rows = []
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_market_cap_by_ticker(date_str, market=market)
        except Exception as e:
            logging.warning("mcap %s %s 실패: %s", date_str, market, e)
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index()
        tcol = next((c for c in ["티커", "종목코드", "ticker", "code"] if c in df.columns), df.columns[0])
        mcol = next((c for c in ["시가총액", "marcap"] if c in df.columns), None)
        if mcol is None:
            continue
        df = df.rename(columns={tcol: "code", mcol: "mcap"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
        df = df.dropna(subset=["mcap"]).sort_values("mcap", ascending=False).reset_index(drop=True)
        df["mcap_rank"] = df.index + 1
        band = df[(df["mcap_rank"] >= RANK_LO) & (df["mcap_rank"] <= RANK_HI)].copy()
        band["market"] = market
        band["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        rows.append(band[["date", "code", "market", "mcap_rank"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    setup_logging()
    load_env()
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    dates = quarter_dates()
    logging.info("PIT 멤버십 복원: %d 시점", len(dates))
    parts = []
    for d in dates:
        b = fetch_rank_band(d)
        if not b.empty:
            parts.append(b)
            logging.info("  %s: %d종목", d, len(b))
    if not parts:
        logging.error("수집 실패")
        return 1
    pit = pd.concat(parts, ignore_index=True)
    pit.to_csv(OUT, index=False, encoding="utf-8-sig")

    pit_codes = set(pit["code"])
    cur = set(pd.read_csv(CUR_UNI, dtype={"code": str})["code"].str.zfill(6))
    dropouts = pit_codes - cur
    print("\n" + "=" * 56)
    print(f"  PIT 시점 수            : {pit['date'].nunique()}")
    print(f"  PIT 101~200위 종목(합집합): {len(pit_codes)}")
    print(f"  현재(오늘) 200과 교집합  : {len(pit_codes & cur)}")
    print(f"  과거 전용 dropout(신규백필): {len(dropouts)}")
    print(f"  → 생존편향 크기 가늠: 과거 밴드의 {len(dropouts)/max(len(pit_codes),1)*100:.0f}%가 현재 밴드 밖")
    print(f"  저장: {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
