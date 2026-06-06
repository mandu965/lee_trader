"""
midcap_backfill_flow_short.py  —  Phase 1: 중형주 수급(flow) + 공매도(short) 백필

운영 검증 패턴(by_ticker, 날짜별 전종목)을 격리 재사용한다.
  - flow : get_market_net_purchases_of_equities_by_ticker (외국인/기관합계 순매수)
           ※ fetch_flow_pykrx.py 와 동일. KRX_ID/KRX_PW 필요.
  - short: get_shorting_balance_by_ticker (공매도 잔고 비율)
           ※ fetch_short_interest.py 와 동일.

출력(격리): data/research_midcap/flow_midcap_raw.csv, short_midcap_raw.csv
원칙: 운영 테이블(flow_daily/short_interest_daily) 미수정. 라이브 무영향.

사용법:
  python python/research/midcap_backfill_flow_short.py --start 20230101 --end 20260530
  python python/research/midcap_backfill_flow_short.py --start 20260525 --end 20260529   # 검증
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests as _rq

# pykrx/KRX 호출에 타임아웃이 없어 무한 대기(hang) 발생.
# requests는 timeout 미지정 시 소켓을 blocking으로 강제 → socket.setdefaulttimeout 무력화.
# 따라서 Session.request에 기본 (connect,read) 타임아웃을 강제 주입한다.
socket.setdefaulttimeout(30)
_orig_request = _rq.sessions.Session.request
def _request_with_timeout(self, *args, **kwargs):
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = (10, 15)  # (connect 10s, read 15s)
    return _orig_request(self, *args, **kwargs)
_rq.sessions.Session.request = _request_with_timeout

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "data" / "research_midcap"
UNI = RESEARCH_DIR / "universe_midcap.csv"
FLOW_OUT = RESEARCH_DIR / "flow_midcap_raw.csv"
SHORT_OUT = RESEARCH_DIR / "short_midcap_raw.csv"
FLUSH_EVERY = 10


def setup_logging() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if val.startswith(('"', "'")):
                q = val[0]
                end = val.find(q, 1)
                val = val[1:end] if end != -1 else val[1:]
            else:
                val = re.sub(r"\s+#.*$", "", val).strip()
            if key and key not in os.environ:
                os.environ[key] = val


def business_dates(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out, cur = [], s
    while cur <= e:
        if cur.weekday() < 5:
            out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def load_targets() -> set[str]:
    df = pd.read_csv(UNI, dtype={"code": str})
    return set(df["code"].str.zfill(6))


def fetch_flow_day(date_str: str, targets: set[str]) -> list[dict]:
    from pykrx import stock
    AMOUNT_COL = "순매수거래대금"
    acc: dict[str, dict] = {}
    for market in ("KOSPI", "KOSDAQ"):
        for investor, key in (("외국인", "foreign_net"), ("기관합계", "inst_net")):
            try:
                df = stock.get_market_net_purchases_of_equities_by_ticker(
                    date_str, date_str, market=market, investor=investor)
            except Exception as e:
                logging.warning("flow %s %s %s 실패: %s", date_str, market, investor, e)
                continue
            if df is None or df.empty:
                continue
            for ticker in df.index:
                code = str(ticker).zfill(6)
                if code not in targets:
                    continue
                acc.setdefault(code, {"foreign_net": None, "inst_net": None})
                try:
                    acc[code][key] = float(df.at[ticker, AMOUNT_COL])
                except Exception:
                    pass
    bdate = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return [{"date": bdate, "code": c, **v} for c, v in acc.items()]


def fetch_short_day(date_str: str, targets: set[str]) -> list[dict]:
    from pykrx import stock
    rows = []
    bdate = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    for market in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_shorting_balance_by_ticker(date_str, market=market)
        except Exception as e:
            logging.warning("short %s %s 실패: %s", date_str, market, e)
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index()
        # 한글 컬럼(정상) + 영문 코드(ISU_CD/BAL_*) 폴백 — KRX 응답이 간헐적으로 영문코드로 옴
        tcol = next((c for c in ["티커", "종목코드", "Ticker", "ticker", "code", "ISU_CD"] if c in df.columns), None)
        if tcol is None:
            continue
        df["code"] = df[tcol].astype(str).str.zfill(6)
        vcol = next((c for c in ["공매도잔고", "공매도 잔고", "BAL_QTY"] if c in df.columns), None)
        lcol = next((c for c in ["상장주식수", "LIST_SHRS"] if c in df.columns), None)
        rcol = next((c for c in ["공매도잔고비율", "공매도 잔고비율", "비중", "BAL_RTO"] if c in df.columns), None)
        for _, r in df[df["code"].isin(targets)].iterrows():
            sv = pd.to_numeric(r.get(vcol), errors="coerce") if vcol else None
            ls = pd.to_numeric(r.get(lcol), errors="coerce") if lcol else None
            sr = pd.to_numeric(r.get(rcol), errors="coerce") if rcol else None
            if (sr is None or pd.isna(sr)) and sv is not None and ls and ls > 0:
                sr = round(sv / ls * 100, 4)
            rows.append({"date": bdate, "code": r["code"], "short_ratio": sr,
                         "short_volume": sv, "listed_shares": ls})
    return rows


def flush(path: Path, rows: list[dict], header_written: dict) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    mode = "a" if header_written.get(str(path)) else "w"
    df.to_csv(path, index=False, mode=mode, header=not header_written.get(str(path)),
              encoding="utf-8-sig")
    header_written[str(path)] = True


def main() -> int:
    setup_logging()
    load_env()
    if not os.environ.get("KRX_ID") or not os.environ.get("KRX_PW"):
        logging.error("KRX_ID/KRX_PW 미설정 — .env 확인")
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--delay", type=float, default=1.2)
    args = ap.parse_args()

    targets = load_targets()
    dates = business_dates(args.start, args.end)
    logging.info("flow+short 백필: %d종목 · %s~%s (%d영업일)", len(targets), args.start, args.end, len(dates))

    # 재개(resume): 이미 수집된 날짜는 건너뛴다. (hang 후 재시작 대비)
    header_written: dict = {}
    done_dates: set[str] = set()
    if FLOW_OUT.exists():
        try:
            prev = pd.read_csv(FLOW_OUT, usecols=["date"])
            done_dates = set(prev["date"].astype(str).unique())
            header_written[str(FLOW_OUT)] = True
            header_written[str(SHORT_OUT)] = True
            logging.info("resume: 기존 %d일 발견 → 건너뜀", len(done_dates))
        except Exception as e:
            logging.warning("기존 CSV 읽기 실패(처음부터 진행): %s", e)
    flow_buf: list[dict] = []
    short_buf: list[dict] = []
    flow_total = short_total = 0

    for i, d in enumerate(dates, 1):
        bdate = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        if bdate in done_dates:
            continue
        fr = fetch_flow_day(d, targets)
        sr = fetch_short_day(d, targets)
        flow_buf.extend(fr)
        short_buf.extend(sr)
        flow_total += len(fr)
        short_total += len(sr)
        if i % FLUSH_EVERY == 0:
            flush(FLOW_OUT, flow_buf, header_written); flow_buf = []
            flush(SHORT_OUT, short_buf, header_written); short_buf = []
            logging.info("[%d/%d] %s — flow累=%d short累=%d", i, len(dates), d, flow_total, short_total)
        time.sleep(args.delay)

    flush(FLOW_OUT, flow_buf, header_written)
    flush(SHORT_OUT, short_buf, header_written)

    print("\n" + "=" * 56)
    print(f"  flow  : {flow_total:,}행 → {FLOW_OUT.name}")
    print(f"  short : {short_total:,}행 → {SHORT_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
