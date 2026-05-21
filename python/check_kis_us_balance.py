"""One-shot script: KIS 해외계좌 잔고 확인"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import requests

BASE       = os.environ["KIS_BASE_URL"]
APP_KEY    = os.environ["KIS_US_APP_KEY"]
APP_SECRET = os.environ["KIS_US_APP_SECRET"]
CANO       = os.environ["KIS_US_CANO"]
ACNT       = os.environ["KIS_US_ACNT_PRDT_CD"]

tok = requests.post(
    f"{BASE}/oauth2/tokenP",
    json={"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET},
    timeout=20,
).json()
token = tok.get("access_token", "")
if not token:
    print(f"[ERROR] Token failed: {tok}")
    sys.exit(1)
print("[OK] Token acquired")

headers = {
    "Authorization": f"Bearer {token}",
    "appkey": APP_KEY,
    "appsecret": APP_SECRET,
    "tr_id": "TTTS3012R",
    "Content-Type": "application/json",
}
params = {
    "CANO": CANO,
    "ACNT_PRDT_CD": ACNT,
    "OVRS_EXCG_CD": "",
    "TR_CRCY_CD": "USD",
    "CTX_AREA_FK200": "",
    "CTX_AREA_NK200": "",
}
r = requests.get(
    f"{BASE}/uapi/overseas-stock/v1/trading/inquire-balance",
    headers=headers, params=params, timeout=20,
).json()

rt  = r.get("rt_cd", "?")
msg = r.get("msg1", "")
print(f"rt_cd={rt}  msg={msg}")

out2 = r.get("output2", {})
if isinstance(out2, list):
    out2 = out2[0] if out2 else {}

print("\n=== output2 전체 필드 (모두) ===")
for k, v in out2.items():
    print(f"  {k}: {v!r}")

print("\n=== output2 주요 잔고 필드 ===")
fields = [
    ("tot_evlu_pfls_amt", "총평가손익금액"),
    ("frcr_dncl_amt_2",   "외화예수금(2)"),
    ("tot_dncl_amt",      "총예수금"),
    ("evlu_amt_smtl_2",   "평가금액합계(2)"),
    ("pchs_amt_smtl_amt", "매입금액합계"),
    ("ovrs_tot_pfls",     "해외총손익"),
    ("tot_pftrt",         "총수익률(%)"),
]
for key, label in fields:
    val = out2.get(key, "-")
    print(f"  {label:20s}: {val}")
