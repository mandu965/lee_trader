"""Sync KIS RULE account holdings (KIS_RULE_CANO) to data/rule_account_holdings.csv.

Runs as part of the daily close batch so the manual-trading monitoring page
always shows the latest positions from the RULE brokerage account.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from kis_client import KISClient
from kis_live_account import KISAccountEnv, inquire_balance, infer_env_dv
from sync_live_account_holdings import normalize_holdings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_HOLDINGS_CSV = DATA_DIR / "rule_account_holdings.csv"
OUT_SUMMARY_JSON = ROOT / "outputs" / "rule_account_holdings_summary.json"


def resolve_rule_account_env() -> KISAccountEnv:
    """Read RULE-account credentials from env vars KIS_RULE_CANO / KIS_RULE_ACNT_PRDT_CD."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    base_url = os.getenv("KIS_BASE_URL")
    env_dv = os.getenv("KIS_ENV_DV") or infer_env_dv(base_url)

    cano = (os.getenv("KIS_RULE_CANO") or "").strip()
    acnt_prdt_cd = (os.getenv("KIS_RULE_ACNT_PRDT_CD") or "").strip()

    if not cano or not acnt_prdt_cd:
        raise ValueError(
            "KIS_RULE_CANO and KIS_RULE_ACNT_PRDT_CD must be set in .env for RULE account sync."
        )

    return KISAccountEnv(
        env_dv=env_dv,
        cano=cano,
        acnt_prdt_cd=acnt_prdt_cd,
    )


def main() -> int:
    account = resolve_rule_account_env()
    logging.info("Syncing RULE account holdings: cano=%s..., acnt=%s", account.cano[:4], account.acnt_prdt_cd)

    client = KISClient.from_rule_env()
    client.issue_access_token()

    holdings_df, summary_df = inquire_balance(client, account)
    holdings = normalize_holdings(holdings_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    holdings.to_csv(OUT_HOLDINGS_CSV, index=False, encoding="utf-8-sig")
    logging.info("rule_account_holdings.csv saved: %d rows -> %s", len(holdings), OUT_HOLDINGS_CSV)

    # 요약 JSON (웹 페이지 상태 표시용)
    cash_row = {}
    if not summary_df.empty:
        row = summary_df.iloc[0]
        for key in ("dnca_tot_amt", "tot_evlu_amt", "evlu_amt_smtl_amt", "evlu_pfls_smtl_amt", "pchs_amt_smtl_amt"):
            val = pd.to_numeric(row.get(key, None), errors="coerce")
            cash_row[key] = float(val) if pd.notna(val) else None

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cano_masked": f"{account.cano[:2]}****{account.cano[-2:]}",
        "holding_count": len(holdings),
        "cash_summary": cash_row,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("rule_account_holdings_summary.json saved -> %s", OUT_SUMMARY_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
