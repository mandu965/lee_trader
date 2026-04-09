from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from kis_client import KISClient
from kis_live_account import inquire_balance, resolve_account_env, summarize_cash


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUT_HOLDINGS_CSV = DATA_DIR / "live_account_holdings.csv"
OUT_SUMMARY_JSON = OUTPUT_DIR / "live_account_balance_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync KIS live/demo account holdings into CSV/JSON.")
    parser.add_argument("--out-holdings-csv", type=Path, default=OUT_HOLDINGS_CSV)
    parser.add_argument("--out-summary-json", type=Path, default=OUT_SUMMARY_JSON)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["code", "name", "qty", "avg_price", "current_price", "eval_amount", "pnl_amount", "pnl_pct"]
        )
    work = df.copy()
    rename_map = {
        "pdno": "code",
        "prdt_name": "name",
        "hldg_qty": "qty",
        "pchs_avg_pric": "avg_price",
        "prpr": "current_price",
        "evlu_amt": "eval_amount",
        "evlu_pfls_amt": "pnl_amount",
        "evlu_pfls_rt": "pnl_pct",
    }
    work = work.rename(columns=rename_map)
    for col in ["code", "name"]:
        work[col] = work.get(col, "").fillna("").astype(str)
    work["code"] = work["code"].str.zfill(6)
    for col in ["qty", "avg_price", "current_price", "eval_amount", "pnl_amount", "pnl_pct"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["weight"] = work["eval_amount"] / work["eval_amount"].sum() if work["eval_amount"].notna().any() and work["eval_amount"].sum() else pd.NA
    work["status"] = "OPEN"
    return work[["code", "name", "qty", "avg_price", "current_price", "eval_amount", "pnl_amount", "pnl_pct", "weight", "status"]].sort_values(["eval_amount", "code"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    client = KISClient.from_env()
    client.issue_access_token()
    account = resolve_account_env()
    holdings_df, summary_df = inquire_balance(client, account)

    holdings = normalize_holdings(holdings_df)
    cash_summary = summarize_cash(summary_df)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_dv": account.env_dv,
        "cano_masked": f"{account.cano[:2]}******",
        "account_product_code": account.acnt_prdt_cd,
        "cash_summary": cash_summary,
        "holding_count": int(len(holdings)),
    }

    out_holdings = _resolve(args.out_holdings_csv)
    out_summary = _resolve(args.out_summary_json)
    out_holdings.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    holdings.to_csv(out_holdings, index=False, encoding="utf-8-sig")
    out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"holdings_csv: {out_holdings}")
    print(f"summary_json: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
