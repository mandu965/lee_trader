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


<<<<<<< HEAD
def _to_number(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(numeric) else float(numeric)


def _normalize_summary_row(summary_df: pd.DataFrame) -> dict[str, float | str | None]:
    if summary_df.empty:
        return {}
    row = summary_df.iloc[0]
    keys = [
        "dnca_tot_amt",
        "nxdy_excc_amt",
        "prvs_rcdl_excc_amt",
        "cma_evlu_amt",
        "bfdy_buy_amt",
        "thdt_buy_amt",
        "nxdy_auto_rdpt_amt",
        "bfdy_sll_amt",
        "thdt_sll_amt",
        "d2_auto_rdpt_amt",
        "bfdy_tlex_amt",
        "thdt_tlex_amt",
        "tot_loan_amt",
        "scts_evlu_amt",
        "tot_evlu_amt",
        "nass_amt",
        "pchs_amt_smtl_amt",
        "evlu_amt_smtl_amt",
        "evlu_pfls_smtl_amt",
        "tot_stln_slng_chgs",
        "bfdy_tot_asst_evlu_amt",
        "asst_icdc_amt",
        "asst_icdc_erng_rt",
    ]
    payload: dict[str, float | str | None] = {}
    for key in keys:
        if key not in row.index:
            continue
        number_value = _to_number(row.get(key))
        payload[key] = number_value if number_value is not None else (str(row.get(key) or "").strip() or None)
    return payload


def _build_derived_metrics(holdings: pd.DataFrame, summary_row: dict[str, float | str | None]) -> dict[str, float | None]:
    holding_eval_amount = _to_number(holdings.get("eval_amount").sum()) if not holdings.empty else 0.0
    holding_pnl_amount = _to_number(holdings.get("pnl_amount").sum()) if not holdings.empty else 0.0
    cash_amount = _to_number(summary_row.get("dnca_tot_amt")) or 0.0
    purchase_amount = _to_number(summary_row.get("pchs_amt_smtl_amt")) or 0.0
    total_assets = _to_number(summary_row.get("tot_evlu_amt")) or (cash_amount + holding_eval_amount)
    cash_ratio = (cash_amount / total_assets) if total_assets else None
    invested_ratio = (holding_eval_amount / total_assets) if total_assets else None
    avg_position_weight = _to_number(holdings.get("weight").mean()) if not holdings.empty else None
    return {
        "holding_eval_amount": holding_eval_amount,
        "holding_pnl_amount": holding_pnl_amount,
        "cash_amount": cash_amount,
        "purchase_amount": purchase_amount,
        "total_assets": total_assets,
        "cash_ratio": cash_ratio,
        "invested_ratio": invested_ratio,
        "avg_position_weight": avg_position_weight,
    }


=======
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
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
<<<<<<< HEAD
    # KIS returns evaluation PnL rate in percentage points (for example 3.99),
    # while the UI formatter expects ratio values (0.0399 -> 3.99%).
    work["pnl_pct"] = (work["pnl_pct"] / 100.0).round(6)
=======
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
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
<<<<<<< HEAD
    summary_row = _normalize_summary_row(summary_df)
    derived_metrics = _build_derived_metrics(holdings, summary_row)
=======
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_dv": account.env_dv,
        "cano_masked": f"{account.cano[:2]}******",
        "account_product_code": account.acnt_prdt_cd,
        "cash_summary": cash_summary,
<<<<<<< HEAD
        "summary_row": summary_row,
        "derived_metrics": derived_metrics,
=======
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
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
