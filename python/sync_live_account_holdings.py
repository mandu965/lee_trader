from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine
from kis_client import KISClient
from kis_live_account import inquire_balance, resolve_account_env, summarize_cash
from sync_live_trade_ledger import sync_live_trade_ledger


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


def _weekly_loss_context(
    total_assets: float | None,
    holding_pnl_amount: float | None,
    cash_amount: float | None,
    as_of_date: datetime,
) -> dict[str, float | str | None]:
    if total_assets is None or total_assets <= 0:
        return {
            "timezone": "Asia/Seoul",
            "week_start_date": None,
            "week_start_snapshot_at": None,
            "week_start_cash_amount": None,
            "week_start_holding_pnl_amount": None,
            "week_start_total_assets": None,
            "weekly_asset_change_amount": None,
            "weekly_realized_pnl": None,
            "weekly_unrealized_pnl": None,
            "weekly_total_pnl": None,
            "weekly_loss_pct": None,
            "weekly_reset_mode": "snapshot_based_no_scheduler",
            "weekly_loss_source": "current_total_assets_missing",
        }
    week_start = pd.Timestamp(as_of_date.date())
    week_start = week_start - pd.Timedelta(days=week_start.weekday())
    try:
        engine = get_engine()
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    WITH first_snapshot AS (
                        SELECT snapshot_at
                        FROM research.live_position_snapshot
                        WHERE snapshot_date >= :week_start
                          AND snapshot_date <= :as_of_date
                          AND total_assets IS NOT NULL
                        ORDER BY snapshot_at ASC
                        LIMIT 1
                    )
                    SELECT
                        MAX(snapshot_at) AS snapshot_at,
                        MAX(snapshot_date) AS snapshot_date,
                        MAX(total_assets) AS total_assets,
                        MAX(cash_amount) AS cash_amount,
                        COALESCE(SUM(pnl_amount), 0) AS holding_pnl_amount
                    FROM research.live_position_snapshot
                    WHERE snapshot_at = (SELECT snapshot_at FROM first_snapshot)
                    """
                ),
                {
                    "week_start": week_start.date().isoformat(),
                    "as_of_date": as_of_date.date().isoformat(),
                },
            ).mappings().first()
            fill_summary = conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*)::integer AS fill_count,
                        COUNT(*) FILTER (WHERE UPPER(side) = 'BUY')::integer AS buy_fill_count,
                        COUNT(*) FILTER (WHERE UPPER(side) = 'SELL')::integer AS sell_fill_count
                    FROM research.live_order_fill
                    WHERE as_of_date >= :week_start
                      AND as_of_date <= :as_of_date
                    """
                ),
                {
                    "week_start": week_start.date().isoformat(),
                    "as_of_date": as_of_date.date().isoformat(),
                },
            ).mappings().first()
    except Exception:
        row = None
        fill_summary = None
    if not row:
        return {
            "timezone": "Asia/Seoul",
            "week_start_date": week_start.date().isoformat(),
            "week_start_snapshot_at": None,
            "week_start_cash_amount": None,
            "week_start_holding_pnl_amount": None,
            "week_start_total_assets": None,
            "weekly_asset_change_amount": None,
            "weekly_realized_pnl": None,
            "weekly_unrealized_pnl": None,
            "weekly_total_pnl": None,
            "weekly_loss_pct": None,
            "weekly_fill_count": None,
            "weekly_buy_fill_count": None,
            "weekly_sell_fill_count": None,
            "weekly_external_cash_flow_amount": None,
            "weekly_loss_guard_basis": "snapshot_total_assets",
            "weekly_reset_mode": "snapshot_based_no_scheduler",
            "weekly_loss_source": "live_position_snapshot_unavailable",
        }

    week_start_total_assets = _to_number(row.get("total_assets"))
    if week_start_total_assets is None or week_start_total_assets <= 0:
        return {
            "timezone": "Asia/Seoul",
            "week_start_date": str(row.get("snapshot_date") or week_start.date().isoformat()),
            "week_start_snapshot_at": str(row.get("snapshot_at") or "") or None,
            "week_start_cash_amount": _to_number(row.get("cash_amount")),
            "week_start_holding_pnl_amount": _to_number(row.get("holding_pnl_amount")),
            "week_start_total_assets": week_start_total_assets,
            "weekly_asset_change_amount": None,
            "weekly_realized_pnl": None,
            "weekly_unrealized_pnl": None,
            "weekly_total_pnl": None,
            "weekly_loss_pct": None,
            "weekly_fill_count": _to_number((fill_summary or {}).get("fill_count")),
            "weekly_buy_fill_count": _to_number((fill_summary or {}).get("buy_fill_count")),
            "weekly_sell_fill_count": _to_number((fill_summary or {}).get("sell_fill_count")),
            "weekly_external_cash_flow_amount": None,
            "weekly_loss_guard_basis": "snapshot_total_assets",
            "weekly_reset_mode": "snapshot_based_no_scheduler",
            "weekly_loss_source": "week_start_total_assets_invalid",
        }

    week_start_cash_amount = _to_number(row.get("cash_amount"))
    week_start_holding_pnl_amount = _to_number(row.get("holding_pnl_amount")) or 0.0
    weekly_fill_count = int(_to_number((fill_summary or {}).get("fill_count")) or 0.0)
    weekly_buy_fill_count = int(_to_number((fill_summary or {}).get("buy_fill_count")) or 0.0)
    weekly_sell_fill_count = int(_to_number((fill_summary or {}).get("sell_fill_count")) or 0.0)
    weekly_asset_change_amount = total_assets - week_start_total_assets
    weekly_unrealized_pnl = (holding_pnl_amount or 0.0) - week_start_holding_pnl_amount
    implied_realized_pnl = weekly_asset_change_amount - weekly_unrealized_pnl
    weekly_realized_pnl = implied_realized_pnl
    weekly_total_pnl = weekly_asset_change_amount
    weekly_loss_pct = weekly_total_pnl / week_start_total_assets
    weekly_external_cash_flow_amount = None
    weekly_loss_guard_basis = "snapshot_total_assets"
    weekly_loss_source = "live_position_snapshot"

    # If the account had no weekly sell fills, a large "realized" residual usually
    # means deposit/withdrawal or broker-side cash movement rather than trading loss.
    no_realizing_trade_this_week = weekly_sell_fill_count == 0
    suspicious_residual_threshold = max(100000.0, week_start_total_assets * 0.03)
    if no_realizing_trade_this_week and abs(implied_realized_pnl) >= suspicious_residual_threshold:
        weekly_realized_pnl = 0.0
        weekly_total_pnl = weekly_unrealized_pnl
        weekly_loss_pct = weekly_total_pnl / week_start_total_assets
        weekly_external_cash_flow_amount = implied_realized_pnl
        weekly_loss_guard_basis = "unrealized_only_no_weekly_sells"
        weekly_loss_source = "live_position_snapshot_excluding_external_cash_flow"
    return {
        "timezone": "Asia/Seoul",
        "week_start_date": str(row.get("snapshot_date") or week_start.date().isoformat()),
        "week_start_snapshot_at": str(row.get("snapshot_at") or "") or None,
        "week_start_cash_amount": week_start_cash_amount,
        "week_start_holding_pnl_amount": week_start_holding_pnl_amount,
        "week_start_total_assets": week_start_total_assets,
        "weekly_asset_change_amount": weekly_asset_change_amount,
        "weekly_realized_pnl": weekly_realized_pnl,
        "weekly_unrealized_pnl": weekly_unrealized_pnl,
        "weekly_total_pnl": weekly_total_pnl,
        "weekly_loss_pct": weekly_loss_pct,
        "weekly_fill_count": weekly_fill_count,
        "weekly_buy_fill_count": weekly_buy_fill_count,
        "weekly_sell_fill_count": weekly_sell_fill_count,
        "weekly_external_cash_flow_amount": weekly_external_cash_flow_amount,
        "weekly_loss_guard_basis": weekly_loss_guard_basis,
        "weekly_reset_mode": "snapshot_based_no_scheduler",
        "weekly_loss_source": weekly_loss_source,
    }


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
    # KIS balance responses can include zero-quantity placeholders after liquidation.
    # Exclude them so downstream previews and the live dashboard reflect actual holdings only.
    work = work[work["qty"].fillna(0) > 0].copy()
    if work.empty:
        return pd.DataFrame(
            columns=["code", "name", "qty", "avg_price", "current_price", "eval_amount", "pnl_amount", "pnl_pct", "weight", "status"]
        )
    # KIS returns evaluation PnL rate in percentage points (for example 3.99),
    # while the UI formatter expects ratio values (0.0399 -> 3.99%).
    work["pnl_pct"] = (work["pnl_pct"] / 100.0).round(6)
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
    summary_row = _normalize_summary_row(summary_df)
    derived_metrics = _build_derived_metrics(holdings, summary_row)
    weekly_context = _weekly_loss_context(
        derived_metrics.get("total_assets"),
        derived_metrics.get("holding_pnl_amount"),
        derived_metrics.get("cash_amount"),
        datetime.now(),
    )
    derived_metrics.update(weekly_context)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_dv": account.env_dv,
        "cano_masked": f"{account.cano[:2]}******",
        "account_product_code": account.acnt_prdt_cd,
        "cash_summary": cash_summary,
        "summary_row": summary_row,
        "derived_metrics": derived_metrics,
        "holding_count": int(len(holdings)),
    }

    out_holdings = _resolve(args.out_holdings_csv)
    out_summary = _resolve(args.out_summary_json)
    out_holdings.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    holdings.to_csv(out_holdings, index=False, encoding="utf-8-sig")
    out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        sync_result = sync_live_trade_ledger(
            holdings_csv=out_holdings,
            balance_summary_payload=payload,
        )
        print(f"live_trade_ledger_sync: {sync_result}")
    except Exception as exc:
        print(f"live_trade_ledger_sync_failed: {exc}")

    print(f"holdings_csv: {out_holdings}")
    print(f"summary_json: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
