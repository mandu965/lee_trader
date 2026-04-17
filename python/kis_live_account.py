from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from kis_client import KISClient

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


BALANCE_API_URL = "/uapi/domestic-stock/v1/trading/inquire-balance"
PSBL_ORDER_API_URL = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
ORDER_CASH_API_URL = "/uapi/domestic-stock/v1/trading/order-cash"


@dataclass(frozen=True)
class KISAccountEnv:
    env_dv: str
    cano: str
    acnt_prdt_cd: str
    hts_id: str | None = None


def infer_env_dv(base_url: str | None) -> str:
    text = str(base_url or "").lower()
    return "demo" if "openapivts" in text else "real"


def resolve_account_env() -> KISAccountEnv:
    if load_dotenv:
        load_dotenv()
    base_url = os.getenv("KIS_BASE_URL")
    env_dv = os.getenv("KIS_ENV_DV") or infer_env_dv(base_url)

    cano = (os.getenv("KIS_CANO") or "").strip()
    acnt_prdt_cd = (os.getenv("KIS_ACNT_PRDT_CD") or "").strip()
    if not cano or not acnt_prdt_cd:
        raw_account = (os.getenv("KIS_ACCOUNT_NO") or "").strip().replace("-", "")
        if len(raw_account) >= 10 and raw_account.isdigit():
            cano = cano or raw_account[:8]
            acnt_prdt_cd = acnt_prdt_cd or raw_account[8:10]

    if not cano or not acnt_prdt_cd:
        raise ValueError("KIS account env missing. Set KIS_CANO and KIS_ACNT_PRDT_CD, or provide KIS_ACCOUNT_NO in 8-2 format.")

    return KISAccountEnv(
        env_dv=env_dv,
        cano=cano,
        acnt_prdt_cd=acnt_prdt_cd,
        hts_id=(os.getenv("KIS_HTS_ID") or "").strip() or None,
    )


def _to_frame(payload: dict[str, Any], key: str) -> pd.DataFrame:
    value = payload.get(key)
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    return pd.DataFrame()


def inquire_balance(
    client: KISClient,
    account: KISAccountEnv,
    *,
    afhr_flpr_yn: str = "N",
    inqr_dvsn: str = "02",
    unpr_dvsn: str = "01",
    fund_sttl_icld_yn: str = "N",
    fncg_amt_auto_rdpt_yn: str = "N",
    prcs_dvsn: str = "00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_id = "VTTC8434R" if account.env_dv == "demo" else "TTTC8434R"
    payload = client.get(
        BALANCE_API_URL,
        tr_id=tr_id,
        params={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "AFHR_FLPR_YN": afhr_flpr_yn,
            "OFL_YN": "",
            "INQR_DVSN": inqr_dvsn,
            "UNPR_DVSN": unpr_dvsn,
            "FUND_STTL_ICLD_YN": fund_sttl_icld_yn,
            "FNCG_AMT_AUTO_RDPT_YN": fncg_amt_auto_rdpt_yn,
            "PRCS_DVSN": prcs_dvsn,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    return _to_frame(payload, "output1"), _to_frame(payload, "output2")


def inquire_psbl_order(
    client: KISClient,
    account: KISAccountEnv,
    *,
    pdno: str,
    ord_unpr: str,
    ord_dvsn: str = "01",
    cma_evlu_amt_icld_yn: str = "N",
    ovrs_icld_yn: str = "N",
) -> pd.DataFrame:
    tr_id = "VTTC8908R" if account.env_dv == "demo" else "TTTC8908R"
    payload = client.get(
        PSBL_ORDER_API_URL,
        tr_id=tr_id,
        params={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "PDNO": str(pdno).zfill(6),
            "ORD_UNPR": str(ord_unpr),
            "ORD_DVSN": ord_dvsn,
            "CMA_EVLU_AMT_ICLD_YN": cma_evlu_amt_icld_yn,
            "OVRS_ICLD_YN": ovrs_icld_yn,
        },
    )
    return _to_frame(payload, "output")


def order_cash(
    client: KISClient,
    account: KISAccountEnv,
    *,
    side: str,
    pdno: str,
    ord_dvsn: str,
    ord_qty: str,
    ord_unpr: str,
    excg_id_dvsn_cd: str = "KRX",
    sll_type: str = "",
    cndt_pric: str = "",
) -> pd.DataFrame:
    side_normalized = side.lower().strip()
    if side_normalized not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if account.env_dv == "demo":
        tr_id = "VTTC0012U" if side_normalized == "buy" else "VTTC0011U"
    else:
        tr_id = "TTTC0012U" if side_normalized == "buy" else "TTTC0011U"

    payload = client.post(
        ORDER_CASH_API_URL,
        tr_id=tr_id,
        payload={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "PDNO": str(pdno).zfill(6),
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(ord_qty),
            "ORD_UNPR": str(ord_unpr),
            "EXCG_ID_DVSN_CD": excg_id_dvsn_cd,
            "SLL_TYPE": sll_type,
            "CNDT_PRIC": cndt_pric,
        },
        require_hashkey=False,
    )
    return _to_frame(payload, "output")


def summarize_cash(balance_summary: pd.DataFrame) -> dict[str, float | None]:
    if balance_summary.empty:
        return {
            "dnca_tot_amt": None,
            "tot_evlu_amt": None,
            "tot_pfls": None,
        }
    row = balance_summary.iloc[0]
    def _num(name: str) -> float | None:
        value = pd.to_numeric(row.get(name), errors="coerce")
        return None if pd.isna(value) else float(value)
    return {
        "dnca_tot_amt": _num("dnca_tot_amt"),
        "tot_evlu_amt": _num("tot_evlu_amt"),
        "tot_pfls": _num("tot_evlu_pfls_amt") or _num("evlu_pfls_smtl_amt"),
    }


def compute_market_order_preview_qty(*, available_cash: float | None, target_weight: float | None, price: float | None) -> int:
    if not (available_cash and target_weight and price):
        return 0
    if available_cash <= 0 or target_weight <= 0 or price <= 0:
        return 0
    budget = available_cash * target_weight
    return max(int(math.floor(budget / price)), 0)
