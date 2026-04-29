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
DAILY_CCLD_API_URL = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
PSBL_RVSECNCL_API_URL = "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
ORDER_RVSECNCL_API_URL = "/uapi/domestic-stock/v1/trading/order-rvsecncl"


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


def inquire_daily_ccld(
    client: KISClient,
    account: KISAccountEnv,
    *,
    start_date: str,
    end_date: str,
    sll_buy_dvsn_cd: str = "00",
    inqr_dvsn: str = "00",
    pdno: str = "",
    odno: str = "",
    ccld_dvsn: str = "00",
    inqr_dvsn_3: str = "00",
    inqr_dvsn_1: str = "",
    inqr_dvsn_2: str = "",
    ord_gno_brno: str = "",
    ctx_area_fk100: str = "",
    ctx_area_nk100: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_id = "VTTC8001R" if account.env_dv == "demo" else "TTTC8001R"
    payload = client.get(
        DAILY_CCLD_API_URL,
        tr_id=tr_id,
        params={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "INQR_STRT_DT": str(start_date).replace("-", ""),
            "INQR_END_DT": str(end_date).replace("-", ""),
            "SLL_BUY_DVSN_CD": sll_buy_dvsn_cd,
            "INQR_DVSN": inqr_dvsn,
            "PDNO": str(pdno).zfill(6) if str(pdno or "").strip() else "",
            "CCLD_DVSN": ccld_dvsn,
            "ORD_GNO_BRNO": ord_gno_brno,
            "ODNO": str(odno or "").strip(),
            "INQR_DVSN_3": inqr_dvsn_3,
            "INQR_DVSN_1": inqr_dvsn_1,
            "INQR_DVSN_2": inqr_dvsn_2,
            "CTX_AREA_FK100": ctx_area_fk100,
            "CTX_AREA_NK100": ctx_area_nk100,
        },
    )
    return _to_frame(payload, "output1"), _to_frame(payload, "output2")


def inquire_psbl_rvsecncl(
    client: KISClient,
    account: KISAccountEnv,
    *,
    inqr_dvsn_1: str = "1",
    inqr_dvsn_2: str = "0",
    ctx_area_fk100: str = "",
    ctx_area_nk100: str = "",
) -> pd.DataFrame:
    # Official KIS sample uses TTTC0084R for this endpoint.
    payload = client.get(
        PSBL_RVSECNCL_API_URL,
        tr_id="TTTC0084R",
        params={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "INQR_DVSN_1": inqr_dvsn_1,
            "INQR_DVSN_2": inqr_dvsn_2,
            "CTX_AREA_FK100": ctx_area_fk100,
            "CTX_AREA_NK100": ctx_area_nk100,
        },
    )
    return _to_frame(payload, "output")


def order_rvsecncl(
    client: KISClient,
    account: KISAccountEnv,
    *,
    krx_fwdg_ord_orgno: str,
    orgn_odno: str,
    ord_dvsn: str,
    rvse_cncl_dvsn_cd: str,
    ord_qty: str,
    ord_unpr: str,
    qty_all_ord_yn: str = "Y",
    excg_id_dvsn_cd: str = "KRX",
    cndt_pric: str = "",
) -> pd.DataFrame:
    tr_id = "VTTC0013U" if account.env_dv == "demo" else "TTTC0013U"
    payload = client.post(
        ORDER_RVSECNCL_API_URL,
        tr_id=tr_id,
        payload={
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": str(krx_fwdg_ord_orgno or "").strip(),
            "ORGN_ODNO": str(orgn_odno or "").strip(),
            "ORD_DVSN": str(ord_dvsn or "").strip(),
            "RVSE_CNCL_DVSN_CD": str(rvse_cncl_dvsn_cd or "").strip(),
            "ORD_QTY": str(ord_qty),
            "ORD_UNPR": str(ord_unpr),
            "QTY_ALL_ORD_YN": str(qty_all_ord_yn or "Y").strip(),
            "EXCG_ID_DVSN_CD": str(excg_id_dvsn_cd or "KRX").strip(),
            "CNDT_PRIC": str(cndt_pric or "").strip(),
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


def compute_market_order_preview_qty(
    *,
    available_cash: float | None,
    target_weight: float | None,
    price: float | None,
    total_assets: float | None = None,
    current_position_value: float | None = None,
) -> int:
    if target_weight is None or price is None:
        return 0
    if available_cash is None or available_cash <= 0 or target_weight <= 0 or price <= 0:
        return 0
    desired_budget = available_cash * target_weight
    if total_assets is not None and total_assets > 0:
        desired_budget = total_assets * target_weight
        if current_position_value is not None and current_position_value > 0:
            desired_budget = max(desired_budget - current_position_value, 0.0)
    budget = min(float(available_cash), float(desired_budget))
    return max(int(math.floor(budget / price)), 0)
