from __future__ import annotations

"""KIS overseas stock order client for US market (Phase 7-3 Live integration).

Endpoints used:
  POST /uapi/overseas-stock/v1/trading/order           — 주문 (매수/매도)
  POST /uapi/overseas-stock/v1/trading/order-rvsecncl  — 정정/취소
  GET  /uapi/overseas-stock/v1/trading/inquire-nccs    — 미체결 조회
  GET  /uapi/overseas-stock/v1/trading/inquire-ccnl    — 일별 체결 조회
  GET  /uapi/overseas-stock/v1/trading/inquire-balance — 잔고/보유 종목 조회

broker_order_id format: "{SYMBOL}:{ODNO}"
  SYMBOL — 종목코드 (e.g. AAPL), needed for cancel PDNO field
  ODNO   — KIS 주문번호, returned by submit_order response output.ODNO
"""

import os
import sys
from pathlib import Path

from brokers.us.base import UsOrderClient


# ── KIS API 엔드포인트 ────────────────────────────────────────────────────────
_ORDER_URL = "/uapi/overseas-stock/v1/trading/order"
_RVSECNCL_URL = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
_NCCS_URL = "/uapi/overseas-stock/v1/trading/inquire-nccs"      # 미체결
_CCNL_URL = "/uapi/overseas-stock/v1/trading/inquire-ccnl"      # 일별체결
_BALANCE_URL = "/uapi/overseas-stock/v1/trading/inquire-balance" # 잔고/보유종목

# ── TR ID (실전 / 모의) ───────────────────────────────────────────────────────
_TR = {
    "real": {
        "buy":    "TTTT1002U",
        "sell":   "TTTT1006U",
        "cancel": "TTTT1004U",
        "nccs":   "TTTS3018R",
        "ccnl":   "TTTS3035R",
        "balance": "TTTS3012R",
    },
    "demo": {
        "buy":    "JTTT1002U",
        "sell":   "JTTT1006U",
        "cancel": "JTTT1004U",
        "nccs":   "JTTS3018R",
        "ccnl":   "JTTS3035R",
        "balance": "JTTS3012R",
    },
}


def _safe_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: object, default: str = "") -> str:
    return str(value or "").strip() or default


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _load_kis() -> tuple:
    """Lazily import KISClient + resolve_account_env from python/ package."""
    python_dir = Path(__file__).resolve().parents[2] / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from kis_client import KISClient  # type: ignore[import]
    from kis_live_account import resolve_account_env  # type: ignore[import]
    return KISClient, resolve_account_env


class KisOverseasOrderClient(UsOrderClient):
    """KIS OpenAPI 해외주식 주문 클라이언트.

    KIS_US_APP_KEY / KIS_US_APP_SECRET / KIS_US_CANO / KIS_US_ACNT_PRDT_CD 를
    우선 사용하며, 미설정 시 KIS_APP_KEY / KIS_CANO 로 폴백한다.
    해외거래소 코드는 KIS_US_EXCD (기본 NASD).
    """

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _client_and_account(self):
        import os
        KISClient, resolve_account_env = _load_kis()

        us_key = _env("KIS_US_APP_KEY")
        us_secret = _env("KIS_US_APP_SECRET")
        if us_key and us_secret:
            base_url = os.environ.get("KIS_BASE_URL", "")
            client = KISClient(base_url=base_url, app_key=us_key, app_secret=us_secret)
        else:
            client = KISClient.from_env()

        us_cano = _env("KIS_US_CANO")
        us_acnt = _env("KIS_US_ACNT_PRDT_CD")
        if us_cano and us_acnt:
            from kis_live_account import KISAccountEnv  # type: ignore[import]
            python_dir = Path(__file__).resolve().parents[2] / "python"
            import sys
            if str(python_dir) not in sys.path:
                sys.path.insert(0, str(python_dir))
            from kis_live_account import KISAccountEnv  # type: ignore[import]
            env_dv = os.environ.get("KIS_ENV_DV") or ("demo" if "demo" in os.environ.get("KIS_BASE_URL", "") else "real")
            account = KISAccountEnv(env_dv=env_dv, cano=us_cano, acnt_prdt_cd=us_acnt)
        else:
            account = resolve_account_env()

        return client, account

    def _tr(self, account, key: str) -> str:
        tier = "demo" if account.env_dv == "demo" else "real"
        return _TR[tier][key]

    def _excd(self) -> str:
        return _env("KIS_US_EXCD", "NASD").upper()

    @staticmethod
    def _make_broker_order_id(symbol: str, odno: str) -> str:
        return f"{symbol.upper()}:{odno}"

    @staticmethod
    def _parse_broker_order_id(broker_order_id: str) -> tuple[str, str]:
        parts = str(broker_order_id or "").split(":", 1)
        if len(parts) == 2:
            return parts[0].upper(), parts[1]
        return "", parts[0]

    @staticmethod
    def _failed(error_code: str, message: str, broker_order_id: str | None = None) -> dict:
        result: dict = {"status": "FAILED", "error_code": error_code, "message": message}
        if broker_order_id is not None:
            result["broker_order_id"] = broker_order_id
        return result

    # ── 주문 제출 ─────────────────────────────────────────────────────────────

    def submit_order(self, order_request: dict) -> dict:
        try:
            client, account = self._client_and_account()
        except Exception as exc:
            return self._failed("KIS_CLIENT_INIT_ERROR", str(exc))

        side = _safe_str(order_request.get("side")).upper()
        if side not in {"BUY", "SELL"}:
            return self._failed("INVALID_SIDE", f"side must be BUY or SELL, got: {side!r}")

        symbol = _safe_str(order_request.get("symbol")).upper()
        if not symbol:
            return self._failed("SYMBOL_MISSING", "symbol is required")

        limit_price = _safe_float(order_request.get("limit_price"))
        if limit_price is None or limit_price <= 0:
            return self._failed("LIMIT_PRICE_INVALID", "limit_price must be a positive number")

        order_qty = _safe_float(order_request.get("order_qty"))
        if order_qty is None or order_qty <= 0:
            return self._failed("ORDER_QTY_INVALID", "order_qty must be a positive number")

        tr_id = self._tr(account, side.lower())
        body = {
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "OVRS_EXCG_CD": self._excd(),
            "PDNO": symbol,
            "ORD_DVSN": "00",                          # 지정가 (해외는 지정가만)
            "ORD_QTY": str(int(order_qty)),
            "OVRS_ORD_UNPR": f"{limit_price:.2f}",
            "ORD_SVR_DVSN_CD": "0",
        }

        try:
            resp = client.post(_ORDER_URL, tr_id=tr_id, payload=body, require_hashkey=True, no_retry=True)
        except Exception as exc:
            return self._failed("KIS_ORDER_API_ERROR", str(exc))

        output = resp.get("output") or {}
        odno = _safe_str(output.get("ODNO"))
        if not odno:
            return self._failed("NO_ORDER_NUMBER", f"KIS response missing ODNO: {resp}")

        broker_order_id = self._make_broker_order_id(symbol, odno)
        return {
            "status": "ACCEPTED",
            "broker_order_id": broker_order_id,
            "message": f"KIS overseas order accepted symbol={symbol} odno={odno}",
            "raw_output": output,
        }

    # ── 주문 취소 ─────────────────────────────────────────────────────────────

    def cancel_order(self, broker_order_id: str) -> dict:
        try:
            client, account = self._client_and_account()
        except Exception as exc:
            return self._failed("KIS_CLIENT_INIT_ERROR", str(exc), broker_order_id)

        symbol, odno = self._parse_broker_order_id(broker_order_id)
        if not odno:
            return self._failed("BROKER_ORDER_ID_INVALID", f"Cannot parse ODNO from: {broker_order_id!r}", broker_order_id)

        tr_id = self._tr(account, "cancel")
        body = {
            "CANO": account.cano,
            "ACNT_PRDT_CD": account.acnt_prdt_cd,
            "OVRS_EXCG_CD": self._excd(),
            "PDNO": symbol,
            "ORGN_ODNO": odno,
            "ORD_SVR_DVSN_CD": "0",
            "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
            "ORD_QTY": "0",
            "OVRS_ORD_UNPR": "0",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "ORD_DVSN": "00",
        }

        try:
            resp = client.post(_RVSECNCL_URL, tr_id=tr_id, payload=body, no_retry=True)
        except Exception as exc:
            return self._failed("KIS_CANCEL_API_ERROR", str(exc), broker_order_id)

        output = resp.get("output") or {}
        return {
            "status": "CANCELED",
            "broker_order_id": broker_order_id,
            "message": f"KIS overseas order canceled odno={odno}",
            "raw_output": output,
        }

    # ── 주문 상태 조회 ────────────────────────────────────────────────────────

    def get_order_status(self, broker_order_id: str) -> dict:
        """미체결 → 일별체결 순서로 조회해 최신 상태 반환."""
        symbol, odno = self._parse_broker_order_id(broker_order_id)
        if not odno:
            return {"status": "SYNC_ERROR", "broker_order_id": broker_order_id,
                    "error_code": "BROKER_ORDER_ID_INVALID",
                    "message": f"Cannot parse ODNO from: {broker_order_id!r}"}

        try:
            client, account = self._client_and_account()
        except Exception as exc:
            return {"status": "SYNC_ERROR", "broker_order_id": broker_order_id,
                    "error_code": "KIS_CLIENT_INIT_ERROR", "message": str(exc)}

        # ① 미체결 조회 (주문 당일 미체결 존재 여부 확인)
        nccs_status = self._query_nccs(client, account, odno)
        if nccs_status is not None:
            nccs_status["broker_order_id"] = broker_order_id
            return nccs_status

        # ② 일별 체결 조회 (이미 체결/취소된 경우)
        ccnl_status = self._query_ccnl(client, account, odno)
        if ccnl_status is not None:
            ccnl_status["broker_order_id"] = broker_order_id
            return ccnl_status

        return {"status": "ORDER_UNKNOWN", "broker_order_id": broker_order_id,
                "message": "Order not found in nccs or ccnl"}

    def _query_nccs(self, client, account, odno: str) -> dict | None:
        """미체결 내역에서 해당 주문번호 검색. 없으면 None 반환."""
        tr_id = self._tr(account, "nccs")
        try:
            resp = client.get(
                _NCCS_URL,
                tr_id=tr_id,
                params={
                    "CANO": account.cano,
                    "ACNT_PRDT_CD": account.acnt_prdt_cd,
                    "OVRS_EXCG_CD": self._excd(),
                    "SORT_SQN": "DS",
                    "CTX_AREA_FK200": "",
                    "CTX_AREA_NK200": "",
                },
            )
        except Exception:
            return None

        rows = resp.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            if _safe_str(row.get("ODNO")) == odno:
                rmn_qty = _safe_float(row.get("RMN_QTY")) or 0.0
                ccld_qty = _safe_float(row.get("CCLD_QTY")) or 0.0
                if rmn_qty > 0 and ccld_qty == 0:
                    status = "ORDER_OPEN"
                elif rmn_qty > 0 and ccld_qty > 0:
                    status = "ORDER_PARTIALLY_FILLED"
                else:
                    status = "ORDER_OPEN"
                return {
                    "status": status,
                    "filled_qty": ccld_qty,
                    "remaining_qty": rmn_qty,
                    "avg_filled_price": _safe_float(row.get("AVG_PRVS")),
                    "raw": row,
                }
        return None

    def _query_ccnl(self, client, account, odno: str) -> dict | None:
        """일별체결 내역에서 해당 주문번호 검색. 없으면 None 반환."""
        tr_id = self._tr(account, "ccnl")
        try:
            resp = client.get(
                _CCNL_URL,
                tr_id=tr_id,
                params={
                    "CANO": account.cano,
                    "ACNT_PRDT_CD": account.acnt_prdt_cd,
                    "PDNO": "",
                    "ORD_STRT_DT": "",
                    "ORD_END_DT": "",
                    "SLL_BUY_DVSN": "00",
                    "CCL_NCCS_DVSN": "00",
                    "OVRS_EXCG_CD": self._excd(),
                    "SORT_SQN": "DS",
                    "ORD_DT": "",
                    "ORD_GNO_BRNO": "",
                    "ODNO": odno,
                    "CTX_AREA_NK200": "",
                    "CTX_AREA_FK200": "",
                },
            )
        except Exception:
            return None

        rows = resp.get("output") or []
        if isinstance(rows, dict):
            rows = [rows]

        # ODNO 기준으로 모든 체결 행을 합산
        matched = [r for r in rows if _safe_str(r.get("ODNO")) == odno]
        if not matched:
            return None

        total_filled = sum(_safe_float(r.get("CCLD_QTY")) or 0.0 for r in matched)
        cncl_yn = _safe_str(matched[0].get("CNCL_YN"))
        rmn_qty = _safe_float(matched[0].get("RMN_QTY")) or 0.0

        if cncl_yn == "Y":
            status = "ORDER_CANCELED"
        elif rmn_qty <= 0 and total_filled > 0:
            status = "ORDER_FILLED"
        elif total_filled > 0:
            status = "ORDER_PARTIALLY_FILLED"
        else:
            status = "ORDER_UNKNOWN"

        avg_price = _safe_float(matched[0].get("AVG_PRVS")) or _safe_float(matched[0].get("CCLD_UNPR"))
        return {
            "status": status,
            "filled_qty": total_filled,
            "remaining_qty": rmn_qty,
            "avg_filled_price": avg_price,
            "raw": matched[0],
        }

    # ── 체결 내역 ─────────────────────────────────────────────────────────────

    def get_order_fills(self, broker_order_id: str) -> list[dict]:
        result = self.get_order_status(broker_order_id)
        filled_qty = result.get("filled_qty")
        if not filled_qty:
            return []
        return [{
            "broker_order_id": broker_order_id,
            "filled_qty": filled_qty,
            "filled_price": result.get("avg_filled_price"),
            "status": result.get("status"),
        }]

    # ── 계좌 잔고 / 보유 종목 조회 ────────────────────────────────────────────

    def get_account_balance(self) -> dict:
        """해외주식 잔고 조회 (TTTS3012R).

        Returns:
            {
              "cash_usd": float,          # 외화예수금(USD)
              "total_eval_usd": float,    # 유가증권 평가금액
              "total_equity_usd": float,  # 총 자산(현금+평가)
              "positions": [
                {
                  "symbol": str,
                  "qty": float,
                  "avg_price": float,
                  "current_price": float,
                  "market_value_usd": float,
                  "unrealized_pnl_usd": float,
                  "unrealized_pnl_pct": float,
                  "raw": dict,
                },
                ...
              ],
              "raw_output1": dict,   # output1 (계좌 요약)
              "raw_output2": list,   # output2 (보유종목 목록)
            }
        """
        try:
            client, account = self._client_and_account()
        except Exception as exc:
            return {"error": str(exc)}

        tr_id = self._tr(account, "balance")
        try:
            resp = client.get(
                _BALANCE_URL,
                tr_id=tr_id,
                params={
                    "CANO": account.cano,
                    "ACNT_PRDT_CD": account.acnt_prdt_cd,
                    "OVRS_EXCG_CD": self._excd(),
                    "TR_CRCY_CD": "USD",
                    "CTX_AREA_FK200": "",
                    "CTX_AREA_NK200": "",
                },
            )
        except Exception as exc:
            return {"error": str(exc)}

        # KIS 응답: output1 = 보유종목 목록(list), output2 = 계좌 요약(dict)
        output1_rows = resp.get("output1") or []
        if isinstance(output1_rows, dict):
            output1_rows = [output1_rows]
        output2 = resp.get("output2") or {}
        if isinstance(output2, list):
            output2 = output2[0] if output2 else {}

        total_eval = _safe_float(output2.get("tot_evlu_pfls_amt")) or 0.0
        total_pnl = _safe_float(output2.get("ovrs_tot_pfls")) or 0.0

        positions = []
        for row in output1_rows:
            symbol = _safe_str(row.get("ovrs_pdno"))
            if not symbol:
                continue
            qty = _safe_float(row.get("ovrs_cblc_qty")) or 0.0
            avg_price = _safe_float(row.get("pchs_avg_pric")) or 0.0
            current_price = _safe_float(row.get("now_pric2")) or 0.0
            market_value = _safe_float(row.get("ovrs_stck_evlu_amt")) or qty * current_price
            pnl = _safe_float(row.get("frcr_evlu_pfls_amt")) or (market_value - qty * avg_price)
            cost = qty * avg_price
            pnl_pct = (pnl / cost) if cost > 0 else None
            positions.append({
                "symbol": symbol,
                "qty": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value_usd": market_value,
                "unrealized_pnl_usd": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "raw": row,
            })

        return {
            "total_eval_usd": total_eval,
            "total_pnl_usd": total_pnl,
            "positions": positions,
            "raw_output1": output1_rows,
            "raw_output2": output2,
        }
