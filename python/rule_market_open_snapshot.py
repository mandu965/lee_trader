from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kis_client import KISClient, KISError
from kis_live_account import KISAccountEnv, infer_env_dv

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OUT = OUTPUT_DIR / "rule_market_open_snapshot.json"
QUOTE_API_URL = "/uapi/domestic-stock/v1/quotations/inquire-price"
QUOTE_TR_ID = "FHKST01010100"


@dataclass(frozen=True)
class RuleAccountEnv(KISAccountEnv):
    rule_app_id: str | None = None
    rule_app_password: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch RULE before-open market snapshot from KIS.")
    parser.add_argument("--symbols", default="", help="Comma-separated 6-digit symbols.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)


def resolve_rule_account_env() -> RuleAccountEnv:
    _load_env()
    base_url = os.getenv("KIS_BASE_URL")
    env_dv = os.getenv("KIS_ENV_DV") or infer_env_dv(base_url)
    cano = (os.getenv("KIS_RULE_CANO") or "").strip()
    acnt_prdt_cd = (os.getenv("KIS_RULE_ACNT_PRDT_CD") or "").strip()
    rule_app_id = (os.getenv("KIS_APP_RULE_ID") or "").strip() or None
    rule_app_password = (os.getenv("KIS_APP_RULE_PASSWORD") or "").strip() or None

    if not cano or not acnt_prdt_cd:
        raise ValueError("RULE account env missing. Set KIS_RULE_CANO and KIS_RULE_ACNT_PRDT_CD.")

    return RuleAccountEnv(
        env_dv=env_dv,
        cano=cano,
        acnt_prdt_cd=acnt_prdt_cd,
        hts_id=rule_app_id,
        rule_app_id=rule_app_id,
        rule_app_password=rule_app_password,
    )


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_single_open_snapshot(client: KISClient, symbol: str) -> dict[str, Any]:
    payload = client.get(
        QUOTE_API_URL,
        tr_id=QUOTE_TR_ID,
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(symbol).zfill(6),
        },
    )
    output = payload.get("output") or {}
    current_price = _to_float(output.get("stck_prpr"))
    open_price = _to_float(output.get("stck_oprc"))
    high_price = _to_float(output.get("stck_hgpr"))
    low_price = _to_float(output.get("stck_lwpr"))
    prev_close = _to_float(output.get("stck_sdpr"))
    acc_volume = _to_float(output.get("acml_vol"))
    acc_trading_value = _to_float(output.get("acml_tr_pbmn"))
    actual_open_gap = None
    if open_price is not None and open_price > 0 and prev_close not in {None, 0.0}:
        actual_open_gap = (open_price / prev_close) - 1.0
    elif open_price is not None and open_price <= 0:
        import logging as _logging
        _logging.warning(
            "[RULE_OPEN_SNAPSHOT] open_price=%s is zero/negative for symbol=%s — actual_open_gap set to None (장 시작 전 또는 시가 미확정)",
            open_price, symbol,
        )
    intraday_return = None
    if current_price is not None and prev_close not in {None, 0.0}:
        intraday_return = (current_price / prev_close) - 1.0
    high_from_open = None
    low_from_open = None
    if open_price not in {None, 0.0}:
        if high_price is not None:
            high_from_open = (high_price / open_price) - 1.0
        if low_price is not None:
            low_from_open = (low_price / open_price) - 1.0

    return {
        "symbol": str(symbol).zfill(6),
        "current_price": current_price,
        "open_price": open_price,
        "high_price": high_price,
        "low_price": low_price,
        "previous_close": prev_close,
        "intraday_return": intraday_return,
        "intraday_volume": acc_volume,
        "intraday_trading_value": acc_trading_value,
        "actual_open_gap": actual_open_gap,
        "high_from_open": high_from_open,
        "low_from_open": low_from_open,
        "market_data_available": (
            current_price is not None and current_price > 0
            and open_price is not None and open_price > 0
            and prev_close is not None and prev_close > 0
        ),
        "raw_output": output,
    }


def fetch_open_snapshot_with_retry(
    symbols: list[str],
    *,
    max_retry: int | None = None,
    retry_interval_sec: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    client = KISClient.from_rule_env()
    client.issue_access_token()
    max_retry = max_retry if max_retry is not None else int(os.getenv("RULE_OPEN_SNAPSHOT_MAX_RETRY", "3"))
    retry_interval_sec = retry_interval_sec if retry_interval_sec is not None else int(os.getenv("RULE_OPEN_SNAPSHOT_RETRY_INTERVAL_SEC", "10"))

    snapshots: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    last_status = "ok"
    for symbol in [str(item).zfill(6) for item in symbols if str(item).strip()]:
        last_error: Exception | None = None
        for attempt in range(max_retry):
            try:
                snapshots[symbol] = fetch_single_open_snapshot(client, symbol)
                last_error = None
                break
            except KISError as exc:
                last_error = exc
                last_status = "market_data_unavailable"
            except Exception as exc:
                last_error = exc
                last_status = "unknown_failed"
            if attempt + 1 < max_retry:
                time.sleep(max(retry_interval_sec, 1))
        if last_error is not None:
            failures.append(
                {
                    "symbol": symbol,
                    "error": str(last_error),
                }
            )

    return snapshots, failures, last_status


def check_market_data_available(symbols: list[str]) -> dict[str, Any]:
    try:
        resolve_rule_account_env()
        snapshots, failures, api_health_status = fetch_open_snapshot_with_retry(symbols)
    except Exception as exc:
        return {
            "market_data_available": False,
            "api_health_status": "auth_failed",
            "api_failure_reason": str(exc),
            "snapshots": {},
            "failures": [{"symbol": None, "error": str(exc)}],
        }

    ok_count = sum(1 for row in snapshots.values() if row.get("market_data_available"))
    return {
        "market_data_available": ok_count == len([s for s in symbols if str(s).strip()]) and not failures,
        "api_health_status": api_health_status if not failures else "market_data_unavailable",
        "api_failure_reason": None if not failures else "; ".join(f"{row['symbol']}:{row['error']}" for row in failures[:10]),
        "snapshots": snapshots,
        "failures": failures,
    }


def save_snapshot_payload(path: Path, payload: dict[str, Any]) -> None:
    resolved = path if path.is_absolute() else ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    symbols = [str(token).strip().zfill(6) for token in str(args.symbols or "").split(",") if str(token).strip()]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": symbols,
        **check_market_data_available(symbols),
    }
    save_snapshot_payload(args.out_json, payload)
    print(f"saved {args.out_json if args.out_json.is_absolute() else (ROOT / args.out_json)}")


if __name__ == "__main__":
    main()
