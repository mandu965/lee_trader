from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from kis_client import KISClient, KISError
from payload_store import upsert_json_payload
from production_config import get_production_config_value, get_production_versions


ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = ROOT / "serving"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

INPUT_DAILY = SERVING_DIR / "daily_recommendations.json"
INPUT_HOLDINGS = DATA_DIR / "live_account_holdings.csv"
OUTPUT_INTRADAY = SERVING_DIR / "intraday_recommendations.json"
OUTPUT_STATUS = OUTPUTS_DIR / "intraday_refresh_status.json"
NAVER_REALTIME_URL = "https://polling.finance.naver.com/api/realtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build midday intraday recommendations for afternoon operations.")
    parser.add_argument("--daily-json", type=Path, default=INPUT_DAILY)
    parser.add_argument("--holdings-csv", type=Path, default=INPUT_HOLDINGS)
    parser.add_argument("--out-json", type=Path, default=OUTPUT_INTRADAY)
    parser.add_argument("--status-json", type=Path, default=OUTPUT_STATUS)
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, dtype=str, encoding="utf-8-sig", low_memory=False)


def to_num(value: Any) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(num) else float(num)


def load_live_holdings_codes(path: Path) -> list[str]:
    df = read_csv(path)
    if df.empty or "code" not in df.columns:
        return []
    return sorted({str(code).strip().zfill(6) for code in df["code"].dropna().tolist() if str(code).strip()})


def fetch_quote_map(codes: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not codes:
        return {}, []

    try:
        client = KISClient.from_env()
        client.issue_access_token()
    except Exception:
        return {}, list(sorted(set(codes)))
    quotes: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for code in sorted(set(codes)):
        try:
            payload = client.get(
                "/uapi/domestic-stock/v1/quotations/inquire-price",
                tr_id="FHKST01010100",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                },
            )
            out = payload.get("output") or payload.get("output1") or {}
            if not isinstance(out, dict) or not out:
                failures.append(code)
                continue
            raw_change_pct = to_num(out.get("prdy_ctrt"))
            change_pct = None if raw_change_pct is None else raw_change_pct / 100.0
            quotes[code] = {
                "current_price": to_num(out.get("stck_prpr")),
                "open_price": to_num(out.get("stck_oprc")),
                "high_price": to_num(out.get("stck_hgpr")),
                "low_price": to_num(out.get("stck_lwpr")),
                "change_pct": change_pct,
                "change_amount": to_num(out.get("prdy_vrss")),
                "volume": to_num(out.get("acml_vol")),
                "trading_value": to_num(out.get("acml_tr_pbmn")),
            }
        except KISError:
            failures.append(code)
        except Exception:
            failures.append(code)

    return quotes, failures


def fetch_naver_quote_map(codes: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    query_codes = [str(code).strip().zfill(6) for code in codes if str(code).strip()]
    if not query_codes:
        return {}, []
    verify_ssl = str(os.getenv("NAVER_FALLBACK_VERIFY_SSL", "0")).strip().lower() in {"1", "true", "yes", "on"}
    timeout_sec = max(5, int(os.getenv("NAVER_FALLBACK_TIMEOUT_SEC", "20")))
    query = "|".join(f"SERVICE_ITEM:{code}" for code in query_codes)
    try:
        response = requests.get(
            NAVER_REALTIME_URL,
            params={"query": query, "_callback": ""},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout_sec,
            verify=verify_ssl,
        )
        response.raise_for_status()
        payload = response.json()
        areas = ((payload or {}).get("result") or {}).get("areas") or []
        quotes: dict[str, dict[str, Any]] = {}
        for area in areas:
            for item in area.get("datas") or []:
                code = str(item.get("cd") or "").strip().zfill(6)
                if not code:
                    continue
                change_pct = to_num(item.get("cr"))
                quotes[code] = {
                    "current_price": to_num(item.get("nv")),
                    "open_price": to_num(item.get("ov")),
                    "high_price": to_num(item.get("hv")),
                    "low_price": to_num(item.get("lv")),
                    "change_pct": None if change_pct is None else change_pct / 100.0,
                    "change_amount": to_num(item.get("cv")),
                    "volume": to_num(item.get("aq")),
                    "trading_value": to_num(item.get("aa")),
                    "source": "naver_fallback",
                }
        failures = [code for code in query_codes if code not in quotes]
        return quotes, failures
    except Exception:
        return {}, query_codes


def classify_intraday_candidate(item: dict[str, Any], quote: dict[str, Any] | None, held_codes: set[str]) -> tuple[str, list[str]]:
    security = item.get("security") or {}
    selection = item.get("selection") or {}
    buy_eligibility = item.get("buy_eligibility") or {}
    scores = item.get("scores") or {}
    code = str(security.get("code") or "").strip().zfill(6)
    buyability_status = str(selection.get("buyability_status") or "").upper()
    entry_rule_pass = selection.get("entry_rule_pass") is not False
    eligibility_status = str(buy_eligibility.get("status") or "").upper()
    confidence_state = str(scores.get("confidence_state_v2") or "").upper()
    change_pct = None if not quote else to_num(quote.get("change_pct"))
    trading_value = None if not quote else to_num(quote.get("trading_value"))

    reasons: list[str] = []
    verdict = "PRIORITY"

    if code in held_codes:
        verdict = "HOLDING_REVIEW"
        reasons.append("기존 보유 종목이라 신규 진입보다 보유 관리 기준으로 봅니다.")
    if buyability_status == "BLOCK" or eligibility_status == "BLOCK":
        verdict = "CAUTION"
        reasons.append("기본 운영 기준에서 이미 매수 제한 상태입니다.")
    if confidence_state in {"BLOCKED", "WEAK"}:
        verdict = "CAUTION"
        reasons.append("confidence_state_v2가 약해 오후장 신규 진입 우선순위가 낮습니다.")
    if entry_rule_pass is False:
        verdict = "CAUTION"
        reasons.append("entry_rule_pass가 false라 장중에는 추격보다 보류가 맞습니다.")
    if quote is None:
        verdict = "CAUTION"
        reasons.append("장중 시세를 받지 못해 오후장 판단 근거가 부족합니다.")
        return verdict, reasons

    if change_pct is not None and change_pct >= 0.10:
        verdict = "CAUTION"
        reasons.append(f"장중 상승률이 {change_pct * 100:.1f}%로 높아 추격 매수는 피하는 편이 맞습니다.")
    elif change_pct is not None and change_pct >= 0.05:
        reasons.append(f"장중 상승률이 {change_pct * 100:.1f}%라 진입 시 체결가 부담을 먼저 확인해야 합니다.")
    elif change_pct is not None and change_pct <= -0.05:
        verdict = "CAUTION"
        reasons.append(f"장중 하락률이 {change_pct * 100:.1f}%라 급락 사유 확인이 먼저입니다.")
    elif change_pct is not None:
        reasons.append(f"장중 변동률은 {change_pct * 100:.1f}% 수준입니다.")

    min_trading_value = float(get_production_config_value(["buy_candidate", "min_trading_value"], 5_000_000_000.0))
    intraday_liquidity_floor = min_trading_value * 0.30
    if trading_value is not None and trading_value < intraday_liquidity_floor:
        verdict = "CAUTION"
        reasons.append(
            f"장중 거래대금이 {trading_value:,.0f}원으로 오후장 집행 기준({intraday_liquidity_floor:,.0f}원)보다 낮습니다."
        )
    elif trading_value is not None:
        reasons.append(f"장중 거래대금은 {trading_value:,.0f}원입니다.")

    if verdict == "PRIORITY" and not reasons:
        reasons.append("기본 후보 상태와 장중 가격 흐름 모두 오후장 검토 범위 안에 있습니다.")
    return verdict, reasons[:3]


def build_candidate_row(item: dict[str, Any], quote: dict[str, Any] | None, verdict: str, reasons: list[str]) -> dict[str, Any]:
    security = item.get("security") or {}
    selection = item.get("selection") or {}
    scores = item.get("scores") or {}
    market_signals = item.get("market_signals") or {}
    buy_eligibility = item.get("buy_eligibility") or {}
    return {
        "code": security.get("code"),
        "name": security.get("name"),
        "market": security.get("market"),
        "sector": security.get("sector"),
        "dominant_theme": security.get("dominant_theme"),
        "buy_rank": item.get("buy_rank"),
        "final_score": scores.get("final_score"),
        "confidence_score": scores.get("confidence_score"),
        "confidence_state_v2": scores.get("confidence_state_v2"),
        "buyability_status": selection.get("buyability_status"),
        "watchlist_tier": selection.get("buyability_watchlist_tier"),
        "promotion_readiness_score": selection.get("buyability_promotion_readiness_score"),
        "buy_eligibility_status": buy_eligibility.get("status"),
        "buy_eligibility_score": buy_eligibility.get("score"),
        "entry_rule_pass": selection.get("entry_rule_pass"),
        "pred_return_60d": market_signals.get("pred_return_60d"),
        "prob_top20_60d": market_signals.get("prob_top20_60d"),
        "pred_mdd_60d": market_signals.get("pred_mdd_60d"),
        "intraday_verdict": verdict,
        "intraday_reasons": reasons,
        "intraday_quote": quote or {},
    }


def main() -> int:
    args = parse_args()
    now = datetime.now()
    daily = read_json(args.daily_json)
    items = daily.get("items") if isinstance(daily.get("items"), list) else []
    if not items:
        payload = {
            "entity": "intraday_recommendations",
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "session_date": now.strftime("%Y-%m-%d"),
            "status": "empty",
            "message": "daily_recommendations.json에 후보가 없어 장중 payload를 만들지 않았습니다.",
        }
        write_json(args.out_json, payload)
        write_json(args.status_json, payload)
        upsert_json_payload(
            "intraday_recommendations",
            payload,
            asof_date=payload.get("asof_date") or payload.get("session_date"),
            generated_at=payload.get("generated_at"),
            source_path=args.out_json,
        )
        return 0

    candidate_codes = [str((item.get("security") or {}).get("code") or "").strip().zfill(6) for item in items]
    held_codes = set(load_live_holdings_codes(args.holdings_csv))
    quote_codes = [code for code in candidate_codes if code]
    for code in held_codes:
        if code not in quote_codes:
            quote_codes.append(code)

    status_payload: dict[str, Any] = {
        "generated_at": now.isoformat(timespec="seconds"),
        "session_date": now.strftime("%Y-%m-%d"),
        "status": "ok",
        "quote_code_count": len(quote_codes),
    }

    try:
        quotes, failures = fetch_quote_map(quote_codes)
        if failures:
            fallback_quotes, fallback_failures = fetch_naver_quote_map(failures)
            quotes.update(fallback_quotes)
            failures = fallback_failures
        status_payload["quote_success_count"] = len(quotes)
        status_payload["quote_failure_count"] = len(failures)
        status_payload["quote_failures"] = failures[:20]
        status_payload["quote_source"] = "kis_with_naver_fallback"
    except Exception as exc:
        fallback_rows = []
        for item in items:
            row = build_candidate_row(
                item,
                None,
                "CAUTION",
                [f"KIS 장중 시세 조회 실패로 오후장 신규 진입 판단을 보수적으로 전환합니다: {exc}"],
            )
            fallback_rows.append(row)
        payload = {
            "entity": "intraday_recommendations",
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "session_date": now.strftime("%Y-%m-%d"),
            "asof_date": daily.get("asof_date"),
            "basis": "intraday",
            "basis_label": "장중 기준",
            "status": "fallback",
            "message": f"장중 시세 조회 실패: {exc}",
            "source_daily_asof_date": daily.get("asof_date"),
            "gate_overall_status": daily.get("gate_overall_status"),
            "walkforward_acceptance_status": daily.get("walkforward_acceptance_status"),
            "checklist": [
                "장중 시세 조회가 실패해 오후장 신규 진입은 보수적으로 해석합니다.",
                "기존 마감 후보를 그대로 추격하지 말고 HTS/MTS 실시간 시세를 직접 확인해야 합니다.",
                "오늘은 전 종목을 CAUTION으로 보고, 가격 급등/급락과 거래대금부터 점검합니다.",
            ],
            "summary": {
                "decision": "오후장은 보수 대응",
                "reason": "KIS 장중 시세 조회가 실패해 자동 재선별 근거가 부족합니다.",
                "action_guide": "수동으로 실시간 호가를 확인하기 전에는 신규 진입보다 관찰이 우선입니다.",
            },
            "priority_candidates": [],
            "caution_candidates": fallback_rows[: args.max_candidates],
        }
        write_json(args.out_json, payload)
        status_payload.update({"status": "fallback", "error": str(exc)})
        write_json(args.status_json, status_payload)
        upsert_json_payload(
            "intraday_recommendations",
            payload,
            asof_date=payload.get("asof_date") or payload.get("session_date"),
            generated_at=payload.get("generated_at"),
            source_path=args.out_json,
        )
        return 0

    priority_rows: list[dict[str, Any]] = []
    caution_rows: list[dict[str, Any]] = []
    for item in items:
        code = str((item.get("security") or {}).get("code") or "").strip().zfill(6)
        quote = quotes.get(code)
        verdict, reasons = classify_intraday_candidate(item, quote, held_codes)
        row = build_candidate_row(item, quote, verdict, reasons)
        if verdict == "PRIORITY":
            priority_rows.append(row)
        else:
            caution_rows.append(row)

    def _priority_key(row: dict[str, Any]) -> tuple[float, float]:
        readiness = to_num(row.get("promotion_readiness_score")) or -1.0
        change_pct = to_num((row.get("intraday_quote") or {}).get("change_pct"))
        change_penalty = abs(change_pct) if change_pct is not None else 9.0
        return (readiness, -change_penalty)

    priority_rows.sort(key=_priority_key, reverse=True)
    caution_rows.sort(key=lambda row: (to_num(row.get("promotion_readiness_score")) or -1.0), reverse=True)

    payload = {
        "entity": "intraday_recommendations",
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "session_date": now.strftime("%Y-%m-%d"),
        "asof_date": daily.get("asof_date"),
        "basis": "intraday",
        "basis_label": "장중 기준",
        "status": "current",
        "source_daily_generated_at": daily.get("generated_at"),
        "source_daily_asof_date": daily.get("asof_date"),
        "gate_overall_status": daily.get("gate_overall_status"),
        "walkforward_acceptance_status": daily.get("walkforward_acceptance_status"),
        "versions": get_production_versions(),
        "counts": {
            "priority": len(priority_rows),
            "caution": len(caution_rows),
            "quote_success": len(quotes),
            "quote_failure": len(failures),
        },
        "summary": {
            "decision": "오후장은 장중 기준으로 재선별",
            "reason": "전일 종가 후보를 그대로 추격하지 않고, 12시 시세와 거래대금으로 오후장 대응 후보를 다시 거릅니다.",
            "action_guide": "PRIORITY만 우선 보고, CAUTION은 추격·급락·유동성 부족 사유를 먼저 확인합니다.",
        },
        "checklist": [
            "PRIORITY 후보라도 실시간 호가와 체결 강도를 최종 확인합니다.",
            "장중 상승률이 큰 종목은 추격보다 눌림 또는 보류를 우선합니다.",
            "CAUTION 후보는 급등·급락·저유동성 사유를 해소하기 전까지 신규 진입을 보류합니다.",
            "보유 종목은 HOLDING_REVIEW로 보고 신규 진입과 분리해서 판단합니다.",
        ],
        "priority_candidates": priority_rows[: args.max_candidates],
        "caution_candidates": caution_rows[: args.max_candidates],
    }
    write_json(args.out_json, payload)
    status_payload.update(
        {
            "status": "ok",
            "basis": "intraday",
            "basis_label": "장중 기준",
            "priority_count": len(priority_rows),
            "caution_count": len(caution_rows),
            "output_json": str(_resolve(args.out_json)),
        }
    )
    write_json(args.status_json, status_payload)
    upsert_json_payload(
        "intraday_recommendations",
        payload,
        asof_date=payload.get("asof_date") or payload.get("session_date"),
        generated_at=payload.get("generated_at"),
        source_path=args.out_json,
    )
    print("[DONE] intraday refresh completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
