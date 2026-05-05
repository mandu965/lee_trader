from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_AI_PREVIEW = OUTPUT_DIR / "order_requests_preview.json"
DEFAULT_RULE_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_HOLDINGS_CSV = DATA_DIR / "live_account_holdings.csv"
DEFAULT_BALANCE_JSON = OUTPUT_DIR / "live_account_balance_summary.json"
DEFAULT_FILLS_JSON = OUTPUT_DIR / "live_order_fills.json"
DEFAULT_MARKET_STATUS_CSV = DATA_DIR / "market_status.csv"
DEFAULT_COMMON_GUARD_JSON = OUTPUT_DIR / "common_live_risk_guard.json"

DEFAULT_APPROVED_JSON = OUTPUT_DIR / "master_approved_orders.json"
DEFAULT_BLOCKED_JSON = OUTPUT_DIR / "master_blocked_orders.json"
DEFAULT_SUMMARY_JSON = OUTPUT_DIR / "master_risk_summary.json"
DEFAULT_SUMMARY_MD = OUTPUT_DIR / "master_risk_summary.md"

SUBMIT_ONLY_BLOCK_REASONS = {
    "paper_mode_no_order_submission",
    "rule_order_submit_disabled",
    "buy_requires_allow_buy",
    "execution blocked: --confirm-text LIVE_ORDER is required",
}


def _resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    raw = resolved.read_text(encoding="utf-8-sig")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    value = json.loads(normalized)
    return value if isinstance(value, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    resolved = _resolve(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    resolved = _resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text else ""


def _text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _is_entry_gate_blocked(status: Any, reason: Any) -> bool:
    status_text = str(status or "").strip().lower()
    reason_text = str(reason or "").strip().lower()
    if not status_text and not reason_text:
        return False
    blocked_tokens = {"live_price_unavailable", "entry_gap_up_hard_blocked", "entry_gap_up_blocked", "entry_gap_down_blocked"}
    return status_text in blocked_tokens or reason_text in blocked_tokens or "blocked" in status_text


def _first_reason(*values: Any) -> str | None:
    for value in values:
        text = _text(value)
        if text:
            return text
    return None


def _sum_today_buy_amount(fills_payload: dict[str, Any], as_of_date: str) -> float:
    total = 0.0
    for item in fills_payload.get("items") or []:
        if str(item.get("side") or "").upper() != "BUY":
            continue
        item_date = str(item.get("as_of_date") or "")[:10]
        if item_date != as_of_date:
            continue
        total += _num(item.get("filled_amount")) or 0.0
    return total


def _holdings_by_code(rows: list[dict[str, Any]]) -> set[str]:
    codes: set[str] = set()
    for row in rows:
        code = _normalize_code(row.get("code"))
        if code:
            codes.add(code)
    return codes


@dataclass
class Candidate:
    source: str
    engine_type: str
    request_id: str
    code: str
    name: str | None
    side: str
    order_amount: float
    qty: float
    price: float | None
    priority: float
    score: float
    sector: str | None
    theme: str | None
    common_risk_allowed: bool | None
    common_risk_block_reasons: list[str]
    entry_gate_status: str | None
    entry_gate_reason: str | None
    source_block_reason: str | None
    raw: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "engine_type": self.engine_type,
            "request_id": self.request_id,
            "code": self.code,
            "name": self.name,
            "side": self.side,
            "order_amount": self.order_amount,
            "order_qty": self.qty,
            "reference_price": self.price,
            "priority": self.priority,
            "score": self.score,
            "sector": self.sector,
            "theme": self.theme,
            "common_risk_allowed": self.common_risk_allowed,
            "common_risk_block_reasons": self.common_risk_block_reasons,
            "entry_gate_status": self.entry_gate_status,
            "entry_gate_reason": self.entry_gate_reason,
            "source_block_reason": self.source_block_reason,
            "source_preview": self.raw,
        }


def _build_ai_candidates(preview_payload: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for item in preview_payload.get("items") or []:
        if str(item.get("side") or "").upper() != "BUY":
            continue
        code = _normalize_code(item.get("code"))
        qty = _num(item.get("final_request_qty")) or _num(item.get("allowed_qty")) or _num(item.get("planned_qty")) or 0.0
        price = _num(item.get("live_price")) or _num(item.get("reference_price")) or _num(item.get("previous_close"))
        order_amount = _num(item.get("order_amount"))
        if order_amount is None and qty > 0 and price is not None:
            order_amount = qty * price
        if not code or order_amount is None or order_amount <= 0:
            continue
        common_reasons = item.get("common_risk_block_reasons") or []
        if not isinstance(common_reasons, list):
            common_reasons = [_text(common_reasons)] if _text(common_reasons) else []
        out.append(
            Candidate(
                source="ai",
                engine_type="ai_model",
                request_id=str(item.get("request_id") or item.get("intent_id") or f"AI:{code}"),
                code=code,
                name=_text(item.get("name")),
                side="BUY",
                order_amount=float(order_amount),
                qty=float(qty),
                price=price,
                priority=float(_num(item.get("priority")) or 0.0),
                score=float(_num(item.get("live_score")) or _num(item.get("final_score")) or 0.0),
                sector=_text(item.get("sector")),
                theme=_text(item.get("dominant_theme")),
                common_risk_allowed=_bool(item.get("common_risk_allowed")),
                common_risk_block_reasons=[str(x) for x in common_reasons if _text(x)],
                entry_gate_status=_text(item.get("entry_price_gate_status")),
                entry_gate_reason=_text(item.get("entry_price_gate_reason")),
                source_block_reason=_text(item.get("blocked_reason")),
                raw=deepcopy(item),
            )
        )
    return out


def _build_rule_candidates(preview_payload: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for item in preview_payload.get("items") or []:
        if str(item.get("side") or "").upper() != "BUY":
            continue
        code = _normalize_code(item.get("code") or item.get("symbol"))
        qty = _num(item.get("order_qty")) or 0.0
        price = _num(item.get("expected_execution_price")) or _num(item.get("limit_price"))
        order_amount = _num(item.get("order_amount"))
        if order_amount is None and qty > 0 and price is not None:
            order_amount = qty * price
        if not code or order_amount is None or order_amount <= 0:
            continue
        common_reasons = item.get("common_risk_block_reasons") or []
        if not isinstance(common_reasons, list):
            common_reasons = [_text(common_reasons)] if _text(common_reasons) else []
        out.append(
            Candidate(
                source="rule",
                engine_type=str(item.get("engine_type") or "rule_based"),
                request_id=str(item.get("order_id") or f"RULE:{code}"),
                code=code,
                name=_text(item.get("name")),
                side="BUY",
                order_amount=float(order_amount),
                qty=float(qty),
                price=price,
                priority=float(_num(item.get("priority")) or _num(item.get("rule_score_v2")) or _num(item.get("rule_score")) or 0.0),
                score=float(_num(item.get("rule_score_v2")) or _num(item.get("rule_score")) or 0.0),
                sector=_text(item.get("sector")),
                theme=_text(item.get("dominant_theme")),
                common_risk_allowed=_bool(item.get("common_risk_allowed")),
                common_risk_block_reasons=[str(x) for x in common_reasons if _text(x)],
                entry_gate_status=_text(item.get("entry_price_gate_status")),
                entry_gate_reason=_text(item.get("entry_price_gate_reason")),
                source_block_reason=_text(item.get("order_block_reason")),
                raw=deepcopy(item),
            )
        )
    return out


def _block_record(candidate: Candidate, reason: str, detail: str | None = None) -> dict[str, Any]:
    record = candidate.to_record()
    record["master_decision"] = "blocked"
    record["master_block_reason"] = reason
    record["master_block_detail"] = detail
    return record


def _approved_record(candidate: Candidate, *, projected_cash_ratio: float | None, engine_spent_after: float, total_spent_after: float) -> dict[str, Any]:
    record = candidate.to_record()
    record["master_decision"] = "approved"
    record["projected_cash_ratio_after"] = projected_cash_ratio
    record["engine_spent_after"] = engine_spent_after
    record["total_spent_after"] = total_spent_after
    return record


def run_master_risk_preview(
    *,
    ai_preview_path: Path = DEFAULT_AI_PREVIEW,
    rule_preview_path: Path = DEFAULT_RULE_PREVIEW,
    holdings_csv_path: Path = DEFAULT_HOLDINGS_CSV,
    balance_json_path: Path = DEFAULT_BALANCE_JSON,
    fills_json_path: Path = DEFAULT_FILLS_JSON,
    market_status_csv_path: Path = DEFAULT_MARKET_STATUS_CSV,
    common_guard_json_path: Path = DEFAULT_COMMON_GUARD_JSON,
    approved_out_path: Path = DEFAULT_APPROVED_JSON,
    blocked_out_path: Path = DEFAULT_BLOCKED_JSON,
    summary_json_path: Path = DEFAULT_SUMMARY_JSON,
    summary_md_path: Path = DEFAULT_SUMMARY_MD,
) -> dict[str, Any]:
    ai_preview = _read_json(ai_preview_path)
    rule_preview = _read_json(rule_preview_path)
    holdings_rows = _read_csv_rows(holdings_csv_path)
    balance_payload = _read_json(balance_json_path)
    fills_payload = _read_json(fills_json_path)
    market_status_rows = _read_csv_rows(market_status_csv_path)
    common_guard_payload = _read_json(common_guard_json_path)

    warnings: list[str] = []
    if not ai_preview:
        warnings.append("ai_preview_missing")
    if not rule_preview:
        warnings.append("rule_preview_missing")
    if not holdings_rows:
        warnings.append("holdings_csv_missing_or_empty")
    if not balance_payload:
        warnings.append("balance_json_missing_or_empty")
    if not fills_payload:
        warnings.append("fills_json_missing_or_empty")
    if not market_status_rows:
        warnings.append("market_status_missing_or_empty")
    if not common_guard_payload:
        warnings.append("common_guard_payload_missing_or_empty")

    ai_candidates = _build_ai_candidates(ai_preview)
    rule_candidates = _build_rule_candidates(rule_preview)
    candidates = ai_candidates + rule_candidates

    as_of_date = str(
        ai_preview.get("asof_date")
        or rule_preview.get("as_of_date")
        or fills_payload.get("end_date")
        or datetime.now().strftime("%Y-%m-%d")
    )[:10]

    engine_budgets = {
        "ai": _float_env("MASTER_RISK_ENGINE_DAILY_BUDGET_AI", 500000.0),
        "rule": _float_env("MASTER_RISK_ENGINE_DAILY_BUDGET_RULE", 500000.0),
    }
    total_daily_budget = _float_env("MASTER_RISK_TOTAL_DAILY_BUY_BUDGET", 1000000.0)
    max_sector_exposure_pct = _float_env("MASTER_RISK_MAX_SECTOR_EXPOSURE_PCT", 0.35)
    max_theme_exposure_pct = _float_env("MASTER_RISK_MAX_THEME_EXPOSURE_PCT", 0.30)
    min_cash_ratio = _float_env("MASTER_RISK_MIN_CASH_RATIO", 0.20)

    cash_ratio = _num(((balance_payload.get("derived_metrics") or {}).get("cash_ratio")))
    cash_amount = _num(((balance_payload.get("derived_metrics") or {}).get("cash_amount")))
    total_assets = _num(((balance_payload.get("derived_metrics") or {}).get("total_assets")))
    if cash_amount is None:
        cash_amount = _num(((balance_payload.get("cash_summary") or {}).get("dnca_tot_amt")))
    if total_assets is None:
        total_assets = _num(((balance_payload.get("cash_summary") or {}).get("tot_evlu_amt")))
    if cash_ratio is None and cash_amount is not None and total_assets not in {None, 0.0}:
        cash_ratio = cash_amount / total_assets

    engine_spent_before = {
        "ai": _sum_today_buy_amount(fills_payload, as_of_date),
        "rule": 0.0,
    }
    total_spent_before = sum(engine_spent_before.values())

    if total_assets in {None, 0.0}:
        warnings.append("total_assets_missing")
    if cash_amount is None:
        warnings.append("cash_amount_missing")
    if cash_ratio is None:
        warnings.append("cash_ratio_missing")

    holdings_codes = _holdings_by_code(holdings_rows)
    current_sector_exposure: dict[str, float] = defaultdict(float)
    current_theme_exposure: dict[str, float] = defaultdict(float)
    for row in holdings_rows:
        sector = _text(row.get("sector"))
        theme = _text(row.get("theme")) or _text(row.get("dominant_theme"))
        weight = _num(row.get("weight")) or 0.0
        if sector:
            current_sector_exposure[sector] += weight
        if theme:
            current_theme_exposure[theme] += weight

    duplicate_code_map: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        duplicate_code_map[candidate.code].add(candidate.source)

    blocked: list[dict[str, Any]] = []
    approved: list[dict[str, Any]] = []
    blocked_reason_counter: Counter[str] = Counter()
    engine_spent = dict(engine_spent_before)
    total_spent = float(total_spent_before)
    planned_sector_exposure: dict[str, float] = defaultdict(float)
    planned_theme_exposure: dict[str, float] = defaultdict(float)

    candidates_sorted = sorted(
        candidates,
        key=lambda item: (-item.priority, -item.score, item.source, item.code),
    )

    for candidate in candidates_sorted:
        if len(duplicate_code_map[candidate.code]) > 1:
            reason = "duplicate_buy_candidate_across_engines"
            blocked.append(_block_record(candidate, reason, "same_code_present_in_ai_and_rule"))
            blocked_reason_counter[reason] += 1
            continue

        if candidate.common_risk_allowed is not True:
            reason = "common_risk_blocked"
            detail = ",".join(candidate.common_risk_block_reasons) if candidate.common_risk_block_reasons else "common_risk_missing_or_false"
            blocked.append(_block_record(candidate, reason, detail))
            blocked_reason_counter[reason] += 1
            continue

        if _is_entry_gate_blocked(candidate.entry_gate_status, candidate.entry_gate_reason):
            reason = "entry_price_gate_blocked"
            blocked.append(_block_record(candidate, reason, candidate.entry_gate_reason or candidate.entry_gate_status))
            blocked_reason_counter[reason] += 1
            continue

        if candidate.source_block_reason and candidate.source_block_reason not in SUBMIT_ONLY_BLOCK_REASONS:
            reason = "source_preview_blocked"
            blocked.append(_block_record(candidate, reason, candidate.source_block_reason))
            blocked_reason_counter[reason] += 1
            continue

        engine_limit = engine_budgets.get(candidate.source, total_daily_budget)
        if engine_spent.get(candidate.source, 0.0) + candidate.order_amount > engine_limit:
            reason = "engine_daily_budget_exceeded"
            blocked.append(_block_record(candidate, reason, f"{candidate.source}:{engine_limit:.0f}"))
            blocked_reason_counter[reason] += 1
            continue

        if total_spent + candidate.order_amount > total_daily_budget:
            reason = "total_daily_buy_budget_exceeded"
            blocked.append(_block_record(candidate, reason, f"{total_daily_budget:.0f}"))
            blocked_reason_counter[reason] += 1
            continue

        if total_assets not in {None, 0.0} and candidate.sector:
            projected_sector = current_sector_exposure.get(candidate.sector, 0.0) + planned_sector_exposure.get(candidate.sector, 0.0) + (candidate.order_amount / float(total_assets))
            if projected_sector > max_sector_exposure_pct:
                reason = "sector_exposure_limit_exceeded"
                blocked.append(_block_record(candidate, reason, f"{candidate.sector}:{projected_sector:.4f}"))
                blocked_reason_counter[reason] += 1
                continue
        elif candidate.sector is None:
            warnings.append(f"sector_missing:{candidate.code}")

        if total_assets not in {None, 0.0} and candidate.theme and candidate.theme != "(none)":
            projected_theme = current_theme_exposure.get(candidate.theme, 0.0) + planned_theme_exposure.get(candidate.theme, 0.0) + (candidate.order_amount / float(total_assets))
            if projected_theme > max_theme_exposure_pct:
                reason = "theme_exposure_limit_exceeded"
                blocked.append(_block_record(candidate, reason, f"{candidate.theme}:{projected_theme:.4f}"))
                blocked_reason_counter[reason] += 1
                continue
        elif candidate.theme in {None, "(none)"}:
            warnings.append(f"theme_missing:{candidate.code}")

        projected_cash_ratio = None
        if cash_amount is not None and total_assets not in {None, 0.0}:
            projected_cash_ratio = (cash_amount - (total_spent + candidate.order_amount - total_spent_before)) / float(total_assets)
            if projected_cash_ratio < min_cash_ratio:
                reason = "cash_ratio_floor_breached"
                blocked.append(_block_record(candidate, reason, f"{projected_cash_ratio:.4f}"))
                blocked_reason_counter[reason] += 1
                continue

        engine_spent[candidate.source] = engine_spent.get(candidate.source, 0.0) + candidate.order_amount
        total_spent += candidate.order_amount
        if total_assets not in {None, 0.0}:
            if candidate.sector:
                planned_sector_exposure[candidate.sector] += candidate.order_amount / float(total_assets)
            if candidate.theme and candidate.theme != "(none)":
                planned_theme_exposure[candidate.theme] += candidate.order_amount / float(total_assets)
        approved.append(
            _approved_record(
                candidate,
                projected_cash_ratio=projected_cash_ratio,
                engine_spent_after=engine_spent[candidate.source],
                total_spent_after=total_spent,
            )
        )

    summary = {
        "generated_at": _now_text(),
        "as_of_date": as_of_date,
        "mode": "preview_only",
        "actual_order_submission_connected": False,
        "input_status": {
            "ai_preview_found": bool(ai_preview),
            "rule_preview_found": bool(rule_preview),
            "holdings_found": bool(holdings_rows),
            "fills_found": bool(fills_payload),
            "market_status_found": bool(market_status_rows),
            "common_guard_found": bool(common_guard_payload),
        },
        "controls": {
            "engine_daily_budget_ai": engine_budgets["ai"],
            "engine_daily_budget_rule": engine_budgets["rule"],
            "total_daily_buy_budget": total_daily_budget,
            "max_sector_exposure_pct": max_sector_exposure_pct,
            "max_theme_exposure_pct": max_theme_exposure_pct,
            "min_cash_ratio": min_cash_ratio,
        },
        "portfolio_snapshot": {
            "cash_amount": cash_amount,
            "cash_ratio": cash_ratio,
            "total_assets": total_assets,
            "holding_count": len(holdings_rows),
            "holding_codes": sorted(holdings_codes),
        },
        "candidate_counts": {
            "ai_buy_candidates": len(ai_candidates),
            "rule_buy_candidates": len(rule_candidates),
            "total_buy_candidates": len(candidates),
            "approved_count": len(approved),
            "blocked_count": len(blocked),
        },
        "budget_usage": {
            "engine_spent_before": engine_spent_before,
            "engine_spent_after": engine_spent,
            "total_spent_before": total_spent_before,
            "total_spent_after": total_spent,
        },
        "exposure_snapshot": {
            "current_sector_exposure": dict(current_sector_exposure),
            "current_theme_exposure": dict(current_theme_exposure),
            "planned_sector_exposure_add": dict(planned_sector_exposure),
            "planned_theme_exposure_add": dict(planned_theme_exposure),
        },
        "blocked_reason_counts": dict(blocked_reason_counter),
        "warnings": sorted(set(warnings)),
    }

    approved_payload = {
        "generated_at": summary["generated_at"],
        "as_of_date": as_of_date,
        "mode": "preview_only",
        "approved_count": len(approved),
        "items": approved,
    }
    blocked_payload = {
        "generated_at": summary["generated_at"],
        "as_of_date": as_of_date,
        "mode": "preview_only",
        "blocked_count": len(blocked),
        "items": blocked,
    }

    md_lines = [
        "# Master Risk Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- as_of_date: {as_of_date}",
        f"- mode: preview_only",
        f"- actual_order_submission_connected: false",
        f"- ai_buy_candidates: {len(ai_candidates)}",
        f"- rule_buy_candidates: {len(rule_candidates)}",
        f"- approved_count: {len(approved)}",
        f"- blocked_count: {len(blocked)}",
        f"- cash_ratio: {cash_ratio if cash_ratio is not None else 'null'}",
        f"- total_assets: {total_assets if total_assets is not None else 'null'}",
        "",
        "## Blocked Reasons",
        "",
    ]
    if blocked_reason_counter:
        for reason, count in blocked_reason_counter.most_common():
            md_lines.append(f"- {reason}: {count}")
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Approved Samples", ""])
    if approved:
        for item in approved[:5]:
            md_lines.append(f"- {item['source']} {item['code']} {item.get('name') or '-'} amount={item['order_amount']}")
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Blocked Samples", ""])
    if blocked:
        for item in blocked[:8]:
            md_lines.append(
                f"- {item['source']} {item['code']} {item.get('name') or '-'} reason={item['master_block_reason']} detail={item.get('master_block_detail') or '-'}"
            )
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        for warning in summary["warnings"]:
            md_lines.append(f"- {warning}")
    else:
        md_lines.append("- none")

    _write_json(approved_out_path, approved_payload)
    _write_json(blocked_out_path, blocked_payload)
    _write_json(summary_json_path, summary)
    _write_text(summary_md_path, "\n".join(md_lines) + "\n")
    return summary


def _write_temp_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_temp_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        holdings_csv = root / "live_account_holdings.csv"
        balance_json = root / "live_account_balance_summary.json"
        fills_json = root / "live_order_fills.json"
        market_csv = root / "market_status.csv"
        common_guard_json = root / "common_live_risk_guard.json"

        _write_temp_csv(
            holdings_csv,
            [{"code": "000001", "name": "Held", "qty": "1", "weight": "0.10", "status": "OPEN"}],
        )
        _write_temp_json(
            balance_json,
            {
                "derived_metrics": {
                    "cash_amount": 800000.0,
                    "cash_ratio": 0.8,
                    "total_assets": 1000000.0,
                }
            },
        )
        _write_temp_json(fills_json, {"items": []})
        _write_temp_csv(
            market_csv,
            [{"date": "2026-04-30", "market_up": "True"}],
        )
        _write_temp_json(common_guard_json, {"available": True})

        def execute_case(name: str, ai_items: list[dict[str, Any]], rule_items: list[dict[str, Any]]) -> dict[str, Any]:
            ai_path = root / f"{name}_ai.json"
            rule_path = root / f"{name}_rule.json"
            approved_path = root / f"{name}_approved.json"
            blocked_path = root / f"{name}_blocked.json"
            summary_json_path = root / f"{name}_summary.json"
            summary_md_path = root / f"{name}_summary.md"
            _write_temp_json(ai_path, {"asof_date": "2026-04-30", "items": ai_items})
            _write_temp_json(rule_path, {"as_of_date": "2026-04-30", "items": rule_items})
            summary = run_master_risk_preview(
                ai_preview_path=ai_path,
                rule_preview_path=rule_path,
                holdings_csv_path=holdings_csv,
                balance_json_path=balance_json,
                fills_json_path=fills_json,
                market_status_csv_path=market_csv,
                common_guard_json_path=common_guard_json,
                approved_out_path=approved_path,
                blocked_out_path=blocked_path,
                summary_json_path=summary_json_path,
                summary_md_path=summary_md_path,
            )
            approved_payload = _read_json(approved_path)
            blocked_payload = _read_json(blocked_path)
            return {
                "name": name,
                "summary": summary,
                "approved": approved_payload.get("items") or [],
                "blocked": blocked_payload.get("items") or [],
                "approved_exists": approved_path.exists(),
                "blocked_exists": blocked_path.exists(),
                "summary_md_exists": summary_md_path.exists(),
            }

        scenarios.append(
            execute_case(
                "ai_only",
                [
                    {
                        "request_id": "AI:BUY:111111",
                        "code": "111111",
                        "name": "AIOnly",
                        "side": "BUY",
                        "final_request_qty": 1,
                        "order_amount": 100000,
                        "priority": 90,
                        "live_score": 80,
                        "common_risk_allowed": True,
                        "common_risk_block_reasons": [],
                        "entry_price_gate_status": "entry_gap_ok",
                    }
                ],
                [],
            )
        )
        scenarios.append(
            execute_case(
                "rule_only",
                [],
                [
                    {
                        "order_id": "RULE:BUY:222222",
                        "code": "222222",
                        "name": "RuleOnly",
                        "side": "BUY",
                        "order_qty": 1,
                        "order_amount": 100000,
                        "priority": 70,
                        "rule_score_v2": 88,
                        "common_risk_allowed": True,
                        "common_risk_block_reasons": [],
                    }
                ],
            )
        )
        scenarios.append(
            execute_case(
                "duplicate_cross_engine",
                [
                    {
                        "request_id": "AI:BUY:333333",
                        "code": "333333",
                        "name": "DupAI",
                        "side": "BUY",
                        "final_request_qty": 1,
                        "order_amount": 100000,
                        "priority": 95,
                        "live_score": 90,
                        "common_risk_allowed": True,
                        "common_risk_block_reasons": [],
                        "entry_price_gate_status": "entry_gap_ok",
                    }
                ],
                [
                    {
                        "order_id": "RULE:BUY:333333",
                        "code": "333333",
                        "name": "DupRule",
                        "side": "BUY",
                        "order_qty": 1,
                        "order_amount": 120000,
                        "priority": 85,
                        "rule_score_v2": 77,
                        "common_risk_allowed": True,
                        "common_risk_block_reasons": [],
                    }
                ],
            )
        )
        scenarios.append(
            execute_case(
                "common_risk_blocked",
                [
                    {
                        "request_id": "AI:BUY:444444",
                        "code": "444444",
                        "name": "GuardBlocked",
                        "side": "BUY",
                        "final_request_qty": 1,
                        "order_amount": 100000,
                        "priority": 80,
                        "live_score": 75,
                        "common_risk_allowed": False,
                        "common_risk_block_reasons": ["global_kill_switch_on"],
                        "entry_price_gate_status": "entry_gap_ok",
                    }
                ],
                [],
            )
        )
        scenarios.append(
            execute_case(
                "entry_gate_blocked",
                [
                    {
                        "request_id": "AI:BUY:555555",
                        "code": "555555",
                        "name": "EntryBlocked",
                        "side": "BUY",
                        "final_request_qty": 1,
                        "order_amount": 100000,
                        "priority": 80,
                        "live_score": 75,
                        "common_risk_allowed": True,
                        "common_risk_block_reasons": [],
                        "entry_price_gate_status": "entry_gap_up_blocked",
                        "entry_price_gate_reason": "entry_gap_up_blocked",
                    }
                ],
                [],
            )
        )

    passed = True
    checks: list[dict[str, Any]] = []
    scenario_map = {item["name"]: item for item in scenarios}
    checks.append({"name": "ai_preview_only", "passed": len(scenario_map["ai_only"]["approved"]) == 1})
    checks.append({"name": "rule_preview_only", "passed": len(scenario_map["rule_only"]["approved"]) == 1})
    checks.append(
        {
            "name": "duplicate_cross_engine_blocked",
            "passed": all(item.get("master_block_reason") == "duplicate_buy_candidate_across_engines" for item in scenario_map["duplicate_cross_engine"]["blocked"]),
        }
    )
    checks.append(
        {
            "name": "common_risk_blocked",
            "passed": scenario_map["common_risk_blocked"]["blocked"][0].get("master_block_reason") == "common_risk_blocked",
        }
    )
    checks.append(
        {
            "name": "entry_gate_blocked",
            "passed": scenario_map["entry_gate_blocked"]["blocked"][0].get("master_block_reason") == "entry_price_gate_blocked",
        }
    )
    checks.append(
        {
            "name": "output_files_created",
            "passed": all(item["approved_exists"] and item["blocked_exists"] and item["summary_md_exists"] for item in scenarios),
        }
    )
    for check in checks:
        passed = passed and bool(check["passed"])

    return {
        "generated_at": _now_text(),
        "scenario_count": len(scenarios),
        "passed": passed,
        "checks": checks,
        "scenarios": [
            {
                "name": item["name"],
                "approved_count": len(item["approved"]),
                "blocked_count": len(item["blocked"]),
                "approved_sample": item["approved"][:2],
                "blocked_sample": item["blocked"][:2],
            }
            for item in scenarios
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview-only master risk approval for AI/RULE orders.")
    parser.add_argument("--ai-preview", default=str(DEFAULT_AI_PREVIEW))
    parser.add_argument("--rule-preview", default=str(DEFAULT_RULE_PREVIEW))
    parser.add_argument("--holdings-csv", default=str(DEFAULT_HOLDINGS_CSV))
    parser.add_argument("--balance-json", default=str(DEFAULT_BALANCE_JSON))
    parser.add_argument("--fills-json", default=str(DEFAULT_FILLS_JSON))
    parser.add_argument("--market-status-csv", default=str(DEFAULT_MARKET_STATUS_CSV))
    parser.add_argument("--common-guard-json", default=str(DEFAULT_COMMON_GUARD_JSON))
    parser.add_argument("--approved-out", default=str(DEFAULT_APPROVED_JSON))
    parser.add_argument("--blocked-out", default=str(DEFAULT_BLOCKED_JSON))
    parser.add_argument("--summary-json-out", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--summary-md-out", default=str(DEFAULT_SUMMARY_MD))
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        payload = run_self_test()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("passed") else 1

    summary = run_master_risk_preview(
        ai_preview_path=Path(args.ai_preview),
        rule_preview_path=Path(args.rule_preview),
        holdings_csv_path=Path(args.holdings_csv),
        balance_json_path=Path(args.balance_json),
        fills_json_path=Path(args.fills_json),
        market_status_csv_path=Path(args.market_status_csv),
        common_guard_json_path=Path(args.common_guard_json),
        approved_out_path=Path(args.approved_out),
        blocked_out_path=Path(args.blocked_out),
        summary_json_path=Path(args.summary_json_out),
        summary_md_path=Path(args.summary_md_out),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
