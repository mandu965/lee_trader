from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_RANKING = DATA_DIR / "ranking_final.csv"
INPUT_CONFIDENCE_V2_JSON = DATA_DIR / "confidence_score_v2.json"
INPUT_GATE_JSON = OUTPUT_DIR / "operational_buy_gate.json"
INPUT_ACCEPTANCE_JSON = OUTPUT_DIR / "walkforward_acceptance.json"

OUT_CSV = OUTPUT_DIR / "top20_buyability_report.csv"
OUT_MD = OUTPUT_DIR / "top20_buyability_report.md"
OUT_JSON = OUTPUT_DIR / "top20_buyability_report.json"
OUT_DATA_JSON = DATA_DIR / "top20_buyability_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Top20 buyability report from ranking and operational evidence.")
    parser.add_argument("--ranking-csv", type=Path, default=INPUT_RANKING)
    parser.add_argument("--confidence-v2-json", type=Path, default=INPUT_CONFIDENCE_V2_JSON)
    parser.add_argument("--gate-json", type=Path, default=INPUT_GATE_JSON)
    parser.add_argument("--acceptance-json", type=Path, default=INPUT_ACCEPTANCE_JSON)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--data-json", type=Path, default=OUT_DATA_JSON)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def load_top20(path: Path, top_n: int) -> tuple[pd.DataFrame, str]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"ranking csv not found: {resolved}")
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    if df["date"].isna().all():
        raise ValueError("ranking csv has no usable date column")
    latest_date = df["date"].max()
    latest = df.loc[df["date"].eq(latest_date)].copy()
    score_col = "final_score_v3" if "final_score_v3" in latest.columns else ("final_score_v2" if "final_score_v2" in latest.columns else "final_score")
    latest[score_col] = pd.to_numeric(latest.get(score_col), errors="coerce")
    for col in ["ret_score", "prob_score", "tech_score", "liquidity_score", "trading_value", "ret_5d", "ret_10d", "rsi_14"]:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce")
    top = latest.sort_values([score_col, "code"], ascending=[False, True]).head(top_n).copy()
    top["score_col_used"] = score_col
    return top.reset_index(drop=True), latest_date.strftime("%Y-%m-%d")


def build_confidence_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        lookup[str(item.get("code", "")).zfill(6)] = item
    return lookup


def gate_bucket_status(gate_payload: dict[str, Any]) -> str | None:
    decisions = gate_payload.get("decisions", [])
    if not isinstance(decisions, list):
        return None
    for item in decisions:
        if not isinstance(item, dict):
            continue
        bucket = pd.to_numeric(item.get("bucket"), errors="coerce")
        if pd.notna(bucket) and int(bucket) == 5:
            return str(item.get("status")) if item.get("status") is not None else None
    return None


def walkforward_soft_watch_allowed(acceptance_payload: dict[str, Any]) -> bool:
    status = str(acceptance_payload.get("status") or "").upper()
    reason_codes = {
        str(item).strip()
        for item in (acceptance_payload.get("reason_codes") or [])
        if str(item).strip()
    }
    if status != "REJECTED" or not reason_codes:
        return False

    required_positive = {
        "top20_excess_return_positive",
        "execution_evidence_ok_or_unavailable",
    }
    soft_failure_codes = {
        "ordering_not_stable",
        "drawdown_too_deep",
        "confidence_monotonicity_missing",
    }
    hard_failure_codes = {
        "top20_performance_not_proven",
        "execution_evidence_too_weak",
        "ordering_reference_missing",
    }

    if not required_positive.issubset(reason_codes):
        return False
    if reason_codes & hard_failure_codes:
        return False
    return bool(reason_codes & soft_failure_codes)


def classify_buyability(
    row: pd.Series,
    confidence_lookup: dict[str, dict[str, Any]],
    gate_payload: dict[str, Any],
    acceptance_payload: dict[str, Any],
) -> tuple[str, list[str], list[str], str, str | None, float]:
    code = str(row["code"]).zfill(6)
    conf_item = confidence_lookup.get(code, {})
    confidence_state = str(conf_item.get("confidence_state_v2") or "WEAK")
    raw_conf_v2 = pd.to_numeric(conf_item.get("raw_confidence_v2"), errors="coerce")
    execution_conf = pd.to_numeric(conf_item.get("execution_confidence"), errors="coerce")
    alpha_conf = pd.to_numeric(conf_item.get("alpha_confidence"), errors="coerce")
    gate_status = str(gate_payload.get("overall_status") or "")
    top5_gate_status = str(gate_bucket_status(gate_payload) or gate_status or "")
    acceptance_status = str(acceptance_payload.get("status") or "")
    acceptance_soft_watch = walkforward_soft_watch_allowed(acceptance_payload)

    supporting: list[str] = []
    blocking: list[str] = []
    if pd.to_numeric(row.get("ret_score"), errors="coerce") >= 80:
        supporting.append("high_return_signal")
    if pd.to_numeric(row.get("prob_score"), errors="coerce") >= 80:
        supporting.append("high_top20_probability")
    if confidence_state in {"TRUSTED", "PROVISIONAL"}:
        supporting.append(f"confidence_{confidence_state.lower()}")
    if execution_conf is not None and pd.notna(execution_conf) and float(execution_conf) >= 60:
        supporting.append("execution_quality_supported")

    liquidity_score = pd.to_numeric(row.get("liquidity_score"), errors="coerce")
    trading_value = pd.to_numeric(row.get("trading_value"), errors="coerce")
    ret_5d = pd.to_numeric(row.get("ret_5d"), errors="coerce")
    ret_10d = pd.to_numeric(row.get("ret_10d"), errors="coerce")
    rsi = pd.to_numeric(row.get("rsi_14"), errors="coerce")
    if pd.notna(liquidity_score) and float(liquidity_score) < 10:
        blocking.append("very_low_liquidity")
    if pd.notna(trading_value) and float(trading_value) < 2_500_000_000.0:
        blocking.append("insufficient_trading_value")
    if pd.notna(ret_5d) and float(ret_5d) >= 0.12:
        blocking.append("ret_5d_overheat")
    if pd.notna(ret_10d) and float(ret_10d) >= 0.20:
        blocking.append("ret_10d_overheat")
    if pd.notna(rsi) and float(rsi) >= 80:
        blocking.append("rsi_overheat")
    if confidence_state == "BLOCKED":
        blocking.append("confidence_blocked")
    elif confidence_state == "WEAK":
        blocking.append("confidence_weak")
    if acceptance_status == "REJECTED" and not acceptance_soft_watch:
        blocking.append("walkforward_rejected")
    elif acceptance_status == "REJECTED" and acceptance_soft_watch:
        blocking.append("walkforward_soft_rejected")
    elif acceptance_status == "CONDITIONAL":
        blocking.append("walkforward_conditional")
    if top5_gate_status in {"BLOCK", "HOLD"}:
        blocking.append(f"gate_{top5_gate_status.lower()}")

    status = "WATCHLIST"
    expected_action = "monitor_only"
    watchlist_tier: str | None = None
    auto_promotion_gate_open = top5_gate_status in {"WATCH", "BUY_ALLOWED"}
    auto_promotion_acceptance_open = acceptance_status == "ACCEPTED"

    if any(reason in blocking for reason in ["very_low_liquidity", "insufficient_trading_value", "confidence_blocked", "walkforward_rejected"]):
        status = "BLOCK"
        expected_action = "do_not_buy"
    elif confidence_state == "TRUSTED" and auto_promotion_gate_open and auto_promotion_acceptance_open:
        status = "BUY_NOW"
        expected_action = "eligible_for_buy"
    elif confidence_state in {"TRUSTED", "PROVISIONAL"} and raw_conf_v2 is not None and pd.notna(raw_conf_v2) and float(raw_conf_v2) >= 64:
        status = "WATCHLIST"
        expected_action = "monitor_for_entry"
        if top5_gate_status == "HOLD" or acceptance_status == "CONDITIONAL":
            expected_action = "watch_only_until_gate_upgrade"
    else:
        status = "PAPER_ONLY" if confidence_state in {"WEAK"} else "WATCHLIST"
        expected_action = "paper_trade_only" if status == "PAPER_ONLY" else "monitor_only"

    promotion_readiness = 0.0
    if raw_conf_v2 is not None and pd.notna(raw_conf_v2):
        promotion_readiness += float(raw_conf_v2) * 0.45
    if execution_conf is not None and pd.notna(execution_conf):
        promotion_readiness += float(execution_conf) * 0.25
    if alpha_conf is not None and pd.notna(alpha_conf):
        promotion_readiness += float(alpha_conf) * 0.20
    if confidence_state == "TRUSTED":
        promotion_readiness += 10.0
    elif confidence_state == "PROVISIONAL":
        promotion_readiness += 5.0
    if "walkforward_conditional" in blocking:
        promotion_readiness -= 5.0
    if "walkforward_soft_rejected" in blocking:
        promotion_readiness -= 7.5
    if "gate_hold" in blocking:
        promotion_readiness -= 5.0
    if any(reason in blocking for reason in ["ret_5d_overheat", "ret_10d_overheat", "rsi_overheat"]):
        promotion_readiness -= 10.0
    promotion_readiness = float(max(0.0, min(100.0, promotion_readiness)))

    if status == "WATCHLIST":
        if promotion_readiness >= 72.0 and confidence_state in {"TRUSTED", "PROVISIONAL"}:
            watchlist_tier = "PROMOTION_READY"
            expected_action = "prioritize_after_gate_upgrade"
        else:
            watchlist_tier = "MONITOR"
            expected_action = "monitor_only" if expected_action == "monitor_for_entry" else expected_action
        if "walkforward_soft_rejected" in blocking:
            expected_action = "watch_only_until_walkforward_upgrade"

    if (
        status == "WATCHLIST"
        and watchlist_tier == "PROMOTION_READY"
        and auto_promotion_gate_open
        and auto_promotion_acceptance_open
        and confidence_state == "TRUSTED"
    ):
        status = "BUY_NOW"
        watchlist_tier = None
        expected_action = "auto_promoted_to_buy"
        blocking = [reason for reason in blocking if reason not in {"walkforward_conditional", "gate_hold"}]

    return status, supporting[:3], blocking[:4], expected_action, watchlist_tier, promotion_readiness


def main() -> int:
    args = parse_args()
    top, asof_date = load_top20(args.ranking_csv, args.top_n)
    confidence_payload = _read_json(args.confidence_v2_json)
    gate_payload = _read_json(args.gate_json)
    acceptance_payload = _read_json(args.acceptance_json)
    confidence_lookup = build_confidence_lookup(confidence_payload)

    records: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        status, supporting, blocking, expected_action, watchlist_tier, promotion_readiness = classify_buyability(row, confidence_lookup, gate_payload, acceptance_payload)
        conf_item = confidence_lookup.get(str(row["code"]).zfill(6), {})
        records.append(
            {
                "code": str(row["code"]).zfill(6),
                "name": row.get("name"),
                "score": pd.to_numeric(row.get(row["score_col_used"]), errors="coerce"),
                "final_score": pd.to_numeric(row.get("final_score"), errors="coerce"),
                "raw_confidence_v2": pd.to_numeric(conf_item.get("raw_confidence_v2"), errors="coerce"),
                "confidence_state_v2": conf_item.get("confidence_state_v2"),
                "buyability_status": status,
                "watchlist_tier": watchlist_tier,
                "promotion_readiness_score": promotion_readiness,
                "expected_action": expected_action,
                "supporting_reasons": supporting,
                "blocking_reasons": blocking,
                "liquidity_score": pd.to_numeric(row.get("liquidity_score"), errors="coerce"),
                "trading_value": pd.to_numeric(row.get("trading_value"), errors="coerce"),
                "ret_5d": pd.to_numeric(row.get("ret_5d"), errors="coerce"),
                "ret_10d": pd.to_numeric(row.get("ret_10d"), errors="coerce"),
                "rsi_14": pd.to_numeric(row.get("rsi_14"), errors="coerce"),
            }
        )

    report = pd.DataFrame(records)
    counts = report["buyability_status"].value_counts().to_dict() if not report.empty else {}
    watchlist_tiers = report["watchlist_tier"].fillna("(none)").value_counts().to_dict() if not report.empty else {}
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof_date,
        "summary": {
            "top_n": int(len(report)),
            "buyability_counts": counts,
            "watchlist_tier_counts": watchlist_tiers,
            "gate_overall_status": gate_payload.get("overall_status"),
            "walkforward_acceptance_status": acceptance_payload.get("status"),
        },
        "items": records,
    }

    out_csv = _resolve(args.out_csv)
    out_md = _resolve(args.out_md)
    out_json = _resolve(args.out_json)
    data_json = _resolve(args.data_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data_json.parent.mkdir(parents=True, exist_ok=True)

    report.assign(
        supporting_reasons=report["supporting_reasons"].map(lambda x: " / ".join(x)),
        blocking_reasons=report["blocking_reasons"].map(lambda x: " / ".join(x)),
    ).to_csv(out_csv, index=False, encoding="utf-8-sig")

    count_rows = [[key, value] for key, value in counts.items()]
    tier_rows = [[key, value] for key, value in watchlist_tiers.items()]
    detail_rows = [
        [
            item["code"],
            item["name"],
            item["buyability_status"],
            item["watchlist_tier"] or "-",
            _fmt_num(item["final_score"]),
            _fmt_num(item["raw_confidence_v2"]),
            _fmt_num(item["promotion_readiness_score"]),
            item["confidence_state_v2"],
            " / ".join(item["supporting_reasons"]) or "-",
            " / ".join(item["blocking_reasons"]) or "-",
        ]
        for item in records
    ]
    md = "\n".join(
        [
            "# Top20 Buyability Report",
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- asof_date: {asof_date}",
            f"- gate_overall_status: {payload['summary']['gate_overall_status']}",
            f"- walkforward_acceptance_status: {payload['summary']['walkforward_acceptance_status']}",
            "",
            "## Status Counts",
            "",
            _markdown_table(count_rows or [["(none)", 0]], ["status", "count"]),
            "",
            "## Watchlist Tiers",
            "",
            _markdown_table(tier_rows or [["(none)", 0]], ["tier", "count"]),
            "",
            "## Top20 Detail",
            "",
            _markdown_table(
                detail_rows or [["NA", "NA", "NA", "-", "NA", "NA", "NA", "NA", "-", "-"]],
                ["code", "name", "status", "tier", "final_score", "raw_conf_v2", "promotion", "confidence_state_v2", "supporting", "blocking"],
            ),
            "",
        ]
    )
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    data_json.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"asof_date: {asof_date}")
    print(f"out_csv: {out_csv}")
    print(f"out_md: {out_md}")
    print(f"out_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
