from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from apply_execution_policy import (
    DEFAULT_CANDIDATES_CSV,
    DEFAULT_GATE_JSON,
    DEFAULT_SNAPSHOT_ARCHIVE_CSV,
    GATE_VERSION,
    POLICY_VERSION,
    PORTFOLIO_VERSION,
    SCORE_FORMULA_VERSION,
    _fmt_pct,
    _markdown_table,
    _safe_read_json,
    build_execution_actions,
    build_reference_maps,
    classify_holdings,
    enrich_candidates_with_snapshot,
    extract_cooldown_map,
    gate_decision,
    load_candidates,
    load_holdings,
    load_snapshot_archive,
    markdown_summary_list,
    select_buy_ready_queue,
)
from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

OUT_JSON = OUTPUT_DIR / "trade_intents.json"
OUT_MD = OUTPUT_DIR / "trade_intents.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized trade intents from execution policy outputs.")
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--snapshot-archive-csv", type=Path, default=DEFAULT_SNAPSHOT_ARCHIVE_CSV)
    parser.add_argument("--holdings-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--max-holdings", type=int, default=int(get_production_config_value(["execution_policy", "max_holdings"], 5)))
    parser.add_argument("--max-position-weight", type=float, default=float(get_production_config_value(["execution_policy", "max_position_weight"], get_production_config_value(["portfolio", "max_weight_top5"], 0.24))))
    parser.add_argument("--sector-cap", type=float, default=float(get_production_config_value(["execution_policy", "sector_cap"], get_production_config_value(["portfolio", "sector_cap"], 0.35))))
    parser.add_argument("--theme-cap", type=float, default=float(get_production_config_value(["execution_policy", "theme_cap"], get_production_config_value(["portfolio", "theme_cap"], 0.35))))
    parser.add_argument("--cash-minimum", type=float, default=float(get_production_config_value(["execution_policy", "cash_minimum"], get_production_config_value(["portfolio", "cash_buffer"], 0.05))))
    parser.add_argument("--min-entry-confidence", type=float, default=float(get_production_config_value(["execution_policy", "min_entry_confidence"], get_production_config_value(["buy_candidate", "min_confidence"], 80.0))))
    parser.add_argument("--min-hold-confidence", type=float, default=float(get_production_config_value(["execution_policy", "min_hold_confidence"], 76.0)))
    parser.add_argument("--force-exit-confidence", type=float, default=float(get_production_config_value(["execution_policy", "force_exit_confidence"], 72.0)))
    parser.add_argument("--confidence-block-below", type=float, default=float(get_production_config_value(["execution_policy", "confidence_block_below"], 55.0)))
    parser.add_argument("--confidence-reduced-below", type=float, default=float(get_production_config_value(["execution_policy", "confidence_reduced_below"], 70.0)))
    parser.add_argument("--confidence-standard-below", type=float, default=float(get_production_config_value(["execution_policy", "confidence_standard_below"], 85.0)))
    parser.add_argument("--confidence-reduced-weight-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_reduced_weight_scale"], 0.45)))
    parser.add_argument("--confidence-standard-weight-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_standard_weight_scale"], 1.00)))
    parser.add_argument("--confidence-expanded-weight-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_expanded_weight_scale"], 1.15)))
    parser.add_argument("--confidence-reduced-position-cap-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_reduced_position_cap_scale"], 0.50)))
    parser.add_argument("--confidence-standard-position-cap-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_standard_position_cap_scale"], 1.00)))
    parser.add_argument("--confidence-expanded-position-cap-scale", type=float, default=float(get_production_config_value(["execution_policy", "confidence_expanded_position_cap_scale"], 1.15)))
    parser.add_argument("--max-hold-rank", type=int, default=int(get_production_config_value(["execution_policy", "max_hold_rank"], 8)))
    parser.add_argument("--min-replace-score-gap", type=float, default=float(get_production_config_value(["execution_policy", "min_replace_score_gap"], 3.0)))
    parser.add_argument("--max-replacements-per-cycle", type=int, default=int(get_production_config_value(["execution_policy", "max_replacements_per_cycle"], 2)))
    parser.add_argument("--reentry-cooldown-days", type=int, default=int(get_production_config_value(["execution_policy", "reentry_cooldown_days"], 10)))
    parser.add_argument("--min-liquidity-score", type=float, default=float(get_production_config_value(["buy_candidate", "min_liquidity_score"], 15.0)))
    parser.add_argument("--min-trading-value", type=float, default=float(get_production_config_value(["buy_candidate", "min_trading_value"], 5_000_000_000.0)))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def aggregate_open_positions_by_code(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    for col in ["weight", "confidence_score", "final_score"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    aggregated = (
        work.groupby("code", as_index=False)
        .agg(
            name=("name", "first"),
            weight=("weight", "sum"),
            sector=("sector", "first"),
            dominant_theme=("dominant_theme", "first"),
            confidence_score=("confidence_score", "max"),
            final_score=("final_score", "max"),
        )
    )
    return aggregated


def map_action_to_intent(action: str) -> tuple[str, bool, int]:
    normalized = str(action or "").upper()
    if normalized == "ENTER":
        return "BUY", True, 80
    if normalized == "TRIM":
        return "TRIM", True, 75
    if normalized in {"EXIT_CANDIDATE", "REPLACE_CANDIDATE"}:
        return "EXIT", True, 85
    if normalized == "HOLD_REVIEW":
        return "REVIEW", False, 55
    if normalized == "HOLD":
        return "HOLD", False, 40
    if normalized == "NO_ACTION":
        return "NO_ACTION", False, 0
    return normalized or "UNKNOWN", False, 10


def build_intent_rows(execution_actions: pd.DataFrame, *, asof_date: pd.Timestamp, gate_status: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in execution_actions.iterrows():
        action = str(row.get("action") or "")
        intent_type, executable, base_priority = map_action_to_intent(action)
        code = str(row.get("code") or "").strip()
        reason = str(row.get("reason") or "").strip()
        target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
        if intent_type == "NO_ACTION":
            executable = False
        rows.append(
            {
                "intent_id": f"{asof_date.strftime('%Y%m%d')}:{intent_type}:{code or '-'}",
                "asof_date": asof_date.strftime("%Y-%m-%d"),
                "code": code or None,
                "name": str(row.get("name") or "").strip() or None,
                "source_action": action or None,
                "intent_type": intent_type,
                "target_weight": float(target_weight) if pd.notna(target_weight) else None,
                "gate_status": gate_status or None,
                "reason": reason or None,
                "priority": base_priority,
                "executable": bool(executable),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.candidates_csv)
    snapshot_archive = load_snapshot_archive(args.snapshot_archive_csv)
    latest_snapshot, previous_snapshot = build_reference_maps(snapshot_archive)
    gate_payload = _safe_read_json(args.gate_json)
    gate = gate_decision(gate_payload)

    candidate_asof_raw = str(candidates["asof_date"].iloc[0]) if "asof_date" in candidates.columns and not candidates.empty else ""
    candidate_asof = pd.to_datetime(candidate_asof_raw, errors="coerce")
    latest_snapshot_date = latest_snapshot["asof_date"].iloc[0] if not latest_snapshot.empty else pd.NaT
    asof_date = pd.Timestamp(candidate_asof if pd.notna(candidate_asof) else latest_snapshot_date if pd.notna(latest_snapshot_date) else pd.Timestamp.today()).normalize()

    holdings_context = load_holdings(args, candidates, latest_snapshot)
    if "paper_trading_positions.csv" in str(holdings_context.source):
        holdings_context.open_positions = aggregate_open_positions_by_code(holdings_context.open_positions)
    cooldown_map = extract_cooldown_map(holdings_context.closed_positions, asof_date, args.reentry_cooldown_days)
    candidates = enrich_candidates_with_snapshot(candidates, latest_snapshot, previous_snapshot)
    holdings_review = classify_holdings(holdings_context.open_positions, candidates, latest_snapshot, gate, args)
    held_codes = set(holdings_context.open_positions["code"].astype(str).str.zfill(6).tolist()) if not holdings_context.open_positions.empty else set()
    buy_ready_queue = select_buy_ready_queue(candidates, held_codes, cooldown_map, args)
    execution_actions = build_execution_actions(holdings_review, buy_ready_queue, gate, args)

    intents = build_intent_rows(execution_actions, asof_date=asof_date, gate_status=str(gate["status"]))
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof_date.strftime("%Y-%m-%d"),
        "policy_version": POLICY_VERSION,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "gate_version": GATE_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "holdings_source": holdings_context.source,
        "gate_status": gate["status"],
        "gate_guidance": gate["guidance"],
        "intent_count": len(intents),
        "intents": intents,
    }

    intents_df = pd.DataFrame(intents)
    md_lines = [
        "# Trade Intents",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- asof_date: {payload['asof_date']}",
        f"- policy_version: {payload['policy_version']}",
        f"- gate_status: {payload['gate_status']}",
        f"- holdings_source: {payload['holdings_source']}",
        f"- intent_count: {payload['intent_count']}",
        "",
        "## Gate Guidance",
        "",
        f"- {payload['gate_guidance']}",
        "",
        "## Intents",
        "",
        _markdown_table(
            intents_df if not intents_df.empty else pd.DataFrame([{"intent_id": "-", "code": "-", "name": "", "intent_type": "NO_ACTION", "target_weight": None, "gate_status": payload["gate_status"], "priority": 0, "executable": False, "reason": "no intents"}]),
            ["intent_id", "code", "name", "intent_type", "target_weight", "gate_status", "priority", "executable", "reason"],
        ),
        "",
        "## Intent Notes",
        "",
        markdown_summary_list([
            "BUY/TRIM/EXIT intents are executable candidates for a later broker submission step.",
            "HOLD/REVIEW intents are observational and help explain why no immediate order is created.",
            "This file is the bridge between execution policy outputs and future live order submission.",
        ]),
        "",
    ]

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"trade_intents_json: {out_json}")
    print(f"trade_intents_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
