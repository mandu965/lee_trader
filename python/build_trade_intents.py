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
    gate_decision_runtime,
    load_candidates,
    markdown_summary_list,
)
from production_config import get_production_config_value
from strategy_core import evaluate_strategy


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
    parser.add_argument("--candidate-universe-top-n", type=int, default=int(get_production_config_value(["execution_policy", "candidate_universe_top_n"], 20)))
    parser.add_argument("--entry-review-top-n", type=int, default=int(get_production_config_value(["execution_policy", "entry_review_top_n"], 8)))
    parser.add_argument("--entry-review-extended-top-n", type=int, default=int(get_production_config_value(["execution_policy", "entry_review_extended_top_n"], 10)))
    parser.add_argument("--standard-entry-confidence", type=float, default=float(get_production_config_value(["execution_policy", "standard_entry_confidence"], 76.0)))
    parser.add_argument("--extended-entry-confidence", type=float, default=float(get_production_config_value(["execution_policy", "extended_entry_confidence"], 70.0)))
    parser.add_argument("--max-holdings", type=int, default=int(get_production_config_value(["execution_policy", "max_holdings"], 5)))
    parser.add_argument("--max-position-weight", type=float, default=float(get_production_config_value(["execution_policy", "max_position_weight"], get_production_config_value(["portfolio", "max_weight_top5"], 0.24))))
    parser.add_argument("--sector-cap", type=float, default=float(get_production_config_value(["execution_policy", "sector_cap"], get_production_config_value(["portfolio", "sector_cap"], 0.35))))
    parser.add_argument("--theme-cap", type=float, default=float(get_production_config_value(["execution_policy", "theme_cap"], get_production_config_value(["portfolio", "theme_cap"], 0.35))))
    parser.add_argument("--cash-minimum", type=float, default=float(get_production_config_value(["execution_policy", "cash_minimum"], get_production_config_value(["portfolio", "cash_buffer"], 0.05))))
    parser.add_argument("--watch-limited-auto-buy-enabled", action="store_true", default=bool(get_production_config_value(["execution_policy", "watch_limited_auto_buy_enabled"], False)))
    parser.add_argument("--watch-limited-max-entries", type=int, default=int(get_production_config_value(["execution_policy", "watch_limited_max_entries"], 2)))
    parser.add_argument("--watch-limited-total-exposure", type=float, default=float(get_production_config_value(["execution_policy", "watch_limited_total_exposure"], 0.15)))
    parser.add_argument("--watch-limited-position-cap", type=float, default=float(get_production_config_value(["execution_policy", "watch_limited_position_cap"], 0.08)))
    parser.add_argument("--pilot-limited-auto-buy-enabled", action="store_true", default=bool(get_production_config_value(["execution_policy", "pilot_limited_auto_buy_enabled"], False)))
    parser.add_argument("--pilot-limited-max-entries", type=int, default=int(get_production_config_value(["execution_policy", "pilot_limited_max_entries"], 4)))
    parser.add_argument("--pilot-limited-total-exposure", type=float, default=float(get_production_config_value(["execution_policy", "pilot_limited_total_exposure"], 0.30)))
    parser.add_argument("--pilot-limited-position-cap", type=float, default=float(get_production_config_value(["execution_policy", "pilot_limited_position_cap"], 0.12)))
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


RANKING_CONTEXT_COLUMNS = [
    "buy_rank",
    "rank_final",
    "live_rank",
    "final_score",
    "live_score",
    "confidence_score",
    "risk_penalty",
    "ret_score",
    "prob_score",
    "qual_score",
    "tech_score",
    "liquidity_score",
    "safety_score",
    "dominant_theme",
    "score_driver_1",
    "score_driver_2",
    "score_driver_3",
    "risk_factor_1",
    "risk_factor_2",
    "action_note",
]


def _num_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


def _int_or_none(value: object) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if pd.notna(numeric) else None


def build_ranking_context(candidates: pd.DataFrame) -> dict[str, dict[str, object]]:
    if candidates.empty or "code" not in candidates.columns:
        return {}
    work = candidates.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    context: dict[str, dict[str, object]] = {}
    for _, row in work.drop_duplicates("code", keep="first").iterrows():
        code = str(row.get("code") or "").zfill(6)
        if not code or code == "000000":
            continue
        item: dict[str, object] = {}
        for col in RANKING_CONTEXT_COLUMNS:
            if col not in row.index:
                continue
            value = row.get(col)
            if col in {
                "buy_rank",
                "rank_final",
                "live_rank",
            }:
                item[col] = _int_or_none(value)
            elif col in {
                "final_score",
                "live_score",
                "confidence_score",
                "risk_penalty",
                "ret_score",
                "prob_score",
                "qual_score",
                "tech_score",
                "liquidity_score",
                "safety_score",
            }:
                item[col] = _num_or_none(value)
            else:
                item[col] = str(value).strip() if pd.notna(value) and str(value).strip() else None
        rank = item.get("live_rank") or item.get("rank_final") or item.get("buy_rank")
        item["ranking_rank"] = rank
        item["final_score"] = item.get("final_score") if item.get("final_score") is not None else item.get("live_score")
        context[code] = item
    return context


def build_intent_rows(
    execution_actions: pd.DataFrame,
    *,
    asof_date: pd.Timestamp,
    gate_status: str,
    ranking_context: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranking_context = ranking_context or {}
    for _, row in execution_actions.iterrows():
        action = str(row.get("action") or "")
        intent_type, executable, base_priority = map_action_to_intent(action)
        code = str(row.get("code") or "").strip().zfill(6)
        reason = str(row.get("reason") or "").strip()
        target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
        if intent_type == "NO_ACTION":
            executable = False
        ranking = ranking_context.get(code, {})
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
                "ranking_rank": ranking.get("ranking_rank"),
                "buy_rank": ranking.get("buy_rank"),
                "rank_final": ranking.get("rank_final"),
                "live_rank": ranking.get("live_rank"),
                "final_score": ranking.get("final_score"),
                "live_score": ranking.get("live_score"),
                "confidence_score": ranking.get("confidence_score"),
                "risk_penalty": ranking.get("risk_penalty"),
                "ret_score": ranking.get("ret_score"),
                "prob_score": ranking.get("prob_score"),
                "qual_score": ranking.get("qual_score"),
                "tech_score": ranking.get("tech_score"),
                "liquidity_score": ranking.get("liquidity_score"),
                "safety_score": ranking.get("safety_score"),
                "dominant_theme": ranking.get("dominant_theme"),
                "score_driver_1": ranking.get("score_driver_1"),
                "score_driver_2": ranking.get("score_driver_2"),
                "score_driver_3": ranking.get("score_driver_3"),
                "risk_factor_1": ranking.get("risk_factor_1"),
                "risk_factor_2": ranking.get("risk_factor_2"),
                "action_note": ranking.get("action_note"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.candidates_csv)
    candidates = candidates.loc[pd.to_numeric(candidates["buy_rank"], errors="coerce").le(args.candidate_universe_top_n)].copy()
    candidates = candidates.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)
    gate_payload = _safe_read_json(args.gate_json)
    gate = gate_decision_runtime(gate_payload)

    candidate_asof_raw = str(candidates["asof_date"].iloc[0]) if "asof_date" in candidates.columns and not candidates.empty else ""
    candidate_asof = pd.to_datetime(candidate_asof_raw, errors="coerce")
    strategy = evaluate_strategy(
        args=args,
        candidates=candidates,
        snapshot_archive_csv=args.snapshot_archive_csv,
        gate=gate,
        aggregate_paper_holdings=aggregate_open_positions_by_code,
    )
    latest_snapshot_date = strategy.latest_snapshot["asof_date"].iloc[0] if not strategy.latest_snapshot.empty else pd.NaT
    asof_date = pd.Timestamp(candidate_asof if pd.notna(candidate_asof) else latest_snapshot_date if pd.notna(latest_snapshot_date) else pd.Timestamp.today()).normalize()
    ranking_context = build_ranking_context(candidates)
    intents = build_intent_rows(
        strategy.execution_actions,
        asof_date=asof_date,
        gate_status=str(gate["status"]),
        ranking_context=ranking_context,
    )
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof_date.strftime("%Y-%m-%d"),
        "policy_version": POLICY_VERSION,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "gate_version": GATE_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "holdings_source": strategy.holdings_context.source,
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
            ["intent_id", "code", "name", "intent_type", "ranking_rank", "final_score", "confidence_score", "risk_penalty", "target_weight", "gate_status", "priority", "executable", "reason"],
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
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    out_md.write_text("\n".join(md_lines), encoding="utf-8-sig")
    print(f"trade_intents_json: {out_json}")
    print(f"trade_intents_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
