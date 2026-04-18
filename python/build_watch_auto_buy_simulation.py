from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from apply_execution_policy import (
    DEFAULT_CANDIDATES_CSV,
    DEFAULT_SNAPSHOT_ARCHIVE_CSV,
    POLICY_VERSION,
    SCORE_FORMULA_VERSION,
    GATE_VERSION,
    PORTFOLIO_VERSION,
    _markdown_table,
    build_execution_actions,
    build_reference_maps,
    classify_holdings,
    enrich_candidates_with_snapshot,
    extract_cooldown_map,
    load_candidates,
    load_holdings,
    load_snapshot_archive,
    parse_args as parse_policy_args,
    select_buy_ready_queue,
)
from payload_store import upsert_json_payload


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

OUT_JSON = OUTPUT_DIR / "watch_auto_buy_simulation.json"
OUT_MD = OUTPUT_DIR / "watch_auto_buy_simulation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WATCH-mode limited auto-buy simulation payloads.")
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--snapshot-archive-csv", type=Path, default=DEFAULT_SNAPSHOT_ARCHIVE_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _watch_gate() -> dict[str, object]:
    return {
        "status": "WATCH",
        "reason": "WATCH simulation payload",
        "allow_new_entries": False,
        "allow_score_replacement": False,
        "allow_risk_trim": True,
        "guidance": "WATCH simulation: limited new entry allowed for review only",
    }


def _empty_holdings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["code", "name", "weight", "sector", "dominant_theme", "confidence_score", "final_score"]
    )


def _build_scenario_payload(
    *,
    scenario_key: str,
    scenario_label: str,
    holdings: pd.DataFrame,
    holdings_source: str,
    candidates: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    previous_snapshot: pd.DataFrame,
    policy_args: argparse.Namespace,
    asof_date: pd.Timestamp,
) -> dict[str, object]:
    gate = _watch_gate()
    cooldown_map = extract_cooldown_map(pd.DataFrame(columns=["code", "name", "exit_date"]), asof_date, policy_args.reentry_cooldown_days)
    enriched = enrich_candidates_with_snapshot(candidates, latest_snapshot, previous_snapshot)
    holdings_review = classify_holdings(holdings, enriched, latest_snapshot, gate, policy_args)
    held_codes = set(holdings["code"].astype(str).str.zfill(6).tolist()) if not holdings.empty else set()
    buy_ready_queue = select_buy_ready_queue(enriched, held_codes, cooldown_map, policy_args)
    execution_actions = build_execution_actions(holdings_review, buy_ready_queue, gate, policy_args)

    actionable = execution_actions.loc[execution_actions["action"].astype(str).eq("ENTER")].copy()
    actionable["target_weight"] = pd.to_numeric(actionable["target_weight"], errors="coerce")
    queue_rows = buy_ready_queue.copy()
    for col in ["buy_rank", "final_score", "confidence_score", "liquidity_score", "trading_value", "target_weight"]:
        if col in queue_rows.columns:
            queue_rows[col] = pd.to_numeric(queue_rows[col], errors="coerce")

    return {
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "gate_status": "WATCH",
        "holdings_source": holdings_source,
        "holding_count": int(len(holdings)),
        "summary": {
            "enter_count": int(len(actionable)),
            "enter_codes": actionable["code"].astype(str).tolist(),
            "total_target_weight": float(actionable["target_weight"].fillna(0).sum()) if not actionable.empty else 0.0,
            "max_single_target_weight": float(actionable["target_weight"].fillna(0).max()) if not actionable.empty else 0.0,
        },
        "actions": execution_actions.fillna(value=pd.NA).to_dict(orient="records"),
        "buy_ready_queue": queue_rows.fillna(value=pd.NA).to_dict(orient="records"),
    }


def main() -> int:
    args = parse_args()
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        policy_args = parse_policy_args()
    finally:
        sys.argv = original_argv
    candidates = load_candidates(args.candidates_csv)
    candidates = candidates.loc[pd.to_numeric(candidates["buy_rank"], errors="coerce").le(policy_args.candidate_universe_top_n)].copy()
    candidates = candidates.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)
    snapshot_archive = load_snapshot_archive(args.snapshot_archive_csv)
    latest_snapshot, previous_snapshot = build_reference_maps(snapshot_archive)

    candidate_asof_raw = str(candidates["asof_date"].iloc[0]) if "asof_date" in candidates.columns and not candidates.empty else ""
    candidate_asof = pd.to_datetime(candidate_asof_raw, errors="coerce")
    latest_snapshot_date = latest_snapshot["asof_date"].iloc[0] if not latest_snapshot.empty else pd.NaT
    asof_date = pd.Timestamp(candidate_asof if pd.notna(candidate_asof) else latest_snapshot_date if pd.notna(latest_snapshot_date) else pd.Timestamp.today()).normalize()

    current_holdings_context = load_holdings(policy_args, candidates, latest_snapshot)
    current_holdings = current_holdings_context.open_positions.copy()
    empty_holdings = _empty_holdings_frame()

    scenarios = [
        _build_scenario_payload(
            scenario_key="watch_current_holdings",
            scenario_label="WATCH 가정 / 현재 실계좌 보유 기준",
            holdings=current_holdings,
            holdings_source=current_holdings_context.source,
            candidates=candidates,
            latest_snapshot=latest_snapshot,
            previous_snapshot=previous_snapshot,
            policy_args=policy_args,
            asof_date=asof_date,
        ),
        _build_scenario_payload(
            scenario_key="watch_empty_account",
            scenario_label="WATCH 가정 / 빈 계좌 기준",
            holdings=empty_holdings,
            holdings_source="empty_account_simulation",
            candidates=candidates,
            latest_snapshot=latest_snapshot,
            previous_snapshot=previous_snapshot,
            policy_args=policy_args,
            asof_date=asof_date,
        ),
    ]

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof_date.strftime("%Y-%m-%d"),
        "policy_version": POLICY_VERSION,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "gate_version": GATE_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "candidate_universe_top_n": int(policy_args.candidate_universe_top_n),
        "entry_review_top_n": int(policy_args.entry_review_top_n),
        "entry_review_extended_top_n": int(policy_args.entry_review_extended_top_n),
        "watch_limited_max_entries": int(policy_args.watch_limited_max_entries),
        "watch_limited_total_exposure": float(policy_args.watch_limited_total_exposure),
        "watch_limited_position_cap": float(policy_args.watch_limited_position_cap),
        "scenarios": scenarios,
    }

    md_lines = [
        "# WATCH Auto-Buy Simulation",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- asof_date: {payload['asof_date']}",
        f"- candidate_universe_top_n: {payload['candidate_universe_top_n']}",
        f"- entry_review_top_n: {payload['entry_review_top_n']}",
        f"- entry_review_extended_top_n: {payload['entry_review_extended_top_n']}",
        f"- watch_limited_max_entries: {payload['watch_limited_max_entries']}",
        f"- watch_limited_total_exposure: {payload['watch_limited_total_exposure']:.2%}",
        f"- watch_limited_position_cap: {payload['watch_limited_position_cap']:.2%}",
        "",
    ]
    for scenario in scenarios:
        actions_df = pd.DataFrame(scenario["actions"])
        md_lines.extend(
            [
                f"## {scenario['scenario_label']}",
                "",
                f"- holdings_source: {scenario['holdings_source']}",
                f"- holding_count: {scenario['holding_count']}",
                f"- enter_count: {scenario['summary']['enter_count']}",
                f"- total_target_weight: {scenario['summary']['total_target_weight']:.2%}",
                "",
                _markdown_table(
                    actions_df if not actions_df.empty else pd.DataFrame([{"code": "-", "name": "", "action": "NO_ACTION", "target_weight": None, "reason": "no actions"}]),
                    ["code", "name", "action", "target_weight", "reason"],
                ),
                "",
            ]
        )

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    upsert_json_payload(
        "watch_auto_buy_simulation",
        payload,
        asof_date=payload.get("asof_date"),
        generated_at=payload.get("generated_at"),
        source_path=out_json,
    )
    print(f"watch_auto_buy_simulation_json: {out_json}")
    print(f"watch_auto_buy_simulation_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
