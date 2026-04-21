from __future__ import annotations

from dataclasses import dataclass

import argparse
import pandas as pd

from apply_execution_policy import (
    HoldingsContext,
    build_execution_actions,
    build_reference_maps,
    classify_holdings,
    enrich_candidates_with_snapshot,
    extract_cooldown_map,
    load_holdings,
    load_snapshot_archive,
    select_buy_ready_queue,
)


@dataclass
class StrategyEvaluation:
    holdings_context: object
    latest_snapshot: pd.DataFrame
    previous_snapshot: pd.DataFrame
    enriched_candidates: pd.DataFrame
    holdings_review: pd.DataFrame
    buy_ready_queue: pd.DataFrame
    execution_actions: pd.DataFrame
    held_codes: set[str]
    cooldown_map: dict[str, dict[str, object]]


def evaluate_strategy(
    *,
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    snapshot_archive_csv,
    gate: dict[str, object],
    aggregate_paper_holdings=None,
    holdings_context_override: HoldingsContext | None = None,
) -> StrategyEvaluation:
    snapshot_archive = load_snapshot_archive(snapshot_archive_csv)
    latest_snapshot, previous_snapshot = build_reference_maps(snapshot_archive)

    holdings_context = holdings_context_override or load_holdings(args, candidates, latest_snapshot)
    if (
        aggregate_paper_holdings is not None
        and holdings_context_override is None
        and "paper_trading_positions.csv" in str(holdings_context.source)
    ):
        holdings_context.open_positions = aggregate_paper_holdings(holdings_context.open_positions)

    candidate_asof_raw = (
        str(candidates["asof_date"].iloc[0])
        if "asof_date" in candidates.columns and not candidates.empty
        else ""
    )
    candidate_asof = pd.to_datetime(candidate_asof_raw, errors="coerce")
    latest_snapshot_date = latest_snapshot["asof_date"].iloc[0] if not latest_snapshot.empty else pd.NaT
    asof_date = pd.Timestamp(
        candidate_asof
        if pd.notna(candidate_asof)
        else latest_snapshot_date
        if pd.notna(latest_snapshot_date)
        else pd.Timestamp.today()
    ).normalize()

    cooldown_map = extract_cooldown_map(
        holdings_context.closed_positions,
        asof_date,
        args.reentry_cooldown_days,
    )
    enriched_candidates = enrich_candidates_with_snapshot(candidates, latest_snapshot, previous_snapshot)
    holdings_review = classify_holdings(
        holdings_context.open_positions,
        enriched_candidates,
        latest_snapshot,
        gate,
        args,
    )
    held_codes = (
        set(holdings_context.open_positions["code"].astype(str).str.zfill(6).tolist())
        if not holdings_context.open_positions.empty
        else set()
    )
    buy_ready_queue = select_buy_ready_queue(enriched_candidates, held_codes, cooldown_map, args)
    execution_actions = build_execution_actions(holdings_review, buy_ready_queue, gate, args)

    return StrategyEvaluation(
        holdings_context=holdings_context,
        latest_snapshot=latest_snapshot,
        previous_snapshot=previous_snapshot,
        enriched_candidates=enriched_candidates,
        holdings_review=holdings_review,
        buy_ready_queue=buy_ready_queue,
        execution_actions=execution_actions,
        held_codes=held_codes,
        cooldown_map=cooldown_map,
    )
