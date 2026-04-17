from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from buy_candidate_builder import cap_count, normalize_input, select_candidates


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
HISTORY_DIR = DATA_DIR / "history" / "ranking"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"

POSITIONS_CSV = DATA_DIR / "paper_trading_positions.csv"
NAV_CSV = DATA_DIR / "paper_trading_nav.csv"
REPORT_MD = OUTPUT_DIR / "paper_trading_report.md"

TARGET_SIZES = [5, 8, 10]
HOLDING_POLICY_CODE = "FIXED_HOLD_TRADING_DAYS"
ENTRY_ACTION_CODE = "BUY_NEW_COHORT"
ENTRY_ACTION_REASON = "selected_from_daily_snapshot"
ENTRY_ACTION_CODE_REPLACEMENT = "BUY_REPLACEMENT"
ENTRY_ACTION_REASON_REPLACEMENT = "filled_vacancy_after_exit"
OPEN_ACTION_CODE = "HOLD_ACTIVE"
OPEN_ACTION_REASON = "planned_holding_period_not_reached"
OPEN_ACTION_CODE_NEW = "HOLD_NEW_ENTRY"
OPEN_ACTION_REASON_NEW = "newly_opened_position"
OPEN_ACTION_CODE_REVIEW = "HOLD_REVIEW_SOON"
OPEN_ACTION_REASON_REVIEW = "approaching_holding_review_window"
OPEN_ACTION_CODE_NEAR_EXIT = "EXIT_REVIEW_SOON"
OPEN_ACTION_REASON_NEAR_EXIT = "near_planned_exit_date"
EXIT_ACTION_CODE = "EXIT_HOLD_D20"
EXIT_ACTION_REASON = "planned_holding_period_reached"
EXIT_ACTION_CODE_STOP_LOSS = "EXIT_STOP_LOSS"
EXIT_ACTION_REASON_STOP_LOSS = "loss_below_minus_8pct"
EXIT_ACTION_CODE_CONFIDENCE = "EXIT_CONFIDENCE_BLOCK"
EXIT_ACTION_REASON_CONFIDENCE = "confidence_below_55"
EXIT_ACTION_CODE_SCORE = "EXIT_SCORE_WEAK"
EXIT_ACTION_REASON_SCORE = "final_score_below_45"
EXIT_ACTION_CODE_RANK_FADE = "EXIT_RANK_FADE"
EXIT_ACTION_REASON_RANK_FADE = "rank_outside_top10_with_confidence_weak"


@dataclass
class Position:
    strategy: str
    code: str
    name: str
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp | None
    entry_price_close: float
    entry_exec_price: float
    shares: float
    entry_notional_gross: float
    entry_cost_amount: float
    source_rank: int
    selection_stage: str
    dominant_theme: str
    confidence_score: float | None
    final_score: float | None
    holding_age_trading_days: int = 0
    remaining_holding_days: int | None = None
    holding_policy_code: str = HOLDING_POLICY_CODE
    entry_action_code: str = ENTRY_ACTION_CODE
    entry_action_reason: str = ENTRY_ACTION_REASON
    exit_date: pd.Timestamp | None = None
    exit_price_close: float | None = None
    exit_exec_price: float | None = None
    exit_notional_net: float | None = None
    exit_cost_amount: float | None = None
    status: str = "OPEN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper trading ledger from historical ranking snapshots.")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--out-positions-csv", type=Path, default=POSITIONS_CSV)
    parser.add_argument("--out-nav-csv", type=Path, default=NAV_CSV)
    parser.add_argument("--out-md", type=Path, default=REPORT_MD)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--entry-fee-bps", type=float, default=0.0)
    parser.add_argument("--exit-fee-bps", type=float, default=0.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=0.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=0.0)
    parser.add_argument("--replacement-mode", action="store_true", help="Keep active positions near target_size by filling vacancies after exits.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def default_candidate_args() -> argparse.Namespace:
    return argparse.Namespace(
        min_confidence=80.0,
        min_liquidity_score=15.0,
        min_trading_value=5_000_000_000.0,
        sector_cap_ratio=0.30,
        theme_cap_ratio=0.30,
        no_theme_cap_ratio=0.50,
        soft_surge_ret5d=0.12,
        soft_surge_ret10d=0.20,
        soft_surge_rsi=70.0,
        hard_surge_ret5d=0.20,
        hard_surge_ret10d=0.35,
        hard_surge_rsi=80.0,
    )


def load_price_panel(prices_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(_resolve(prices_csv), dtype={"code": str}, low_memory=False)
    close_col = "adj_close" if "adj_close" in prices.columns else "close"
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["close"] = pd.to_numeric(prices[close_col], errors="coerce")
    prices = prices.dropna(subset=["date", "code", "close"]).sort_values(["date", "code"]).reset_index(drop=True)
    panel = prices.pivot(index="date", columns="code", values="close").sort_index().ffill()
    return prices, panel


def build_candidate_history(history_dir: Path) -> tuple[dict[int, dict[pd.Timestamp, pd.DataFrame]], pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    args = default_candidate_args()
    candidate_history: dict[int, dict[pd.Timestamp, pd.DataFrame]] = {size: {} for size in TARGET_SIZES}
    signal_rows: list[dict[str, object]] = []
    snapshot_history: dict[pd.Timestamp, pd.DataFrame] = {}

    for path in sorted(_resolve(history_dir).glob("*_ranking_final.csv")):
        df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if df.empty or "date" not in df.columns:
            continue
        latest_date = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(latest_date):
            continue
        latest = df.loc[pd.to_datetime(df["date"], errors="coerce").eq(latest_date)].copy()
        for col, default in {
            "dominant_theme": "(none)",
            "theme_score": 0.0,
            "explain_text": "",
            "market": "",
            "sector": "(unknown)",
            "ret_5d": pd.NA,
            "ret_10d": pd.NA,
            "mom_20": pd.NA,
            "rsi_14": pd.NA,
            "confidence_score": pd.NA,
            "liquidity_score": pd.NA,
            "volume": pd.NA,
            "close": pd.NA,
            "rank_final": pd.NA,
            "final_score": pd.NA,
        }.items():
            if col not in latest.columns:
                latest[col] = default
        latest_norm = normalize_input(latest, asof_date=latest_date.strftime("%Y-%m-%d"), args=args)
        snapshot_history[latest_date.normalize()] = latest_norm.copy()
        for target_size in TARGET_SIZES:
            selected, summary = select_candidates(
                latest_norm,
                target_size=target_size,
                sector_cap=cap_count(target_size, args.sector_cap_ratio),
                theme_cap=cap_count(target_size, args.theme_cap_ratio),
                no_theme_cap=cap_count(target_size, args.no_theme_cap_ratio),
            )
            candidate_history[target_size][latest_date.normalize()] = selected.copy()
            signal_rows.append(
                {
                    "date": latest_date.normalize(),
                    "strategy": f"top{target_size}",
                    "target_size": target_size,
                    "selected_count": int(len(selected)),
                    "strict_selected_count": int(summary.get("strict_selected_count", 0)),
                    "eligible_count": int(summary.get("eligible_count", 0)),
                }
            )

    return candidate_history, pd.DataFrame(signal_rows).sort_values(["date", "target_size"]).reset_index(drop=True), snapshot_history


def compute_exit_date(price_dates_by_code: dict[str, list[pd.Timestamp]], code: str, entry_date: pd.Timestamp, hold_days: int) -> pd.Timestamp | None:
    dates = price_dates_by_code.get(code)
    if not dates:
        return None
    try:
        idx = dates.index(entry_date)
    except ValueError:
        return None
    exit_idx = idx + hold_days
    if exit_idx >= len(dates):
        return None
    return dates[exit_idx]


def compute_holding_age_trading_days(
    price_dates_by_code: dict[str, list[pd.Timestamp]],
    code: str,
    entry_date: pd.Timestamp,
    asof_date: pd.Timestamp | None,
) -> int | None:
    if asof_date is None:
        return None
    dates = price_dates_by_code.get(code)
    if not dates:
        return None
    try:
        entry_idx = dates.index(entry_date)
        asof_idx = dates.index(asof_date)
    except ValueError:
        return None
    if asof_idx < entry_idx:
        return 0
    return int(asof_idx - entry_idx)


def resolve_open_action(holding_age_trading_days: int | None, remaining_holding_days: int | None) -> tuple[str, str]:
    if holding_age_trading_days is None:
        return OPEN_ACTION_CODE, OPEN_ACTION_REASON
    if holding_age_trading_days <= 1:
        return OPEN_ACTION_CODE_NEW, OPEN_ACTION_REASON_NEW
    if remaining_holding_days is not None and remaining_holding_days <= 3:
        return OPEN_ACTION_CODE_NEAR_EXIT, OPEN_ACTION_REASON_NEAR_EXIT
    if holding_age_trading_days >= 15:
        return OPEN_ACTION_CODE_REVIEW, OPEN_ACTION_REASON_REVIEW
    return OPEN_ACTION_CODE, OPEN_ACTION_REASON


def classify_early_exit(
    *,
    holding_age_trading_days: int | None,
    planned_exit_date: pd.Timestamp | None,
    current_date: pd.Timestamp,
    current_return: float | None,
    final_score: float | None,
    confidence_score: float | None,
    live_rank: float | None,
) -> tuple[str, str] | None:
    if pd.notna(holding_age_trading_days) and float(holding_age_trading_days) >= 5:
        if pd.notna(current_return) and float(current_return) <= -0.08:
            return EXIT_ACTION_CODE_STOP_LOSS, EXIT_ACTION_REASON_STOP_LOSS
        if pd.notna(confidence_score) and float(confidence_score) < 55:
            return EXIT_ACTION_CODE_CONFIDENCE, EXIT_ACTION_REASON_CONFIDENCE
        if pd.notna(final_score) and float(final_score) < 45:
            return EXIT_ACTION_CODE_SCORE, EXIT_ACTION_REASON_SCORE
    if pd.notna(holding_age_trading_days) and float(holding_age_trading_days) >= 10:
        if pd.notna(live_rank) and float(live_rank) > 10 and pd.notna(confidence_score) and float(confidence_score) < 70:
            return EXIT_ACTION_CODE_RANK_FADE, EXIT_ACTION_REASON_RANK_FADE
    if planned_exit_date is not None and planned_exit_date == current_date:
        return EXIT_ACTION_CODE, EXIT_ACTION_REASON
    return None


def run_strategy_ledger(
    *,
    strategy_name: str,
    target_size: int,
    candidates_by_date: dict[pd.Timestamp, pd.DataFrame],
    snapshot_by_date: dict[pd.Timestamp, pd.DataFrame],
    price_panel: pd.DataFrame,
    price_dates_by_code: dict[str, list[pd.Timestamp]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entry_fee_rate = args.entry_fee_bps / 10_000.0
    exit_fee_rate = args.exit_fee_bps / 10_000.0
    entry_slip_rate = args.entry_slippage_bps / 10_000.0
    exit_slip_rate = args.exit_slippage_bps / 10_000.0

    trading_dates = [
        date
        for date in price_panel.index.tolist()
        if date >= min(candidates_by_date.keys(), default=price_panel.index.min())
    ]

    cash = float(args.initial_nav)
    active_positions: list[Position] = []
    closed_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []
    prior_nav = float(args.initial_nav)

    for current_date in trading_dates:
        current_date = pd.Timestamp(current_date).normalize()
        day_prices = price_panel.loc[current_date]
        day_snapshot = snapshot_by_date.get(current_date, pd.DataFrame())
        snapshot_lookup = (
            day_snapshot.sort_values(["code"]).drop_duplicates("code", keep="last").set_index("code")
            if not day_snapshot.empty and "code" in day_snapshot.columns
            else pd.DataFrame()
        )

        remaining_positions: list[Position] = []
        for pos in active_positions:
            close_price = pd.to_numeric(day_prices.get(pos.code), errors="coerce")
            if pd.isna(close_price):
                remaining_positions.append(pos)
                continue
            holding_age_trading_days = compute_holding_age_trading_days(
                price_dates_by_code,
                pos.code,
                pos.entry_date,
                current_date,
            )
            snapshot_row = snapshot_lookup.loc[pos.code] if not snapshot_lookup.empty and pos.code in snapshot_lookup.index else pd.Series(dtype="object")
            current_return = (float(close_price) / pos.entry_price_close - 1.0) if pos.entry_price_close else None
            exit_decision = classify_early_exit(
                holding_age_trading_days=holding_age_trading_days,
                planned_exit_date=pos.planned_exit_date,
                current_date=current_date,
                current_return=current_return,
                final_score=pd.to_numeric(snapshot_row.get("final_score"), errors="coerce") if not snapshot_row.empty else pd.NA,
                confidence_score=pd.to_numeric(snapshot_row.get("confidence_score"), errors="coerce") if not snapshot_row.empty else pd.NA,
                live_rank=pd.to_numeric(snapshot_row.get("rank_source"), errors="coerce") if not snapshot_row.empty else pd.NA,
            )
            if not exit_decision:
                remaining_positions.append(pos)
                continue

            exit_action_code, exit_action_reason = exit_decision
            exit_exec_price = float(close_price) * (1.0 - exit_slip_rate)
            exit_notional = pos.shares * exit_exec_price
            exit_cost = exit_notional * exit_fee_rate
            cash += exit_notional - exit_cost
            pos.exit_date = current_date
            pos.exit_price_close = float(close_price)
            pos.exit_exec_price = exit_exec_price
            pos.exit_notional_net = exit_notional - exit_cost
            pos.exit_cost_amount = exit_cost
            pos.status = "CLOSED"
            gross_return = (float(close_price) / pos.entry_price_close - 1.0) if pos.entry_price_close else None
            net_return = ((pos.exit_notional_net or 0.0) / pos.entry_notional_gross - 1.0) if pos.entry_notional_gross else None
            closed_rows.append(
                {
                    "strategy": strategy_name,
                    "code": pos.code,
                    "name": pos.name,
                    "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                    "planned_exit_date": pos.planned_exit_date.strftime("%Y-%m-%d") if pos.planned_exit_date is not None else None,
                    "exit_date": pos.exit_date.strftime("%Y-%m-%d"),
                    "entry_price_close": pos.entry_price_close,
                    "entry_exec_price": pos.entry_exec_price,
                    "exit_price_close": pos.exit_price_close,
                    "exit_exec_price": pos.exit_exec_price,
                    "shares": pos.shares,
                    "entry_notional_gross": pos.entry_notional_gross,
                    "exit_notional_net": pos.exit_notional_net,
                    "entry_cost_amount": pos.entry_cost_amount,
                    "exit_cost_amount": pos.exit_cost_amount,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "source_rank": pos.source_rank,
                    "selection_stage": pos.selection_stage,
                    "dominant_theme": pos.dominant_theme,
                    "confidence_score": pos.confidence_score,
                    "final_score": pos.final_score,
                    "holding_age_trading_days": holding_age_trading_days,
                    "remaining_holding_days": 0,
                    "holding_policy_code": pos.holding_policy_code,
                    "entry_action_code": pos.entry_action_code,
                    "entry_action_reason": pos.entry_action_reason,
                    "current_action_code": "POSITION_CLOSED",
                    "current_action_reason": exit_action_reason,
                    "exit_action_code": exit_action_code,
                    "exit_action_reason": exit_action_reason,
                    "status": "CLOSED",
                }
            )
        active_positions = remaining_positions

        candidates = candidates_by_date.get(current_date, pd.DataFrame())
        duplicate_skips = 0
        opened_today = 0
        deployed_today = 0.0
        if not candidates.empty:
            open_codes = {pos.code for pos in active_positions}
            tradable = candidates.copy()
            tradable["current_close"] = tradable["code"].map(day_prices.to_dict())
            tradable["current_close"] = pd.to_numeric(tradable["current_close"], errors="coerce")
            tradable = tradable.dropna(subset=["current_close"]).copy()
            tradable = tradable.loc[~tradable["code"].isin(open_codes)].copy()
            duplicate_skips = int(len(candidates) - len(tradable))
            if not tradable.empty:
                nav_before_entry = cash + sum(
                    pos.shares * float(pd.to_numeric(day_prices.get(pos.code), errors="coerce"))
                    for pos in active_positions
                    if pd.notna(pd.to_numeric(day_prices.get(pos.code), errors="coerce"))
                )
                tranche_budget = min(cash, nav_before_entry / float(args.hold_days))
                if tranche_budget > 0:
                    tradable_entries = tradable.copy()
                    entry_action_code = ENTRY_ACTION_CODE
                    entry_action_reason = ENTRY_ACTION_REASON
                    if args.replacement_mode:
                        vacancy_count = max(target_size - len(active_positions), 0)
                        if vacancy_count <= 0:
                            tradable_entries = tradable_entries.iloc[0:0].copy()
                        else:
                            tradable_entries = tradable_entries.head(vacancy_count).copy()
                            desired_slot_budget = nav_before_entry / float(max(target_size, 1))
                            tranche_budget = min(cash, desired_slot_budget * float(vacancy_count))
                            entry_action_code = ENTRY_ACTION_CODE_REPLACEMENT
                            entry_action_reason = ENTRY_ACTION_REASON_REPLACEMENT
                    if tradable_entries.empty or tranche_budget <= 0:
                        tradable_entries = tradable_entries.iloc[0:0].copy()
                    per_name_budget = tranche_budget / float(len(tradable_entries)) if len(tradable_entries) else 0.0
                    for _, row in tradable_entries.iterrows():
                        close_price = float(row["current_close"])
                        entry_exec_price = close_price * (1.0 + entry_slip_rate)
                        if entry_exec_price <= 0:
                            continue
                        shares = per_name_budget / (entry_exec_price * (1.0 + entry_fee_rate))
                        gross_notional = shares * entry_exec_price
                        entry_cost = gross_notional * entry_fee_rate
                        total_cash_use = gross_notional + entry_cost
                        if shares <= 0 or total_cash_use <= 0 or total_cash_use > cash + 1e-9:
                            continue
                        planned_exit_date = compute_exit_date(
                            price_dates_by_code=price_dates_by_code,
                            code=str(row["code"]),
                            entry_date=current_date,
                            hold_days=args.hold_days,
                        )
                        pos = Position(
                            strategy=strategy_name,
                            code=str(row["code"]),
                            name=str(row.get("name", "")),
                            entry_date=current_date,
                            planned_exit_date=planned_exit_date,
                            entry_price_close=close_price,
                            entry_exec_price=entry_exec_price,
                            shares=shares,
                            entry_notional_gross=gross_notional,
                            entry_cost_amount=entry_cost,
                            source_rank=int(pd.to_numeric(row.get("rank_source"), errors="coerce")),
                            selection_stage=str(row.get("selection_stage", "")),
                            dominant_theme=str(row.get("dominant_theme", "(none)")),
                            confidence_score=pd.to_numeric(row.get("confidence_score"), errors="coerce"),
                            final_score=pd.to_numeric(row.get("final_score"), errors="coerce"),
                            holding_age_trading_days=0,
                            remaining_holding_days=args.hold_days,
                            holding_policy_code=HOLDING_POLICY_CODE,
                            entry_action_code=entry_action_code,
                            entry_action_reason=entry_action_reason,
                        )
                        active_positions.append(pos)
                        cash -= total_cash_use
                        deployed_today += total_cash_use
                        opened_today += 1
        signal_rows.append(
            {
                "strategy": strategy_name,
                "date": current_date.strftime("%Y-%m-%d"),
                "signal_count": int(len(candidates)),
                "opened_count": opened_today,
                "duplicate_skip_count": duplicate_skips,
                "deployed_cash": deployed_today,
            }
        )

        market_value = 0.0
        open_position_count = 0
        for pos in active_positions:
            close_price = pd.to_numeric(day_prices.get(pos.code), errors="coerce")
            if pd.isna(close_price):
                continue
            market_value += pos.shares * float(close_price)
            open_position_count += 1
        nav = cash + market_value
        daily_return = (nav / prior_nav - 1.0) if prior_nav > 0 else None
        nav_rows.append(
            {
                "strategy": strategy_name,
                "date": current_date.strftime("%Y-%m-%d"),
                "cash": cash,
                "market_value": market_value,
                "nav": nav,
                "daily_return": daily_return,
                "active_position_count": open_position_count,
                "opened_today": opened_today,
                "duplicate_skip_count": duplicate_skips,
                "deployed_cash": deployed_today,
            }
        )
        prior_nav = nav

    open_rows = []
    last_date = pd.Timestamp(trading_dates[-1]).normalize() if trading_dates else None
    last_prices = price_panel.loc[last_date] if last_date is not None else pd.Series(dtype=float)
    for pos in active_positions:
        mark_price = pd.to_numeric(last_prices.get(pos.code), errors="coerce")
        unrealized_return = (float(mark_price) / pos.entry_price_close - 1.0) if pd.notna(mark_price) and pos.entry_price_close else None
        holding_age_trading_days = compute_holding_age_trading_days(
            price_dates_by_code,
            pos.code,
            pos.entry_date,
            last_date,
        )
        remaining_holding_days = max(args.hold_days - holding_age_trading_days, 0) if holding_age_trading_days is not None else None
        current_action_code, current_action_reason = resolve_open_action(
            holding_age_trading_days,
            remaining_holding_days,
        )
        open_rows.append(
            {
                "strategy": strategy_name,
                "code": pos.code,
                "name": pos.name,
                "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                "planned_exit_date": pos.planned_exit_date.strftime("%Y-%m-%d") if pos.planned_exit_date is not None else None,
                "exit_date": None,
                "entry_price_close": pos.entry_price_close,
                "entry_exec_price": pos.entry_exec_price,
                "exit_price_close": None,
                "exit_exec_price": None,
                "shares": pos.shares,
                "entry_notional_gross": pos.entry_notional_gross,
                "exit_notional_net": None,
                "entry_cost_amount": pos.entry_cost_amount,
                "exit_cost_amount": None,
                "gross_return": unrealized_return,
                "net_return": unrealized_return,
                "source_rank": pos.source_rank,
                "selection_stage": pos.selection_stage,
                "dominant_theme": pos.dominant_theme,
                "confidence_score": pos.confidence_score,
                "final_score": pos.final_score,
                "holding_age_trading_days": holding_age_trading_days,
                "remaining_holding_days": remaining_holding_days,
                "holding_policy_code": pos.holding_policy_code,
                "entry_action_code": pos.entry_action_code,
                "entry_action_reason": pos.entry_action_reason,
                "current_action_code": current_action_code,
                "current_action_reason": current_action_reason,
                "exit_action_code": None,
                "exit_action_reason": None,
                "status": "OPEN",
            }
        )

    positions = pd.DataFrame(closed_rows + open_rows)
    nav_df = pd.DataFrame(nav_rows)
    signal_df = pd.DataFrame(signal_rows)
    return positions, nav_df, signal_df


def enrich_nav(nav_df: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    out = nav_df.copy()
    if out.empty:
        return out
    out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
    out["cumulative_return"] = out["nav"] / float(out["nav"].iloc[0]) - 1.0
    out["running_nav_max"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_nav_max"] - 1.0
    strategy = str(out["strategy"].iloc[0])
    closed = positions.loc[(positions["strategy"] == strategy) & (positions["status"] == "CLOSED")].copy()
    if closed.empty:
        out["closed_trade_count"] = 0
        out["closed_win_rate"] = pd.NA
        return out

    closed["exit_date"] = pd.to_datetime(closed["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wins_by_day = (
        closed.assign(win=(pd.to_numeric(closed["net_return"], errors="coerce") > 0).astype(int))
        .groupby("exit_date")
        .agg(closed_trade_count=("code", "count"), closed_win_count=("win", "sum"))
        .reset_index()
        .rename(columns={"exit_date": "date"})
    )
    out = out.merge(wins_by_day, on="date", how="left")
    out["closed_trade_count"] = pd.to_numeric(out["closed_trade_count"], errors="coerce").fillna(0).astype(int)
    out["closed_win_count"] = pd.to_numeric(out.get("closed_win_count"), errors="coerce").fillna(0).astype(int)
    out["closed_trade_count_cum"] = out["closed_trade_count"].cumsum()
    out["closed_win_count_cum"] = out["closed_win_count"].cumsum()
    out["closed_win_rate"] = out["closed_win_count_cum"] / out["closed_trade_count_cum"].replace(0, pd.NA)
    return out


def build_report(
    *,
    signal_summary: pd.DataFrame,
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strategy_rows: list[dict[str, object]] = []
    exit_action_rollup_rows: list[dict[str, object]] = []
    for strategy, nav_df in nav.groupby("strategy", sort=True):
        pos_df = positions.loc[positions["strategy"] == strategy].copy()
        closed = pos_df.loc[pos_df["status"] == "CLOSED"].copy()
        last = nav_df.sort_values("date").iloc[-1]
        strategy_rows.append(
            {
                "strategy": strategy,
                "final_nav": _fmt_num(last["nav"]),
                "cumulative_return": _fmt_pct(last["cumulative_return"]),
                "max_drawdown": _fmt_pct(nav_df["drawdown"].min()),
                "closed_trades": int(len(closed)),
                "win_rate": _fmt_pct((pd.to_numeric(closed["net_return"], errors="coerce") > 0).mean()) if not closed.empty else "NA",
                "open_positions": int((pos_df["status"] == "OPEN").sum()),
                "duplicate_skips_total": int(pd.to_numeric(nav_df["duplicate_skip_count"], errors="coerce").fillna(0).sum()),
            }
        )
        if not closed.empty and "exit_action_code" in closed.columns:
            exit_counts = (
                closed.assign(exit_action_code=closed["exit_action_code"].fillna("UNKNOWN"))
                .groupby("exit_action_code", as_index=False)
                .agg(count=("code", "count"))
            )
            for _, row in exit_counts.iterrows():
                exit_action_rollup_rows.append(
                    {
                        "strategy": strategy,
                        "exit_action_code": row["exit_action_code"],
                        "count": int(row["count"]),
                    }
                )
    strategy_summary = pd.DataFrame(strategy_rows)
    exit_action_rollup = pd.DataFrame(exit_action_rollup_rows)

    signal_rollup = (
        signal_summary.groupby("strategy", as_index=False)
        .agg(
            signal_dates=("date", "nunique"),
            signal_count_total=("signal_count", "sum"),
            opened_count_total=("opened_count", "sum"),
            duplicate_skip_count_total=("duplicate_skip_count", "sum"),
        )
        .sort_values("strategy")
    )

    lines = [
        "# Paper Trading Report",
        "",
        f"- generated_at: {generated_at}",
        f"- history_snapshot_dir: {_resolve(args.history_dir)}",
        f"- price_source: {_resolve(args.prices_csv)}",
        f"- hold_days: {args.hold_days}",
        f"- initial_nav_per_strategy: {_fmt_num(args.initial_nav)}",
        f"- entry_fee_bps: {args.entry_fee_bps}",
        f"- exit_fee_bps: {args.exit_fee_bps}",
        f"- entry_slippage_bps: {args.entry_slippage_bps}",
        f"- exit_slippage_bps: {args.exit_slippage_bps}",
        f"- replacement_mode: {'on' if args.replacement_mode else 'off'}",
        "",
        "## Trading Rules",
        "- Each signal date reconstructs `buy_candidates_top5/top8/top10` from the stored ranking snapshot using the current operational buy-candidate rules.",
        (
            f"- Replacement mode: keep active positions near target_size by filling daily vacancies from the latest candidate set."
            if args.replacement_mode
            else f"- Each strategy deploys roughly `1/{args.hold_days}` of current NAV into the day cohort and holds each position for {args.hold_days} trading days."
        ),
        "- Duplicate entry rule: if a code is already open in the same strategy, the new signal is skipped until the existing position exits.",
        "- Entry and exit are both modeled at daily close with optional fee/slippage adjustments.",
        "",
        "## Strategy Summary",
        _markdown_table(
            strategy_summary,
            ["strategy", "final_nav", "cumulative_return", "max_drawdown", "closed_trades", "win_rate", "open_positions", "duplicate_skips_total"],
        ),
        "",
        "## Signal Summary",
        _markdown_table(
            signal_rollup,
            ["strategy", "signal_dates", "signal_count_total", "opened_count_total", "duplicate_skip_count_total"],
        ),
    ]

    latest_nav = nav.sort_values(["strategy", "date"]).groupby("strategy").tail(5).copy()
    latest_nav["nav"] = latest_nav["nav"].map(_fmt_num)
    latest_nav["daily_return"] = latest_nav["daily_return"].map(_fmt_pct)
    latest_nav["cumulative_return"] = latest_nav["cumulative_return"].map(_fmt_pct)
    latest_nav["drawdown"] = latest_nav["drawdown"].map(_fmt_pct)
    lines.extend(
        [
            "",
            "## Recent NAV",
            _markdown_table(
                latest_nav,
                ["strategy", "date", "nav", "daily_return", "cumulative_return", "drawdown", "active_position_count", "opened_today", "duplicate_skip_count"],
            ),
        ]
    )

    if not exit_action_rollup.empty:
        lines.extend(
            [
                "",
                "## Exit Action Summary",
                _markdown_table(
                    exit_action_rollup.sort_values(["strategy", "count", "exit_action_code"], ascending=[True, False, True]),
                    ["strategy", "exit_action_code", "count"],
                ),
            ]
        )

    open_positions = positions.loc[positions["status"] == "OPEN"].copy()
    if not open_positions.empty:
        open_positions["gross_return"] = open_positions["gross_return"].map(_fmt_pct)
        lines.extend(
            [
                "",
                "## Open Positions",
                _markdown_table(
                    open_positions.sort_values(["strategy", "entry_date", "code"]),
                    ["strategy", "code", "name", "entry_date", "planned_exit_date", "current_action_code", "selection_stage", "gross_return"],
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    prices, price_panel = load_price_panel(args.prices_csv)
    price_dates_by_code = {
        code: group["date"].dt.normalize().tolist()
        for code, group in prices.groupby("code", sort=False)
    }

    candidate_history, _, snapshot_history = build_candidate_history(args.history_dir)

    all_positions: list[pd.DataFrame] = []
    all_nav: list[pd.DataFrame] = []
    all_signal_logs: list[pd.DataFrame] = []
    for size in TARGET_SIZES:
        positions_df, nav_df, signal_df = run_strategy_ledger(
            strategy_name=f"top{size}",
            target_size=size,
            candidates_by_date=candidate_history[size],
            snapshot_by_date=snapshot_history,
            price_panel=price_panel,
            price_dates_by_code=price_dates_by_code,
            args=args,
        )
        nav_df = enrich_nav(nav_df, positions_df)
        all_positions.append(positions_df)
        all_nav.append(nav_df)
        all_signal_logs.append(signal_df.assign(date=pd.to_datetime(signal_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")))

    positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()
    nav = pd.concat(all_nav, ignore_index=True) if all_nav else pd.DataFrame()
    signal_summary = pd.concat(all_signal_logs, ignore_index=True) if all_signal_logs else pd.DataFrame()

    out_positions = _resolve(args.out_positions_csv)
    out_nav = _resolve(args.out_nav_csv)
    out_md = _resolve(args.out_md)
    out_positions.parent.mkdir(parents=True, exist_ok=True)
    out_nav.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(out_positions, index=False, encoding="utf-8-sig")
    nav.to_csv(out_nav, index=False, encoding="utf-8-sig")
    out_md.write_text(build_report(signal_summary=signal_summary, positions=positions, nav=nav, args=args), encoding="utf-8")

    print(f"positions_csv: {out_positions}")
    print(f"nav_csv: {out_nav}")
    print(f"report_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
