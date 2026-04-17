from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from outcome_maturity import attach_forward_outcomes, load_price_history


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SERVING_DIR = ROOT / "serving"

RECOMMENDATIONS_JSON = SERVING_DIR / "daily_recommendations.json"
MODEL_PORTFOLIO_JSON = SERVING_DIR / "model_portfolio.json"
BUY_GATE_JSON = SERVING_DIR / "buy_gate_status.json"
SNAPSHOT_ARCHIVE_CSV = DATA_DIR / "ranking_snapshot_archive.csv"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
PAPER_POSITIONS_CSV = DATA_DIR / "paper_trading_positions.csv"

OUT_REAL_LOG_CSV = DATA_DIR / "real_trade_log.csv"
OUT_DIFF_CSV = DATA_DIR / "real_vs_model_diff.csv"
OUT_REPORT_MD = OUTPUT_DIR / "real_execution_report.md"

DATE_COLUMNS = ["actual_entry_time", "actual_exit_time"]
NUMERIC_COLUMNS = [
    "recommendation_price_ref",
    "target_weight",
    "final_score",
    "confidence_score",
    "liquidity_score",
    "theme_score",
    "ret_score",
    "prob_score",
    "tech_score",
    "quality_score",
    "risk_penalty",
    "trading_value",
    "ret_5d",
    "ret_10d",
    "mom_20",
    "rsi_14",
    "actual_entry_price",
    "actual_entry_qty",
    "actual_exit_price",
    "actual_exit_qty",
]
MANUAL_COLUMNS = [
    "actual_entry_status",
    "actual_entry_price",
    "actual_entry_time",
    "actual_entry_qty",
    "actual_entry_notes",
    "actual_exit_status",
    "actual_exit_price",
    "actual_exit_time",
    "actual_exit_qty",
    "actual_exit_notes",
    "operator_memo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shadow execution logs for real-money vs model recommendations.")
    parser.add_argument("--recommendations-json", type=Path, default=RECOMMENDATIONS_JSON)
    parser.add_argument("--model-portfolio-json", type=Path, default=MODEL_PORTFOLIO_JSON)
    parser.add_argument("--buy-gate-json", type=Path, default=BUY_GATE_JSON)
    parser.add_argument("--snapshot-archive-csv", type=Path, default=SNAPSHOT_ARCHIVE_CSV)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--paper-positions-csv", type=Path, default=PAPER_POSITIONS_CSV)
    parser.add_argument("--out-real-log-csv", type=Path, default=OUT_REAL_LOG_CSV)
    parser.add_argument("--out-diff-csv", type=Path, default=OUT_DIFF_CSV)
    parser.add_argument("--out-report-md", type=Path, default=OUT_REPORT_MD)
    parser.add_argument("--paper-horizon-days", type=int, default=20)
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


def load_json(path: Path) -> dict[str, object]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"required json not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_snapshot_archive(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"snapshot archive not found: {resolved}")
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.normalize()
    for col in [
        "rank",
        "final_score",
        "confidence_score",
        "ret_score",
        "prob_score",
        "tech_score",
        "quality_score",
        "risk_penalty",
        "theme_score",
    ]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df.dropna(subset=["asof_date"]).reset_index(drop=True)


def build_recommendation_frame(
    recommendations: dict[str, object],
    model_portfolio: dict[str, object],
    buy_gate: dict[str, object],
    snapshot_archive: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    items = recommendations.get("items", [])
    if not isinstance(items, list) or not items:
        return pd.DataFrame()

    portfolio_map: dict[str, dict[str, object]] = {}
    for holding in model_portfolio.get("holdings", []):
        if not isinstance(holding, dict):
            continue
        portfolio_map[str(holding.get("code", "")).zfill(6)] = holding

    gate_map: dict[int, dict[str, object]] = {}
    for decision in buy_gate.get("decisions", []):
        if not isinstance(decision, dict):
            continue
        bucket = pd.to_numeric(decision.get("bucket"), errors="coerce")
        if pd.notna(bucket):
            gate_map[int(bucket)] = decision

    price_lookup = prices.rename(columns={"date": "asof_date", "close": "recommendation_price_ref"}).copy()
    price_lookup["asof_date"] = pd.to_datetime(price_lookup["asof_date"], errors="coerce").dt.normalize()

    rows: list[dict[str, object]] = []
    generated_at = recommendations.get("generated_at")
    gate_generated_at = buy_gate.get("generated_at")
    gate_overall_status = buy_gate.get("overall_status") or recommendations.get("gate_overall_status")
    for item in items:
        if not isinstance(item, dict):
            continue
        security = item.get("security", {}) if isinstance(item.get("security"), dict) else {}
        scores = item.get("scores", {}) if isinstance(item.get("scores"), dict) else {}
        market_signals = item.get("market_signals", {}) if isinstance(item.get("market_signals"), dict) else {}
        selection = item.get("selection", {}) if isinstance(item.get("selection"), dict) else {}
        target_bucket = int(pd.to_numeric(item.get("target_bucket"), errors="coerce") or 0)
        code = str(security.get("code", "")).zfill(6)
        gate_decision = gate_map.get(target_bucket, {})
        holding = portfolio_map.get(code, {})
        rows.append(
            {
                "recommendation_id": str(item.get("recommendation_id", f"{item.get('asof_date')}:{target_bucket}:{code}")),
                "recommendation_generated_at": generated_at,
                "asof_date": item.get("asof_date"),
                "gate_generated_at": gate_generated_at,
                "gate_overall_status": gate_overall_status,
                "gate_bucket_status": gate_decision.get("status"),
                "gate_reason_summary": gate_decision.get("reason_summary"),
                "target_bucket": target_bucket,
                "buy_rank": pd.to_numeric(item.get("buy_rank"), errors="coerce"),
                "rank_source": pd.to_numeric(item.get("rank_source"), errors="coerce"),
                "code": code,
                "name": security.get("name"),
                "market": security.get("market"),
                "sector": security.get("sector"),
                "dominant_theme": security.get("dominant_theme"),
                "final_score": pd.to_numeric(scores.get("final_score"), errors="coerce"),
                "confidence_score": pd.to_numeric(scores.get("confidence_score"), errors="coerce"),
                "liquidity_score": pd.to_numeric(scores.get("liquidity_score"), errors="coerce"),
                "theme_score": pd.to_numeric(scores.get("theme_score"), errors="coerce"),
                "score_drift_vs_latest_snapshot": pd.to_numeric(scores.get("score_drift_vs_latest_snapshot"), errors="coerce"),
                "trading_value": pd.to_numeric(market_signals.get("trading_value"), errors="coerce"),
                "ret_5d": pd.to_numeric(market_signals.get("ret_5d"), errors="coerce"),
                "ret_10d": pd.to_numeric(market_signals.get("ret_10d"), errors="coerce"),
                "mom_20": pd.to_numeric(market_signals.get("mom_20"), errors="coerce"),
                "rsi_14": pd.to_numeric(market_signals.get("rsi_14"), errors="coerce"),
                "selection_stage": selection.get("selection_stage"),
                "selection_notes": selection.get("selection_notes"),
                "entry_rule_pass": selection.get("entry_rule_pass"),
                "recent_surge_soft_flag": selection.get("recent_surge_soft_flag"),
                "latest_snapshot_rank": pd.to_numeric(selection.get("latest_snapshot_rank"), errors="coerce"),
                "target_weight": pd.to_numeric(holding.get("target_weight"), errors="coerce"),
                "portfolio_buy_rank": pd.to_numeric(holding.get("buy_rank"), errors="coerce"),
            }
        )

    rec_df = pd.DataFrame(rows)
    rec_df["code"] = rec_df["code"].astype(str).str.zfill(6)
    rec_df["asof_date"] = pd.to_datetime(rec_df["asof_date"], errors="coerce").dt.normalize()
    if rec_df.empty:
        return rec_df

    rec_df = rec_df.merge(
        snapshot_archive[
            [
                "asof_date",
                "code",
                "name",
                "rank",
                "ret_score",
                "prob_score",
                "tech_score",
                "quality_score",
                "risk_penalty",
                "theme_score",
                "explain_text",
            ]
        ].rename(columns={"name": "snapshot_name", "rank": "snapshot_rank", "theme_score": "snapshot_theme_score"}),
        on=["asof_date", "code"],
        how="left",
    )
    rec_df = rec_df.merge(
        price_lookup[["asof_date", "code", "recommendation_price_ref"]],
        on=["asof_date", "code"],
        how="left",
    )
    rec_df["name"] = rec_df["snapshot_name"].fillna(rec_df["name"])
    rec_df["theme_score"] = rec_df["theme_score"].fillna(rec_df["snapshot_theme_score"])
    rec_df = rec_df.drop(columns=["snapshot_name", "snapshot_theme_score"], errors="ignore")
    return rec_df.sort_values(["asof_date", "target_bucket", "buy_rank", "code"]).reset_index(drop=True)


def load_existing_log(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str, "recommendation_id": str}, low_memory=False)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["recommendation_id"] = df["recommendation_id"].astype(str)
    df["asof_date"] = pd.to_datetime(df.get("asof_date"), errors="coerce").dt.normalize()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df.get(col), errors="coerce")
    return df


def merge_real_log(recommendation_df: pd.DataFrame, existing_log: pd.DataFrame) -> pd.DataFrame:
    base = recommendation_df.copy()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for column in MANUAL_COLUMNS:
        if column not in base.columns:
            base[column] = pd.NA
    base["record_created_at"] = now_text
    base["record_updated_at"] = now_text

    if existing_log.empty:
        base["actual_entry_status"] = base["actual_entry_status"].fillna("PENDING")
        base["actual_exit_status"] = base["actual_exit_status"].fillna("OPEN")
        return base

    existing = existing_log.copy().drop_duplicates("recommendation_id", keep="last").set_index("recommendation_id")
    merged = base.drop_duplicates("recommendation_id", keep="last").set_index("recommendation_id")

    for column in MANUAL_COLUMNS + ["record_created_at"]:
        if column in existing.columns:
            merged[column] = existing[column]

    missing_rows = existing.loc[~existing.index.isin(merged.index)].copy()
    if not missing_rows.empty:
        for column in merged.columns:
            if column not in missing_rows.columns:
                missing_rows[column] = pd.NA
        merged = pd.concat([merged, missing_rows[merged.columns]], axis=0)

    merged["record_created_at"] = merged["record_created_at"].fillna(now_text)
    merged["record_updated_at"] = now_text
    merged["actual_entry_status"] = merged["actual_entry_status"].fillna("PENDING")
    merged["actual_exit_status"] = merged["actual_exit_status"].fillna("OPEN")
    merged = merged.reset_index()

    merged["code"] = merged["code"].astype(str).str.zfill(6)
    merged["asof_date"] = pd.to_datetime(merged["asof_date"], errors="coerce").dt.normalize()
    for col in NUMERIC_COLUMNS:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")
    for col in DATE_COLUMNS:
        merged[col] = pd.to_datetime(merged.get(col), errors="coerce")
    return merged.sort_values(["asof_date", "target_bucket", "buy_rank", "code"]).reset_index(drop=True)


def load_paper_positions(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["entry_date"] = pd.to_datetime(df.get("entry_date"), errors="coerce").dt.normalize()
    df["planned_exit_date"] = pd.to_datetime(df.get("planned_exit_date"), errors="coerce").dt.normalize()
    df["exit_date"] = pd.to_datetime(df.get("exit_date"), errors="coerce").dt.normalize()
    for col in ["entry_exec_price", "exit_exec_price", "entry_price_close", "exit_price_close"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def build_diff_frame(log_df: pd.DataFrame, prices: pd.DataFrame, paper_positions: pd.DataFrame, paper_horizon_days: int) -> pd.DataFrame:
    diff = log_df.copy()
    diff["asof_date"] = pd.to_datetime(diff["asof_date"], errors="coerce").dt.normalize()
    diff["actual_entry_date"] = pd.to_datetime(diff["actual_entry_time"], errors="coerce").dt.normalize()
    diff["actual_exit_date"] = pd.to_datetime(diff["actual_exit_time"], errors="coerce").dt.normalize()

    latest_price_rows = prices.sort_values(["code", "date"]).drop_duplicates(["code"], keep="last").rename(
        columns={"date": "latest_price_date", "close": "latest_close"}
    )
    diff = diff.merge(latest_price_rows[["code", "latest_price_date", "latest_close"]], on="code", how="left")

    entry_close_lookup = prices.rename(columns={"date": "actual_entry_date", "close": "actual_entry_close_ref"})
    exit_close_lookup = prices.rename(columns={"date": "actual_exit_date", "close": "paper_exit_close_ref"})
    diff = diff.merge(entry_close_lookup[["code", "actual_entry_date", "actual_entry_close_ref"]], on=["code", "actual_entry_date"], how="left")
    diff = diff.merge(exit_close_lookup[["code", "actual_exit_date", "paper_exit_close_ref"]], on=["code", "actual_exit_date"], how="left")

    if not paper_positions.empty:
        paper = paper_positions.copy()
        paper["target_bucket"] = paper["strategy"].astype(str).str.extract(r"top(\d+)")[0].astype("Int64")
        paper = paper.sort_values(["entry_date", "target_bucket", "code"]).drop_duplicates(
            ["entry_date", "target_bucket", "code"], keep="last"
        )
        diff = diff.merge(
            paper[
                [
                    "entry_date",
                    "target_bucket",
                    "code",
                    "entry_exec_price",
                    "entry_price_close",
                    "exit_exec_price",
                    "exit_price_close",
                    "exit_date",
                    "status",
                ]
            ].rename(
                columns={
                    "entry_date": "asof_date",
                    "entry_exec_price": "paper_entry_exec_price",
                    "entry_price_close": "paper_entry_close_ref",
                    "exit_exec_price": "paper_exit_exec_price",
                    "exit_price_close": "paper_exit_close_ref_ledger",
                    "exit_date": "paper_exit_date_ledger",
                    "status": "paper_position_status",
                }
            ),
            on=["asof_date", "target_bucket", "code"],
            how="left",
        )
        diff["paper_trade_source_mode"] = "paper_ledger"
        diff.loc[diff["paper_entry_exec_price"].isna(), "paper_trade_source_mode"] = "price_reference_proxy"
    else:
        diff["paper_trade_source_mode"] = "price_reference_proxy"
        diff["paper_entry_exec_price"] = pd.NA
        diff["paper_entry_close_ref"] = pd.NA
        diff["paper_exit_exec_price"] = pd.NA
        diff["paper_exit_close_ref_ledger"] = pd.NA
        diff["paper_exit_date_ledger"] = pd.NaT
        diff["paper_position_status"] = pd.NA

    paper_forward = attach_forward_outcomes(prices, horizon_days=paper_horizon_days).rename(
        columns={
            "date": "asof_date",
            "realized_return": f"paper_forward_return_{paper_horizon_days}d",
            "realized_mdd": f"paper_forward_mdd_{paper_horizon_days}d",
        }
    )
    diff = diff.merge(
        paper_forward[["code", "asof_date", f"paper_forward_return_{paper_horizon_days}d", f"paper_forward_mdd_{paper_horizon_days}d"]],
        on=["code", "asof_date"],
        how="left",
    )

    actual_entry_qty = pd.to_numeric(diff["actual_entry_qty"], errors="coerce")
    actual_exit_qty = pd.to_numeric(diff["actual_exit_qty"], errors="coerce")
    diff["filled_flag"] = diff["actual_entry_price"].notna() & actual_entry_qty.gt(0)
    diff["exit_filled_flag"] = diff["actual_exit_price"].notna() & actual_exit_qty.gt(0)
    diff["unfilled_flag"] = ~diff["filled_flag"]

    diff["paper_entry_price"] = pd.to_numeric(diff["paper_entry_exec_price"], errors="coerce")
    diff["paper_entry_price"] = diff["paper_entry_price"].fillna(pd.to_numeric(diff["recommendation_price_ref"], errors="coerce"))

    diff["entry_price_drift_pct"] = diff["actual_entry_price"] / diff["recommendation_price_ref"] - 1.0
    diff["entry_price_drift_bps"] = diff["entry_price_drift_pct"] * 10_000.0
    diff["slippage_bps"] = diff["entry_price_drift_bps"]
    diff["actual_vs_paper_entry_pct"] = diff["actual_entry_price"] / diff["paper_entry_price"] - 1.0
    diff["actual_vs_paper_entry_bps"] = diff["actual_vs_paper_entry_pct"] * 10_000.0

    diff["fill_delay_minutes"] = (
        pd.to_datetime(diff["actual_entry_time"], errors="coerce") - pd.to_datetime(diff["recommendation_generated_at"], errors="coerce")
    ).dt.total_seconds() / 60.0
    diff["fill_delay_days"] = (
        pd.to_datetime(diff["actual_entry_time"], errors="coerce").dt.normalize() - pd.to_datetime(diff["asof_date"], errors="coerce")
    ).dt.days

    diff["actual_realized_return"] = diff["actual_exit_price"] / diff["actual_entry_price"] - 1.0
    diff["actual_mtm_return_latest"] = diff["latest_close"] / diff["actual_entry_price"] - 1.0
    diff["paper_mtm_return_latest"] = diff["latest_close"] / diff["paper_entry_price"] - 1.0
    diff["actual_minus_paper_mtm_return"] = diff["actual_mtm_return_latest"] - diff["paper_mtm_return_latest"]

    diff["paper_realized_return_same_exit"] = diff["paper_exit_close_ref"] / diff["paper_entry_price"] - 1.0
    diff["actual_minus_paper_realized_return"] = diff["actual_realized_return"] - diff["paper_realized_return_same_exit"]

    diff["model_expected_hit_prob"] = diff["confidence_score"] / 100.0
    diff["actual_positive_outcome"] = pd.Series(pd.NA, index=diff.index, dtype="boolean")
    realized_mask = diff["actual_realized_return"].notna()
    diff.loc[realized_mask, "actual_positive_outcome"] = diff.loc[realized_mask, "actual_realized_return"].gt(0).astype("boolean")
    diff["actual_minus_expected_hit_prob"] = pd.to_numeric(diff["actual_positive_outcome"], errors="coerce") - diff["model_expected_hit_prob"]
    diff["model_expected_return_proxy_score"] = diff["ret_score"]

    diff.loc[
        ~diff["filled_flag"],
        ["entry_price_drift_pct", "entry_price_drift_bps", "slippage_bps", "actual_vs_paper_entry_pct", "actual_vs_paper_entry_bps", "fill_delay_minutes", "fill_delay_days"],
    ] = pd.NA
    diff.loc[
        ~diff["exit_filled_flag"],
        ["actual_realized_return", "paper_realized_return_same_exit", "actual_minus_paper_realized_return", "actual_positive_outcome", "actual_minus_expected_hit_prob"],
    ] = pd.NA

    ordered = [
        "recommendation_id",
        "asof_date",
        "code",
        "name",
        "target_bucket",
        "buy_rank",
        "gate_overall_status",
        "gate_bucket_status",
        "final_score",
        "confidence_score",
        "model_expected_return_proxy_score",
        "recommendation_price_ref",
        "actual_entry_status",
        "actual_entry_price",
        "actual_entry_time",
        "actual_entry_qty",
        "filled_flag",
        "unfilled_flag",
        "entry_price_drift_pct",
        "entry_price_drift_bps",
        "slippage_bps",
        "fill_delay_minutes",
        "fill_delay_days",
        "paper_trade_source_mode",
        "paper_entry_price",
        "actual_vs_paper_entry_pct",
        "actual_vs_paper_entry_bps",
        "actual_exit_status",
        "actual_exit_price",
        "actual_exit_time",
        "actual_exit_qty",
        "exit_filled_flag",
        "actual_realized_return",
        f"paper_forward_return_{paper_horizon_days}d",
        "paper_realized_return_same_exit",
        "actual_minus_paper_realized_return",
        "actual_mtm_return_latest",
        "paper_mtm_return_latest",
        "actual_minus_paper_mtm_return",
        "model_expected_hit_prob",
        "actual_positive_outcome",
        "actual_minus_expected_hit_prob",
        "latest_price_date",
        "latest_close",
    ]
    return diff[ordered].sort_values(["asof_date", "target_bucket", "buy_rank", "code"]).reset_index(drop=True)


def build_report(log_df: pd.DataFrame, diff_df: pd.DataFrame, paper_horizon_days: int) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = int(len(diff_df))
    filled = int(diff_df["filled_flag"].fillna(False).sum()) if not diff_df.empty else 0
    exits = int(diff_df["exit_filled_flag"].fillna(False).sum()) if not diff_df.empty else 0
    unfilled = int(diff_df["unfilled_flag"].fillna(False).sum()) if not diff_df.empty else 0

    avg_slippage_bps = pd.to_numeric(diff_df.loc[diff_df["filled_flag"].fillna(False), "slippage_bps"], errors="coerce").mean()
    avg_fill_delay_min = pd.to_numeric(diff_df.loc[diff_df["filled_flag"].fillna(False), "fill_delay_minutes"], errors="coerce").mean()
    avg_actual_return = pd.to_numeric(diff_df["actual_realized_return"], errors="coerce").mean()
    avg_paper_same_exit = pd.to_numeric(diff_df["paper_realized_return_same_exit"], errors="coerce").mean()
    avg_return_gap = pd.to_numeric(diff_df["actual_minus_paper_realized_return"], errors="coerce").mean()
    avg_mtm_gap = pd.to_numeric(diff_df["actual_minus_paper_mtm_return"], errors="coerce").mean()

    worst_slippage = diff_df.loc[diff_df["filled_flag"].fillna(False)].copy()
    if not worst_slippage.empty:
        worst_slippage = worst_slippage.sort_values("slippage_bps", ascending=False).head(10)
        worst_slippage["actual_entry_price"] = worst_slippage["actual_entry_price"].map(_fmt_num)
        worst_slippage["recommendation_price_ref"] = worst_slippage["recommendation_price_ref"].map(_fmt_num)
        worst_slippage["slippage_bps"] = worst_slippage["slippage_bps"].map(lambda x: _fmt_num(x, 1))
        worst_slippage["entry_price_drift_pct"] = worst_slippage["entry_price_drift_pct"].map(_fmt_pct)

    unfilled_table = diff_df.loc[diff_df["unfilled_flag"].fillna(False), ["asof_date", "code", "name", "target_bucket", "buy_rank", "gate_bucket_status"]].copy()
    if not unfilled_table.empty:
        unfilled_table["asof_date"] = pd.to_datetime(unfilled_table["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    realized_table = diff_df.loc[diff_df["exit_filled_flag"].fillna(False)].copy()
    if not realized_table.empty:
        realized_table = realized_table.sort_values("actual_minus_paper_realized_return").head(10)
        for col in ["actual_realized_return", "paper_realized_return_same_exit", "actual_minus_paper_realized_return"]:
            realized_table[col] = realized_table[col].map(_fmt_pct)
        realized_table["model_expected_hit_prob"] = realized_table["model_expected_hit_prob"].map(_fmt_pct)

    mtm_table = diff_df.loc[diff_df["filled_flag"].fillna(False)].copy()
    if not mtm_table.empty:
        mtm_table = mtm_table.sort_values("actual_minus_paper_mtm_return").head(10)
        for col in ["actual_mtm_return_latest", "paper_mtm_return_latest", "actual_minus_paper_mtm_return"]:
            mtm_table[col] = mtm_table[col].map(_fmt_pct)

    latest_asof = pd.to_datetime(log_df["asof_date"], errors="coerce").max()
    latest_rows = diff_df.loc[pd.to_datetime(diff_df["asof_date"], errors="coerce").eq(latest_asof)].copy() if not diff_df.empty and pd.notna(latest_asof) else pd.DataFrame()
    if not latest_rows.empty:
        latest_rows["asof_date"] = pd.to_datetime(latest_rows["asof_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        latest_rows["recommendation_price_ref"] = latest_rows["recommendation_price_ref"].map(_fmt_num)
        latest_rows["actual_entry_price"] = latest_rows["actual_entry_price"].map(_fmt_num)
        latest_rows["slippage_bps"] = latest_rows["slippage_bps"].map(lambda x: _fmt_num(x, 1))

    lines = [
        "# Real Execution Shadow Report",
        "",
        f"- generated_at: {generated_at}",
        f"- recommendation_rows: {total}",
        f"- actual_entry_filled_rows: {filled}",
        f"- actual_exit_filled_rows: {exits}",
        f"- unfilled_rows: {unfilled}",
        f"- avg_slippage_bps: {_fmt_num(avg_slippage_bps, 1)}",
        f"- avg_fill_delay_minutes: {_fmt_num(avg_fill_delay_min, 1)}",
        f"- avg_actual_realized_return: {_fmt_pct(avg_actual_return)}",
        f"- avg_paper_realized_return_same_exit: {_fmt_pct(avg_paper_same_exit)}",
        f"- avg_actual_minus_paper_realized_return: {_fmt_pct(avg_return_gap)}",
        f"- avg_actual_minus_paper_mtm_return: {_fmt_pct(avg_mtm_gap)}",
        f"- paper_forward_proxy_horizon_days: {paper_horizon_days}",
        "",
        "## Coverage",
        "- `real_trade_log.csv` keeps recommendation snapshots plus manual actual execution fields.",
        "- `real_vs_model_diff.csv` computes price drift, fill gaps, slippage, realized return, and paper/reference comparison.",
        "- `model expected vs actual` uses `ret_score` and `confidence_score` as proxies because the live payload does not emit an explicit expected return number.",
        "",
    ]

    if not latest_rows.empty:
        lines.extend(
            [
                "## Latest Recommendation Set",
                _markdown_table(
                    latest_rows[["asof_date", "code", "name", "buy_rank", "gate_bucket_status", "recommendation_price_ref", "actual_entry_price", "slippage_bps"]],
                    ["asof_date", "code", "name", "buy_rank", "gate_bucket_status", "recommendation_price_ref", "actual_entry_price", "slippage_bps"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Execution Drift",
            _markdown_table(
                worst_slippage[["asof_date", "code", "name", "actual_entry_price", "recommendation_price_ref", "entry_price_drift_pct", "slippage_bps"]]
                if not worst_slippage.empty
                else pd.DataFrame(),
                ["asof_date", "code", "name", "actual_entry_price", "recommendation_price_ref", "entry_price_drift_pct", "slippage_bps"],
            ),
            "",
            "## Unfilled Recommendations",
            _markdown_table(unfilled_table, ["asof_date", "code", "name", "target_bucket", "buy_rank", "gate_bucket_status"]),
            "",
            "## Realized Return vs Paper",
            _markdown_table(
                realized_table[
                    ["asof_date", "code", "name", "actual_realized_return", "paper_realized_return_same_exit", "actual_minus_paper_realized_return", "model_expected_hit_prob"]
                ]
                if not realized_table.empty
                else pd.DataFrame(),
                ["asof_date", "code", "name", "actual_realized_return", "paper_realized_return_same_exit", "actual_minus_paper_realized_return", "model_expected_hit_prob"],
            ),
            "",
            "## Open MTM vs Paper",
            _markdown_table(
                mtm_table[["asof_date", "code", "name", "actual_mtm_return_latest", "paper_mtm_return_latest", "actual_minus_paper_mtm_return"]]
                if not mtm_table.empty
                else pd.DataFrame(),
                ["asof_date", "code", "name", "actual_mtm_return_latest", "paper_mtm_return_latest", "actual_minus_paper_mtm_return"],
            ),
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    recommendations = load_json(args.recommendations_json)
    model_portfolio = load_json(args.model_portfolio_json)
    buy_gate = load_json(args.buy_gate_json)
    snapshot_archive = load_snapshot_archive(args.snapshot_archive_csv)
    prices = load_price_history(prices_csv=_resolve(args.prices_csv))
    paper_positions = load_paper_positions(args.paper_positions_csv)

    recommendation_df = build_recommendation_frame(
        recommendations=recommendations,
        model_portfolio=model_portfolio,
        buy_gate=buy_gate,
        snapshot_archive=snapshot_archive,
        prices=prices,
    )
    existing_log = load_existing_log(args.out_real_log_csv)
    real_log = merge_real_log(recommendation_df, existing_log)
    diff_df = build_diff_frame(real_log, prices, paper_positions, args.paper_horizon_days)

    out_real_log_csv = _resolve(args.out_real_log_csv)
    out_diff_csv = _resolve(args.out_diff_csv)
    out_report_md = _resolve(args.out_report_md)
    out_real_log_csv.parent.mkdir(parents=True, exist_ok=True)
    out_diff_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report_md.parent.mkdir(parents=True, exist_ok=True)

    real_log.to_csv(out_real_log_csv, index=False, encoding="utf-8-sig")
    diff_df.to_csv(out_diff_csv, index=False, encoding="utf-8-sig")
    out_report_md.write_text(build_report(real_log, diff_df, args.paper_horizon_days), encoding="utf-8")

    print(f"real_trade_log_csv: {out_real_log_csv}")
    print(f"real_vs_model_diff_csv: {out_diff_csv}")
    print(f"real_execution_report_md: {out_report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
