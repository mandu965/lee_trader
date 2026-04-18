from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_CANDIDATES_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_GATE_JSON = OUTPUT_DIR / "operational_buy_gate.json"
DEFAULT_SNAPSHOT_ARCHIVE_CSV = DATA_DIR / "ranking_snapshot_archive.csv"
DEFAULT_REPORT_MD = OUTPUT_DIR / "execution_policy_simulation_report.md"

AUTO_HOLDINGS_PATHS = [
    DATA_DIR / "current_holdings.csv",
    DATA_DIR / "live_account_holdings.csv",
    DATA_DIR / "paper_trading_positions.csv",
]

POLICY_VERSION = str(
    get_production_config_value(["metadata", "execution_policy_version"], "execution_policy_v1")
)
SCORE_FORMULA_VERSION = str(
    get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v8_return_prob_tech_regime")
)
GATE_VERSION = str(
    get_production_config_value(["metadata", "gate_version"], "operational_buy_gate_v1")
)
PORTFOLIO_VERSION = str(
    get_production_config_value(["metadata", "portfolio_version"], "model_portfolio_constructor_v1")
)

STATUS_BUY_ALLOWED = "BUY_ALLOWED"
STATUS_PILOT = "PILOT"
STATUS_WATCH = "WATCH"
STATUS_HOLD = "HOLD"
STATUS_BLOCK = "BLOCK"


@dataclass
class HoldingsContext:
    open_positions: pd.DataFrame
    closed_positions: pd.DataFrame
    source: str


@dataclass(frozen=True)
class ConfidencePolicy:
    band: str
    label: str
    entry_allowed: bool
    hold_allowed: bool
    weight_scale: float
    position_cap_scale: float
    guidance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply operational execution policy to recommended stocks.")
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--snapshot-archive-csv", type=Path, default=DEFAULT_SNAPSHOT_ARCHIVE_CSV)
    parser.add_argument("--holdings-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT_MD)
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


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
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


def _safe_read_json(path: Path) -> dict[str, object]:
    resolved = _resolve(path)
    if resolved is None or not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def _normalize_theme(series: pd.Series) -> pd.Series:
    return series.fillna("(none)").astype(str).replace({"": "(none)", "nan": "(none)"})


def markdown_summary_list(items: Iterable[str]) -> str:
    values = [item for item in items if item]
    if not values:
        return "- none"
    return "\n".join(f"- {item}" for item in values)


def load_candidates(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        raise ValueError(f"candidate file is empty: {path}")
    work = df.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["dominant_theme"] = _normalize_theme(work.get("dominant_theme", pd.Series("(none)", index=work.index)))
    for col in [
        "buy_rank",
        "rank_final",
        "rank_source",
        "final_score",
        "confidence_score",
        "liquidity_score",
        "theme_score",
        "trading_value",
        "close",
        "volume",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
    ]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    if "trading_value" not in work.columns or work["trading_value"].isna().all():
        work["trading_value"] = pd.to_numeric(work.get("close"), errors="coerce") * pd.to_numeric(work.get("volume"), errors="coerce")
    work["recent_surge_soft_flag"] = work.get("recent_surge_soft_flag", pd.Series(False, index=work.index)).astype(str).str.lower().isin(["true", "1"])
    work["selection_stage"] = work.get("selection_stage", pd.Series("", index=work.index)).fillna("").astype(str)
    if "buy_rank" not in work.columns or work["buy_rank"].isna().all():
        work["buy_rank"] = pd.to_numeric(work.get("rank_final"), errors="coerce")
    if work["buy_rank"].isna().all():
        work["buy_rank"] = pd.to_numeric(work.get("rank_source"), errors="coerce")
    if work["buy_rank"].isna().all():
        work["buy_rank"] = (
            pd.to_numeric(work["final_score"], errors="coerce")
            .rank(method="first", ascending=False)
            .astype(float)
        )
    work["buy_rank"] = pd.to_numeric(work["buy_rank"], errors="coerce")
    if "asof_date" in work.columns and work["asof_date"].notna().any():
        work["asof_date"] = work.get("asof_date", "").fillna("").astype(str)
    elif "date" in work.columns:
        parsed_date = pd.to_datetime(work.get("date"), errors="coerce")
        work["asof_date"] = parsed_date.dt.strftime("%Y-%m-%d").fillna("")
    else:
        work["asof_date"] = ""
    return work.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)


def load_snapshot_archive(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if resolved is None or not resolved.exists():
        return pd.DataFrame()
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce").dt.normalize()
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
    df["confidence_score"] = pd.to_numeric(df.get("confidence_score"), errors="coerce")
    df["dominant_theme"] = _normalize_theme(df.get("dominant_theme", pd.Series("(none)", index=df.index)))
    if "sector" not in df.columns:
        df["sector"] = "(unknown)"
    else:
        df["sector"] = df["sector"].fillna("(unknown)").astype(str)
    return df.dropna(subset=["asof_date"]).sort_values(["asof_date", "rank", "code"]).reset_index(drop=True)


def build_reference_maps(snapshot_archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshot_archive.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = sorted(snapshot_archive["asof_date"].dropna().unique().tolist())
    latest_date = dates[-1]
    latest = snapshot_archive.loc[snapshot_archive["asof_date"].eq(latest_date)].copy()
    previous = pd.DataFrame()
    if len(dates) >= 2:
        previous = snapshot_archive.loc[snapshot_archive["asof_date"].eq(dates[-2])].copy()
    return latest.reset_index(drop=True), previous.reset_index(drop=True)


def load_holdings(args: argparse.Namespace, candidates: pd.DataFrame, latest_snapshot: pd.DataFrame) -> HoldingsContext:
    candidate_paths: list[Path] = []
    explicit_holdings_provided = args.holdings_csv is not None
    if args.holdings_csv is not None:
        candidate_paths.append(args.holdings_csv)
    if not explicit_holdings_provided:
        candidate_paths.extend(AUTO_HOLDINGS_PATHS)

    candidate_lookup = candidates.drop_duplicates("code").set_index("code")
    latest_lookup = latest_snapshot.drop_duplicates("code").set_index("code") if not latest_snapshot.empty else pd.DataFrame()

    for path in candidate_paths:
        resolved = _resolve(path)
        if resolved is None or not resolved.exists():
            if explicit_holdings_provided and path == args.holdings_csv:
                return HoldingsContext(open_positions=pd.DataFrame(), closed_positions=pd.DataFrame(columns=["code", "name", "exit_date"]), source=str(path))
            continue
        df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
        if df.empty or "code" not in df.columns:
            if explicit_holdings_provided and path == args.holdings_csv:
                return HoldingsContext(open_positions=pd.DataFrame(), closed_positions=pd.DataFrame(columns=["code", "name", "exit_date"]), source=str(path))
            continue

        work = df.copy()
        work["code"] = work["code"].astype(str).str.zfill(6)

        if "status" in work.columns:
            work["status"] = work["status"].fillna("").astype(str).str.upper()
            open_positions = work.loc[work["status"].eq("OPEN")].copy()
            closed_positions = work.loc[work["status"].eq("CLOSED")].copy()
        else:
            open_positions = work.copy()
            closed_positions = work.iloc[0:0].copy()

        for frame in [open_positions, closed_positions]:
            if frame.empty:
                continue
            if "name" not in frame.columns:
                frame["name"] = ""
            if "current_weight" in frame.columns:
                frame["weight"] = pd.to_numeric(frame["current_weight"], errors="coerce")
            elif "weight" in frame.columns:
                frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
            elif "entry_notional_gross" in frame.columns:
                notionals = pd.to_numeric(frame["entry_notional_gross"], errors="coerce")
                total = notionals.sum()
                frame["weight"] = notionals / total if total and pd.notna(total) else pd.NA
            else:
                frame["weight"] = pd.NA
            if frame["weight"].isna().all() and len(frame) > 0:
                frame["weight"] = 1.0 / len(frame)

            frame["sector"] = frame.get("sector", pd.Series(index=frame.index, dtype="object"))
            frame["dominant_theme"] = frame.get("dominant_theme", pd.Series(index=frame.index, dtype="object"))
            frame["confidence_score"] = pd.to_numeric(frame.get("confidence_score"), errors="coerce")
            frame["final_score"] = pd.to_numeric(frame.get("final_score"), errors="coerce")
            exit_series = frame.get("exit_date")
            if exit_series is None:
                frame["exit_date"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
            else:
                frame["exit_date"] = pd.to_datetime(exit_series, errors="coerce").dt.normalize()

            for idx, row in frame.iterrows():
                code = row["code"]
                if pd.isna(row.get("sector")) or str(row.get("sector", "")).strip() == "":
                    if code in candidate_lookup.index:
                        frame.at[idx, "sector"] = candidate_lookup.at[code, "sector"]
                    elif not latest_lookup.empty and code in latest_lookup.index:
                        frame.at[idx, "sector"] = latest_lookup.at[code, "sector"]
                    else:
                        frame.at[idx, "sector"] = "(unknown)"
                if pd.isna(row.get("dominant_theme")) or str(row.get("dominant_theme", "")).strip() == "":
                    if code in candidate_lookup.index:
                        frame.at[idx, "dominant_theme"] = candidate_lookup.at[code, "dominant_theme"]
                    elif not latest_lookup.empty and code in latest_lookup.index:
                        frame.at[idx, "dominant_theme"] = latest_lookup.at[code, "dominant_theme"]
                    else:
                        frame.at[idx, "dominant_theme"] = "(none)"
                if pd.isna(row.get("name")) or str(row.get("name", "")).strip() == "":
                    if code in candidate_lookup.index:
                        frame.at[idx, "name"] = candidate_lookup.at[code, "name"]
                    elif not latest_lookup.empty and code in latest_lookup.index:
                        frame.at[idx, "name"] = latest_lookup.at[code, "name"]

        source = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)
        return HoldingsContext(
            open_positions=open_positions.reset_index(drop=True),
            closed_positions=closed_positions.reset_index(drop=True),
            source=source,
        )

    empty = pd.DataFrame(columns=["code", "name", "weight", "sector", "dominant_theme", "confidence_score", "final_score"])
    return HoldingsContext(open_positions=empty, closed_positions=pd.DataFrame(columns=["code", "name", "exit_date"]), source="(none)")


def business_days_since(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    if pd.isna(start_date) or pd.isna(end_date) or start_date >= end_date:
        return 0
    return max(len(pd.bdate_range(start_date + pd.Timedelta(days=1), end_date)), 0)


def extract_cooldown_map(closed_positions: pd.DataFrame, asof_date: pd.Timestamp, cooldown_days: int) -> dict[str, dict[str, object]]:
    cooldowns: dict[str, dict[str, object]] = {}
    if closed_positions.empty:
        return cooldowns
    work = closed_positions.dropna(subset=["code", "exit_date"]).copy()
    if work.empty:
        return cooldowns
    latest_exit = work.sort_values(["code", "exit_date"]).groupby("code", as_index=False).tail(1)
    for _, row in latest_exit.iterrows():
        days_since = business_days_since(pd.Timestamp(row["exit_date"]).normalize(), asof_date)
        remaining = max(cooldown_days - days_since, 0)
        cooldowns[str(row["code"]).zfill(6)] = {
            "exit_date": pd.Timestamp(row["exit_date"]).normalize(),
            "days_since_exit": days_since,
            "cooldown_remaining": remaining,
            "active": remaining > 0,
        }
    return cooldowns


def gate_decision(gate_payload: dict[str, object]) -> dict[str, object]:
    status = str(gate_payload.get("overall_status") or STATUS_HOLD).upper()
    primary_bucket = int(gate_payload.get("primary_bucket") or 5)
    reason = ""
    for decision in gate_payload.get("decisions", []):
        if isinstance(decision, dict) and int(decision.get("bucket") or -1) == primary_bucket:
            reason = str(decision.get("reason_summary") or "")
            break

    if status == STATUS_BUY_ALLOWED:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": True,
            "allow_score_replacement": True,
            "allow_risk_trim": True,
            "guidance": "신규 진입, 교체, 목표 비중 재조정 허용",
        }
    if status == STATUS_WATCH:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": False,
            "allow_score_replacement": False,
            "allow_risk_trim": True,
            "guidance": "신규 진입은 보류하고 위험 축소 목적 행동만 허용",
        }
    if status == STATUS_BLOCK:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": False,
            "allow_score_replacement": False,
            "allow_risk_trim": True,
            "guidance": "신규 진입 금지, cap 위반과 약한 포지션부터 축소 또는 청산",
        }
    return {
        "status": STATUS_HOLD,
        "reason": reason,
        "allow_new_entries": False,
        "allow_score_replacement": False,
        "allow_risk_trim": True,
        "guidance": "신규 진입 금지, 기존 보유 유지 또는 비중 축소만 허용",
    }

def gate_decision_runtime(gate_payload: dict[str, object]) -> dict[str, object]:
    status = str(gate_payload.get("overall_status") or STATUS_HOLD).upper()
    primary_bucket = int(gate_payload.get("primary_bucket") or 5)
    reason = ""
    for decision in gate_payload.get("decisions", []):
        if isinstance(decision, dict) and int(decision.get("bucket") or -1) == primary_bucket:
            reason = str(decision.get("reason_summary") or "")
            break

    if status == STATUS_BUY_ALLOWED:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": True,
            "allow_score_replacement": True,
            "allow_risk_trim": True,
            "limited_entry_mode": None,
            "guidance": "정식 자동매수: 신규 진입, 교체, 목표 비중 재조정 허용",
        }
    if status == STATUS_PILOT:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": False,
            "allow_score_replacement": False,
            "allow_risk_trim": True,
            "limited_entry_mode": "pilot",
            "guidance": "PILOT 제한 실운용: 제한된 신규 진입만 허용하고 교체매매는 보류",
        }
    if status == STATUS_WATCH:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": False,
            "allow_score_replacement": False,
            "allow_risk_trim": True,
            "limited_entry_mode": "watch",
            "guidance": "WATCH 제한 진입: 소액 신규 진입만 허용하고 위험 축소는 계속 허용",
        }
    if status == STATUS_BLOCK:
        return {
            "status": status,
            "reason": reason,
            "allow_new_entries": False,
            "allow_score_replacement": False,
            "allow_risk_trim": True,
            "limited_entry_mode": None,
            "guidance": "신규 진입 금지, cap 위반과 과도한 위험 포지션만 축소 또는 청산",
        }
    return {
        "status": STATUS_HOLD,
        "reason": reason,
        "allow_new_entries": False,
        "allow_score_replacement": False,
        "allow_risk_trim": True,
        "limited_entry_mode": None,
        "guidance": "신규 진입 금지, 기존 보유 유지 또는 비중 축소만 허용",
    }


def resolve_confidence_policy(confidence_value: object, args: argparse.Namespace) -> ConfidencePolicy:
    confidence = pd.to_numeric(confidence_value, errors="coerce")
    if pd.isna(confidence):
        return ConfidencePolicy(
            band="UNKNOWN",
            label="미확인",
            entry_allowed=False,
            hold_allowed=False,
            weight_scale=0.0,
            position_cap_scale=0.0,
            guidance="신뢰도 정보가 없어 신규 진입 금지",
        )

    value = float(confidence)
    if value < args.confidence_block_below:
        return ConfidencePolicy(
            band="BLOCKED",
            label="진입 금지",
            entry_allowed=False,
            hold_allowed=False,
            weight_scale=0.0,
            position_cap_scale=0.0,
            guidance=f"confidence < {_fmt_num(args.confidence_block_below)}: 신규 진입 금지, 기존 보유도 축소/청산 우선",
        )
    if value < args.confidence_reduced_below:
        return ConfidencePolicy(
            band="REDUCED",
            label="소액",
            entry_allowed=True,
            hold_allowed=True,
            weight_scale=args.confidence_reduced_weight_scale,
            position_cap_scale=args.confidence_reduced_position_cap_scale,
            guidance=f"{_fmt_num(args.confidence_block_below)}~{_fmt_num(args.confidence_reduced_below)}: 소액 진입만 허용",
        )
    if value < args.confidence_standard_below:
        return ConfidencePolicy(
            band="STANDARD",
            label="표준",
            entry_allowed=True,
            hold_allowed=True,
            weight_scale=args.confidence_standard_weight_scale,
            position_cap_scale=args.confidence_standard_position_cap_scale,
            guidance=f"{_fmt_num(args.confidence_reduced_below)}~{_fmt_num(args.confidence_standard_below)}: 표준 비중 허용",
        )
    return ConfidencePolicy(
        band="EXPANDED",
        label="확대 가능",
        entry_allowed=True,
        hold_allowed=True,
        weight_scale=args.confidence_expanded_weight_scale,
        position_cap_scale=args.confidence_expanded_position_cap_scale,
        guidance=f"{_fmt_num(args.confidence_standard_below)}+: 확대 비중 검토 가능",
    )


def score_for_weight(row: pd.Series, args: argparse.Namespace) -> float:
    final_score = float(pd.to_numeric(row.get("final_score"), errors="coerce") or 0.0)
    confidence = float(pd.to_numeric(row.get("confidence_score"), errors="coerce") or 0.0)
    liquidity = float(pd.to_numeric(row.get("liquidity_score"), errors="coerce") or 0.0)
    trading_value = float(pd.to_numeric(row.get("trading_value"), errors="coerce") or 0.0)
    confidence_policy = resolve_confidence_policy(confidence, args)
    base = (final_score / 100.0) ** 1.10 * (confidence / 100.0) ** 1.20
    liquidity_mult = 1.0
    if liquidity < 10.0 or (trading_value > 0 and trading_value < 4_000_000_000.0):
        liquidity_mult = 0.55
    elif liquidity < 20.0 or (trading_value > 0 and trading_value < 8_000_000_000.0):
        liquidity_mult = 0.80
    confidence_boost = 1.0 + max(confidence - 80.0, 0.0) / 200.0
    weighted = base * liquidity_mult * confidence_boost * max(confidence_policy.weight_scale, 0.0)
    return max(weighted, 1e-9) if confidence_policy.entry_allowed else 1e-9


def compute_target_weights(selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if selected.empty:
        result = selected.copy()
        result["target_weight"] = pd.Series(dtype="float64")
        return result

    work = selected.copy().reset_index(drop=True)
    work["confidence_band"] = work["confidence_score"].map(lambda value: resolve_confidence_policy(value, args).band)
    work["confidence_guidance"] = work["confidence_score"].map(lambda value: resolve_confidence_policy(value, args).guidance)
    work["confidence_position_cap"] = work["confidence_score"].map(
        lambda value: min(args.max_position_weight * resolve_confidence_policy(value, args).position_cap_scale, 1.0)
    )
    work["policy_score"] = work.apply(lambda row: score_for_weight(row, args), axis=1)
    work["target_weight"] = 0.0
    remaining = max(1.0 - args.cash_minimum, 0.0)

    sector_alloc: dict[str, float] = {}
    theme_alloc: dict[str, float] = {}
    eligible = work.index.tolist()

    while remaining > 1e-8 and eligible:
        total_score = float(work.loc[eligible, "policy_score"].sum())
        if total_score <= 0:
            break
        advanced = False
        next_eligible: list[int] = []
        for idx in eligible:
            row = work.loc[idx]
            sector = str(row.get("sector") or "(unknown)")
            theme = str(row.get("dominant_theme") or "(none)")
            current = float(work.at[idx, "target_weight"])
            sector_room = max(args.sector_cap - sector_alloc.get(sector, 0.0), 0.0)
            theme_room = 1.0 if theme == "(none)" else max(args.theme_cap - theme_alloc.get(theme, 0.0), 0.0)
            confidence_position_cap = float(pd.to_numeric(row.get("confidence_position_cap"), errors="coerce") or 0.0)
            name_room = max(confidence_position_cap - current, 0.0)
            cap_room = min(name_room, sector_room, theme_room, remaining)
            if cap_room <= 1e-8:
                continue
            desired = remaining * (float(row["policy_score"]) / total_score)
            allocation = min(desired, cap_room)
            if allocation <= 1e-8:
                next_eligible.append(idx)
                continue
            work.at[idx, "target_weight"] = current + allocation
            sector_alloc[sector] = sector_alloc.get(sector, 0.0) + allocation
            if theme != "(none)":
                theme_alloc[theme] = theme_alloc.get(theme, 0.0) + allocation
            remaining -= allocation
            advanced = True
            if work.at[idx, "target_weight"] + 1e-8 < confidence_position_cap and cap_room - allocation > 1e-8:
                next_eligible.append(idx)
        if not advanced:
            break
        eligible = next_eligible

    return work


def candidate_eligibility_notes(row: pd.Series, cooldown_map: dict[str, dict[str, object]], args: argparse.Namespace) -> tuple[bool, list[str]]:
    notes: list[str] = []
    eligible = True
    rank_value = pd.to_numeric(row.get("buy_rank"), errors="coerce")
    confidence = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    liquidity = pd.to_numeric(row.get("liquidity_score"), errors="coerce")
    trading_value = pd.to_numeric(row.get("trading_value"), errors="coerce")
    confidence_policy = resolve_confidence_policy(confidence, args)
    if pd.isna(rank_value) or float(rank_value) > args.entry_review_extended_top_n:
        eligible = False
        notes.append(f"outside entry review range top{args.entry_review_extended_top_n}")
    elif float(rank_value) <= args.entry_review_top_n:
        if pd.isna(confidence) or float(confidence) < args.standard_entry_confidence:
            eligible = False
            notes.append(
                f"rank<=top{args.entry_review_top_n} requires confidence >= {_fmt_num(args.standard_entry_confidence)}"
            )
    else:
        if pd.isna(confidence) or float(confidence) < args.extended_entry_confidence:
            eligible = False
            notes.append(
                f"rank<=top{args.entry_review_extended_top_n} extended entry requires confidence >= {_fmt_num(args.extended_entry_confidence)}"
            )
    if pd.isna(confidence) or float(confidence) < args.min_entry_confidence:
        eligible = False
        notes.append(f"confidence {_fmt_num(confidence)} < minimum entry floor {_fmt_num(args.min_entry_confidence)}")
    if not confidence_policy.entry_allowed:
        eligible = False
        notes.append(confidence_policy.guidance)
    else:
        notes.append(f"confidence band {confidence_policy.band}: {confidence_policy.label}")
    if pd.isna(liquidity) or float(liquidity) < args.min_liquidity_score:
        eligible = False
        notes.append(f"liquidity {_fmt_num(liquidity)} < floor {_fmt_num(args.min_liquidity_score)}")
    if pd.isna(trading_value) or float(trading_value) < args.min_trading_value:
        eligible = False
        notes.append(f"trading_value {_fmt_num(trading_value, 0)} < floor {_fmt_num(args.min_trading_value, 0)}")
    cooldown = cooldown_map.get(str(row["code"]).zfill(6))
    if cooldown and bool(cooldown.get("active")):
        eligible = False
        notes.append(f"re-entry cooldown {int(cooldown['cooldown_remaining'])}d remaining")
    return eligible, notes


def enrich_candidates_with_snapshot(candidates: pd.DataFrame, latest_snapshot: pd.DataFrame, previous_snapshot: pd.DataFrame) -> pd.DataFrame:
    work = candidates.copy()
    latest_lookup = latest_snapshot.set_index("code") if not latest_snapshot.empty else pd.DataFrame()
    previous_lookup = previous_snapshot.set_index("code") if not previous_snapshot.empty else pd.DataFrame()
    work["latest_snapshot_rank"] = pd.NA
    work["latest_snapshot_final_score"] = pd.NA
    work["latest_snapshot_confidence_score"] = pd.NA
    work["previous_snapshot_rank"] = pd.NA
    work["score_drift_vs_latest_snapshot"] = pd.NA

    for idx, row in work.iterrows():
        code = row["code"]
        if not latest_lookup.empty and code in latest_lookup.index:
            work.at[idx, "latest_snapshot_rank"] = latest_lookup.at[code, "rank"]
            work.at[idx, "latest_snapshot_final_score"] = latest_lookup.at[code, "final_score"]
            work.at[idx, "latest_snapshot_confidence_score"] = latest_lookup.at[code, "confidence_score"]
            work.at[idx, "score_drift_vs_latest_snapshot"] = pd.to_numeric(latest_lookup.at[code, "final_score"], errors="coerce") - pd.to_numeric(row.get("final_score"), errors="coerce")
        if not previous_lookup.empty and code in previous_lookup.index:
            work.at[idx, "previous_snapshot_rank"] = previous_lookup.at[code, "rank"]
    return work


def classify_holdings(
    holdings: pd.DataFrame,
    candidates: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    gate: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=["code", "name", "current_weight", "action", "reason"])

    candidate_lookup = candidates.set_index("code")
    latest_lookup = latest_snapshot.set_index("code") if not latest_snapshot.empty else pd.DataFrame()
    sector_weights = holdings.groupby("sector", dropna=False)["weight"].sum().to_dict() if "weight" in holdings.columns else {}
    theme_weights = holdings.groupby("dominant_theme", dropna=False)["weight"].sum().to_dict() if "weight" in holdings.columns else {}

    rows: list[dict[str, object]] = []
    for _, row in holdings.iterrows():
        code = str(row["code"]).zfill(6)
        current_weight = pd.to_numeric(row.get("weight"), errors="coerce")
        sector = str(row.get("sector") or "(unknown)")
        theme = str(row.get("dominant_theme") or "(none)")
        candidate_rank = pd.NA
        candidate_score = pd.NA
        latest_rank = pd.NA
        confidence = pd.to_numeric(row.get("confidence_score"), errors="coerce")
        reasons: list[str] = []
        action = "HOLD"
        confidence_policy = resolve_confidence_policy(confidence, args)
        policy_cap_weight = min(args.max_position_weight * confidence_policy.position_cap_scale, 1.0)

        if code in candidate_lookup.index:
            candidate_rank = candidate_lookup.at[code, "buy_rank"]
            candidate_score = candidate_lookup.at[code, "final_score"]
            confidence = pd.to_numeric(candidate_lookup.at[code, "confidence_score"], errors="coerce")
            confidence_policy = resolve_confidence_policy(confidence, args)
            policy_cap_weight = min(args.max_position_weight * confidence_policy.position_cap_scale, 1.0)
        if not latest_lookup.empty and code in latest_lookup.index:
            latest_rank = latest_lookup.at[code, "rank"]
            if pd.isna(confidence):
                confidence = pd.to_numeric(latest_lookup.at[code, "confidence_score"], errors="coerce")
                confidence_policy = resolve_confidence_policy(confidence, args)
                policy_cap_weight = min(args.max_position_weight * confidence_policy.position_cap_scale, 1.0)

        if pd.notna(confidence) and float(confidence) < args.force_exit_confidence:
            action = "EXIT_CANDIDATE"
            reasons.append(f"confidence {_fmt_num(confidence)} < force exit floor {_fmt_num(args.force_exit_confidence)}")
        elif pd.notna(confidence) and float(confidence) < args.min_hold_confidence:
            action = "REPLACE_CANDIDATE"
            reasons.append(f"confidence {_fmt_num(confidence)} < hold floor {_fmt_num(args.min_hold_confidence)}")
        if pd.notna(current_weight) and float(current_weight) > args.max_position_weight + 1e-8:
            action = "TRIM"
            reasons.append(f"position cap {_fmt_pct(args.max_position_weight)} exceeded")
        if pd.notna(current_weight) and policy_cap_weight > 0 and float(current_weight) > policy_cap_weight + 1e-8:
            action = "TRIM"
            reasons.append(f"confidence band cap {_fmt_pct(policy_cap_weight)} exceeded ({confidence_policy.band})")
        if sector_weights.get(sector, 0.0) > args.sector_cap + 1e-8:
            action = "TRIM"
            reasons.append(f"sector cap {_fmt_pct(args.sector_cap)} exceeded")
        if theme != "(none)" and theme_weights.get(theme, 0.0) > args.theme_cap + 1e-8:
            action = "TRIM"
            reasons.append(f"theme cap {_fmt_pct(args.theme_cap)} exceeded")
        if not confidence_policy.hold_allowed:
            action = "EXIT_CANDIDATE"
            reasons.append(confidence_policy.guidance)
        elif pd.notna(candidate_rank) and int(candidate_rank) > args.max_holdings:
            if pd.notna(latest_rank) and int(latest_rank) <= args.max_hold_rank and confidence_policy.hold_allowed:
                if action == "HOLD":
                    action = "TRIM"
                reasons.append(f"out of top{args.max_holdings} but within hold range top{args.max_hold_rank}")
            else:
                action = "REPLACE_CANDIDATE"
                reasons.append(f"outside hold range top{args.max_hold_rank}")
        elif pd.isna(candidate_rank):
            if pd.notna(latest_rank) and int(latest_rank) <= args.max_hold_rank and confidence_policy.hold_allowed:
                if action == "HOLD":
                    action = "TRIM"
                reasons.append("not in official top5, latest snapshot still inside hold range")
            else:
                action = "REPLACE_CANDIDATE"
                reasons.append("missing from official recommendation set")

        if gate["status"] in {STATUS_HOLD, STATUS_BLOCK} and action == "REPLACE_CANDIDATE":
            action = "HOLD_REVIEW"
            reasons.append(f"gate {gate['status']} blocks score-driven replacement")
        if gate["status"] == STATUS_BLOCK and action == "HOLD":
            action = "TRIM"
            reasons.append("gate BLOCK prefers risk reduction over full hold")
        if not reasons:
            reasons.append("meets hold criteria")

        rows.append(
            {
                "code": code,
                "name": row.get("name"),
                "current_weight": current_weight,
                "candidate_rank": candidate_rank,
                "latest_snapshot_rank": latest_rank,
                "confidence_score": confidence,
                "confidence_band": confidence_policy.band,
                "candidate_final_score": candidate_score,
                "policy_cap_weight": policy_cap_weight,
                "action": action,
                "reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(rows).sort_values(["action", "candidate_rank", "latest_snapshot_rank", "code"], na_position="last").reset_index(drop=True)


def select_buy_ready_queue(
    candidates: pd.DataFrame,
    held_codes: set[str],
    cooldown_map: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    review_universe = candidates.loc[pd.to_numeric(candidates["buy_rank"], errors="coerce").le(args.candidate_universe_top_n)].copy()
    for _, row in review_universe.iterrows():
        code = str(row["code"]).zfill(6)
        eligible, notes = candidate_eligibility_notes(row, cooldown_map, args)
        confidence_policy = resolve_confidence_policy(row.get("confidence_score"), args)
        rows.append(
            {
                "code": code,
                "name": row["name"],
                "buy_rank": row["buy_rank"],
                "sector": row.get("sector"),
                "dominant_theme": row.get("dominant_theme"),
                "final_score": row["final_score"],
                "confidence_score": row["confidence_score"],
                "liquidity_score": row["liquidity_score"],
                "trading_value": row["trading_value"],
                "confidence_band": confidence_policy.band,
                "position_guidance": confidence_policy.label,
                "latest_snapshot_rank": row.get("latest_snapshot_rank"),
                "score_drift_vs_latest_snapshot": row.get("score_drift_vs_latest_snapshot"),
                "held_now": code in held_codes,
                "entry_eligible": eligible,
                "entry_note": "; ".join(notes) if notes else "meets entry criteria",
            }
        )
    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    selectable = queue.loc[queue["entry_eligible"]].copy()
    if selectable.empty:
        queue["target_weight"] = pd.NA
        queue["confidence_guidance"] = pd.NA
        queue["confidence_position_cap"] = pd.NA
        return queue.sort_values(["buy_rank", "code"]).reset_index(drop=True)
    weight_source = review_universe.loc[review_universe["code"].isin(selectable["code"])].copy()
    weights = compute_target_weights(weight_source, args)[["code", "target_weight", "confidence_guidance", "confidence_position_cap"]]
    queue = queue.merge(weights, on="code", how="left")
    return queue.sort_values(["buy_rank", "code"]).reset_index(drop=True)


def build_execution_actions(
    holdings_review: pd.DataFrame,
    buy_ready_queue: pd.DataFrame,
    gate: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    def _replacement_plan() -> tuple[dict[str, dict[str, object]], set[str]]:
        if holdings_review.empty or buy_ready_queue.empty:
            return {}, set()

        replace_pool = holdings_review.loc[
            holdings_review["action"].astype(str).eq("REPLACE_CANDIDATE")
        ].copy()
        if replace_pool.empty:
            return {}, set()

        entry_pool = buy_ready_queue.loc[
            buy_ready_queue["entry_eligible"].fillna(False)
            & ~buy_ready_queue["code"].isin(set(holdings_review["code"].astype(str)))
        ].copy()
        if entry_pool.empty:
            return {}, set()

        replace_pool["candidate_rank_sort"] = pd.to_numeric(replace_pool.get("candidate_rank"), errors="coerce").fillna(999999.0)
        replace_pool["candidate_final_score_sort"] = pd.to_numeric(replace_pool.get("candidate_final_score"), errors="coerce")
        replace_pool["current_weight_sort"] = pd.to_numeric(replace_pool.get("current_weight"), errors="coerce").fillna(0.0)
        replace_pool = replace_pool.sort_values(
            ["candidate_rank_sort", "candidate_final_score_sort", "current_weight_sort", "code"],
            ascending=[False, True, False, True],
            na_position="last",
        ).reset_index(drop=True)

        entry_pool["buy_rank_sort"] = pd.to_numeric(entry_pool.get("buy_rank"), errors="coerce").fillna(999999.0)
        entry_pool["final_score_sort"] = pd.to_numeric(entry_pool.get("final_score"), errors="coerce")
        entry_pool = entry_pool.sort_values(
            ["buy_rank_sort", "final_score_sort", "code"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

        max_replacements = max(int(args.max_replacements_per_cycle), 0)
        approved: dict[str, dict[str, object]] = {}
        selected_entry_codes: set[str] = set()
        if max_replacements <= 0:
            return approved, selected_entry_codes

        entry_idx = 0
        for _, replace_row in replace_pool.iterrows():
            replace_code = str(replace_row.get("code") or "").zfill(6)
            if not replace_code:
                continue
            if len(approved) >= max_replacements:
                approved[replace_code] = {
                    "approved": False,
                    "reason": f"replacement limit {max_replacements} reached",
                }
                continue
            if entry_idx >= len(entry_pool):
                approved[replace_code] = {
                    "approved": False,
                    "reason": "no eligible replacement entry available",
                }
                continue

            entry_row = entry_pool.iloc[entry_idx]
            entry_idx += 1

            held_score = pd.to_numeric(replace_row.get("candidate_final_score"), errors="coerce")
            new_score = pd.to_numeric(entry_row.get("final_score"), errors="coerce")
            if pd.isna(held_score) or pd.isna(new_score):
                approved[replace_code] = {
                    "approved": False,
                    "reason": "replacement blocked: missing score evidence",
                }
                continue

            score_gap = float(new_score) - float(held_score)
            if score_gap < float(args.min_replace_score_gap):
                approved[replace_code] = {
                    "approved": False,
                    "reason": (
                        f"replacement blocked: score gap {_fmt_num(score_gap)} "
                        f"< minimum {_fmt_num(args.min_replace_score_gap)}"
                    ),
                }
                continue

            entry_code = str(entry_row.get("code") or "").zfill(6)
            approved[replace_code] = {
                "approved": True,
                "entry_code": entry_code,
                "entry_name": entry_row.get("name"),
                "entry_reason": (
                    f"replacement entry approved vs {replace_code} "
                    f"(score gap {_fmt_num(score_gap)})"
                ),
                "reason": (
                    f"replacement approved with {entry_code} "
                    f"(score gap {_fmt_num(score_gap)})"
                ),
            }
            selected_entry_codes.add(entry_code)

        return approved, selected_entry_codes

    def _resolved_entry_rows(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if not selected_rows:
            return []
        selected = pd.DataFrame(selected_rows)
        weighted = compute_target_weights(selected, args)
        weighted_lookup = weighted.drop_duplicates("code").set_index("code")
        resolved_rows: list[dict[str, object]] = []
        for item in selected_rows:
            code = str(item["code"])
            target_weight = weighted_lookup.at[code, "target_weight"] if code in weighted_lookup.index else item.get("target_weight")
            resolved_rows.append({**item, "target_weight": target_weight})
        return resolved_rows

    rows: list[dict[str, object]] = []
    held_codes = set(holdings_review["code"].astype(str).tolist()) if not holdings_review.empty else set()
    replacement_decisions, reserved_entry_codes = _replacement_plan()
    kept_codes: set[str] = set()

    for _, row in holdings_review.iterrows():
        action = str(row["action"])
        code = str(row["code"])
        replacement_decision = replacement_decisions.get(code, {})
        target_weight = pd.NA
        reason = str(row.get("reason") or "")
        if action == "REPLACE_CANDIDATE" and not bool(replacement_decision.get("approved")):
            action = "HOLD_REVIEW"
            block_reason = str(replacement_decision.get("reason") or "replacement review deferred")
            reason = f"{reason}; {block_reason}" if reason else block_reason
        if action == "HOLD":
            target_weight = row.get("current_weight")
            kept_codes.add(code)
        elif action == "TRIM":
            current_weight = pd.to_numeric(row.get("current_weight"), errors="coerce")
            policy_cap_weight = pd.to_numeric(row.get("policy_cap_weight"), errors="coerce")
            trim_cap = float(policy_cap_weight) if pd.notna(policy_cap_weight) else args.max_position_weight
            target_weight = min(float(current_weight), trim_cap) if pd.notna(current_weight) else pd.NA
            kept_codes.add(code)
        elif action == "HOLD_REVIEW":
            target_weight = row.get("current_weight")
            kept_codes.add(code)
        elif action == "REPLACE_CANDIDATE" and bool(replacement_decision.get("approved")):
            reason = f"{reason}; {replacement_decision['reason']}" if reason else str(replacement_decision["reason"])
        rows.append(
            {
                "code": code,
                "name": row.get("name"),
                "action": action,
                "target_weight": target_weight,
                "reason": reason,
            }
        )

    slots_remaining = max(args.max_holdings - len(kept_codes), 0)
    limited_entry_mode = ""
    if gate["allow_new_entries"] and slots_remaining > 0:
        selected_entries: list[dict[str, object]] = []
        for _, row in buy_ready_queue.iterrows():
            row_code = str(row["code"])
            if not bool(row["entry_eligible"]) or row_code in held_codes or slots_remaining <= 0:
                continue
            if reserved_entry_codes and row_code not in reserved_entry_codes and len(selected_entries) < len(reserved_entry_codes):
                continue
            if row_code in {str(item["code"]) for item in selected_entries}:
                continue
            selected_entries.append(
                {
                    "code": row_code,
                    "name": row["name"],
                    "action": "ENTER",
                    "target_weight": row.get("target_weight"),
                    "sector": row.get("sector"),
                    "dominant_theme": row.get("dominant_theme"),
                    "final_score": row.get("final_score"),
                    "confidence_score": row.get("confidence_score"),
                    "liquidity_score": row.get("liquidity_score"),
                    "trading_value": row.get("trading_value"),
                    "reason": next(
                        (
                            str(value.get("entry_reason"))
                            for value in replacement_decisions.values()
                            if str(value.get("entry_code") or "") == row_code and value.get("approved")
                        ),
                        "BUY_ALLOWED gate and entry criteria satisfied",
                    ),
                }
            )
            slots_remaining -= 1
        if reserved_entry_codes:
            for _, row in buy_ready_queue.iterrows():
                row_code = str(row["code"])
                if row_code in reserved_entry_codes and row_code not in {str(item["code"]) for item in selected_entries} and slots_remaining > 0:
                    selected_entries.append(
                        {
                            "code": row_code,
                            "name": row["name"],
                            "action": "ENTER",
                            "target_weight": row.get("target_weight"),
                            "sector": row.get("sector"),
                            "dominant_theme": row.get("dominant_theme"),
                            "final_score": row.get("final_score"),
                            "confidence_score": row.get("confidence_score"),
                            "liquidity_score": row.get("liquidity_score"),
                            "trading_value": row.get("trading_value"),
                            "reason": next(
                                (
                                    str(value.get("entry_reason"))
                                    for value in replacement_decisions.values()
                                    if str(value.get("entry_code") or "") == row_code and value.get("approved")
                                ),
                                "BUY_ALLOWED gate and entry criteria satisfied",
                            ),
                        }
                    )
                    slots_remaining -= 1
        for item in _resolved_entry_rows(selected_entries):
            rows.append(
                {
                    "code": item["code"],
                    "name": item["name"],
                    "action": item["action"],
                    "target_weight": item["target_weight"],
                    "reason": item["reason"],
                }
            )

    else:
        limited_entry_mode = str(gate.get("limited_entry_mode") or "").strip().lower()
        limited_mode_enabled = (
            (limited_entry_mode == "watch" and bool(args.watch_limited_auto_buy_enabled))
            or (limited_entry_mode == "pilot" and bool(args.pilot_limited_auto_buy_enabled))
        )
        if not (limited_entry_mode and limited_mode_enabled and slots_remaining > 0):
            limited_entry_mode = ""

    if limited_entry_mode:
        if limited_entry_mode == "pilot":
            limited_slots_remaining = min(slots_remaining, max(int(args.pilot_limited_max_entries), 0))
            limited_exposure_remaining = max(float(args.pilot_limited_total_exposure), 0.0)
            limited_position_cap = max(float(args.pilot_limited_position_cap), 0.0)
            limited_label = "PILOT"
        else:
            limited_slots_remaining = min(slots_remaining, max(int(args.watch_limited_max_entries), 0))
            limited_exposure_remaining = max(float(args.watch_limited_total_exposure), 0.0)
            limited_position_cap = max(float(args.watch_limited_position_cap), 0.0)
            limited_label = "WATCH"
        selected_entries: list[dict[str, object]] = []
        for _, row in buy_ready_queue.iterrows():
            row_code = str(row["code"])
            if not bool(row["entry_eligible"]) or row_code in held_codes or limited_slots_remaining <= 0:
                continue
            if reserved_entry_codes and row_code not in reserved_entry_codes and len(selected_entries) < len(reserved_entry_codes):
                continue
            if row_code in {str(item["code"]) for item in selected_entries}:
                continue
            selected_entries.append(
                {
                    "code": row_code,
                    "name": row["name"],
                    "action": "ENTER",
                    "target_weight": row.get("target_weight"),
                    "sector": row.get("sector"),
                    "dominant_theme": row.get("dominant_theme"),
                    "final_score": row.get("final_score"),
                    "confidence_score": row.get("confidence_score"),
                    "liquidity_score": row.get("liquidity_score"),
                    "trading_value": row.get("trading_value"),
                    "reason": "",
                }
            )
            limited_slots_remaining -= 1
        for item in _resolved_entry_rows(selected_entries):
            base_target_weight = pd.to_numeric(item.get("target_weight"), errors="coerce")
            if pd.isna(base_target_weight) or float(base_target_weight) <= 0:
                continue
            capped_target_weight = min(float(base_target_weight), limited_position_cap, limited_exposure_remaining)
            if capped_target_weight <= 0:
                continue
            rows.append(
                {
                    "code": item["code"],
                    "name": item["name"],
                    "action": "ENTER",
                    "target_weight": capped_target_weight,
                    "reason": (
                        f"{limited_label} limited auto-buy window: constrained live entry allowed "
                        f"(cap {_fmt_pct(limited_position_cap)}, total remaining {_fmt_pct(limited_exposure_remaining)})"
                    ),
                }
            )
            limited_exposure_remaining = max(limited_exposure_remaining - capped_target_weight, 0.0)

    if not rows:
        rows.append(
            {
                "code": "-",
                "name": "",
                "action": "NO_ACTION",
                "target_weight": pd.NA,
                "reason": f"gate {gate['status']} does not permit new exposure and no current holdings require change",
            }
        )

    return pd.DataFrame(rows)


def top5_members(snapshot: pd.DataFrame) -> set[str]:
    if snapshot.empty:
        return set()
    return set(snapshot.loc[pd.to_numeric(snapshot["rank"], errors="coerce").le(5), "code"].astype(str).str.zfill(6))


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidates_csv)
    candidates = candidates.loc[pd.to_numeric(candidates["buy_rank"], errors="coerce").le(args.candidate_universe_top_n)].copy()
    candidates = candidates.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)
    snapshot_archive = load_snapshot_archive(args.snapshot_archive_csv)
    latest_snapshot, previous_snapshot = build_reference_maps(snapshot_archive)
    gate_payload = _safe_read_json(args.gate_json)
    gate = gate_decision_runtime(gate_payload)

    candidate_asof_raw = str(candidates["asof_date"].iloc[0]) if "asof_date" in candidates.columns and not candidates.empty else ""
    candidate_asof = pd.to_datetime(candidate_asof_raw, errors="coerce")
    latest_snapshot_date = latest_snapshot["asof_date"].iloc[0] if not latest_snapshot.empty else pd.NaT
    if pd.notna(candidate_asof):
        asof_date = pd.Timestamp(candidate_asof).normalize()
    elif pd.notna(latest_snapshot_date):
        asof_date = pd.Timestamp(latest_snapshot_date).normalize()
    else:
        asof_date = pd.Timestamp.today().normalize()

    holdings_context = load_holdings(args, candidates, latest_snapshot)
    cooldown_map = extract_cooldown_map(holdings_context.closed_positions, asof_date, args.reentry_cooldown_days)
    candidates = enrich_candidates_with_snapshot(candidates, latest_snapshot, previous_snapshot)
    holdings_review = classify_holdings(holdings_context.open_positions, candidates, latest_snapshot, gate, args)
    held_codes = set(holdings_context.open_positions["code"].astype(str).str.zfill(6).tolist()) if not holdings_context.open_positions.empty else set()
    buy_ready_queue = select_buy_ready_queue(candidates, held_codes, cooldown_map, args)
    execution_actions = build_execution_actions(holdings_review, buy_ready_queue, gate, args)

    candidate_top = set(candidates["code"].astype(str).str.zfill(6).tolist())
    latest_top = top5_members(latest_snapshot)
    new_vs_latest = sorted(latest_top - candidate_top)
    dropped_vs_latest = sorted(candidate_top - latest_top)

    operator_notes: list[str] = []
    if pd.notna(candidate_asof) and pd.notna(latest_snapshot_date) and pd.Timestamp(candidate_asof).normalize() < pd.Timestamp(latest_snapshot_date).normalize():
        operator_notes.append(
            f"official buy candidate file is stale: candidate asof {pd.Timestamp(candidate_asof).strftime('%Y-%m-%d')} vs latest ranking snapshot {pd.Timestamp(latest_snapshot_date).strftime('%Y-%m-%d')}"
        )
    if new_vs_latest:
        operator_notes.append(f"latest snapshot top5 has new names vs official candidates: {', '.join(new_vs_latest)}")
    if dropped_vs_latest:
        operator_notes.append(f"official candidates include names no longer in latest snapshot top5: {', '.join(dropped_vs_latest)}")
    if gate["status"] in {STATUS_HOLD, STATUS_BLOCK}:
        operator_notes.append(f"gate status {gate['status']} blocks 신규 진입. current recommendation set is observation-only until gate improves.")
    if holdings_context.source == "(none)":
        operator_notes.append("no holdings file found, report is an empty-account simulation rather than a live rebalance instruction.")
    if all(str(value) == "(none)" for value in candidates["dominant_theme"].tolist()):
        operator_notes.append("all current top5 candidates have dominant_theme=(none); theme cap is not binding and sector cap is the active concentration control.")

    queue_display = buy_ready_queue.copy()
    if not queue_display.empty:
        queue_display["held_now"] = queue_display["held_now"].map({True: "Y", False: "N"})
        queue_display["entry_eligible"] = queue_display["entry_eligible"].map({True: "Y", False: "N"})
        queue_display["final_score"] = queue_display["final_score"].map(_fmt_num)
        queue_display["confidence_score"] = queue_display["confidence_score"].map(_fmt_num)
        queue_display["liquidity_score"] = queue_display["liquidity_score"].map(_fmt_num)
        queue_display["trading_value"] = queue_display["trading_value"].map(lambda v: _fmt_num(v, 0))
        queue_display["score_drift_vs_latest_snapshot"] = queue_display["score_drift_vs_latest_snapshot"].map(_fmt_num)
        queue_display["target_weight"] = queue_display["target_weight"].map(_fmt_pct)
        queue_display["confidence_position_cap"] = queue_display["confidence_position_cap"].map(_fmt_pct)

    holdings_display = holdings_review.copy()
    if not holdings_display.empty:
        holdings_display["current_weight"] = holdings_display["current_weight"].map(_fmt_pct)
        holdings_display["confidence_score"] = holdings_display["confidence_score"].map(_fmt_num)
        holdings_display["candidate_final_score"] = holdings_display["candidate_final_score"].map(_fmt_num)
        holdings_display["policy_cap_weight"] = holdings_display["policy_cap_weight"].map(_fmt_pct)

    actions_display = execution_actions.copy()
    if not actions_display.empty:
        actions_display["target_weight"] = actions_display["target_weight"].map(_fmt_pct)

    report_lines = [
        "# Operational Execution Policy Simulation Report",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- asof_date: {asof_date.strftime('%Y-%m-%d')}",
        f"- policy_version: {POLICY_VERSION}",
        f"- score_formula_version: {SCORE_FORMULA_VERSION}",
        f"- gate_version: {GATE_VERSION}",
        f"- portfolio_version: {PORTFOLIO_VERSION}",
        f"- holdings_source: {holdings_context.source}",
        "",
        "## Gate Context",
        "",
        f"- overall_status: {gate['status']}",
        f"- guidance: {gate['guidance']}",
        f"- reason_summary: {gate['reason'] or 'NA'}",
        "",
        "## Policy Thresholds",
        "",
        f"- max_holdings: {args.max_holdings}",
        f"- max_position_weight: {_fmt_pct(args.max_position_weight)}",
        f"- sector_cap: {_fmt_pct(args.sector_cap)}",
        f"- theme_cap: {_fmt_pct(args.theme_cap)}",
        f"- cash_minimum: {_fmt_pct(args.cash_minimum)}",
        f"- watch_limited_auto_buy_enabled: {'Y' if args.watch_limited_auto_buy_enabled else 'N'}",
        f"- watch_limited_max_entries: {args.watch_limited_max_entries}",
        f"- watch_limited_total_exposure: {_fmt_pct(args.watch_limited_total_exposure)}",
        f"- watch_limited_position_cap: {_fmt_pct(args.watch_limited_position_cap)}",
        f"- pilot_limited_auto_buy_enabled: {'Y' if args.pilot_limited_auto_buy_enabled else 'N'}",
        f"- pilot_limited_max_entries: {args.pilot_limited_max_entries}",
        f"- pilot_limited_total_exposure: {_fmt_pct(args.pilot_limited_total_exposure)}",
        f"- pilot_limited_position_cap: {_fmt_pct(args.pilot_limited_position_cap)}",
        f"- entry_confidence_floor: {_fmt_num(args.min_entry_confidence)}",
        f"- hold_confidence_floor: {_fmt_num(args.min_hold_confidence)}",
        f"- force_exit_confidence_floor: {_fmt_num(args.force_exit_confidence)}",
        f"- confidence bands: <{_fmt_num(args.confidence_block_below)} block / {_fmt_num(args.confidence_block_below)}~{_fmt_num(args.confidence_reduced_below)} reduced / {_fmt_num(args.confidence_reduced_below)}~{_fmt_num(args.confidence_standard_below)} standard / {_fmt_num(args.confidence_standard_below)}+ expanded",
        f"- reduced weight scale: {_fmt_num(args.confidence_reduced_weight_scale)} / expanded weight scale: {_fmt_num(args.confidence_expanded_weight_scale)}",
        f"- reduced cap scale: {_fmt_num(args.confidence_reduced_position_cap_scale)} / expanded cap scale: {_fmt_num(args.confidence_expanded_position_cap_scale)}",
        f"- max_hold_rank: top{args.max_hold_rank}",
        f"- reentry_cooldown_days: {args.reentry_cooldown_days}",
        "",
        "## Current Holdings Review",
        "",
        _markdown_table(holdings_display, ["code", "name", "current_weight", "policy_cap_weight", "candidate_rank", "latest_snapshot_rank", "confidence_score", "confidence_band", "candidate_final_score", "action", "reason"]),
        "",
        "## Recommended Actions",
        "",
        _markdown_table(actions_display, ["code", "name", "action", "target_weight", "reason"]),
        "",
        "## Buy-Ready Queue",
        "",
        _markdown_table(queue_display, ["code", "name", "buy_rank", "held_now", "entry_eligible", "confidence_band", "position_guidance", "confidence_position_cap", "target_weight", "final_score", "confidence_score", "latest_snapshot_rank", "score_drift_vs_latest_snapshot", "entry_note"]),
        "",
        "## Operator Notes",
        "",
        markdown_summary_list(operator_notes),
        "",
    ]

    out_path = _resolve(args.out_md)
    if out_path is None:
        raise ValueError("output path could not be resolved")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
