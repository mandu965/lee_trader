from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from production_config import get_production_config_value


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_CSV = DATA_DIR / "ranking_final.csv"
OUTPUT_TOP5_CSV = DATA_DIR / "buy_candidates_top5.csv"
OUTPUT_TOP8_CSV = DATA_DIR / "buy_candidates_top8.csv"
OUTPUT_TOP10_CSV = DATA_DIR / "buy_candidates_top10.csv"
OUTPUT_MD = OUTPUT_DIR / "buy_candidate_builder_report.md"

TARGET_SIZES = [5, 8, 10]
DATE_CANDIDATE_COLUMNS = ["as_of_date", "date", "trade_date", "snapshot_date"]
BUY_CANDIDATE_VERSION = str(
    get_production_config_value(["metadata", "candidate_selection_version"], "buy_candidate_builder_v1")
)
SCORE_FORMULA_VERSION = str(
    get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v8_return_prob_tech_regime")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build operational buy candidate lists from ranking_final.csv.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--out-top5-csv", type=Path, default=OUTPUT_TOP5_CSV)
    parser.add_argument("--out-top8-csv", type=Path, default=OUTPUT_TOP8_CSV)
    parser.add_argument("--out-top10-csv", type=Path, default=OUTPUT_TOP10_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--min-confidence", type=float, default=float(get_production_config_value(["buy_candidate", "min_confidence"], 80.0)))
    parser.add_argument("--min-liquidity-score", type=float, default=float(get_production_config_value(["buy_candidate", "min_liquidity_score"], 15.0)))
    parser.add_argument("--min-trading-value", type=float, default=float(get_production_config_value(["buy_candidate", "min_trading_value"], 5_000_000_000.0)))
    parser.add_argument("--sector-cap-ratio", type=float, default=float(get_production_config_value(["buy_candidate", "sector_cap_ratio"], 0.30)))
    parser.add_argument("--theme-cap-ratio", type=float, default=float(get_production_config_value(["buy_candidate", "theme_cap_ratio"], 0.30)))
    parser.add_argument("--no-theme-cap-ratio", type=float, default=float(get_production_config_value(["buy_candidate", "no_theme_cap_ratio"], 0.50)))
    parser.add_argument("--soft-surge-ret5d", type=float, default=float(get_production_config_value(["buy_candidate", "soft_surge_ret5d"], 0.12)))
    parser.add_argument("--soft-surge-ret10d", type=float, default=float(get_production_config_value(["buy_candidate", "soft_surge_ret10d"], 0.20)))
    parser.add_argument("--soft-surge-rsi", type=float, default=float(get_production_config_value(["buy_candidate", "soft_surge_rsi"], 70.0)))
    parser.add_argument("--hard-surge-ret5d", type=float, default=float(get_production_config_value(["buy_candidate", "hard_surge_ret5d"], 0.20)))
    parser.add_argument("--hard-surge-ret10d", type=float, default=float(get_production_config_value(["buy_candidate", "hard_surge_ret10d"], 0.35)))
    parser.add_argument("--hard-surge-rsi", type=float, default=float(get_production_config_value(["buy_candidate", "hard_surge_rsi"], 80.0)))
    return parser.parse_args()


def _fmt(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _fmt_krw(value: object) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{int(round(float(x))):,}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows_"
    table = frame[columns].copy().fillna("")
    rendered = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    headers = columns
    widths = [len(col) for col in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(val.ljust(widths[idx]) for idx, val in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def resolve_latest_slice(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    for column in DATE_CANDIDATE_COLUMNS:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            latest = parsed.max().normalize()
            mask = parsed.dt.normalize().eq(latest)
            return df.loc[mask].copy(), latest.strftime("%Y-%m-%d"), column
    raise ValueError("could not resolve latest date from ranking csv")


def cap_count(target_size: int, ratio: float) -> int:
    return max(1, int(math.ceil(target_size * ratio)))


def normalize_input(df: pd.DataFrame, *, asof_date: str, args: argparse.Namespace) -> pd.DataFrame:
    work = df.copy()
    if "code" not in work.columns:
        raise ValueError("ranking csv must contain code")

    work["asof_date"] = asof_date
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    work["market"] = work.get("market", "").fillna("").astype(str)
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["dominant_theme"] = (
        work.get("dominant_theme", "(none)")
        .fillna("(none)")
        .astype(str)
        .replace({"": "(none)", "nan": "(none)"})
    )
    work["theme_score"] = pd.to_numeric(work.get("theme_score"), errors="coerce")
    work["final_score"] = pd.to_numeric(work.get("final_score"), errors="coerce")
    work["confidence_score"] = pd.to_numeric(work.get("confidence_score"), errors="coerce")
    work["liquidity_score"] = pd.to_numeric(work.get("liquidity_score"), errors="coerce")
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce")
    work["close"] = pd.to_numeric(work.get("close"), errors="coerce")
    work["ret_5d"] = pd.to_numeric(work.get("ret_5d"), errors="coerce")
    work["ret_10d"] = pd.to_numeric(work.get("ret_10d"), errors="coerce")
    work["mom_20"] = pd.to_numeric(work.get("mom_20"), errors="coerce")
    work["rsi_14"] = pd.to_numeric(work.get("rsi_14"), errors="coerce")
    work["explain_text"] = work.get("explain_text", "").fillna("").astype(str)
    work["rank_source"] = pd.to_numeric(work.get("rank_final"), errors="coerce")
    work["score_formula_version"] = work.get("score_formula_version", SCORE_FORMULA_VERSION)
    work["candidate_selection_version"] = BUY_CANDIDATE_VERSION
    if work["rank_source"].isna().all():
        work["rank_source"] = (
            pd.to_numeric(work["final_score"], errors="coerce")
            .rank(method="first", ascending=False)
            .astype(float)
        )
    work["rank_source"] = work["rank_source"].round().astype("Int64")
    work["trading_value"] = work["close"] * work["volume"]
    work["has_theme"] = work["dominant_theme"].ne("(none)")

    liq_from_score = work["liquidity_score"].ge(args.min_liquidity_score)
    liq_from_value = work["trading_value"].ge(args.min_trading_value)
    work["liquidity_gate_pass"] = liq_from_score.fillna(False) | liq_from_value.fillna(False)
    work["liquidity_gate_source"] = "failed"
    work.loc[liq_from_score.fillna(False) & liq_from_value.fillna(False), "liquidity_gate_source"] = "score_and_trading_value"
    work.loc[liq_from_score.fillna(False) & ~liq_from_value.fillna(False), "liquidity_gate_source"] = "score_only"
    work.loc[~liq_from_score.fillna(False) & liq_from_value.fillna(False), "liquidity_gate_source"] = "trading_value_only"

    soft_surge = (
        work["ret_5d"].ge(args.soft_surge_ret5d).fillna(False)
        | work["ret_10d"].ge(args.soft_surge_ret10d).fillna(False)
        | work["rsi_14"].ge(args.soft_surge_rsi).fillna(False)
    )
    hard_surge = (
        work["ret_5d"].ge(args.hard_surge_ret5d).fillna(False)
        | work["ret_10d"].ge(args.hard_surge_ret10d).fillna(False)
        | work["rsi_14"].ge(args.hard_surge_rsi).fillna(False)
    )
    work["recent_surge_soft_flag"] = soft_surge
    work["recent_surge_hard_flag"] = hard_surge

    work["hard_filter_reason"] = ""
    low_conf = work["confidence_score"].lt(args.min_confidence) | work["confidence_score"].isna()
    low_liq = ~work["liquidity_gate_pass"].fillna(False)
    invalid_trade = work["volume"].le(0).fillna(True) | work["trading_value"].le(0).fillna(True)
    hard_surge_mask = work["recent_surge_hard_flag"].fillna(False)

    reason_parts = []
    for idx in work.index:
        parts: list[str] = []
        if bool(low_conf.loc[idx]):
            parts.append("low_confidence")
        if bool(low_liq.loc[idx]):
            parts.append("low_liquidity")
        if bool(invalid_trade.loc[idx]):
            parts.append("invalid_trading_activity")
        if bool(hard_surge_mask.loc[idx]):
            parts.append("extreme_recent_surge")
        reason_parts.append("; ".join(parts))
    work["hard_filter_reason"] = reason_parts
    work["hard_filter_excluded"] = work["hard_filter_reason"].ne("")
    work["selection_notes"] = ""
    work["selection_stage"] = ""
    work["exclusion_reason"] = work["hard_filter_reason"]

    sort_columns = ["recent_surge_soft_flag", "rank_source", "final_score", "confidence_score", "code"]
    sort_ascending = [True, True, False, False, True]
    work = work.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)
    return work


def select_candidates(
    universe: pd.DataFrame,
    *,
    target_size: int,
    sector_cap: int,
    theme_cap: int,
    no_theme_cap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    eligible = universe.loc[~universe["hard_filter_excluded"]].copy().reset_index(drop=True)
    selected_indices: list[int] = []
    phase_counts: dict[str, int] = {}

    phases = [
        {
            "name": "strict",
            "enforce_sector_cap": True,
            "enforce_theme_cap": True,
            "enforce_no_theme_cap": True,
            "note": "strict rules",
        },
        {
            "name": "relax_no_theme_cap",
            "enforce_sector_cap": True,
            "enforce_theme_cap": True,
            "enforce_no_theme_cap": False,
            "note": "filled after relaxing no-theme cap",
        },
        {
            "name": "relax_theme_cap",
            "enforce_sector_cap": True,
            "enforce_theme_cap": False,
            "enforce_no_theme_cap": False,
            "note": "filled after relaxing theme cap",
        },
        {
            "name": "relax_sector_cap",
            "enforce_sector_cap": False,
            "enforce_theme_cap": False,
            "enforce_no_theme_cap": False,
            "note": "filled after relaxing sector cap",
        },
    ]

    def _current_counts() -> tuple[dict[str, int], dict[str, int], int]:
        sector_counts: dict[str, int] = {}
        theme_counts: dict[str, int] = {}
        no_theme_count = 0
        for idx in selected_indices:
            row = eligible.iloc[idx]
            sector = row["sector"]
            theme = row["dominant_theme"]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if theme == "(none)":
                no_theme_count += 1
            else:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        return sector_counts, theme_counts, no_theme_count

    deferred_records: list[dict[str, object]] = []

    for phase in phases:
        if len(selected_indices) >= target_size:
            break
        sector_counts, theme_counts, no_theme_count = _current_counts()
        selected_set = set(selected_indices)
        for idx, row in eligible.iterrows():
            if idx in selected_set:
                continue
            sector = row["sector"]
            theme = row["dominant_theme"]
            if phase["enforce_sector_cap"] and sector_counts.get(sector, 0) >= sector_cap:
                deferred_records.append(
                    {
                        "code": row["code"],
                        "name": row["name"],
                        "reason": "sector_concentration_cap",
                        "phase": phase["name"],
                        "source_rank": int(row["rank_source"]),
                    }
                )
                continue
            if phase["enforce_theme_cap"] and theme != "(none)" and theme_counts.get(theme, 0) >= theme_cap:
                deferred_records.append(
                    {
                        "code": row["code"],
                        "name": row["name"],
                        "reason": "theme_concentration_cap",
                        "phase": phase["name"],
                        "source_rank": int(row["rank_source"]),
                    }
                )
                continue
            if phase["enforce_no_theme_cap"] and theme == "(none)" and no_theme_count >= no_theme_cap:
                deferred_records.append(
                    {
                        "code": row["code"],
                        "name": row["name"],
                        "reason": "no_theme_cap",
                        "phase": phase["name"],
                        "source_rank": int(row["rank_source"]),
                    }
                )
                continue

            selected_indices.append(idx)
            selected_set.add(idx)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if theme == "(none)":
                no_theme_count += 1
            else:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            phase_counts[phase["name"]] = phase_counts.get(phase["name"], 0) + 1
            if len(selected_indices) >= target_size:
                break

    selected = eligible.iloc[selected_indices].copy().reset_index(drop=True)
    selected["buy_rank"] = range(1, len(selected) + 1)
    selected["target_bucket"] = target_size
    selected["selection_stage"] = "strict"
    selected["selection_notes"] = "selected under strict rules"
    selected["exclusion_reason"] = ""

    strict_count = phase_counts.get("strict", 0)
    if strict_count < len(selected):
        selected.loc[selected.index >= strict_count, "selection_stage"] = "relaxed"

    offset = strict_count
    for phase in phases[1:]:
        phase_count = phase_counts.get(phase["name"], 0)
        if phase_count <= 0:
            continue
        end = offset + phase_count
        selected.loc[(selected.index >= offset) & (selected.index < end), "selection_stage"] = phase["name"]
        selected.loc[(selected.index >= offset) & (selected.index < end), "selection_notes"] = phase["note"]
        offset = end

    selected["sector_cap"] = sector_cap
    selected["theme_cap"] = theme_cap
    selected["no_theme_cap"] = no_theme_cap

    summary = {
        "target_size": target_size,
        "eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
        "strict_selected_count": int(phase_counts.get("strict", 0)),
        "phase_counts": phase_counts,
        "sector_cap": sector_cap,
        "theme_cap": theme_cap,
        "no_theme_cap": no_theme_cap,
        "deferred_records": deferred_records,
    }
    return selected, summary


def build_report(
    *,
    asof_date: str,
    date_column: str,
    normalized: pd.DataFrame,
    hard_excluded: pd.DataFrame,
    selections: dict[int, pd.DataFrame],
    summaries: dict[int, dict[str, object]],
    args: argparse.Namespace,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hard_reason_counts = (
        hard_excluded.assign(reason_split=hard_excluded["hard_filter_reason"].str.split("; "))
        .explode("reason_split")
        .dropna(subset=["reason_split"])
        ["reason_split"]
        .value_counts()
        .reset_index()
    )
    if not hard_reason_counts.empty:
        hard_reason_counts.columns = ["reason", "count"]

    lines = [
        "# Buy Candidate Builder Report",
        "",
        f"- generated_at: {generated_at}",
        f"- source_csv: {INPUT_CSV}",
        f"- source_latest_date: {asof_date}",
        f"- source_date_column: {date_column}",
        f"- source_rows_latest_date: {len(normalized)}",
        f"- hard_filter_excluded_rows: {int(normalized['hard_filter_excluded'].sum())}",
        f"- eligible_rows_after_hard_filters: {int((~normalized['hard_filter_excluded']).sum())}",
        "",
        "## Rule Config",
        f"- candidate_selection_version: {BUY_CANDIDATE_VERSION}",
        f"- score_formula_version: {SCORE_FORMULA_VERSION}",
        f"- confidence_score >= {args.min_confidence:.1f}",
        f"- liquidity_score >= {args.min_liquidity_score:.1f} or trading_value >= {_fmt_krw(args.min_trading_value)} KRW",
        f"- sector cap ratio: {args.sector_cap_ratio:.2f}",
        f"- theme cap ratio: {args.theme_cap_ratio:.2f}",
        f"- no-theme cap ratio: {args.no_theme_cap_ratio:.2f}",
        f"- recent surge soft gate: ret_5d >= {_fmt_pct(args.soft_surge_ret5d)} or ret_10d >= {_fmt_pct(args.soft_surge_ret10d)} or rsi_14 >= {_fmt(args.soft_surge_rsi)}",
        f"- recent surge hard gate: ret_5d >= {_fmt_pct(args.hard_surge_ret5d)} or ret_10d >= {_fmt_pct(args.hard_surge_ret10d)} or rsi_14 >= {_fmt(args.hard_surge_rsi)}",
        "",
        "## Hard Filter Breakdown",
    ]

    if hard_reason_counts.empty:
        lines.append("- no hard-filter exclusions")
    else:
        lines.append(
            _markdown_table(
                hard_reason_counts,
                ["reason", "count"],
            )
        )

    excluded_preview = hard_excluded.sort_values(["rank_source", "final_score"], ascending=[True, False]).head(12).copy()
    if not excluded_preview.empty:
        excluded_preview["final_score"] = excluded_preview["final_score"].map(_fmt)
        excluded_preview["confidence_score"] = excluded_preview["confidence_score"].map(_fmt)
        excluded_preview["liquidity_score"] = excluded_preview["liquidity_score"].map(_fmt)
        excluded_preview["trading_value"] = excluded_preview["trading_value"].map(_fmt_krw)
        lines.extend(
            [
                "",
                "## Excluded Preview",
                _markdown_table(
                    excluded_preview,
                    [
                        "rank_source",
                        "code",
                        "name",
                        "final_score",
                        "confidence_score",
                        "liquidity_score",
                        "trading_value",
                        "hard_filter_reason",
                    ],
                ),
            ]
        )

    for target_size in TARGET_SIZES:
        selected = selections[target_size].copy()
        summary = summaries[target_size]
        selected["final_score"] = selected["final_score"].map(_fmt)
        selected["confidence_score"] = selected["confidence_score"].map(_fmt)
        selected["liquidity_score"] = selected["liquidity_score"].map(_fmt)
        selected["trading_value"] = selected["trading_value"].map(_fmt_krw)
        selected["ret_5d"] = selected["ret_5d"].map(_fmt_pct)
        selected["ret_10d"] = selected["ret_10d"].map(_fmt_pct)

        lines.extend(
            [
                "",
                f"## Top{target_size}",
                f"- selected_count: {summary['selected_count']}",
                f"- strict_selected_count: {summary['strict_selected_count']}",
                f"- sector_cap: {summary['sector_cap']}",
                f"- theme_cap: {summary['theme_cap']}",
                f"- no_theme_cap: {summary['no_theme_cap']}",
                f"- phase_counts: {summary['phase_counts']}",
                _markdown_table(
                    selected,
                    [
                        "buy_rank",
                        "rank_source",
                        "code",
                        "name",
                        "sector",
                        "dominant_theme",
                        "final_score",
                        "confidence_score",
                        "liquidity_score",
                        "trading_value",
                        "ret_5d",
                        "ret_10d",
                        "selection_stage",
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Output Files",
            f"- [buy_candidates_top5.csv]({OUTPUT_TOP5_CSV.as_posix()})",
            f"- [buy_candidates_top8.csv]({OUTPUT_TOP8_CSV.as_posix()})",
            f"- [buy_candidates_top10.csv]({OUTPUT_TOP10_CSV.as_posix()})",
        ]
    )
    return "\n".join(lines) + "\n"


def output_columns() -> list[str]:
    return [
        "asof_date",
        "target_bucket",
        "buy_rank",
        "rank_source",
        "code",
        "name",
        "market",
        "sector",
        "dominant_theme",
        "theme_score",
        "final_score",
        "confidence_score",
        "liquidity_score",
        "liquidity_gate_source",
        "trading_value",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
        "recent_surge_soft_flag",
        "selection_stage",
        "selection_notes",
        "exclusion_reason",
        "explain_text",
        "score_formula_version",
        "candidate_selection_version",
    ]


def save_selection(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns()].to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    input_path = args.input_csv if args.input_csv.is_absolute() else ROOT / args.input_csv
    if not input_path.exists():
        raise FileNotFoundError(f"input ranking csv not found: {input_path}")

    raw = pd.read_csv(input_path, dtype={"code": str}, low_memory=False)
    latest_slice, asof_date, date_column = resolve_latest_slice(raw)
    normalized = normalize_input(latest_slice, asof_date=asof_date, args=args)
    hard_excluded = normalized.loc[normalized["hard_filter_excluded"]].copy()

    selections: dict[int, pd.DataFrame] = {}
    summaries: dict[int, dict[str, object]] = {}
    for target_size in TARGET_SIZES:
        sector_cap = cap_count(target_size, args.sector_cap_ratio)
        theme_cap = cap_count(target_size, args.theme_cap_ratio)
        no_theme_cap = cap_count(target_size, args.no_theme_cap_ratio)
        selected, summary = select_candidates(
            normalized,
            target_size=target_size,
            sector_cap=sector_cap,
            theme_cap=theme_cap,
            no_theme_cap=no_theme_cap,
        )
        selections[target_size] = selected
        summaries[target_size] = summary

    save_selection(selections[5], args.out_top5_csv if args.out_top5_csv.is_absolute() else ROOT / args.out_top5_csv)
    save_selection(selections[8], args.out_top8_csv if args.out_top8_csv.is_absolute() else ROOT / args.out_top8_csv)
    save_selection(selections[10], args.out_top10_csv if args.out_top10_csv.is_absolute() else ROOT / args.out_top10_csv)

    report = build_report(
        asof_date=asof_date,
        date_column=date_column,
        normalized=normalized,
        hard_excluded=hard_excluded,
        selections=selections,
        summaries=summaries,
        args=args,
    )
    out_md = args.out_md if args.out_md.is_absolute() else ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")

    print(f"asof_date: {asof_date}")
    for target_size in TARGET_SIZES:
        summary = summaries[target_size]
        print(
            f"top{target_size}: selected={summary['selected_count']} strict={summary['strict_selected_count']} "
            f"phase_counts={summary['phase_counts']}"
        )
    print(f"report_path: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
