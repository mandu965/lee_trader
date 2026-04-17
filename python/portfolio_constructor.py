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

INPUT_TOP5 = DATA_DIR / "buy_candidates_top5.csv"
INPUT_TOP8 = DATA_DIR / "buy_candidates_top8.csv"
INPUT_TOP10 = DATA_DIR / "buy_candidates_top10.csv"
OUTPUT_TOP5 = DATA_DIR / "model_portfolio_top5.csv"
OUTPUT_TOP8 = DATA_DIR / "model_portfolio_top8.csv"
OUTPUT_TOP10 = DATA_DIR / "model_portfolio_top10.csv"
OUTPUT_MD = OUTPUT_DIR / "portfolio_construction_report.md"
PORTFOLIO_VERSION = str(
    get_production_config_value(["metadata", "portfolio_version"], "model_portfolio_constructor_v1")
)
SCORE_FORMULA_VERSION = str(
    get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v8_return_prob_tech_regime")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct investable model portfolios from buy candidate files.")
    parser.add_argument("--input-top5", type=Path, default=INPUT_TOP5)
    parser.add_argument("--input-top8", type=Path, default=INPUT_TOP8)
    parser.add_argument("--input-top10", type=Path, default=INPUT_TOP10)
    parser.add_argument("--out-top5", type=Path, default=OUTPUT_TOP5)
    parser.add_argument("--out-top8", type=Path, default=OUTPUT_TOP8)
    parser.add_argument("--out-top10", type=Path, default=OUTPUT_TOP10)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--cash-buffer", type=float, default=float(get_production_config_value(["portfolio", "cash_buffer"], 0.05)))
    parser.add_argument("--max-weight-top5", type=float, default=float(get_production_config_value(["portfolio", "max_weight_top5"], 0.24)))
    parser.add_argument("--max-weight-top8", type=float, default=float(get_production_config_value(["portfolio", "max_weight_top8"], 0.17)))
    parser.add_argument("--max-weight-top10", type=float, default=float(get_production_config_value(["portfolio", "max_weight_top10"], 0.13)))
    parser.add_argument("--sector-cap", type=float, default=float(get_production_config_value(["portfolio", "sector_cap"], 0.35)))
    parser.add_argument("--theme-cap", type=float, default=float(get_production_config_value(["portfolio", "theme_cap"], 0.35)))
    parser.add_argument("--no-theme-cap", type=float, default=float(get_production_config_value(["portfolio", "no_theme_cap"], 0.60)))
    parser.add_argument("--min-keep-weight", type=float, default=float(get_production_config_value(["portfolio", "min_keep_weight"], 0.03)))
    parser.add_argument("--turnover-keep-slots", type=int, default=int(get_production_config_value(["portfolio", "turnover_keep_slots"], 2)))
    parser.add_argument("--keep-rank-buffer", type=int, default=int(get_production_config_value(["portfolio", "keep_rank_buffer"], 3)))
    parser.add_argument("--liquidity-score-low", type=float, default=float(get_production_config_value(["portfolio", "liquidity_score_low"], 20.0)))
    parser.add_argument("--liquidity-score-very-low", type=float, default=float(get_production_config_value(["portfolio", "liquidity_score_very_low"], 10.0)))
    parser.add_argument("--trading-value-low", type=float, default=float(get_production_config_value(["portfolio", "trading_value_low"], 8_000_000_000.0)))
    parser.add_argument("--trading-value-very-low", type=float, default=float(get_production_config_value(["portfolio", "trading_value_very_low"], 4_000_000_000.0)))
    return parser.parse_args()


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


def load_candidates(path: Path, *, target_size: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"candidate file is empty: {path}")

    work = df.copy()
    work["target_size"] = target_size
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work.get("name", "").fillna("").astype(str)
    work["sector"] = work.get("sector", "(unknown)").fillna("(unknown)").astype(str)
    work["dominant_theme"] = (
        work.get("dominant_theme", "(none)")
        .fillna("(none)")
        .astype(str)
        .replace({"": "(none)", "nan": "(none)"})
    )
    for col in [
        "buy_rank",
        "rank_source",
        "final_score",
        "confidence_score",
        "liquidity_score",
        "theme_score",
        "trading_value",
        "ret_5d",
        "ret_10d",
        "mom_20",
        "rsi_14",
    ]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["asof_date"] = work.get("asof_date", "").astype(str)
    work["explain_text"] = work.get("explain_text", "").fillna("").astype(str)
    work["selection_stage"] = work.get("selection_stage", "").fillna("").astype(str)
    work["recent_surge_soft_flag"] = work.get("recent_surge_soft_flag", False).astype(str).str.lower().isin(["true", "1"])
    return work.sort_values(["buy_rank", "rank_source", "code"]).reset_index(drop=True)


def load_previous_portfolio(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def liquidity_haircut(row: pd.Series, args: argparse.Namespace) -> tuple[float, str]:
    liq = pd.to_numeric(row.get("liquidity_score"), errors="coerce")
    value = pd.to_numeric(row.get("trading_value"), errors="coerce")
    if (pd.notna(liq) and liq < args.liquidity_score_very_low) or (
        pd.notna(value) and value < args.trading_value_very_low
    ):
        return 0.55, "very_low_liquidity_haircut"
    if (pd.notna(liq) and liq < args.liquidity_score_low) or (pd.notna(value) and value < args.trading_value_low):
        return 0.80, "low_liquidity_haircut"
    return 1.0, "none"


def build_score(row: pd.Series, args: argparse.Namespace) -> tuple[float, float, str]:
    final_score = float(pd.to_numeric(row.get("final_score"), errors="coerce") or 0.0)
    confidence = float(pd.to_numeric(row.get("confidence_score"), errors="coerce") or 0.0)
    liquidity = float(pd.to_numeric(row.get("liquidity_score"), errors="coerce") or 0.0)
    base = (final_score / 100.0) ** 1.10 * (confidence / 100.0) ** 1.20
    liquidity_multiplier, liquidity_note = liquidity_haircut(row, args)
    confidence_boost = 1.0 + max(0.0, confidence - 80.0) / 200.0
    score = max(base * confidence_boost * liquidity_multiplier, 1e-9)
    detail = f"base={base:.4f}; confidence_boost={confidence_boost:.3f}; liquidity_mult={liquidity_multiplier:.2f}; liq={liquidity:.1f}"
    return score, liquidity_multiplier, liquidity_note if liquidity_note != "none" else detail


def allocate_weights(
    candidates: pd.DataFrame,
    *,
    target_size: int,
    max_weight: float,
    previous_portfolio: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = candidates.copy()
    work["base_weight_score"] = 0.0
    work["liquidity_multiplier"] = 1.0
    work["weight_reason"] = ""
    work["keep_from_previous"] = False
    prev_codes = set(previous_portfolio["code"].tolist()) if not previous_portfolio.empty else set()

    for idx, row in work.iterrows():
        score, liq_mult, note = build_score(row, args)
        work.at[idx, "base_weight_score"] = score
        work.at[idx, "liquidity_multiplier"] = liq_mult
        work.at[idx, "weight_reason"] = note
        if row["code"] in prev_codes:
            work.at[idx, "keep_from_previous"] = True

    slot_count = min(target_size, len(work))
    sector_cap_slots = max(1, int(math.ceil(slot_count * args.sector_cap)))
    theme_cap_slots = max(1, int(math.ceil(slot_count * args.theme_cap)))
    no_theme_cap_slots = max(1, int(math.ceil(slot_count * args.no_theme_cap)))

    chosen_rows: list[int] = []
    kept_rows: list[int] = []
    sector_counts: dict[str, int] = {}
    theme_counts: dict[str, int] = {}
    no_theme_count = 0
    selection_mode = "strict"

    ranked_keep = work.loc[work["keep_from_previous"]].sort_values(["buy_rank", "final_score"], ascending=[True, False])
    for idx, row in ranked_keep.iterrows():
        if len(kept_rows) >= args.turnover_keep_slots:
            break
        rank_ok = int(pd.to_numeric(row["buy_rank"], errors="coerce") or 9999) <= target_size + args.keep_rank_buffer
        if not rank_ok:
            continue
        sector = row["sector"]
        theme = row["dominant_theme"]
        if sector_counts.get(sector, 0) >= sector_cap_slots:
            continue
        if theme == "(none)":
            if no_theme_count >= no_theme_cap_slots:
                continue
        elif theme_counts.get(theme, 0) >= theme_cap_slots:
            continue
        kept_rows.append(idx)
        chosen_rows.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if theme == "(none)":
            no_theme_count += 1
        else:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    phases = [
        ("strict", True, True, True),
        ("relax_no_theme_cap", True, True, False),
        ("relax_theme_cap", True, False, False),
        ("relax_sector_cap", False, False, False),
    ]
    ranked_candidates = work.sort_values(["base_weight_score", "buy_rank", "final_score"], ascending=[False, True, False])
    for phase_name, enforce_sector, enforce_theme, enforce_no_theme in phases:
        if len(chosen_rows) >= slot_count:
            break
        for idx, row in ranked_candidates.iterrows():
            if len(chosen_rows) >= slot_count:
                break
            if idx in chosen_rows:
                continue
            sector = row["sector"]
            theme = row["dominant_theme"]
            if enforce_sector and sector_counts.get(sector, 0) >= sector_cap_slots:
                continue
            if enforce_theme:
                if theme == "(none)":
                    if enforce_no_theme and no_theme_count >= no_theme_cap_slots:
                        continue
                elif theme_counts.get(theme, 0) >= theme_cap_slots:
                    continue
            chosen_rows.append(idx)
            selection_mode = phase_name
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if theme == "(none)":
                no_theme_count += 1
            else:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

    selected = work.loc[chosen_rows].copy()
    if selected.empty:
        raise ValueError(f"no holdings selected for target size {target_size}")

    investable_weight = max(0.0, 1.0 - args.cash_buffer)
    raw = selected["base_weight_score"].clip(lower=1e-9)
    selected["target_weight_pre_cap"] = raw / raw.sum() * investable_weight
    selected["target_weight"] = selected["target_weight_pre_cap"]

    max_iter = 20
    for _ in range(max_iter):
        changed = False
        excess = 0.0

        over_name = selected["target_weight"] > max_weight
        if over_name.any():
            excess += (selected.loc[over_name, "target_weight"] - max_weight).sum()
            selected.loc[over_name, "target_weight"] = max_weight
            changed = True

        sector_sum = selected.groupby("sector")["target_weight"].sum()
        for sector, total in sector_sum.items():
            if total <= args.sector_cap + 1e-12:
                continue
            factor = args.sector_cap / total
            mask = selected["sector"].eq(sector)
            before = selected.loc[mask, "target_weight"].sum()
            selected.loc[mask, "target_weight"] *= factor
            excess += before - selected.loc[mask, "target_weight"].sum()
            changed = True

        distinct_non_none_themes = {theme for theme in selected["dominant_theme"].tolist() if theme != "(none)"}
        effective_no_theme_cap = investable_weight if not distinct_non_none_themes else args.no_theme_cap
        theme_sum = selected.groupby("dominant_theme")["target_weight"].sum()
        for theme, total in theme_sum.items():
            cap = effective_no_theme_cap if theme == "(none)" else args.theme_cap
            if total <= cap + 1e-12:
                continue
            factor = cap / total
            mask = selected["dominant_theme"].eq(theme)
            before = selected.loc[mask, "target_weight"].sum()
            selected.loc[mask, "target_weight"] *= factor
            excess += before - selected.loc[mask, "target_weight"].sum()
            changed = True

        if excess <= 1e-12:
            if not changed:
                break
            continue

        headroom = pd.Series(max_weight, index=selected.index) - selected["target_weight"]
        sector_headroom = selected["sector"].map(
            lambda x: max(0.0, args.sector_cap - float(selected.loc[selected["sector"].eq(x), "target_weight"].sum()))
        )
        theme_headroom = selected["dominant_theme"].map(
            lambda x: max(
                0.0,
                (effective_no_theme_cap if x == "(none)" else args.theme_cap)
                - float(selected.loc[selected["dominant_theme"].eq(x), "target_weight"].sum()),
            )
        )
        effective_room = pd.concat([headroom, sector_headroom, theme_headroom], axis=1).min(axis=1).clip(lower=0.0)
        receivers = effective_room > 1e-12
        if not receivers.any():
            break
        receiver_score = selected.loc[receivers, "base_weight_score"] * effective_room.loc[receivers]
        receiver_score = receiver_score / receiver_score.sum()
        add = receiver_score * excess
        selected.loc[receivers, "target_weight"] += add

    selected["target_weight"] = selected["target_weight"].clip(lower=0.0)
    total_weight = selected["target_weight"].sum()
    if total_weight > investable_weight + 1e-9:
        selected["target_weight"] *= investable_weight / total_weight
        total_weight = selected["target_weight"].sum()

    min_keep_mask = selected["keep_from_previous"] & selected["target_weight"].lt(args.min_keep_weight)
    if min_keep_mask.any():
        uplift = args.min_keep_weight - selected.loc[min_keep_mask, "target_weight"]
        required = uplift.sum()
        donors = selected.index.difference(selected.loc[min_keep_mask].index)
        donor_room = (selected.loc[donors, "target_weight"] - 0.0).clip(lower=0.0)
        if required > 0 and donor_room.sum() > required:
            donor_cut = donor_room / donor_room.sum() * required
            selected.loc[donors, "target_weight"] -= donor_cut
            selected.loc[min_keep_mask, "target_weight"] = args.min_keep_weight

    selected["target_weight"] = selected["target_weight"].clip(lower=0.0)
    selected["portfolio_cash_weight"] = max(0.0, 1.0 - selected["target_weight"].sum())
    selected["turnover_action"] = selected["keep_from_previous"].map(lambda x: "kept" if x else "new")
    selected["portfolio_rank"] = selected["target_weight"].rank(method="first", ascending=False).astype(int)
    selected = selected.sort_values(["portfolio_rank", "buy_rank", "code"]).reset_index(drop=True)

    summary = {
        "holdings": len(selected),
        "cash_weight": float(selected["portfolio_cash_weight"].iloc[0]),
        "kept_count": int(selected["keep_from_previous"].sum()),
        "new_count": int((~selected["keep_from_previous"]).sum()),
        "sector_max_weight": float(selected.groupby("sector")["target_weight"].sum().max()),
        "theme_max_weight": float(selected.groupby("dominant_theme")["target_weight"].sum().max()),
        "selection_mode": selection_mode,
    }
    return selected, summary


def finalize_output(portfolio: pd.DataFrame, *, label: str, summary: dict[str, object]) -> pd.DataFrame:
    out = portfolio.copy()
    out["strategy"] = label
    out["portfolio_version"] = PORTFOLIO_VERSION
    out["score_formula_version"] = out.get("score_formula_version", SCORE_FORMULA_VERSION)
    out["cash_buffer"] = summary["cash_weight"]
    out["sector_weight"] = out.groupby("sector")["target_weight"].transform("sum")
    out["theme_weight"] = out.groupby("dominant_theme")["target_weight"].transform("sum")
    out["target_weight_pct"] = out["target_weight"] * 100.0
    out["cash_buffer_pct"] = out["cash_buffer"] * 100.0
    return out[
        [
            "strategy",
            "portfolio_version",
            "score_formula_version",
            "asof_date",
            "portfolio_rank",
            "buy_rank",
            "code",
            "name",
            "sector",
            "dominant_theme",
            "final_score",
            "confidence_score",
            "liquidity_score",
            "trading_value",
            "target_weight",
            "target_weight_pct",
            "sector_weight",
            "theme_weight",
            "cash_buffer",
            "cash_buffer_pct",
            "keep_from_previous",
            "turnover_action",
            "selection_stage",
            "weight_reason",
            "explain_text",
        ]
    ]


def build_report(
    portfolios: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, object]],
    *,
    generated_at: str,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = [
        "# Portfolio Construction Report",
        "",
        f"- generated_at: {generated_at}",
        f"- portfolio_version: {PORTFOLIO_VERSION}",
        f"- score_formula_version: {SCORE_FORMULA_VERSION}",
        f"- cash_buffer: {_fmt_pct(args.cash_buffer)}",
        f"- sector_cap: {_fmt_pct(args.sector_cap)}",
        f"- theme_cap: {_fmt_pct(args.theme_cap)}",
        f"- no_theme_cap: {_fmt_pct(args.no_theme_cap)}",
        f"- turnover_keep_slots: {args.turnover_keep_slots}",
        "",
        "## Construction Rules",
        "- Weight seed is a composite of final_score, confidence_score, and a liquidity haircut.",
        "- Position weight is capped by strategy-level max weight and rebalanced under sector/theme caps.",
        "- Low-liquidity names receive a haircut before allocation, not a hard exclusion.",
        "- A cash buffer is reserved by default, and a small number of prior holdings can be retained to reduce turnover.",
        "",
    ]

    summary_rows = []
    for strategy, summary in summaries.items():
        summary_rows.append(
            {
                "strategy": strategy,
                "holdings": summary["holdings"],
                "cash_weight": _fmt_pct(summary["cash_weight"]),
                "kept_count": summary["kept_count"],
                "new_count": summary["new_count"],
                "sector_max_weight": _fmt_pct(summary["sector_max_weight"]),
                "theme_max_weight": _fmt_pct(summary["theme_max_weight"]),
                "selection_mode": summary["selection_mode"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    lines.extend(["## Strategy Summary", _markdown_table(summary_df, list(summary_df.columns)), ""])

    for strategy, frame in portfolios.items():
        shown = frame.copy()
        shown["target_weight_pct"] = shown["target_weight"].map(lambda x: f"{x * 100:.2f}%")
        shown["sector_weight"] = shown["sector_weight"].map(lambda x: f"{x * 100:.2f}%")
        shown["theme_weight"] = shown["theme_weight"].map(lambda x: f"{x * 100:.2f}%")
        shown["keep_from_previous"] = shown["keep_from_previous"].map(lambda x: "yes" if bool(x) else "no")
        lines.extend(
            [
                f"## {strategy.upper()} Portfolio",
                _markdown_table(
                    shown,
                    [
                        "portfolio_rank",
                        "code",
                        "name",
                        "target_weight_pct",
                        "sector",
                        "sector_weight",
                        "dominant_theme",
                        "theme_weight",
                        "confidence_score",
                        "liquidity_score",
                        "turnover_action",
                        "selection_stage",
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def run_one(
    *,
    strategy_label: str,
    target_size: int,
    input_path: Path,
    output_path: Path,
    max_weight: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = load_candidates(input_path, target_size=target_size)
    previous = load_previous_portfolio(output_path)
    portfolio, summary = allocate_weights(
        candidates,
        target_size=target_size,
        max_weight=max_weight,
        previous_portfolio=previous,
        args=args,
    )
    out = finalize_output(portfolio, label=strategy_label, summary=summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out, summary


def main() -> int:
    args = parse_args()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    portfolios: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, object]] = {}

    top5, top5_summary = run_one(
        strategy_label="top5",
        target_size=5,
        input_path=args.input_top5,
        output_path=args.out_top5,
        max_weight=args.max_weight_top5,
        args=args,
    )
    portfolios["top5"] = top5
    summaries["top5"] = top5_summary

    top8, top8_summary = run_one(
        strategy_label="top8",
        target_size=8,
        input_path=args.input_top8,
        output_path=args.out_top8,
        max_weight=args.max_weight_top8,
        args=args,
    )
    portfolios["top8"] = top8
    summaries["top8"] = top8_summary

    if args.input_top10.exists():
        top10, top10_summary = run_one(
            strategy_label="top10",
            target_size=10,
            input_path=args.input_top10,
            output_path=args.out_top10,
            max_weight=args.max_weight_top10,
            args=args,
        )
        portfolios["top10"] = top10
        summaries["top10"] = top10_summary

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_report(portfolios, summaries, generated_at=generated_at, args=args),
        encoding="utf-8",
    )

    print(f"top5_csv: {args.out_top5}")
    print(f"top8_csv: {args.out_top8}")
    if args.input_top10.exists():
        print(f"top10_csv: {args.out_top10}")
    print(f"report_md: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
