from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_theme_overlay_acceptance_report import (
    build_explain_consistency_sample,
    build_new_entry_quality,
    build_no_theme_retention,
    build_theme_concentration,
    build_top20_churn,
    load_latest_ranking,
    load_latest_stock_theme_date,
)
from outcome_maturity import attach_forward_outcomes, load_price_history


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

HISTORY_DIR = DATA_DIR / "history" / "ranking"
PRICES_CSV = DATA_DIR / "prices_daily_adjusted.csv"
OUTPUT_MD = OUTPUT_DIR / "theme_overlay_acceptance_operational.md"

TOP_N = 20
NEAR_TOP_N = 40
HORIZONS = [5, 20, 60, 90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run operational theme overlay acceptance gates before promotion.")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--prices-csv", type=Path, default=PRICES_CSV)
    parser.add_argument("--out-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--min-matured-dates", type=int, default=3)
    parser.add_argument("--max-churn-ratio", type=float, default=0.40)
    parser.add_argument("--min-no-theme-retention", type=float, default=0.50)
    parser.add_argument("--max-theme-share", type=float, default=0.35)
    parser.add_argument("--max-theme-share-delta", type=float, default=0.15)
    parser.add_argument("--min-entry-explainable-ratio", type=float, default=0.60)
    parser.add_argument("--min-explain-consistency-ratio", type=float, default=0.80)
    parser.add_argument("--max-no-theme-themeword-ratio", type=float, default=0.10)
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


def resolve_overlay_score_column(df: pd.DataFrame) -> str:
    preferred = []
    if "shadow_final_score_v3" in df.columns:
        preferred.append("shadow_final_score_v3")
    if "final_score_v3" in df.columns:
        preferred.append("final_score_v3")
    preferred.append("final_score")

    baseline = pd.to_numeric(df.get("final_score"), errors="coerce")
    for column in preferred:
        score = pd.to_numeric(df.get(column), errors="coerce")
        if score.notna().any():
            diff = (score - baseline).abs()
            if diff.fillna(0).gt(1e-9).any():
                return column
    for column in preferred:
        if column in df.columns:
            return column
    return "final_score"


def normalize_snapshot(df: pd.DataFrame, *, top_n: int) -> tuple[pd.DataFrame, str]:
    work = df.copy()
    if "date" not in work.columns:
        raise ValueError("snapshot must contain date")
    latest_date = pd.to_datetime(work["date"], errors="coerce").max()
    work = work.loc[pd.to_datetime(work["date"], errors="coerce").eq(latest_date)].copy()
    work["date"] = latest_date.strftime("%Y-%m-%d")
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["name"] = work["name"].fillna("").astype(str) if "name" in work.columns else pd.Series("", index=work.index, dtype="object")
    work["dominant_theme"] = (
        work["dominant_theme"].fillna("(none)").astype(str).replace({"": "(none)", "nan": "(none)"})
        if "dominant_theme" in work.columns
        else pd.Series("(none)", index=work.index, dtype="object")
    )
    work["explain_text"] = (
        work["explain_text"].fillna("").astype(str)
        if "explain_text" in work.columns
        else pd.Series("", index=work.index, dtype="object")
    )
    for column in ["final_score", "final_score_v3", "shadow_final_score_v3", "ret_score", "confidence_score", "tech_score", "theme_confidence"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")

    overlay_score_column = resolve_overlay_score_column(work)
    work["overlay_score_eval"] = pd.to_numeric(work.get(overlay_score_column), errors="coerce")
    work["overlay_score_eval"] = work["overlay_score_eval"].fillna(pd.to_numeric(work["final_score"], errors="coerce"))
    work["baseline_rank"] = pd.to_numeric(work["final_score"], errors="coerce").rank(method="first", ascending=False).astype(int)
    work["overlay_rank"] = pd.to_numeric(work["overlay_score_eval"], errors="coerce").rank(method="first", ascending=False).astype(int)
    work["is_no_theme"] = work["dominant_theme"].eq("(none)")
    work["has_theme_explain"] = work.apply(
        lambda row: (
            (str(row["dominant_theme"]).strip() not in {"", "(none)"})
            and (
                str(row["dominant_theme"]).strip() in str(row["explain_text"])
                or "theme=" in str(row["explain_text"]).lower()
            )
        ),
        axis=1,
    )
    work["quality_composite"] = work[["ret_score", "confidence_score", "tech_score"]].mean(axis=1, skipna=True)
    work["in_baseline_top20"] = work["baseline_rank"].le(top_n)
    work["in_overlay_top20"] = work["overlay_rank"].le(top_n)
    work["in_baseline_near_top20"] = work["baseline_rank"].between(top_n + 1, NEAR_TOP_N)
    return work, overlay_score_column


def load_history_snapshots(history_dir: Path, *, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    if not history_dir.exists():
        raise FileNotFoundError(f"history dir not found: {history_dir}")
    frames: list[pd.DataFrame] = []
    columns_used: list[str] = []
    for path in sorted(history_dir.glob("*_ranking_final.csv")):
        df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if df.empty or "date" not in df.columns or "code" not in df.columns:
            continue
        snapshot, overlay_col = normalize_snapshot(df, top_n=top_n)
        snapshot["snapshot_file"] = path.name
        snapshot["overlay_score_column_used"] = overlay_col
        columns_used.append(overlay_col)
        frames.append(snapshot)
    if not frames:
        raise ValueError("no usable historical ranking snapshots found")
    history = pd.concat(frames, ignore_index=True)
    return history, sorted(set(columns_used))


def evaluate_static_gates(latest: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    latest = latest.copy()
    latest["in_baseline_top20"] = latest["baseline_rank"].le(args.top_n)
    latest["in_overlay_top20"] = latest["overlay_rank"].le(args.top_n)
    latest["in_baseline_near_top20"] = latest["baseline_rank"].between(args.top_n + 1, NEAR_TOP_N)
    latest["quality_composite"] = latest[["ret_score", "confidence_score", "tech_score"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1, skipna=True)

    churn_df, churn_meta = build_top20_churn(latest)
    no_theme_df, no_theme_meta = build_no_theme_retention(latest)
    concentration_df, concentration_meta = build_theme_concentration(latest)
    new_entry_df, new_entry_meta = build_new_entry_quality(latest)
    explain_df, explain_meta = build_explain_consistency_sample(latest)

    overlay_top = latest.loc[latest["overlay_rank"].le(args.top_n)].copy()
    baseline_top = latest.loc[latest["baseline_rank"].le(args.top_n)].copy()
    overlay_no_theme_count = int(overlay_top["is_no_theme"].sum())
    baseline_no_theme_count = int(baseline_top["is_no_theme"].sum())

    overlay_entries = latest.loc[latest["in_overlay_top20"] & ~latest["in_baseline_top20"]].copy()
    exited = latest.loc[latest["in_baseline_top20"] & ~latest["in_overlay_top20"]].copy()
    entered_avg_quality = float(pd.to_numeric(overlay_entries["quality_composite"], errors="coerce").mean()) if not overlay_entries.empty else None
    exited_avg_quality = float(pd.to_numeric(exited["quality_composite"], errors="coerce").mean()) if not exited.empty else None
    entry_quality_delta = (
        None
        if overlay_entries.empty or exited.empty
        else float(entered_avg_quality - exited_avg_quality)
    )

    themed_overlay = overlay_top.loc[~overlay_top["is_no_theme"]].copy()
    explain_consistency_ratio = (
        float(themed_overlay["has_theme_explain"].mean()) if not themed_overlay.empty else 1.0
    )
    no_theme_themeword_ratio = (
        float(
            overlay_top.loc[overlay_top["is_no_theme"], "explain_text"]
            .astype(str)
            .str.lower()
            .str.contains("theme=")
            .mean()
        )
        if overlay_no_theme_count > 0
        else 0.0
    )

    baseline_theme_counts = baseline_top.loc[~baseline_top["is_no_theme"], "dominant_theme"].value_counts()
    overlay_theme_counts = overlay_top.loc[~overlay_top["is_no_theme"], "dominant_theme"].value_counts()
    baseline_theme_max_share = float((baseline_theme_counts / args.top_n).max()) if not baseline_theme_counts.empty else 0.0
    overlay_theme_max_share = float((overlay_theme_counts / args.top_n).max()) if not overlay_theme_counts.empty else 0.0
    theme_share_delta = overlay_theme_max_share - baseline_theme_max_share
    overlay_max_theme = str(overlay_theme_counts.index[0]) if not overlay_theme_counts.empty else "(none)"

    gate_rows = [
        {
            "gate_name": "top20_churn_not_excessive",
            "status": "PASS" if float(churn_meta["churn_ratio"]) <= args.max_churn_ratio else "FAIL",
            "rule": f"churn_ratio <= {_fmt_pct(args.max_churn_ratio)}",
            "value": _fmt_pct(churn_meta["churn_ratio"]),
            "detail": f"entered={churn_meta['entered_count']}, exited={churn_meta['exited_count']}",
        },
        {
            "gate_name": "no_theme_not_fully_excluded",
            "status": "PASS"
            if baseline_no_theme_count == 0
            or overlay_no_theme_count > 0
            or float(no_theme_meta["retention_ratio"]) >= args.min_no_theme_retention
            else "FAIL",
            "rule": f"overlay_no_theme_count > 0 or retention_ratio >= {_fmt_pct(args.min_no_theme_retention)}",
            "value": f"overlay_no_theme={overlay_no_theme_count}, retention={_fmt_pct(no_theme_meta['retention_ratio'])}",
            "detail": f"baseline_no_theme={baseline_no_theme_count}",
        },
        {
            "gate_name": "near_top20_entry_quality_improves",
            "status": "PASS"
            if overlay_entries.empty
            or (
                (entry_quality_delta is not None and entry_quality_delta >= 0.0)
                and float(new_entry_meta["explainable_ratio"]) >= args.min_entry_explainable_ratio
            )
            else "FAIL",
            "rule": f"entry_quality_delta >= 0 and explainable_ratio >= {_fmt_pct(args.min_entry_explainable_ratio)}",
            "value": f"entry_quality_delta={_fmt(entry_quality_delta)}, explainable_ratio={_fmt_pct(new_entry_meta['explainable_ratio'])}",
            "detail": f"entries={len(overlay_entries)}, exited={len(exited)}",
        },
        {
            "gate_name": "theme_concentration_not_excessive",
            "status": "PASS"
            if overlay_theme_max_share <= args.max_theme_share and theme_share_delta <= args.max_theme_share_delta
            else "FAIL",
            "rule": f"overlay_max_share <= {_fmt_pct(args.max_theme_share)} and delta <= {_fmt_pct(args.max_theme_share_delta)}",
            "value": f"overlay={_fmt_pct(overlay_theme_max_share)}, baseline={_fmt_pct(baseline_theme_max_share)}, delta={_fmt_pct(theme_share_delta)}",
            "detail": f"max_theme={overlay_max_theme}",
        },
        {
            "gate_name": "explain_consistency_maintained",
            "status": "PASS"
            if explain_consistency_ratio >= args.min_explain_consistency_ratio
            and no_theme_themeword_ratio <= args.max_no_theme_themeword_ratio
            else "FAIL",
            "rule": f"themed_consistency >= {_fmt_pct(args.min_explain_consistency_ratio)} and no_theme_themeword <= {_fmt_pct(args.max_no_theme_themeword_ratio)}",
            "value": f"themed_consistency={_fmt_pct(explain_consistency_ratio)}, no_theme_themeword={_fmt_pct(no_theme_themeword_ratio)}",
            "detail": f"sample_manual_review={explain_meta['review_count']}/{explain_meta['sample_count']}",
        },
    ]
    gates = pd.DataFrame(gate_rows)
    details = {
        "churn_df": churn_df,
        "no_theme_df": no_theme_df,
        "concentration_df": concentration_df,
        "new_entry_df": new_entry_df,
        "explain_df": explain_df,
        "overlay_entries": overlay_entries,
        "exited": exited,
        "overlay_no_theme_count": overlay_no_theme_count,
        "baseline_no_theme_count": baseline_no_theme_count,
        "entry_quality_delta": entry_quality_delta,
        "explain_consistency_ratio": explain_consistency_ratio,
        "no_theme_themeword_ratio": no_theme_themeword_ratio,
        "theme_share_delta": theme_share_delta,
    }
    return gates, details


def build_operational_forward_gate(history: pd.DataFrame, prices_csv: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    prices = load_price_history(prices_csv=prices_csv)
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    outcome_frames = {
        horizon: attach_forward_outcomes(prices, horizon_days=horizon).rename(
            columns={"date": "asof_date", "realized_return": f"forward_return_{horizon}d"}
        )
        for horizon in HORIZONS
    }

    rows: list[dict[str, object]] = []
    per_date_rows: list[dict[str, object]] = []
    dates = sorted(history["date"].dropna().unique().tolist())
    for horizon in HORIZONS:
        diffs: list[float] = []
        hit_diffs: list[float] = []
        matured_dates = 0
        for date in dates:
            day = history.loc[history["date"] == date].copy()
            outcome = outcome_frames[horizon]
            joined = day.merge(
                outcome[["code", "asof_date", f"forward_return_{horizon}d"]],
                left_on=["code", "date"],
                right_on=["code", "asof_date"],
                how="left",
            )
            ret_col = f"forward_return_{horizon}d"
            baseline = joined.loc[joined["baseline_rank"].le(args.top_n)].copy()
            overlay = joined.loc[joined["overlay_rank"].le(args.top_n)].copy()
            baseline_ret = pd.to_numeric(baseline[ret_col], errors="coerce")
            overlay_ret = pd.to_numeric(overlay[ret_col], errors="coerce")
            baseline_matured = int(baseline_ret.notna().sum())
            overlay_matured = int(overlay_ret.notna().sum())
            if baseline_matured < args.top_n or overlay_matured < args.top_n:
                per_date_rows.append(
                    {
                        "date": date,
                        "horizon": f"{horizon}d",
                        "maturity_status": "immature",
                        "baseline_avg_return": None,
                        "overlay_avg_return": None,
                        "overlay_minus_baseline": None,
                        "baseline_hit_rate": None,
                        "overlay_hit_rate": None,
                        "overlay_minus_baseline_hit_rate": None,
                    }
                )
                continue
            matured_dates += 1
            baseline_avg = float(baseline_ret.mean())
            overlay_avg = float(overlay_ret.mean())
            baseline_hit = float((baseline_ret > 0).mean())
            overlay_hit = float((overlay_ret > 0).mean())
            diff = overlay_avg - baseline_avg
            hit_diff = overlay_hit - baseline_hit
            diffs.append(diff)
            hit_diffs.append(hit_diff)
            per_date_rows.append(
                {
                    "date": date,
                    "horizon": f"{horizon}d",
                    "maturity_status": f"matured_{horizon}d",
                    "baseline_avg_return": baseline_avg,
                    "overlay_avg_return": overlay_avg,
                    "overlay_minus_baseline": diff,
                    "baseline_hit_rate": baseline_hit,
                    "overlay_hit_rate": overlay_hit,
                    "overlay_minus_baseline_hit_rate": hit_diff,
                }
            )
        rows.append(
            {
                "horizon": f"{horizon}d",
                "matured_dates": matured_dates,
                "avg_overlay_minus_baseline_return": float(pd.Series(diffs).mean()) if diffs else None,
                "median_overlay_minus_baseline_return": float(pd.Series(diffs).median()) if diffs else None,
                "positive_diff_date_ratio": float((pd.Series(diffs) > 0).mean()) if diffs else None,
                "avg_overlay_minus_baseline_hit_rate": float(pd.Series(hit_diffs).mean()) if hit_diffs else None,
            }
        )

    summary = pd.DataFrame(rows)
    eligible = summary.loc[summary["matured_dates"] >= args.min_matured_dates].copy()
    if eligible.empty:
        gate_status = "FAIL"
        gate_value = "insufficient_matured_dates"
        gate_detail = f"need >= {args.min_matured_dates} matured dates on at least one horizon"
    else:
        avg_return_delta = float(pd.to_numeric(eligible["avg_overlay_minus_baseline_return"], errors="coerce").mean())
        avg_hit_delta = float(pd.to_numeric(eligible["avg_overlay_minus_baseline_hit_rate"], errors="coerce").mean())
        positive_ratio = float(pd.to_numeric(eligible["positive_diff_date_ratio"], errors="coerce").mean())
        gate_status = "PASS" if avg_return_delta > 0 and avg_hit_delta >= 0 and positive_ratio >= 0.50 else "FAIL"
        gate_value = (
            f"avg_return_delta={_fmt_pct(avg_return_delta)}, "
            f"avg_hit_delta={_fmt_pct(avg_hit_delta)}, "
            f"positive_date_ratio={_fmt_pct(positive_ratio)}"
        )
        gate_detail = f"eligible_horizons={','.join(eligible['horizon'].astype(str).tolist())}"

    gate = pd.DataFrame(
        [
            {
                "gate_name": "operational_forward_return_improves_vs_baseline",
                "status": gate_status,
                "rule": "at least one horizon with enough matured dates and positive overlay-minus-baseline return improvement",
                "value": gate_value,
                "detail": gate_detail,
            }
        ]
    )
    return gate, {"summary": summary, "per_date": pd.DataFrame(per_date_rows), "eligible": eligible}


def final_status(gates: pd.DataFrame) -> str:
    return "PASS" if gates["status"].eq("PASS").all() else "FAIL"


def build_markdown(
    latest: pd.DataFrame,
    overlay_meta: dict[str, object],
    overlay_columns_used: list[str],
    static_gates: pd.DataFrame,
    static_details: dict[str, object],
    forward_gate: pd.DataFrame,
    forward_details: dict[str, object],
    args: argparse.Namespace,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_gates = pd.concat([static_gates, forward_gate], ignore_index=True)
    decision = final_status(all_gates)

    gate_table = all_gates.copy()
    forward_summary = forward_details["summary"].copy()
    if not forward_summary.empty:
        for col in [
            "avg_overlay_minus_baseline_return",
            "median_overlay_minus_baseline_return",
            "positive_diff_date_ratio",
            "avg_overlay_minus_baseline_hit_rate",
        ]:
            if "ratio" in col or "hit_rate" in col or "return" in col:
                forward_summary[col] = forward_summary[col].map(_fmt_pct)

    entry_preview = static_details["overlay_entries"].copy()
    if not entry_preview.empty:
        for col in ["quality_composite", "ret_score", "confidence_score", "tech_score", "theme_confidence"]:
            if col in entry_preview.columns:
                entry_preview[col] = entry_preview[col].map(_fmt)

    per_date_preview = forward_details["per_date"].copy()
    if not per_date_preview.empty:
        for col in [
            "baseline_avg_return",
            "overlay_avg_return",
            "overlay_minus_baseline",
            "baseline_hit_rate",
            "overlay_hit_rate",
            "overlay_minus_baseline_hit_rate",
        ]:
            per_date_preview[col] = per_date_preview[col].map(_fmt_pct)

    lines = [
        "# Theme Overlay Acceptance Operational",
        "",
        f"- generated_at: {generated_at}",
        f"- latest_ranking_date: {latest['date'].max() if not latest.empty else 'NA'}",
        f"- stock_theme_daily_latest_date: {load_latest_stock_theme_date()}",
        f"- resolved_mode: {overlay_meta['resolved_mode']}",
        f"- evaluation_profile: {overlay_meta['evaluation_profile']}",
        f"- overlay_score_column_used_latest: {overlay_meta['overlay_score_column']}",
        f"- overlay_score_columns_seen_in_history: {', '.join(overlay_columns_used)}",
        f"- latest_row_count: {len(latest)}",
        "",
        "## Decision",
        f"- final_gate_status: {decision}",
        f"- pass_gate_count: {int(all_gates['status'].eq('PASS').sum())}",
        f"- fail_gate_count: {int(all_gates['status'].eq('FAIL').sum())}",
        "",
        "## Gate Table",
        _markdown_table(gate_table, ["gate_name", "status", "rule", "value", "detail"]),
        "",
        "## Static Acceptance Context",
        f"- overlay_no_theme_count_top20: {static_details['overlay_no_theme_count']}",
        f"- baseline_no_theme_count_top20: {static_details['baseline_no_theme_count']}",
        f"- entry_quality_delta: {_fmt(static_details['entry_quality_delta'])}",
        f"- explain_consistency_ratio: {_fmt_pct(static_details['explain_consistency_ratio'])}",
        f"- no_theme_themeword_ratio: {_fmt_pct(static_details['no_theme_themeword_ratio'])}",
        f"- theme_share_delta: {_fmt_pct(static_details['theme_share_delta'])}",
        "",
        "## Operational Forward Summary",
        _markdown_table(
            forward_summary,
            [
                "horizon",
                "matured_dates",
                "avg_overlay_minus_baseline_return",
                "median_overlay_minus_baseline_return",
                "positive_diff_date_ratio",
                "avg_overlay_minus_baseline_hit_rate",
            ],
        ),
    ]

    if not entry_preview.empty:
        lines.extend(
            [
                "",
                "## Near-Top20 Entries",
                _markdown_table(
                    entry_preview,
                    [
                        "date",
                        "code",
                        "name",
                        "baseline_rank",
                        "overlay_rank",
                        "quality_composite",
                        "ret_score",
                        "confidence_score",
                        "tech_score",
                        "dominant_theme",
                        "theme_confidence",
                        "has_theme_explain",
                    ],
                ),
            ]
        )

    if not per_date_preview.empty:
        lines.extend(
            [
                "",
                "## Forward By Date",
                _markdown_table(
                    per_date_preview,
                    [
                        "date",
                        "horizon",
                        "maturity_status",
                        "baseline_avg_return",
                        "overlay_avg_return",
                        "overlay_minus_baseline",
                        "baseline_hit_rate",
                        "overlay_hit_rate",
                        "overlay_minus_baseline_hit_rate",
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- This gate is promotion-oriented. Any single FAIL blocks operational promotion.",
            "- The operational forward-return gate requires matured historical snapshots. If that evidence is missing, the gate fails by design rather than assuming improvement.",
            "- The current framework compares baseline `final_score` top20 against overlay score top20 reconstructed from historical ranking snapshots.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    latest, _, overlay_meta = load_latest_ranking()
    static_gates, static_details = evaluate_static_gates(latest, args)
    history, overlay_columns_used = load_history_snapshots(
        args.history_dir if args.history_dir.is_absolute() else ROOT / args.history_dir,
        top_n=args.top_n,
    )
    forward_gate, forward_details = build_operational_forward_gate(
        history,
        prices_csv=args.prices_csv if args.prices_csv.is_absolute() else ROOT / args.prices_csv,
        args=args,
    )

    out_md = args.out_md if args.out_md.is_absolute() else ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        build_markdown(
            latest=latest,
            overlay_meta=overlay_meta,
            overlay_columns_used=overlay_columns_used,
            static_gates=static_gates,
            static_details=static_details,
            forward_gate=forward_gate,
            forward_details=forward_details,
            args=args,
        ),
        encoding="utf-8",
    )

    all_gates = pd.concat([static_gates, forward_gate], ignore_index=True)
    print(f"final_gate_status: {final_status(all_gates)}")
    print(f"out_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
