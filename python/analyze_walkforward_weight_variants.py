from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from build_walk_forward_score_validation import attach_realized_outcomes, build_rank_history  # noqa: E402


OUTPUT_MD = ROOT / "outputs" / "walkforward_weight_variant_analysis.md"
OUTPUT_CSV = ROOT / "outputs" / "walkforward_weight_variant_metrics.csv"
TARGET_HORIZON = 60
MIN_RUN_ROWS = 50
TOP_N = 20
TOP50_N = 50


@dataclass(frozen=True)
class VariantProfile:
    name: str
    description: str
    bull: tuple[float, float, float, float, float]
    neutral: tuple[float, float, float, float, float]
    defensive: tuple[float, float, float, float, float]
    mode: str = "weights"


BASELINE_PROFILE = VariantProfile(
    name="baseline",
    description="Current production operating weights.",
    bull=(0.38, 0.27, 0.27, 0.08, 0.40),
    neutral=(0.32, 0.26, 0.24, 0.18, 0.65),
    defensive=(0.26, 0.22, 0.18, 0.34, 0.80),
)

REBALANCED_PROFILE = VariantProfile(
    name="rebalanced_weights",
    description="Reduce same-date tech/prob dominance and raise quality/risk control.",
    bull=(0.38, 0.22, 0.20, 0.20, 0.60),
    neutral=(0.32, 0.22, 0.18, 0.28, 0.85),
    defensive=(0.24, 0.18, 0.12, 0.46, 1.00),
)

QUALITY_RISK_GUARD_PROFILE = VariantProfile(
    name="quality_risk_guard",
    description="Keep baseline core but add extra soft penalty for low quality / high risk names.",
    bull=BASELINE_PROFILE.bull,
    neutral=BASELINE_PROFILE.neutral,
    defensive=BASELINE_PROFILE.defensive,
    mode="guard",
)


def _fmt_pct(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _fmt_num(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(item) for item in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def _apply_variant_score(df: pd.DataFrame, profile: VariantProfile) -> pd.DataFrame:
    out = df.copy()
    regime = out.get("regime", pd.Series("defensive", index=out.index)).fillna("defensive").astype(str).str.lower()

    def _select_weights(item: str) -> tuple[float, float, float, float, float]:
        if item == "bull":
            return profile.bull
        if item == "neutral":
            return profile.neutral
        return profile.defensive

    weights = regime.map(_select_weights)
    out["variant_w_ret"] = weights.map(lambda item: item[0])
    out["variant_w_prob"] = weights.map(lambda item: item[1])
    out["variant_w_tech"] = weights.map(lambda item: item[2])
    out["variant_w_qual"] = weights.map(lambda item: item[3])
    out["variant_w_risk"] = weights.map(lambda item: item[4])

    raw = (
        out["variant_w_ret"] * pd.to_numeric(out["ret_score"], errors="coerce").fillna(0.0)
        + out["variant_w_prob"] * pd.to_numeric(out["prob_score"], errors="coerce").fillna(0.0)
        + out["variant_w_tech"] * pd.to_numeric(out["tech_score"], errors="coerce").fillna(0.0)
        + out["variant_w_qual"] * pd.to_numeric(out["qual_score"], errors="coerce").fillna(0.0)
        - out["variant_w_risk"] * pd.to_numeric(out["risk_penalty"], errors="coerce").fillna(0.0)
    )

    if profile.mode == "guard":
        qual = pd.to_numeric(out["qual_score"], errors="coerce")
        risk = pd.to_numeric(out["risk_penalty"], errors="coerce")
        guard_penalty = np.where(qual < 20.0, 6.0, 0.0) + np.where(risk >= 12.0, 4.0, 0.0)
        raw = raw - pd.Series(guard_penalty, index=out.index, dtype="float64")

    out["variant_score"] = pd.to_numeric(raw, errors="coerce").clip(lower=0.0, upper=100.0)
    out["variant_rank"] = out["variant_score"].rank(method="first", ascending=False)
    return out


def _selection_metrics(df: pd.DataFrame, rank_col: str, score_col: str) -> dict[str, object]:
    work = df.copy()
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
    top20 = work.loc[work[rank_col] <= TOP_N].copy()
    top50 = work.loc[work[rank_col] <= TOP50_N].copy()
    universe = work.copy()
    realized = pd.to_numeric(work["realized_return_60d"], errors="coerce")
    score = pd.to_numeric(work[score_col], errors="coerce")
    valid = realized.notna() & score.notna()

    return {
        "top20_avg_return": float(pd.to_numeric(top20["realized_return_60d"], errors="coerce").mean()),
        "top50_avg_return": float(pd.to_numeric(top50["realized_return_60d"], errors="coerce").mean()),
        "universe_avg_return": float(pd.to_numeric(universe["realized_return_60d"], errors="coerce").mean()),
        "top20_hit_rate": float((pd.to_numeric(top20["realized_return_60d"], errors="coerce") > 0).mean()),
        "top50_hit_rate": float((pd.to_numeric(top50["realized_return_60d"], errors="coerce") > 0).mean()),
        "top20_avg_mdd": float(pd.to_numeric(top20["realized_mdd_60d"], errors="coerce").mean()),
        "top50_avg_mdd": float(pd.to_numeric(top50["realized_mdd_60d"], errors="coerce").mean()),
        "ordering_ok": bool(
            pd.notna(pd.to_numeric(top20["realized_return_60d"], errors="coerce").mean())
            and pd.notna(pd.to_numeric(top50["realized_return_60d"], errors="coerce").mean())
            and float(pd.to_numeric(top20["realized_return_60d"], errors="coerce").mean())
            > float(pd.to_numeric(top50["realized_return_60d"], errors="coerce").mean())
            > float(pd.to_numeric(universe["realized_return_60d"], errors="coerce").mean())
        ),
        "score_return_corr": float(score[valid].corr(realized[valid])) if valid.sum() >= 3 else float("nan"),
        "top20_low_qual_count": int((pd.to_numeric(top20["qual_score"], errors="coerce") < 20.0).sum()),
        "top20_high_risk_count": int((pd.to_numeric(top20["risk_penalty"], errors="coerce") >= 12.0).sum()),
    }


def _build_latest_run() -> tuple[pd.DataFrame, str]:
    ranked, _ = build_rank_history()
    work = attach_realized_outcomes(ranked, TARGET_HORIZON)
    work = work.loc[pd.to_numeric(work[f"realized_return_{TARGET_HORIZON}d"], errors="coerce").notna()].copy()
    eligible_dates = work.groupby("date")["code"].size()
    eligible_dates = eligible_dates[eligible_dates >= MIN_RUN_ROWS]
    work = work.loc[work["date"].isin(eligible_dates.index)].copy()
    latest = str(work["date"].max())
    run = work.loc[work["date"] == latest].copy()
    run["baseline_rank"] = pd.to_numeric(run["final_score"], errors="coerce").rank(method="first", ascending=False)
    return run, latest


def build_report() -> tuple[pd.DataFrame, str]:
    latest_run, latest_date = _build_latest_run()
    variants = [BASELINE_PROFILE, REBALANCED_PROFILE, QUALITY_RISK_GUARD_PROFILE]

    metric_rows: list[dict[str, object]] = []
    detail_lines: list[str] = []
    baseline_top20 = set(latest_run.loc[latest_run["baseline_rank"] <= TOP_N, "code"].astype(str))

    for profile in variants:
        if profile.name == "baseline":
            variant_df = latest_run.copy()
            variant_df["variant_score"] = pd.to_numeric(variant_df["final_score"], errors="coerce")
            variant_df["variant_rank"] = pd.to_numeric(variant_df["baseline_rank"], errors="coerce")
        else:
            variant_df = _apply_variant_score(latest_run, profile)

        metrics = _selection_metrics(variant_df, "variant_rank", "variant_score")
        metric_rows.append(
            {
                "variant": profile.name,
                "description": profile.description,
                **metrics,
            }
        )

        top20 = variant_df.loc[pd.to_numeric(variant_df["variant_rank"], errors="coerce") <= TOP_N].copy()
        top20_codes = set(top20["code"].astype(str))
        new_entries = sorted(top20_codes - baseline_top20)
        exits = sorted(baseline_top20 - top20_codes)
        detail_lines.extend(
            [
                f"## Variant `{profile.name}`",
                f"- description: {profile.description}",
                f"- ordering_ok: {metrics['ordering_ok']}",
                f"- top20_avg_return: {_fmt_pct(metrics['top20_avg_return'])}",
                f"- top50_avg_return: {_fmt_pct(metrics['top50_avg_return'])}",
                f"- universe_avg_return: {_fmt_pct(metrics['universe_avg_return'])}",
                f"- score_return_corr: {_fmt_num(metrics['score_return_corr'])}",
                f"- top20_low_qual_count: {metrics['top20_low_qual_count']}",
                f"- top20_high_risk_count: {metrics['top20_high_risk_count']}",
                f"- top20_new_entries_vs_baseline: {', '.join(new_entries[:10]) if new_entries else '(none)'}",
                f"- top20_exits_vs_baseline: {', '.join(exits[:10]) if exits else '(none)'}",
                "",
            ]
        )

    metrics_df = pd.DataFrame(metric_rows)
    summary_rows = [
        [
            row["variant"],
            _fmt_pct(row["top20_avg_return"]),
            _fmt_pct(row["top50_avg_return"]),
            _fmt_pct(row["universe_avg_return"]),
            _fmt_pct(row["top20_avg_mdd"]),
            _fmt_num(row["score_return_corr"]),
            "YES" if bool(row["ordering_ok"]) else "NO",
            int(row["top20_low_qual_count"]),
            int(row["top20_high_risk_count"]),
        ]
        for _, row in metrics_df.iterrows()
    ]

    report = "\n".join(
        [
            "# Walk-Forward Weight Variant Analysis",
            "",
            f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- target_horizon_days: {TARGET_HORIZON}",
            f"- latest_matured_date: {latest_date}",
            "",
            "## Summary",
            _markdown_table(
                summary_rows,
                [
                    "variant",
                    "top20_avg_return",
                    "top50_avg_return",
                    "universe_avg_return",
                    "top20_avg_mdd",
                    "score_return_corr",
                    "ordering_ok",
                    "top20_low_qual",
                    "top20_high_risk",
                ],
            ),
            "",
            *detail_lines,
        ]
    )
    return metrics_df, report + "\n"


def main() -> None:
    metrics_df, report = build_report()
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_MD}")
    print(f"[ok] wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
