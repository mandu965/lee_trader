"""
select_best_backtest_configs.py

Selection-only utility for backtest experiment summaries.
It reads experiment_summary.csv, filters invalid runs, and chooses three
practical candidate configurations: aggressive, defensive, and balanced.

This script never modifies backtest outputs, production configs, or live
trading code. It only produces analysis artifacts in a separate directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = ROOT / "outputs" / "backtest_experiments" / "experiment_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "backtest_experiments" / "selection"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Select the best backtest configurations.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--max-mdd", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    """Resolve path relative to project root."""
    return path if path.is_absolute() else ROOT / path


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce to finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _json_safe(value: Any) -> Any:
    """Recursively sanitize JSON payload values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _validate_input(path: Path) -> None:
    """Validate input CSV exists."""
    if not path.exists():
        raise FileNotFoundError(f"experiment summary csv not found: {path}")


def load_summary_csv(path: Path) -> pd.DataFrame:
    """Load experiment summary CSV and normalize numeric fields."""
    df = pd.read_csv(path, low_memory=False)
    required = [
        "experiment_id",
        "initial_cash",
        "stop_loss",
        "trailing_stop",
        "max_holding_days",
        "final_total_value",
        "total_return",
        "CAGR",
        "MDD",
        "Sharpe",
        "win_rate",
        "total_trades",
        "profit_factor",
        "payoff_ratio",
        "ranking_score",
        "output_dir",
        "status",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("experiment_summary.csv missing required columns: " + ", ".join(missing))

    for col in [
        "initial_cash",
        "stop_loss",
        "trailing_stop",
        "max_holding_days",
        "final_total_value",
        "total_return",
        "CAGR",
        "MDD",
        "Sharpe",
        "win_rate",
        "total_trades",
        "avg_return",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "profit_factor",
        "ranking_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df["error_message"] = df.get("error_message", "").fillna("").astype(str)
    df["output_dir"] = df["output_dir"].fillna("").astype(str)
    return df


def build_rejected_experiments(
    df: pd.DataFrame,
    *,
    min_trades: int,
    max_mdd: float,
) -> pd.DataFrame:
    """Build rejected experiment table with explicit reject reasons."""
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        reject_reason = None
        status = str(row.get("status") or "").lower()
        total_trades = _safe_float(row.get("total_trades"))
        total_return = _safe_float(row.get("total_return"))
        mdd_abs = abs(_safe_float(row.get("MDD")))
        final_total_value = _safe_float(row.get("final_total_value"))
        initial_cash = _safe_float(row.get("initial_cash"))

        metric_fields = ["final_total_value", "total_return", "CAGR", "MDD", "Sharpe", "win_rate"]
        if any(pd.isna(row.get(col)) for col in metric_fields):
            reject_reason = "missing_metrics"
        elif status not in {"success", "completed"}:
            reject_reason = "failed"
        elif total_trades < float(min_trades):
            reject_reason = "too_few_trades"
        elif mdd_abs > float(max_mdd):
            reject_reason = "excessive_mdd"
        elif final_total_value <= initial_cash or total_return <= 0:
            reject_reason = "negative_return"

        if reject_reason is None:
            continue

        rows.append({
            "experiment_id": row.get("experiment_id"),
            "reject_reason": reject_reason,
            "total_trades": row.get("total_trades"),
            "total_return": row.get("total_return"),
            "CAGR": row.get("CAGR"),
            "MDD": row.get("MDD"),
            "Sharpe": row.get("Sharpe"),
            "win_rate": row.get("win_rate"),
            "output_dir": row.get("output_dir"),
        })
    return pd.DataFrame(rows)


def filter_valid_experiments(
    df: pd.DataFrame,
    *,
    min_trades: int,
    max_mdd: float,
) -> pd.DataFrame:
    """
    Filter valid experiments.

    Metric values in experiment_summary.csv are treated as decimal fractions,
    not whole-number percentages. MDD may be stored as a negative decimal, so
    abs(MDD) is used for threshold filtering.
    """
    work = df.copy()
    metric_fields = ["final_total_value", "total_return", "CAGR", "MDD", "Sharpe", "win_rate"]
    for field in metric_fields:
        work = work.loc[work[field].notna()].copy()

    valid = work.loc[
        work["status"].isin(["success", "completed"])
        & (pd.to_numeric(work["total_trades"], errors="coerce") >= float(min_trades))
        & (pd.to_numeric(work["MDD"], errors="coerce").abs() <= float(max_mdd))
        & (pd.to_numeric(work["final_total_value"], errors="coerce") > pd.to_numeric(work["initial_cash"], errors="coerce"))
    ].copy()
    return valid.reset_index(drop=True)


def _pick_candidate(df: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> pd.Series | None:
    """Pick top row after sorting, or None if empty."""
    if df.empty:
        return None
    ordered = df.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
    if ordered.empty:
        return None
    return ordered.iloc[0]


def select_candidates(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Select aggressive, defensive, and balanced candidates."""
    columns = [
        "profile",
        "experiment_id",
        "stop_loss",
        "trailing_stop",
        "max_holding_days",
        "total_return",
        "CAGR",
        "MDD",
        "Sharpe",
        "win_rate",
        "total_trades",
        "profit_factor",
        "payoff_ratio",
        "ranking_score",
        "output_dir",
        "reason",
    ]
    if filtered_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    aggressive = _pick_candidate(
        filtered_df,
        ["CAGR", "total_return", "profit_factor", "Sharpe", "win_rate"],
        [False, False, False, False, False],
    )
    defensive = _pick_candidate(
        filtered_df,
        ["MDD", "Sharpe", "win_rate", "profit_factor", "CAGR"],
        [False, False, False, False, False],
    )
    defensive = _pick_candidate(
        filtered_df.assign(mdd_abs=filtered_df["MDD"].abs()),
        ["mdd_abs", "Sharpe", "win_rate", "profit_factor", "CAGR"],
        [True, False, False, False, False],
    )
    balanced = _pick_candidate(
        filtered_df,
        ["ranking_score", "CAGR", "Sharpe", "win_rate", "total_trades"],
        [False, False, False, False, False],
    )

    selections = [
        ("aggressive", aggressive, "High CAGR and total_return with acceptable drawdown."),
        ("defensive", defensive, "Low drawdown priority with strong Sharpe and win_rate."),
        ("balanced", balanced, "Highest balance across ranking_score, CAGR, MDD, Sharpe, and trade count."),
    ]

    for profile, candidate, reason in selections:
        if candidate is None:
            continue
        rows.append({
            "profile": profile,
            "experiment_id": candidate.get("experiment_id"),
            "stop_loss": candidate.get("stop_loss"),
            "trailing_stop": candidate.get("trailing_stop"),
            "max_holding_days": candidate.get("max_holding_days"),
            "total_return": candidate.get("total_return"),
            "CAGR": candidate.get("CAGR"),
            "MDD": candidate.get("MDD"),
            "Sharpe": candidate.get("Sharpe"),
            "win_rate": candidate.get("win_rate"),
            "total_trades": candidate.get("total_trades"),
            "profit_factor": candidate.get("profit_factor"),
            "payoff_ratio": candidate.get("payoff_ratio"),
            "ranking_score": candidate.get("ranking_score"),
            "output_dir": candidate.get("output_dir"),
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=columns)


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """Render a compact markdown table."""
    if df.empty:
        return "_none_"
    work = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if limit is not None:
        work = work.head(limit).copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


def build_selection_report(
    *,
    input_file: Path,
    min_trades: int,
    max_mdd: float,
    top_n: int,
    filtered_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> str:
    """Build markdown config selection report."""
    aggressive = selected_df.loc[selected_df["profile"] == "aggressive"] if not selected_df.empty else pd.DataFrame()
    defensive = selected_df.loc[selected_df["profile"] == "defensive"] if not selected_df.empty else pd.DataFrame()
    balanced = selected_df.loc[selected_df["profile"] == "balanced"] if not selected_df.empty else pd.DataFrame()

    if filtered_df.empty:
        selected_section = "No valid candidates"
    else:
        selected_section = _markdown_table(
            selected_df,
            [
                "profile",
                "experiment_id",
                "stop_loss",
                "trailing_stop",
                "max_holding_days",
                "total_return",
                "CAGR",
                "MDD",
                "Sharpe",
                "win_rate",
                "ranking_score",
            ],
        )

    lines = [
        "# Backtest Config Selection Report",
        "",
        "## 1. Selection Scope",
        f"- input_file: `{input_file}`",
        f"- candidate_count_before_filter: `{len(filtered_df) + len(rejected_df)}`",
        f"- candidate_count_after_filter: `{len(filtered_df)}`",
        f"- selected_profiles: `{len(selected_df)}`",
        "",
        "## 2. Filtering Rules",
        f"- status in `success`, `completed`",
        f"- total_trades >= `{min_trades}`",
        f"- abs(MDD) <= `{max_mdd}`",
        f"- final_total_value > initial_cash",
        f"- top_n report window: `{top_n}`",
        f"- note: return-like values are treated as decimal fractions, not percent units",
        "",
        "## 3. Selected Configurations",
        selected_section,
        "",
        "## 4. Aggressive Candidate",
        _markdown_table(
            aggressive,
            ["profile", "experiment_id", "stop_loss", "trailing_stop", "max_holding_days", "total_return", "CAGR", "MDD", "profit_factor", "reason"],
        ),
        "",
        "## 5. Defensive Candidate",
        _markdown_table(
            defensive,
            ["profile", "experiment_id", "stop_loss", "trailing_stop", "max_holding_days", "MDD", "Sharpe", "win_rate", "profit_factor", "reason"],
        ),
        "",
        "## 6. Balanced Candidate",
        _markdown_table(
            balanced,
            ["profile", "experiment_id", "stop_loss", "trailing_stop", "max_holding_days", "ranking_score", "CAGR", "MDD", "Sharpe", "reason"],
        ),
        "",
        "## 7. Rejected Patterns",
        _markdown_table(
            rejected_df,
            ["experiment_id", "reject_reason", "total_trades", "total_return", "CAGR", "MDD", "Sharpe", "win_rate"],
            limit=top_n,
        ),
        "",
        "## 8. Risk Notes",
        "- 백테스트 최적 설정은 과최적화 위험이 있다.",
        "- 2023~2026 3년 구간만으로 실전 확정하면 안 된다.",
        "- 선택된 설정은 실전 투입 전 forward test 또는 paper trading이 필요하다.",
        "- 기존 RULE 백테스트 기준선과 결과 차이를 반드시 비교해야 한다.",
        "",
        "## 9. Recommended Next Step",
        "- balanced 후보를 실전 적용 1순위 가설로 두고 forward test를 먼저 진행합니다.",
        "- aggressive 후보는 수익 추구형 대안으로, defensive 후보는 손실 방어형 대안으로 paper trading에 병렬 투입합니다.",
        "- 선택된 3개 후보의 개별 실험 폴더 안 `analysis/analysis_report.md`를 함께 비교해 weak market 특성을 재검토합니다.",
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    output_dir: Path,
    *,
    filtered_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    selection_report: str,
) -> None:
    """Persist selection outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_df.to_csv(output_dir / "selected_configs.csv", index=False, encoding="utf-8-sig")
    filtered_df.to_csv(output_dir / "filtered_experiments.csv", index=False, encoding="utf-8-sig")
    rejected_df.to_csv(output_dir / "rejected_experiments.csv", index=False, encoding="utf-8-sig")

    selected_json_payload = {
        "selected_configs": selected_df.to_dict(orient="records"),
    }
    (output_dir / "selected_configs.json").write_text(
        json.dumps(_json_safe(selected_json_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "selection_report.md").write_text(selection_report, encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_file = _resolve(args.input_file)
    output_dir = _resolve(args.output_dir)

    _validate_input(input_file)
    df = load_summary_csv(input_file)
    rejected_df = build_rejected_experiments(df, min_trades=args.min_trades, max_mdd=args.max_mdd)
    filtered_df = filter_valid_experiments(df, min_trades=args.min_trades, max_mdd=args.max_mdd)
    selected_df = select_candidates(filtered_df)
    report = build_selection_report(
        input_file=input_file,
        min_trades=args.min_trades,
        max_mdd=args.max_mdd,
        top_n=args.top_n,
        filtered_df=filtered_df,
        rejected_df=rejected_df,
        selected_df=selected_df,
    )
    save_outputs(
        output_dir,
        filtered_df=filtered_df,
        rejected_df=rejected_df,
        selected_df=selected_df,
        selection_report=report,
    )


if __name__ == "__main__":
    main()
