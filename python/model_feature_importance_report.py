from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model_train import MODEL_FEATURE_IMPORTANCE_DIR
else:
    from .model_train import MODEL_FEATURE_IMPORTANCE_DIR

DEFAULT_OUTPUT_MD = Path("outputs") / "model_feature_importance_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown report from model feature importance CSVs.")
    parser.add_argument(
        "--importance-dir",
        type=Path,
        default=MODEL_FEATURE_IMPORTANCE_DIR,
        help="Directory containing feature importance CSV outputs from model_train.py",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Output markdown report path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top features to show for the summary and each target",
    )
    return parser.parse_args()


def _render_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows_"
    table = df[columns].copy().fillna("")
    rows = [[str(item) for item in row] for row in table.to_numpy().tolist()]
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [render_line(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(render_line(row) for row in rows)
    return "\n".join(lines)


def build_markdown_report(importance_dir: Path, top_n: int = 15) -> str:
    summary_path = importance_dir / "feature_importance_summary.csv"
    combined_path = importance_dir / "feature_importance_all_targets.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined file not found: {combined_path}")

    summary = pd.read_csv(summary_path)
    combined = pd.read_csv(combined_path)
    latest_trained_at = str(combined.get("trained_at", pd.Series(dtype=str)).dropna().astype(str).max() or "")
    model_version = str(combined.get("model_version", pd.Series(dtype=str)).dropna().astype(str).max() or "")
    target_names = sorted(combined["target"].dropna().astype(str).unique().tolist())

    lines = [
        "# Model Feature Importance Report",
        "",
        f"- source_dir: `{importance_dir.as_posix()}`",
        f"- model_version: `{model_version or 'unknown'}`",
        f"- trained_at: `{latest_trained_at or 'unknown'}`",
        f"- targets: `{', '.join(target_names)}`",
        "",
        "## Overall Top Features",
        "",
        _render_table(
            summary.head(top_n),
            ["rank", "feature", "target_count", "mean_split_pct", "mean_gain_pct", "composite_score"],
        ),
        "",
    ]

    for target in target_names:
        target_df = combined[combined["target"].astype(str) == target].copy()
        target_df = target_df.sort_values(["importance_gain", "importance_split"], ascending=False, na_position="last")
        lines.extend(
            [
                f"## {target}",
                "",
                _render_table(
                    target_df.head(top_n),
                    ["rank", "feature", "importance_split", "importance_gain", "importance_split_pct", "importance_gain_pct"],
                ),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    report = build_markdown_report(args.importance_dir, top_n=args.top_n)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
