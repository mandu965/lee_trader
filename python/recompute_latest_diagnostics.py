import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RANKING_CSV = DATA_DIR / "ranking_final.csv"
SUMMARY_MD = OUTPUT_DIR / "latest_diagnostics_summary.md"
FAILURES_MD = OUTPUT_DIR / "latest_diagnostics_failures.md"


@dataclass
class RunResult:
    name: str
    command: list[str]
    expected_outputs: list[Path]
    returncode: int
    stdout: str
    stderr: str
    generated_outputs: list[Path]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_ranking_metadata() -> dict[str, str]:
    if not RANKING_CSV.exists():
      raise FileNotFoundError(f"ranking CSV not found: {RANKING_CSV}")

    df = pd.read_csv(RANKING_CSV)
    latest_date = "NA"
    if "date" in df.columns and df["date"].notna().any():
        latest_date = str(pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().max())
    score_formula_version = "NA"
    if "score_formula_version" in df.columns and df["score_formula_version"].notna().any():
        score_formula_version = str(df["score_formula_version"].dropna().iloc[0])
    stat = RANKING_CSV.stat()
    return {
        "generated_at": now_text(),
        "latest_date": latest_date,
        "score_formula_version": score_formula_version,
        "source_ranking_file": f"{RANKING_CSV.name}; rows={len(df)}; modified_at={datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
        "recomputed_from_current_code": "true",
    }


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_command(name: str, command: list[str], expected_outputs: list[Path]) -> RunResult:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    generated = [path for path in expected_outputs if path.exists()]
    return RunResult(
        name=name,
        command=command,
        expected_outputs=expected_outputs,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        generated_outputs=generated,
    )


def prepend_markdown_metadata(path: Path, metadata: dict[str, str]) -> None:
    body = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    meta_lines = [f"- {key}: {value}" for key, value in metadata.items()]
    stamped = "\n".join(["## Metadata", *meta_lines, "", body.lstrip()])
    path.write_text(stamped, encoding="utf-8")


def prepend_csv_metadata(path: Path, metadata: dict[str, str]) -> None:
    original = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    comment_lines = [f"# {key}: {value}" for key, value in metadata.items()]
    stamped = "\n".join([*comment_lines, original.lstrip("\ufeff")])
    path.write_text(stamped, encoding="utf-8")


def stamp_output(path: Path, metadata: dict[str, str]) -> None:
    if not path.exists():
        return
    if path.suffix.lower() == ".md":
        prepend_markdown_metadata(path, metadata)
    elif path.suffix.lower() == ".csv":
        prepend_csv_metadata(path, metadata)


def write_tech_md(result: RunResult, metadata: dict[str, str], output_path: Path) -> None:
    lines = [
        "## Metadata",
        *[f"- {key}: {value}" for key, value in metadata.items()],
        "",
        "# Tech Score Diagnostics",
        "",
        "```text",
        result.stdout.rstrip(),
        "```",
    ]
    if result.stderr.strip():
        lines.extend(["", "## stderr", "", "```text", result.stderr.rstrip(), "```"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scripts() -> list[tuple[str, list[str], list[Path]]]:
    py = sys.executable
    return [
        (
            "final_score_dominance",
            [py, str(PYTHON_DIR / "check_final_score_dominance.py")],
            [
                OUTPUT_DIR / "final_score_dominance_report.md",
                OUTPUT_DIR / "final_score_top20_components.csv",
            ],
        ),
        (
            "ranking_trend_alignment",
            [py, str(PYTHON_DIR / "check_ranking_trend_alignment.py")],
            [
                OUTPUT_DIR / "top20_score_breakdown.csv",
                OUTPUT_DIR / "sector_score_summary.csv",
                OUTPUT_DIR / "ranking_trend_alignment.md",
                OUTPUT_DIR / "confidence_anomaly_report.md",
            ],
        ),
        (
            "regime_weight_effect",
            [py, str(PYTHON_DIR / "check_regime_weight_effect.py")],
            [OUTPUT_DIR / "regime_weight_effect.md"],
        ),
        (
            "tech_score",
            [
                py,
                str(PYTHON_DIR / "check_tech_score.py"),
                "--out-csv",
                str(OUTPUT_DIR / "tech_score_top20_summary.csv"),
            ],
            [
                OUTPUT_DIR / "tech_score_top20_summary.csv",
                OUTPUT_DIR / "tech_score_diagnostics.md",
            ],
        ),
        (
            "prob_score_alignment",
            [py, str(PYTHON_DIR / "check_prob_score_alignment.py")],
            [OUTPUT_DIR / "prob_score_diagnostics.md"],
        ),
    ]


def write_failures_md(metadata: dict[str, str], results: list[RunResult]) -> None:
    failed = [r for r in results if r.returncode != 0 or len(r.generated_outputs) != len(r.expected_outputs)]
    lines = [
        "## Metadata",
        *[f"- {key}: {value}" for key, value in metadata.items()],
        "",
        "# Latest Diagnostics Failures",
        "",
    ]
    if not failed:
        lines.append("- none")
    else:
        for result in failed:
            missing = [str(path.relative_to(ROOT)) for path in result.expected_outputs if not path.exists()]
            lines.append(f"## {result.name}")
            lines.append(f"- returncode: {result.returncode}")
            lines.append(f"- command: `{' '.join(result.command)}`")
            lines.append(f"- missing_outputs: {missing or ['none']}")
            if result.stderr.strip():
                lines.extend(["", "```text", result.stderr.rstrip(), "```", ""])
            elif result.stdout.strip():
                lines.extend(["", "```text", result.stdout.rstrip(), "```", ""])
    FAILURES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(metadata: dict[str, str], ranking_meta: dict[str, str], results: list[RunResult]) -> None:
    lines = [
        "## Metadata",
        *[f"- {key}: {value}" for key, value in metadata.items()],
        "",
        "# Latest Diagnostics Summary",
        "",
        "## Ranking Rebuild",
        f"- ranking_file: `{RANKING_CSV.relative_to(ROOT)}`",
        f"- latest_date: {ranking_meta['latest_date']}",
        f"- score_formula_version: {ranking_meta['score_formula_version']}",
        f"- source_ranking_file: {ranking_meta['source_ranking_file']}",
        "",
        "## Refreshed Files",
    ]
    refreshed = []
    for result in results:
        for path in result.generated_outputs:
            refreshed.append((result.name, path))
    if not refreshed:
        lines.append("- none")
    else:
        for name, path in refreshed:
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- `{path.relative_to(ROOT)}` from `{name}` updated_at={stamp}")

    lines.extend(["", "## Script Status"])
    for result in results:
        status = "ok" if result.returncode == 0 and len(result.generated_outputs) == len(result.expected_outputs) else "failed"
        missing = [str(path.relative_to(ROOT)) for path in result.expected_outputs if not path.exists()]
        lines.append(f"- {result.name}: {status}; returncode={result.returncode}; outputs={len(result.generated_outputs)}/{len(result.expected_outputs)}; missing={missing or ['none']}")

    if FAILURES_MD.exists():
        lines.extend(["", "## Failure Log", f"- `{FAILURES_MD.relative_to(ROOT)}`"])

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dir()

    ranking_build = run_command("ranking_builder", [sys.executable, str(PYTHON_DIR / "ranking_builder.py")], [RANKING_CSV])
    if ranking_build.returncode != 0 or not RANKING_CSV.exists():
        metadata = {
            "generated_at": now_text(),
            "latest_date": "NA",
            "score_formula_version": "NA",
            "source_ranking_file": "ranking rebuild failed",
            "recomputed_from_current_code": "true",
        }
        write_failures_md(metadata, [ranking_build])
        write_summary_md(metadata, metadata, [ranking_build])
        raise SystemExit(ranking_build.returncode or 1)

    ranking_meta = read_ranking_metadata()
    results: list[RunResult] = []
    for name, command, expected_outputs in build_scripts():
        result = run_command(name, command, expected_outputs)
        for path in result.generated_outputs:
            stamp_output(path, ranking_meta)
        results.append(result)

    write_failures_md(ranking_meta, results)
    write_summary_md(ranking_meta, ranking_meta, results)

    if any(result.returncode != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
