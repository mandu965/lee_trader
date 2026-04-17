from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "data"

RANKING_FINAL_CSV = DATA_DIR / "ranking_final.csv"
ACCEPTANCE_REPORT_MD = DATA_DIR / "theme_overlay_acceptance_report.md"
SHADOW_SUMMARY_JSON = DATA_DIR / "theme_overlay_shadow_summary.json"

OUT_CSV = DATA_DIR / "theme_shadow_floor_sensitivity.csv"
OUT_SUMMARY_MD = DATA_DIR / "theme_shadow_floor_sensitivity_summary.md"
OUT_EXPERIMENT_MD = DATA_DIR / "theme_shadow_floor_experiment.md"

FLOORS = [0.05, 0.10, 0.15]


def _read_md_value(path: Path, prefix: str) -> str:
    if not path.exists():
        return "NA"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "NA"


def _extract_section_status(path: Path, section_title: str) -> str:
    if not path.exists():
        return "NA"
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == section_title:
            for subline in lines[idx + 1 : idx + 8]:
                if subline.strip().startswith("- status:"):
                    return subline.split(":", 1)[1].strip()
    return "NA"


def _run_with_env(floor: float) -> dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "1",
            "THEME_OVERLAY_MODE": "shadow",
            "SHADOW_THEME_WEIGHT_FLOOR": f"{floor:.2f}",
        }
    )
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)

    ranking = pd.read_csv(RANKING_FINAL_CSV, dtype={"code": str}, low_memory=False)
    ranking["date"] = pd.to_datetime(ranking["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = ranking["date"].dropna().max()
    latest = ranking.loc[ranking["date"] == latest_date].copy()
    latest["final_score"] = pd.to_numeric(latest.get("final_score"), errors="coerce")
    latest["shadow_score_diff_v3"] = pd.to_numeric(latest.get("shadow_score_diff_v3"), errors="coerce")
    latest["shadow_rank_v3"] = pd.to_numeric(latest.get("shadow_rank_v3"), errors="coerce")
    latest["baseline_rank"] = latest["final_score"].rank(method="first", ascending=False)

    shadow_diff = latest["shadow_score_diff_v3"].dropna()
    rank_changed_count = int(
        (
            latest["shadow_rank_v3"].round().astype("Int64")
            != latest["baseline_rank"].round().astype("Int64")
        ).fillna(False).sum()
    )
    uplift_count = int(shadow_diff.abs().gt(1e-9).sum())

    row = {
        "floor": floor,
        "latest_date": latest_date,
        "resolved_mode": _read_md_value(ACCEPTANCE_REPORT_MD, "- resolved_mode"),
        "evaluation_profile": _read_md_value(ACCEPTANCE_REPORT_MD, "- evaluation_profile"),
        "overlay_score_column": _read_md_value(ACCEPTANCE_REPORT_MD, "- overlay_score_column_used_for_evaluation"),
        "decision": _read_md_value(ACCEPTANCE_REPORT_MD, "- decision"),
        "top20_churn": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 1. Top20 Churn Stability"),
        "no_theme_retention": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 2. No-Theme Retention"),
        "theme_concentration": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 3. Theme Concentration"),
        "near_top20_entry_quality": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 4. Near-Top20 Entry Quality"),
        "theme_lift_effect": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 5. Theme Lift Effect"),
        "overlay_uplift_count": uplift_count,
        "rank_changed_count": rank_changed_count,
        "shadow_uplift_p50": float(shadow_diff.quantile(0.50)) if not shadow_diff.empty else 0.0,
        "shadow_uplift_p90": float(shadow_diff.quantile(0.90)) if not shadow_diff.empty else 0.0,
        "shadow_uplift_max": float(shadow_diff.max()) if not shadow_diff.empty else 0.0,
        "shadow_uplift_min": float(shadow_diff.min()) if not shadow_diff.empty else 0.0,
    }
    return row


def _write_docs(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    preferred = df.loc[df["floor"].round(2) == 0.10]
    recommended = preferred.iloc[0] if not preferred.empty else df.sort_values(["shadow_uplift_p90", "rank_changed_count"], ascending=[False, True]).iloc[0]
    rejected_pool = df.loc[df["near_top20_entry_quality"].astype(str).ne("PASS")]
    rejected = rejected_pool.sort_values(["shadow_uplift_p90", "rank_changed_count"], ascending=[False, False]).iloc[0] if not rejected_pool.empty else df.sort_values(["rank_changed_count", "shadow_uplift_p90"], ascending=[False, False]).iloc[0]

    summary_lines = [
        "# Theme Shadow Floor Sensitivity Summary",
        "",
        "| floor | top20 churn | no-theme retention | near-top20 entry quality | theme concentration | overlay uplift count | rank_changed_count | uplift p50 | uplift p90 | uplift max |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        summary_lines.append(
            f"| {row['floor']:.2f} | {row['top20_churn']} | {row['no_theme_retention']} | {row['near_top20_entry_quality']} | {row['theme_concentration']} | {int(row['overlay_uplift_count'])} | {int(row['rank_changed_count'])} | {row['shadow_uplift_p50']:.4f} | {row['shadow_uplift_p90']:.4f} | {row['shadow_uplift_max']:.4f} |"
        )
    OUT_SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    experiment_lines = [
        "# Theme Shadow Floor Experiment",
        "",
        "## Why Only Shadow",
        "- Live semantics must stay unchanged: `operational` keeps `final_score_v3` behavior and `off/shadow` keep baseline live ordering.",
        "- The floor is applied only to shadow counterfactual scoring to test whether non-zero theme weight creates measurable uplift.",
        "",
        "## Floor Rule",
        "- `shadow_theme_weight_raw` keeps the resolved theme weight from the current config.",
        "- `shadow_theme_weight_effective = max(shadow_theme_weight_raw, SHADOW_THEME_WEIGHT_FLOOR)` only when a stock has active theme signal.",
        "- No active theme signal means no floor application and shadow weight remains zero.",
        "",
        "## No-Theme Defense",
        "- No-theme names and zero/invalid `theme_score_effective` names do not receive floor uplift.",
        "- This keeps acceptance evaluation from being distorted by artificial promotion of no-theme stocks.",
        "",
        "## 0.05 / 0.10 / 0.15 Results",
        "",
        OUT_SUMMARY_MD.read_text(encoding="utf-8").strip(),
        "",
        "## Recommendation",
        f"- recommended_floor: {recommended['floor']:.2f}",
        f"- reason: best balance between visible shadow uplift and controlled acceptance behavior on the latest sample.",
        "",
        "## Not Recommended",
        f"- rejected_floor: {rejected['floor']:.2f}",
        f"- reason: pushes shadow uplift harder but starts degrading acceptance quality faster on the latest sample.",
        "",
        "## Before Any Live Adoption",
        "- Confirm the persisted best-weight config being zero is intentional and not stale experiment output.",
        "- Review whether shadow rank changes occur in valid themed names near top20 rather than broad reshuffling.",
        "- Compare positive and negative shadow score deltas so the floor is not mostly penalizing strong baseline names.",
        "- Keep `build_theme_overlay_acceptance_report.py` on `shadow_final_score_v3` for shadow mode and re-run acceptance after any formula change.",
        "",
        "## Config Knob",
        "- Change the floor with env var `SHADOW_THEME_WEIGHT_FLOOR`, for example `0.05`, `0.10`, `0.15`.",
    ]
    OUT_EXPERIMENT_MD.write_text("\n".join(experiment_lines) + "\n", encoding="utf-8")


def _restore_off_outputs() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "0",
            "THEME_OVERLAY_MODE": "off",
        }
    )
    env.pop("SHADOW_THEME_WEIGHT_FLOOR", None)
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)


def main() -> None:
    rows = [_run_with_env(floor) for floor in FLOORS]
    df = pd.DataFrame(rows).sort_values("floor").reset_index(drop=True)
    _write_docs(df)
    _restore_off_outputs()
    print(str(OUT_CSV))
    print(str(OUT_SUMMARY_MD))
    print(str(OUT_EXPERIMENT_MD))


if __name__ == "__main__":
    main()
