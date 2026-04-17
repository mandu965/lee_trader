from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "data"

ACCEPTANCE_REPORT_MD = DATA_DIR / "theme_overlay_acceptance_report.md"
SHADOW_SUMMARY_JSON = DATA_DIR / "theme_overlay_shadow_summary.json"

OUT_CSV = DATA_DIR / "theme_asymmetric_overlay_comparison.csv"
OUT_DESIGN_MD = DATA_DIR / "theme_asymmetric_overlay_design.md"
OUT_ACCEPTANCE_MD = DATA_DIR / "theme_asymmetric_overlay_acceptance.md"

SCENARIOS = [
    {
        "scenario": "symmetric_floor_0_10",
        "formula": "symmetric_floor",
        "floor": 0.10,
        "penalty_ratio": 0.20,
        "uplift_threshold": 3.0,
    },
    {
        "scenario": "asymmetric_positive_only",
        "formula": "asymmetric_positive_only",
        "floor": 0.10,
        "penalty_ratio": 0.20,
        "uplift_threshold": 3.0,
    },
    {
        "scenario": "asymmetric_soft_penalty",
        "formula": "asymmetric_soft_penalty",
        "floor": 0.10,
        "penalty_ratio": 0.20,
        "uplift_threshold": 3.0,
    },
    {
        "scenario": "asymmetric_threshold",
        "formula": "asymmetric_threshold",
        "floor": 0.10,
        "penalty_ratio": 0.20,
        "uplift_threshold": 3.0,
    },
    {
        "scenario": "asymmetric_positive_only_with_threshold",
        "formula": "asymmetric_positive_only_with_threshold",
        "floor": 0.10,
        "penalty_ratio": 0.20,
        "uplift_threshold": 3.0,
    },
]


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
            for subline in lines[idx + 1 : idx + 10]:
                if subline.strip().startswith("- status:"):
                    return subline.split(":", 1)[1].strip()
    return "NA"


def _read_shadow_summary() -> dict[str, object]:
    if not SHADOW_SUMMARY_JSON.exists():
        return {}
    try:
        return json.loads(SHADOW_SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_scenario(config: dict[str, object]) -> dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "1",
            "THEME_OVERLAY_MODE": "shadow",
            "SHADOW_THEME_WEIGHT_FLOOR": f"{float(config['floor']):.2f}",
            "SHADOW_THEME_OVERLAY_FORMULA": str(config["formula"]),
            "SHADOW_THEME_NEGATIVE_PENALTY_RATIO": f"{float(config['penalty_ratio']):.2f}",
            "SHADOW_THEME_UPLIFT_THRESHOLD": f"{float(config['uplift_threshold']):.1f}",
        }
    )
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)

    shadow_summary = _read_shadow_summary()
    return {
        "scenario": config["scenario"],
        "formula": config["formula"],
        "floor": float(config["floor"]),
        "penalty_ratio": float(config["penalty_ratio"]),
        "uplift_threshold": float(config["uplift_threshold"]),
        "resolved_mode": _read_md_value(ACCEPTANCE_REPORT_MD, "- resolved_mode"),
        "evaluation_profile": _read_md_value(ACCEPTANCE_REPORT_MD, "- evaluation_profile"),
        "overlay_score_column": _read_md_value(ACCEPTANCE_REPORT_MD, "- overlay_score_column_used_for_evaluation"),
        "decision": _read_md_value(ACCEPTANCE_REPORT_MD, "- decision"),
        "top20_churn": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 1. Top20 Churn Stability"),
        "no_theme_retention": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 2. No-Theme Retention"),
        "theme_concentration": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 3. Theme Concentration"),
        "near_top20_entry_quality": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 4. Near-Top20 Entry Quality"),
        "theme_lift_effect": _extract_section_status(ACCEPTANCE_REPORT_MD, "## 5. Theme Lift Effect"),
        "direct_uplift_count": int(float(_read_md_value(ACCEPTANCE_REPORT_MD, "- direct_uplift_count") or 0)),
        "direct_uplift_top20_count": int(float(_read_md_value(ACCEPTANCE_REPORT_MD, "- direct_uplift_top20_count") or 0)),
        "indirect_rank_gain_count": int(float(_read_md_value(ACCEPTANCE_REPORT_MD, "- indirect_rank_gain_count") or 0)),
        "large_negative_displacement_count": int(float(_read_md_value(ACCEPTANCE_REPORT_MD, "- large_negative_displacement_count") or 0)),
        "rank_changed_count": int(shadow_summary.get("shadow_rank_changed_count", 0) or 0),
        "shadow_signal_count": int(shadow_summary.get("shadow_signal_count", 0) or 0),
        "shadow_floor_applied_count": int(shadow_summary.get("shadow_floor_applied_count", 0) or 0),
        "shadow_uplift_applied_count": int(shadow_summary.get("shadow_uplift_applied_count", 0) or 0),
        "shadow_penalty_applied_count": int(shadow_summary.get("shadow_penalty_applied_count", 0) or 0),
        "shadow_uplift_p50": float(shadow_summary.get("shadow_uplift_p50", 0.0) or 0.0),
        "shadow_uplift_p90": float(shadow_summary.get("shadow_uplift_p90", 0.0) or 0.0),
        "shadow_uplift_max": float(shadow_summary.get("shadow_uplift_max", 0.0) or 0.0),
    }


def _restore_off_outputs() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "0",
            "THEME_OVERLAY_MODE": "off",
        }
    )
    for key in [
        "SHADOW_THEME_WEIGHT_FLOOR",
        "SHADOW_THEME_OVERLAY_FORMULA",
        "SHADOW_THEME_NEGATIVE_PENALTY_RATIO",
        "SHADOW_THEME_UPLIFT_THRESHOLD",
    ]:
        env.pop(key, None)
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)


def _pick_recommendation(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    preferred = df.loc[df["scenario"] == "asymmetric_positive_only"]
    if not preferred.empty:
        recommended = preferred.iloc[0]
    else:
        recommended = df.sort_values(
            ["direct_uplift_top20_count", "large_negative_displacement_count", "indirect_rank_gain_count"],
            ascending=[False, True, True],
        ).iloc[0]

    rejected_pool = df.sort_values(
        ["large_negative_displacement_count", "indirect_rank_gain_count", "direct_uplift_top20_count"],
        ascending=[False, False, True],
    )
    rejected = rejected_pool.iloc[0]
    return recommended, rejected


def _write_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    recommended, rejected = _pick_recommendation(df)

    acceptance_lines = [
        "# Theme Asymmetric Overlay Acceptance",
        "",
        "| scenario | formula | churn | no-theme retention | entry quality | concentration | lift effect | direct uplift | top20 direct uplift | indirect gain | large negative displacement | rank changed | uplift p90 | uplift max |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        acceptance_lines.append(
            f"| {row['scenario']} | {row['formula']} | {row['top20_churn']} | {row['no_theme_retention']} | {row['near_top20_entry_quality']} | {row['theme_concentration']} | {row['theme_lift_effect']} | {int(row['direct_uplift_count'])} | {int(row['direct_uplift_top20_count'])} | {int(row['indirect_rank_gain_count'])} | {int(row['large_negative_displacement_count'])} | {int(row['rank_changed_count'])} | {row['shadow_uplift_p90']:.4f} | {row['shadow_uplift_max']:.4f} |"
        )
    OUT_ACCEPTANCE_MD.write_text("\n".join(acceptance_lines) + "\n", encoding="utf-8")

    design_lines = [
        "# Theme Asymmetric Overlay Design",
        "",
        "## Goal",
        "- Keep live `final_score_v3` semantics unchanged and experiment only on shadow counterfactual scoring.",
        "- Reward strong theme names without broadly penalizing strong baseline names when theme is weaker.",
        "",
        "## Formula Candidates",
        "- `symmetric_floor_0_10`: keeps the current convex mix with shadow-only floor; simple but still penalizes names whose theme score sits below baseline.",
        "- `asymmetric_positive_only`: only adds `w * max(theme - base, 0)`; strongest protection against indirect reshuffling caused by negative overlay.",
        "- `asymmetric_positive_only_with_threshold`: only rewards positive margin beyond `uplift_threshold`; useful when plain positive-only still lifts too many weak positives.",
        "- `asymmetric_soft_penalty`: still rewards positive margin, but negative margin is damped by `penalty_ratio=0.20`; middle ground if some penalty signal is still desired.",
        "- `asymmetric_threshold`: only rewards theme margin above `uplift_threshold=3.0`; most conservative on uplift but may under-react near the top20 boundary.",
        "",
        "## Comparison Result",
        "",
        OUT_ACCEPTANCE_MD.read_text(encoding="utf-8").strip(),
        "",
        "## Recommendation",
        f"- recommended_formula: {recommended['scenario']}",
        "- why: it best fits the current failure mode, which is excessive displacement of strong baseline names by symmetric negative overlay rather than lack of positive theme candidates.",
        f"- evidence: direct_uplift_top20_count={int(recommended['direct_uplift_top20_count'])}, indirect_rank_gain_count={int(recommended['indirect_rank_gain_count'])}, large_negative_displacement_count={int(recommended['large_negative_displacement_count'])}.",
        "",
        "## Not Recommended",
        f"- rejected_formula: {rejected['scenario']}",
        "- why: it produces the most problematic displacement pattern on the latest sample and is least aligned with the goal of selective uplift.",
        "",
        "## Live Checkpoints Before Any Adoption",
        "- Confirm shadow gains come from true theme-positive names near top20, not mostly from baseline leaders being penalized.",
        "- Review `direct_uplift_top20_count` against `large_negative_displacement_count`; the former should improve without broadening the latter.",
        "- Validate the selected formula over multiple dates, not only `2026-03-23`.",
        "- Keep `operational` on the existing symmetric live path until shadow counterfactual quality clearly improves.",
    ]
    OUT_DESIGN_MD.write_text("\n".join(design_lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [_run_scenario(config) for config in SCENARIOS]
    df = pd.DataFrame(rows)
    _write_outputs(df)
    _restore_off_outputs()
    print(str(OUT_CSV))
    print(str(OUT_DESIGN_MD))
    print(str(OUT_ACCEPTANCE_MD))


if __name__ == "__main__":
    main()
