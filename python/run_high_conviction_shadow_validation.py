from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TMP_DIR = DATA_DIR / "tmp_high_conviction_validation_restore"

RESTORE_FILES = [
    OUTPUT_DIR / "stock_theme_daily.csv",
    OUTPUT_DIR / "stock_theme_daily_summary.csv",
    DATA_DIR / "ranking_final.csv",
    DATA_DIR / "theme_overlay_acceptance_report.md",
    DATA_DIR / "top20_churn_analysis.csv",
    DATA_DIR / "new_entry_quality_report.csv",
    DATA_DIR / "no_theme_retention.csv",
    DATA_DIR / "theme_concentration.csv",
    DATA_DIR / "theme_lift_analysis.csv",
    DATA_DIR / "theme_overlay_mode_resolution.md",
    DATA_DIR / "theme_overlay_acceptance_mode_note.md",
    DATA_DIR / "theme_overlay_shadow_mode_update.md",
    DATA_DIR / "theme_overlay_shadow_summary.json",
]

OUT_CSV = DATA_DIR / "theme_high_conviction_validation_comparison.csv"
OUT_MD = DATA_DIR / "theme_high_conviction_validation_comparison.md"

SCENARIOS = [
    {
        "scenario": "high_conviction_v1_repro",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.85,
        "strong_source_theme_level_min": 85.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
    {
        "scenario": "high_conviction_v2_blend080",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.80,
        "strong_source_theme_level_min": 85.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
    {
        "scenario": "high_conviction_v2_blend075",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.75,
        "strong_source_theme_level_min": 85.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
    {
        "scenario": "high_conviction_v2_blend070",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.70,
        "strong_source_theme_level_min": 85.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
    {
        "scenario": "high_conviction_v4_blend075_tl90",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.75,
        "strong_source_theme_level_min": 90.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
    {
        "scenario": "high_conviction_v3_tl95",
        "mapping_floor": 0.65,
        "confidence_floor": 0.00,
        "component_blend_floor": 0.85,
        "strong_source_theme_level_min": 95.0,
        "strong_source_signal_conf_min": 0.55,
        "strong_source_conf_floor": 0.74,
    },
]


def _read_md_value(path: Path, prefix: str) -> str:
    if not path.exists():
        return "NA"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "NA"


def _backup_restore_files() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESTORE_FILES:
        if not path.exists():
            continue
        dst = TMP_DIR / path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _restore_restore_files() -> None:
    for path in RESTORE_FILES:
        backup = TMP_DIR / path.relative_to(ROOT)
        if not backup.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _off_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "0",
            "THEME_OVERLAY_MODE": "off",
        }
    )
    for key in [
        "STOCK_THEME_TRANSMISSION_MODE",
        "STOCK_THEME_MAPPING_FLOOR",
        "STOCK_THEME_CONFIDENCE_FLOOR",
        "STOCK_THEME_COMPONENT_BLEND_FLOOR",
        "STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN",
        "STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN",
        "STOCK_THEME_STRONG_SOURCE_CONF_FLOOR",
        "THEME_OVERLAY_SHADOW_MODE",
        "THEME_OVERLAY_SHADOW_GAIN",
        "THEME_OVERLAY_SHADOW_CAP",
        "THEME_OVERLAY_SHADOW_SOFT_CONF_ENABLED",
    ]:
        env.pop(key, None)
    return env


def _scenario_env(config: dict[str, float | str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STOCK_THEME_TRANSMISSION_MODE": "high_conviction",
            "STOCK_THEME_MAPPING_FLOOR": f"{float(config['mapping_floor']):.2f}",
            "STOCK_THEME_CONFIDENCE_FLOOR": f"{float(config['confidence_floor']):.2f}",
            "STOCK_THEME_COMPONENT_BLEND_FLOOR": f"{float(config['component_blend_floor']):.2f}",
            "STOCK_THEME_STRONG_SOURCE_THEME_LEVEL_MIN": f"{float(config['strong_source_theme_level_min']):.1f}",
            "STOCK_THEME_STRONG_SOURCE_SIGNAL_CONF_MIN": f"{float(config['strong_source_signal_conf_min']):.2f}",
            "STOCK_THEME_STRONG_SOURCE_CONF_FLOOR": f"{float(config['strong_source_conf_floor']):.2f}",
            "ENABLE_THEME_OVERLAY": "1",
            "THEME_OVERLAY_MODE": "shadow",
            "THEME_OVERLAY_SHADOW_MODE": "asymmetric_positive_only",
            "THEME_OVERLAY_SHADOW_GAIN": "0.12",
            "THEME_OVERLAY_SHADOW_CAP": "6.0",
            "THEME_OVERLAY_SHADOW_SOFT_CONF_ENABLED": "1",
        }
    )
    return env


def _build_theme_etf_for_date(end_date: str) -> None:
    env = os.environ.copy()
    _run(["python", "python/compute_theme_etf_daily.py", "--end-date", end_date], env)


def _copy_snapshot(scenario: str) -> None:
    shutil.copy2(OUTPUT_DIR / "stock_theme_daily.csv", DATA_DIR / f"stock_theme_daily_{scenario}.csv")
    shutil.copy2(DATA_DIR / "ranking_final.csv", DATA_DIR / f"ranking_final_{scenario}.csv")
    shutil.copy2(DATA_DIR / "theme_overlay_acceptance_report.md", DATA_DIR / f"theme_overlay_acceptance_report_{scenario}.md")
    shutil.copy2(DATA_DIR / "theme_lift_analysis.csv", DATA_DIR / f"theme_lift_analysis_{scenario}.csv")


def _summarize_scenario(config: dict[str, float | str]) -> dict[str, object]:
    scenario = str(config["scenario"])
    ranking = pd.read_csv(DATA_DIR / f"ranking_final_{scenario}.csv", dtype={"code": str}, low_memory=False)
    ranking["date"] = pd.to_datetime(ranking["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_ranking_date = ranking["date"].dropna().max()
    if latest_ranking_date:
        ranking = ranking.loc[ranking["date"] == latest_ranking_date].copy()
    ranking["final_score"] = pd.to_numeric(ranking["final_score"], errors="coerce").fillna(0.0)
    ranking["shadow_final_score_v3"] = pd.to_numeric(ranking["shadow_final_score_v3"], errors="coerce").fillna(ranking["final_score"])
    ranking["baseline_rank"] = ranking["final_score"].rank(method="first", ascending=False).astype(int)
    ranking["overlay_rank"] = ranking["shadow_final_score_v3"].rank(method="first", ascending=False).astype(int)
    ranking["score_delta_v3"] = ranking["shadow_final_score_v3"] - ranking["final_score"]
    ranking["rank_change_shadow"] = ranking["baseline_rank"] - ranking["overlay_rank"]
    ranking["large_negative_displacement"] = (
        ranking["rank_change_shadow"].le(-5) | ranking["score_delta_v3"].le(-1.0)
    )
    entry = ranking.loc[(ranking["baseline_rank"] > 20) & (ranking["overlay_rank"] <= 20)].copy()

    stock_theme = pd.read_csv(DATA_DIR / f"stock_theme_daily_{scenario}.csv", dtype={"code": str}, low_memory=False)
    stock_theme["date"] = pd.to_datetime(stock_theme["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = stock_theme["date"].dropna().max()
    stock_theme = stock_theme.loc[
        (stock_theme["date"] == latest_date)
        & (pd.to_numeric(stock_theme["theme_rank_within_stock"], errors="coerce").fillna(999).eq(1))
    ].copy()
    stock_theme["strong_source_gate_passed"] = stock_theme["strong_source_gate_passed"].fillna(False).astype(bool)
    stock_theme["component_floor_applied"] = stock_theme["component_floor_applied"].fillna(False).astype(bool)

    return {
        "scenario": scenario,
        "latest_date": latest_date,
        "decision": _read_md_value(DATA_DIR / f"theme_overlay_acceptance_report_{scenario}.md", "- decision"),
        "top20_churn": _read_md_value(DATA_DIR / f"theme_overlay_acceptance_report_{scenario}.md", "- churn_ratio"),
        "entry_count": int(len(entry)),
        "entry_names": " | ".join(entry["name"].astype(str).tolist()) if not entry.empty else "",
        "entry_codes": " | ".join(entry["code"].astype(str).tolist()) if not entry.empty else "",
        "entry_score_delta_sum": float(entry["score_delta_v3"].sum()) if not entry.empty else 0.0,
        "large_negative_displacement_count": int(ranking["large_negative_displacement"].sum()),
        "direct_uplift_count": int(((ranking["score_delta_v3"] > 0.0) & (ranking["rank_change_shadow"] > 0.0)).sum()),
        "direct_uplift_top20_count": int(((ranking["score_delta_v3"] > 0.0) & (ranking["rank_change_shadow"] > 0.0) & (ranking["overlay_rank"] <= 20)).sum()),
        "theme_lift_effect": _read_md_value(DATA_DIR / f"theme_overlay_acceptance_report_{scenario}.md", "- status"),
        "gated_top1_count": int(stock_theme["strong_source_gate_passed"].sum()),
        "component_floor_top1_count": int(stock_theme["component_floor_applied"].sum()),
        "gated_themes": json.dumps(stock_theme.loc[stock_theme["strong_source_gate_passed"], "dominant_theme"].value_counts().to_dict(), ensure_ascii=False),
        "component_blend_floor": float(config["component_blend_floor"]),
        "strong_source_theme_level_min": float(config["strong_source_theme_level_min"]),
        "strong_source_signal_conf_min": float(config["strong_source_signal_conf_min"]),
    }


def _write_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    lines = [
        "# Theme High Conviction Validation Comparison",
        "",
        "| scenario | decision | entry_count | entry_names | large_negative_displacement | direct_uplift | direct_uplift_top20 | gated_top1 | component_floor_top1 | blend_floor | theme_level_min | signal_conf_min |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['decision']} | {int(row['entry_count'])} | {row['entry_names'] or '-'} | "
            f"{int(row['large_negative_displacement_count'])} | {int(row['direct_uplift_count'])} | {int(row['direct_uplift_top20_count'])} | "
            f"{int(row['gated_top1_count'])} | {int(row['component_floor_top1_count'])} | {row['component_blend_floor']:.2f} | "
            f"{row['strong_source_theme_level_min']:.1f} | {row['strong_source_signal_conf_min']:.2f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    end_date = sys.argv[1] if len(sys.argv) > 1 else ""
    _backup_restore_files()
    rows: list[dict[str, object]] = []
    try:
        if end_date:
            _build_theme_etf_for_date(end_date)
        for config in SCENARIOS:
            env = _scenario_env(config)
            _run(["python", "python/build_stock_theme_daily.py"], env)
            _run(["python", "python/ranking_builder.py"], env)
            _run(["python", "python/build_theme_overlay_acceptance_report.py"], env)
            _copy_snapshot(str(config["scenario"]))
            rows.append(_summarize_scenario(config))
    finally:
        _restore_restore_files()
        off_env = _off_env()
        _run(["python", "python/ranking_builder.py"], off_env)
        _run(["python", "python/build_theme_overlay_acceptance_report.py"], off_env)

    df = pd.DataFrame(rows)
    _write_outputs(df)
    print(str(OUT_CSV))
    print(str(OUT_MD))


if __name__ == "__main__":
    main()
