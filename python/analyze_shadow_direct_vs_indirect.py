from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "data"

RANKING_FINAL_CSV = DATA_DIR / "ranking_final.csv"
SENSITIVITY_CSV = DATA_DIR / "theme_shadow_floor_sensitivity.csv"
SENSITIVITY_MD = DATA_DIR / "theme_shadow_floor_sensitivity_summary.md"

OUT_MD = DATA_DIR / "theme_shadow_direct_vs_indirect.md"
OUT_DIRECT_CSV = DATA_DIR / "theme_shadow_direct_uplift_candidates.csv"
OUT_INDIRECT_CSV = DATA_DIR / "theme_shadow_indirect_rank_gain_candidates.csv"
OUT_PENALTY_CSV = DATA_DIR / "theme_shadow_penalty_candidates.csv"

TARGET_FLOOR = 0.10
EPS = 1e-9


def _rerun_shadow_floor_010() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_THEME_OVERLAY": "1",
            "THEME_OVERLAY_MODE": "shadow",
            "SHADOW_THEME_WEIGHT_FLOOR": f"{TARGET_FLOOR:.2f}",
        }
    )
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)


def _restore_off() -> None:
    env = os.environ.copy()
    env.update({"ENABLE_THEME_OVERLAY": "0", "THEME_OVERLAY_MODE": "off"})
    env.pop("SHADOW_THEME_WEIGHT_FLOOR", None)
    subprocess.run(["python", "python/ranking_builder.py"], cwd=ROOT, env=env, check=True)
    subprocess.run(["python", "python/build_theme_overlay_acceptance_report.py"], cwd=ROOT, env=env, check=True)


def _load_latest() -> pd.DataFrame:
    df = pd.read_csv(RANKING_FINAL_CSV, dtype={"code": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date = df["date"].dropna().max()
    latest = df.loc[df["date"] == latest_date].copy()
    latest["code"] = latest["code"].astype(str).str.zfill(6)

    numeric_cols = [
        "final_score",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "theme_score",
        "theme_confidence",
        "shadow_theme_weight_raw",
        "shadow_theme_weight_effective",
        "shadow_rank_v3",
    ]
    for col in numeric_cols:
        latest[col] = pd.to_numeric(latest.get(col), errors="coerce")

    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    latest["shadow_floor_applied"] = latest.get("shadow_floor_applied", False).fillna(False).astype(bool)
    latest["baseline_rank"] = latest["final_score"].rank(method="first", ascending=False).astype(int)
    latest["shadow_rank_v3"] = latest["shadow_rank_v3"].round().astype("Int64")
    latest["rank_change_shadow"] = latest["baseline_rank"] - latest["shadow_rank_v3"]
    return latest


def _export(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "code",
        "name",
        "baseline_rank",
        "shadow_rank_v3",
        "rank_change_shadow",
        "final_score",
        "shadow_final_score_v3",
        "shadow_score_diff_v3",
        "theme_score",
        "theme_confidence",
        "dominant_theme",
        "shadow_theme_weight_raw",
        "shadow_theme_weight_effective",
        "shadow_floor_applied",
    ]
    df.loc[:, cols].to_csv(path, index=False, encoding="utf-8-sig")


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    _rerun_shadow_floor_010()
    latest = _load_latest()

    direct = latest.loc[
        latest["shadow_score_diff_v3"].gt(EPS)
        & latest["rank_change_shadow"].gt(0)
    ].copy()
    direct = direct.sort_values(["rank_change_shadow", "shadow_score_diff_v3"], ascending=[False, False])

    indirect = latest.loc[
        latest["rank_change_shadow"].gt(0)
        & (
            latest["shadow_score_diff_v3"].isna()
            | latest["shadow_score_diff_v3"].le(EPS)
        )
    ].copy()
    indirect = indirect.sort_values(["rank_change_shadow", "baseline_rank"], ascending=[False, True])

    penalty = latest.loc[
        latest["shadow_score_diff_v3"].lt(-EPS)
        & latest["rank_change_shadow"].lt(0)
    ].copy()
    penalty = penalty.sort_values(["rank_change_shadow", "shadow_score_diff_v3"], ascending=[True, True])

    _export(direct, OUT_DIRECT_CSV)
    _export(indirect, OUT_INDIRECT_CSV)
    _export(penalty, OUT_PENALTY_CSV)

    top20_direct = direct.loc[direct["shadow_rank_v3"].le(20)].copy()
    near_top20_entries = direct.loc[
        direct["baseline_rank"].between(21, 40)
        & direct["shadow_rank_v3"].le(20)
    ].copy()

    floor_row = pd.read_csv(SENSITIVITY_CSV).loc[lambda d: (d["floor"].round(2) == TARGET_FLOOR)].iloc[0]
    sensitivity_summary = SENSITIVITY_MD.read_text(encoding="utf-8").strip() if SENSITIVITY_MD.exists() else "NA"

    cases = {
        "삼성전기": latest.loc[latest["name"] == "삼성전기"].head(1),
        "하나마이크론": latest.loc[latest["name"] == "하나마이크론"].head(1),
        "대한항공": latest.loc[latest["name"] == "대한항공"].head(1),
        "BNK금융지주": latest.loc[latest["name"] == "BNK금융지주"].head(1),
        "LG화학": latest.loc[latest["name"] == "LG화학"].head(1),
        "삼성증권": latest.loc[latest["name"] == "삼성증권"].head(1),
        "NH투자증권": latest.loc[latest["name"] == "NH투자증권"].head(1),
        "레이크머티리얼즈": latest.loc[latest["name"] == "레이크머티리얼즈"].head(1),
        "솔루스첨단소재": latest.loc[latest["name"] == "솔루스첨단소재"].head(1),
        "LG전자": latest.loc[latest["name"] == "LG전자"].head(1),
        "JYP Ent.": latest.loc[latest["name"] == "JYP Ent."].head(1),
        "현대차우": latest.loc[latest["name"] == "현대차우"].head(1),
    }

    def explain_case(name: str) -> str:
        row_df = cases[name]
        if row_df.empty:
            return f"- {name}: not found"
        row = row_df.iloc[0]
        return (
            f"- {name}: baseline {int(row['baseline_rank'])} -> shadow {int(row['shadow_rank_v3'])}, "
            f"rank_change {int(row['rank_change_shadow'])}, "
            f"shadow_diff {float(row['shadow_score_diff_v3']):.4f}, "
            f"theme `{row['dominant_theme']}`, conf {float(row['theme_confidence']):.3f}, "
            f"floor_applied={bool(row['shadow_floor_applied'])}"
        )

    lines = [
        "# Theme Shadow Direct Vs Indirect",
        "",
        "## Definitions",
        "- direct uplift: `shadow_score_diff_v3 > 0` and `rank_change_shadow > 0`",
        "- indirect rank gain: `rank_change_shadow > 0` but own `shadow_score_diff_v3 <= 0` or effectively zero",
        "- direct penalty: `shadow_score_diff_v3 < 0` and `rank_change_shadow < 0`",
        "",
        "## 0.10 Floor Summary",
        f"- floor: {TARGET_FLOOR:.2f}",
        f"- overlay_uplift_count: {int(floor_row['overlay_uplift_count'])}",
        f"- rank_changed_count: {int(floor_row['rank_changed_count'])}",
        f"- shadow_uplift_p90: {float(floor_row['shadow_uplift_p90']):.4f}",
        f"- shadow_uplift_max: {float(floor_row['shadow_uplift_max']):.4f}",
        "",
        "## Conclusion",
        f"- direct uplift names: {len(direct)}",
        f"- indirect rank gain names: {len(indirect)}",
        f"- direct penalty names: {len(penalty)}",
        "- Actual theme selection effect exists, but it is narrow. Most positive rank gains are indirect and come from penalties applied to other names.",
        "- This is not yet a clean live-ready ranking improvement pattern.",
        "",
        "## Top20 Direct Uplift",
        _table(top20_direct.loc[:, [
            'code','name','baseline_rank','shadow_rank_v3','rank_change_shadow','final_score',
            'shadow_final_score_v3','shadow_score_diff_v3','theme_score','theme_confidence',
            'dominant_theme','shadow_theme_weight_raw','shadow_theme_weight_effective','shadow_floor_applied'
        ]]),
        "",
        "## Near-Top20 Direct Entry Candidates",
        _table(near_top20_entries.loc[:, [
            'code','name','baseline_rank','shadow_rank_v3','rank_change_shadow','final_score',
            'shadow_final_score_v3','shadow_score_diff_v3','theme_score','theme_confidence',
            'dominant_theme','shadow_theme_weight_raw','shadow_theme_weight_effective','shadow_floor_applied'
        ]]),
        "",
        "## Direct Uplift Candidates",
        _table(direct.loc[:, [
            'code','name','baseline_rank','shadow_rank_v3','rank_change_shadow','final_score',
            'shadow_final_score_v3','shadow_score_diff_v3','theme_score','theme_confidence',
            'dominant_theme','shadow_theme_weight_raw','shadow_theme_weight_effective','shadow_floor_applied'
        ]].head(20)),
        "",
        "## Indirect Rank Gain Candidates",
        _table(indirect.loc[:, [
            'code','name','baseline_rank','shadow_rank_v3','rank_change_shadow','final_score',
            'shadow_final_score_v3','shadow_score_diff_v3','theme_score','theme_confidence',
            'dominant_theme','shadow_theme_weight_raw','shadow_theme_weight_effective','shadow_floor_applied'
        ]].head(20)),
        "",
        "## Direct Penalty Candidates",
        _table(penalty.loc[:, [
            'code','name','baseline_rank','shadow_rank_v3','rank_change_shadow','final_score',
            'shadow_final_score_v3','shadow_score_diff_v3','theme_score','theme_confidence',
            'dominant_theme','shadow_theme_weight_raw','shadow_theme_weight_effective','shadow_floor_applied'
        ]].head(20)),
        "",
        "## Case Notes",
        explain_case("삼성전기"),
        explain_case("하나마이크론"),
        explain_case("대한항공"),
        explain_case("BNK금융지주"),
        explain_case("LG화학"),
        explain_case("삼성증권"),
        explain_case("NH투자증권"),
        explain_case("레이크머티리얼즈"),
        explain_case("솔루스첨단소재"),
        explain_case("LG전자"),
        explain_case("JYP Ent."),
        explain_case("현대차우"),
        "",
        "## Interpretation",
        "- `삼성전기` and `하나마이크론` are genuine direct uplift names. Their shadow score increased and that translated into rank improvement.",
        "- `대한항공` is also direct uplift, but the score increase is small and only modestly changes rank.",
        "- `BNK금융지주`, `LG화학`, `삼성증권`, `NH투자증권`, `레이크머티리얼즈` are direct penalty cases. Their theme-effective score is materially below baseline, so floor blending drags them down.",
        "- `솔루스첨단소재`, `LG전자`, `JYP Ent.`, `현대차우` are indirect rank gains. Their own shadow uplift is zero, but they move up because other names are penalized harder.",
        "",
        "## Live-Risk Patterns",
        "- A stock can rise in shadow ranking without any positive theme signal of its own, which is dangerous for live adoption.",
        "- Strong baseline names with mediocre theme-effective score are heavily penalized, which can create unstable rotation.",
        "",
        "## Next Candidates",
        "- asymmetric positive overlay: only grant positive overlay when `theme_score_effective > final_score`",
        "- downside clamp: weaken or clamp to zero the penalty when `theme_score_effective < final_score`",
        "",
        "## Reference",
        sensitivity_summary,
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _restore_off()
    print(str(OUT_MD))
    print(str(OUT_DIRECT_CSV))
    print(str(OUT_INDIRECT_CSV))
    print(str(OUT_PENALTY_CSV))


if __name__ == "__main__":
    main()
