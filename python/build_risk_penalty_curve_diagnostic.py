from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
RANKING_CSV = DATA_DIR / "ranking_final.csv"
OUT_MD = DATA_DIR / "risk_penalty_curve_diagnostic.md"
OUT_CSV = DATA_DIR / "risk_penalty_curve_summary.csv"

BUCKETS = [
    ("<=0.10", None, 0.10),
    ("0.10~0.15", 0.10, 0.15),
    ("0.15~0.20", 0.15, 0.20),
    ("0.20~0.25", 0.20, 0.25),
    ("0.25~0.30", 0.25, 0.30),
    (">0.30", 0.30, None),
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def load_latest_ranking() -> pd.DataFrame:
    if not RANKING_CSV.exists():
        raise FileNotFoundError(f"ranking file not found: {RANKING_CSV}")

    df = pd.read_csv(RANKING_CSV)
    if df.empty:
        raise ValueError("ranking_final.csv is empty")

    if "date" not in df.columns:
        raise ValueError("ranking_final.csv missing date column")

    latest_date = df["date"].astype(str).max()
    latest = df.loc[df["date"].astype(str) == latest_date].copy()
    latest["pred_mdd_mix"] = pd.to_numeric(latest.get("pred_mdd_mix"), errors="coerce")
    latest["risk_penalty"] = pd.to_numeric(latest.get("risk_penalty"), errors="coerce").fillna(0.0)
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").replace("", "(none)")
    return latest


def assign_bucket(series: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=series.index, dtype="object")
    for label, lower, upper in BUCKETS:
        mask = pd.Series(True, index=series.index)
        if lower is not None:
            mask &= series.gt(lower)
        if upper is not None:
            mask &= series.le(upper)
        out.loc[mask] = label
    return out


def build_bucket_summary(latest: pd.DataFrame) -> pd.DataFrame:
    df = latest.copy()
    df["mdd_bucket"] = assign_bucket(df["pred_mdd_mix"])
    df["is_cap18"] = df["risk_penalty"].round(6).eq(18.0)
    grouped = (
        df.groupby("mdd_bucket", as_index=False)
        .agg(
            stock_count=("code", "count"),
            avg_pred_mdd_mix=("pred_mdd_mix", "mean"),
            avg_risk_penalty=("risk_penalty", "mean"),
            median_risk_penalty=("risk_penalty", "median"),
            max_risk_penalty=("risk_penalty", "max"),
            cap18_count=("is_cap18", "sum"),
        )
    )
    grouped["cap18_ratio"] = grouped["cap18_count"] / grouped["stock_count"].clip(lower=1)
    return grouped


def build_report(latest: pd.DataFrame, bucket_summary: pd.DataFrame) -> str:
    rows = len(latest)
    mix = latest["pred_mdd_mix"]
    rp = latest["risk_penalty"]
    cap18 = rp.round(6).eq(18.0)

    top_theme_cap = (
        latest.loc[cap18]
        .groupby("dominant_theme", as_index=False)
        .agg(stock_count=("code", "count"))
        .sort_values(["stock_count", "dominant_theme"], ascending=[False, True])
    )
    top_cap_names = latest.loc[cap18, ["code", "name", "dominant_theme", "pred_mdd_mix", "risk_penalty"]].head(20)

    lines = [
        "# Risk Penalty Curve Diagnostic",
        "",
        "## 1. Formula",
        "- mix <= 0.10 -> 0.0",
        "- 0.10 < mix <= 0.15 -> (mix - 0.10) * 40",
        "- 0.15 < mix <= 0.20 -> 2.0 + (mix - 0.15) * 80",
        "- mix > 0.20 -> 6.0 + (mix - 0.20) * 120",
        "- final clip: 0 ~ 18",
        "",
        "## 2. Latest Summary",
        f"- latest_date={latest['date'].astype(str).max()}",
        f"- rows={rows}",
        f"- pred_mdd_mix_mean={float(mix.mean()):.4f}",
        f"- pred_mdd_mix_median={float(mix.median()):.4f}",
        f"- pred_mdd_mix_min={float(mix.min()):.4f}",
        f"- pred_mdd_mix_max={float(mix.max()):.4f}",
        f"- risk_penalty_mean={float(rp.mean()):.4f}",
        f"- risk_penalty_median={float(rp.median()):.4f}",
        f"- risk_penalty_max={float(rp.max()):.4f}",
        f"- cap18_count={int(cap18.sum())}",
        f"- cap18_ratio={float(cap18.mean()):.4f}",
        "",
        "## 3. Saturation Check",
        f"- mix >= 0.20 count={int(mix.ge(0.20).sum())}",
        f"- mix >= 0.25 count={int(mix.ge(0.25).sum())}",
        f"- mix >= 0.30 count={int(mix.ge(0.30).sum())}",
        f"- mix >= 0.30 and cap18 count={int((mix.ge(0.30) & cap18).sum())}",
        "- Interpretation: current curve structurally saturates at 18 once pred_mdd_mix moves above roughly 0.30.",
        "",
        "## 4. Bucket Summary",
    ]

    for row in bucket_summary.itertuples(index=False):
        lines.append(
            f"- {row.mdd_bucket}: stock_count={int(row.stock_count)}, "
            f"avg_mix={float(row.avg_pred_mdd_mix):.4f}, "
            f"avg_penalty={float(row.avg_risk_penalty):.4f}, "
            f"cap18_count={int(row.cap18_count)}, cap18_ratio={float(row.cap18_ratio):.4f}"
        )

    lines.extend(["", "## 5. Cap18 Theme Distribution"])
    if top_theme_cap.empty:
        lines.append("- none")
    else:
        for row in top_theme_cap.itertuples(index=False):
            lines.append(f"- {row.dominant_theme}: stock_count={int(row.stock_count)}")

    lines.extend(["", "## 6. Cap18 Sample"])
    if top_cap_names.empty:
        lines.append("- none")
    else:
        for row in top_cap_names.itertuples(index=False):
            lines.append(
                f"- {row.code} {row.name}: theme={row.dominant_theme}, mix={float(row.pred_mdd_mix):.4f}, penalty={float(row.risk_penalty):.4f}"
            )

    lines.extend([
        "",
        "## 7. Diagnosis",
        "- The current issue is not only theme overlay. The baseline penalty curve itself is saturating for a large share of the universe.",
        "- Because many names are already clipped at 18, any soft-factor experiment will naturally create repeated delta values such as 2.7.",
        "- This means the next tuning priority should be the penalty curve or upper-cap structure, not a broader theme_weight increase.",
        "",
        "## 8. Next Step",
        "- Build a sidecar experiment that changes only the risk_penalty curve above mix 0.25 or moves the hard cap threshold higher, then compare cap18_count and near-top20 movers.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    latest = load_latest_ranking()
    bucket_summary = build_bucket_summary(latest)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bucket_summary.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_MD.write_text(build_report(latest, bucket_summary), encoding="utf-8")
    logging.info("Saved risk penalty curve summary: %s", OUT_CSV.resolve())
    logging.info("Saved risk penalty curve diagnostic: %s", OUT_MD.resolve())
    print(f"generated_files={[str(OUT_CSV), str(OUT_MD)]}")
    print("example=python python\\build_risk_penalty_curve_diagnostic.py")


if __name__ == "__main__":
    main()
