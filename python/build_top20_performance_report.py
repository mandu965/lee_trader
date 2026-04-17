from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from outcome_maturity import attach_forward_outcomes, load_price_history


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANKING_HISTORY_DIR = DATA_DIR / "history" / "ranking"
DEFAULT_RANKING_CURRENT = DATA_DIR / "ranking_final.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "top20_performance_report.csv"
DEFAULT_OUTPUT_MD = DATA_DIR / "top20_performance_report.md"
DEFAULT_OUTPUT_JSON = DATA_DIR / "top20_performance_dashboard.json"
HORIZONS = [20, 60, 90]
TOP_N = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Top20 forward performance report from ranking snapshots.")
    p.add_argument("--ranking-history-dir", type=Path, default=RANKING_HISTORY_DIR)
    p.add_argument("--ranking-current-csv", type=Path, default=DEFAULT_RANKING_CURRENT)
    p.add_argument("--prices-csv", type=Path, default=DATA_DIR / "prices_daily_adjusted.csv")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUTPUT_MD)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return p.parse_args()


def _fmt(value: object, digits: int = 4) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(item) for item in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def confidence_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        pd.Series(
            pd.NA,
            index=series.index,
            dtype="object",
        )
    ).mask(values >= 80, "high").mask((values >= 60) & (values < 80), "medium").fillna("low")


def load_ranking_snapshots(history_dir: Path, current_csv: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if history_dir.exists():
        for path in sorted(history_dir.glob("*_ranking_final.csv")):
            df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            df["snapshot_file"] = path.name
            frames.append(df)

    if current_csv.exists():
        current = pd.read_csv(current_csv, dtype={"code": str}, low_memory=False)
        current["snapshot_file"] = current_csv.name
        frames.append(current)

    if not frames:
        raise FileNotFoundError("No ranking snapshots found.")

    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["rank_final"] = pd.to_numeric(df.get("rank_final"), errors="coerce")
    df["confidence_score"] = pd.to_numeric(df.get("confidence_score"), errors="coerce")
    df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
    df = df.dropna(subset=["date", "code", "rank_final"]).copy()

    # Prefer archived snapshots over current file duplicates for the same date/code.
    df["snapshot_priority"] = df["snapshot_file"].eq(current_csv.name).astype(int)
    df = (
        df.sort_values(["date", "code", "snapshot_priority"])
        .drop_duplicates(["date", "code"], keep="first")
        .drop(columns=["snapshot_priority"])
        .reset_index(drop=True)
    )
    return df


def attach_outcomes(ranking: pd.DataFrame, prices_csv: Path) -> pd.DataFrame:
    price_history = load_price_history(prices_csv=prices_csv)
    work = ranking.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()

    for horizon in HORIZONS:
        outcome = attach_forward_outcomes(price_history, horizon_days=horizon).rename(
            columns={
                "realized_return": f"realized_return_{horizon}d",
                "realized_mdd": f"realized_mdd_{horizon}d",
            }
        )
        outcome["date"] = pd.to_datetime(outcome["date"], errors="coerce").dt.normalize()
        work = work.merge(outcome, on=["code", "date"], how="left")

    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    return work


def build_daily_top20_frame(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = df.loc[df["rank_final"] <= top_n].copy()
    top = top.sort_values(["date", "rank_final", "final_score"], ascending=[True, True, False]).reset_index(drop=True)
    top["confidence_bucket"] = confidence_bucket(top["confidence_score"])

    prev_codes_by_date: dict[str, set[str]] = {}
    ordered_dates = sorted(top["date"].dropna().unique().tolist())
    previous_codes: set[str] = set()
    for date_value in ordered_dates:
        prev_codes_by_date[date_value] = set(previous_codes)
        previous_codes = set(top.loc[top["date"] == date_value, "code"].astype(str))

    top["is_new_entry"] = top.apply(
        lambda row: row["code"] not in prev_codes_by_date.get(row["date"], set()),
        axis=1,
    )
    return top


def summarize_daily(top: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_value, group in top.groupby("date", sort=True):
        row: dict[str, object] = {
            "date": date_value,
            "top_n": int(len(group)),
            "new_entry_count": int(group["is_new_entry"].sum()),
            "high_confidence_count": int((group["confidence_bucket"] == "high").sum()),
            "medium_confidence_count": int((group["confidence_bucket"] == "medium").sum()),
            "low_confidence_count": int((group["confidence_bucket"] == "low").sum()),
            "avg_final_score": float(pd.to_numeric(group["final_score"], errors="coerce").mean()),
            "avg_confidence_score": float(pd.to_numeric(group["confidence_score"], errors="coerce").mean()),
        }
        for horizon in HORIZONS:
            ret_col = f"realized_return_{horizon}d"
            mdd_col = f"realized_mdd_{horizon}d"
            matured = group.loc[pd.to_numeric(group[ret_col], errors="coerce").notna()].copy()
            row[f"matured_count_{horizon}d"] = int(len(matured))
            row[f"avg_return_{horizon}d"] = float(pd.to_numeric(matured[ret_col], errors="coerce").mean()) if not matured.empty else None
            row[f"median_return_{horizon}d"] = float(pd.to_numeric(matured[ret_col], errors="coerce").median()) if not matured.empty else None
            row[f"hit_rate_{horizon}d"] = float((pd.to_numeric(matured[ret_col], errors="coerce") > 0).mean()) if not matured.empty else None
            row[f"avg_mdd_{horizon}d"] = float(pd.to_numeric(matured[mdd_col], errors="coerce").mean()) if not matured.empty else None
            new_entries = matured.loc[matured["is_new_entry"]].copy()
            row[f"new_entry_avg_return_{horizon}d"] = float(pd.to_numeric(new_entries[ret_col], errors="coerce").mean()) if not new_entries.empty else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_bucket(top: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        ret_col = f"realized_return_{horizon}d"
        mdd_col = f"realized_mdd_{horizon}d"
        matured = top.loc[pd.to_numeric(top[ret_col], errors="coerce").notna()].copy()
        if matured.empty:
            continue
        grouped = (
            matured.groupby("confidence_bucket", dropna=False)
            .agg(
                n=("code", "size"),
                avg_return=(ret_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
                median_return=(ret_col, lambda s: float(pd.to_numeric(s, errors="coerce").median())),
                hit_rate=(ret_col, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
                avg_mdd=(mdd_col, lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            )
            .reset_index()
        )
        grouped["horizon_days"] = horizon
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)


def summarize_overall(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        matured_col = f"matured_count_{horizon}d"
        available = daily.loc[pd.to_numeric(daily[matured_col], errors="coerce") > 0].copy()
        if available.empty:
            continue
        rows.append(
            {
                "horizon_days": horizon,
                "dates": int(len(available)),
                "avg_return": float(pd.to_numeric(available[f"avg_return_{horizon}d"], errors="coerce").mean()),
                "median_return": float(pd.to_numeric(available[f"median_return_{horizon}d"], errors="coerce").mean()),
                "hit_rate": float(pd.to_numeric(available[f"hit_rate_{horizon}d"], errors="coerce").mean()),
                "avg_mdd": float(pd.to_numeric(available[f"avg_mdd_{horizon}d"], errors="coerce").mean()),
                "avg_new_entry_return": float(pd.to_numeric(available[f"new_entry_avg_return_{horizon}d"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def build_markdown(daily: pd.DataFrame, overall: pd.DataFrame, bucket: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Top20 성능 리포트",
        "",
        f"- 생성 시각: {generated_at}",
        f"- 분석 기준: 일자별 Top20 스냅샷 누적",
        f"- 사용 horizon: {', '.join(str(x) for x in HORIZONS)}일",
        "",
    ]

    if overall.empty:
        lines.extend(["성숙한 성과 데이터가 아직 없습니다.", ""])
    else:
        overall_rows = [
            [
                int(row["horizon_days"]),
                int(row["dates"]),
                _fmt_pct(row["avg_return"]),
                _fmt_pct(row["median_return"]),
                _fmt(row["hit_rate"]),
                _fmt_pct(row["avg_mdd"]),
                _fmt_pct(row["avg_new_entry_return"]),
            ]
            for _, row in overall.iterrows()
        ]
        lines.extend(
            [
                "## 전체 요약",
                "",
                _markdown_table(
                    overall_rows,
                    ["horizon", "dates", "avg_return", "median_return", "hit_rate", "avg_mdd", "new_entry_avg_return"],
                ),
                "",
            ]
        )

    if not bucket.empty:
        bucket_rows = [
            [
                int(row["horizon_days"]),
                row["confidence_bucket"],
                int(row["n"]),
                _fmt_pct(row["avg_return"]),
                _fmt_pct(row["median_return"]),
                _fmt(row["hit_rate"]),
                _fmt_pct(row["avg_mdd"]),
            ]
            for _, row in bucket.iterrows()
        ]
        lines.extend(
            [
                "## Confidence Bucket 요약",
                "",
                _markdown_table(
                    bucket_rows,
                    ["horizon", "bucket", "n", "avg_return", "median_return", "hit_rate", "avg_mdd"],
                ),
                "",
            ]
        )

    if not daily.empty:
        recent = daily.tail(10)
        recent_rows = []
        for _, row in recent.iterrows():
            recent_rows.append(
                [
                    row["date"],
                    int(row["top_n"]),
                    int(row["new_entry_count"]),
                    _fmt(row["avg_final_score"], 2),
                    _fmt(row["avg_confidence_score"], 2),
                    _fmt_pct(row["avg_return_20d"]),
                    _fmt_pct(row["avg_return_60d"]),
                    _fmt_pct(row["avg_return_90d"]),
                ]
            )
        lines.extend(
            [
                "## 최근 일자별 요약",
                "",
                _markdown_table(
                    recent_rows,
                    ["date", "top_n", "new_entry", "avg_score", "avg_conf", "ret_20d", "ret_60d", "ret_90d"],
                ),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    ranking = load_ranking_snapshots(args.ranking_history_dir, args.ranking_current_csv)
    ranking = attach_outcomes(ranking, args.prices_csv)
    top = build_daily_top20_frame(ranking, args.top_n)
    daily = summarize_daily(top)
    bucket = summarize_bucket(top)
    overall = summarize_overall(daily)

    report = daily.copy()
    report.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    md = build_markdown(daily, overall, bucket)
    args.out_md.write_text(md, encoding="utf-8")

    dashboard = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizons": HORIZONS,
        "daily": daily.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "confidence_bucket": bucket.to_dict(orient="records"),
    }
    args.out_json.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
