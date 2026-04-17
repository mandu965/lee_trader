import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from db import get_engine
except Exception:
    get_engine = None


OUTPUT_DIR = Path("output")
SUMMARY_CSV = OUTPUT_DIR / "flow_ingestion_summary.csv"
VALIDATION_MD = OUTPUT_DIR / "flow_ingestion_validation.md"
FAILURE_CSV = OUTPUT_DIR / "flow_ingestion_failures.csv"
DEFAULT_SAMPLE_CODES = ["005930", "000660", "035420"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_codes() -> List[str]:
    raw = os.getenv("FLOW_SAMPLE_CODES") or os.getenv("SYMBOLS", "")
    if raw.strip():
        codes = [part.strip().zfill(6) for part in raw.split(",") if part.strip()]
        if codes:
            return codes[:3]
    return DEFAULT_SAMPLE_CODES


def load_failure_rows() -> List[Dict[str, Any]]:
    if not FAILURE_CSV.exists():
        return []
    with FAILURE_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_flow_daily_df() -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("DB engine unavailable")
    engine = get_engine()
    with engine.begin() as conn:
        return pd.read_sql("SELECT * FROM flow_daily", conn)


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"latest_date": None, "total_rows": 0, "unique_codes": 0}
    latest_ts = pd.to_datetime(df["date"]).max()
    latest_df = df.loc[pd.to_datetime(df["date"]) == latest_ts].copy()
    return {
        "latest_date": latest_ts.date().isoformat(),
        "total_rows": int(len(latest_df)),
        "unique_codes": int(latest_df["code"].astype(str).nunique()),
    }


def investor_type_distribution(df: pd.DataFrame, latest_date: str | None) -> List[Dict[str, Any]]:
    if df.empty or not latest_date:
        return []
    latest_df = df.loc[pd.to_datetime(df["date"]).dt.date.astype(str) == latest_date].copy()
    counts = latest_df.groupby("investor_type").size().reset_index(name="row_count")
    return counts.to_dict(orient="records")


def code_level_presence(df: pd.DataFrame, latest_date: str | None) -> pd.DataFrame:
    if df.empty or not latest_date:
        return pd.DataFrame(
            columns=["date", "code", "row_count", "presence_ratio", "pair_coverage", "investor_types", "fetch_statuses"]
        )
    latest_df = df.loc[pd.to_datetime(df["date"]).dt.date.astype(str) == latest_date].copy()
    grouped = (
        latest_df.groupby(["date", "code"])
        .agg(
            row_count=("investor_type", "size"),
            investor_types=("investor_type", lambda s: ",".join(sorted(set(map(str, s))))),
            fetch_statuses=("fetch_status", lambda s: ",".join(sorted(set(filter(None, map(str, s)))))),
        )
        .reset_index()
    )
    grouped["presence_ratio"] = grouped["row_count"] / 2.0
    grouped["pair_coverage"] = grouped["investor_types"].eq("foreign,institution")
    return grouped.sort_values(["date", "code"]).reset_index(drop=True)


def fetch_status_distribution(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty or "fetch_status" not in df.columns:
        return []
    counts = (
        df.assign(fetch_status=df["fetch_status"].fillna("NULL").astype(str))
        .groupby("fetch_status")
        .size()
        .reset_index(name="row_count")
    )
    return counts.to_dict(orient="records")


def error_code_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty or "error_code" not in df.columns:
        return []
    filtered = df.loc[df["error_code"].notna() & (df["error_code"].astype(str).str.strip() != "")]
    if filtered.empty:
        return []
    return filtered.groupby("error_code").size().reset_index(name="row_count").to_dict(orient="records")


def recent_business_days(df: pd.DataFrame, count: int = 5) -> List[str]:
    if df.empty:
        return []
    dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
    return [d.isoformat() for d in dates[-count:]]


def sample_code_validation(df: pd.DataFrame, codes: List[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    dates = recent_business_days(df, count=5)
    if not dates:
        return []
    subset = df.loc[
        df["code"].astype(str).isin(codes)
        & pd.to_datetime(df["date"]).dt.date.astype(str).isin(dates)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for code in codes:
        for date_value in dates:
            item = subset.loc[
                (subset["code"].astype(str) == code)
                & (pd.to_datetime(subset["date"]).dt.date.astype(str) == date_value)
            ]
            investor_types = ",".join(sorted(set(item["investor_type"].astype(str)))) if not item.empty else ""
            rows.append(
                {
                    "date": date_value,
                    "code": code,
                    "row_count": int(len(item)),
                    "pair_coverage": investor_types == "foreign,institution",
                    "investor_types": investor_types,
                    "missing_flag": len(item) < 2,
                }
            )
    return rows


def write_summary_csv(rows: pd.DataFrame) -> None:
    rows.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")


def next_fix_points(*, empty_df: bool, failures: List[Dict[str, Any]], pair_cov_ratio: float, has_partial: bool) -> List[str]:
    points: List[str] = []
    if empty_df:
        points.append("No rows exist in flow_daily. Check KIS connectivity/authentication first, then verify that schema.sql has been applied.")
    if failures:
        first = failures[0]
        points.append(
            f"Recent failure report exists. Prioritize stage={first.get('stage')} and error_type={first.get('error_type')}."
        )
    if not empty_df and pair_cov_ratio < 0.95:
        points.append("Pair coverage on latest_date is below target. Review normalize mapping and paging completeness.")
    if has_partial:
        points.append("Some rows have is_partial_page=true. Recheck continuation handling and duplicate-page safety.")
    if not points:
        points.append("No major issue is visible under the current validation rules. Keep monitoring 5-day sample-code completeness.")
    return points


def write_validation_md(
    *,
    summary: Dict[str, Any],
    inv_dist: List[Dict[str, Any]],
    code_presence: pd.DataFrame,
    fetch_dist: List[Dict[str, Any]],
    error_summary: List[Dict[str, Any]],
    sample_validation: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    has_partial: bool,
) -> None:
    latest_date = summary.get("latest_date")
    pair_cov_ratio = float(code_presence["pair_coverage"].mean()) if not code_presence.empty else 0.0
    fix_points = next_fix_points(
        empty_df=code_presence.empty,
        failures=failures,
        pair_cov_ratio=pair_cov_ratio,
        has_partial=has_partial,
    )

    lines: List[str] = []
    lines.append("# Flow Ingestion Validation")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- latest_date: {latest_date or 'unavailable'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| latest_date | {latest_date or 'unavailable'} |")
    lines.append(f"| total_rows | {summary.get('total_rows', 0)} |")
    lines.append(f"| unique_codes | {summary.get('unique_codes', 0)} |")
    lines.append(f"| foreign/institution pair coverage | {pair_cov_ratio:.2%} |")
    lines.append("")

    lines.append("## Investor Type Distribution")
    lines.append("")
    if inv_dist:
        lines.append("| investor_type | row_count |")
        lines.append("| --- | ---: |")
        for row in inv_dist:
            lines.append(f"| {row['investor_type']} | {row['row_count']} |")
    else:
        lines.append("No rows available.")
    lines.append("")

    lines.append("## Code-Level Presence")
    lines.append("")
    if not code_presence.empty:
        lines.append("| date | code | row_count | presence_ratio | pair_coverage | investor_types | fetch_statuses |")
        lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
        for row in code_presence.to_dict(orient="records"):
            lines.append(
                f"| {row['date']} | {row['code']} | {row['row_count']} | {row['presence_ratio']:.2f} | {row['pair_coverage']} | {row['investor_types']} | {row['fetch_statuses']} |"
            )
    else:
        lines.append("No rows available.")
    lines.append("")

    lines.append("## Fetch Status Distribution")
    lines.append("")
    if fetch_dist:
        lines.append("| fetch_status | row_count |")
        lines.append("| --- | ---: |")
        for row in fetch_dist:
            lines.append(f"| {row['fetch_status']} | {row['row_count']} |")
    else:
        lines.append("No rows available.")
    lines.append("")

    lines.append("## Error Code Summary")
    lines.append("")
    if error_summary:
        lines.append("| error_code | row_count |")
        lines.append("| --- | ---: |")
        for row in error_summary:
            lines.append(f"| {row['error_code']} | {row['row_count']} |")
    else:
        lines.append("No in-table error_code rows.")
    lines.append("")

    lines.append("## Sample Code Recent 5-Day Check")
    lines.append("")
    if sample_validation:
        lines.append("| date | code | row_count | pair_coverage | investor_types | missing_flag |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        for row in sample_validation:
            lines.append(
                f"| {row['date']} | {row['code']} | {row['row_count']} | {row['pair_coverage']} | {row['investor_types']} | {row['missing_flag']} |"
            )
    else:
        lines.append("No sample validation rows available.")
    lines.append("")

    lines.append("## Failure Snapshot")
    lines.append("")
    if failures:
        lines.append("| code | error_type | stage | failed_at |")
        lines.append("| --- | --- | --- | --- |")
        for row in failures[:10]:
            lines.append(f"| {row.get('code','')} | {row.get('error_type','')} | {row.get('stage','')} | {row.get('failed_at','')} |")
    else:
        lines.append("No failure file found.")
    lines.append("")

    lines.append("## Next Fix Points")
    lines.append("")
    for point in fix_points:
        lines.append(f"- {point}")
    lines.append("")

    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    setup_logging()
    ensure_output_dir()

    failures = load_failure_rows()
    try:
        df = fetch_flow_daily_df()
    except Exception as exc:
        logging.warning("Unable to load flow_daily: %s", exc)
        df = pd.DataFrame()

    if not df.empty and "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)

    summary = compute_summary(df)
    inv_dist = investor_type_distribution(df, summary.get("latest_date"))
    code_presence = code_level_presence(df, summary.get("latest_date"))
    fetch_dist = fetch_status_distribution(df)
    error_summary = error_code_summary(df)
    sample_validation = sample_code_validation(df, sample_codes())
    has_partial = bool(df["is_partial_page"].fillna(False).astype(bool).any()) if not df.empty and "is_partial_page" in df.columns else False

    write_summary_csv(code_presence)
    write_validation_md(
        summary=summary,
        inv_dist=inv_dist,
        code_presence=code_presence,
        fetch_dist=fetch_dist,
        error_summary=error_summary,
        sample_validation=sample_validation,
        failures=failures,
        has_partial=has_partial,
    )

    logging.info(
        "Flow ingestion validation complete: latest_date=%s total_rows=%s unique_codes=%s summary_csv=%s md=%s",
        summary.get("latest_date"),
        summary.get("total_rows"),
        summary.get("unique_codes"),
        SUMMARY_CSV,
        VALIDATION_MD,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
