import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine


LOGGER = logging.getLogger("calculate_theme_score")
CALC_VERSION = "theme_score_v1"
DEFAULT_REPORT_CSV = Path("outputs") / "theme_score_distribution.csv"
WINSORIZE_LOWER_Q = 0.05
WINSORIZE_UPPER_Q = 0.95
CONFIDENCE_WEIGHTS = {
    "etf_count": 0.3,
    "exposure_weight": 0.4,
    "mapping_confidence": 0.3,
}
MAX_ETF_COUNT_FOR_CONFIDENCE = 5.0
MAX_EXPOSURE_WEIGHT_FOR_CONFIDENCE = 100.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate theme_score_daily from theme exposure and ETF strength.")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Theme score date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--save-report-csv",
        action="store_true",
        help="Save distribution report CSV for inspection.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=DEFAULT_REPORT_CSV,
        help="Output path for the optional theme score report CSV.",
    )
    return parser.parse_args()


def get_table(name: str) -> Table:
    metadata = MetaData()
    return Table(name, metadata, autoload_with=get_engine())


def load_theme_source_frame(as_of_date: str) -> pd.DataFrame:
    query = text(
        """
        WITH exposure AS (
            SELECT
                as_of_date,
                stock_code,
                theme_code,
                exposure_score,
                exposure_weight,
                supporting_etf_count,
                primary_etf_code
            FROM stock_theme_exposure_daily
            WHERE as_of_date = :as_of_date
        ),
        theme_etf AS (
            SELECT
                h.as_of_date,
                m.theme_code,
                h.etf_code,
                SUM(COALESCE(h.holding_weight, 0)) AS total_holding_weight,
                AVG(COALESCE(m.mapping_confidence, 0)) AS avg_mapping_confidence,
                MAX(COALESCE(s.signal_score, 0)) AS etf_signal_score
            FROM etf_holdings_snapshot h
            JOIN etf_theme_map m
              ON h.etf_code = m.etf_code
             AND h.as_of_date >= m.valid_from
             AND (m.valid_to IS NULL OR h.as_of_date <= m.valid_to)
            LEFT JOIN etf_signal_daily s
              ON h.as_of_date = s.as_of_date
             AND h.etf_code = s.etf_code
            WHERE h.as_of_date = :as_of_date
            GROUP BY h.as_of_date, m.theme_code, h.etf_code
        ),
        leader_stock AS (
            SELECT DISTINCT ON (as_of_date, theme_code)
                as_of_date,
                theme_code,
                stock_code AS leader_stock_code,
                exposure_score AS leader_stock_exposure
            FROM exposure
            ORDER BY as_of_date, theme_code, exposure_score DESC, stock_code
        ),
        leader_etf AS (
            SELECT DISTINCT ON (as_of_date, theme_code)
                as_of_date,
                theme_code,
                etf_code AS leader_etf_code,
                (total_holding_weight * avg_mapping_confidence * etf_signal_score) AS leader_etf_contribution
            FROM theme_etf
            ORDER BY as_of_date, theme_code, (total_holding_weight * avg_mapping_confidence * etf_signal_score) DESC, etf_code
        )
        SELECT
            e.as_of_date,
            e.theme_code,
            SUM(COALESCE(e.exposure_score, 0)) AS exposure_score_total,
            SUM(COALESCE(e.exposure_weight, 0)) AS total_exposure_weight,
            COUNT(DISTINCT e.stock_code) AS stock_count,
            COUNT(DISTINCT te.etf_code) AS etf_count,
            AVG(COALESCE(te.avg_mapping_confidence, 0)) AS avg_mapping_confidence,
            AVG(COALESCE(te.etf_signal_score, 0)) AS avg_etf_signal_score,
            MAX(ls.leader_stock_code) AS leader_stock_code,
            MAX(le.leader_etf_code) AS leader_etf_code
        FROM exposure e
        LEFT JOIN theme_etf te
          ON e.as_of_date = te.as_of_date
         AND e.theme_code = te.theme_code
        LEFT JOIN leader_stock ls
          ON e.as_of_date = ls.as_of_date
         AND e.theme_code = ls.theme_code
        LEFT JOIN leader_etf le
          ON e.as_of_date = le.as_of_date
         AND e.theme_code = le.theme_code
        GROUP BY e.as_of_date, e.theme_code
        ORDER BY e.theme_code
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query, {"as_of_date": as_of_date}).mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [
        "exposure_score_total",
        "total_exposure_weight",
        "stock_count",
        "etf_count",
        "avg_mapping_confidence",
        "avg_etf_signal_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if series.empty:
        return series
    clean = pd.to_numeric(series, errors="coerce")
    lower = clean.quantile(lower_q)
    upper = clean.quantile(upper_q)
    return clean.clip(lower=lower, upper=upper)


def normalize_rank_pct(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if len(series) == 1:
        return pd.Series([1.0], index=series.index, dtype=float)
    return series.rank(method="average", pct=True)


def compute_theme_confidence(df: pd.DataFrame) -> pd.Series:
    etf_count_component = (df["etf_count"].fillna(0.0) / MAX_ETF_COUNT_FOR_CONFIDENCE).clip(0.0, 1.0)
    exposure_weight_component = (df["total_exposure_weight"].fillna(0.0) / MAX_EXPOSURE_WEIGHT_FOR_CONFIDENCE).clip(0.0, 1.0)
    mapping_conf_component = df["avg_mapping_confidence"].fillna(0.0).clip(0.0, 1.0)
    return (
        etf_count_component * CONFIDENCE_WEIGHTS["etf_count"]
        + exposure_weight_component * CONFIDENCE_WEIGHTS["exposure_weight"]
        + mapping_conf_component * CONFIDENCE_WEIGHTS["mapping_confidence"]
    ).round(4)


def calculate_theme_score_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()

    out = source_df.copy()
    out["dominant_theme"] = out["theme_code"]
    out["raw_theme_score"] = (
        out["exposure_score_total"].fillna(0.0)
        * out["avg_etf_signal_score"].fillna(0.0)
    )
    out["winsorized_raw_theme_score"] = winsorize_series(
        out["raw_theme_score"],
        lower_q=WINSORIZE_LOWER_Q,
        upper_q=WINSORIZE_UPPER_Q,
    )
    out["normalized_theme_score"] = normalize_rank_pct(out["winsorized_raw_theme_score"]).round(6)
    out["theme_confidence"] = compute_theme_confidence(out)
    out["calc_version"] = CALC_VERSION
    return out.sort_values(["normalized_theme_score", "theme_code"], ascending=[False, True]).reset_index(drop=True)


def upsert_theme_score_daily(theme_df: pd.DataFrame) -> int:
    if theme_df.empty:
        return 0

    table = get_table("theme_score_daily")
    payload = []
    for row in theme_df.to_dict(orient="records"):
        payload.append(
            {
                "as_of_date": row["as_of_date"],
                "theme_code": row["theme_code"],
                "theme_score": row["normalized_theme_score"],
                "signal_score": row["raw_theme_score"],
                "breadth_count": int(row["etf_count"]) if pd.notna(row["etf_count"]) else 0,
                "leader_etf_code": row.get("leader_etf_code"),
                "leader_stock_code": row.get("leader_stock_code"),
                "calc_version": row["calc_version"],
            }
        )

    stmt = insert(table).values(payload)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["as_of_date", "theme_code"],
        set_={
            "theme_score": stmt.excluded.theme_score,
            "signal_score": stmt.excluded.signal_score,
            "breadth_count": stmt.excluded.breadth_count,
            "leader_etf_code": stmt.excluded.leader_etf_code,
            "leader_stock_code": stmt.excluded.leader_stock_code,
            "calc_version": stmt.excluded.calc_version,
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def save_report_csv(theme_df: pd.DataFrame, out_path: Path) -> None:
    if theme_df.empty:
        LOGGER.info("Skip report CSV because theme score frame is empty")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "as_of_date",
        "theme_code",
        "dominant_theme",
        "raw_theme_score",
        "winsorized_raw_theme_score",
        "normalized_theme_score",
        "theme_confidence",
        "etf_count",
        "stock_count",
        "total_exposure_weight",
        "avg_mapping_confidence",
        "avg_etf_signal_score",
        "leader_etf_code",
        "leader_stock_code",
        "calc_version",
    ]
    theme_df.loc[:, columns].to_csv(out_path, index=False, encoding="utf-8")
    LOGGER.info("Saved theme score distribution report path=%s rows=%s", out_path, len(theme_df))


def print_summary(*, as_of_date: str, theme_rows: int, upserted_rows: int) -> None:
    print(
        "Theme score calculation completed "
        f"as_of_date={as_of_date} theme_rows={theme_rows} upserted_rows={upserted_rows}"
    )


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        source_df = load_theme_source_frame(args.as_of_date)
        if source_df.empty:
            LOGGER.warning("No theme source rows found for as_of_date=%s", args.as_of_date)
            print_summary(as_of_date=args.as_of_date, theme_rows=0, upserted_rows=0)
            return 0

        theme_df = calculate_theme_score_frame(source_df)
        upserted_rows = upsert_theme_score_daily(theme_df)

        if args.save_report_csv:
            save_report_csv(theme_df, args.report_csv)

        LOGGER.info(
            "Theme score calculation finished as_of_date=%s theme_rows=%s upserted_rows=%s",
            args.as_of_date,
            len(theme_df),
            upserted_rows,
        )
        if not theme_df.empty:
            preview = theme_df.head(10).to_dict(orient="records")
            LOGGER.info("Theme score preview: %s", json.dumps(preview, ensure_ascii=False))

        print_summary(as_of_date=args.as_of_date, theme_rows=len(theme_df), upserted_rows=upserted_rows)
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while calculating theme score: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Theme score calculation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
