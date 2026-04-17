import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine


LOGGER = logging.getLogger("build_stock_theme_exposure")
CALC_VERSION = "etf_theme_exposure_v1"
DEFAULT_SAMPLE_CSV = Path("outputs") / "stock_theme_exposure_top20.csv"
DEFAULT_MAPPING_WEIGHT = 1.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stock_theme_exposure_daily from ETF holdings and theme mappings.")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Exposure date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--save-sample-csv",
        action="store_true",
        help="Save top 20 exposure sample CSV for validation.",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=DEFAULT_SAMPLE_CSV,
        help="Output path for optional top 20 sample CSV.",
    )
    return parser.parse_args()


def get_table(name: str) -> Table:
    metadata = MetaData()
    return Table(name, metadata, autoload_with=get_engine())


def load_source_frame(as_of_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT
            h.as_of_date,
            h.etf_code,
            h.stock_code,
            h.stock_name,
            h.holding_weight,
            h.holding_quantity,
            h.market_value,
            m.theme_code,
            m.mapping_confidence,
            m.is_primary
        FROM etf_holdings_snapshot h
        JOIN etf_theme_map m
          ON h.etf_code = m.etf_code
         AND h.as_of_date >= m.valid_from
         AND (m.valid_to IS NULL OR h.as_of_date <= m.valid_to)
        WHERE h.as_of_date = :as_of_date
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query, {"as_of_date": as_of_date}).mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric_cols = [
        "holding_weight",
        "holding_quantity",
        "market_value",
        "mapping_confidence",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_primary"] = df["is_primary"].fillna(False).astype(bool)
    return df


def compute_mapping_weight(df: pd.DataFrame) -> pd.Series:
    return pd.Series(DEFAULT_MAPPING_WEIGHT, index=df.index, dtype=float)


def calculate_exposure_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()

    working = source_df.copy()
    working["holding_weight"] = working["holding_weight"].fillna(0.0)
    working["mapping_confidence"] = working["mapping_confidence"].fillna(0.0)
    working["mapping_weight"] = compute_mapping_weight(working)
    working["exposure_component"] = (
        working["holding_weight"] * working["mapping_weight"] * working["mapping_confidence"]
    )

    grouped = (
        working.groupby(["as_of_date", "stock_code", "theme_code"], as_index=False)
        .agg(
            exposure_score=("exposure_component", "sum"),
            exposure_weight=("holding_weight", "sum"),
            supporting_etf_count=("etf_code", "nunique"),
        )
    )

    leader_rows = (
        working.sort_values(
            ["as_of_date", "stock_code", "theme_code", "exposure_component", "is_primary"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(subset=["as_of_date", "stock_code", "theme_code"], keep="first")
        .loc[:, ["as_of_date", "stock_code", "theme_code", "etf_code"]]
        .rename(columns={"etf_code": "primary_etf_code"})
    )

    out = grouped.merge(
        leader_rows,
        on=["as_of_date", "stock_code", "theme_code"],
        how="left",
    )
    out["calc_version"] = CALC_VERSION
    return out.sort_values(["theme_code", "exposure_score", "stock_code"], ascending=[True, False, True]).reset_index(drop=True)


def upsert_stock_theme_exposure(exposure_df: pd.DataFrame) -> int:
    if exposure_df.empty:
        return 0

    table = get_table("stock_theme_exposure_daily")
    payload = []
    for row in exposure_df.to_dict(orient="records"):
        payload.append(
            {
                "as_of_date": row["as_of_date"],
                "stock_code": row["stock_code"],
                "theme_code": row["theme_code"],
                "exposure_score": row["exposure_score"],
                "exposure_weight": row["exposure_weight"],
                "supporting_etf_count": int(row["supporting_etf_count"]),
                "primary_etf_code": row.get("primary_etf_code"),
                "calc_version": row["calc_version"],
            }
        )

    stmt = insert(table).values(payload)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["as_of_date", "stock_code", "theme_code"],
        set_={
            "exposure_score": stmt.excluded.exposure_score,
            "exposure_weight": stmt.excluded.exposure_weight,
            "supporting_etf_count": stmt.excluded.supporting_etf_count,
            "primary_etf_code": stmt.excluded.primary_etf_code,
            "calc_version": stmt.excluded.calc_version,
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def save_sample_csv(exposure_df: pd.DataFrame, out_path: Path) -> None:
    if exposure_df.empty:
        LOGGER.info("Skip sample CSV save because exposure frame is empty")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = (
        exposure_df.sort_values(["exposure_score", "theme_code", "stock_code"], ascending=[False, True, True])
        .head(20)
        .copy()
    )
    sample.to_csv(out_path, index=False, encoding="utf-8")
    LOGGER.info("Saved stock theme exposure sample CSV path=%s rows=%s", out_path, len(sample))


def print_summary(*, as_of_date: str, source_rows: int, exposure_rows: int, upserted_rows: int) -> None:
    print(
        "Stock theme exposure build completed "
        f"as_of_date={as_of_date} source_rows={source_rows} exposure_rows={exposure_rows} upserted_rows={upserted_rows}"
    )


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        source_df = load_source_frame(args.as_of_date)
        if source_df.empty:
            LOGGER.warning("No source rows found for as_of_date=%s", args.as_of_date)
            print_summary(as_of_date=args.as_of_date, source_rows=0, exposure_rows=0, upserted_rows=0)
            return 0

        exposure_df = calculate_exposure_frame(source_df)
        upserted_rows = upsert_stock_theme_exposure(exposure_df)

        if args.save_sample_csv:
            save_sample_csv(exposure_df, args.sample_csv)

        LOGGER.info(
            "Stock theme exposure build finished as_of_date=%s source_rows=%s exposure_rows=%s upserted_rows=%s",
            args.as_of_date,
            len(source_df),
            len(exposure_df),
            upserted_rows,
        )
        if not exposure_df.empty:
            preview = exposure_df.head(10).to_dict(orient="records")
            LOGGER.info("Exposure preview: %s", json.dumps(preview, ensure_ascii=False))

        print_summary(
            as_of_date=args.as_of_date,
            source_rows=len(source_df),
            exposure_rows=len(exposure_df),
            upserted_rows=upserted_rows,
        )
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while building stock theme exposure: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Stock theme exposure build failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
