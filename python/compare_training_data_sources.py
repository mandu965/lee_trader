from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy import create_engine, text


TABLES = [
    "fact_price_daily",
    "prices_adjusted",
    "features",
    "labels",
    "predictions",
    "daily_ranking",
]


@dataclass
class TableSnapshot:
    table_name: str
    total_rows: int
    min_date: str | None
    latest_date: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training-data related tables between local Docker DB and web DB."
    )
    parser.add_argument("--local-url", default=os.environ.get("LOCAL_DATABASE_URL"), help="Local Docker Postgres URL.")
    parser.add_argument("--web-url", default=os.environ.get("WEB_DATABASE_URL"), help="Web Postgres URL.")
    parser.add_argument("--code", default="096530", help="Code to inspect in predictions/daily_ranking.")
    parser.add_argument(
        "--output-json",
        default="outputs/training_data_compare_report.json",
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def snapshot_tables(engine) -> list[TableSnapshot]:
    snapshots: list[TableSnapshot] = []
    with engine.connect() as conn:
        for table in TABLES:
            row = conn.execute(
                text(
                    f"""
                    SELECT
                        COUNT(*) AS total_rows,
                        TO_CHAR(MIN(date), 'YYYY-MM-DD') AS min_date,
                        TO_CHAR(MAX(date), 'YYYY-MM-DD') AS latest_date
                    FROM {table}
                    """
                )
            ).mappings().one()
            snapshots.append(
                TableSnapshot(
                    table_name=table,
                    total_rows=int(row["total_rows"] or 0),
                    min_date=row["min_date"],
                    latest_date=row["latest_date"],
                )
            )
    return snapshots


def load_training_overlap(engine) -> dict[str, object]:
    query = text(
        """
        SELECT
            COUNT(*) AS merged_rows,
            TO_CHAR(MIN(f.date), 'YYYY-MM-DD') AS min_date,
            TO_CHAR(MAX(f.date), 'YYYY-MM-DD') AS latest_date
        FROM features f
        INNER JOIN labels l
            ON f.date = l.date
           AND f.code = l.code
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().one()
    return {
        "merged_rows": int(row["merged_rows"] or 0),
        "min_date": row["min_date"],
        "latest_date": row["latest_date"],
    }


def load_code_snapshot(engine, code: str) -> dict[str, object]:
    queries = {
        "fact_price_daily": text(
            """
            SELECT
                COUNT(*) AS rows,
                TO_CHAR(MIN(date), 'YYYY-MM-DD') AS min_date,
                TO_CHAR(MAX(date), 'YYYY-MM-DD') AS latest_date
            FROM fact_price_daily
            WHERE code = :code
            """
        ),
        "predictions": text(
            """
            SELECT
                code,
                TO_CHAR(date, 'YYYY-MM-DD') AS date,
                pred_return_60d,
                pred_return_90d,
                prob_top20_60d,
                prob_top20_90d,
                pred_mdd_60d,
                pred_mdd_90d
            FROM predictions
            WHERE code = :code
            ORDER BY date DESC
            LIMIT 1
            """
        ),
        "daily_ranking": text(
            """
            SELECT
                code,
                TO_CHAR(date, 'YYYY-MM-DD') AS date,
                final_score,
                rank_final,
                live_rank,
                live_score,
                live_score_source,
                ret_score,
                prob_score,
                tech_score,
                qual_score,
                pred_score,
                safety_score,
                liquidity_score,
                risk_penalty,
                pred_return_60d,
                prob_top20_60d,
                pred_mdd_60d,
                weight_profile,
                score_formula_version
            FROM daily_ranking
            WHERE code = :code
            ORDER BY date DESC
            LIMIT 1
            """
        ),
    }
    out: dict[str, object] = {}
    with engine.connect() as conn:
        for name, query in queries.items():
            row = conn.execute(query, {"code": code}).mappings().first()
            out[name] = dict(row) if row else None
    return out


def build_comparison(local_snapshots: list[TableSnapshot], web_snapshots: list[TableSnapshot]) -> pd.DataFrame:
    local_df = pd.DataFrame([asdict(item) for item in local_snapshots]).rename(
        columns={
            "total_rows": "local_rows",
            "min_date": "local_min_date",
            "latest_date": "local_latest_date",
        }
    )
    web_df = pd.DataFrame([asdict(item) for item in web_snapshots]).rename(
        columns={
            "total_rows": "web_rows",
            "min_date": "web_min_date",
            "latest_date": "web_latest_date",
        }
    )
    merged = local_df.merge(web_df, on="table_name", how="outer")
    merged["row_diff"] = merged["web_rows"].fillna(0).astype(int) - merged["local_rows"].fillna(0).astype(int)
    return merged.sort_values("table_name").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.local_url or not args.web_url:
        raise SystemExit("Both --local-url and --web-url are required. You can also set LOCAL_DATABASE_URL and WEB_DATABASE_URL.")

    local_engine = create_engine(args.local_url, future=True)
    web_engine = create_engine(args.web_url, future=True)

    local_snapshots = snapshot_tables(local_engine)
    web_snapshots = snapshot_tables(web_engine)
    compare_df = build_comparison(local_snapshots, web_snapshots)
    local_overlap = load_training_overlap(local_engine)
    web_overlap = load_training_overlap(web_engine)
    local_code = load_code_snapshot(local_engine, args.code)
    web_code = load_code_snapshot(web_engine, args.code)

    print("\n[Table Snapshots]")
    print(compare_df.to_string(index=False))

    print("\n[Training Overlap: features INNER JOIN labels]")
    print(
        pd.DataFrame(
            [
                {"source": "local", **local_overlap},
                {"source": "web", **web_overlap},
            ]
        ).to_string(index=False)
    )

    print(f"\n[Code Snapshot: {args.code}]")
    print(json.dumps({"local": local_code, "web": web_code}, ensure_ascii=False, indent=2, default=str))

    report = {
        "tables": compare_df.to_dict(orient="records"),
        "training_overlap": {"local": local_overlap, "web": web_overlap},
        "code_snapshot": {"code": args.code, "local": local_code, "web": web_code},
    }
    output_json = args.output_json
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved JSON report: {output_json}")


if __name__ == "__main__":
    main()
