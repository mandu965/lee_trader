from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine


BASE_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = BASE_DIR / "data" / "history" / "ranking"
META_DIR = BASE_DIR / "data" / "history" / "ranking_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild dated ranking snapshots from DB history tables."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Calendar days to scan backwards from --end-date (default: 10).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing snapshot/meta files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a date if both snapshot and meta already exist.",
    )
    return parser.parse_args()


def resolve_date_window(days: int, end_date_text: str | None) -> tuple[date, date]:
    if days <= 0:
        raise ValueError("--days must be >= 1")

    if end_date_text:
        end_date = datetime.strptime(end_date_text, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date


def load_rankings(start_date: date, end_date: date) -> pd.DataFrame:
    query = text(
        """
        WITH latest_runs AS (
            SELECT
                as_of_date,
                MAX(run_id) AS run_id
            FROM research.ranking_history
            WHERE as_of_date BETWEEN :start_date AND :end_date
            GROUP BY as_of_date
        )
        SELECT
            rh.run_id,
            rh.as_of_date,
            rh.code,
            s.name,
            s.market,
            s.sector,
            rh.model_version,
            rh.horizon_days,
            rh.rank,
            rh.final_score,
            rh.ret_score,
            rh.prob_score,
            rh.qual_score,
            rh.tech_score,
            rh.risk_penalty,
            rh.in_top_n,
            rh.top_n,
            ph.pred_return_30d,
            ph.pred_return_60d,
            ph.pred_return_90d,
            ph.pred_mdd_30d,
            ph.pred_mdd_60d,
            ph.pred_mdd_90d,
            ph.prob_top20_30d,
            ph.prob_top20_60d,
            ph.prob_top20_90d,
            rh.created_at
        FROM research.ranking_history rh
        JOIN latest_runs lr
          ON rh.as_of_date = lr.as_of_date
         AND rh.run_id = lr.run_id
        LEFT JOIN research.prediction_history ph
          ON ph.run_id = rh.run_id
         AND ph.as_of_date = rh.as_of_date
         AND ph.code = rh.code
         AND ph.horizon_days = rh.horizon_days
        LEFT JOIN stocks s
          ON s.code = rh.code
        ORDER BY rh.as_of_date ASC, rh.rank ASC, rh.code ASC
        """
    )

    engine = get_engine()
    df = pd.read_sql(
        query,
        con=engine,
        params={"start_date": start_date, "end_date": end_date},
        parse_dates=["as_of_date", "created_at"],
    )
    return df


def build_snapshot_frame(day_df: pd.DataFrame) -> pd.DataFrame:
    out = day_df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.strftime("%Y-%m-%d")
    out["date"] = out["as_of_date"]
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["rank_final"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    out["snapshot_source"] = "db_rebuild"
    out["snapshot_source_table"] = "research.ranking_history"
    out["snapshot_rebuilt_at"] = datetime.now().isoformat(timespec="seconds")
    out["snapshot_note"] = (
        "Rebuilt from DB history tables; not a byte-identical copy of ranking_final.csv."
    )

    preferred_columns = [
        "as_of_date",
        "date",
        "code",
        "name",
        "market",
        "sector",
        "run_id",
        "model_version",
        "horizon_days",
        "rank_final",
        "final_score",
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "risk_penalty",
        "pred_return_30d",
        "pred_return_60d",
        "pred_return_90d",
        "pred_mdd_30d",
        "pred_mdd_60d",
        "pred_mdd_90d",
        "prob_top20_30d",
        "prob_top20_60d",
        "prob_top20_90d",
        "in_top_n",
        "top_n",
        "created_at",
        "snapshot_source",
        "snapshot_source_table",
        "snapshot_rebuilt_at",
        "snapshot_note",
    ]
    return out[preferred_columns].sort_values(["rank_final", "code"], ascending=[True, True])


def build_meta(snapshot_df: pd.DataFrame, snapshot_path: Path) -> dict[str, object]:
    top20 = (
        snapshot_df.sort_values(["rank_final", "code"], ascending=[True, True])
        .head(20)["code"]
        .astype(str)
        .tolist()
    )
    return {
        "as_of_date": str(snapshot_df["as_of_date"].iloc[0]),
        "snapshot_file": str(snapshot_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "row_count": int(len(snapshot_df)),
        "top20_tickers": top20,
        "score_column_used": "final_score",
        "rank_column_used": "rank_final",
        "source_mode": "db_rebuild",
        "source_tables": ["research.ranking_history", "research.prediction_history", "stocks"],
        "run_id": int(snapshot_df["run_id"].iloc[0]),
        "model_version": str(snapshot_df["model_version"].iloc[0]),
        "horizon_days": int(snapshot_df["horizon_days"].iloc[0]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Rebuilt from DB history tables; columns may differ from original ranking_final.csv snapshot.",
    }


def save_snapshot(snapshot_df: pd.DataFrame, snapshot_path: Path, overwrite: bool) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists() and not overwrite:
        raise FileExistsError(f"snapshot already exists: {snapshot_path}")
    snapshot_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")


def save_meta(meta: dict[str, object], meta_path: Path, overwrite: bool) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"meta already exists: {meta_path}")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    start_date, end_date = resolve_date_window(args.days, args.end_date)
    df = load_rankings(start_date, end_date)

    print(f"scan_window: {start_date} -> {end_date}")
    if df.empty:
        print("status: no_rows_found")
        return 0

    saved = 0
    skipped = 0
    failed = 0
    found_dates = sorted(df["as_of_date"].dt.strftime("%Y-%m-%d").unique().tolist())

    print(f"found_dates: {', '.join(found_dates)}")
    for as_of_date_text, day_df in df.groupby(df["as_of_date"].dt.strftime("%Y-%m-%d"), sort=True):
        date_token = as_of_date_text.replace("-", "")
        snapshot_path = SNAPSHOT_DIR / f"{date_token}_ranking_final.csv"
        meta_path = META_DIR / f"{date_token}_ranking_meta.json"

        if args.skip_existing and snapshot_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"skip_existing: {as_of_date_text}")
            skipped += 1
            continue

        snapshot_df = build_snapshot_frame(day_df)
        meta = build_meta(snapshot_df, snapshot_path)

        try:
            save_snapshot(snapshot_df, snapshot_path, overwrite=args.overwrite)
            save_meta(meta, meta_path, overwrite=True if meta_path.exists() else args.overwrite)
            print(
                f"saved: {as_of_date_text} rows={len(snapshot_df)} "
                f"run_id={int(snapshot_df['run_id'].iloc[0])} path={snapshot_path}"
            )
            saved += 1
        except Exception as exc:
            print(f"save_failed: {as_of_date_text} error={exc}")
            failed += 1

    print(f"summary: saved={saved} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
