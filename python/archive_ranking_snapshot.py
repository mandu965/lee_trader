from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data" / "ranking_final.csv"
SNAPSHOT_DIR = BASE_DIR / "data" / "history" / "ranking"
META_DIR = BASE_DIR / "data" / "history" / "ranking_meta"
ARCHIVE_CSV = BASE_DIR / "data" / "ranking_snapshot_archive.csv"

DATE_CANDIDATE_COLUMNS = ["as_of_date", "date", "trade_date", "snapshot_date"]
THEME_COLUMNS = ["theme_score", "dominant_theme", "theme_confidence", "regime", "explain", "base_score"]
ARCHIVE_COLUMNS = [
    "asof_date",
    "rank",
    "code",
    "name",
    "final_score",
    "confidence_score",
    "ret_score",
    "prob_score",
    "tech_score",
    "quality_score",
    "safety_score",
    "risk_penalty",
    "dominant_theme",
    "theme_score",
    "explain_text",
]
ARCHIVE_KEY_COLUMNS = ["asof_date", "rank", "code"]
TOP_ARCHIVE_LIMIT = 20


def resolve_as_of_date(df: pd.DataFrame, cli_as_of_date: str | None = None) -> tuple[str, str]:
    for column in DATE_CANDIDATE_COLUMNS:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            return parsed.max().strftime("%Y-%m-%d"), column

    if cli_as_of_date:
        parsed = pd.to_datetime(cli_as_of_date, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d"), "cli_as_of_date"
        raise ValueError(f"invalid --as-of-date: {cli_as_of_date}")

    raise ValueError("could not resolve as_of_date from CSV columns or CLI")


def validate_required_columns(df: pd.DataFrame) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if "ticker" not in df.columns and "code" not in df.columns:
        raise ValueError("missing required identifier column: ticker or code")

    if "name" not in df.columns:
        warnings.append("name column missing; continuing with blank name")

    score_column_used = ""
    for column in ("final_score", "final_score_v3", "final_score_v2"):
        if column in df.columns:
            score_column_used = column
            break
    if not score_column_used:
        raise ValueError("missing required score column: final_score, final_score_v3, or final_score_v2")

    return score_column_used, warnings


def save_snapshot(df: pd.DataFrame, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"snapshot already exists: {out_path}")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def build_meta(
    df: pd.DataFrame,
    as_of_date: str,
    snapshot_path: Path,
    score_column_used: str,
    archive_path: Path,
    archive_rows_added: int,
) -> dict[str, object]:
    top = df.copy()
    top[score_column_used] = pd.to_numeric(top.get(score_column_used), errors="coerce")
    ticker_col = "ticker" if "ticker" in top.columns else "code"
    top[ticker_col] = top[ticker_col].astype(str).str.zfill(6)
    top = top.sort_values([score_column_used, ticker_col], ascending=[False, True])
    top20 = top.head(20)
    top10 = top.head(10)
    top5 = top.head(5)
    theme_present = [column for column in THEME_COLUMNS if column in df.columns]
    return {
        "as_of_date": as_of_date,
        "snapshot_file": str(snapshot_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "archive_file": str(archive_path.relative_to(BASE_DIR)).replace("\\", "/"),
        "row_count": int(len(df)),
        "archive_top20_row_count": int(min(len(top20), TOP_ARCHIVE_LIMIT)),
        "archive_rows_added": int(archive_rows_added),
        "top20_tickers": top20[ticker_col].astype(str).tolist(),
        "top10_tickers": top10[ticker_col].astype(str).tolist(),
        "top5_tickers": top5[ticker_col].astype(str).tolist(),
        "score_column_used": score_column_used,
        "theme_columns_present": theme_present,
        "regime_present": "regime" in df.columns,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_meta(meta: dict[str, object], meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _zero_pad_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(6)


def _resolve_code_column(df: pd.DataFrame) -> str:
    return "ticker" if "ticker" in df.columns else "code"


def _resolve_name_series(df: pd.DataFrame) -> pd.Series:
    if "name" not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df["name"].fillna("").astype(str)


def _resolve_rank_series(df: pd.DataFrame, score_column: str) -> pd.Series:
    if "rank_final" in df.columns:
        return pd.to_numeric(df["rank_final"], errors="coerce")
    if "rank" in df.columns:
        return pd.to_numeric(df["rank"], errors="coerce")
    return (
        pd.to_numeric(df[score_column], errors="coerce")
        .rank(method="first", ascending=False)
        .astype(float)
    )


def _resolve_quality_series(df: pd.DataFrame) -> pd.Series:
    if "qual_score" in df.columns:
        return pd.to_numeric(df["qual_score"], errors="coerce")
    return pd.to_numeric(df.get("quality_score"), errors="coerce")


def build_archive_frame(df: pd.DataFrame, as_of_date: str, score_column_used: str) -> pd.DataFrame:
    date_mask = pd.to_datetime(df[_resolve_date_column(df)], errors="coerce").dt.strftime("%Y-%m-%d").eq(as_of_date)
    latest = df.loc[date_mask].copy()
    if latest.empty:
        raise ValueError(f"no rows found for latest as_of_date={as_of_date}")

    code_col = _resolve_code_column(latest)
    latest["asof_date"] = as_of_date
    latest["rank"] = _resolve_rank_series(latest, score_column_used)
    latest["code"] = _zero_pad_code(latest[code_col])
    latest["name"] = _resolve_name_series(latest)
    latest["final_score"] = pd.to_numeric(latest.get("final_score"), errors="coerce")
    if latest["final_score"].isna().all():
        latest["final_score"] = pd.to_numeric(latest.get(score_column_used), errors="coerce")
    latest["confidence_score"] = pd.to_numeric(latest.get("confidence_score"), errors="coerce")
    latest["ret_score"] = pd.to_numeric(latest.get("ret_score"), errors="coerce")
    latest["prob_score"] = pd.to_numeric(latest.get("prob_score"), errors="coerce")
    latest["tech_score"] = pd.to_numeric(latest.get("tech_score"), errors="coerce")
    latest["quality_score"] = _resolve_quality_series(latest)
    latest["safety_score"] = pd.to_numeric(latest.get("safety_score"), errors="coerce")
    latest["risk_penalty"] = pd.to_numeric(latest.get("risk_penalty"), errors="coerce")
    latest["dominant_theme"] = latest.get("dominant_theme", "").fillna("").astype(str)
    latest["theme_score"] = pd.to_numeric(latest.get("theme_score"), errors="coerce")
    latest["explain_text"] = latest.get("explain_text", "").fillna("").astype(str)

    archive = latest[ARCHIVE_COLUMNS].copy()
    archive["rank"] = pd.to_numeric(archive["rank"], errors="coerce")
    archive = archive.dropna(subset=["rank", "code"])
    archive["rank"] = archive["rank"].round().astype(int)
    archive = archive.loc[archive["rank"] <= TOP_ARCHIVE_LIMIT].copy()
    archive = archive.sort_values(["rank", "code"], ascending=[True, True]).drop_duplicates(subset=ARCHIVE_KEY_COLUMNS, keep="first")
    return archive.reset_index(drop=True)


def _resolve_date_column(df: pd.DataFrame) -> str:
    for column in DATE_CANDIDATE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError("missing usable date column")


def load_archive_csv(archive_path: Path) -> pd.DataFrame:
    if not archive_path.exists():
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    df = pd.read_csv(archive_path, low_memory=False)
    for col in ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ARCHIVE_COLUMNS].copy()


def archive_rows_equal(existing: pd.DataFrame, new_rows: pd.DataFrame) -> bool:
    if len(existing) != len(new_rows):
        return False
    if existing.empty and new_rows.empty:
        return True
    left = existing.sort_values(ARCHIVE_KEY_COLUMNS).reset_index(drop=True).fillna("")
    right = new_rows.sort_values(ARCHIVE_KEY_COLUMNS).reset_index(drop=True).fillna("")
    return left.equals(right)


def upsert_archive_rows(
    archive_path: Path,
    new_rows: pd.DataFrame,
    *,
    overwrite_existing_date: bool = False,
    skip_if_exists: bool = False,
) -> tuple[str, int]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_archive_csv(archive_path)
    as_of_date = str(new_rows["asof_date"].iloc[0]) if not new_rows.empty else ""
    existing_same_date = existing.loc[existing["asof_date"].astype(str).eq(as_of_date)].copy()

    if archive_rows_equal(existing_same_date, new_rows):
        return "skipped_existing", 0

    if not existing_same_date.empty and skip_if_exists and not overwrite_existing_date:
        return "skipped_existing", 0

    remaining = existing.loc[~existing["asof_date"].astype(str).eq(as_of_date)].copy()
    if remaining.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([remaining, new_rows], ignore_index=True)
    combined = combined[ARCHIVE_COLUMNS].copy()
    combined = combined.drop_duplicates(subset=ARCHIVE_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["asof_date", "rank", "code"], ascending=[True, True, True]).reset_index(drop=True)
    combined.to_csv(archive_path, index=False, encoding="utf-8-sig")
    return "saved", int(len(new_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive latest ranking snapshot and top20 recommendation rows.")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Input ranking CSV path")
    parser.add_argument("--archive-csv", default=str(ARCHIVE_CSV), help="Archive CSV path for latest top20 rows")
    parser.add_argument("--as-of-date", default=None, help="Fallback as_of_date when no usable date column exists")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dated snapshot and archive rows for the same date")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip archive when the same date snapshot rows already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    archive_path = Path(args.archive_csv)
    if not archive_path.is_absolute():
        archive_path = BASE_DIR / archive_path

    if not input_path.exists():
        print(f"FILE_ERROR: input not found: {input_path}")
        return 1

    try:
        df = pd.read_csv(input_path, low_memory=False)
        as_of_date, date_column_used = resolve_as_of_date(df, cli_as_of_date=args.as_of_date)
        score_column_used, warnings = validate_required_columns(df)
    except Exception as exc:
        print(f"INPUT_ERROR: {exc}")
        return 1

    ticker_col = _resolve_code_column(df)
    df = df.copy()
    df[ticker_col] = _zero_pad_code(df[ticker_col])

    date_token = as_of_date.replace("-", "")
    snapshot_path = SNAPSHOT_DIR / f"{date_token}_ranking_final.csv"
    meta_path = META_DIR / f"{date_token}_ranking_meta.json"

    archive_status = "saved"
    archive_rows_added = 0

    if snapshot_path.exists() and args.skip_if_exists and not args.overwrite:
        try:
            archive_frame = build_archive_frame(df, as_of_date, score_column_used)
            archive_status, archive_rows_added = upsert_archive_rows(
                archive_path,
                archive_frame,
                overwrite_existing_date=args.overwrite,
                skip_if_exists=args.skip_if_exists,
            )
        except Exception as exc:
            print(f"WRITE_ERROR: {exc}")
            return 1
        print(f"snapshot saved path: {snapshot_path}")
        print(f"meta saved path: {meta_path}")
        print(f"archive saved path: {archive_path}")
        print(f"as_of_date: {as_of_date}")
        print(f"date_column_used: {date_column_used}")
        print(f"archive_status: {archive_status}")
        print(f"archive_rows_added: {archive_rows_added}")
        print("status: skipped_existing")
        return 0

    try:
        save_snapshot(df, snapshot_path, overwrite=args.overwrite)
        archive_frame = build_archive_frame(df, as_of_date, score_column_used)
        archive_status, archive_rows_added = upsert_archive_rows(
            archive_path,
            archive_frame,
            overwrite_existing_date=args.overwrite,
            skip_if_exists=args.skip_if_exists,
        )
        meta = build_meta(
            df,
            as_of_date,
            snapshot_path,
            score_column_used,
            archive_path,
            archive_rows_added,
        )
        save_meta(meta, meta_path)
    except FileExistsError as exc:
        print(f"WRITE_ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"WRITE_ERROR: {exc}")
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"snapshot saved path: {snapshot_path}")
    print(f"meta saved path: {meta_path}")
    print(f"archive saved path: {archive_path}")
    print(f"as_of_date: {as_of_date}")
    print(f"date_column_used: {date_column_used}")
    print(f"row_count: {len(df)}")
    print(f"archive_status: {archive_status}")
    print(f"archive_rows_added: {archive_rows_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
