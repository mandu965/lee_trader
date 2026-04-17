from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = ROOT / "data"
DEFAULT_OUTPUT_JSON = ROOT / "outputs" / "ranking_output_compare_report.json"

COMPARE_FILES = {
    "ranking": "ranking_final.csv",
    "features": "features.csv",
    "labels": "labels.csv",
    "universe": "universe.csv",
    "prices_adjusted": "prices_daily_adjusted.csv",
}


@dataclass
class FileSnapshot:
    name: str
    filename: str
    local_exists: bool
    remote_exists: bool
    local_rows: int | None
    remote_rows: int | None
    row_diff: int | None
    local_min_date: str | None
    remote_min_date: str | None
    local_latest_date: str | None
    remote_latest_date: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local ranking-related CSV outputs against downloaded GitHub Actions artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded artifact CSV files.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local data directory. Defaults to {DEFAULT_LOCAL_DIR}.",
    )
    parser.add_argument(
        "--code",
        default="096530",
        help="Stock code to inspect in ranking/features/labels/prices.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N ranking rows to compare. Defaults to 20.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Optional JSON report path. Defaults to {DEFAULT_OUTPUT_JSON}.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"code": str}, low_memory=False)


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def min_date(df: pd.DataFrame) -> str | None:
    if "date" not in df.columns or df.empty:
        return None
    values = pd.to_datetime(df["date"], errors="coerce").dropna()
    return values.min().strftime("%Y-%m-%d") if not values.empty else None


def latest_date(df: pd.DataFrame) -> str | None:
    if "date" not in df.columns or df.empty:
        return None
    values = pd.to_datetime(df["date"], errors="coerce").dropna()
    return values.max().strftime("%Y-%m-%d") if not values.empty else None


def file_snapshot(name: str, filename: str, local_dir: Path, artifact_dir: Path) -> FileSnapshot:
    local_path = local_dir / filename
    remote_path = artifact_dir / filename
    local_df = normalize_date_column(load_csv(local_path))
    remote_df = normalize_date_column(load_csv(remote_path))

    local_exists = local_path.exists()
    remote_exists = remote_path.exists()
    local_rows = len(local_df) if local_exists else None
    remote_rows = len(remote_df) if remote_exists else None
    row_diff = (remote_rows - local_rows) if local_rows is not None and remote_rows is not None else None

    return FileSnapshot(
        name=name,
        filename=filename,
        local_exists=local_exists,
        remote_exists=remote_exists,
        local_rows=local_rows,
        remote_rows=remote_rows,
        row_diff=row_diff,
        local_min_date=min_date(local_df),
        remote_min_date=min_date(remote_df),
        local_latest_date=latest_date(local_df),
        remote_latest_date=latest_date(remote_df),
    )


def resolve_latest_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df.iloc[0:0].copy()
    out = normalize_date_column(df)
    latest = out["date"].dropna().max()
    if not latest:
        return out.iloc[0:0].copy()
    return out.loc[out["date"] == latest].copy()


def build_topn_frame(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    latest = resolve_latest_slice(df)
    if latest.empty:
        return latest
    if "rank_final" in latest.columns:
        latest["rank_final_num"] = pd.to_numeric(latest["rank_final"], errors="coerce")
        latest = latest.sort_values(["rank_final_num", "code"], na_position="last")
    elif "final_score" in latest.columns:
        latest["final_score_num"] = pd.to_numeric(latest["final_score"], errors="coerce")
        latest = latest.sort_values(["final_score_num", "code"], ascending=[False, True], na_position="last")
    return latest.head(top_n).copy()


def compare_topn(local_ranking: pd.DataFrame, remote_ranking: pd.DataFrame, top_n: int) -> dict[str, object]:
    local_top = build_topn_frame(local_ranking, top_n)
    remote_top = build_topn_frame(remote_ranking, top_n)
    local_codes = [str(code).zfill(6) for code in local_top.get("code", pd.Series(dtype=str)).tolist()]
    remote_codes = [str(code).zfill(6) for code in remote_top.get("code", pd.Series(dtype=str)).tolist()]
    local_set = set(local_codes)
    remote_set = set(remote_codes)

    local_only = sorted(local_set - remote_set)
    remote_only = sorted(remote_set - local_set)
    common = sorted(local_set & remote_set)

    def _rank_map(df: pd.DataFrame) -> dict[str, int | None]:
        if df.empty:
            return {}
        work = df.copy()
        if "rank_final" in work.columns:
            work["rank_final_num"] = pd.to_numeric(work["rank_final"], errors="coerce")
        else:
            work["rank_final_num"] = pd.NA
        return {
            str(row["code"]).zfill(6): (int(row["rank_final_num"]) if pd.notna(row["rank_final_num"]) else None)
            for _, row in work.iterrows()
            if str(row.get("code") or "").strip()
        }

    local_rank_map = _rank_map(local_top)
    remote_rank_map = _rank_map(remote_top)
    common_rank_diff = []
    for code in common:
        local_rank = local_rank_map.get(code)
        remote_rank = remote_rank_map.get(code)
        rank_diff = None
        if local_rank is not None and remote_rank is not None:
            rank_diff = remote_rank - local_rank
        common_rank_diff.append(
            {
                "code": code,
                "local_rank": local_rank,
                "remote_rank": remote_rank,
                "rank_diff": rank_diff,
            }
        )

    return {
        "local_latest_date": latest_date(local_top),
        "remote_latest_date": latest_date(remote_top),
        "local_codes": local_codes,
        "remote_codes": remote_codes,
        "local_only": local_only,
        "remote_only": remote_only,
        "common_rank_diff": common_rank_diff,
    }


def compare_code_detail(code: str, local_dir: Path, artifact_dir: Path) -> dict[str, object]:
    code = str(code).zfill(6)
    result: dict[str, object] = {"code": code}
    for name, filename in COMPARE_FILES.items():
        local_df = normalize_date_column(load_csv(local_dir / filename))
        remote_df = normalize_date_column(load_csv(artifact_dir / filename))
        if "code" not in local_df.columns and "code" not in remote_df.columns:
            result[name] = None
            continue

        def _slice(df: pd.DataFrame) -> list[dict[str, object]]:
            if df.empty or "code" not in df.columns:
                return []
            out = df.copy()
            out["code"] = out["code"].astype(str).str.zfill(6)
            out = out.loc[out["code"] == code].copy()
            if out.empty:
                return []
            if "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                out = out.sort_values("date", ascending=False)
            return out.head(3).to_dict(orient="records")

        result[name] = {
            "local_rows": _slice(local_df),
            "remote_rows": _slice(remote_df),
        }
    return result


def print_snapshot_table(snapshots: list[FileSnapshot]) -> None:
    frame = pd.DataFrame([asdict(item) for item in snapshots])
    if frame.empty:
      return
    print("\n[File Snapshots]")
    print(frame.to_string(index=False))


def print_topn_summary(topn: dict[str, object], top_n: int) -> None:
    print(f"\n[Top{top_n} Code Diff]")
    print(f"local latest date : {topn.get('local_latest_date')}")
    print(f"remote latest date: {topn.get('remote_latest_date')}")
    print(f"local only : {', '.join(topn.get('local_only', [])) or '-'}")
    print(f"remote only: {', '.join(topn.get('remote_only', [])) or '-'}")
    common_diff = pd.DataFrame(topn.get("common_rank_diff", []))
    if not common_diff.empty:
        print("\n[Common Rank Diff]")
        print(common_diff.to_string(index=False))


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    local_dir = args.local_dir if args.local_dir.is_absolute() else ROOT / args.local_dir

    snapshots = [
        file_snapshot(name, filename, local_dir, artifact_dir)
        for name, filename in COMPARE_FILES.items()
    ]
    print_snapshot_table(snapshots)

    local_ranking = load_csv(local_dir / COMPARE_FILES["ranking"])
    remote_ranking = load_csv(artifact_dir / COMPARE_FILES["ranking"])
    topn = compare_topn(local_ranking, remote_ranking, args.top_n)
    print_topn_summary(topn, args.top_n)

    code_detail = compare_code_detail(args.code, local_dir, artifact_dir)
    print(f"\n[Code Detail: {args.code}]")
    print(json.dumps(code_detail, ensure_ascii=False, indent=2, default=str))

    report = {
        "snapshots": [asdict(item) for item in snapshots],
        "topn": topn,
        "code_detail": code_detail,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
