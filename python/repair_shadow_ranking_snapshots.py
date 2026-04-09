from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "data" / "history"
INVENTORY_CSV = HISTORY_DIR / "ranking_snapshot_inventory.csv"
CURRENT_RANKING_CSV = ROOT / "data" / "ranking_final.csv"

REQUIRED_COLUMNS = {
    "date",
    "code",
    "live_rank",
    "shadow_rank_quality_risk_guard",
    "shadow_quality_risk_guard_penalty",
}


def load_inventory() -> list[dict[str, str]]:
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"inventory file not found: {INVENTORY_CSV}")
    with INVENTORY_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [row for row in rows if str(row.get("snapshot_file") or "").strip()]


def read_header(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    return pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()


def has_required_columns(csv_path: Path) -> bool:
    columns = set(read_header(csv_path))
    return REQUIRED_COLUMNS.issubset(columns)


def resolve_output_snapshot(as_of_date: str) -> Path:
    compact = str(as_of_date).replace("-", "")
    return ROOT / "outputs" / "snapshots" / compact / f"ranking_final_{compact}.csv"


def fallback_candidates(as_of_date: str) -> list[Path]:
    candidates = [resolve_output_snapshot(as_of_date)]
    if CURRENT_RANKING_CSV.exists():
        candidates.append(CURRENT_RANKING_CSV)
    return candidates


def load_snapshot_rows(csv_path: Path, as_of_date: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"code": str}, low_memory=False)
    if df.empty:
        return df
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.loc[dates.eq(as_of_date)].copy()
    return df


def is_compatible_candidate(target_path: Path, candidate_path: Path, as_of_date: str) -> tuple[bool, str]:
    if not candidate_path.exists():
        return False, "missing_candidate"
    if not has_required_columns(candidate_path):
        return False, "missing_required_columns"

    target_df = load_snapshot_rows(target_path, as_of_date)
    candidate_df = load_snapshot_rows(candidate_path, as_of_date)
    if candidate_df.empty:
        return False, "candidate_date_mismatch"
    if target_df.empty:
        return True, "target_empty"

    if len(target_df) != len(candidate_df):
        return False, "row_count_mismatch"

    target_codes = target_df["code"].astype(str).sort_values().tolist()
    candidate_codes = candidate_df["code"].astype(str).sort_values().tolist()
    if target_codes != candidate_codes:
        return False, "code_set_mismatch"
    return True, "ok"


def repair_snapshot(row: dict[str, str]) -> tuple[str, str]:
    as_of_date = str(row.get("as_of_date") or "").strip()
    snapshot_rel = str(row.get("snapshot_file") or "").strip()
    if not as_of_date or not snapshot_rel:
        return "skipped", "invalid_inventory_row"

    target_path = ROOT / snapshot_rel
    if not target_path.exists():
        return "skipped", "missing_target"
    if has_required_columns(target_path):
        return "skipped", "already_has_shadow_columns"

    for candidate in fallback_candidates(as_of_date):
        ok, reason = is_compatible_candidate(target_path, candidate, as_of_date)
        if not ok:
            continue
        repaired = pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_csv(target_path, index=False, encoding="utf-8-sig")
        return "repaired", str(candidate.relative_to(ROOT)).replace("\\", "/")

    return "skipped", "no_compatible_fallback"


def main() -> int:
    repaired_count = 0
    skipped_count = 0
    for row in load_inventory():
        status, detail = repair_snapshot(row)
        as_of_date = str(row.get("as_of_date") or "").strip()
        snapshot_rel = str(row.get("snapshot_file") or "").strip()
        print(f"{status.upper()} as_of_date={as_of_date} snapshot={snapshot_rel} detail={detail}")
        if status == "repaired":
            repaired_count += 1
        else:
            skipped_count += 1

    print(f"repaired_count: {repaired_count}")
    print(f"skipped_count: {skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
