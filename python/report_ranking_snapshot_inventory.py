from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = BASE_DIR / "data" / "history" / "ranking"
PRICE_CSV = BASE_DIR / "data" / "prices_daily_adjusted.csv"
OUTPUT_CSV = BASE_DIR / "data" / "history" / "ranking_snapshot_inventory.csv"
OUTPUT_MD = BASE_DIR / "data" / "history" / "ranking_snapshot_inventory.md"
FILE_DATE_PATTERN = re.compile(r"(?P<yyyymmdd>\d{8})_ranking_final\.csv$", re.IGNORECASE)


def _extract_as_of_date(path: Path) -> pd.Timestamp | None:
    match = FILE_DATE_PATTERN.search(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group("yyyymmdd"), format="%Y%m%d", errors="coerce")


def _load_price_dates() -> pd.Series | None:
    if not PRICE_CSV.exists():
        return None
    try:
        prices = pd.read_csv(PRICE_CSV, usecols=["date"], low_memory=False)
    except Exception:
        return None
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    return dates


def _is_matured(as_of_date: pd.Timestamp, horizon: int, price_dates: pd.Series | None, today: pd.Timestamp) -> bool:
    if pd.isna(as_of_date):
        return False
    if price_dates is not None and not price_dates.empty:
        future_dates = price_dates[price_dates > as_of_date]
        return int(len(future_dates)) >= int(horizon)
    return (today - as_of_date).days >= int(horizon)


def main() -> int:
    if not SNAPSHOT_DIR.exists():
        print(f"FILE_ERROR: snapshot directory not found: {SNAPSHOT_DIR}")
        return 1

    snapshot_files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    price_dates = _load_price_dates()
    today = pd.Timestamp.today().normalize()
    rows: list[dict[str, object]] = []

    for path in snapshot_files:
        as_of_date = _extract_as_of_date(path)
        if pd.isna(as_of_date):
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            row_count = int(len(df))
            has_top20 = row_count >= 20
        except Exception:
            row_count = 0
            has_top20 = False

        rows.append(
            {
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "snapshot_file": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
                "row_count": row_count,
                "has_top20": bool(has_top20),
                "matured_20d": _is_matured(as_of_date, 20, price_dates, today),
                "matured_60d": _is_matured(as_of_date, 60, price_dates, today),
                "matured_90d": _is_matured(as_of_date, 90, price_dates, today),
            }
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "as_of_date",
                "snapshot_file",
                "row_count",
                "has_top20",
                "matured_20d",
                "matured_60d",
                "matured_90d",
            ]
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total_count = int(len(inventory))
    matured_20d_count = int(pd.to_numeric(inventory.get("matured_20d"), errors="coerce").fillna(False).astype(bool).sum()) if not inventory.empty else 0
    matured_60d_count = int(pd.to_numeric(inventory.get("matured_60d"), errors="coerce").fillna(False).astype(bool).sum()) if not inventory.empty else 0
    matured_90d_count = int(pd.to_numeric(inventory.get("matured_90d"), errors="coerce").fillna(False).astype(bool).sum()) if not inventory.empty else 0
    oldest_date = inventory["as_of_date"].min() if total_count else "NA"
    latest_date = inventory["as_of_date"].max() if total_count else "NA"
    if matured_60d_count > 0:
        readiness_60d = "READY"
        readiness_note = "60d confidence calibration rerun is now possible."
    else:
        readiness_60d = "WAIT"
        readiness_note = "Keep accumulating dated ranking snapshots until at least one 60d-matured snapshot exists."

    lines = [
        "# Ranking Snapshot Inventory",
        "",
        f"- total snapshot count: {total_count}",
        f"- matured snapshot count 20d: {matured_20d_count}",
        f"- matured snapshot count 60d: {matured_60d_count}",
        f"- matured snapshot count 90d: {matured_90d_count}",
        f"- oldest snapshot date: {oldest_date}",
        f"- latest snapshot date: {latest_date}",
        f"- confidence calibration readiness 60d: {readiness_60d}",
        f"- note: {readiness_note}",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"total snapshots: {total_count}")
    print(f"matured_20d count: {matured_20d_count}")
    print(f"matured_60d count: {matured_60d_count}")
    print(f"matured_90d count: {matured_90d_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
