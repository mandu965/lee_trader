from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
HISTORY_DIR = DATA_DIR / "history"
INVENTORY_CSV = HISTORY_DIR / "ranking_snapshot_inventory.csv"

OUT_CSV = OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.csv"
OUT_JSON = OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.json"
OUT_MD = OUTPUT_DIR / "shadow_quality_risk_guard_repeatability_report.md"

REQUIRED_COLUMNS = {
    "date",
    "code",
    "name",
    "market",
    "sector",
    "live_rank",
    "shadow_rank_quality_risk_guard",
    "shadow_quality_risk_guard_penalty",
}


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_text(value: object, default: str = "-") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _fmt_int(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(float(value)))


def _fmt_float(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_inventory() -> list[dict[str, str]]:
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"inventory file not found: {INVENTORY_CSV}")
    with INVENTORY_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [row for row in rows if str(row.get("snapshot_file") or "").strip()]
    rows.sort(key=lambda row: str(row.get("as_of_date") or ""))
    return rows


def load_shadow_snapshot(snapshot_path: Path, as_of_date: str) -> pd.DataFrame | None:
    if not snapshot_path.exists():
        return None

    df = pd.read_csv(snapshot_path, encoding="utf-8-sig")
    if df.empty or not REQUIRED_COLUMNS.issubset(df.columns):
        return None

    out = df.copy()
    out["as_of_date"] = as_of_date
    out["live_rank"] = _to_num(out["live_rank"])
    out["shadow_rank_quality_risk_guard"] = _to_num(out["shadow_rank_quality_risk_guard"])
    out["shadow_quality_risk_guard_penalty"] = _to_num(out["shadow_quality_risk_guard_penalty"])
    out["shadow_rank_delta_quality_risk_guard"] = out["live_rank"] - out["shadow_rank_quality_risk_guard"]
    out = out.loc[
        out["shadow_rank_delta_quality_risk_guard"].notna()
        & (out["shadow_rank_delta_quality_risk_guard"] > 0)
    ].copy()
    if out.empty:
        return out

    keep_cols = [
        "as_of_date",
        "date",
        "code",
        "name",
        "market",
        "sector",
        "live_rank",
        "shadow_rank_quality_risk_guard",
        "shadow_rank_delta_quality_risk_guard",
        "shadow_quality_risk_guard_penalty",
    ]
    return out[keep_cols].copy()


def build_repeatability_frame(frames: list[pd.DataFrame], usable_dates: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["as_of_date"] = combined["as_of_date"].astype(str).str.slice(0, 10)
    combined = combined.sort_values(["as_of_date", "shadow_rank_delta_quality_risk_guard"], ascending=[True, False])

    latest_date = combined["as_of_date"].max()
    latest_rows = (
        combined.loc[combined["as_of_date"] == latest_date]
        .sort_values(["shadow_rank_delta_quality_risk_guard", "live_rank"], ascending=[False, True])
        .drop_duplicates(subset=["code"], keep="first")
        .set_index("code")
    )

    rows: list[dict[str, object]] = []
    for code, group in combined.groupby("code", sort=False):
        group = group.sort_values("as_of_date")
        dates = group["as_of_date"].tolist()
        latest = latest_rows.loc[code] if code in latest_rows.index else group.iloc[-1]
        appearance_set = set(dates)
        consecutive_recent_days = 0
        for day in reversed(usable_dates):
            if day in appearance_set:
                consecutive_recent_days += 1
            elif consecutive_recent_days > 0:
                break

        rows.append(
            {
                "code": code,
                "name": _safe_text(latest.get("name")),
                "market": _safe_text(latest.get("market")),
                "sector": _safe_text(latest.get("sector")),
                "appearance_days": int(group["as_of_date"].nunique()),
                "consecutive_recent_days": int(consecutive_recent_days),
                "avg_rank_delta": round(float(group["shadow_rank_delta_quality_risk_guard"].mean()), 2),
                "max_rank_delta": int(group["shadow_rank_delta_quality_risk_guard"].max()),
                "latest_live_rank": int(latest.get("live_rank")) if pd.notna(latest.get("live_rank")) else pd.NA,
                "latest_shadow_rank": int(latest.get("shadow_rank_quality_risk_guard"))
                if pd.notna(latest.get("shadow_rank_quality_risk_guard"))
                else pd.NA,
                "latest_rank_delta": int(latest.get("shadow_rank_delta_quality_risk_guard"))
                if pd.notna(latest.get("shadow_rank_delta_quality_risk_guard"))
                else pd.NA,
                "latest_penalty": round(float(latest.get("shadow_quality_risk_guard_penalty")), 2)
                if pd.notna(latest.get("shadow_quality_risk_guard_penalty"))
                else pd.NA,
                "latest_asof_date": latest_date,
            }
        )

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        return report_df

    report_df = report_df.sort_values(
        by=["appearance_days", "consecutive_recent_days", "avg_rank_delta", "max_rank_delta", "latest_live_rank"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return report_df


def build_payload(report_df: pd.DataFrame, inventory_rows: list[dict[str, str]], usable_dates: list[str]) -> dict[str, object]:
    total_snapshot_count = len(inventory_rows)
    usable_snapshot_count = len(usable_dates)
    repeated = report_df.loc[report_df["appearance_days"] >= 2].copy() if not report_df.empty else report_df.copy()
    top_repeaters = repeated.head(10).copy() if not repeated.empty else repeated.copy()
    latest_date = usable_dates[-1] if usable_dates else (inventory_rows[-1]["as_of_date"] if inventory_rows else "")

    if usable_snapshot_count >= 2 and not repeated.empty:
        judgment = "emerging_repeatability"
    elif usable_snapshot_count >= 2:
        judgment = "no_repeaters_yet"
    else:
        judgment = "insufficient_history"

    summary = {
        "latest_asof_date": latest_date,
        "total_snapshot_count": total_snapshot_count,
        "usable_snapshot_count": usable_snapshot_count,
        "repeated_candidate_count": int(len(repeated)),
        "top_repeater_count": int(len(top_repeaters)),
        "judgment": judgment,
    }

    return {
        "summary": summary,
        "usable_dates": usable_dates,
        "top_repeaters": top_repeaters.to_dict(orient="records"),
        "all_repeaters": repeated.to_dict(orient="records"),
    }


def build_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    usable_dates = payload.get("usable_dates", []) if isinstance(payload, dict) else []
    top_repeaters = payload.get("top_repeaters", []) if isinstance(payload, dict) else []

    lines = [
        "# Shadow Quality/Risk Guard Repeatability Report",
        "",
        f"- latest_asof_date: {summary.get('latest_asof_date', '-')}",
        f"- total_snapshot_count: {summary.get('total_snapshot_count', 0)}",
        f"- usable_snapshot_count: {summary.get('usable_snapshot_count', 0)}",
        f"- repeated_candidate_count: {summary.get('repeated_candidate_count', 0)}",
        f"- judgment: {summary.get('judgment', '-')}",
        "",
        "## Usable Snapshot Dates",
        "",
        f"- {', '.join(usable_dates) if usable_dates else 'none'}",
        "",
    ]

    if not top_repeaters:
        lines.extend(
            [
                "## Interpretation",
                "",
                "- Repeated shadow improvers cannot be judged yet because usable historical snapshots are insufficient.",
                "- Historical ranking snapshots do not yet contain the `shadow_quality_risk_guard_*` columns needed for repeatability tracking.",
                "- Continue archiving daily rankings with the new shadow columns and revisit this report after at least 2 to 5 usable dates accumulate.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Top Repeaters",
            "",
            "| code | name | appearance_days | recent_streak | avg_delta | max_delta | latest_live | latest_shadow | latest_penalty |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_repeaters:
        lines.append(
            f"| {_safe_text(row.get('code'))}"
            f" | {_safe_text(row.get('name'))}"
            f" | {_fmt_int(row.get('appearance_days'))}"
            f" | {_fmt_int(row.get('consecutive_recent_days'))}"
            f" | {_fmt_float(row.get('avg_rank_delta'))}"
            f" | {_fmt_int(row.get('max_rank_delta'))}"
            f" | {_fmt_int(row.get('latest_live_rank'))}"
            f" | {_fmt_int(row.get('latest_shadow_rank'))}"
            f" | {_fmt_float(row.get('latest_penalty'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report tracks whether the same names keep improving under the quality/risk guard shadow across multiple archived ranking dates.",
            "- Repeated appearance and recent streak matter more than a single large one-day rank delta.",
            "- Use this report as promotion evidence only after repeated candidates continue to appear across several usable dates.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    inventory_rows = load_inventory()

    usable_frames: list[pd.DataFrame] = []
    usable_dates: list[str] = []
    for row in inventory_rows:
        as_of_date = str(row.get("as_of_date") or "").strip()
        snapshot_rel = str(row.get("snapshot_file") or "").strip()
        if not as_of_date or not snapshot_rel:
            continue
        snapshot_path = ROOT / snapshot_rel
        frame = load_shadow_snapshot(snapshot_path, as_of_date)
        if frame is None:
            continue
        usable_dates.append(as_of_date)
        usable_frames.append(frame)

    report_df = build_repeatability_frame(usable_frames, usable_dates)
    payload = build_payload(report_df, inventory_rows, usable_dates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")

    print(f"shadow_repeatability_report_csv: {OUT_CSV}")
    print(f"shadow_repeatability_report_json: {OUT_JSON}")
    print(f"shadow_repeatability_report_md: {OUT_MD}")
    print(f"usable_snapshot_count: {payload['summary']['usable_snapshot_count']}")
    print(f"judgment: {payload['summary']['judgment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
