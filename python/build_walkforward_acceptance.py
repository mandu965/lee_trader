from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from payload_store import upsert_json_payload

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_WF_CSV = OUTPUT_DIR / "walk_forward_score_validation.csv"
INPUT_WF_MD = OUTPUT_DIR / "walk_forward_score_validation.md"
INPUT_CONFIDENCE_OPERATIONAL = DATA_DIR / "confidence_calibration_operational.csv"
INPUT_REAL_DIFF = DATA_DIR / "real_vs_model_diff.csv"

OUT_JSON = OUTPUT_DIR / "walkforward_acceptance.json"
OUT_MD = OUTPUT_DIR / "walkforward_acceptance.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build walk-forward acceptance verdict from current validation artifacts.")
    parser.add_argument("--walkforward-csv", type=Path, default=INPUT_WF_CSV)
    parser.add_argument("--walkforward-md", type=Path, default=INPUT_WF_MD)
    parser.add_argument("--confidence-operational-csv", type=Path, default=INPUT_CONFIDENCE_OPERATIONAL)
    parser.add_argument("--real-diff-csv", type=Path, default=INPUT_REAL_DIFF)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in rendered)
    return "\n".join(lines)


def parse_walkforward_markdown(path: Path) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        return pd.DataFrame()
    text = resolved.read_text(encoding="utf-8")
    rows: list[dict[str, object]] = []
    in_summary = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "## Summary":
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary or not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[0] in {"horizon_days", "------------"}:
            continue
        if len(parts) == 9:
            rows.append(
                {
                    "section": "selection_summary",
                    "horizon_days": pd.to_numeric(parts[0], errors="coerce"),
                    "selection": parts[1],
                    "run_dates": pd.to_numeric(parts[2], errors="coerce"),
                    "avg_return": pd.to_numeric(parts[3].replace("%", ""), errors="coerce") / 100.0,
                    "benchmark_return": pd.to_numeric(parts[4].replace("%", ""), errors="coerce") / 100.0,
                    "excess_return": pd.to_numeric(parts[5].replace("%", ""), errors="coerce") / 100.0,
                    "hit_rate": pd.to_numeric(parts[6], errors="coerce"),
                    "avg_mdd": pd.to_numeric(parts[7].replace("%", ""), errors="coerce") / 100.0,
                    "median_return": pd.to_numeric(parts[8].replace("%", ""), errors="coerce") / 100.0,
                }
            )
        elif len(parts) == 8:
            rows.append(
                {
                    "section": "selection_summary",
                    "horizon_days": pd.to_numeric(parts[0], errors="coerce"),
                    "selection": "top20",
                    "run_dates": pd.to_numeric(parts[2], errors="coerce"),
                    "avg_return": pd.to_numeric(parts[4].replace("%", ""), errors="coerce") / 100.0,
                    "benchmark_return": pd.NA,
                    "excess_return": pd.to_numeric(parts[5].replace("%", ""), errors="coerce") / 100.0,
                    "hit_rate": pd.to_numeric(parts[6], errors="coerce"),
                    "avg_mdd": pd.to_numeric(parts[7].replace("%", ""), errors="coerce") / 100.0,
                    "median_return": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def load_selection_summary(csv_path: Path, md_path: Path) -> pd.DataFrame:
    df = _read_csv(csv_path)
    if not df.empty:
        df = df.loc[df.get("section", pd.Series(index=df.index)).astype(str).eq("selection_summary")].copy()
    if df.empty:
        df = parse_walkforward_markdown(md_path)
    required_cols = ["horizon_days", "selection", "run_dates", "avg_return", "benchmark_return", "excess_return", "hit_rate", "avg_mdd", "median_return"]
    if df.empty:
        return pd.DataFrame(columns=required_cols)
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["horizon_days", "run_dates", "avg_return", "benchmark_return", "excess_return", "hit_rate", "avg_mdd", "median_return"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def confidence_summary(path: Path) -> dict[str, object]:
    df = _read_csv(path)
    if df.empty:
        return {"available": False, "stable_bucket_count_5d": 0, "monotonic_hit_rate_5d": False}
    work = df.loc[df.get("source_mode", pd.Series(index=df.index)).astype(str).eq("operational") & pd.to_numeric(df.get("horizon_days"), errors="coerce").eq(5)].copy()
    if work.empty:
        return {"available": False, "stable_bucket_count_5d": 0, "monotonic_hit_rate_5d": False}
    work["rows"] = pd.to_numeric(work.get("rows"), errors="coerce")
    work["hit_rate"] = pd.to_numeric(work.get("hit_rate"), errors="coerce")
    work["bucket_mid"] = pd.to_numeric(work.get("bucket_mid"), errors="coerce")
    stable = work.loc[work.get("status", "").astype(str).eq("stable") & work["rows"].fillna(0).ge(20)].sort_values("bucket_mid")
    return {
        "available": True,
        "stable_bucket_count_5d": int(len(stable)),
        "monotonic_hit_rate_5d": bool(stable["hit_rate"].fillna(-1).is_monotonic_increasing) if len(stable) >= 2 else False,
    }


def execution_summary(path: Path) -> dict[str, object]:
    df = _read_csv(path)
    if df.empty:
        return {"available": False, "fill_ratio": None, "median_abs_slippage": None}
    entry_status = df.get("actual_entry_status", pd.Series(index=df.index)).fillna("").astype(str).str.upper()
    resolved = ~entry_status.isin(["PENDING", ""])
    if resolved.sum() == 0:
        return {"available": False, "fill_ratio": None, "median_abs_slippage": None}
    filled = entry_status.isin(["FILLED", "PARTIAL", "ENTERED"])
    fill_ratio = float(filled[resolved].mean()) if resolved.sum() > 0 else None
    slippage = pd.to_numeric(df.get("entry_slippage"), errors="coerce")
    slippage = slippage if isinstance(slippage, pd.Series) else pd.Series([slippage])
    if slippage.isna().all():
        slippage = pd.to_numeric(df.get("entry_price_vs_ref_return"), errors="coerce")
        slippage = slippage if isinstance(slippage, pd.Series) else pd.Series([slippage])
    return {
        "available": True,
        "fill_ratio": fill_ratio,
        "median_abs_slippage": float(slippage.abs().median()) if slippage.notna().any() else None,
    }


def build_payload(selection: pd.DataFrame, conf: dict[str, object], execution: dict[str, object]) -> dict[str, object]:
    required_cols = ["horizon_days", "selection", "run_dates", "avg_return", "benchmark_return", "excess_return", "hit_rate", "avg_mdd", "median_return"]
    work = selection.copy()
    for col in required_cols:
        if col not in work.columns:
            work[col] = pd.NA

    top20 = work.loc[(work["horizon_days"].eq(60)) & (work["selection"].astype(str).eq("top20"))].head(1)
    top50 = work.loc[(work["horizon_days"].eq(60)) & (work["selection"].astype(str).eq("top50"))].head(1)
    universe = work.loc[(work["horizon_days"].eq(60)) & (work["selection"].astype(str).eq("universe"))].head(1)

    reasons: list[str] = []
    status = "CONDITIONAL"
    if top20.empty:
        status = "REJECTED"
        reasons.append("walkforward_summary_missing")
    else:
        top20_row = top20.iloc[0]
        top50_row = top50.iloc[0] if not top50.empty else None
        universe_row = universe.iloc[0] if not universe.empty else None
        performance_ok = float(top20_row["excess_return"]) > 0 and float(top20_row["hit_rate"]) >= 0.55
        ordering_ok = (
            top50_row is not None
            and universe_row is not None
            and pd.notna(top20_row["avg_return"])
            and pd.notna(top50_row["avg_return"])
            and pd.notna(universe_row["avg_return"])
            and float(top20_row["avg_return"]) > float(top50_row["avg_return"]) > float(universe_row["avg_return"])
        )
        risk_ok = float(top20_row["avg_mdd"]) >= -0.25
        confidence_ok = bool(conf.get("monotonic_hit_rate_5d")) and int(conf.get("stable_bucket_count_5d") or 0) >= 2
        execution_ok = not execution.get("available") or (
            (execution.get("fill_ratio") is not None and float(execution.get("fill_ratio")) >= 0.6)
            and (execution.get("median_abs_slippage") is None or float(execution.get("median_abs_slippage")) <= 0.03)
        )

        reasons.append("top20_excess_return_positive" if performance_ok else "top20_performance_not_proven")
        if top50_row is None or universe_row is None:
            reasons.append("ordering_reference_missing")
        else:
            reasons.append("top20_top50_universe_ordering_ok" if ordering_ok else "ordering_not_stable")
        reasons.append("drawdown_within_limit" if risk_ok else "drawdown_too_deep")
        reasons.append("confidence_monotonicity_confirmed" if confidence_ok else "confidence_monotonicity_missing")
        reasons.append("execution_evidence_ok_or_unavailable" if execution_ok else "execution_evidence_too_weak")

        if performance_ok and ordering_ok and risk_ok and confidence_ok and execution_ok:
            status = "ACCEPTED"
        elif performance_ok and risk_ok:
            status = "CONDITIONAL"
        else:
            status = "REJECTED"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "reason_codes": reasons,
        "selection_summary": {
            "top20": top20.iloc[0].to_dict() if not top20.empty else None,
            "top50": top50.iloc[0].to_dict() if not top50.empty else None,
            "universe": universe.iloc[0].to_dict() if not universe.empty else None,
        },
        "confidence_summary": conf,
        "execution_summary": execution,
    }


def build_markdown(payload: dict[str, object]) -> str:
    top20 = payload.get("selection_summary", {}).get("top20") if isinstance(payload.get("selection_summary"), dict) else None
    top50 = payload.get("selection_summary", {}).get("top50") if isinstance(payload.get("selection_summary"), dict) else None
    universe = payload.get("selection_summary", {}).get("universe") if isinstance(payload.get("selection_summary"), dict) else None
    rows = []
    for label, item in [("top20", top20), ("top50", top50), ("universe", universe)]:
        if not isinstance(item, dict):
            continue
        rows.append([label, int(pd.to_numeric(item.get("run_dates"), errors="coerce")) if pd.notna(pd.to_numeric(item.get("run_dates"), errors="coerce")) else "NA", _fmt_pct(item.get("avg_return")), _fmt_pct(item.get("excess_return")), _fmt_num(item.get("hit_rate"), 4), _fmt_pct(item.get("avg_mdd"))])
    reason_rows = [[code] for code in payload.get("reason_codes", [])]
    return "\n".join(
        [
            "# Walk-Forward Acceptance",
            "",
            f"- generated_at: {payload.get('generated_at')}",
            f"- status: {payload.get('status')}",
            "",
            "## 60d Selection Summary",
            "",
            _markdown_table(rows or [["NA", "NA", "NA", "NA", "NA", "NA"]], ["selection", "run_dates", "avg_return", "excess_return", "hit_rate", "avg_mdd"]),
            "",
            "## Reason Codes",
            "",
            _markdown_table(reason_rows or [["(none)"]], ["reason_code"]),
            "",
            f"- confidence_monotonic_hit_rate_5d: {payload.get('confidence_summary', {}).get('monotonic_hit_rate_5d')}",
            f"- confidence_stable_bucket_count_5d: {payload.get('confidence_summary', {}).get('stable_bucket_count_5d')}",
            f"- execution_fill_ratio: {_fmt_pct(payload.get('execution_summary', {}).get('fill_ratio'))}",
            f"- execution_median_abs_slippage: {_fmt_pct(payload.get('execution_summary', {}).get('median_abs_slippage'))}",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    selection = load_selection_summary(args.walkforward_csv, args.walkforward_md)
    conf = confidence_summary(args.confidence_operational_csv)
    execution = execution_summary(args.real_diff_csv)
    payload = build_payload(selection, conf, execution)

    out_json = _resolve(args.out_json)
    out_md = _resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(payload), encoding="utf-8")
    upsert_json_payload(
        "walkforward_acceptance",
        payload,
        generated_at=payload.get("generated_at"),
        source_path=args.out_json,
    )

    print(f"status: {payload['status']}")
    print(f"out_json: {out_json}")
    print(f"out_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
