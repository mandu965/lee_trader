from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.append(str(ROOT / "python"))

from ranking_builder import detect_market_regime  # noqa: E402


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

INPUT_MARKET = DATA_DIR / "market_status.csv"
INPUT_FEATURES = DATA_DIR / "features.csv"

OUT_DAILY_CSV = DATA_DIR / "market_regime_validation_daily.csv"
OUT_MD = OUTPUT_DIR / "market_status_validation_report.md"
OUT_JSON = OUTPUT_DIR / "market_status_validation_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate market_status inputs and summarize current regime logic.")
    parser.add_argument("--market-csv", type=Path, default=INPUT_MARKET)
    parser.add_argument("--features-csv", type=Path, default=INPUT_FEATURES)
    parser.add_argument("--window-days", type=int, default=126, help="Recent trading days to include in the report")
    parser.add_argument("--out-daily-csv", type=Path, default=OUT_DAILY_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt_num(value: object, digits: int = 2) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_pct(value: object, digits: int = 1) -> str:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _bool_text(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return "true" if bool(value) else "false"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in text_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_line(row) for row in text_rows)
    return "\n".join(lines)


def load_inputs(market_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(_resolve(market_path), low_memory=False)
    features = pd.read_csv(_resolve(features_path), usecols=["date", "code", "close_over_ma20"], low_memory=False)
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["code"] = features["code"].astype(str).str.zfill(6)
    market = market.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    features = features.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
    return market, features


def compute_daily_validation(market: pd.DataFrame, features: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()

    recent_market = market.tail(window_days).copy()
    rows: list[dict[str, Any]] = []

    for _, market_row in recent_market.iterrows():
        day = pd.Timestamp(market_row["date"]).normalize()
        day_features = features.loc[features["date"].eq(day)].copy()
        history = market.loc[market["date"].le(day), ["date", "kospi_close"]].copy()
        history["kospi_close"] = pd.to_numeric(history["kospi_close"], errors="coerce")
        history = history.dropna(subset=["date", "kospi_close"]).sort_values("date")
        history["kospi_ma60"] = history["kospi_close"].rolling(60, min_periods=20).mean()
        history["recent_20d_return"] = history["kospi_close"] / history["kospi_close"].shift(20) - 1.0

        last = history.iloc[-1] if not history.empty else pd.Series(dtype=object)
        close = pd.to_numeric(market_row.get("kospi_close"), errors="coerce")
        ma20 = pd.to_numeric(market_row.get("kospi_ma20"), errors="coerce")
        ma60 = pd.to_numeric(last.get("kospi_ma60"), errors="coerce")
        recent_20d_return = pd.to_numeric(last.get("recent_20d_return"), errors="coerce")
        volatility_5d = pd.to_numeric(market_row.get("volatility_5d"), errors="coerce")
        foreign_net_5d = pd.to_numeric(market_row.get("foreign_net_5d"), errors="coerce")
        breadth_20d = pd.to_numeric(day_features.get("close_over_ma20"), errors="coerce").gt(0).mean() if not day_features.empty else pd.NA

        close_gt_ma20 = close > ma20 if pd.notna(close) and pd.notna(ma20) else pd.NA
        ma20_gt_ma60 = ma20 > ma60 if pd.notna(ma20) and pd.notna(ma60) else pd.NA
        recent_gt_3 = recent_20d_return > 0.03 if pd.notna(recent_20d_return) else pd.NA
        breadth_gt_55 = breadth_20d > 0.55 if pd.notna(breadth_20d) else pd.NA
        vol_risk_flag = volatility_5d > 0.025 if pd.notna(volatility_5d) else pd.NA
        vol_ok = (not bool(vol_risk_flag)) if pd.notna(vol_risk_flag) else pd.NA

        condition_map = {
            "close_gt_ma20": close_gt_ma20,
            "ma20_gt_ma60": ma20_gt_ma60,
            "recent_20d_return_gt_0.03": recent_gt_3,
            "breadth_20d_gt_0.55": breadth_gt_55,
            "volatility_risk_flag_false": vol_ok,
        }
        true_conditions = [name for name, flag in condition_map.items() if pd.notna(flag) and bool(flag)]
        regime, reason = detect_market_regime(day_features, market_row.to_dict(), history)

        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "kospi_close": close,
                "kospi_ma20": ma20,
                "kospi_ma60": ma60,
                "recent_20d_return": recent_20d_return,
                "breadth_20d": breadth_20d,
                "volatility_5d": volatility_5d,
                "foreign_net_5d": foreign_net_5d,
                "close_gt_ma20": close_gt_ma20,
                "ma20_gt_ma60": ma20_gt_ma60,
                "recent_20d_return_gt_0.03": recent_gt_3,
                "breadth_20d_gt_0.55": breadth_gt_55,
                "volatility_risk_flag": vol_risk_flag,
                "volatility_risk_flag_false": vol_ok,
                "true_count": len(true_conditions),
                "met_conditions": ",".join(true_conditions) if true_conditions else "none",
                "regime": regime,
                "regime_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def build_summary_payload(df: pd.DataFrame) -> dict[str, Any]:
    regime_ratio = df["regime"].value_counts(normalize=True).to_dict()
    latest = df.iloc[-1].to_dict() if not df.empty else {}
    bool_cols = [
        "close_gt_ma20",
        "ma20_gt_ma60",
        "recent_20d_return_gt_0.03",
        "breadth_20d_gt_0.55",
        "volatility_risk_flag_false",
    ]
    condition_summary = {}
    for col in bool_cols:
        valid = df[col].dropna()
        condition_summary[col] = {
            "true_ratio": float(valid.astype(bool).mean()) if len(valid) else None,
            "false_ratio": float((~valid.astype(bool)).mean()) if len(valid) else None,
            "missing_ratio": float(df[col].isna().mean()),
        }

    latest_reasons: list[str] = []
    if latest:
        if latest.get("regime") == "neutral":
            latest_reasons.append("최신 일자는 5개 조건 중 2~3개만 충족해 neutral로 남았습니다.")
        if latest.get("close_gt_ma20") is False:
            latest_reasons.append("코스피 종가가 20일선 아래라서 bull 전환 조건이 하나 빠졌습니다.")
        if latest.get("breadth_20d_gt_0.55") is False:
            latest_reasons.append("종목 breadth가 0.55를 넘지 못해 시장 확산 신호가 약합니다.")
        if latest.get("volatility_risk_flag") is True:
            latest_reasons.append("5일 변동성 위험 플래그가 켜져 bullish 판정을 제한했습니다.")

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_date": latest.get("date"),
        "window_rows": int(len(df)),
        "regime_ratio": regime_ratio,
        "condition_summary": condition_summary,
        "latest": latest,
        "latest_diagnosis": latest_reasons,
    }


def build_markdown_report(payload: dict[str, Any], df: pd.DataFrame) -> str:
    latest = payload.get("latest") or {}
    regime_rows = []
    regime_counts = df["regime"].value_counts()
    for regime in ["bull", "neutral", "defensive"]:
        count = int(regime_counts.get(regime, 0))
        regime_rows.append([regime, count, _fmt_pct(count / max(len(df), 1))])

    condition_rows = []
    for col, info in (payload.get("condition_summary") or {}).items():
        condition_rows.append(
            [
                col,
                _fmt_pct(info.get("true_ratio")),
                _fmt_pct(info.get("false_ratio")),
                _fmt_pct(info.get("missing_ratio")),
            ]
        )

    recent_rows = []
    for _, row in df.tail(15).iterrows():
        recent_rows.append(
            [
                row["date"],
                row["regime"],
                int(row["true_count"]),
                row["met_conditions"],
                _fmt_num(row["breadth_20d"], 3),
                _fmt_pct(row["recent_20d_return"], 1),
                _bool_text(row["close_gt_ma20"]),
                _bool_text(row["ma20_gt_ma60"]),
                _bool_text(row["volatility_risk_flag"]),
            ]
        )

    lines: list[str] = []
    lines.append("# Market Status Validation Report")
    lines.append("")
    lines.append(f"- generated_at: {payload.get('generated_at')}")
    lines.append(f"- latest_date: {payload.get('latest_date')}")
    lines.append(f"- window_rows: {payload.get('window_rows')}")
    lines.append("")
    lines.append("## 핵심 결론")
    if latest:
        lines.append(
            f"- 최신 상태는 `{latest.get('regime')}` 입니다. true_count={int(latest.get('true_count') or 0)} / 5,"
            f" breadth_20d={_fmt_num(latest.get('breadth_20d'), 3)}, recent_20d_return={_fmt_pct(latest.get('recent_20d_return'), 1)},"
            f" volatility_5d={_fmt_num(latest.get('volatility_5d'), 4)}"
        )
    for item in payload.get("latest_diagnosis") or []:
        lines.append(f"- {item}")
    if not payload.get("latest_diagnosis"):
        lines.append("- 최신 판정은 입력값 범위상 특별한 이상 없이 계산되었습니다.")
    lines.append("")
    lines.append("## 레짐 분포")
    lines.append(_markdown_table(regime_rows, ["regime", "days", "ratio"]))
    lines.append("")
    lines.append("## 조건 충족률")
    lines.append(_markdown_table(condition_rows, ["condition", "true_ratio", "false_ratio", "missing_ratio"]))
    lines.append("")
    lines.append("## 최근 15거래일 상세")
    lines.append(
        _markdown_table(
            recent_rows,
            [
                "date",
                "regime",
                "true_count",
                "met_conditions",
                "breadth_20d",
                "recent_20d_return",
                "close_gt_ma20",
                "ma20_gt_ma60",
                "volatility_risk_flag",
            ],
        )
    )
    lines.append("")
    lines.append("## 운영 해석")
    lines.append("- 이 리포트는 현재 `market_status.csv`와 `features.csv`만 검증하며, 기존 레짐 로직은 변경하지 않습니다.")
    lines.append("- `neutral` 비중이 높다면 입력 이상일 수도 있지만, threshold가 보수적이라서 그럴 수도 있습니다.")
    lines.append("- 최신 레짐이 계속 `neutral`이면 buy gate 변경 전에 threshold 완화 실험과 원천 지수 검증을 분리해서 보시는 편이 안전합니다.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    market, features = load_inputs(args.market_csv, args.features_csv)
    daily = compute_daily_validation(market, features, args.window_days)
    payload = build_summary_payload(daily)

    out_daily = _resolve(args.out_daily_csv)
    out_md = _resolve(args.out_md)
    out_json = _resolve(args.out_json)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
    out_md.write_text(build_markdown_report(payload, daily), encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_daily_csv: {out_daily}")
    print(f"out_md: {out_md}")
    print(f"out_json: {out_json}")


if __name__ == "__main__":
    main()
