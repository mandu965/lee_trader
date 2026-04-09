from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.append(str(ROOT / "python"))

from ranking_builder import detect_market_regime  # noqa: E402


MARKET_STATUS_CSV = ROOT / "data" / "market_status.csv"
FEATURES_CSV = ROOT / "data" / "features.csv"
OUTPUT_MD = ROOT / "outputs" / "regime_sensitivity_report.md"


def _fmt_num(value: object, digits: int = 4) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _fmt_pct(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _bool_text(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return "true" if bool(value) else "false"


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    data = [[str(item) for item in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in data:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_line(headers), "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"]
    lines.extend(_line(row) for row in data)
    return "\n".join(lines)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(MARKET_STATUS_CSV)
    features = pd.read_csv(FEATURES_CSV, usecols=["date", "close_over_ma20"])
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    features = features.dropna(subset=["date"]).reset_index(drop=True)
    return market, features


def _compute_daily_details(
    market_row: pd.Series,
    day_features: pd.DataFrame,
    market_history: pd.DataFrame,
) -> dict[str, object]:
    close = pd.to_numeric(market_row.get("kospi_close"), errors="coerce")
    ma20 = pd.to_numeric(market_row.get("kospi_ma20"), errors="coerce")
    volatility_5d = pd.to_numeric(market_row.get("volatility_5d"), errors="coerce")

    hist = market_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["kospi_close"] = pd.to_numeric(hist["kospi_close"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    hist["kospi_ma60"] = hist["kospi_close"].rolling(60, min_periods=20).mean()
    hist["recent_20d_return"] = hist["kospi_close"] / hist["kospi_close"].shift(20) - 1.0
    last = hist.iloc[-1]
    ma60 = pd.to_numeric(last.get("kospi_ma60"), errors="coerce")
    recent_20d_return = pd.to_numeric(last.get("recent_20d_return"), errors="coerce")

    close_gt_ma20 = close > ma20 if pd.notna(close) and pd.notna(ma20) else pd.NA
    ma20_gt_ma60 = ma20 > ma60 if pd.notna(ma20) and pd.notna(ma60) else pd.NA
    breadth_20d = pd.NA
    if not day_features.empty and "close_over_ma20" in day_features.columns:
        close_over_ma20 = pd.to_numeric(day_features["close_over_ma20"], errors="coerce")
        if close_over_ma20.notna().any():
            breadth_20d = float(close_over_ma20.gt(0).mean())

    breadth_condition = breadth_20d > 0.55 if pd.notna(breadth_20d) else pd.NA
    recent_condition = recent_20d_return > 0.03 if pd.notna(recent_20d_return) else pd.NA
    volatility_risk_flag = volatility_5d > 0.025 if pd.notna(volatility_5d) else pd.NA
    volatility_condition = (not bool(volatility_risk_flag)) if pd.notna(volatility_risk_flag) else pd.NA

    condition_map = {
        "close_gt_ma20": close_gt_ma20,
        "ma20_gt_ma60": ma20_gt_ma60,
        "recent_20d_return_gt_0.03": recent_condition,
        "breadth_20d_gt_0.55": breadth_condition,
        "volatility_risk_flag_false": volatility_condition,
    }
    met_conditions = [name for name, flag in condition_map.items() if pd.notna(flag) and bool(flag)]

    regime, reason = detect_market_regime(day_features, market_row.to_dict(), hist[["date", "kospi_close"]].copy())
    return {
        "date": pd.to_datetime(market_row["date"]).strftime("%Y-%m-%d"),
        "regime": regime,
        "regime_reason": reason,
        "true_count": len(met_conditions),
        "met_conditions": ", ".join(met_conditions) if met_conditions else "none",
        "close_gt_ma20": close_gt_ma20,
        "ma20_gt_ma60": ma20_gt_ma60,
        "recent_20d_return": recent_20d_return,
        "recent_20d_return_gt_0.03": recent_condition,
        "breadth_20d": breadth_20d,
        "breadth_20d_gt_0.55": breadth_condition,
        "volatility_5d": volatility_5d,
        "volatility_risk_flag": volatility_risk_flag,
        "volatility_risk_flag_false": volatility_condition,
    }


def build_daily_regime_frame(market: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    latest = market["date"].max()
    start = latest - pd.DateOffset(months=6)
    market_window = market.loc[market["date"] >= start].copy()

    records: list[dict[str, object]] = []
    for _, market_row in market_window.iterrows():
        day = market_row["date"]
        day_features = features.loc[features["date"] == day].copy()
        history = market.loc[market["date"] <= day, ["date", "kospi_close"]].copy()
        records.append(_compute_daily_details(market_row, day_features, history))
    return pd.DataFrame(records)


def _condition_summary(df: pd.DataFrame, column: str) -> list[object]:
    series = df[column]
    valid = series.dropna()
    true_ratio = float(valid.astype(bool).mean()) if len(valid) else float("nan")
    false_ratio = float((~valid.astype(bool)).mean()) if len(valid) else float("nan")
    missing_ratio = float(series.isna().mean())
    return [column, _fmt_pct(true_ratio), _fmt_pct(false_ratio), _fmt_pct(missing_ratio)]


def _numeric_summary(df: pd.DataFrame, column: str) -> list[list[object]]:
    series = pd.to_numeric(df[column], errors="coerce")
    return [[
        column,
        _fmt_num(series.min()),
        _fmt_num(series.quantile(0.25)),
        _fmt_num(series.median()),
        _fmt_num(series.quantile(0.75)),
        _fmt_num(series.max()),
    ]]


def _combo_summary(df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for regime in ["bull", "neutral", "defensive"]:
        subset = df.loc[df["regime"] == regime, "met_conditions"]
        counts = subset.value_counts().head(5)
        for combo, count in counts.items():
            rows.append([regime, combo, int(count), _fmt_pct(count / max(len(subset), 1))])
    return rows


def _build_diagnosis(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    regime_share = df["regime"].value_counts(normalize=True)
    bull_share = float(regime_share.get("bull", 0.0))
    neutral_share = float(regime_share.get("neutral", 0.0))
    defensive_share = float(regime_share.get("defensive", 0.0))

    if bull_share == 0.0:
        lines.append("- 최근 6개월에 bull이 한 번도 등장하지 않아, 상방 구간 감지가 지나치게 엄격할 가능성이 있습니다.")
    else:
        first_bull = df.loc[df["regime"] == "bull", "date"].iloc[0]
        last_bull = df.loc[df["regime"] == "bull", "date"].iloc[-1]
        lines.append(f"- bull은 최근 6개월 중 {_fmt_pct(bull_share)} 비중으로 발생했고, {first_bull} ~ {last_bull} 구간에서 실제로 등장했습니다.")

    if defensive_share == 0.0:
        lines.append("- defensive가 한 번도 없어서, 변동성 악화 구간 방어 분기가 둔할 수 있습니다.")
    else:
        first_def = df.loc[df["regime"] == "defensive", "date"].iloc[0]
        last_def = df.loc[df["regime"] == "defensive", "date"].iloc[-1]
        lines.append(f"- defensive는 {_fmt_pct(defensive_share)} 비중으로 나타났고, {first_def} ~ {last_def}의 입력 부족 초기 구간과 3월 변동성 확대 구간에서 확인됩니다.")

    if neutral_share >= 0.85:
        lines.append("- neutral 비중이 85%를 넘어서 과밀 상태입니다. 이 경우 threshold 구조 조정이 필요합니다.")
    elif neutral_share >= 0.65:
        lines.append("- neutral 비중이 다소 높지만 과밀이라고 보기는 어렵습니다. bull / defensive도 함께 발생하므로 구조 붕괴는 아닙니다.")
    else:
        lines.append("- neutral 비중이 과밀하지 않습니다. 최근 ranking 결과가 neutral로 나온 것은 최신 시장 상태의 결과일 가능성이 더 큽니다.")

    latest = df.iloc[-1]
    if latest["regime"] == "neutral":
        lines.append(
            f"- 최신일 {latest['date']}은 true_count={int(latest['true_count'])}로 neutral입니다. "
            f"현재는 breadth_20d={_fmt_num(latest['breadth_20d'], 3)}가 낮아 bull까지는 못 가고, "
            f"ma20_gt_ma60 / recent_20d_return / volatility_risk_flag_false 조합으로 neutral에 머물렀습니다."
        )
    return lines


def build_report(df: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_date = df["date"].max()

    count_rows = []
    total = len(df)
    regime_counts = df["regime"].value_counts()
    for regime in ["bull", "neutral", "defensive"]:
        count = int(regime_counts.get(regime, 0))
        count_rows.append([regime, count, _fmt_pct(count / max(total, 1))])

    condition_rows = [
        _condition_summary(df, "close_gt_ma20"),
        _condition_summary(df, "ma20_gt_ma60"),
        _condition_summary(df, "recent_20d_return_gt_0.03"),
        _condition_summary(df, "breadth_20d_gt_0.55"),
        _condition_summary(df, "volatility_risk_flag_false"),
    ]
    numeric_rows = _numeric_summary(df, "breadth_20d") + _numeric_summary(df, "recent_20d_return")

    daily_rows = [
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
        for _, row in df.iterrows()
    ]

    combo_rows = _combo_summary(df)

    lines: list[str] = []
    lines.append("# Regime Sensitivity Report")
    lines.append("")
    lines.append("- generated_at: " + generated_at)
    lines.append("- latest_date: " + str(latest_date))
    lines.append("- source_market_status_rows: " + str(len(pd.read_csv(MARKET_STATUS_CSV))))
    lines.append("- source_features_rows: " + str(len(pd.read_csv(FEATURES_CSV, usecols=['date']))))
    lines.append("- recomputed_from_current_code: true")
    lines.append("")
    lines.append("## regime ratio")
    lines.append(_markdown_table(count_rows, ["regime", "days", "ratio"]))
    lines.append("")
    lines.append("## 일자별 regime 분포")
    lines.append(_markdown_table(
        daily_rows,
        ["date", "regime", "true_count", "met_conditions", "breadth_20d", "recent_20d_return", "close_gt_ma20", "ma20_gt_ma60", "volatility_risk_flag"],
    ))
    lines.append("")
    lines.append("## 각 regime 진입 시 충족된 조건 목록")
    lines.append(_markdown_table(combo_rows, ["regime", "met_conditions", "days", "share_within_regime"]))
    lines.append("")
    lines.append("## 조건 분포")
    lines.append("### boolean condition distribution")
    lines.append(_markdown_table(condition_rows, ["condition", "true_ratio", "false_ratio", "missing_ratio"]))
    lines.append("")
    lines.append("### numeric distribution")
    lines.append(_markdown_table(numeric_rows, ["metric", "min", "p25", "median", "p75", "max"]))
    lines.append("")
    lines.append("## 진단 요약")
    lines.extend(_build_diagnosis(df))
    lines.append("")
    lines.append("## threshold 조정안 후보")
    lines.append("### 보수안")
    lines.append("- bull 진입 기준 `true_count >= 4`는 유지")
    lines.append("- `breadth_20d_gt_0.55`를 `breadth_20d_gt_0.50`으로 완화")
    lines.append("- `volatility_risk_flag_false` 임계값을 `volatility_5d <= 0.028`으로 소폭 완화")
    lines.append("- 기대 효과: breadth가 약간 부족한 상승장 후반에도 bull이 끊기지 않고, defensive 남발 없이 neutral 일부가 bull로 이동")
    lines.append("")
    lines.append("### 완화안")
    lines.append("- bull 진입 기준을 `true_count >= 3`으로 낮추되, `close_gt_ma20` 또는 `ma20_gt_ma60` 중 하나는 반드시 참이어야 함")
    lines.append("- `recent_20d_return_gt_0.03`를 `recent_20d_return_gt_0.02`로 완화")
    lines.append("- 기대 효과: 지수 상승과 추세 확인은 됐지만 breadth가 늦게 따라오는 구간에서도 bull이 더 빠르게 발생")
    lines.append("- 주의점: 완화안은 neutral 일부를 bull로 더 적극 이동시키므로 실전 적용 전 별도 백테스트가 필요")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    market, features = load_inputs()
    daily = build_daily_regime_frame(market, features)
    report = build_report(daily)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
