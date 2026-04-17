import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path("data")
DEFAULT_THEME_ETF_MASTER = DATA_DIR / "theme_etf_master.csv"
DEFAULT_PRICES_RAW = DATA_DIR / "prices_daily_raw.csv"
DEFAULT_REPORT_MD = DATA_DIR / "etf_price_universe_integration_report.md"
DEFAULT_DEBUG_CSV = DATA_DIR / "etf_price_universe_match_debug.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ETF price universe integration into prices_daily_raw.csv.")
    parser.add_argument("--theme-etf-master", type=Path, default=DEFAULT_THEME_ETF_MASTER)
    parser.add_argument("--prices-raw", type=Path, default=DEFAULT_PRICES_RAW)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--debug-csv", type=Path, default=DEFAULT_DEBUG_CSV)
    return parser.parse_args()


def normalize_code_value(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.zfill(6) if text.isdigit() else text


def normalize_code_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_code_value)


def load_theme_etf_master_codes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"theme_etf_master file not found: {path}")
    df = pd.read_csv(path, dtype=str)
    code_col = None
    for candidate in ["etf_code", "code", "symbol"]:
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        raise ValueError("theme_etf_master missing code column. expected one of: etf_code, code, symbol")
    out = pd.DataFrame({"etf_code": normalize_code_series(df[code_col])})
    out = out.loc[out["etf_code"] != ""].drop_duplicates(subset=["etf_code"]).reset_index(drop=True)
    out["in_theme_etf_master"] = True
    return out


def load_prices_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"prices_daily_raw file not found: {path}")
    df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    required = {"date", "code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"prices_daily_raw missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["code"] = normalize_code_series(df["code"])
    df = df.loc[(df["date"].notna()) & (df["code"] != "")].copy()
    return df


def detect_latest_date_status(prices_raw: pd.DataFrame, theme_codes: set[str]) -> dict[str, Any]:
    latest_date = str(prices_raw["date"].max()) if not prices_raw.empty else ""
    latest = prices_raw.loc[prices_raw["date"] == latest_date].copy() if latest_date else pd.DataFrame(columns=prices_raw.columns)
    latest_code_set = set(latest["code"].tolist())
    latest_etf_codes = sorted(latest_code_set & theme_codes)
    return {
        "raw_latest_date": latest_date,
        "matched_etf_row_count_on_latest_date": int(latest["code"].isin(theme_codes).sum()),
        "matched_etf_code_count_on_latest_date": int(len(latest_etf_codes)),
        "stock_row_count_on_latest_date": int((~latest["code"].isin(theme_codes)).sum()),
        "stock_code_count_on_latest_date": int(len(latest_code_set - theme_codes)),
        "latest_etf_code_set": set(latest_etf_codes),
    }


def build_verdict(stats: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    matched_count = int(stats["matched_etf_code_count"])
    latest_count = int(stats["matched_etf_code_count_on_latest_date"])
    ratio = float(stats["matched_ratio_vs_theme_etf_master"])

    if matched_count == 0 or latest_count == 0:
        if matched_count == 0:
            reasons.append("theme_etf_master ETF codes did not match any raw price code")
        if latest_count == 0:
            reasons.append("latest raw date has zero ETF codes")
        return "FAIL", reasons

    if ratio >= 0.5 and latest_count > 0:
        reasons.append("ETF codes are present in raw prices and latest-date coverage is meaningful")
        return "STRONG PASS", reasons

    if ratio < 0.3:
        reasons.append("ETF codes are present but raw coverage ratio is weak")
        return "WARN", reasons

    reasons.append("ETF codes are present in raw prices and latest-date coverage is non-zero")
    return "PASS", reasons


def compute_etf_price_universe_match(theme_df: pd.DataFrame, prices_raw: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    theme_codes = set(theme_df["etf_code"].tolist())
    raw_code_set = set(prices_raw["code"].tolist())
    matched_codes = sorted(theme_codes & raw_code_set)
    unmatched_codes = sorted(theme_codes - raw_code_set)
    latest_stats = detect_latest_date_status(prices_raw, theme_codes)

    debug = theme_df.copy()
    debug["in_prices_raw"] = debug["etf_code"].isin(raw_code_set)
    debug["matched"] = debug["in_prices_raw"]
    debug["latest_date_present"] = debug["etf_code"].isin(latest_stats["latest_etf_code_set"])

    if "asset_type" in prices_raw.columns:
        asset_map = (
            prices_raw.loc[:, ["code", "asset_type"]]
            .dropna(subset=["code"])
            .drop_duplicates(subset=["code"], keep="last")
            .rename(columns={"code": "etf_code", "asset_type": "asset_type_detected"})
        )
        debug = debug.merge(asset_map, on="etf_code", how="left")
    else:
        debug["asset_type_detected"] = pd.NA

    if "universe_source" in prices_raw.columns:
        source_map = (
            prices_raw.loc[:, ["code", "universe_source"]]
            .dropna(subset=["code"])
            .drop_duplicates(subset=["code"], keep="last")
            .rename(columns={"code": "etf_code", "universe_source": "universe_source_detected"})
        )
        debug = debug.merge(source_map, on="etf_code", how="left")
    else:
        debug["universe_source_detected"] = pd.NA

    stats = {
        "theme_etf_master_etf_count": int(len(theme_df)),
        "unique_theme_etf_code_count": int(len(theme_codes)),
        "raw_total_row_count": int(len(prices_raw)),
        "raw_unique_code_count": int(len(raw_code_set)),
        "matched_etf_row_count": int(prices_raw["code"].isin(theme_codes).sum()),
        "matched_etf_code_count": int(len(matched_codes)),
        "unmatched_theme_etf_code_count": int(len(unmatched_codes)),
        "matched_ratio_vs_theme_etf_master": float(len(matched_codes) / max(len(theme_codes), 1)),
        "non_etf_row_count": int((~prices_raw["code"].isin(theme_codes)).sum()),
        "non_etf_unique_code_count": int(len(raw_code_set - theme_codes)),
        "matched_etf_sample": matched_codes[:10],
        "unmatched_etf_sample": unmatched_codes[:10],
    }
    stats.update(latest_stats)
    verdict, reasons = build_verdict(stats)
    stats["verdict"] = verdict
    stats["verdict_reasons"] = reasons

    if "asset_type" in prices_raw.columns:
        stats["asset_type_distribution"] = prices_raw["asset_type"].fillna("unknown").value_counts().to_dict()
    else:
        stats["asset_type_distribution"] = {}
    if "universe_source" in prices_raw.columns:
        stats["universe_source_distribution"] = prices_raw["universe_source"].fillna("unknown").value_counts().to_dict()
    else:
        stats["universe_source_distribution"] = {}

    return stats, debug.loc[:, [
        "etf_code",
        "in_theme_etf_master",
        "in_prices_raw",
        "matched",
        "latest_date_present",
        "asset_type_detected",
        "universe_source_detected",
    ]]


def write_debug_csv(debug_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(path, index=False, encoding="utf-8-sig")


def write_markdown_report(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ETF Price Universe Integration Report",
        "",
        f"- verdict: {stats['verdict']}",
        f"- reason: {'; '.join(stats['verdict_reasons'])}",
        "",
        "## Core Metrics",
        "",
        f"- theme_etf_master unique ETF codes: {stats['unique_theme_etf_code_count']}",
        f"- prices_daily_raw unique codes: {stats['raw_unique_code_count']}",
        f"- matched ETF codes: {stats['matched_etf_code_count']}",
        f"- matched ETF rows: {stats['matched_etf_row_count']}",
        f"- unmatched theme ETF codes: {stats['unmatched_theme_etf_code_count']}",
        f"- matched ratio vs theme_etf_master: {stats['matched_ratio_vs_theme_etf_master']:.2%}",
        "",
        "## Latest Date Status",
        "",
        f"- raw_latest_date: {stats['raw_latest_date']}",
        f"- matched ETF rows on latest date: {stats['matched_etf_row_count_on_latest_date']}",
        f"- matched ETF codes on latest date: {stats['matched_etf_code_count_on_latest_date']}",
        f"- stock rows on latest date: {stats['stock_row_count_on_latest_date']}",
        f"- stock codes on latest date: {stats['stock_code_count_on_latest_date']}",
        "",
        "## Stock Path Preservation",
        "",
        f"- non_etf_row_count: {stats['non_etf_row_count']}",
        f"- non_etf_unique_code_count: {stats['non_etf_unique_code_count']}",
        "",
        "## Optional Column Check",
        "",
        f"- asset_type_distribution: {stats['asset_type_distribution']}",
        f"- universe_source_distribution: {stats['universe_source_distribution']}",
        "",
        "## Samples",
        "",
        f"- matched_etf_sample: {stats['matched_etf_sample']}",
        f"- unmatched_etf_sample: {stats['unmatched_etf_sample']}",
        "",
        "## Interpretation",
        "",
        "- 지금 병목은 `theme_etf_master` 부족이 아니라 `ETF 가격 universe 부재`였다.",
        "- 이 검증이 PASS 이상이면 `compute_theme_etf_daily.py`가 proxy-only 상태에서 벗어날 기반이 생긴다.",
        "- breadth-aware 집계의 다음 단계는 `build_stock_theme_daily.py` 활용이며, 이번 검증은 그 이전 prerequisite 확인이다.",
        "",
        "## Next Step",
        "",
        "- PASS/STRONG PASS면 `compute_theme_etf_daily.py`를 다시 실행해 ETF real price 기반 점수 전환 여부를 확인한다.",
        "- WARN/FAIL이면 `download_prices_kis.py --include-theme-etf-codes` 경로를 다시 실행하고 raw 최신 적재 상태를 먼저 복구한다.",
        "- 중기적으로는 ETF 전용 raw 파일 분리도 검토할 수 있다.",
        "",
        "## Run Example",
        "",
        "- `python python/check_etf_price_universe_integration.py`",
        "- `python python/check_etf_price_universe_integration.py --theme-etf-master data/theme_etf_master.csv --prices-raw data/prices_daily_raw.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(stats: dict[str, Any]) -> None:
    print(f"theme_etf_master unique ETF codes: {stats['unique_theme_etf_code_count']}")
    print(f"prices_daily_raw unique codes: {stats['raw_unique_code_count']}")
    print(f"matched ETF codes: {stats['matched_etf_code_count']}")
    print(f"matched ETF rows: {stats['matched_etf_row_count']}")
    print(f"latest date: {stats['raw_latest_date']}")
    print(f"matched ETF codes on latest date: {stats['matched_etf_code_count_on_latest_date']}")
    print(f"verdict: {stats['verdict']}")


def main() -> int:
    args = parse_args()
    theme_df = load_theme_etf_master_codes(args.theme_etf_master)
    prices_raw = load_prices_raw(args.prices_raw)
    stats, debug_df = compute_etf_price_universe_match(theme_df, prices_raw)
    write_debug_csv(debug_df, args.debug_csv)
    write_markdown_report(stats, args.report_md)
    print_console_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
