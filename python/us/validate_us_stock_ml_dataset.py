from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python.us.us_config import USDatasetValidationConfig, load_us_dataset_validation_config
from python.us.us_db import (
    fetch_active_tickers,
    fetch_daily_feature_rows,
    fetch_financial_feature_rows,
    fetch_label_rows,
    fetch_relative_strength_feature_rows,
    get_us_engine,
)


LOGGER = logging.getLogger("us_dataset")
SUPPORTED_FEATURE_TABLES = {"feature.us_stock_feature_daily"}
SUPPORTED_FINANCIAL_TABLES = {"feature.us_stock_financial_feature"}
SUPPORTED_RS_TABLES = {"feature.us_stock_relative_strength_daily"}
SUPPORTED_LABEL_TABLES = {"label.us_stock_label_daily"}


@dataclass(frozen=True)
class USDatasetValidationResult:
    feature_row_count: int
    label_row_count: int
    joined_row_count: int
    ticker_count: int
    trade_date_min: str | None
    trade_date_max: str | None
    feature_null_ratio_summary: dict[str, float]
    label_null_ratio_summary: dict[str, float]
    label_distribution: dict[str, dict[str, float]]
    duplicate_key_count: int
    financial_reported_date_null_ratio: float | None
    leakage_risk_notes: list[str]
    report_path: Path


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_dataset_validation_config()
    parser = argparse.ArgumentParser(description="Validate US stock ML dataset readiness.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag used to load active tickers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for test runs.")
    return parser.parse_args()


def _null_ratio_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if frame.empty:
        return {}
    out: dict[str, float] = {}
    for column in columns:
        if column in frame.columns:
            out[column] = round(float(frame[column].isna().mean()), 4)
    return out


def _label_distribution(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if frame.empty:
        return out
    for column in ["label_positive_20d", "label_positive_60d", "label_top20_20d", "label_top20_60d"]:
        if column in frame.columns:
            valid = frame[column].dropna()
            out[column] = {
                "ones_ratio": 0.0 if valid.empty else round(float((valid == 1).mean()), 4),
                "zeros_ratio": 0.0 if valid.empty else round(float((valid == 0).mean()), 4),
            }
    return out


def _write_report(path: Path, *, result: USDatasetValidationResult, financial_row_count: int, rs_row_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# US Stock Dataset Validation",
        "",
        f"- feature_row_count: {result.feature_row_count}",
        f"- relative_strength_row_count: {rs_row_count}",
        f"- financial_feature_row_count: {financial_row_count}",
        f"- financial_reported_date_null_ratio: {result.financial_reported_date_null_ratio}",
        f"- label_row_count: {result.label_row_count}",
        f"- joined_row_count: {result.joined_row_count}",
        f"- ticker_count: {result.ticker_count}",
        f"- trade_date_range: {result.trade_date_min} .. {result.trade_date_max}",
        f"- duplicate_key_count: {result.duplicate_key_count}",
        "",
        "## Feature Null Ratio",
    ]
    for key, value in result.feature_null_ratio_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Null Ratio"])
    for key, value in result.label_null_ratio_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Distribution"])
    for key, value in result.label_distribution.items():
        lines.append(f"- {key}: ones_ratio={value['ones_ratio']} zeros_ratio={value['zeros_ratio']}")
    lines.extend(["", "## Leakage Risk Notes"])
    for note in result.leakage_risk_notes:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_us_stock_ml_dataset(
    *,
    cfg: USDatasetValidationConfig,
    universe_tag: str,
    limit: int | None = None,
    ticker_loader=fetch_active_tickers,
    feature_fetcher=fetch_daily_feature_rows,
    financial_fetcher=fetch_financial_feature_rows,
    rs_fetcher=fetch_relative_strength_feature_rows,
    label_fetcher=fetch_label_rows,
) -> USDatasetValidationResult:
    if not cfg.enabled:
        LOGGER.info("[US_DATASET] US_DATASET_VALIDATE_ENABLED=0. Skip validator.")
        return USDatasetValidationResult(0, 0, 0, 0, None, None, {}, {}, {}, 0, None, [], cfg.report_path)
    if cfg.feature_table not in SUPPORTED_FEATURE_TABLES:
        raise ValueError(f"Unsupported feature table '{cfg.feature_table}'.")
    if cfg.financial_feature_table not in SUPPORTED_FINANCIAL_TABLES:
        raise ValueError(f"Unsupported financial feature table '{cfg.financial_feature_table}'.")
    if cfg.relative_strength_table not in SUPPORTED_RS_TABLES:
        raise ValueError(f"Unsupported relative strength table '{cfg.relative_strength_table}'.")
    if cfg.label_table not in SUPPORTED_LABEL_TABLES:
        raise ValueError(f"Unsupported label table '{cfg.label_table}'.")

    tickers = ticker_loader(universe_tag)
    if limit is not None:
        tickers = tickers[:limit]
    if not tickers:
        LOGGER.info("[US_DATASET] No active tickers found. universe=%s", universe_tag)
        return USDatasetValidationResult(0, 0, 0, 0, None, None, {}, {}, {}, 0, None, [], cfg.report_path)

    feature_rows = feature_fetcher(tickers)
    rs_rows = rs_fetcher(tickers)
    label_rows = label_fetcher(tickers)
    financial_rows = financial_fetcher(tickers)

    feature_df = pd.DataFrame(feature_rows)
    rs_df = pd.DataFrame(rs_rows)
    label_df = pd.DataFrame(label_rows)
    financial_df = pd.DataFrame(financial_rows)

    if not feature_df.empty:
        feature_df["trade_date"] = pd.to_datetime(feature_df["trade_date"], errors="coerce").dt.date
    if not rs_df.empty:
        rs_df["trade_date"] = pd.to_datetime(rs_df["trade_date"], errors="coerce").dt.date
    if not label_df.empty:
        label_df["trade_date"] = pd.to_datetime(label_df["trade_date"], errors="coerce").dt.date
    if not financial_df.empty:
        financial_df["fiscal_date"] = pd.to_datetime(financial_df["fiscal_date"], errors="coerce").dt.date
        if "reported_date" in financial_df.columns:
            financial_df["reported_date"] = pd.to_datetime(financial_df["reported_date"], errors="coerce").dt.date

    feature_duplicates = 0 if feature_df.empty else int(feature_df.duplicated(subset=["ticker", "trade_date"]).sum())
    rs_duplicates = 0 if rs_df.empty else int(rs_df.duplicated(subset=["ticker", "trade_date", "source"]).sum())
    label_duplicates = 0 if label_df.empty else int(label_df.duplicated(subset=["ticker", "trade_date", "source"]).sum())

    joined = feature_df.merge(label_df, on=["ticker", "trade_date"], how="inner", suffixes=("_feature", "_label")) if not feature_df.empty and not label_df.empty else pd.DataFrame()
    if not rs_df.empty and not joined.empty:
        joined = joined.merge(
            rs_df.drop(columns=["market"], errors="ignore"),
            on=["ticker", "trade_date"],
            how="left",
            suffixes=("", "_rs"),
        )

    feature_null_summary = _null_ratio_summary(
        feature_df,
        ["ret_5d", "ret_10d", "ret_20d", "ret_60d", "ma_20", "ma_60"],
    )
    if not rs_df.empty:
        feature_null_summary.update(
            _null_ratio_summary(rs_df, ["rs_spy_20d", "rs_spy_60d", "rs_qqq_20d", "rs_qqq_60d"])
        )
    label_null_summary = _null_ratio_summary(
        label_df,
        ["future_ret_5d", "future_ret_20d", "future_ret_60d", "label_top20_20d", "label_top20_60d"],
    )
    distribution = _label_distribution(label_df)

    trade_date_min = None if joined.empty else str(joined["trade_date"].min())
    trade_date_max = None if joined.empty else str(joined["trade_date"].max())
    financial_reported_date_null_ratio = None
    if not financial_df.empty and "reported_date" in financial_df.columns:
        financial_reported_date_null_ratio = round(float(financial_df["reported_date"].isna().mean()), 4)
    leakage_notes = [
        "feature rows must use trade_date or earlier information only",
        "labels use future trading-day prices only",
        "financial features are joined in train/predict with reported_date-aware as-of logic when available",
        "if reported_date is missing, the system currently falls back to fiscal_date and residual leakage risk remains",
        "recent rows near the dataset tail are expected to have null forward-return labels",
    ]
    if financial_reported_date_null_ratio is not None and financial_reported_date_null_ratio > 0:
        leakage_notes.append(
            f"financial reported_date is missing for {financial_reported_date_null_ratio:.4f} of rows; collector/source enrichment is still needed"
        )

    result = USDatasetValidationResult(
        feature_row_count=len(feature_df) + len(rs_df),
        label_row_count=len(label_df),
        joined_row_count=len(joined),
        ticker_count=len(tickers),
        trade_date_min=trade_date_min,
        trade_date_max=trade_date_max,
        feature_null_ratio_summary=feature_null_summary,
        label_null_ratio_summary=label_null_summary,
        label_distribution=distribution,
        duplicate_key_count=feature_duplicates + rs_duplicates + label_duplicates,
        financial_reported_date_null_ratio=financial_reported_date_null_ratio,
        leakage_risk_notes=leakage_notes,
        report_path=cfg.report_path,
    )
    _write_report(cfg.report_path, result=result, financial_row_count=len(financial_df), rs_row_count=len(rs_df))

    LOGGER.info("[US_DATASET] dataset validation started")
    LOGGER.info("[US_DATASET] feature_tables=%s,%s", cfg.feature_table, cfg.relative_strength_table)
    LOGGER.info("[US_DATASET] label_table=%s", cfg.label_table)
    LOGGER.info("[US_DATASET] feature_row_count=%s label_row_count=%s joined_row_count=%s", result.feature_row_count, result.label_row_count, result.joined_row_count)
    LOGGER.info("[US_DATASET] trade_date_range=%s..%s", result.trade_date_min, result.trade_date_max)
    LOGGER.info("[US_DATASET] feature_null_ratio_summary=%s", result.feature_null_ratio_summary)
    LOGGER.info("[US_DATASET] label_null_ratio_summary=%s", result.label_null_ratio_summary)
    LOGGER.info("[US_DATASET] label_distribution=%s", result.label_distribution)
    LOGGER.info("[US_DATASET] financial_reported_date_null_ratio=%s", result.financial_reported_date_null_ratio)
    LOGGER.info("[US_DATASET] duplicate_key_count=%s", result.duplicate_key_count)
    LOGGER.info("[US_DATASET] dataset validation finished report=%s", cfg.report_path)
    return result


def main() -> None:
    args = parse_args()
    cfg = load_us_dataset_validation_config()
    setup_logging(cfg.log_level)
    runtime_cfg = USDatasetValidationConfig(
        enabled=cfg.enabled,
        feature_table=cfg.feature_table,
        financial_feature_table=cfg.financial_feature_table,
        relative_strength_table=cfg.relative_strength_table,
        label_table=cfg.label_table,
        report_path=cfg.report_path,
        log_level=cfg.log_level,
        universe=str(args.universe or cfg.universe).strip().upper() or cfg.universe,
    )
    if not runtime_cfg.enabled:
        validate_us_stock_ml_dataset(cfg=runtime_cfg, universe_tag=runtime_cfg.universe, limit=args.limit)
        return
    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_DATASET] DB connection failed: {exc}") from exc
    try:
        validate_us_stock_ml_dataset(cfg=runtime_cfg, universe_tag=runtime_cfg.universe, limit=args.limit)
    except KeyboardInterrupt:
        LOGGER.info("[US_DATASET] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_DATASET] Failed error=%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
