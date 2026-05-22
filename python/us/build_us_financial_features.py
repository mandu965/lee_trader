from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.us_config import USFinancialFeatureConfig, load_us_financial_feature_config
from python.us.us_db import (
    fetch_active_tickers,
    fetch_financial_metric_rows,
    fetch_financial_statement_rows,
    get_us_engine,
    upsert_financial_feature_rows,
)


LOGGER = logging.getLogger("us_financial_feature")
SUPPORTED_SOURCE_TABLES = {"raw.us_stock_financial_statement"}
SUPPORTED_METRIC_TABLES = {"raw.us_stock_financial_metric"}
SUPPORTED_TARGET_TABLES = {"feature.us_stock_financial_feature"}
SUPPORTED_WRITE_MODE = "upsert"
SUPPORTED_PERIOD_TYPES = {"annual", "quarterly"}

KEY_COLUMNS = ["ticker", "market", "period_type", "fiscal_date", "source"]
BASE_NUMERIC_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "eps",
    "forward_eps",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cash_flow",
    "free_cash_flow",
    "shares_outstanding",
    "market_cap",
    "roe",
    "roa",
    "debt_to_equity",
    "current_ratio",
    "per",
    "forward_pe",
    "peg_ratio",
    "pbr",
    "psr",
    "ev_ebitda",
    "dividend_yield",
    "analyst_target_price",
    "analyst_recommendation",
    "analyst_count",
]
FEATURE_COLUMNS = [
    "revenue_growth_yoy",
    "revenue_growth_qoq",
    "net_income_growth_yoy",
    "net_income_growth_qoq",
    "eps_growth_yoy",
    "eps_growth_qoq",
    "free_cash_flow_growth_yoy",
    "free_cash_flow_growth_qoq",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "debt_ratio",
    "equity_ratio",
    "debt_to_equity",
    "current_ratio",
    "free_cash_flow_margin",
    "fcf_yield",
    "asset_turnover",
    "gross_margin_trend",
    "revenue_growth_accel",
    "roic_approx",
    "analyst_target_price",
    "analyst_recommendation",
    "analyst_count",
    "analyst_target_upside",
    "financial_quality_score",
    "financial_growth_score",
    "financial_value_score",
]


@dataclass(frozen=True)
class FinancialFeatureBuildResult:
    processed_ticker_count: int
    raw_row_count: int
    built_row_count: int
    upserted_row_count: int
    skipped_row_count: int
    duplicate_key_count: int
    null_ratio_summary: dict[str, float]
    failed_tickers: list[str]


def setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_financial_feature_config()
    parser = argparse.ArgumentParser(description="Build US stock financial features from raw financial tables.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag used to load active tickers.")
    parser.add_argument("--ticker", action="append", default=None, help="Build features for explicit ticker(s).")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for test runs.")
    parser.add_argument("--period-types", default=None, help="Override period types. Example: annual,quarterly")
    return parser.parse_args()


def _normalize_period_types(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    out = tuple(part for part in parts if part)
    if not out:
        raise ValueError("No period types configured.")
    unsupported = [name for name in out if name not in SUPPORTED_PERIOD_TYPES]
    if unsupported:
        raise ValueError(f"Unsupported period types: {unsupported}. Supported: annual, quarterly.")
    return out


def _safe_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(value) or value == float("inf") or value == float("-inf"):
        return None
    return value


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    left = _safe_numeric(numerator)
    right = _safe_numeric(denominator)
    if left is None or right in {None, 0.0}:
        return None
    value = left / right
    return None if pd.isna(value) or value in {float("inf"), float("-inf")} else float(value)


def _safe_growth(current: Any, previous: Any) -> float | None:
    cur = _safe_numeric(current)
    prev = _safe_numeric(previous)
    if cur is None or prev in {None, 0.0}:
        return None
    value = cur / prev - 1.0
    return None if pd.isna(value) or value in {float("inf"), float("-inf")} else float(value)


def _average_flags(flags: list[bool | None]) -> float | None:
    values = [1.0 if flag else 0.0 for flag in flags if flag is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _quality_score(row: pd.Series) -> float | None:
    return _average_flags(
        [
            (_safe_numeric(row.get("roe")) or -1.0) > 0 if row.get("roe") is not None else None,
            (_safe_numeric(row.get("net_margin")) or -1.0) > 0 if row.get("net_margin") is not None else None,
            (_safe_numeric(row.get("free_cash_flow")) or -1.0) > 0 if row.get("free_cash_flow") is not None else None,
            (_safe_numeric(row.get("debt_ratio")) or 9.0) < 0.7 if row.get("debt_ratio") is not None else None,
        ]
    )


def _growth_score(row: pd.Series) -> float | None:
    period_type = str(row.get("period_type") or "")
    if period_type == "quarterly":
        # yfinance provides only ~4-5 quarters, making quarterly YoY (4-period shift) unreliable.
        # QoQ (1-period shift) has adequate coverage.
        return _average_flags(
            [
                (_safe_numeric(row.get("revenue_growth_qoq")) or -9.0) > 0 if row.get("revenue_growth_qoq") is not None else None,
                (_safe_numeric(row.get("net_income_growth_qoq")) or -9.0) > 0 if row.get("net_income_growth_qoq") is not None else None,
                (_safe_numeric(row.get("eps_growth_qoq")) or -9.0) > 0 if row.get("eps_growth_qoq") is not None else None,
            ]
        )
    return _average_flags(
        [
            (_safe_numeric(row.get("revenue_growth_yoy")) or -9.0) > 0 if row.get("revenue_growth_yoy") is not None else None,
            (_safe_numeric(row.get("net_income_growth_yoy")) or -9.0) > 0
            if row.get("net_income_growth_yoy") is not None
            else None,
            (_safe_numeric(row.get("eps_growth_yoy")) or -9.0) > 0 if row.get("eps_growth_yoy") is not None else None,
        ]
    )


def _value_score(row: pd.Series) -> float | None:
    # Prefer forward PE (analyst consensus) over trailing PE when available
    pe = _safe_numeric(row.get("forward_pe")) or _safe_numeric(row.get("per"))
    pe_flag = (0 < (pe or -1.0) <= 35) if pe is not None else None
    peg = _safe_numeric(row.get("peg_ratio"))
    peg_flag = (0 < (peg or -1.0) <= 2.0) if peg is not None else None
    pbr_flag = (0 < (_safe_numeric(row.get("pbr")) or -1.0) <= 5) if row.get("pbr") is not None else None
    psr_flag = (0 < (_safe_numeric(row.get("psr")) or -1.0) <= 10) if row.get("psr") is not None else None
    # PEG available: use pe + peg + pbr; otherwise: pe + pbr + psr (original set)
    if peg_flag is not None:
        return _average_flags([pe_flag, peg_flag, pbr_flag])
    return _average_flags([pe_flag, pbr_flag, psr_flag])


def _prepare_financial_raw_frame(
    statement_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> pd.DataFrame:
    statement_df = pd.DataFrame(statement_rows)
    metric_df = pd.DataFrame(metric_rows)

    if statement_df.empty and metric_df.empty:
        return pd.DataFrame(columns=KEY_COLUMNS)
    if statement_df.empty:
        combined = metric_df.copy()
    elif metric_df.empty:
        combined = statement_df.copy()
    else:
        combined = statement_df.merge(
            metric_df,
            on=KEY_COLUMNS,
            how="outer",
            suffixes=("_stmt", "_metric"),
        )
        for column in ["reported_date", "currency", "source_updated_at", "collected_at"]:
            stmt_col = f"{column}_stmt"
            metric_col = f"{column}_metric"
            if stmt_col in combined.columns or metric_col in combined.columns:
                combined[column] = combined.get(stmt_col).combine_first(combined.get(metric_col))
                combined = combined.drop(columns=[name for name in [stmt_col, metric_col] if name in combined.columns])

    for column in ["fiscal_date", "reported_date"]:
        if column in combined.columns:
            combined[column] = pd.to_datetime(combined[column], errors="coerce").dt.date
    if "collected_at" in combined.columns:
        combined["collected_at"] = pd.to_datetime(combined["collected_at"], errors="coerce", utc=True)

    for column in BASE_NUMERIC_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    combined = combined.dropna(subset=["ticker", "period_type", "fiscal_date", "source"]).copy()
    combined = combined.sort_values(KEY_COLUMNS + ["collected_at"], na_position="last")
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)
    return combined


def build_financial_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=KEY_COLUMNS + FEATURE_COLUMNS)

    frame = raw_df.copy()
    frame["fiscal_date"] = pd.to_datetime(frame["fiscal_date"], errors="coerce").dt.date
    frame = frame.sort_values(["ticker", "period_type", "fiscal_date"]).reset_index(drop=True)

    for column in BASE_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    groups: list[pd.DataFrame] = []
    for (_, period_type), group in frame.groupby(["ticker", "period_type"], sort=False):
        ordered = group.sort_values("fiscal_date").reset_index(drop=True).copy()
        yoy_shift = 1 if period_type == "annual" else 4
        qoq_shift = None if period_type == "annual" else 1

        ordered["revenue_growth_yoy"] = [
            _safe_growth(cur, prev)
            for cur, prev in zip(ordered["revenue"], ordered["revenue"].shift(yoy_shift), strict=False)
        ]
        ordered["net_income_growth_yoy"] = [
            _safe_growth(cur, prev)
            for cur, prev in zip(ordered["net_income"], ordered["net_income"].shift(yoy_shift), strict=False)
        ]
        ordered["eps_growth_yoy"] = [
            _safe_growth(cur, prev)
            for cur, prev in zip(ordered["eps"], ordered["eps"].shift(yoy_shift), strict=False)
        ]
        ordered["free_cash_flow_growth_yoy"] = [
            _safe_growth(cur, prev)
            for cur, prev in zip(ordered["free_cash_flow"], ordered["free_cash_flow"].shift(yoy_shift), strict=False)
        ]

        if qoq_shift is None:
            ordered["revenue_growth_qoq"] = pd.NA
            ordered["net_income_growth_qoq"] = pd.NA
            ordered["eps_growth_qoq"] = pd.NA
            ordered["free_cash_flow_growth_qoq"] = pd.NA
        else:
            ordered["revenue_growth_qoq"] = [
                _safe_growth(cur, prev)
                for cur, prev in zip(ordered["revenue"], ordered["revenue"].shift(qoq_shift), strict=False)
            ]
            ordered["net_income_growth_qoq"] = [
                _safe_growth(cur, prev)
                for cur, prev in zip(ordered["net_income"], ordered["net_income"].shift(qoq_shift), strict=False)
            ]
            ordered["eps_growth_qoq"] = [
                _safe_growth(cur, prev)
                for cur, prev in zip(ordered["eps"], ordered["eps"].shift(qoq_shift), strict=False)
            ]
            ordered["free_cash_flow_growth_qoq"] = [
                _safe_growth(cur, prev)
                for cur, prev in zip(ordered["free_cash_flow"], ordered["free_cash_flow"].shift(qoq_shift), strict=False)
            ]

        ordered["gross_margin"] = [_safe_ratio(a, b) for a, b in zip(ordered["gross_profit"], ordered["revenue"], strict=False)]
        ordered["operating_margin"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["operating_income"], ordered["revenue"], strict=False)
        ]
        ordered["net_margin"] = [_safe_ratio(a, b) for a, b in zip(ordered["net_income"], ordered["revenue"], strict=False)]
        ordered["ebitda_margin"] = [_safe_ratio(a, b) for a, b in zip(ordered["ebitda"], ordered["revenue"], strict=False)]
        ordered["free_cash_flow_margin"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["free_cash_flow"], ordered["revenue"], strict=False)
        ]

        ordered["debt_ratio"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["total_liabilities"], ordered["total_assets"], strict=False)
        ]
        ordered["equity_ratio"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["total_equity"], ordered["total_assets"], strict=False)
        ]
        ordered["debt_to_equity"] = [
            existing if _safe_numeric(existing) is not None else _safe_ratio(liab, equity)
            for existing, liab, equity in zip(
                ordered["debt_to_equity"],
                ordered["total_liabilities"],
                ordered["total_equity"],
                strict=False,
            )
        ]

        # FCF yield: free cash flow / market cap (positive = cash generative, low = expensive)
        ordered["fcf_yield"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["free_cash_flow"], ordered["market_cap"], strict=False)
        ]

        # Asset turnover: revenue / total assets (efficiency of asset utilization)
        ordered["asset_turnover"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["revenue"], ordered["total_assets"], strict=False)
        ]

        # ROIC approximation: operating_income / total_assets (operating return on total assets)
        ordered["roic_approx"] = [
            _safe_ratio(a, b) for a, b in zip(ordered["operating_income"], ordered["total_assets"], strict=False)
        ]

        # Gross margin trend: QoQ change in gross margin (momentum of margin expansion)
        gm = ordered["gross_margin"]
        ordered["gross_margin_trend"] = [
            _safe_numeric(cur) - _safe_numeric(prev)
            if _safe_numeric(cur) is not None and _safe_numeric(prev) is not None
            else None
            for cur, prev in zip(gm, gm.shift(1), strict=False)
        ]

        # Revenue growth acceleration: QoQ change in revenue_growth_qoq (is growth speeding up?)
        rg = ordered["revenue_growth_qoq"]
        ordered["revenue_growth_accel"] = [
            _safe_numeric(cur) - _safe_numeric(prev)
            if _safe_numeric(cur) is not None and _safe_numeric(prev) is not None
            else None
            for cur, prev in zip(rg, rg.shift(1), strict=False)
        ]

        # Analyst target upside: (target_price / current_price) - 1
        # Current price = market_cap / shares_outstanding
        def _analyst_upside(row: pd.Series) -> float | None:
            target = _safe_numeric(row.get("analyst_target_price"))
            mktcap = _safe_numeric(row.get("market_cap"))
            shares = _safe_numeric(row.get("shares_outstanding"))
            if target is None or mktcap is None or shares in {None, 0.0}:
                return None
            current_price = mktcap / shares
            if current_price <= 0:
                return None
            return target / current_price - 1.0

        ordered["analyst_target_upside"] = ordered.apply(_analyst_upside, axis=1)

        ordered["financial_quality_score"] = ordered.apply(_quality_score, axis=1)
        ordered["financial_growth_score"] = ordered.apply(_growth_score, axis=1)
        ordered["financial_value_score"] = ordered.apply(_value_score, axis=1)
        groups.append(ordered)

    out = pd.concat(groups, ignore_index=True) if groups else frame.iloc[0:0].copy()
    out["raw_collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out["feature_created_at"] = datetime.now(timezone.utc)
    for column in FEATURE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].where(pd.notna(out[column]), None)
    return out


def _frame_to_feature_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    columns = [
        "ticker",
        "market",
        "period_type",
        "fiscal_date",
        "reported_date",
        "source",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "ebitda",
        "eps",
        "forward_eps",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "operating_cash_flow",
        "free_cash_flow",
        "shares_outstanding",
        "market_cap",
        "revenue_growth_yoy",
        "revenue_growth_qoq",
        "net_income_growth_yoy",
        "net_income_growth_qoq",
        "eps_growth_yoy",
        "eps_growth_qoq",
        "free_cash_flow_growth_yoy",
        "free_cash_flow_growth_qoq",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "ebitda_margin",
        "roe",
        "roa",
        "debt_to_equity",
        "debt_ratio",
        "equity_ratio",
        "current_ratio",
        "free_cash_flow_margin",
        "fcf_yield",
        "asset_turnover",
        "gross_margin_trend",
        "revenue_growth_accel",
        "roic_approx",
        "analyst_target_price",
        "analyst_recommendation",
        "analyst_count",
        "analyst_target_upside",
        "per",
        "forward_pe",
        "peg_ratio",
        "pbr",
        "psr",
        "ev_ebitda",
        "dividend_yield",
        "financial_quality_score",
        "financial_growth_score",
        "financial_value_score",
        "raw_collected_at",
        "feature_created_at",
    ]
    rows: list[dict[str, object]] = []
    for record in frame[columns].to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.to_pydatetime()
            elif pd.isna(value) if not isinstance(value, (str, bytes)) else False:
                clean[key] = None
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def _null_ratio_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    summary: dict[str, float] = {}
    for column in [
        "revenue_growth_yoy",
        "revenue_growth_qoq",
        "gross_margin",
        "debt_ratio",
        "financial_quality_score",
        "financial_growth_score",
        "financial_value_score",
    ]:
        if column in frame.columns:
            summary[column] = round(float(frame[column].isna().mean()), 4)
    return summary


def _validate_config(cfg: USFinancialFeatureConfig) -> tuple[str, ...]:
    if cfg.write_mode != SUPPORTED_WRITE_MODE:
        raise ValueError(
            f"Unsupported US_FINANCIAL_FEATURE_WRITE_MODE='{cfg.write_mode}'. Only '{SUPPORTED_WRITE_MODE}' is supported."
        )
    if cfg.source_statement_table not in SUPPORTED_SOURCE_TABLES:
        raise ValueError(
            f"Unsupported US_FINANCIAL_FEATURE_SOURCE_STATEMENT_TABLE='{cfg.source_statement_table}'."
        )
    if cfg.source_metric_table not in SUPPORTED_METRIC_TABLES:
        raise ValueError(
            f"Unsupported US_FINANCIAL_FEATURE_SOURCE_METRIC_TABLE='{cfg.source_metric_table}'."
        )
    if cfg.target_table not in SUPPORTED_TARGET_TABLES:
        raise ValueError(f"Unsupported US_FINANCIAL_FEATURE_TARGET_TABLE='{cfg.target_table}'.")
    return _normalize_period_types(cfg.period_types)


def build_us_financial_features(
    *,
    cfg: USFinancialFeatureConfig,
    universe_tag: str,
    explicit_tickers: list[str] | None = None,
    limit: int | None = None,
    statement_fetcher=fetch_financial_statement_rows,
    metric_fetcher=fetch_financial_metric_rows,
    row_writer=upsert_financial_feature_rows,
) -> FinancialFeatureBuildResult:
    if not cfg.enabled:
        LOGGER.info("[US_FIN_FEATURE] US_FINANCIAL_FEATURE_BUILD_ENABLED=0. Skip feature builder.")
        return FinancialFeatureBuildResult(0, 0, 0, 0, 0, 0, {}, [])

    period_types = _validate_config(cfg)
    tickers = (
        [str(ticker).strip().upper() for ticker in explicit_tickers if str(ticker).strip()]
        if explicit_tickers
        else fetch_active_tickers(universe_tag)
    )
    if limit is not None:
        tickers = tickers[:limit]

    if not tickers:
        LOGGER.info("[US_FIN_FEATURE] No active tickers found. universe=%s", universe_tag)
        return FinancialFeatureBuildResult(0, 0, 0, 0, 0, 0, {}, [])

    min_fiscal_date = date.today() - timedelta(days=max(1, cfg.lookback_years) * 366)
    statement_rows = statement_fetcher(tickers, period_types=list(period_types), min_fiscal_date=min_fiscal_date)
    metric_rows = metric_fetcher(tickers, period_types=list(period_types), min_fiscal_date=min_fiscal_date)
    raw_row_count = max(len(statement_rows), len(metric_rows))
    combined = _prepare_financial_raw_frame(statement_rows, metric_rows)
    duplicate_key_count = max(0, raw_row_count - len(combined))

    LOGGER.info("[US_FIN_FEATURE] feature build started")
    LOGGER.info("[US_FIN_FEATURE] source_statement_table=%s", cfg.source_statement_table)
    LOGGER.info("[US_FIN_FEATURE] source_metric_table=%s", cfg.source_metric_table)
    LOGGER.info("[US_FIN_FEATURE] target_table=%s", cfg.target_table)
    LOGGER.info("[US_FIN_FEATURE] raw_rows=%s tickers=%s", len(combined), len(tickers))

    built_rows = 0
    upserted_rows = 0
    skipped_rows = 0
    failed_tickers: list[str] = []
    built_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            ticker_raw = combined.loc[combined["ticker"] == ticker].copy()
            if ticker_raw.empty:
                skipped_rows += 1
                LOGGER.info("[US_FIN_FEATURE] %s SKIPPED reason=no raw rows", ticker)
                continue
            built = build_financial_feature_frame(ticker_raw)
            if built.empty:
                skipped_rows += 1
                LOGGER.info("[US_FIN_FEATURE] %s SKIPPED reason=no feature rows", ticker)
                continue
            rows = _frame_to_feature_rows(built)
            saved = row_writer(rows)
            built_rows += len(rows)
            upserted_rows += saved
            built_frames.append(built)
            LOGGER.info("[US_FIN_FEATURE] %s SUCCESS rows=%s", ticker, saved)
        except Exception as exc:
            failed_tickers.append(ticker)
            LOGGER.info("[US_FIN_FEATURE] %s FAILED error=%s", ticker, exc)

    final_frame = pd.concat(built_frames, ignore_index=True) if built_frames else pd.DataFrame()
    null_summary = _null_ratio_summary(final_frame)
    if not final_frame.empty:
        period_counts = final_frame["period_type"].value_counts().to_dict()
        LOGGER.info("[US_FIN_FEATURE] period_type_counts=%s", period_counts)
        reported_ratio = float(final_frame["reported_date"].notna().mean()) if "reported_date" in final_frame.columns else 0.0
        LOGGER.info("[US_FIN_FEATURE] reported_date_coverage=%s", round(reported_ratio, 4))
        if reported_ratio < 1.0:
            LOGGER.info(
                "[US_FIN_FEATURE] reported_date_missing_rows=%s",
                int(final_frame["reported_date"].isna().sum()) if "reported_date" in final_frame.columns else len(final_frame),
            )
    LOGGER.info("[US_FIN_FEATURE] built_rows=%s upserted_rows=%s skipped_rows=%s", built_rows, upserted_rows, skipped_rows)
    LOGGER.info("[US_FIN_FEATURE] null_ratio_summary=%s", null_summary)
    LOGGER.info("[US_FIN_FEATURE] duplicate_key_count=%s", duplicate_key_count)
    if failed_tickers:
        LOGGER.info("[US_FIN_FEATURE] failed_tickers=%s", ",".join(failed_tickers))
    LOGGER.info("[US_FIN_FEATURE] feature build finished")

    return FinancialFeatureBuildResult(
        processed_ticker_count=len(tickers),
        raw_row_count=len(combined),
        built_row_count=built_rows,
        upserted_row_count=upserted_rows,
        skipped_row_count=skipped_rows,
        duplicate_key_count=duplicate_key_count,
        null_ratio_summary=null_summary,
        failed_tickers=failed_tickers,
    )


def main() -> None:
    args = parse_args()
    cfg = load_us_financial_feature_config()
    setup_logging(cfg.log_level)

    runtime_cfg = USFinancialFeatureConfig(
        enabled=cfg.enabled,
        source_statement_table=cfg.source_statement_table,
        source_metric_table=cfg.source_metric_table,
        target_table=cfg.target_table,
        period_types=_normalize_period_types(args.period_types or cfg.period_types),
        lookback_years=cfg.lookback_years,
        write_mode=cfg.write_mode,
        log_level=cfg.log_level,
        universe=str(args.universe or cfg.universe).strip().upper() or cfg.universe,
    )
    tickers = [str(name).strip().upper() for name in (args.ticker or []) if str(name).strip()] or None

    if not runtime_cfg.enabled:
        build_us_financial_features(
            cfg=runtime_cfg,
            universe_tag=runtime_cfg.universe,
            explicit_tickers=tickers,
            limit=args.limit,
        )
        return

    try:
        with get_us_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise SystemExit(f"[US_FIN_FEATURE] DB connection failed: {exc}") from exc

    try:
        build_us_financial_features(
            cfg=runtime_cfg,
            universe_tag=runtime_cfg.universe,
            explicit_tickers=tickers,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        LOGGER.info("[US_FIN_FEATURE] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_FIN_FEATURE] Failed error=%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
