from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import os
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


def _flag(name: str, default: str) -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class USStockConfig:
    enabled: bool
    universe: str
    data_source: str
    root_dir: Path
    data_dir: Path
    us_data_dir: Path
    default_universe_csv: Path
    price_backfill_years: int
    price_start_date: str
    price_end_date: str
    stale_days_limit: int
    batch_size: int
    request_sleep_sec: float
    paper_trading_enabled: bool
    live_trading_enabled: bool
    live_broker: str


@dataclass(frozen=True)
class USFinancialCollectorConfig:
    enabled: bool
    source: str
    universe: str
    period_types: tuple[str, ...]
    lookback_years: int
    max_tickers_per_run: int
    sleep_sec: float
    retry_count: int
    retry_sleep_sec: float
    fail_fast: bool
    write_mode: str
    log_level: str


@dataclass(frozen=True)
class USFinancialFeatureConfig:
    enabled: bool
    source_statement_table: str
    source_metric_table: str
    target_table: str
    period_types: tuple[str, ...]
    lookback_years: int
    write_mode: str
    log_level: str
    universe: str


@dataclass(frozen=True)
class USRelativeStrengthConfig:
    enabled: bool
    source_table: str
    target_table: str
    benchmarks: tuple[str, ...]
    windows: tuple[int, ...]
    price_column: str
    write_mode: str
    log_level: str
    universe: str


@dataclass(frozen=True)
class USLabelConfig:
    enabled: bool
    source_price_table: str
    target_table: str
    price_column: str
    windows: tuple[int, ...]
    top_percentile: float
    min_universe_size: int
    exclude_benchmarks: tuple[str, ...]
    write_mode: str
    log_level: str
    universe: str


@dataclass(frozen=True)
class USDatasetValidationConfig:
    enabled: bool
    feature_table: str
    financial_feature_table: str
    relative_strength_table: str
    label_table: str
    report_path: Path
    log_level: str
    universe: str


@dataclass(frozen=True)
class USRankingTableConfig:
    enabled: bool
    target_table: str
    default_source: str
    verify_sample_trade_date: date
    log_level: str


@dataclass(frozen=True)
class USRuleRankingConfig:
    enabled: bool
    source: str
    min_feature_quality_score: float
    apply_fundamental_quality_to_etf: bool
    volatility_20d_threshold: float
    volatility_60d_threshold: float
    return_20d_overheat_threshold: float
    strong_buy_score: float
    buy_score: float
    watch_score: float
    hold_score: float
    log_level: str


@dataclass(frozen=True)
class USUniverseFilterConfig:
    min_market_cap: float
    min_avg_volume: float
    min_feature_quality_score: float
    include_etf: bool
    exclude_leveraged: bool
    exclude_inverse: bool


@dataclass(frozen=True)
class USRankReportConfig:
    output_dir: Path
    top_n: int
    email_enabled: bool
    log_level: str


@dataclass(frozen=True)
class USBacktestReportConfig:
    output_dir: Path
    default_format: str
    recent_days: int
    best_worst_limit: int
    min_test_days_warning: int
    missing_rate_warning: float
    log_level: str


@dataclass(frozen=True)
class USMarketRegimeConfig:
    spy_vol20_high_threshold: float
    qqq_vol20_high_threshold: float
    min_test_days_warning: int
    report_output_dir: Path
    report_default_format: str
    log_level: str


@dataclass(frozen=True)
class USWeightExperimentConfig:
    output_dir: Path
    default_holding_days: tuple[int, ...]
    default_strategies: tuple[str, ...]
    min_test_days: int
    baseline_weight_config_id: str
    log_level: str


@dataclass(frozen=True)
class USForwardTestConfig:
    enabled: bool
    forward_test_id: str
    holding_days: tuple[int, ...]
    strategies: tuple[str, ...]
    output_dir: Path
    auto_register: bool
    auto_update: bool
    auto_report: bool
    log_level: str


@dataclass(frozen=True)
class USPaperTradingConfig:
    enabled: bool
    account_id: str
    account_name: str
    initial_cash: float
    base_currency: str
    max_positions: int
    max_position_weight: float
    max_sector_weight: float
    min_cash_weight: float
    max_daily_new_buys: int
    allow_fractional_shares: bool
    buy_grades: tuple[str, ...] = ()
    sell_grades: tuple[str, ...] = ()
    max_rank_no: int = 20
    rebalance_enabled: bool = False
    rebalance_frequency: str = "DAILY"
    rebalance_sell_first: bool = True
    rebalance_allow_rebuy_same_day: bool = False
    rebalance_min_amount: float = 100.0
    rebalance_min_weight_diff: float = 0.02
    rebalance_full_sell_on_rank_exit: bool = True
    rebalance_full_sell_on_grade_downgrade: bool = True
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0
    real_order_blocked: bool = True
    output_dir: Path | None = None
    report_output_dir: Path | None = None
    use_previous_close_if_missing: bool = True
    auto_snapshot_enabled: bool = False
    auto_report_enabled: bool = False
    validation_enabled: bool = True
    validation_fail_on_error: bool = False
    scheduler_enabled: bool = False
    scheduler_account_id: str = ""
    scheduler_run_rebalance_plan: bool = True
    scheduler_generate_orders: bool = False
    scheduler_simulate_fills: bool = False
    scheduler_update_snapshot: bool = False
    scheduler_validate: bool = True
    scheduler_report: bool = True
    log_level: str = "INFO"
    config_path: Path | None = None


def load_us_stock_config() -> USStockConfig:
    """
    Load Project C Phase 1 environment configuration.

    This module does not implement data collection or trading logic.
    """
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data"
    us_data_dir = data_dir / "us"
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    return USStockConfig(
        enabled=_flag("US_STOCK_ENABLED", "0"),
        universe=universe,
        data_source=str(os.environ.get("US_STOCK_DATA_SOURCE", "yfinance")).strip() or "yfinance",
        root_dir=root_dir,
        data_dir=data_dir,
        us_data_dir=us_data_dir,
        default_universe_csv=us_data_dir / f"{universe.lower()}_universe.csv",
        price_backfill_years=int(os.environ.get("US_STOCK_PRICE_BACKFILL_YEARS", "5")),
        price_start_date=str(os.environ.get("US_STOCK_PRICE_START_DATE", "")).strip(),
        price_end_date=str(os.environ.get("US_STOCK_PRICE_END_DATE", "")).strip(),
        stale_days_limit=int(os.environ.get("US_STOCK_STALE_DAYS_LIMIT", "3")),
        batch_size=int(os.environ.get("US_STOCK_BATCH_SIZE", "20")),
        request_sleep_sec=float(os.environ.get("US_STOCK_REQUEST_SLEEP_SEC", "1")),
        paper_trading_enabled=_flag("US_PAPER_TRADING_ENABLED", "0"),
        live_trading_enabled=_flag("US_LIVE_TRADING_ENABLED", "0"),
        live_broker=str(os.environ.get("US_LIVE_BROKER", "none")).strip() or "none",
    )


def _split_csv(value: str | None) -> tuple[str, ...]:
    parts = [part.strip() for part in str(value or "").split(",")]
    return tuple(part for part in parts if part)


def _parse_scalar(text: str):
    value = text.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in {".", "e", "E"}):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml_mapping(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, remainder = line.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not remainder:
            child: dict = {}
            current[key] = child
            stack.append((indent, child))
            continue
        current[key] = _parse_scalar(remainder)
    return root


def load_us_financial_collector_config() -> USFinancialCollectorConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    period_types = tuple(name.lower() for name in _split_csv(os.environ.get("US_FINANCIAL_PERIOD_TYPES", "annual,quarterly")))
    return USFinancialCollectorConfig(
        enabled=_flag("US_FINANCIAL_COLLECT_ENABLED", "0"),
        source=str(os.environ.get("US_FINANCIAL_SOURCE", "yfinance")).strip() or "yfinance",
        universe=universe,
        period_types=period_types or ("annual", "quarterly"),
        lookback_years=int(os.environ.get("US_FINANCIAL_LOOKBACK_YEARS", "5")),
        max_tickers_per_run=int(os.environ.get("US_FINANCIAL_MAX_TICKERS_PER_RUN", "100")),
        sleep_sec=float(os.environ.get("US_FINANCIAL_SLEEP_SEC", "1.0")),
        retry_count=int(os.environ.get("US_FINANCIAL_RETRY_COUNT", "3")),
        retry_sleep_sec=float(os.environ.get("US_FINANCIAL_RETRY_SLEEP_SEC", "5")),
        fail_fast=_flag("US_FINANCIAL_FAIL_FAST", "0"),
        write_mode=str(os.environ.get("US_FINANCIAL_WRITE_MODE", "upsert")).strip().lower() or "upsert",
        log_level=str(os.environ.get("US_FINANCIAL_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_financial_feature_config() -> USFinancialFeatureConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    period_types = tuple(name.lower() for name in _split_csv(os.environ.get("US_FINANCIAL_FEATURE_PERIOD_TYPES", "annual,quarterly")))
    return USFinancialFeatureConfig(
        enabled=_flag("US_FINANCIAL_FEATURE_BUILD_ENABLED", "0"),
        source_statement_table=str(
            os.environ.get("US_FINANCIAL_FEATURE_SOURCE_STATEMENT_TABLE", "raw.us_stock_financial_statement")
        ).strip()
        or "raw.us_stock_financial_statement",
        source_metric_table=str(
            os.environ.get("US_FINANCIAL_FEATURE_SOURCE_METRIC_TABLE", "raw.us_stock_financial_metric")
        ).strip()
        or "raw.us_stock_financial_metric",
        target_table=str(
            os.environ.get("US_FINANCIAL_FEATURE_TARGET_TABLE", "feature.us_stock_financial_feature")
        ).strip()
        or "feature.us_stock_financial_feature",
        period_types=period_types or ("annual", "quarterly"),
        lookback_years=int(os.environ.get("US_FINANCIAL_FEATURE_LOOKBACK_YEARS", "5")),
        write_mode=str(os.environ.get("US_FINANCIAL_FEATURE_WRITE_MODE", "upsert")).strip().lower() or "upsert",
        log_level=str(os.environ.get("US_FINANCIAL_FEATURE_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        universe=universe,
    )


def load_us_relative_strength_config() -> USRelativeStrengthConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    benchmarks = tuple(name.upper() for name in _split_csv(os.environ.get("US_RELATIVE_STRENGTH_BENCHMARKS", "SPY,QQQ")))
    windows = tuple(
        int(part)
        for part in _split_csv(os.environ.get("US_RELATIVE_STRENGTH_WINDOWS", "5,20,60,120,252"))
        if str(part).strip()
    )
    return USRelativeStrengthConfig(
        enabled=_flag("US_RELATIVE_STRENGTH_BUILD_ENABLED", "0"),
        source_table=str(os.environ.get("US_RELATIVE_STRENGTH_SOURCE_TABLE", "market.us_stock_daily_price")).strip()
        or "market.us_stock_daily_price",
        target_table=str(
            os.environ.get("US_RELATIVE_STRENGTH_TARGET_TABLE", "feature.us_stock_relative_strength_daily")
        ).strip()
        or "feature.us_stock_relative_strength_daily",
        benchmarks=benchmarks or ("SPY", "QQQ"),
        windows=windows or (5, 20, 60, 120, 252),
        price_column=str(os.environ.get("US_RELATIVE_STRENGTH_PRICE_COLUMN", "auto")).strip().lower() or "auto",
        write_mode=str(os.environ.get("US_RELATIVE_STRENGTH_WRITE_MODE", "upsert")).strip().lower() or "upsert",
        log_level=str(os.environ.get("US_RELATIVE_STRENGTH_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        universe=universe,
    )


def load_us_label_config() -> USLabelConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    windows = tuple(
        int(part)
        for part in _split_csv(os.environ.get("US_LABEL_WINDOWS", "5,20,60"))
        if str(part).strip()
    )
    exclude_benchmarks = tuple(
        name.upper() for name in _split_csv(os.environ.get("US_LABEL_EXCLUDE_BENCHMARKS", "SPY,QQQ"))
    )
    return USLabelConfig(
        enabled=_flag("US_LABEL_BUILD_ENABLED", "0"),
        source_price_table=str(os.environ.get("US_LABEL_SOURCE_PRICE_TABLE", "market.us_stock_daily_price")).strip()
        or "market.us_stock_daily_price",
        target_table=str(os.environ.get("US_LABEL_TARGET_TABLE", "label.us_stock_label_daily")).strip()
        or "label.us_stock_label_daily",
        price_column=str(os.environ.get("US_LABEL_PRICE_COLUMN", "auto")).strip().lower() or "auto",
        windows=windows or (5, 20, 60),
        top_percentile=float(os.environ.get("US_LABEL_TOP_PERCENTILE", "0.20")),
        min_universe_size=int(os.environ.get("US_LABEL_MIN_UNIVERSE_SIZE", "30")),
        exclude_benchmarks=exclude_benchmarks or ("SPY", "QQQ"),
        write_mode=str(os.environ.get("US_LABEL_WRITE_MODE", "upsert")).strip().lower() or "upsert",
        log_level=str(os.environ.get("US_LABEL_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        universe=universe,
    )


def load_us_dataset_validation_config() -> USDatasetValidationConfig:
    universe = str(os.environ.get("US_STOCK_UNIVERSE", "NASDAQ100")).strip().upper() or "NASDAQ100"
    root_dir = Path(__file__).resolve().parents[2]
    report_path_raw = str(
        os.environ.get("US_DATASET_REPORT_PATH", "reports/us_stock_dataset_validation.md")
    ).strip() or "reports/us_stock_dataset_validation.md"
    report_path = Path(report_path_raw)
    if not report_path.is_absolute():
        report_path = root_dir / report_path
    return USDatasetValidationConfig(
        enabled=_flag("US_DATASET_VALIDATE_ENABLED", "0"),
        feature_table=str(os.environ.get("US_DATASET_FEATURE_TABLE", "feature.us_stock_feature_daily")).strip()
        or "feature.us_stock_feature_daily",
        financial_feature_table=str(
            os.environ.get("US_DATASET_FINANCIAL_FEATURE_TABLE", "feature.us_stock_financial_feature")
        ).strip()
        or "feature.us_stock_financial_feature",
        relative_strength_table=str(
            os.environ.get("US_DATASET_RELATIVE_STRENGTH_TABLE", "feature.us_stock_relative_strength_daily")
        ).strip()
        or "feature.us_stock_relative_strength_daily",
        label_table=str(os.environ.get("US_DATASET_LABEL_TABLE", "label.us_stock_label_daily")).strip()
        or "label.us_stock_label_daily",
        report_path=report_path,
        log_level=str(os.environ.get("US_DATASET_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        universe=universe,
    )


def load_us_ranking_table_config() -> USRankingTableConfig:
    verify_sample_trade_date = parse_iso_date(
        os.environ.get("US_RANKING_VERIFY_SAMPLE_TRADE_DATE", "2099-12-31"),
        field_name="US_RANKING_VERIFY_SAMPLE_TRADE_DATE",
    )
    if verify_sample_trade_date is None:
        verify_sample_trade_date = date.today() + timedelta(days=365 * 50)
    return USRankingTableConfig(
        enabled=_flag("US_RANKING_TABLE_ENABLED", "0"),
        target_table=str(os.environ.get("US_RANKING_TARGET_TABLE", "recommend.us_stock_rank_daily")).strip()
        or "recommend.us_stock_rank_daily",
        default_source=str(os.environ.get("US_RANKING_DEFAULT_SOURCE", "rule_v1")).strip() or "rule_v1",
        verify_sample_trade_date=verify_sample_trade_date,
        log_level=str(os.environ.get("US_RANKING_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_rule_ranking_config() -> USRuleRankingConfig:
    return USRuleRankingConfig(
        enabled=_flag("US_RULE_RANKING_ENABLED", "1"),
        source=str(os.environ.get("US_RULE_RANKING_SOURCE", "rule_v1")).strip() or "rule_v1",
        min_feature_quality_score=float(os.environ.get("US_RANK_MIN_FEATURE_QUALITY_SCORE", "40")),
        apply_fundamental_quality_to_etf=_flag("US_RANK_APPLY_FUNDAMENTAL_QUALITY_TO_ETF", "0"),
        volatility_20d_threshold=float(os.environ.get("US_RANK_VOLATILITY_20D_THRESHOLD", "0.05")),
        volatility_60d_threshold=float(os.environ.get("US_RANK_VOLATILITY_60D_THRESHOLD", "0.04")),
        return_20d_overheat_threshold=float(os.environ.get("US_RANK_RETURN_20D_OVERHEAT_THRESHOLD", "0.25")),
        strong_buy_score=float(os.environ.get("US_RANK_STRONG_BUY_SCORE", "80")),
        buy_score=float(os.environ.get("US_RANK_BUY_SCORE", "70")),
        watch_score=float(os.environ.get("US_RANK_WATCH_SCORE", "60")),
        hold_score=float(os.environ.get("US_RANK_HOLD_SCORE", "50")),
        log_level=str(os.environ.get("US_RANKING_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_universe_filter_config() -> USUniverseFilterConfig:
    return USUniverseFilterConfig(
        min_market_cap=float(os.environ.get("US_UNIVERSE_MIN_MARKET_CAP", "10000000000")),
        min_avg_volume=float(os.environ.get("US_UNIVERSE_MIN_AVG_VOLUME", "1000000")),
        min_feature_quality_score=float(os.environ.get("US_UNIVERSE_MIN_FEATURE_QUALITY_SCORE", "40")),
        include_etf=_flag("US_UNIVERSE_INCLUDE_ETF", "1"),
        exclude_leveraged=_flag("US_UNIVERSE_EXCLUDE_LEVERAGED", "1"),
        exclude_inverse=_flag("US_UNIVERSE_EXCLUDE_INVERSE", "1"),
    )


def load_us_rank_report_config() -> USRankReportConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(
        os.environ.get("US_RANK_REPORT_OUTPUT_DIR", "outputs/us_stock_top_rank")
    ).strip() or "outputs/us_stock_top_rank"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    return USRankReportConfig(
        output_dir=output_dir,
        top_n=max(1, int(os.environ.get("US_RANK_REPORT_TOP_N", "20"))),
        email_enabled=_flag("US_RANK_REPORT_EMAIL_ENABLED", "0"),
        log_level=str(os.environ.get("US_RANK_REPORT_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_backtest_report_config() -> USBacktestReportConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(
        os.environ.get("US_BACKTEST_REPORT_OUTPUT_DIR", "outputs/us_stock_backtest")
    ).strip() or "outputs/us_stock_backtest"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    return USBacktestReportConfig(
        output_dir=output_dir,
        default_format=str(os.environ.get("US_BACKTEST_REPORT_DEFAULT_FORMAT", "console")).strip().lower() or "console",
        recent_days=max(1, int(os.environ.get("US_BACKTEST_REPORT_RECENT_DAYS", "10"))),
        best_worst_limit=max(1, int(os.environ.get("US_BACKTEST_REPORT_BEST_WORST_LIMIT", "10"))),
        min_test_days_warning=max(1, int(os.environ.get("US_BACKTEST_MIN_TEST_DAYS_WARNING", "30"))),
        missing_rate_warning=float(os.environ.get("US_BACKTEST_MISSING_RATE_WARNING", "0.1")),
        log_level=str(os.environ.get("US_BACKTEST_REPORT_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_market_regime_config() -> USMarketRegimeConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(
        os.environ.get("US_REGIME_REPORT_OUTPUT_DIR", "outputs/us_stock_backtest")
    ).strip() or "outputs/us_stock_backtest"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    return USMarketRegimeConfig(
        spy_vol20_high_threshold=float(os.environ.get("US_REGIME_SPY_VOL20_HIGH_THRESHOLD", "0.025")),
        qqq_vol20_high_threshold=float(os.environ.get("US_REGIME_QQQ_VOL20_HIGH_THRESHOLD", "0.030")),
        min_test_days_warning=max(1, int(os.environ.get("US_REGIME_MIN_TEST_DAYS_WARNING", "20"))),
        report_output_dir=output_dir,
        report_default_format=str(os.environ.get("US_REGIME_REPORT_DEFAULT_FORMAT", "console")).strip().lower() or "console",
        log_level=str(os.environ.get("US_REGIME_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_weight_experiment_config() -> USWeightExperimentConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(
        os.environ.get("US_WEIGHT_EXPERIMENT_OUTPUT_DIR", "outputs/us_stock_weight_experiment")
    ).strip() or "outputs/us_stock_weight_experiment"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    holding_days = tuple(
        int(part)
        for part in _split_csv(os.environ.get("US_WEIGHT_EXPERIMENT_DEFAULT_HOLDING_DAYS", "5,20,60"))
        if str(part).strip()
    )
    strategies = tuple(
        str(part).strip().upper()
        for part in _split_csv(os.environ.get("US_WEIGHT_EXPERIMENT_DEFAULT_STRATEGIES", "TOP5,TOP10,TOP20,BUY_OR_BETTER"))
        if str(part).strip()
    )
    return USWeightExperimentConfig(
        output_dir=output_dir,
        default_holding_days=holding_days or (5, 20, 60),
        default_strategies=strategies or ("TOP5", "TOP10", "TOP20", "BUY_OR_BETTER"),
        min_test_days=max(1, int(os.environ.get("US_WEIGHT_EXPERIMENT_MIN_TEST_DAYS", "30"))),
        baseline_weight_config_id=str(os.environ.get("US_WEIGHT_EXPERIMENT_BASELINE", "RULE_V1_BASELINE")).strip().upper() or "RULE_V1_BASELINE",
        log_level=str(os.environ.get("US_WEIGHT_EXPERIMENT_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def load_us_forward_test_config() -> USForwardTestConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(
        os.environ.get("US_FORWARD_TEST_OUTPUT_DIR", "outputs/us_stock_forward_test")
    ).strip() or "outputs/us_stock_forward_test"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    holding_days = tuple(
        int(part)
        for part in _split_csv(os.environ.get("US_FORWARD_TEST_HOLDING_DAYS", "5,20,60"))
        if str(part).strip()
    )
    strategies = tuple(
        str(part).strip().upper()
        for part in _split_csv(os.environ.get("US_FORWARD_TEST_STRATEGIES", "TOP5,TOP10,TOP20,BUY_OR_BETTER,STRONG_BUY"))
        if str(part).strip()
    )
    return USForwardTestConfig(
        enabled=_flag("US_FORWARD_TEST_ENABLED", "0"),
        forward_test_id=str(os.environ.get("US_FORWARD_TEST_ID", "US_RANK_FORWARD_RULE_V1")).strip() or "US_RANK_FORWARD_RULE_V1",
        holding_days=holding_days or (5, 20, 60),
        strategies=strategies or ("TOP5", "TOP10", "TOP20", "BUY_OR_BETTER", "STRONG_BUY"),
        output_dir=output_dir,
        auto_register=_flag("US_FORWARD_TEST_AUTO_REGISTER", "0"),
        auto_update=_flag("US_FORWARD_TEST_AUTO_UPDATE", "1"),
        auto_report=_flag("US_FORWARD_TEST_AUTO_REPORT", "0"),
        log_level=str(os.environ.get("US_FORWARD_TEST_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
    )


def resolve_us_paper_trading_config_path() -> Path:
    root_dir = Path(__file__).resolve().parents[2]
    raw = str(os.environ.get("US_PAPER_CONFIG_PATH", "config/us_stock_paper_trading.yaml")).strip() or "config/us_stock_paper_trading.yaml"
    path = Path(raw)
    return path if path.is_absolute() else root_dir / path


def load_us_paper_trading_profiles() -> dict[str, object]:
    path = resolve_us_paper_trading_config_path()
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = _load_simple_yaml_mapping(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Paper trading config must be a mapping: {path}")
    return payload


def load_us_paper_trading_config(account_id: str | None = None) -> USPaperTradingConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(os.environ.get("US_PAPER_OUTPUT_DIR", "outputs/us_stock_paper_trading")).strip() or "outputs/us_stock_paper_trading"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    report_output_dir_raw = str(os.environ.get("US_PAPER_REPORT_OUTPUT_DIR", str(output_dir))).strip() or str(output_dir)
    report_output_dir = Path(report_output_dir_raw)
    if not report_output_dir.is_absolute():
        report_output_dir = root_dir / report_output_dir
    config_path = resolve_us_paper_trading_config_path()
    profile_id = str(account_id or os.environ.get("US_PAPER_ACCOUNT_ID", "US_PAPER_RULE_V1")).strip() or "US_PAPER_RULE_V1"
    profiles = load_us_paper_trading_profiles()
    profile = profiles.get(profile_id, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    strategy = profile.get("strategy", {}) if isinstance(profile.get("strategy"), dict) else {}
    risk = profile.get("risk", {}) if isinstance(profile.get("risk"), dict) else {}
    execution = profile.get("execution", {}) if isinstance(profile.get("execution"), dict) else {}
    safety = profile.get("safety", {}) if isinstance(profile.get("safety"), dict) else {}
    return USPaperTradingConfig(
        enabled=_flag("US_PAPER_TRADING_ENABLED", "0"),
        account_id=profile_id,
        account_name=str(profile.get("account_name") or "미국주식 Rule v1 Paper Trading"),
        initial_cash=float(os.environ.get("US_PAPER_INITIAL_CASH", profile.get("initial_cash", 100000) or 100000)),
        base_currency=str(os.environ.get("US_PAPER_BASE_CURRENCY", profile.get("base_currency", "USD") or "USD")).strip().upper() or "USD",
        max_positions=int(os.environ.get("US_PAPER_MAX_POSITIONS", risk.get("max_positions", strategy.get("max_positions", 20)) or 20)),
        max_position_weight=float(os.environ.get("US_PAPER_MAX_POSITION_WEIGHT", risk.get("max_position_weight", 0.10) or 0.10)),
        max_sector_weight=float(os.environ.get("US_PAPER_MAX_SECTOR_WEIGHT", risk.get("max_sector_weight", 0.30) or 0.30)),
        min_cash_weight=float(os.environ.get("US_PAPER_MIN_CASH_WEIGHT", risk.get("min_cash_weight", 0.05) or 0.05)),
        max_daily_new_buys=int(os.environ.get("US_PAPER_MAX_DAILY_NEW_BUYS", risk.get("max_daily_new_buys", 5) or 5)),
        allow_fractional_shares=_flag("US_PAPER_ALLOW_FRACTIONAL_SHARES", "1" if bool(risk.get("allow_fractional_shares", True)) else "0"),
        commission_per_trade=float(os.environ.get("US_PAPER_COMMISSION_PER_TRADE", execution.get("commission_per_trade", 0) or 0)),
        slippage_bps=float(os.environ.get("US_PAPER_SLIPPAGE_BPS", execution.get("slippage_bps", 5) or 5)),
        real_order_blocked=_flag("US_PAPER_REAL_ORDER_BLOCKED", "1" if bool(safety.get("real_order_blocked", True)) else "0"),
        output_dir=output_dir,
        report_output_dir=report_output_dir,
        use_previous_close_if_missing=_flag("US_PAPER_USE_PREVIOUS_CLOSE_IF_MISSING", "1"),
        auto_snapshot_enabled=_flag("US_PAPER_AUTO_SNAPSHOT_ENABLED", "0"),
        auto_report_enabled=_flag("US_PAPER_AUTO_REPORT_ENABLED", "0"),
        log_level=str(os.environ.get("US_PAPER_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        config_path=config_path,
    )


def load_us_paper_trading_config(account_id: str | None = None) -> USPaperTradingConfig:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir_raw = str(os.environ.get("US_PAPER_OUTPUT_DIR", "outputs/us_stock_paper_trading")).strip() or "outputs/us_stock_paper_trading"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    report_output_dir_raw = str(os.environ.get("US_PAPER_REPORT_OUTPUT_DIR", str(output_dir))).strip() or str(output_dir)
    report_output_dir = Path(report_output_dir_raw)
    if not report_output_dir.is_absolute():
        report_output_dir = root_dir / report_output_dir
    config_path = resolve_us_paper_trading_config_path()
    profile_id = str(account_id or os.environ.get("US_PAPER_ACCOUNT_ID", "US_PAPER_RULE_V1")).strip() or "US_PAPER_RULE_V1"
    profiles = load_us_paper_trading_profiles()
    profile = profiles.get(profile_id, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    strategy = profile.get("strategy", {}) if isinstance(profile.get("strategy"), dict) else {}
    rebalance = profile.get("rebalance", {}) if isinstance(profile.get("rebalance"), dict) else {}
    risk = profile.get("risk", {}) if isinstance(profile.get("risk"), dict) else {}
    execution = profile.get("execution", {}) if isinstance(profile.get("execution"), dict) else {}
    safety = profile.get("safety", {}) if isinstance(profile.get("safety"), dict) else {}
    buy_grades_raw = strategy.get("buy_grades", ("STRONG_BUY", "BUY"))
    sell_grades_raw = strategy.get("sell_grades", ("HOLD", "EXCLUDE"))
    if isinstance(buy_grades_raw, str):
        buy_grades = tuple(part.strip().upper() for part in buy_grades_raw.split(",") if part.strip())
    else:
        buy_grades = tuple(str(part).strip().upper() for part in buy_grades_raw if str(part).strip())
    if isinstance(sell_grades_raw, str):
        sell_grades = tuple(part.strip().upper() for part in sell_grades_raw.split(",") if part.strip())
    else:
        sell_grades = tuple(str(part).strip().upper() for part in sell_grades_raw if str(part).strip())
    return USPaperTradingConfig(
        enabled=_flag("US_PAPER_TRADING_ENABLED", "0"),
        account_id=profile_id,
        account_name=str(profile.get("account_name") or "US Stock Rule v1 Paper Trading"),
        initial_cash=float(os.environ.get("US_PAPER_INITIAL_CASH", profile.get("initial_cash", 100000) or 100000)),
        base_currency=str(os.environ.get("US_PAPER_BASE_CURRENCY", profile.get("base_currency", "USD") or "USD")).strip().upper() or "USD",
        max_positions=int(os.environ.get("US_PAPER_MAX_POSITIONS", risk.get("max_positions", strategy.get("max_positions", 20)) or 20)),
        max_position_weight=float(os.environ.get("US_PAPER_MAX_POSITION_WEIGHT", risk.get("max_position_weight", 0.10) or 0.10)),
        max_sector_weight=float(os.environ.get("US_PAPER_MAX_SECTOR_WEIGHT", risk.get("max_sector_weight", 0.30) or 0.30)),
        min_cash_weight=float(os.environ.get("US_PAPER_MIN_CASH_WEIGHT", risk.get("min_cash_weight", 0.05) or 0.05)),
        max_daily_new_buys=int(os.environ.get("US_PAPER_MAX_DAILY_NEW_BUYS", risk.get("max_daily_new_buys", 5) or 5)),
        allow_fractional_shares=_flag("US_PAPER_ALLOW_FRACTIONAL_SHARES", "1" if bool(risk.get("allow_fractional_shares", True)) else "0"),
        buy_grades=buy_grades or ("STRONG_BUY", "BUY"),
        sell_grades=sell_grades or ("HOLD", "EXCLUDE"),
        max_rank_no=int(os.environ.get("US_PAPER_MAX_RANK_NO", strategy.get("max_rank_no", 20) or 20)),
        rebalance_enabled=_flag("US_PAPER_REBALANCE_ENABLED", "1" if bool(rebalance.get("enabled", False)) else "0"),
        rebalance_frequency=str(os.environ.get("US_PAPER_REBALANCE_FREQUENCY", rebalance.get("frequency", "DAILY") or "DAILY")).strip().upper() or "DAILY",
        rebalance_sell_first=_flag("US_PAPER_REBALANCE_SELL_FIRST", "1" if bool(rebalance.get("sell_first", True)) else "0"),
        rebalance_allow_rebuy_same_day=_flag("US_PAPER_REBALANCE_ALLOW_REBUY_SAME_DAY", "1" if bool(rebalance.get("allow_rebuy_same_day", False)) else "0"),
        rebalance_min_amount=float(os.environ.get("US_PAPER_REBALANCE_MIN_AMOUNT", rebalance.get("min_rebalance_amount", 100) or 100)),
        rebalance_min_weight_diff=float(os.environ.get("US_PAPER_REBALANCE_MIN_WEIGHT_DIFF", rebalance.get("min_weight_diff", 0.02) or 0.02)),
        rebalance_full_sell_on_rank_exit=_flag("US_PAPER_REBALANCE_FULL_SELL_ON_RANK_EXIT", "1" if bool(rebalance.get("full_sell_on_rank_exit", True)) else "0"),
        rebalance_full_sell_on_grade_downgrade=_flag("US_PAPER_REBALANCE_FULL_SELL_ON_GRADE_DOWNGRADE", "1" if bool(rebalance.get("full_sell_on_grade_downgrade", True)) else "0"),
        commission_per_trade=float(os.environ.get("US_PAPER_COMMISSION_PER_TRADE", execution.get("commission_per_trade", 0) or 0)),
        slippage_bps=float(os.environ.get("US_PAPER_SLIPPAGE_BPS", execution.get("slippage_bps", 5) or 5)),
        real_order_blocked=_flag("US_PAPER_REAL_ORDER_BLOCKED", "1" if bool(safety.get("real_order_blocked", True)) else "0"),
        output_dir=output_dir,
        report_output_dir=report_output_dir,
        use_previous_close_if_missing=_flag("US_PAPER_USE_PREVIOUS_CLOSE_IF_MISSING", "1"),
        auto_snapshot_enabled=_flag("US_PAPER_AUTO_SNAPSHOT_ENABLED", "0"),
        auto_report_enabled=_flag("US_PAPER_AUTO_REPORT_ENABLED", "0"),
        validation_enabled=_flag("US_PAPER_VALIDATION_ENABLED", "1"),
        validation_fail_on_error=_flag("US_PAPER_VALIDATION_FAIL_ON_ERROR", "0"),
        scheduler_enabled=_flag("US_PAPER_SCHEDULER_ENABLED", "0"),
        scheduler_account_id=str(os.environ.get("US_PAPER_SCHEDULER_ACCOUNT_ID", profile_id)).strip() or profile_id,
        scheduler_run_rebalance_plan=_flag("US_PAPER_SCHEDULER_RUN_REBALANCE_PLAN", "1"),
        scheduler_generate_orders=_flag("US_PAPER_SCHEDULER_GENERATE_ORDERS", "0"),
        scheduler_simulate_fills=_flag("US_PAPER_SCHEDULER_SIMULATE_FILLS", "0"),
        scheduler_update_snapshot=_flag("US_PAPER_SCHEDULER_UPDATE_SNAPSHOT", "0"),
        scheduler_validate=_flag("US_PAPER_SCHEDULER_VALIDATE", "1"),
        scheduler_report=_flag("US_PAPER_SCHEDULER_REPORT", "1"),
        log_level=str(os.environ.get("US_PAPER_LOG_LEVEL", "INFO")).strip().upper() or "INFO",
        config_path=config_path,
    )


def resolve_universe_csv_path(universe_tag: str, override: str | None = None) -> Path:
    cfg = load_us_stock_config()
    if override:
        path = Path(override)
        return path if path.is_absolute() else cfg.root_dir / path
    return cfg.us_data_dir / f"{str(universe_tag).strip().lower()}_universe.csv"


def parse_iso_date(value: str | None, *, field_name: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: '{text}'. Expected YYYY-MM-DD.") from exc


def resolve_price_window(
    *,
    cfg: USStockConfig,
    cli_start_date: str | None = None,
    cli_end_date: str | None = None,
    backfill: bool = False,
    today: date | None = None,
) -> tuple[date, date]:
    base_today = today or date.today()
    start_date = parse_iso_date(cli_start_date, field_name="start_date") or parse_iso_date(
        cfg.price_start_date, field_name="US_STOCK_PRICE_START_DATE"
    )
    end_date = parse_iso_date(cli_end_date, field_name="end_date") or parse_iso_date(
        cfg.price_end_date, field_name="US_STOCK_PRICE_END_DATE"
    )
    if end_date is None:
        end_date = base_today

    if start_date is None:
        if backfill:
            start_date = end_date - timedelta(days=max(1, cfg.price_backfill_years) * 366)
        else:
            start_date = end_date

    if start_date > end_date:
        raise ValueError(
            f"Invalid date window: start_date {start_date.isoformat()} is after end_date {end_date.isoformat()}."
        )
    return start_date, end_date
