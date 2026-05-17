from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from python.us.build_us_features import FeatureBuildResult, build_us_features
from python.us.build_us_macro_features import build_us_macro_features
from python.us.build_us_sector_features import build_us_sector_features
from python.us.buy_automation.scheduler_job import run_buy_scheduler_job
from python.us.download_us_prices import PriceCollectResult, collect_us_prices
from python.us.load_us_universe import UniverseLoadResult, load_universe
from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.trade_orchestration.scheduler_job import run_trade_scheduler_job
from python.us.us_config import load_us_stock_config, parse_iso_date, resolve_universe_csv_path
from python.us.validate_us_price_data import ValidationResult, validate_and_save_quality_report


LOGGER = logging.getLogger("us_pipeline")


@dataclass(frozen=True)
class PipelineResult:
    final_status: str
    universe_result: UniverseLoadResult | None
    price_result: PriceCollectResult | None
    quality_result: ValidationResult | None
    feature_result: FeatureBuildResult | None
    buy_scheduler_result: dict[str, object] | None
    trade_scheduler_result: dict[str, object] | None
    elapsed_sec: float
    mode: str


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    cfg = load_us_stock_config()
    parser = argparse.ArgumentParser(description="Run the standalone Project C US daily pipeline.")
    parser.add_argument("--universe", default=cfg.universe, help="Universe tag. Default: env US_STOCK_UNIVERSE.")
    parser.add_argument("--force", action="store_true", help="Allow execution even when US_STOCK_ENABLED=false.")
    parser.add_argument("--backfill", action="store_true", help="Run price collection in backfill mode.")
    parser.add_argument("--incremental", action="store_true", help="Run price collection in incremental mode.")
    parser.add_argument("--start-date", default=None, help="Optional custom pipeline start date. Format: YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional custom pipeline end date. Format: YYYY-MM-DD.")
    parser.add_argument("--skip-universe", action="store_true", help="Skip universe load step.")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price collection step.")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality validation step.")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature build step.")
    parser.add_argument("--force-features", action="store_true", help="Build features even if quality status is FAILED.")
    parser.add_argument("--limit", type=int, default=None, help="Limit ticker count for testing.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed stage logs.")
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.backfill and args.incremental:
        raise ValueError("Choose only one mode: --backfill or --incremental.")
    if args.incremental:
        if args.start_date or args.end_date:
            raise ValueError("--incremental cannot be combined with --start-date or --end-date.")
        return "incremental"
    if args.backfill:
        return "backfill"
    if args.start_date or args.end_date:
        return "custom"
    return "custom"


def _pipeline_status_from_quality(status: str) -> str:
    if status == "OK":
        return "SUCCESS"
    if status == "WARN":
        return "WARN"
    return "FAILED"


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    cfg = load_us_stock_config()
    universe_tag = str(args.universe or cfg.universe).strip().upper() or "NASDAQ100"
    mode = _resolve_mode(args)
    start_ts = time.perf_counter()

    if not cfg.enabled and not args.force:
        LOGGER.info("[US_PIPELINE] Start Project C Phase 1 pipeline universe=%s", universe_tag)
        LOGGER.info("[US_PIPELINE] US_STOCK_ENABLED=false and --force not provided. Skip pipeline.")
        elapsed = time.perf_counter() - start_ts
        LOGGER.info("[US_PIPELINE] Finished status=SKIPPED elapsed_sec=%.2f", elapsed)
        return PipelineResult(
            final_status="SKIPPED",
            universe_result=None,
            price_result=None,
            quality_result=None,
            feature_result=None,
            buy_scheduler_result=None,
            trade_scheduler_result=None,
            elapsed_sec=elapsed,
            mode=mode,
        )

    LOGGER.info("[US_PIPELINE] Start Project C Phase 1 pipeline universe=%s", universe_tag)
    if not cfg.enabled and args.force:
        LOGGER.info("[US_PIPELINE] US_STOCK_ENABLED=false but --force provided. Continue.")

    universe_result: UniverseLoadResult | None = None
    price_result: PriceCollectResult | None = None
    quality_result: ValidationResult | None = None
    feature_result: FeatureBuildResult | None = None
    buy_scheduler_result: dict[str, object] | None = None
    trade_scheduler_result: dict[str, object] | None = None
    final_status = "SUCCESS"

    as_of_date = parse_iso_date(args.end_date, field_name="end_date") or date.today()

    if args.skip_universe:
        LOGGER.info("[US_PIPELINE] Step 1/4 Load Universe skipped")
    else:
        LOGGER.info("[US_PIPELINE] Step 1/4 Load Universe started")
        csv_path = resolve_universe_csv_path(universe_tag, None)
        universe_result = load_universe(
            universe_tag=universe_tag,
            csv_path=csv_path,
            load_date=as_of_date,
            deactivate_missing=False,
            dry_run=False,
            data_source="static_csv",
        )
        LOGGER.info(
            "[US_PIPELINE] Step 1/4 Load Universe completed inserted=%s updated=%s",
            universe_result.inserted,
            universe_result.updated,
        )

    if args.skip_prices:
        LOGGER.info("[US_PIPELINE] Step 2/4 Collect Prices skipped")
    else:
        LOGGER.info("[US_PIPELINE] Step 2/4 Collect Prices started mode=%s", mode)
        price_result = collect_us_prices(
            universe_tag=universe_tag,
            mode="incremental" if mode == "incremental" else ("backfill" if mode == "backfill" else "date_range"),
            limit=args.limit,
            start_date_text=args.start_date,
            end_date_text=args.end_date,
        )
        LOGGER.info(
            "[US_PIPELINE] Step 2/4 Collect Prices completed success=%s failed=%s skipped=%s",
            price_result.success_count,
            price_result.failed_count,
            price_result.skipped_count,
        )
        if price_result.active_ticker_count == 0 or (price_result.success_count == 0 and price_result.total_rows == 0):
            final_status = "FAILED"

    if args.skip_quality:
        LOGGER.info("[US_PIPELINE] Step 3/4 Validate Quality skipped")
    else:
        LOGGER.info("[US_PIPELINE] Step 3/4 Validate Quality started")
        quality_result = validate_and_save_quality_report(
            universe_tag=universe_tag,
            as_of_date=as_of_date,
            verbose=bool(args.verbose),
        )
        LOGGER.info(
            "[US_PIPELINE] Step 3/4 Validate Quality completed status=%s missing=%s stale=%s",
            quality_result.quality_status,
            quality_result.price_missing_count,
            quality_result.stale_ticker_count,
        )
        quality_pipeline_status = _pipeline_status_from_quality(quality_result.quality_status)
        if quality_pipeline_status == "FAILED":
            final_status = "FAILED"
        elif quality_pipeline_status == "WARN" and final_status != "FAILED":
            final_status = "WARN"

    should_run_features = not args.skip_features
    if (
        not args.skip_features
        and quality_result is not None
        and quality_result.quality_status == "FAILED"
        and not args.force_features
    ):
        should_run_features = False
        LOGGER.info("[US_PIPELINE] Step 4/4 Build Features skipped due to quality_status=FAILED")

    if args.skip_features:
        LOGGER.info("[US_PIPELINE] Step 4/4 Build Features skipped")
    elif should_run_features:
        LOGGER.info("[US_PIPELINE] Step 4/4 Build Features started")
        feature_result = build_us_features(
            universe_tag=universe_tag,
            start_date=parse_iso_date(args.start_date, field_name="start_date"),
            end_date=parse_iso_date(args.end_date, field_name="end_date"),
            limit=args.limit,
            verbose=bool(args.verbose),
        )
        LOGGER.info(
            "[US_PIPELINE] Step 4/4 Build Features completed success=%s skipped=%s failed=%s",
            feature_result.success_count,
            feature_result.skipped_count,
            feature_result.failed_count,
        )
        if feature_result.failed_count > 0 and final_status != "FAILED":
            final_status = "WARN"

    # Macro features (VIX/SPY/QQQ regime context) — lightweight, always runs after feature build
    if should_run_features and not args.skip_features:
        LOGGER.info("[US_PIPELINE] Step 4b/4 Build Macro Features started")
        try:
            macro_rows = build_us_macro_features(
                end_date=parse_iso_date(args.end_date, field_name="end_date"),
            )
            LOGGER.info("[US_PIPELINE] Step 4b/4 Build Macro Features completed rows=%s", macro_rows)
        except Exception as exc:
            LOGGER.warning("[US_PIPELINE] Step 4b/4 Build Macro Features failed (non-fatal): %s", exc)

    # Sector relative strength — runs after feature build, batch across all tickers
    if should_run_features and not args.skip_features:
        LOGGER.info("[US_PIPELINE] Step 4c/4 Build Sector Features started")
        try:
            sector_rows = build_us_sector_features(
                universe_tag=universe_tag,
                end_date=parse_iso_date(args.end_date, field_name="end_date"),
            )
            LOGGER.info("[US_PIPELINE] Step 4c/4 Build Sector Features completed rows=%s", sector_rows)
        except Exception as exc:
            LOGGER.warning("[US_PIPELINE] Step 4c/4 Build Sector Features failed (non-fatal): %s", exc)

    trade_cfg = load_trade_orchestration_config()
    should_run_trade_scheduler = trade_cfg.scheduler_enabled
    disable_buy_scheduler = (
        should_run_trade_scheduler
        and trade_cfg.disable_buy_only_scheduler_when_orchestration
    )

    # US Trade Orchestration must run after ranking/score generation.
    # This stage is SHADOW/PAPER only. Live order execution is prohibited.
    if should_run_trade_scheduler:
        LOGGER.info("[US_PIPELINE] Step 5/6 US Trade Scheduler Job started")
        trade_scheduler_result = run_trade_scheduler_job(
            trade_date=args.end_date or str(as_of_date),
            emit_console=False,
        )
        if trade_scheduler_result.get("success"):
            LOGGER.info(
                "[US_PIPELINE] Step 5/6 US Trade Scheduler Job completed sell=%s buy_candidates=%s conflict_blocked=%s final_buy=%s",
                (trade_scheduler_result.get("summary") or {}).get("sell_signals", 0),
                (trade_scheduler_result.get("summary") or {}).get("buy_candidates", 0),
                (trade_scheduler_result.get("summary") or {}).get("conflict_blocked", 0),
                (trade_scheduler_result.get("summary") or {}).get("final_buy_candidates", 0),
            )
        else:
            LOGGER.info(
                "[US_PIPELINE] Step 5/6 US Trade Scheduler Job completed with errors=%s pipeline_should_fail=%s",
                ",".join(trade_scheduler_result.get("errors") or []),
                trade_scheduler_result.get("pipeline_should_fail"),
            )
            if trade_scheduler_result.get("pipeline_should_fail"):
                final_status = "FAILED"
            elif final_status == "SUCCESS":
                final_status = "WARN"
    else:
        LOGGER.info("[US_PIPELINE] Step 5/6 US Trade Scheduler Job skipped")

    if disable_buy_scheduler:
        LOGGER.info("[US_PIPELINE] Step 6/6 US BUY Scheduler Job skipped reason=ORCHESTRATION_ENABLED")
    else:
        LOGGER.info("[US_PIPELINE] Step 6/6 US BUY Scheduler Job started")
        buy_scheduler_result = run_buy_scheduler_job(
            trade_date=args.end_date or str(as_of_date),
            emit_console=False,
        )
        if not buy_scheduler_result.get("enabled"):
            LOGGER.info(
                "[US_PIPELINE] Step 6/6 US BUY Scheduler Job skipped error=%s",
                buy_scheduler_result.get("error"),
            )
        elif buy_scheduler_result.get("success"):
            LOGGER.info(
                "[US_PIPELINE] Step 6/6 US BUY Scheduler Job completed candidates=%s allowed=%s blocked=%s paper_orders=%s",
                (buy_scheduler_result.get("summary") or {}).get("loaded_candidates", 0),
                (buy_scheduler_result.get("summary") or {}).get("allowed_candidates", 0),
                (buy_scheduler_result.get("summary") or {}).get("blocked_candidates", 0),
                (buy_scheduler_result.get("summary") or {}).get("paper_orders", 0),
            )
        else:
            LOGGER.info(
                "[US_PIPELINE] Step 6/6 US BUY Scheduler Job completed with error=%s pipeline_should_fail=%s",
                buy_scheduler_result.get("error"),
                buy_scheduler_result.get("pipeline_should_fail"),
            )
            if buy_scheduler_result.get("pipeline_should_fail"):
                final_status = "FAILED"
            elif final_status == "SUCCESS":
                final_status = "WARN"

    if price_result is not None and price_result.failed_count > 0 and final_status == "SUCCESS":
        final_status = "WARN"

    elapsed = time.perf_counter() - start_ts
    LOGGER.info("[US_PIPELINE] Summary universe=%s", universe_tag)
    LOGGER.info("[US_PIPELINE] Summary data_source=%s", cfg.data_source)
    LOGGER.info("[US_PIPELINE] Summary mode=%s", mode)
    LOGGER.info(
        "[US_PIPELINE] Summary universe_load=%s",
        "skipped" if universe_result is None and args.skip_universe else (
            f"inserted={universe_result.inserted} updated={universe_result.updated}" if universe_result else "not_run"
        ),
    )
    LOGGER.info(
        "[US_PIPELINE] Summary price_collect=%s",
        "skipped" if price_result is None and args.skip_prices else (
            f"success={price_result.success_count} failed={price_result.failed_count} skipped={price_result.skipped_count}"
            if price_result
            else "not_run"
        ),
    )
    LOGGER.info(
        "[US_PIPELINE] Summary quality=%s",
        "skipped" if quality_result is None and args.skip_quality else (quality_result.quality_status if quality_result else "not_run"),
    )
    LOGGER.info(
        "[US_PIPELINE] Summary features=%s",
        "skipped"
        if feature_result is None and (args.skip_features or (quality_result is not None and quality_result.quality_status == "FAILED" and not args.force_features))
        else (
            f"success={feature_result.success_count} skipped={feature_result.skipped_count} failed={feature_result.failed_count}"
            if feature_result
            else "not_run"
        ),
    )
    LOGGER.info(
        "[US_PIPELINE] Summary trade_scheduler=%s",
        "not_run"
        if trade_scheduler_result is None
        else (
            f"success={trade_scheduler_result.get('success')} "
            f"errors={','.join(trade_scheduler_result.get('errors') or [])} "
            f"buy_candidates={(trade_scheduler_result.get('summary') or {}).get('buy_candidates', 0)} "
            f"conflict_blocked={(trade_scheduler_result.get('summary') or {}).get('conflict_blocked', 0)}"
        ),
    )
    LOGGER.info(
        "[US_PIPELINE] Summary buy_scheduler=%s",
        "not_run"
        if buy_scheduler_result is None
        else (
            f"success={buy_scheduler_result.get('success')} error={buy_scheduler_result.get('error')} "
            f"candidates={(buy_scheduler_result.get('summary') or {}).get('loaded_candidates', 0)} "
            f"allowed={(buy_scheduler_result.get('summary') or {}).get('allowed_candidates', 0)} "
            f"blocked={(buy_scheduler_result.get('summary') or {}).get('blocked_candidates', 0)}"
        ),
    )
    LOGGER.info("[US_PIPELINE] Summary final_status=%s", final_status)
    LOGGER.info("[US_PIPELINE] Summary elapsed_sec=%.2f", elapsed)
    LOGGER.info("[US_PIPELINE] Summary Korean auto-trading impact=none")
    LOGGER.info("[US_PIPELINE] Finished status=%s elapsed_sec=%.2f", final_status, elapsed)

    return PipelineResult(
        final_status=final_status,
        universe_result=universe_result,
        price_result=price_result,
        quality_result=quality_result,
        feature_result=feature_result,
        buy_scheduler_result=buy_scheduler_result,
        trade_scheduler_result=trade_scheduler_result,
        elapsed_sec=elapsed,
        mode=mode,
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        result = run_pipeline(args)
    except KeyboardInterrupt:
        LOGGER.info("[US_PIPELINE] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.info("[US_PIPELINE] Failed error=%s", exc)
        raise SystemExit(1) from exc

    if result.final_status == "SKIPPED":
        raise SystemExit(0)
    if result.final_status == "FAILED":
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
