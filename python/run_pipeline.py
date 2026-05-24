# python/run_pipeline.py
import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import text

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from performance_profiling import log_profile_result, profile_begin
from production_config import get_production_config_value, is_operational_runtime_mode

try:
    from db import create_research_model_run, get_engine, log_pipeline_history
except Exception:
    get_engine = None
    log_pipeline_history = None
    create_research_model_run = None

DATA_DIR = Path("data")
ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = DATA_DIR / "lee_trader.db"
OUTPUT_DIR = Path("output")
FUNDAMENTALS_CSV = DATA_DIR / "fundamentals.csv"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
THEME_STOCK_DAILY_CSV = OUTPUT_DIR / "stock_theme_daily.csv"
THEME_STOCK_DAILY_BACKUP_CSV = OUTPUT_DIR / "stock_theme_daily.pipeline_backup.csv"
THEME_MODE_NOTE_MD = DATA_DIR / "theme_pipeline_mode_note.md"
ARCHIVE_SNAPSHOT_SCRIPT = Path("python") / "archive_ranking_snapshot.py"
SNAPSHOT_INVENTORY_SCRIPT = Path("python") / "report_ranking_snapshot_inventory.py"
CONFIDENCE_CALIBRATION_SCRIPT = Path("python") / "build_confidence_calibration_map.py"
TOP20_MEANINGFULNESS_SCRIPT = Path("python") / "analyze_top20_meaningfulness.py"
CONFIDENCE_V2_SCRIPT = Path("python") / "build_confidence_score_v2.py"
TOP20_BUYABILITY_SCRIPT = Path("python") / "build_top20_buyability_report.py"
WALKFORWARD_ACCEPTANCE_SCRIPT = Path("python") / "build_walkforward_acceptance.py"
CSV_DB_PARITY_SCRIPT = Path("python") / "sync_csv_db_parity.py"

# Daily pipeline order:
# market/universe -> prices/fundamentals -> features/labels ->
# model training -> prediction -> final ranking
STEPS: List[Tuple[str, str]] = [
    ("fetch_market_data", "python/fetch_market_data.py"),
    ("fetch_top_universe", "python/fetch_top_universe.py"),
    ("merge_universe", "python/merge_universe.py"),
    ("download_prices_kis", "python/download_prices_kis.py"),
    ("clean_prices", "python/clean_prices.py"),
    ("create_adjusted_prices", "python/create_adjusted_prices.py"),
    ("fetch_fundamentals_dart", "python/fetch_fundamentals_dart.py"),
    ("quality_builder", "python/quality_builder.py"),
    ("feature_builder", "python/feature_builder.py"),
    # Labels feed the supervised training targets.
    ("label_builder", "python/label_builder.py"),
    # Model training/prediction produces the inputs consumed by ranking_builder.
    ("model_train", "python/model_train.py"),
    ("model_predict", "python/model_predict.py"),
    # Archive today's predictions for future accuracy evaluation (60d/90d later).
    ("archive_predictions", "python/archive_predictions.py"),
    # Final single scoring/ranking step.
    ("ranking_builder", "python/ranking_builder.py"),
]

CORE_CODES = ["005930", "000660", "035420"]
DEFAULT_MARKET_TIMEZONE = "Asia/Seoul"
DEFAULT_MARKET_DATE_CUTOFF = "18:10"
PIPELINE_ADVISORY_LOCK_KEY = 2026050801

THEME_OVERLAY_OFF = "off"
THEME_OVERLAY_RESEARCH = "research"
THEME_OVERLAY_OPERATIONAL = "operational"
THEME_OVERLAY_ALLOWED_MODES = {
    THEME_OVERLAY_OFF,
    THEME_OVERLAY_RESEARCH,
    THEME_OVERLAY_OPERATIONAL,
}
PROFILE_STEP_GROUP_BY_STEP = {
    "fetch_market_data": "data_collection",
    "fetch_top_universe": "data_collection",
    "merge_universe": "data_collection",
    "download_prices_kis": "data_collection",
    "clean_prices": "data_collection",
    "create_adjusted_prices": "data_collection",
    "fetch_fundamentals_dart": "data_collection",
    "quality_builder": "quality_build",
    "feature_builder": "feature_build",
    "model_train": "model_train",
    "model_predict": "prediction",
    "ranking_builder": "ranking_build",
}
PROFILE_STEP_GROUP_LAST_STEP = {
    "data_collection": "fetch_fundamentals_dart",
    "quality_build": "quality_builder",
    "feature_build": "feature_builder",
    "model_train": "model_train",
    "prediction": "model_predict",
    "ranking_build": "ranking_builder",
}


def _env_model_version() -> str:
    return os.environ.get("MODEL_VERSION", "v1")


def _env_horizon_days() -> int:
    return int(os.environ.get("HORIZON_DAYS", "60"))


def _env_top_n() -> int:
    return int(os.environ.get("TOP_N", "20"))


def _env_score_formula_version() -> str:
    configured = str(get_production_config_value(["metadata", "score_formula_version"], "ranking_builder_v1")).strip()
    if is_operational_runtime_mode():
        return configured or "ranking_builder_v1"
    return os.environ.get("SCORE_FORMULA_VERSION", configured or "ranking_builder_v1")


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_market_date_override() -> date | None:
    raw = str(os.environ.get("MARKET_DATE", "")).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    logging.warning("Invalid MARKET_DATE format in run_pipeline: %s", raw)
    return None


def _parse_daily_time(raw: str) -> tuple[int, int]:
    text = str(raw or DEFAULT_MARKET_DATE_CUTOFF).strip() or DEFAULT_MARKET_DATE_CUTOFF
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception:
        logging.warning("Invalid cutoff time '%s' -> fallback to %s", text, DEFAULT_MARKET_DATE_CUTOFF)
        return 16, 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        logging.warning("Out-of-range cutoff time '%s' -> fallback to %s", text, DEFAULT_MARKET_DATE_CUTOFF)
        return 16, 0
    return hour, minute


def _fallback_previous_business_day(base_date: date) -> date:
    try:
        prior = pd.bdate_range(end=pd.Timestamp(base_date) - pd.Timedelta(days=1), periods=1)
        if len(prior) > 0:
            return prior[-1].date()
    except Exception:
        pass
    return base_date - timedelta(days=1)


def _resolve_pipeline_market_date() -> date:
    override = _parse_market_date_override()
    if override is not None:
        return override

    tz_name = str(os.environ.get("MARKET_TIMEZONE", DEFAULT_MARKET_TIMEZONE)).strip() or DEFAULT_MARKET_TIMEZONE
    try:
        now = datetime.now(ZoneInfo(tz_name))
    except Exception:
        logging.warning("Invalid MARKET_TIMEZONE '%s' -> fallback to local time", tz_name)
        now = datetime.now()
    cutoff_hour, cutoff_minute = _parse_daily_time(
        os.environ.get("MARKET_DATE_CUTOFF") or os.environ.get("SCHEDULER_DAILY_TIME", DEFAULT_MARKET_DATE_CUTOFF)
    )

    try:
        from fetch_market_data import _resolve_last_completed_trading_day, _resolve_last_trading_day

        latest_visible = _resolve_last_trading_day().date()
        if latest_visible < now.date():
            return latest_visible
        if (now.hour, now.minute) >= (cutoff_hour, cutoff_minute):
            return latest_visible
        return _resolve_last_completed_trading_day().date()
    except Exception as exc:
        logging.warning("Failed to resolve trading date from market sources -> fallback to clock-based rule: %s", exc)

    if (now.hour, now.minute) >= (cutoff_hour, cutoff_minute):
        return now.date()
    return _fallback_previous_business_day(now.date())


def _effective_market_date() -> date:
    return _parse_market_date_override() or _resolve_pipeline_market_date()


def _env_theme_overlay_mode() -> str:
    configured = str(get_production_config_value(["ranking", "theme_overlay", "mode"], THEME_OVERLAY_OFF)).strip().lower() or THEME_OVERLAY_OFF
    raw = str(os.environ.get("THEME_OVERLAY_MODE", configured)).strip().lower()
    if raw not in THEME_OVERLAY_ALLOWED_MODES:
        logging.warning("Invalid THEME_OVERLAY_MODE='%s' -> fallback to '%s'", raw, THEME_OVERLAY_OFF)
        return THEME_OVERLAY_OFF
    return raw


def _theme_config() -> dict[str, object]:
    enabled_default = "1" if bool(get_production_config_value(["ranking", "theme_overlay", "enabled"], False)) else "0"
    validation_default = "1" if bool(get_production_config_value(["ranking", "theme_overlay", "validation_enabled"], False)) else "0"
    enabled = _env_flag("ENABLE_THEME_OVERLAY", enabled_default)
    validation_enabled = _env_flag("ENABLE_THEME_VALIDATION", validation_default)
    mode = _env_theme_overlay_mode()
    if not enabled and mode != THEME_OVERLAY_OFF:
        logging.info(
            "ENABLE_THEME_OVERLAY is disabled, so THEME_OVERLAY_MODE='%s' will behave as '%s'",
            mode,
            THEME_OVERLAY_OFF,
        )
        mode = THEME_OVERLAY_OFF
    ranking_enabled = enabled and mode == THEME_OVERLAY_OPERATIONAL
    research_enabled = enabled and mode == THEME_OVERLAY_RESEARCH
    return {
        "overlay_enabled": enabled,
        "validation_enabled": validation_enabled,
        "mode": mode,
        "ranking_enabled": ranking_enabled,
        "research_enabled": research_enabled,
    }


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def load_runtime_env() -> None:
    if load_dotenv is None:
        logging.info("python-dotenv unavailable -> keep existing process environment")
        return
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        logging.info(".env not found at %s -> keep existing process environment", env_path)
        return
    load_dotenv(dotenv_path=env_path, override=True)
    logging.info("Loaded runtime .env from %s", env_path)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily Lee_trader pipeline.")
    parser.add_argument(
        "--continue-on-archive-error",
        action="store_true",
        help="Continue pipeline even if ranking snapshot archive fails.",
    )
    parser.add_argument(
        "--archive-as-of-date",
        default=None,
        help="Optional as_of_date passed to archive_ranking_snapshot.py when CSV date resolution is unavailable.",
    )
    return parser.parse_args()


@contextmanager
def acquire_pipeline_run_lock():
    if not get_engine:
        yield
        return

    eng = get_engine()
    conn = eng.connect()
    acquired = False
    try:
        acquired = bool(
            conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": PIPELINE_ADVISORY_LOCK_KEY},
            ).scalar()
        )
        if not acquired:
            raise RuntimeError(
                "Another run_pipeline instance is already running. "
                "Check for duplicate scheduler or task triggers."
            )
        logging.info("Acquired run_pipeline advisory lock key=%s", PIPELINE_ADVISORY_LOCK_KEY)
        yield
    finally:
        if acquired:
            try:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": PIPELINE_ADVISORY_LOCK_KEY},
                )
                logging.info("Released run_pipeline advisory lock key=%s", PIPELINE_ADVISORY_LOCK_KEY)
            except Exception:
                logging.warning("Failed to release run_pipeline advisory lock", exc_info=True)
        conn.close()


def create_model_run_id() -> str:
    """
    Insert a row into research.dim_model_run and return numeric run_id as text.
    """
    if not get_engine or not create_research_model_run:
        raise RuntimeError("run_id creation requires DB engine and create_research_model_run")

    run_type = os.environ.get("RUN_TYPE", "daily_pipeline")
    model_version = _env_model_version()
    horizon_days = _env_horizon_days()
    top_n = _env_top_n()
    config_json = {}
    config_json["score_formula_version"] = _env_score_formula_version()
    if os.environ.get("SCORE_WEIGHTS_JSON"):
        try:
            config_json["score_weights"] = json.loads(os.environ["SCORE_WEIGHTS_JSON"])
        except Exception:
            logging.warning("Invalid SCORE_WEIGHTS_JSON; storing empty config_json")
            config_json["score_weights"] = {}

    try:
        run_id_db = create_research_model_run(
            run_type=run_type,
            model_version=model_version,
            horizon_days=horizon_days,
            top_n=top_n,
            config_json=config_json if config_json else None,
        )
        return str(int(run_id_db))
    except Exception:
        logging.exception("create_model_run_id failed; research history append cannot continue")
        raise


def _require_numeric_run_id(run_id: str, *, context: str) -> int:
    run_id_text = str(run_id).strip()
    if not run_id_text:
        raise ValueError(f"{context}: run_id is empty")
    if not run_id_text.isdigit():
        raise ValueError(f"{context}: run_id must be numeric, got '{run_id_text}'")
    return int(run_id_text)


def _log_run_id_event(run_id: str, message: str) -> None:
    logging.info("%s (run_id=%s)", message, run_id)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def write_theme_pipeline_mode_note(theme_cfg: dict[str, object]) -> None:
    lines = [
        "# Theme Pipeline Mode Note",
        "",
        "## Modes",
        "- `off`: baseline pipeline only. Theme CSVs are not connected to ranking_builder.",
        "- `research`: theme build steps may run, but ranking_builder stays baseline-only.",
        "- `operational`: theme build steps run and ranking_builder is allowed to consume theme overlay inputs.",
        "",
        "## Environment Variables",
        "- `ENABLE_THEME_OVERLAY`: theme build steps master switch.",
        "- `ENABLE_THEME_VALIDATION`: optional validation runner switch.",
        "- `THEME_OVERLAY_MODE=off|research|operational`: selects whether theme outputs stay detached or are connected to ranking.",
        "",
        "## Current Effective Config",
        f"- overlay_enabled: {bool(theme_cfg.get('overlay_enabled'))}",
        f"- validation_enabled: {bool(theme_cfg.get('validation_enabled'))}",
        f"- mode: {theme_cfg.get('mode')}",
        f"- ranking_enabled: {bool(theme_cfg.get('ranking_enabled'))}",
        f"- research_enabled: {bool(theme_cfg.get('research_enabled'))}",
        "",
        "## Safety Rules",
        "- Optional theme steps are wrapped so baseline ranking generation continues after failure.",
        "- When theme overlay is not operational, run_pipeline temporarily disconnects `output/stock_theme_daily.csv` from ranking_builder to avoid stale theme influence.",
    ]
    THEME_MODE_NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_schema_if_available() -> None:
    """
    Apply schema.sql only when the target database looks uninitialized.
    Useful in environments without shell access (e.g., Render).
    """
    if not get_engine:
        logging.info("No DB engine available -> skip schema apply")
        return

    schema_path = Path("schema.sql")
    if not schema_path.exists():
        logging.info("schema.sql not found -> skip schema apply")
        return

    try:
        eng = get_engine()
        with eng.begin() as conn:
            existing_research_schema = conn.exec_driver_sql(
                "select 1 from information_schema.schemata where schema_name = 'research' limit 1"
            ).scalar()
            if existing_research_schema:
                logging.info("research schema already exists -> skip schema.sql apply")
                return

            sql_text = schema_path.read_text(encoding="utf-8-sig")
            conn.exec_driver_sql(sql_text)
        logging.info("schema.sql applied successfully")
    except Exception:
        logging.exception("Failed to apply schema.sql (skipped)")


def run_step(name: str, script: str, *, run_id: str) -> float:
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for step {name}: {script_path}")

    start_ts = time.perf_counter()
    _log_run_id_event(run_id, f"==> Running step: {name} ({script_path})")
    env = os.environ.copy()
    env["RUN_ID"] = str(run_id)
    result = subprocess.run([sys.executable, str(script_path)], check=True, env=env, cwd=ROOT_DIR)
    elapsed = time.perf_counter() - start_ts
    if result.returncode != 0:
        raise RuntimeError(f"Step {name} failed with return code {result.returncode}")
    _log_run_id_event(run_id, f"<== Step completed: {name} (elapsed={elapsed:.2f}s)")
    return elapsed


def should_skip_fundamentals_step() -> bool:
    if not FUNDAMENTALS_CSV.exists():
        return False
    fundamentals_mtime = datetime.fromtimestamp(FUNDAMENTALS_CSV.stat().st_mtime)
    if fundamentals_mtime.date() != date.today():
        return False
    if UNIVERSE_CSV.exists():
        universe_mtime = datetime.fromtimestamp(UNIVERSE_CSV.stat().st_mtime)
        if universe_mtime > fundamentals_mtime:
            return False
    return True


def run_command_step(name: str, command: list[str], *, run_id: str) -> float:
    start_ts = time.perf_counter()
    _log_run_id_event(run_id, f"==> Running step: {name} ({' '.join(command)})")
    env = os.environ.copy()
    env["RUN_ID"] = str(run_id)
    result = subprocess.run(command, check=True, env=env, cwd=ROOT_DIR)
    elapsed = time.perf_counter() - start_ts
    if result.returncode != 0:
        raise RuntimeError(f"Step {name} failed with return code {result.returncode}")
    _log_run_id_event(run_id, f"<== Step completed: {name} (elapsed={elapsed:.2f}s)")
    return elapsed


def run_optional_command_step(name: str, command: list[str], *, run_id: str) -> None:
    try:
        if log_pipeline_history:
            log_pipeline_history(run_id, name, "start", None, None)
        elapsed = run_command_step(name, command, run_id=run_id)
        if log_pipeline_history:
            log_pipeline_history(run_id, name, "success", elapsed, None)
    except Exception as exc:
        logging.warning("Optional step failed and pipeline will continue: %s -> %s", name, exc)
        if log_pipeline_history:
            try:
                log_pipeline_history(run_id, name, "failed", None, str(exc))
            except Exception:
                pass


def run_archive_snapshot_step(
    *,
    run_id: str,
    archive_as_of_date: str | None = None,
    continue_on_error: bool = False,
) -> dict[str, object]:
    step_name = "archive_ranking_snapshot"
    command = [sys.executable, str(ARCHIVE_SNAPSHOT_SCRIPT), "--skip-if-exists"]
    if archive_as_of_date:
        command.extend(["--as-of-date", str(archive_as_of_date)])

    status = {
        "archived": False,
        "snapshot_path": None,
        "snapshot_date": None,
        "meta_path": None,
        "status": None,
        "error": None,
    }

    logging.info("[START] %s", step_name)
    if log_pipeline_history:
        log_pipeline_history(run_id, step_name, "start", None, None)

    try:
        ranking_df = pd.read_csv(DATA_DIR / "ranking_final.csv", low_memory=False)
        snapshot_date = None
        for col in ("as_of_date", "date", "trade_date", "snapshot_date"):
            if col in ranking_df.columns:
                parsed = pd.to_datetime(ranking_df[col], errors="coerce")
                if parsed.notna().any():
                    snapshot_date = parsed.max().strftime("%Y-%m-%d")
                    break
        if snapshot_date:
            token = snapshot_date.replace("-", "")
            snapshot_path = (DATA_DIR / "history" / "ranking" / f"{token}_ranking_final.csv").resolve()
            meta_path = (DATA_DIR / "history" / "ranking_meta" / f"{token}_ranking_meta.json").resolve()
            status["snapshot_path"] = str(snapshot_path)
            status["meta_path"] = str(meta_path)
            status["snapshot_date"] = snapshot_date
            archive_preexisting = snapshot_path.exists()
        else:
            archive_preexisting = False

        elapsed = run_command_step(step_name, command, run_id=run_id)
        status["archived"] = True
        status["status"] = "skipped_existing" if archive_preexisting else "saved"
        if log_pipeline_history:
            log_pipeline_history(run_id, step_name, "success", elapsed, None)
        logging.info("[OK] %s", step_name)
        return status
    except Exception as exc:
        status["error"] = str(exc)
        if log_pipeline_history:
            try:
                log_pipeline_history(run_id, step_name, "failed", None, str(exc))
            except Exception:
                pass
        logging.error("[FAIL] %s", step_name)
        if continue_on_error:
            logging.warning("Archive step failed but pipeline will continue: %s", exc)
            return status
        raise


@contextmanager
def temporarily_disable_theme_overlay_file(disabled: bool, *, run_id: str):
    backup_created = False
    if disabled and THEME_STOCK_DAILY_CSV.exists():
        try:
            if THEME_STOCK_DAILY_BACKUP_CSV.exists():
                THEME_STOCK_DAILY_BACKUP_CSV.unlink()
            THEME_STOCK_DAILY_CSV.replace(THEME_STOCK_DAILY_BACKUP_CSV)
            backup_created = True
            _log_run_id_event(
                run_id,
                f"[theme] overlay disconnected for baseline ranking: {THEME_STOCK_DAILY_CSV} -> {THEME_STOCK_DAILY_BACKUP_CSV}",
            )
        except Exception as exc:
            logging.warning("[theme] failed to disconnect overlay file and pipeline will continue: %s", exc)
    try:
        yield
    finally:
        if backup_created and THEME_STOCK_DAILY_BACKUP_CSV.exists():
            try:
                if THEME_STOCK_DAILY_CSV.exists():
                    THEME_STOCK_DAILY_CSV.unlink()
                THEME_STOCK_DAILY_BACKUP_CSV.replace(THEME_STOCK_DAILY_CSV)
                _log_run_id_event(run_id, f"[theme] overlay file restored: {THEME_STOCK_DAILY_CSV}")
            except Exception as exc:
                logging.warning("[theme] failed to restore overlay file: %s", exc)


def _load_codes_from_csv(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype={"code": str})
    if "code" not in df.columns:
        return set()
    return set(df["code"].astype(str).str.zfill(6).unique())


def _flow_codes_arg() -> str:
    universe_path = DATA_DIR / "universe.csv"
    if universe_path.exists():
        try:
            df = pd.read_csv(universe_path, dtype={"code": str})
            if "code" in df.columns:
                codes = [str(c).zfill(6) for c in df["code"].dropna().astype(str).tolist() if str(c).strip()]
                if codes:
                    return ",".join(sorted(set(codes)))
        except Exception as exc:
            logging.warning("Failed to build flow codes from universe.csv: %s", exc)

    raw = os.environ.get("SYMBOLS", "")
    if raw.strip():
        codes = [part.strip().zfill(6) for part in raw.split(",") if part.strip()]
        if codes:
            return ",".join(sorted(set(codes)))

    return ",".join(CORE_CODES)


def _flow_window() -> tuple[str, str]:
    start_env = os.environ.get("FLOW_START_DATE", "").strip()
    end_env = os.environ.get("FLOW_END_DATE", "").strip()
    if start_env and end_env:
        return start_env, end_env
    end = datetime.combine(_effective_market_date(), datetime.min.time())
    lookback = int(os.environ.get("FLOW_LOOKBACK_DAYS", "7"))
    start = end - timedelta(days=lookback)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def maybe_run_short_interest_ingestion(run_id: str) -> None:
    """공매도 잔고 일별 수집 (pykrx). feature_builder.merge_short_interest()의 입력."""
    if _env_flag("RUN_PIPELINE_SKIP_SHORT_INTEREST", "0"):
        _log_run_id_event(run_id, "RUN_PIPELINE_SKIP_SHORT_INTEREST enabled -> skip short interest ingestion")
        return

    short_script = Path("python") / "fetch_short_interest.py"
    if not short_script.exists():
        logging.info("fetch_short_interest.py not found -> skip short interest ingestion")
        return

    # 최근 7일 수집 — short_interest.csv 캐시가 이미 존재하면 미수집 날짜만 API 호출
    end_date = datetime.now(ZoneInfo(DEFAULT_MARKET_TIMEZONE)).strftime("%Y%m%d")
    start_date = (datetime.now(ZoneInfo(DEFAULT_MARKET_TIMEZONE)) - timedelta(days=7)).strftime("%Y%m%d")

    run_optional_command_step(
        "fetch_short_interest",
        [
            sys.executable,
            str(short_script),
            "--start", start_date,
            "--end", end_date,
        ],
        run_id=run_id,
    )


def maybe_run_flow_ingestion(run_id: str) -> None:
    if _env_flag("RUN_PIPELINE_SKIP_FLOW_INGESTION", "0"):
        _log_run_id_event(run_id, "RUN_PIPELINE_SKIP_FLOW_INGESTION enabled -> skip optional flow ingestion")
        return

    if not all([os.environ.get("KIS_BASE_URL"), os.environ.get("KIS_APP_KEY"), os.environ.get("KIS_APP_SECRET")]):
        logging.info("KIS env not fully configured -> skip optional flow ingestion/validation")
        return

    flow_script = Path("python") / "download_flows_kis.py"
    check_script = Path("python") / "check_flow_ingestion.py"
    if not flow_script.exists():
        logging.info("download_flows_kis.py not found -> skip optional flow ingestion")
        return
    if not check_script.exists():
        logging.info("check_flow_ingestion.py not found -> skip optional flow validation")
        return

    start_date, end_date = _flow_window()
    flow_codes = _flow_codes_arg()
    run_optional_command_step(
        "download_flows_kis",
        [
            sys.executable,
            str(flow_script),
            "--codes",
            flow_codes,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--save-raw",
        ],
        run_id=run_id,
    )
    run_optional_command_step(
        "check_flow_ingestion",
        [sys.executable, str(check_script)],
        run_id=run_id,
    )


def maybe_run_theme_overlay_steps(run_id: str, theme_cfg: dict[str, object]) -> None:
    if not bool(theme_cfg.get("overlay_enabled")):
        _log_run_id_event(run_id, "[theme] overlay disabled -> skip optional theme build steps")
        return

    mode = str(theme_cfg.get("mode") or THEME_OVERLAY_OFF)
    logging.info("[theme] start optional build sequence mode=%s", mode)
    run_optional_command_step(
        "theme_compute_etf_daily",
        [sys.executable, "python/compute_theme_etf_daily.py"],
        run_id=run_id,
    )
    run_optional_command_step(
        "theme_build_stock_daily",
        [sys.executable, "python/build_stock_theme_daily.py"],
        run_id=run_id,
    )
    logging.info("[theme] optional build sequence finished mode=%s", mode)


def maybe_run_theme_validation(run_id: str, theme_cfg: dict[str, object]) -> None:
    if not bool(theme_cfg.get("validation_enabled")):
        _log_run_id_event(run_id, "[theme] validation disabled -> skip theme validation runner")
        return
    logging.info("[theme] start optional validation mode=%s", theme_cfg.get("mode"))
    run_optional_command_step(
        "theme_overlay_validation",
        [sys.executable, "python/run_theme_overlay_validation.py"],
        run_id=run_id,
    )
    logging.info("[theme] optional validation completed mode=%s", theme_cfg.get("mode"))


def maybe_run_snapshot_readiness_reports(run_id: str) -> None:
    if SNAPSHOT_INVENTORY_SCRIPT.exists():
        run_optional_command_step(
            "ranking_snapshot_inventory",
            [sys.executable, str(SNAPSHOT_INVENTORY_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("report_ranking_snapshot_inventory.py not found -> skip snapshot inventory report")

    if CONFIDENCE_CALIBRATION_SCRIPT.exists():
        run_optional_command_step(
            "confidence_calibration_map",
            [sys.executable, str(CONFIDENCE_CALIBRATION_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("build_confidence_calibration_map.py not found -> skip confidence calibration report")


def maybe_run_top20_meaningfulness_report(run_id: str) -> None:
    if TOP20_MEANINGFULNESS_SCRIPT.exists():
        run_optional_command_step(
            "top20_meaningfulness_report",
            [sys.executable, str(TOP20_MEANINGFULNESS_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("analyze_top20_meaningfulness.py not found -> skip top20 meaningfulness report")


def maybe_run_operational_decision_reports(run_id: str) -> None:
    if CONFIDENCE_V2_SCRIPT.exists():
        run_optional_command_step(
            "confidence_score_v2",
            [sys.executable, str(CONFIDENCE_V2_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("build_confidence_score_v2.py not found -> skip confidence v2")

    if WALKFORWARD_ACCEPTANCE_SCRIPT.exists():
        run_optional_command_step(
            "walkforward_acceptance",
            [sys.executable, str(WALKFORWARD_ACCEPTANCE_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("build_walkforward_acceptance.py not found -> skip walkforward acceptance")

    if TOP20_BUYABILITY_SCRIPT.exists():
        run_optional_command_step(
            "top20_buyability_report",
            [sys.executable, str(TOP20_BUYABILITY_SCRIPT)],
            run_id=run_id,
        )
    else:
        logging.info("build_top20_buyability_report.py not found -> skip top20 buyability report")


def maybe_sync_csv_db_parity(run_id: str) -> None:
    if not get_engine:
        logging.info("No DB engine available -> skip csv/db parity sync")
        return
    if not CSV_DB_PARITY_SCRIPT.exists():
        logging.info("sync_csv_db_parity.py not found -> skip csv/db parity sync")
        return
    run_optional_command_step(
        "csv_db_parity_sync",
        [sys.executable, str(CSV_DB_PARITY_SCRIPT)],
        run_id=run_id,
    )


def _load_codes_from_db(table: str) -> set:
    if get_engine:
        try:
            eng = get_engine()
            with eng.connect() as conn:
                stmt = text(f"SELECT DISTINCT code FROM {table}")
                rows = conn.execute(stmt).fetchall()

            codes = {str(r[0]).zfill(6) for r in rows if r and r[0] is not None}
            logging.info("Loaded %d codes from Postgres table '%s'", len(codes), table)
            return codes
        except Exception as e:
            logging.warning("Postgres load failed for table '%s': %s", table, e)

    if not DB_PATH.exists():
        logging.info("No sqlite DB at %s -> skip DB check for table '%s'", DB_PATH, table)
        return set()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            rows = cur.execute(f"SELECT DISTINCT code FROM {table}").fetchall()
            codes = {str(r[0]).zfill(6) for r in rows if r and r[0] is not None}
            logging.info("Loaded %d codes from sqlite table '%s'", len(codes), table)
            return codes
    except Exception as e:
        logging.warning("sqlite load failed for table '%s': %s", table, e)
        return set()


def verify_core_codes(codes: list[str]) -> None:
    """Ensure critical codes exist in both CSV outputs and DB tables."""
    codes = [str(c).zfill(6) for c in codes]
    files = {
        "features.csv": DATA_DIR / "features.csv",
        "predictions.csv": DATA_DIR / "predictions.csv",
        "ranking_final.csv": DATA_DIR / "ranking_final.csv",
    }
    tables = ["features", "predictions", "daily_ranking"]

    missing = []

    for name, path in files.items():
        present = _load_codes_from_csv(path)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{name}: {','.join(miss)}")

    for tbl in tables:
        present = _load_codes_from_db(tbl)
        miss = [c for c in codes if c not in present]
        if miss:
            missing.append(f"{tbl} (DB): {','.join(miss)}")

    if missing:
        raise RuntimeError(f"Core code check failed -> { '; '.join(missing) }")

    logging.info("Core code check passed: %s", ", ".join(codes))


def _load_table(engine, table: str) -> pd.DataFrame:
    """Load full table from Postgres via SQLAlchemy engine."""
    return pd.read_sql(f"SELECT * FROM {table}", con=engine)


def _get_table_columns(engine, schema: str, table: str) -> list[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"schema": schema, "table": table}).fetchall()
    return [row[0] for row in rows]


def _latest_as_of_date(df: pd.DataFrame, date_col: str = "date") -> date:
    if date_col not in df.columns or df.empty:
        return date.today()
    try:
        dates = pd.to_datetime(df[date_col]).dt.date.dropna().unique()
        if len(dates) == 0:
            return date.today()
        return max(dates)
    except Exception:
        return date.today()


def append_prediction_history(run_id: str) -> None:
    """Append today's predictions into research.prediction_history."""
    if not get_engine:
        logging.info("No DB engine available -> skip prediction_history append")
        return
    run_id_num = _require_numeric_run_id(run_id, context="append_prediction_history")
    _log_run_id_event(run_id, "Entering prediction_history append")

    horizon_days = _env_horizon_days()
    model_version = _env_model_version()

    try:
        eng = get_engine()
        preds = _load_table(eng, "predictions")
        if preds.empty:
            logging.warning("predictions table empty -> skip prediction_history append")
            return

        score_cols = [
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "live_score",
            "live_rank",
            "live_score_source",
            "final_score",
            "final_score_custom",
            "score_formula_version",
        ]
        try:
            ranking = _load_table(eng, "daily_ranking")
            score_subset = ranking[["date", "code"] + [c for c in score_cols if c in ranking.columns]]
            merged = preds.merge(score_subset, on=["date", "code"], how="left")
        except Exception:
            logging.warning("daily_ranking load failed; inserting predictions without score columns", exc_info=True)
            merged = preds.copy()

        as_of = _latest_as_of_date(merged, "date")
        merged["as_of_date"] = as_of
        merged["run_id"] = run_id_num
        merged["model_version"] = model_version
        merged["horizon_days"] = horizon_days
        if "final_score_custom" not in merged.columns:
            merged["final_score_custom"] = None

        cols = [
            "run_id",
            "as_of_date",
            "code",
            "model_version",
            "horizon_days",
            "pred_return_60d",
            "pred_return_90d",
            "pred_mdd_60d",
            "pred_mdd_90d",
            "prob_top20_60d",
            "prob_top20_90d",
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "live_score",
            "live_rank",
            "live_score_source",
            "final_score",
            "final_score_custom",
        ]
        missing = [c for c in cols if c not in merged.columns]
        for c in missing:
            merged[c] = None

        out = merged[cols].copy()

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.prediction_history
                    WHERE run_id = :run_id AND as_of_date = :as_of_date
                    """
                ),
                {"run_id": run_id_num, "as_of_date": as_of},
            )
        actual_columns = _get_table_columns(eng, "research", "prediction_history")
        if actual_columns:
            out = out[[col for col in out.columns if col in actual_columns]].copy()
        if merged["run_id"].isna().any():
            raise ValueError("append_prediction_history: run_id contains null values before save")
        out.to_sql("prediction_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Appended prediction_history: run_id=%s rows=%d, as_of=%s", run_id_num, len(out), as_of)
    except Exception:
        logging.exception("append_prediction_history failed")
        raise


def append_ranking_history(run_id: str) -> None:
    """Append today's final ranking into research.ranking_history."""
    if not get_engine:
        logging.info("No DB engine available -> skip ranking_history append")
        return
    run_id_num = _require_numeric_run_id(run_id, context="append_ranking_history")
    _log_run_id_event(run_id, "Entering ranking_history append")

    horizon_days = _env_horizon_days()
    model_version = _env_model_version()
    top_n = _env_top_n()

    try:
        eng = get_engine()
        ranking = _load_table(eng, "daily_ranking")
        if ranking.empty:
            logging.warning("daily_ranking table empty -> skip ranking_history append")
            return

        as_of = _latest_as_of_date(ranking, "date")
        ranking["as_of_date"] = as_of
        sort_col = "live_score" if "live_score" in ranking.columns else "final_score"
        ranking = ranking.sort_values(["date", sort_col], ascending=[False, False]).copy()
        if "live_rank" in ranking.columns:
            ranking["rank"] = pd.to_numeric(ranking["live_rank"], errors="coerce")
        else:
            ranking["rank"] = ranking.groupby("date")[sort_col].rank(method="first", ascending=False)
        ranking["rank"] = ranking["rank"].fillna(
            ranking.groupby("date")[sort_col].rank(method="first", ascending=False)
        ).astype(int)
        ranking["in_top_n"] = ranking["rank"] <= top_n
        ranking["top_n"] = top_n
        ranking["run_id"] = run_id_num
        if "model_version" in ranking.columns:
            ranking["model_version"] = ranking["model_version"].fillna(model_version)
        else:
            ranking["model_version"] = model_version
        ranking["horizon_days"] = horizon_days

        cols = [
            "run_id",
            "as_of_date",
            "code",
            "model_version",
            "horizon_days",
            "rank",
            "live_score",
            "live_score_source",
            "final_score",
            "score_formula_version",
            "ret_score",
            "prob_score",
            "qual_score",
            "tech_score",
            "risk_penalty",
            "in_top_n",
            "top_n",
        ]
        missing = [c for c in cols if c not in ranking.columns]
        for c in missing:
            ranking[c] = None

        out = ranking[cols].copy()

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.ranking_history
                    WHERE run_id = :run_id AND as_of_date = :as_of_date
                    """
                ),
                {"run_id": run_id_num, "as_of_date": as_of},
            )
        actual_columns = _get_table_columns(eng, "research", "ranking_history")
        if actual_columns:
            out = out[[col for col in out.columns if col in actual_columns]].copy()
        if out["run_id"].isna().any():
            raise ValueError("append_ranking_history: run_id contains null values before save")
        out.to_sql("ranking_history", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info("Appended ranking_history: run_id=%s rows=%d, as_of=%s", run_id_num, len(out), as_of)
    except Exception:
        logging.exception("append_ranking_history failed")
        raise


def _table_stat_query(conn, table: str, date_cols: list[str]) -> None:
    """Log max(date-col) if present and count(*) for a table."""
    col = None
    try:
        schema, _, name = table.partition(".")
        if not name:
            schema = "public"
            name = table
        cols = conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=:schema AND table_name=:name
                """
            ),
            {"schema": schema, "name": name},
        ).fetchall()
        colnames = {c[0] for c in cols}
        if not colnames:
            logging.warning("Table stats skipped (not found): %s", table)
            return
        for cand in date_cols:
            if cand in colnames:
                col = cand
                break
    except Exception:
        col = None

    try:
        if col:
            res = conn.execute(text(f"SELECT MAX({col}) AS max_col, COUNT(*) AS cnt FROM {table}"))
            row = res.fetchone()
            logging.info("Table %s -> max_%s=%s, count=%s", table, col, row.max_col, row.cnt)
        else:
            res = conn.execute(text(f"SELECT COUNT(*) AS cnt FROM {table}"))
            row = res.fetchone()
            logging.info("Table %s -> count=%s (no date column found)", table, row.cnt)
    except Exception as e:
        logging.warning("Table stats failed for %s: %s", table, e)
        try:
            conn.rollback()
        except Exception:
            pass


def log_table_stats() -> None:
    """Log MAX(date-like) and row count for core/research tables."""
    if not get_engine:
        logging.info("No DB engine available -> skip table stats logging")
        return

    date_cols = ["date", "as_of_date", "trade_date", "created_at"]
    tables = [
        "market_status",
        "stocks",
        "prices_raw",
        "prices_clean",
        "prices_adjusted",
        "fact_price_daily",
        "quality",
        "fundamentals",
        "features",
        "daily_scores",
        "labels",
        "predictions",
        "daily_ranking",
        "trades",
        "backtest_trades",
        "research.dim_model_run",
        "research.prediction_history",
        "research.ranking_history",
        "research.backtest_outcome",
    ]

    try:
        eng = get_engine()
        with eng.connect() as conn:
            for tbl in tables:
                _table_stat_query(conn, tbl, date_cols)
            try:
                res = conn.execute(
                    text(
                        """
                        SELECT run_id, run_type, model_version, horizon_days, top_n, created_at
                        FROM research.dim_model_run
                        ORDER BY run_id DESC
                        LIMIT 5
                        """
                    )
                )
                rows = res.fetchall()
                logging.info("Recent dim_model_run (latest 5): %s", rows)
            except Exception as e:
                logging.warning("dim_model_run fetch failed: %s", e)
    except Exception:
        logging.exception("log_table_stats failed")


def _label_cols_for_horizon(h: int) -> tuple[str | None, str | None]:
    """Return (realized_return_col, realized_mdd_col) names for the given horizon days."""
    ret_col = f"realized_return_{h}d"
    mdd_col_candidates = [f"realized_mdd_{h}d", f"target_mdd_{h}d"]
    return ret_col, mdd_col_candidates


def append_backtest_outcome(run_id: str) -> None:
    """Append realized outcomes (from labels) into research.backtest_outcome."""
    if not get_engine:
        logging.info("No DB engine available -> skip backtest_outcome append")
        return
    run_id_num = _require_numeric_run_id(run_id, context="append_backtest_outcome")
    _log_run_id_event(run_id, "Entering backtest_outcome append")

    horizon_days = _env_horizon_days()
    ret_col, mdd_candidates = _label_cols_for_horizon(horizon_days)

    try:
        eng = get_engine()
        query = text(
            "SELECT run_id, as_of_date, code, horizon_days "
            "FROM research.prediction_history WHERE run_id = :run_id"
        )
        preds_hist = pd.read_sql(query, con=eng, params={"run_id": run_id_num}, parse_dates=["as_of_date"])
        if preds_hist.empty:
            logging.warning("prediction_history empty for run_id=%s -> skip backtest_outcome append", run_id)
            return

        labels = _load_table(eng, "labels")
        if labels.empty:
            logging.warning("labels table empty -> skip backtest_outcome append")
            return
        labels["date"] = pd.to_datetime(labels["date"])

        latest_label_date = labels["date"].max()
        preds_hist = preds_hist[preds_hist["as_of_date"] <= latest_label_date]
        if preds_hist.empty:
            logging.info(
                "No prediction_history rows within label coverage (latest_label_date=%s) -> skip backtest_outcome append",
                latest_label_date.date(),
            )
            return

        if ret_col not in labels.columns:
            logging.warning("%s not in labels -> skip backtest_outcome append", ret_col)
            return
        mdd_col = next((c for c in mdd_candidates if c in labels.columns), None)

        merged = preds_hist.merge(
            labels[["date", "code", ret_col] + ([mdd_col] if mdd_col else [])],
            left_on=["as_of_date", "code"],
            right_on=["date", "code"],
            how="left",
        )
        merged["realized_return"] = merged[ret_col]
        merged["realized_mdd"] = merged[mdd_col] if mdd_col else None
        merged["label_source"] = "from_labels"

        out_cols = ["run_id", "as_of_date", "code", "horizon_days", "realized_return", "realized_mdd", "label_source"]
        out = merged[out_cols].copy().dropna(subset=["realized_return"])
        if out.empty:
            logging.warning("No realized_return rows after merge -> skip backtest_outcome append")
            return

        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM research.backtest_outcome
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id_num},
            )
        if out["run_id"].isna().any():
            raise ValueError("append_backtest_outcome: run_id contains null values before save")
        out.to_sql("backtest_outcome", eng, schema="research", if_exists="append", index=False, method="multi")
        logging.info(
            "Appended backtest_outcome: run_id=%s rows=%d, horizon=%d, ret_col=%s, mdd_col=%s",
            run_id_num,
            len(out),
            horizon_days,
            ret_col,
            mdd_col,
        )
    except Exception:
        logging.exception("append_backtest_outcome failed")
        raise


def main() -> None:
    args = parse_cli_args()
    setup_logging()
    load_runtime_env()
    ensure_data_dir()
    apply_schema_if_available()
    market_date = _resolve_pipeline_market_date()
    os.environ["MARKET_DATE"] = market_date.isoformat()
    theme_cfg = _theme_config()
    write_theme_pipeline_mode_note(theme_cfg)

    run_id: str | None = None
    try:
        with acquire_pipeline_run_lock():
            run_id = create_model_run_id()
            _log_run_id_event(run_id, "Created pipeline run_id")
            pipeline_start = time.perf_counter()
            archive_status = {
                "archived": False,
                "snapshot_path": None,
                "snapshot_date": None,
                "meta_path": None,
                "error": None,
            }
            grouped_profile_state: dict[str, dict[str, object]] = {}
            _log_run_id_event(run_id, f"Pipeline step order: {' -> '.join(name for name, _ in STEPS)}")
            _log_run_id_event(run_id, f"Resolved pipeline MARKET_DATE={market_date.isoformat()}")
            _log_run_id_event(
                run_id,
                "[theme] effective config: "
                f"overlay_enabled={theme_cfg['overlay_enabled']} "
                f"validation_enabled={theme_cfg['validation_enabled']} "
                f"mode={theme_cfg['mode']} "
                f"ranking_enabled={theme_cfg['ranking_enabled']}",
            )
            for name, script in STEPS:
                profile_group = PROFILE_STEP_GROUP_BY_STEP.get(name)
                if profile_group and profile_group not in grouped_profile_state:
                    grouped_profile_state[profile_group] = {
                        "elapsed_sec": 0.0,
                        "sample": profile_begin(),
                    }
                if name == "ranking_builder":
                    maybe_run_theme_overlay_steps(run_id, theme_cfg)
                if log_pipeline_history:
                    log_pipeline_history(run_id, name, "start", None, None)
                if name == "fetch_fundamentals_dart" and should_skip_fundamentals_step():
                    logging.info(
                        "Skipping %s because %s is already fresh for today and not older than %s",
                        name,
                        FUNDAMENTALS_CSV.resolve(),
                        UNIVERSE_CSV.resolve(),
                    )
                    elapsed = 0.0
                    if log_pipeline_history:
                        log_pipeline_history(run_id, name, "success", elapsed, "skipped_fresh_today")
                else:
                    if name == "ranking_builder":
                        disable_overlay = not bool(theme_cfg.get("ranking_enabled"))
                        with temporarily_disable_theme_overlay_file(disable_overlay, run_id=run_id):
                            elapsed = run_step(name, script, run_id=run_id)
                    else:
                        elapsed = run_step(name, script, run_id=run_id)
                    if log_pipeline_history:
                        log_pipeline_history(run_id, name, "success", elapsed, None)
                    if name == "ranking_builder":
                        archive_status = run_archive_snapshot_step(
                            run_id=run_id,
                            archive_as_of_date=args.archive_as_of_date,
                            continue_on_error=bool(args.continue_on_archive_error),
                        )
                        maybe_run_snapshot_readiness_reports(run_id)

                if profile_group:
                    state = grouped_profile_state[profile_group]
                    state["elapsed_sec"] = float(state.get("elapsed_sec", 0.0)) + float(elapsed)
                    if PROFILE_STEP_GROUP_LAST_STEP.get(profile_group) == name:
                        log_profile_result(
                            profile_group,
                            float(state["elapsed_sec"]),
                            sample=state.get("sample") if isinstance(state.get("sample"), object) else None,
                            extra={"run_id": run_id, "script": "run_pipeline.py"},
                        )

            walkforward_sample = profile_begin()
            maybe_run_flow_ingestion(run_id)
            maybe_run_short_interest_ingestion(run_id)
            maybe_run_theme_validation(run_id, theme_cfg)
            maybe_sync_csv_db_parity(run_id)

            verify_core_codes(CORE_CODES)

            append_prediction_history(run_id)
            append_ranking_history(run_id)
            append_backtest_outcome(run_id)
            maybe_run_top20_meaningfulness_report(run_id)
            maybe_run_operational_decision_reports(run_id)
            log_profile_result(
                "walkforward",
                time.perf_counter() - walkforward_sample.wall_start,
                sample=walkforward_sample,
                extra={
                    "run_id": run_id,
                    "script": "run_pipeline.py",
                    "notes": "includes walkforward acceptance and related operational decision reports",
                },
            )
            log_table_stats()

            try:
                snapshot_script = Path("scripts") / "snapshot_scores.py"
                if snapshot_script.exists():
                    run_step("snapshot_scores", str(snapshot_script), run_id=run_id)
                else:
                    logging.warning("snapshot_scores.py not found -> skip snapshot step")
            except Exception:
                logging.exception("Snapshot step failed (pipeline continues)")

            total_elapsed = time.perf_counter() - pipeline_start
            if log_pipeline_history:
                log_pipeline_history(run_id, "pipeline", "success", total_elapsed, "all steps completed")
            logging.info(
                "Pipeline summary: snapshot_archived=%s snapshot_path=%s snapshot_date=%s",
                archive_status.get("archived"),
                archive_status.get("snapshot_path"),
                archive_status.get("snapshot_date"),
            )
            logging.info("Pipeline finished successfully (total=%.2fs)", total_elapsed)
    except Exception as e:
        logging.exception("Pipeline error: %s", e)
        if log_pipeline_history and run_id:
            try:
                log_pipeline_history(run_id, "pipeline", "failed", None, str(e))
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
