import argparse
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from db import use_sqlite_mirror
except Exception:
    def use_sqlite_mirror() -> bool:
        return False

from kis_client import KISClient, KISError

try:
    from pykrx import stock as pykrx_stock
except Exception:
    pykrx_stock = None


DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "prices_daily_raw.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
DEFAULT_THEME_ETF_MASTER_PATH = DATA_DIR / "theme_etf_master.csv"
DEFAULT_SYMBOLS = ["005930", "000660", "035420"]
RAW_DB_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume"]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download stock prices from KIS and optionally include theme ETF codes.")
    parser.add_argument(
        "--include-theme-etf-codes",
        action="store_true",
        default=str(os.getenv("INCLUDE_THEME_ETF_CODES", "0")).strip().lower() in {"1", "true", "yes", "on"},
        help="Include ETF codes from theme_etf_master.csv in the download universe.",
    )
    parser.add_argument(
        "--theme-etf-master-path",
        type=Path,
        default=Path(os.getenv("THEME_ETF_MASTER_PATH", str(DEFAULT_THEME_ETF_MASTER_PATH))),
        help="Path to theme_etf_master.csv.",
    )
    return parser.parse_args()


def _resolve_end_date() -> datetime:
    market_date = os.getenv("MARKET_DATE", "").strip()
    if market_date:
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                resolved = datetime.strptime(market_date, fmt)
                logging.info("Using MARKET_DATE override for price download: %s", resolved.date())
                return resolved
            except ValueError:
                continue
        logging.warning("Invalid MARKET_DATE format for price download: %s", market_date)
    resolved = datetime.today() - timedelta(days=1)
    logging.info("Using previous-close end date for price download: %s", resolved.date())
    return resolved


def _normalize_code(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.zfill(6) if text.isdigit() else text


def _env_symbols() -> list[str]:
    try:
        uni_path = DATA_DIR / "universe.csv"
        if uni_path.exists():
            dfu = pd.read_csv(uni_path, dtype={"code": str})
            codes = [_normalize_code(c) for c in dfu["code"].dropna().tolist()]
            codes = [c for c in codes if c]
            if codes:
                return sorted(set(codes))
    except Exception as exc:
        logging.warning("Failed to load universe.csv, fallback to .env SYMBOLS: %s", exc)

    val = os.getenv("SYMBOLS")
    if not val:
        return DEFAULT_SYMBOLS[:]
    symbols = [_normalize_code(s) for s in val.split(",")]
    symbols = [s for s in symbols if s]
    return sorted(set(symbols)) or DEFAULT_SYMBOLS[:]


def load_theme_etf_codes(path: Path) -> list[str]:
    if not path.exists():
        logging.warning("theme_etf_master path not found -> skip ETF include: %s", path)
        return []
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        logging.warning("Failed to read theme_etf_master -> skip ETF include: %s", exc)
        return []

    code_col = None
    for candidate in ["etf_code", "code", "symbol"]:
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        logging.warning("theme_etf_master missing code column(etf_code/code/symbol) -> skip ETF include")
        return []

    codes = [_normalize_code(v) for v in df[code_col].tolist()]
    codes = sorted({code for code in codes if code})
    logging.info("Loaded theme ETF codes path=%s count=%d sample=%s", path, len(codes), codes[:10])
    return codes


def merge_price_universe(base_codes: list[str], etf_codes: list[str], include_theme_etf_codes: bool) -> dict[str, Any]:
    stock_set = {code for code in map(_normalize_code, base_codes) if code}
    etf_set = {code for code in map(_normalize_code, etf_codes) if code}
    merged_set = set(stock_set)
    if include_theme_etf_codes:
        merged_set |= etf_set

    overlap = stock_set & etf_set
    summary = {
        "include_theme_etf_codes": include_theme_etf_codes,
        "stock_codes": len(stock_set),
        "theme_etf_codes": len(etf_set) if include_theme_etf_codes else 0,
        "merged_codes": len(merged_set),
        "duplicate_overlap": len(overlap) if include_theme_etf_codes else 0,
        "stock_code_set": stock_set,
        "theme_etf_code_set": etf_set if include_theme_etf_codes else set(),
        "merged_code_list": sorted(merged_set),
        "theme_etf_sample": sorted(etf_set)[:10] if include_theme_etf_codes else [],
    }
    return summary


def summarize_universe_merge(summary: dict[str, Any]) -> None:
    logging.info("include_theme_etf_codes=%s", summary["include_theme_etf_codes"])
    logging.info("stock_codes=%d", summary["stock_codes"])
    logging.info("theme_etf_codes=%d", summary["theme_etf_codes"])
    logging.info("merged_codes=%d", summary["merged_codes"])
    logging.info("duplicate_overlap=%d", summary["duplicate_overlap"])
    if summary["theme_etf_sample"]:
        logging.info("theme_etf_sample=%s", summary["theme_etf_sample"])


def annotate_universe_source(df: pd.DataFrame, stock_code_set: set[str], theme_etf_code_set: set[str]) -> pd.DataFrame:
    out = df.copy()
    out["code"] = out["code"].astype(str).map(_normalize_code)
    in_stock = out["code"].isin(stock_code_set)
    in_etf = out["code"].isin(theme_etf_code_set)
    out["asset_type"] = np.where(in_etf & ~in_stock, "etf", "stock")
    out["universe_source"] = np.select(
        [in_stock & in_etf, in_stock, in_etf],
        ["both", "stock_master", "theme_etf_master"],
        default="unknown",
    )
    return out


def generate_demo_prices(symbols: list[str], days: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_date = datetime.today()
    dates = []
    d = end_date
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d.replace(hour=0, minute=0, second=0, microsecond=0))
        d -= timedelta(days=1)
    dates = sorted(dates)

    rows = []
    for code in symbols:
        base_price = rng.uniform(40000, 90000)
        price = base_price
        for dt in dates:
            ret = rng.normal(0, 0.01)
            price = max(1000, price * (1 + ret))
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = (high + low) / 2 * (1 + rng.normal(0, 0.001))
            volume = int(rng.uniform(1e5, 5e6))
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": int(round(open_)),
                    "high": int(round(high)),
                    "low": int(round(low)),
                    "close": int(round(price)),
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def _kis_fetch_daily_prices(
    client: KISClient,
    code: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> Optional[pd.DataFrame]:
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }
    try:
        data = client.get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST03010100",
            params=params,
        )
        arr = data.get("output2") or data.get("output") or []
        if not isinstance(arr, list) or not arr:
            logging.warning("KIS daily price empty(%s): %s", code, data)
            return None

        rows = []
        for item in arr:
            ymd = str(item.get("stck_bsop_date") or "")
            try:
                date_str = datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")
            except Exception:
                continue
            try:
                rows.append(
                    {
                        "date": date_str,
                        "code": code,
                        "open": int(float(item.get("stck_oprc", 0))),
                        "high": int(float(item.get("stck_hgpr", 0))),
                        "low": int(float(item.get("stck_lwpr", 0))),
                        "close": int(float(item.get("stck_clpr", 0))),
                        "volume": int(float(item.get("acml_vol", 0))),
                    }
                )
            except Exception:
                continue
        if not rows:
            return None
        return pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
    except KISError as exc:
        logging.warning("KIS daily price failed(%s): %s", code, exc)
        return None
    except Exception as exc:
        logging.warning("KIS daily price exception(%s): %s", code, exc)
        return None


def _summarize_download_results(success_codes: list[str], failed_codes: list[str], stock_code_set: set[str], theme_etf_code_set: set[str], source_name: str) -> None:
    success_stock = sum(1 for code in success_codes if code in stock_code_set)
    success_etf = sum(1 for code in success_codes if code in theme_etf_code_set and code not in stock_code_set)
    failed_stock = sum(1 for code in failed_codes if code in stock_code_set)
    failed_etf = sum(1 for code in failed_codes if code in theme_etf_code_set and code not in stock_code_set)
    logging.info(
        "%s summary success_total=%d success_stock=%d success_etf=%d failed_total=%d failed_stock=%d failed_etf=%d",
        source_name,
        len(success_codes),
        success_stock,
        success_etf,
        len(failed_codes),
        failed_stock,
        failed_etf,
    )
    if failed_codes:
        logging.warning("%s failed_codes_sample=%s", source_name, failed_codes[:20])


def try_kis_download(universe_summary: dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        client = KISClient.from_env()
    except ValueError:
        logging.info("KIS env missing or incomplete -> skip KIS")
        return None

    try:
        client.issue_access_token()
    except KISError as exc:
        logging.warning("KIS token issuance failed: %s", exc)
        return None

    end = _resolve_end_date()
    start = end - timedelta(days=365 * 3)
    start_ymd = start.strftime("%Y%m%d")
    end_ymd = end.strftime("%Y%m%d")

    frames: list[pd.DataFrame] = []
    success_codes: list[str] = []
    failed_codes: list[str] = []
    for code in universe_summary["merged_code_list"]:
        df_code = _kis_fetch_daily_prices(client=client, code=code, start_yyyymmdd=start_ymd, end_yyyymmdd=end_ymd)
        if df_code is not None and not df_code.empty:
            frames.append(df_code)
            success_codes.append(code)
        else:
            failed_codes.append(code)

    _summarize_download_results(
        success_codes,
        failed_codes,
        universe_summary["stock_code_set"],
        universe_summary["theme_etf_code_set"],
        "KIS",
    )
    if not frames:
        logging.warning("KIS daily prices fetched nothing")
        return None
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)


def try_pykrx_download(universe_summary: dict[str, Any]) -> Optional[pd.DataFrame]:
    if pykrx_stock is None:
        logging.info("pykrx not available -> skip pykrx fallback")
        return None
    try:
        end = _resolve_end_date()
        start = end - timedelta(days=365 * 3)
        start_ymd = start.strftime("%Y%m%d")
        end_ymd = end.strftime("%Y%m%d")

        import socket as _socket
        _pykrx_timeout = int(os.environ.get("PYKRX_PER_STOCK_TIMEOUT", "30"))
        _prev_timeout = _socket.getdefaulttimeout()
        _socket.setdefaulttimeout(_pykrx_timeout)

        frames: list[pd.DataFrame] = []
        success_codes: list[str] = []
        failed_codes: list[str] = []
        for code in universe_summary["merged_code_list"]:
            try:
                df = pykrx_stock.get_market_ohlcv_by_date(start_ymd, end_ymd, code)
                if df is None or df.empty:
                    failed_codes.append(code)
                    continue
                df = df.reset_index()
                rename_map = {}
                for col in df.columns:
                    normalized = str(col).strip()
                    if normalized in {"날짜", "date"}:
                        rename_map[col] = "date"
                    elif normalized in {"시가", "open"}:
                        rename_map[col] = "open"
                    elif normalized in {"고가", "high"}:
                        rename_map[col] = "high"
                    elif normalized in {"저가", "low"}:
                        rename_map[col] = "low"
                    elif normalized in {"종가", "close"}:
                        rename_map[col] = "close"
                    elif normalized in {"거래량", "volume"}:
                        rename_map[col] = "volume"
                df = df.rename(columns=rename_map)
                required = ["date", "open", "high", "low", "close", "volume"]
                if any(col not in df.columns for col in required):
                    failed_codes.append(code)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df["code"] = code
                df = df[["date", "code", "open", "high", "low", "close", "volume"]]
                frames.append(df)
                success_codes.append(code)
            except Exception as exc:
                logging.warning("pykrx fetch error(%s): %s", code, exc)
                failed_codes.append(code)
                continue

        _socket.setdefaulttimeout(_prev_timeout)
        _summarize_download_results(
            success_codes,
            failed_codes,
            universe_summary["stock_code_set"],
            universe_summary["theme_etf_code_set"],
            "pykrx",
        )
        if not frames:
            logging.warning("pykrx daily prices fetched nothing")
            return None
        return pd.concat(frames, ignore_index=True).sort_values(["code", "date"]).reset_index(drop=True)
    except Exception as exc:
        logging.warning("pykrx fallback exception: %s", exc)
        return None


def log_output_row_summary(df: pd.DataFrame) -> None:
    if df.empty:
        logging.info("raw output row summary: empty")
        return
    if "asset_type" in df.columns:
        counts = df["asset_type"].fillna("unknown").value_counts().to_dict()
        logging.info("raw output asset_type rows=%s", counts)
    if "universe_source" in df.columns:
        counts = df["universe_source"].fillna("unknown").value_counts().to_dict()
        logging.info("raw output universe_source rows=%s", counts)


def write_prices_raw_csv(df: pd.DataFrame) -> None:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    logging.info("Saved raw prices: %s (rows=%d)", RAW_CSV.resolve(), len(df))


def save_prices_raw_db(df: pd.DataFrame) -> None:
    if not use_sqlite_mirror():
        logging.info("Skipping sqlite mirror for raw prices (USE_SQLITE_MIRROR=0)")
        return
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices_raw (
                date    DATE NOT NULL,
                code    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL,
                PRIMARY KEY (date, code)
            );
            """
        )
        db_df = df.loc[:, RAW_DB_COLUMNS].copy()
        records = db_df.to_dict(orient="records")
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices_raw
            (date, code, open, high, low, close, volume)
            VALUES (:date, :code, :open, :high, :low, :close, :volume)
            """,
            records,
        )
        conn.commit()
        logging.info("Saved raw prices to DB: %s (rows=%d)", DB_PATH.resolve(), len(db_df))
    except Exception:
        logging.exception("Failed to save raw prices to DB")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


def update_demo_marker(demo_mode: bool) -> None:
    try:
        marker = DATA_DIR / ".demo"
        if demo_mode:
            marker.write_text("demo", encoding="utf-8")
        elif marker.exists():
            marker.unlink()
    except Exception as exc:
        logging.warning("Failed to update demo marker: %s", exc)


def main() -> None:
    setup_logging()
    ensure_data_dir()
    args = parse_args()

    # Skip download if env flag set and existing CSV is fresh enough
    skip_flag = os.environ.get("RUN_PIPELINE_SKIP_PRICE_DOWNLOAD", "0").strip()
    if skip_flag in ("1", "true", "yes"):
        if RAW_CSV.exists():
            logging.info("RUN_PIPELINE_SKIP_PRICE_DOWNLOAD=1 and %s exists -> skip download", RAW_CSV)
            return
        logging.warning("RUN_PIPELINE_SKIP_PRICE_DOWNLOAD=1 but %s missing -> proceed with download", RAW_CSV)

    base_codes = _env_symbols()
    theme_etf_codes = load_theme_etf_codes(args.theme_etf_master_path) if args.include_theme_etf_codes else []
    universe_summary = merge_price_universe(base_codes, theme_etf_codes, args.include_theme_etf_codes)
    summarize_universe_merge(universe_summary)

    df = try_kis_download(universe_summary)
    if df is None:
        logging.info("KIS unavailable -> trying pykrx fallback...")
        df = try_pykrx_download(universe_summary)

    demo_mode = False
    if df is None:
        # If real price data already exists, preserve it rather than overwriting with demo prices
        if RAW_CSV.exists():
            logging.warning(
                "KIS and pykrx both unavailable, but %s already exists -> skip overwrite with demo prices", RAW_CSV
            )
            return
        logging.info("Generating demo prices...")
        df = generate_demo_prices(universe_summary["merged_code_list"])
        demo_mode = True

    df = annotate_universe_source(df, universe_summary["stock_code_set"], universe_summary["theme_etf_code_set"])
    log_output_row_summary(df)
    write_prices_raw_csv(df)
    update_demo_marker(demo_mode)
    save_prices_raw_db(df)


if __name__ == "__main__":
    main()
