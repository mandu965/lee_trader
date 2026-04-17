import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine

try:
    import pykrx.website.krx.etx.core as etx_core
except Exception:
    etx_core = None


LOGGER = logging.getLogger("calculate_etf_signals")
LOOKBACK_DAYS = 120
MIN_REQUIRED_HISTORY = 61
RETURN_WINDOWS = (5, 20, 60)
RET_20D_WEIGHT = 0.5
RET_5D_WEIGHT = 0.3
VOLUME_WEIGHT = 0.2
SOURCE_NAME = "prices_raw_or_pykrx"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate ETF daily signals and upsert etf_signal_daily.")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Signal date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def parse_as_of_date(raw_value: str) -> tuple[date, str]:
    parsed = datetime.strptime(raw_value, "%Y-%m-%d").date()
    return parsed, parsed.strftime("%Y%m%d")


def get_table(name: str) -> Table:
    metadata = MetaData()
    return Table(name, metadata, autoload_with=get_engine())


def load_active_etfs() -> list[dict[str, Any]]:
    query = text(
        """
        SELECT etf_code, etf_name
        FROM etf_master
        WHERE is_active = true
        ORDER BY etf_code
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def build_isin_map() -> dict[str, str]:
    if etx_core is None:
        return {}

    etf_basic_cls = getattr(etx_core, "ETF_\uc804\uc885\ubaa9\uae30\ubcf8\uc885\ubaa9")
    df = etf_basic_cls().fetch()
    if df is None or getattr(df, "empty", True):
        return {}

    isin_map: dict[str, str] = {}
    for row in df.to_dict(orient="records"):
        etf_code = str(row.get("ISU_SRT_CD") or "").strip().zfill(6)
        isin = str(row.get("ISU_CD") or "").strip()
        if etf_code and isin:
            isin_map[etf_code] = isin
    return isin_map


def load_prices_raw_history(etf_codes: list[str], start_date: date, as_of_date: date) -> pd.DataFrame:
    if not etf_codes:
        return pd.DataFrame(columns=["date", "code", "close", "volume"])

    query = text(
        """
        SELECT date, code, close, volume
        FROM prices_raw
        WHERE code IN :codes
          AND date BETWEEN :start_date AND :as_of_date
        ORDER BY code, date
        """
    ).bindparams(bindparam("codes", expanding=True))
    with get_engine().begin() as conn:
        rows = conn.execute(
            query,
            {
                "codes": list(etf_codes),
                "start_date": start_date,
                "as_of_date": as_of_date,
            },
        ).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["date", "code", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for column in ("close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fetch_price_history_from_pykrx(
    *,
    etf_code: str,
    isin: str,
    start_krx_date: str,
    end_krx_date: str,
) -> pd.DataFrame:
    if etx_core is None:
        raise RuntimeError("pykrx ETX core module is not available")

    price_cls = getattr(etx_core, "\uac1c\ubcc4\uc885\ubaa9\uc2dc\uc138_ETF")
    df = price_cls().fetch(start_krx_date, end_krx_date, isin)
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["date", "code", "close", "volume", "trading_value", "nav_price"])

    required_cols = {"TRD_DD", "TDD_CLSPRC", "ACC_TRDVOL", "ACC_TRDVAL", "LST_NAV"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise RuntimeError(f"ETF price response missing columns: {sorted(missing_cols)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["TRD_DD"], format="%Y/%m/%d", errors="coerce"),
            "code": etf_code,
            "close": pd.to_numeric(df["TDD_CLSPRC"].astype(str).str.replace(",", ""), errors="coerce"),
            "volume": pd.to_numeric(df["ACC_TRDVOL"].astype(str).str.replace(",", ""), errors="coerce"),
            "trading_value": pd.to_numeric(df["ACC_TRDVAL"].astype(str).str.replace(",", ""), errors="coerce"),
            "nav_price": pd.to_numeric(df["LST_NAV"].astype(str).str.replace(",", ""), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def load_price_history(
    *,
    etf_code: str,
    as_of_date: date,
    start_date: date,
    prices_raw_df: pd.DataFrame,
    isin_map: dict[str, str],
) -> pd.DataFrame:
    history = prices_raw_df[prices_raw_df["code"] == etf_code].copy()
    if not history.empty and history["date"].max().date() >= as_of_date and len(history) >= MIN_REQUIRED_HISTORY:
        history["trading_value"] = np.nan
        history["nav_price"] = np.nan
        return history.sort_values("date").reset_index(drop=True)

    isin = isin_map.get(etf_code)
    if not isin:
        return history.sort_values("date").reset_index(drop=True)

    LOGGER.info(
        "prices_raw coverage insufficient for etf_code=%s -> fetching ETF price history from pykrx. "
        "If this becomes the common path, add a dedicated etf_price_daily table.",
        etf_code,
    )
    fetched = fetch_price_history_from_pykrx(
        etf_code=etf_code,
        isin=isin,
        start_krx_date=start_date.strftime("%Y%m%d"),
        end_krx_date=as_of_date.strftime("%Y%m%d"),
    )
    if fetched.empty:
        history["trading_value"] = np.nan
        history["nav_price"] = np.nan
        return history.sort_values("date").reset_index(drop=True)
    return fetched.sort_values("date").reset_index(drop=True)


def compute_return(close_series: pd.Series, window: int) -> float | None:
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if len(series) <= window:
        return None
    current = float(series.iloc[-1])
    base = float(series.iloc[-(window + 1)])
    if base == 0:
        return None
    return (current / base) - 1.0


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean_value = values.mean(skipna=True)
    std_value = values.std(skipna=True, ddof=0)
    if pd.isna(std_value) or std_value == 0:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    scores = (values - mean_value) / std_value
    return scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_signal_score(row: pd.Series) -> float:
    ret_20d_z = float(row.get("ret_20d_zscore", 0.0) or 0.0)
    ret_5d_z = float(row.get("ret_5d_zscore", 0.0) or 0.0)
    volume_z = float(row.get("volume_zscore", 0.0) or 0.0)
    return (
        (RET_20D_WEIGHT * ret_20d_z)
        + (RET_5D_WEIGHT * ret_5d_z)
        + (VOLUME_WEIGHT * volume_z)
    )


def build_signal_frame(
    *,
    active_etfs: list[dict[str, Any]],
    as_of_date: date,
    prices_raw_df: pd.DataFrame,
    isin_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_date = as_of_date - timedelta(days=LOOKBACK_DAYS)

    for etf in active_etfs:
        etf_code = str(etf["etf_code"])
        etf_name = str(etf.get("etf_name") or "")
        try:
            history = load_price_history(
                etf_code=etf_code,
                as_of_date=as_of_date,
                start_date=start_date,
                prices_raw_df=prices_raw_df,
                isin_map=isin_map,
            )
            if history.empty:
                LOGGER.warning("No ETF history available etf_code=%s etf_name=%s", etf_code, etf_name)
                continue

            history = history.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if history.empty or history["date"].max().date() < as_of_date:
                LOGGER.warning("ETF history stale etf_code=%s etf_name=%s last_date=%s", etf_code, etf_name, history["date"].max())
                continue

            latest = history.iloc[-1]
            rows.append(
                {
                    "etf_code": etf_code,
                    "close_price": float(latest["close"]) if pd.notna(latest["close"]) else None,
                    "nav_price": float(latest["nav_price"]) if "nav_price" in history.columns and pd.notna(latest["nav_price"]) else None,
                    "nav_gap_pct": (
                        (float(latest["close"]) / float(latest["nav_price"]) - 1.0)
                        if "nav_price" in history.columns
                        and pd.notna(latest["close"])
                        and pd.notna(latest["nav_price"])
                        and float(latest["nav_price"]) != 0.0
                        else None
                    ),
                    "return_1d": compute_return(history["close"], 1),
                    "return_5d": compute_return(history["close"], 5),
                    "return_20d": compute_return(history["close"], 20),
                    "return_60d": compute_return(history["close"], 60),
                    "volume": float(latest["volume"]) if pd.notna(latest["volume"]) else None,
                    "trading_value": (
                        float(latest["trading_value"])
                        if "trading_value" in history.columns and pd.notna(latest["trading_value"])
                        else None
                    ),
                    "source_name": SOURCE_NAME,
                }
            )
        except Exception as exc:
            LOGGER.warning("ETF signal history build failed etf_code=%s etf_name=%s error=%s", etf_code, etf_name, exc)
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ret_20d_zscore"] = zscore(df["return_20d"])
    df["ret_5d_zscore"] = zscore(df["return_5d"])
    df["volume_zscore"] = zscore(df["volume"])
    df["relative_strength_score"] = df["ret_20d_zscore"].fillna(0.0)
    df["signal_score"] = df.apply(compute_signal_score, axis=1)
    df["as_of_date"] = as_of_date.isoformat()
    return df


def upsert_etf_signals(signal_df: pd.DataFrame) -> int:
    if signal_df.empty:
        return 0

    table = get_table("etf_signal_daily")
    payload = []
    for row in signal_df.to_dict(orient="records"):
        payload.append(
            {
                "as_of_date": row["as_of_date"],
                "etf_code": row["etf_code"],
                "close_price": row.get("close_price"),
                "nav_price": row.get("nav_price"),
                "nav_gap_pct": row.get("nav_gap_pct"),
                "return_1d": row.get("return_1d"),
                "return_5d": row.get("return_5d"),
                "return_20d": row.get("return_20d"),
                "volume": row.get("volume"),
                "trading_value": row.get("trading_value"),
                "aum_amount": None,
                "relative_strength_score": row.get("relative_strength_score"),
                "signal_score": row.get("signal_score"),
                "signal_payload_json": json.dumps(
                    {
                        "return_60d": row.get("return_60d"),
                        "ret_20d_zscore": row.get("ret_20d_zscore"),
                        "ret_5d_zscore": row.get("ret_5d_zscore"),
                        "volume_zscore": row.get("volume_zscore"),
                        "source_name": row.get("source_name"),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    stmt = insert(table).values(payload)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["as_of_date", "etf_code"],
        set_={
            "close_price": stmt.excluded.close_price,
            "nav_price": stmt.excluded.nav_price,
            "nav_gap_pct": stmt.excluded.nav_gap_pct,
            "return_1d": stmt.excluded.return_1d,
            "return_5d": stmt.excluded.return_5d,
            "return_20d": stmt.excluded.return_20d,
            "volume": stmt.excluded.volume,
            "trading_value": stmt.excluded.trading_value,
            "aum_amount": stmt.excluded.aum_amount,
            "relative_strength_score": stmt.excluded.relative_strength_score,
            "signal_score": stmt.excluded.signal_score,
            "signal_payload_json": stmt.excluded.signal_payload_json,
            "updated_at": text("now()"),
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def print_summary(*, as_of_date: str, etf_count: int, loaded_rows: int) -> None:
    print(
        "ETF signal calculation completed "
        f"as_of_date={as_of_date} etf_count={etf_count} upserted_rows={loaded_rows}"
    )


def main() -> int:
    setup_logging()
    args = parse_args()

    try:
        as_of_date, _ = parse_as_of_date(args.as_of_date)
        active_etfs = load_active_etfs()
        if not active_etfs:
            print_summary(as_of_date=as_of_date.isoformat(), etf_count=0, loaded_rows=0)
            return 0

        start_date = as_of_date - timedelta(days=LOOKBACK_DAYS)
        prices_raw_df = load_prices_raw_history(
            [str(etf["etf_code"]) for etf in active_etfs],
            start_date=start_date,
            as_of_date=as_of_date,
        )
        isin_map = build_isin_map()
        signal_df = build_signal_frame(
            active_etfs=active_etfs,
            as_of_date=as_of_date,
            prices_raw_df=prices_raw_df,
            isin_map=isin_map,
        )
        loaded_rows = upsert_etf_signals(signal_df)

        LOGGER.info(
            "ETF signal calculation finished as_of_date=%s active_etfs=%s built_rows=%s upserted_rows=%s",
            as_of_date,
            len(active_etfs),
            0 if signal_df.empty else len(signal_df),
            loaded_rows,
        )
        print_summary(as_of_date=as_of_date.isoformat(), etf_count=len(active_etfs), loaded_rows=loaded_rows)
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while calculating ETF signals: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("ETF signal calculation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
