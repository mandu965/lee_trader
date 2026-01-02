"""
rebalance_ranking.py

운용 규칙에 맞춰 research.prediction_history에서 최신 랭킹/포트폴리오를 생성한다.
- 메인 run_id: 90d (기본 41)
- 서브 run_id: 60d (기본 4)
- 보조 run_id: 30d (기본 40)

규칙 반영:
- 리밸런스: 주 1회 종가 기준 (스크립트 호출 시점의 최신 as_of_date 사용)
- pred_mdd 필터: -20% 이하 제외
- 변동성 필터: vol_20 상위 5% 제외 (features 최신일 기준, 옵션)
- 섹터 cap: 섹터당 40% cap (Top20 기준 최대 8개), 섹터 정보 없으면 스킵
- Top20 동가중, 종목당 5% cap
- 3주 연속 편입 시 이번 회차 제외(최근 2회 ranking_history 비교)

출력:
- outputs/rebalance/ranking_main.csv  (메인 run_id 기준 Top20)
"""
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

try:
    from db import get_engine
except Exception:
    get_engine = None

OUTPUT_DIR = Path("outputs/rebalance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID_MAIN = int(os.environ.get("RUN_ID_MAIN", "41"))  # 90d
RUN_ID_SUB = int(os.environ.get("RUN_ID_SUB", "4"))     # 60d
RUN_ID_AUX = int(os.environ.get("RUN_ID_AUX", "40"))    # 30d
TOP_N = int(os.environ.get("TOP_N", "20"))
PRED_MDD_THRESHOLD = -0.20  # -20% 이하 제외
SECTOR_CAP_RATIO = 0.40
FEATURES_CSV = Path(os.environ.get("FEATURES_CSV", "data/features.csv"))
UNIVERSE_CSV = Path(os.environ.get("UNIVERSE_CSV", "data/universe.csv"))
# score weights (aligned with adjusted scoring.py)
W_RET, W_PROB, W_QUAL, W_TECH, W_PRED = 0.28, 0.25, 0.20, 0.17, 0.10
PRED_SCORE_DEFAULT = 60.0


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def load_latest_predictions(run_id: int) -> pd.DataFrame:
    if not get_engine:
        raise RuntimeError("DB engine not available")
    eng = get_engine()
    with eng.connect() as conn:
        # latest as_of_date for run_id/horizon
        res = conn.execute(
            text(
                """
                SELECT MAX(as_of_date) AS as_of_date
                FROM research.prediction_history
                WHERE run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).fetchone()
        if not res or not res.as_of_date:
            raise RuntimeError(f"No prediction_history for run_id={run_id}")
        as_of = res.as_of_date
        logging.info("Latest as_of_date for run_id=%s: %s", run_id, as_of)
        df = pd.read_sql(
            text(
                """
                SELECT *
                FROM research.prediction_history
                WHERE run_id = :run_id AND as_of_date = :as_of
                """
            ),
            conn,
            params={"run_id": run_id, "as_of": as_of},
        )
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        return df


def apply_risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    # pred_mdd_xxd 컬럼 중 사용 가능한 것 하나 선택
    mdd_col = None
    for c in ["pred_mdd_90d", "pred_mdd_60d", "pred_mdd_30d"]:
        if c in df.columns:
            mdd_col = c
            break
    if not mdd_col:
        logging.warning("No pred_mdd column found; skip risk filter")
        return df
    filtered = df[pd.to_numeric(df[mdd_col], errors="coerce") >= PRED_MDD_THRESHOLD].copy()
    logging.info("Risk filter (mdd >= %.2f): %d -> %d rows", PRED_MDD_THRESHOLD, len(df), len(filtered))
    return filtered


def attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Attach sector from universe.csv or stocks table (if available)."""
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    sector_map = {}
    # Try Postgres stocks table
    if get_engine:
        try:
            eng = get_engine()
            sql = text("SELECT code, sector FROM stocks")
            rows = eng.execute(sql).fetchall()
            sector_map = {str(r.code).zfill(6): r.sector for r in rows if r and r.code is not None}
            logging.info("Loaded sector map from stocks table (rows=%d)", len(sector_map))
        except Exception:
            logging.warning("Failed to load sector from stocks table; fallback to universe.csv")
    if not sector_map and UNIVERSE_CSV.exists():
        try:
            uni = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
            if "code" in uni.columns and "sector" in uni.columns:
                uni["code"] = uni["code"].astype(str).str.zfill(6)
                sector_map = dict(zip(uni["code"], uni["sector"]))
                logging.info("Loaded sector map from universe.csv (rows=%d)", len(sector_map))
        except Exception:
            logging.warning("Failed to load universe.csv for sector", exc_info=True)
    if sector_map:
        df["sector"] = df["code"].map(sector_map)
    else:
        logging.info("Sector info not available; sector cap will be skipped")
        df["sector"] = None
    return df


def apply_vol_filter(df: pd.DataFrame, cutoff_pct: float = 0.95) -> pd.DataFrame:
    """Exclude codes in top X% vol_20 (from latest features). If features missing, skip."""
    if get_engine:
        try:
            eng = get_engine()
            sql = text(
                """
                WITH latest AS (
                  SELECT MAX(date) AS max_date FROM features
                )
                SELECT code, vol_20
                FROM features, latest
                WHERE date = latest.max_date
                """
            )
            feats = pd.read_sql(sql, eng, dtype={"code": str})
        except Exception:
            feats = pd.DataFrame()
    else:
        feats = pd.DataFrame()
    if feats.empty and FEATURES_CSV.exists():
        try:
            feats = pd.read_csv(FEATURES_CSV, dtype={"code": str})
        except Exception:
            feats = pd.DataFrame()
    if feats.empty or "vol_20" not in feats.columns:
        logging.info("Volatility data missing; skip vol filter")
        return df

    feats["code"] = feats["code"].astype(str).str.zfill(6)
    feats = feats.dropna(subset=["vol_20"])
    if feats.empty:
        logging.info("Volatility data empty after dropna; skip vol filter")
        return df
    threshold = feats["vol_20"].quantile(cutoff_pct)
    high_vol = set(feats[feats["vol_20"] >= threshold]["code"])
    before = len(df)
    df = df[~df["code"].isin(high_vol)].copy()
    logging.info("Vol filter (top %.0f%% vol_20): %d -> %d rows", (1 - cutoff_pct) * 100, before, len(df))
    return df


def apply_cooldown(df: pd.DataFrame, run_id: int, lookback_rebalances: int = 2) -> pd.DataFrame:
    """
    Exclude codes that appeared in TopN for the last `lookback_rebalances` as_of_date values (3회 연속 방지).
    """
    if not get_engine:
        logging.info("No DB engine; skip cooldown filter")
        return df
    try:
        eng = get_engine()
        recent_dates = pd.read_sql(
            text(
                """
                SELECT DISTINCT as_of_date
                FROM research.ranking_history
                WHERE run_id = :run_id
                ORDER BY as_of_date DESC
                LIMIT :n
                """
            ),
            eng,
            params={"run_id": run_id, "n": lookback_rebalances},
            parse_dates=["as_of_date"],
        )
        if recent_dates.empty:
            return df
        dates = recent_dates["as_of_date"].dt.date.tolist()
        rh = pd.read_sql(
            text(
                """
                SELECT code, as_of_date, in_top_n
                FROM research.ranking_history
                WHERE run_id = :run_id AND as_of_date = ANY(:dates)
                """
            ),
            eng,
            params={"run_id": run_id, "dates": dates},
            parse_dates=["as_of_date"],
        )
        if rh.empty:
            return df
        rh["code"] = rh["code"].astype(str).str.zfill(6)
        # codes present in all selected rebalances with in_top_n = True
        pivot = (
            rh[rh["in_top_n"] == True]
            .groupby("code")["as_of_date"]
            .nunique()
            .reset_index()
        )
        cooldown_codes = set(pivot[pivot["as_of_date"] >= lookback_rebalances]["code"])
        before = len(df)
        df = df[~df["code"].isin(cooldown_codes)].copy()
        logging.info("Cooldown filter (appeared in last %d rebalances): %d -> %d rows", lookback_rebalances, before, len(df))
        return df
    except Exception:
        logging.warning("Cooldown filter failed; skip", exc_info=True)
        return df


def apply_sector_cap(df: pd.DataFrame, top_n: int, cap_ratio: float) -> pd.DataFrame:
    if "sector" not in df.columns or df["sector"].isna().all():
        logging.info("Sector info missing; skip sector cap")
        return df
    max_per_sector = max(1, int(top_n * cap_ratio))
    kept = []
    sector_counts: Dict[str, int] = {}
    for _, row in df.sort_values("final_score_custom", ascending=False).iterrows():
        sector = row.get("sector")
        cnt = sector_counts.get(sector, 0)
        if cnt < max_per_sector and len(kept) < top_n:
            kept.append(row)
            sector_counts[sector] = cnt + 1
        if len(kept) >= top_n:
            break
    filtered = pd.DataFrame(kept)
    logging.info("Sector cap applied (max_per_sector=%d): %d -> %d rows", max_per_sector, len(df), len(filtered))
    return filtered


def build_topn(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Recompute custom score with adjusted weights
    df = df.copy()
    df["final_score_custom"] = (
        W_RET * df["ret_score"].fillna(0)
        + W_PROB * df["prob_score"].fillna(0)
        + W_QUAL * df["qual_score"].fillna(0)
        + W_TECH * df["tech_score"].fillna(0)
        + W_PRED * PRED_SCORE_DEFAULT
    )
    if "risk_penalty" in df.columns:
        df["final_score_custom"] = df["final_score_custom"] * df["risk_penalty"].fillna(1.0)
    df = df.sort_values("final_score_custom", ascending=False).head(top_n).copy()
    # equal weight capped at 5%
    weight = min(0.05, 1.0 / top_n if top_n > 0 else 0)
    df["weight"] = weight
    return df


def save_ranking(df: pd.DataFrame, name: str, run_id: int | None = None) -> Path:
    suffix = name
    if run_id is not None:
        suffix = f"{suffix}_run{run_id}"
    if "horizon_days" in df.columns and not df.empty:
        try:
            h = int(df["horizon_days"].iloc[0])
            suffix = f"{suffix}_h{h}"
        except Exception:
            pass
    out = OUTPUT_DIR / f"ranking_{suffix}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    logging.info("Saved ranking (%s): %s (rows=%d)", name, out, len(df))
    return out


def main() -> None:
    setup_logging()
    logging.info("Rebalance ranking start (main run_id=%s)", RUN_ID_MAIN)
    # main run
    df_main = load_latest_predictions(RUN_ID_MAIN)
    df_main = apply_risk_filter(df_main)
    # merge sector info
    df_main = attach_sector(df_main)
    # volatility filter
    df_main = apply_vol_filter(df_main)
    # cooldown filter using ranking_history
    df_main = apply_cooldown(df_main, RUN_ID_MAIN, lookback_rebalances=2)
    top_main = build_topn(df_main, TOP_N)
    top_main = apply_sector_cap(top_main, TOP_N, SECTOR_CAP_RATIO)
    top_main["weight"] = min(0.05, 1.0 / len(top_main)) if len(top_main) > 0 else 0
    save_ranking(top_main, "main", run_id=RUN_ID_MAIN)
    logging.info("Rebalance ranking done.")


if __name__ == "__main__":
    main()
