"""
Financial Momentum Feature Builder — Phase 2

Phase 1(fetch_financials_dart_quarterly.py)이 수집한 DART 분기 누적 재무 데이터를
읽어 재무 모멘텀 feature를 생성한다.

처리 단계:
  1. 누적값 → 진짜 분기값 역산 (P&L 항목: revenue, op_income, net_income, ocf)
  2. 분기 영업이익률(op_margin) 재계산 (누적 아닌 진짜 분기 기준)
  3. YoY 성장률 및 전환 flag 계산
  4. QoQ 성장률 및 둔화 flag 계산
  5. 연속 둔화 횟수 계산
  6. Financial Momentum Phase 분류 (ACCELERATING/GROWING/SLOWING/WEAKENING/DECLINING/TURNAROUND)
  7. 재무 모멘텀 점수 계산 (financial_momentum_score, financial_risk_score, turnaround_score)
  8. 위험 flag 설정 (fundamental_decline_flag, hard_fundamental_risk)

출력:
  - data/dart/financial_momentum_quarterly.csv
  - DB 테이블 financial_momentum_quarterly (Postgres 우선, SQLite fallback)

주의:
  - disclosed_at 기준 point-in-time 원칙: 해당 날짜 이후에만 feature 사용
  - is_sector_exception 종목(금융/바이오 등)은 Phase 미분류 (Phase 4에서 처리)
  - 설계 문서: doc/modules/Lee_trader_ai/FINANCIAL_MOMENTUM_DESIGN.md
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from db import get_engine, replace_table_rows_pg, use_sqlite_fallback_writes
except Exception:
    get_engine = None
    replace_table_rows_pg = None

    def use_sqlite_fallback_writes() -> bool:
        return False


# ---------------------------------------------------------------------------
# 상수 / 경로
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "dart"
INPUT_CSV = CACHE_DIR / "financial_quarterly.csv"
OUTPUT_CSV = CACHE_DIR / "financial_momentum_quarterly.csv"
DB_PATH = DATA_DIR / "lee_trader.db"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

INPUT_DB_TABLE = "financial_statement_quarterly"
OUTPUT_DB_TABLE = "financial_momentum_quarterly"

# P&L 항목: DART는 연초 누적값을 제공 → 진짜 분기값 역산 필요
CUMULATIVE_COLS = ["revenue", "op_income", "net_income", "ocf"]
# Balance sheet: 기말 시점값 → 그대로 사용
AS_OF_COLS = ["assets", "liabilities", "equity"]

QUARTER_NUM = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# 환경변수로 조정 가능한 임계값
OP_MARGIN_DROP_THRESHOLD = float(os.getenv("FINANCIAL_OP_MARGIN_DROP_THRESHOLD", "-2.0"))
SLOWDOWN_CONSECUTIVE_QUARTERS = int(os.getenv("FINANCIAL_SLOWDOWN_CONSECUTIVE_QUARTERS", "2"))

# 업종 예외 코드 (금융, 바이오 등): 향후 universe.csv / DB에서 읽도록 확장 예정
SECTOR_EXCEPTION_NAMES: List[str] = [
    "은행", "보험", "증권", "금융투자", "금융",
    "바이오", "제약", "의약품",
]

OUTPUT_COLUMNS: List[str] = [
    "stock_code", "fiscal_year", "quarter",
    "source_report_date", "disclosed_at", "fs_div",
    # 진짜 분기값 (누적 역산)
    "true_revenue", "true_op_income", "true_net_income", "true_ocf",
    # Balance sheet (시점값)
    "assets", "liabilities", "equity",
    # 분기 영업이익률 (진짜 분기 기준 재계산)
    "op_margin",
    # YoY 성장률
    "revenue_yoy", "op_profit_yoy", "op_margin_change_yoy",
    # QoQ 성장률
    "revenue_qoq", "op_profit_qoq", "op_margin_change_qoq",
    # 둔화 flag
    "revenue_yoy_slowdown", "op_profit_yoy_slowdown",
    "revenue_slowdown_count", "op_profit_slowdown_count",
    # 전환 flag
    "revenue_negative_turn", "op_profit_negative_turn",
    "op_profit_turnaround_flag", "op_profit_loss_expansion_flag", "op_profit_loss_reduction_flag",
    # Phase 분류 및 점수
    "financial_momentum_phase",
    "financial_momentum_score", "financial_risk_score", "turnaround_score",
    # 위험 flag
    "fundamental_decline_flag", "hard_fundamental_risk",
    # 업종 예외
    "is_sector_exception",
]


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_quarterly() -> pd.DataFrame:
    """Phase 1 수집 데이터 로드 (DB 우선, CSV fallback)"""
    try:
        if get_engine is not None:
            eng = get_engine()
            with eng.connect() as conn:
                df = pd.read_sql(
                    f"SELECT * FROM {INPUT_DB_TABLE}",
                    conn,
                    dtype={"stock_code": str},
                )
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
            logging.info("Loaded %d rows from DB: %s", len(df), INPUT_DB_TABLE)
            return df
    except Exception:
        pass

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"quarterly data not found: {INPUT_CSV}. "
            "fetch_financials_dart_quarterly.py를 먼저 실행하세요."
        )
    df = pd.read_csv(INPUT_CSV, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    logging.info("Loaded %d rows from CSV: %s", len(df), INPUT_CSV)
    return df


# ---------------------------------------------------------------------------
# Step 1: 누적값 → 진짜 분기값 역산
# ---------------------------------------------------------------------------

def _deaccumulate(df: pd.DataFrame) -> pd.DataFrame:
    """DART 누적 P&L 항목을 진짜 분기값으로 역산.

    Q1_true = Q1_cum
    Q2_true = Q2_cum - Q1_cum  (Q1이 없으면 NaN)
    Q3_true = Q3_cum - Q2_cum  (Q2이 없으면 NaN)
    Q4_true = Q4_cum - Q3_cum  (Q3이 없으면 NaN)
    """
    df = df.copy()
    df["quarter_num"] = df["quarter"].map(QUARTER_NUM)
    df = df.sort_values(["stock_code", "fiscal_year", "quarter_num"]).reset_index(drop=True)

    for col in CUMULATIVE_COLS:
        if col not in df.columns:
            df[f"true_{col}"] = np.nan
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        prev_val = df.groupby(["stock_code", "fiscal_year"])[col].shift(1)
        prev_qnum = df.groupby(["stock_code", "fiscal_year"])["quarter_num"].shift(1)

        is_q1 = df["quarter_num"] == 1
        is_sequential = (df["quarter_num"] - prev_qnum) == 1

        df[f"true_{col}"] = np.where(
            is_q1,
            df[col],
            np.where(
                is_sequential & prev_val.notna() & df[col].notna(),
                df[col] - prev_val,
                np.nan,
            ),
        )

    for col in AS_OF_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Step 2: 분기 영업이익률 재계산
# ---------------------------------------------------------------------------

def _compute_op_margin(df: pd.DataFrame) -> pd.DataFrame:
    """true_revenue / true_op_income 기준으로 op_margin 재계산.

    Phase 1의 op_margin은 누적 기준이므로, 진짜 분기값 기준으로 재계산한다.
    """
    df = df.copy()
    rev = pd.to_numeric(df["true_revenue"], errors="coerce")
    op = pd.to_numeric(df["true_op_income"], errors="coerce")
    valid = rev.notna() & op.notna() & (rev != 0)
    df["op_margin"] = np.where(valid, op / rev, np.nan)
    return df


# ---------------------------------------------------------------------------
# Step 3: YoY 성장률 및 전환 flag
# ---------------------------------------------------------------------------

def _compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """전년 동기 대비 성장률 및 전환 flag 계산.

    YoY는 같은 quarter 기준으로 직전 fiscal_year 데이터와 비교한다.
    적자↔흑자 전환 시 YoY 계산 대신 flag를 사용한다.
    """
    df = df.copy()

    # 전년 동기 데이터 merge (fiscal_year + 1로 shift하여 현재 행에 붙임)
    yoy_src = df[["stock_code", "fiscal_year", "quarter",
                   "true_revenue", "true_op_income", "op_margin"]].copy()
    yoy_src = yoy_src.rename(columns={
        "true_revenue":   "py_revenue",
        "true_op_income": "py_op_income",
        "op_margin":      "py_op_margin",
    })
    yoy_src["fiscal_year"] = yoy_src["fiscal_year"] + 1

    df = df.merge(yoy_src, on=["stock_code", "fiscal_year", "quarter"], how="left")

    curr_rev = pd.to_numeric(df["true_revenue"], errors="coerce")
    prev_rev = pd.to_numeric(df["py_revenue"], errors="coerce")
    curr_op  = pd.to_numeric(df["true_op_income"], errors="coerce")
    prev_op  = pd.to_numeric(df["py_op_income"], errors="coerce")
    curr_om  = pd.to_numeric(df["op_margin"], errors="coerce")
    prev_om  = pd.to_numeric(df["py_op_margin"], errors="coerce")

    both_valid_rev = curr_rev.notna() & prev_rev.notna() & (prev_rev != 0)
    both_valid_op  = curr_op.notna() & prev_op.notna()

    # Revenue YoY
    df["revenue_yoy"] = np.where(both_valid_rev, (curr_rev - prev_rev) / prev_rev.abs(), np.nan)
    df["revenue_negative_turn"] = (
        curr_rev.notna() & prev_rev.notna() & (prev_rev > 0) & (curr_rev < 0)
    )

    # Op profit 전환 flag
    df["op_profit_turnaround_flag"]     = (both_valid_op & (prev_op < 0) & (curr_op > 0))
    df["op_profit_negative_turn"]       = (both_valid_op & (prev_op > 0) & (curr_op < 0))
    df["op_profit_loss_reduction_flag"] = (
        both_valid_op & (prev_op < 0) & (curr_op < 0) & (curr_op.abs() < prev_op.abs())
    )
    df["op_profit_loss_expansion_flag"] = (
        both_valid_op & (prev_op < 0) & (curr_op < 0) & (curr_op.abs() >= prev_op.abs())
    )

    # Op profit YoY: 적자→흑자 전환 시 제외, 그 외 계산
    normal_op = both_valid_op & ~df["op_profit_turnaround_flag"] & (prev_op != 0)
    df["op_profit_yoy"] = np.where(normal_op, (curr_op - prev_op) / prev_op.abs(), np.nan)

    # Op margin change YoY
    df["op_margin_change_yoy"] = np.where(
        curr_om.notna() & prev_om.notna(), curr_om - prev_om, np.nan
    )

    # 전년 데이터 임시 컬럼 정리 (QoQ step에서 필요하므로 이름만 변경)
    df = df.rename(columns={
        "py_revenue":   "_py_revenue",
        "py_op_income": "_py_op_income",
        "py_op_margin": "_py_op_margin",
    })

    return df


# ---------------------------------------------------------------------------
# Step 4: QoQ 성장률 및 둔화 flag
# ---------------------------------------------------------------------------

def _compute_qoq(df: pd.DataFrame) -> pd.DataFrame:
    """직전 분기 대비 성장률 및 YoY 둔화 flag 계산.

    QoQ는 period(fiscal_year*4 + quarter_num) 기준으로 직전 period 데이터와 비교한다.
    Q4→Q1(다음 연도) 경계를 자연스럽게 처리한다.
    """
    df = df.copy()
    df["period"] = df["fiscal_year"] * 4 + df["quarter_num"]

    # 직전 period 데이터 merge
    qoq_src = df[["stock_code", "period",
                   "true_revenue", "true_op_income", "op_margin",
                   "revenue_yoy", "op_profit_yoy"]].copy()
    qoq_src = qoq_src.rename(columns={
        "true_revenue":   "pq_revenue",
        "true_op_income": "pq_op_income",
        "op_margin":      "pq_op_margin",
        "revenue_yoy":    "pq_revenue_yoy",
        "op_profit_yoy":  "pq_op_profit_yoy",
    })
    qoq_src["period"] = qoq_src["period"] + 1  # 직전 period → 현재 period로 올려서 merge

    df = df.merge(qoq_src, on=["stock_code", "period"], how="left")

    curr_rev = pd.to_numeric(df["true_revenue"], errors="coerce")
    prev_rev = pd.to_numeric(df["pq_revenue"], errors="coerce")
    curr_op  = pd.to_numeric(df["true_op_income"], errors="coerce")
    prev_op  = pd.to_numeric(df["pq_op_income"], errors="coerce")
    curr_om  = pd.to_numeric(df["op_margin"], errors="coerce")
    prev_om  = pd.to_numeric(df["pq_op_margin"], errors="coerce")

    # Revenue QoQ
    valid_rev = curr_rev.notna() & prev_rev.notna() & (prev_rev != 0)
    df["revenue_qoq"] = np.where(valid_rev, (curr_rev - prev_rev) / prev_rev.abs(), np.nan)

    # Op profit QoQ
    valid_op = curr_op.notna() & prev_op.notna() & (prev_op != 0)
    df["op_profit_qoq"] = np.where(valid_op, (curr_op - prev_op) / prev_op.abs(), np.nan)

    # Op margin change QoQ
    df["op_margin_change_qoq"] = np.where(
        curr_om.notna() & prev_om.notna(), curr_om - prev_om, np.nan
    )

    # YoY 둔화 flag: 이번 분기 YoY < 직전 분기 YoY
    curr_ry  = pd.to_numeric(df["revenue_yoy"], errors="coerce")
    prev_ry  = pd.to_numeric(df["pq_revenue_yoy"], errors="coerce")
    curr_opy = pd.to_numeric(df["op_profit_yoy"], errors="coerce")
    prev_opy = pd.to_numeric(df["pq_op_profit_yoy"], errors="coerce")

    df["revenue_yoy_slowdown"] = np.where(
        curr_ry.notna() & prev_ry.notna(), curr_ry < prev_ry, False
    )
    df["op_profit_yoy_slowdown"] = np.where(
        curr_opy.notna() & prev_opy.notna(), curr_opy < prev_opy, False
    )

    return df


# ---------------------------------------------------------------------------
# Step 5: 연속 둔화 횟수
# ---------------------------------------------------------------------------

def _compute_slowdown_counts(df: pd.DataFrame) -> pd.DataFrame:
    """연속 YoY 둔화 분기 수 계산. 둔화가 끊기면 0으로 초기화."""

    def consecutive_true(series: pd.Series) -> pd.Series:
        counts, c = [], 0
        for val in series:
            if pd.isna(val) or not bool(val):
                c = 0
            else:
                c += 1
            counts.append(c)
        return pd.Series(counts, index=series.index, dtype=int)

    df = df.copy()
    df = df.sort_values(["stock_code", "period"])

    for flag_col, count_col in [
        ("revenue_yoy_slowdown", "revenue_slowdown_count"),
        ("op_profit_yoy_slowdown", "op_profit_slowdown_count"),
    ]:
        df[count_col] = (
            df.groupby("stock_code")[flag_col]
            .transform(consecutive_true)
        )

    return df


# ---------------------------------------------------------------------------
# Step 6: 업종 예외 설정
# ---------------------------------------------------------------------------

def _add_sector_exception(df: pd.DataFrame) -> pd.DataFrame:
    """업종 예외 종목 식별. universe.csv에 sector 컬럼이 있으면 사용.

    금융, 바이오 등 재무 모멘텀 로직이 왜곡될 수 있는 업종을 is_sector_exception으로 표시.
    Phase 4 이전에는 해당 종목의 Phase를 SECTOR_EXCEPTION으로 설정만 하고 scoring은 skip.
    """
    df = df.copy()
    df["is_sector_exception"] = False

    if not UNIVERSE_CSV.exists():
        return df

    try:
        uni = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
        if "sector" not in uni.columns:
            return df

        uni["code"] = uni["code"].astype(str).str.zfill(6)
        uni = uni[["code", "sector"]].dropna(subset=["sector"]).drop_duplicates(subset=["code"])
        sector_map = dict(zip(uni["code"], uni["sector"]))

        sectors = df["stock_code"].map(sector_map).fillna("")
        is_exception = sectors.str.contains("|".join(SECTOR_EXCEPTION_NAMES), na=False)
        df["is_sector_exception"] = is_exception
        logging.info("Sector exception count: %d", is_exception.sum())
    except Exception as e:
        logging.warning("Failed to load sector info: %s", e)

    return df


# ---------------------------------------------------------------------------
# Step 7: Financial Momentum Phase 분류
# ---------------------------------------------------------------------------

def _classify_phase(df: pd.DataFrame) -> pd.DataFrame:
    """ACCELERATING / GROWING / SLOWING / WEAKENING / DECLINING / TURNAROUND 분류.

    적용 순서 (먼저 해당하는 것 채택, 설계문서 섹션 7 기준):
      1. is_sector_exception → SECTOR_EXCEPTION
      2. op_profit_turnaround_flag AND revenue_qoq > 0 → TURNAROUND
      3. rv_yoy > 0 AND op_yoy > 0 AND 모두 가속 AND margin 개선 → ACCELERATING
      4. rv_yoy > 0 AND op_yoy > 0 AND margin 개선 → GROWING
      5. rv_yoy > 0 AND op_yoy > 0 AND 둔화 → SLOWING
      6. rv_yoy > 0 AND (op 역전 OR margin 급락) → WEAKENING
      7. rv_yoy < 0 AND op_yoy < 0 AND QoQ 개선 → TURNAROUND
      8. rv_yoy < 0 AND op_yoy < 0 → DECLINING
    """
    df = df.copy()

    rv_yoy     = pd.to_numeric(df["revenue_yoy"], errors="coerce")
    op_yoy     = pd.to_numeric(df["op_profit_yoy"], errors="coerce")
    rv_qoq     = pd.to_numeric(df["revenue_qoq"], errors="coerce")
    op_qoq     = pd.to_numeric(df["op_profit_qoq"], errors="coerce")
    om_chg_yoy = pd.to_numeric(df["op_margin_change_yoy"], errors="coerce")

    ta_flag  = df["op_profit_turnaround_flag"].fillna(False).astype(bool)
    lr_flag  = df["op_profit_loss_reduction_flag"].fillna(False).astype(bool)
    nt_flag  = df["op_profit_negative_turn"].fillna(False).astype(bool)
    rv_slow  = df["revenue_yoy_slowdown"].fillna(False).astype(bool)
    op_slow  = df["op_profit_yoy_slowdown"].fillna(False).astype(bool)
    is_exc   = df["is_sector_exception"].fillna(False).astype(bool)

    # 가속 판단: 이번 분기 YoY > 직전 분기 YoY
    pq_rv_yoy = pd.to_numeric(df.get("pq_revenue_yoy", pd.Series(np.nan, index=df.index)), errors="coerce")
    pq_op_yoy = pd.to_numeric(df.get("pq_op_profit_yoy", pd.Series(np.nan, index=df.index)), errors="coerce")
    rv_accel  = rv_yoy.notna() & pq_rv_yoy.notna() & (rv_yoy > pq_rv_yoy)
    op_accel  = op_yoy.notna() & pq_op_yoy.notna() & (op_yoy > pq_op_yoy)

    # np.select로 우선순위 순서 적용 (첫 번째 True 조건 채택)
    conditions = [
        is_exc,
        ~is_exc & ta_flag & (rv_qoq > 0),
        ~is_exc & (rv_yoy > 0) & (op_yoy > 0) & rv_accel & op_accel & (om_chg_yoy > 0),
        ~is_exc & (rv_yoy > 0) & (op_yoy > 0) & (om_chg_yoy >= 0),
        ~is_exc & (rv_yoy > 0) & (op_yoy > 0) & (rv_slow | op_slow),
        ~is_exc & (rv_yoy > 0) & (nt_flag | (om_chg_yoy < OP_MARGIN_DROP_THRESHOLD)),
        ~is_exc & (rv_yoy < 0) & (op_yoy < 0) & ((rv_qoq > 0) | (op_qoq > 0) | lr_flag),
        ~is_exc & (rv_yoy < 0) & (op_yoy < 0),
    ]
    choices = [
        "SECTOR_EXCEPTION",
        "TURNAROUND",
        "ACCELERATING",
        "GROWING",
        "SLOWING",
        "WEAKENING",
        "TURNAROUND",
        "DECLINING",
    ]

    df["financial_momentum_phase"] = np.select(conditions, choices, default="UNKNOWN")
    return df


# ---------------------------------------------------------------------------
# Step 8: 점수 및 위험 flag 계산
# ---------------------------------------------------------------------------

def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """financial_momentum_score / financial_risk_score / turnaround_score 및 위험 flag.

    설계문서 섹션 8, 9 기준.
    """
    df = df.copy()

    rv_yoy     = pd.to_numeric(df["revenue_yoy"], errors="coerce")
    op_yoy     = pd.to_numeric(df["op_profit_yoy"], errors="coerce")
    om_chg_yoy = pd.to_numeric(df["op_margin_change_yoy"], errors="coerce")
    rv_qoq     = pd.to_numeric(df["revenue_qoq"], errors="coerce")
    op_qoq     = pd.to_numeric(df["op_profit_qoq"], errors="coerce")
    om_chg_qoq = pd.to_numeric(df["op_margin_change_qoq"], errors="coerce")
    rv_sc      = pd.to_numeric(df["revenue_slowdown_count"], errors="coerce").fillna(0)
    op_sc      = pd.to_numeric(df["op_profit_slowdown_count"], errors="coerce").fillna(0)

    ta_flag = df["op_profit_turnaround_flag"].fillna(False).astype(bool)
    lr_flag = df["op_profit_loss_reduction_flag"].fillna(False).astype(bool)
    rv_slow = df["revenue_yoy_slowdown"].fillna(False).astype(bool)
    op_slow = df["op_profit_yoy_slowdown"].fillna(False).astype(bool)
    nt_flag = df["op_profit_negative_turn"].fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # financial_momentum_score (0~100, 높을수록 좋음)
    # ------------------------------------------------------------------
    score = pd.Series(50.0, index=df.index)
    score += np.where(rv_yoy > 0, 10, 0)
    score += np.where(op_yoy > 0, 15, 0)
    # 영업레버리지: 영업이익 증가율 > 매출 증가율
    leverage = rv_yoy.notna() & op_yoy.notna() & (op_yoy > rv_yoy)
    score += np.where(leverage, 10, 0)
    score += np.where(om_chg_yoy > 0, 10, 0)
    score += np.where(rv_slow, -10, 0)
    score += np.where(op_slow, -15, 0)
    score += np.where(rv_yoy < 0, -15, 0)
    score += np.where(op_yoy < 0, -20, 0)
    score += np.where(om_chg_yoy < OP_MARGIN_DROP_THRESHOLD, -15, 0)
    score += np.where(rv_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS, -10, 0)
    score += np.where(op_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS, -15, 0)
    df["financial_momentum_score"] = score.clip(0, 100)

    # ------------------------------------------------------------------
    # financial_risk_score (0~100, 높을수록 위험)
    # ------------------------------------------------------------------
    risk = pd.Series(0.0, index=df.index)
    risk += np.where(rv_yoy < 0, 20, 0)
    risk += np.where(op_yoy < 0, 25, 0)
    risk += np.where(om_chg_yoy < OP_MARGIN_DROP_THRESHOLD, 15, 0)
    risk += np.where(rv_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS, 10, 0)
    risk += np.where(op_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS, 15, 0)
    # 매출·영업이익 동시 감소 가중
    risk += np.where((rv_yoy < 0) & (op_yoy < 0), 20, 0)
    df["financial_risk_score"] = risk.clip(0, 100)

    # ------------------------------------------------------------------
    # turnaround_score (0~100)
    # ------------------------------------------------------------------
    tscore = pd.Series(0.0, index=df.index)
    tscore += np.where(ta_flag, 35, 0)
    tscore += np.where(lr_flag, 20, 0)
    tscore += np.where(rv_qoq > 0, 10, 0)
    tscore += np.where(op_qoq > 0, 20, 0)
    tscore += np.where(om_chg_qoq > 0, 15, 0)
    df["turnaround_score"] = tscore.clip(0, 100)

    # ------------------------------------------------------------------
    # 위험 flag (섹션 9 기준)
    # ------------------------------------------------------------------
    df["fundamental_decline_flag"] = (
        (rv_yoy < 0)
        | (op_yoy < 0)
        | (rv_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS)
        | (op_sc >= SLOWDOWN_CONSECUTIVE_QUARTERS)
        | (om_chg_yoy < OP_MARGIN_DROP_THRESHOLD)
    )

    df["hard_fundamental_risk"] = (
        ((rv_yoy < 0) & (op_yoy < 0) & (om_chg_yoy < 0))
        | (nt_flag & (rv_yoy <= 0))
    )

    return df


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """전체 feature 빌드 파이프라인."""
    logging.info("Step 1: de-accumulate quarterly P&L")
    df = _deaccumulate(df)

    logging.info("Step 2: compute op_margin from true quarterly")
    df = _compute_op_margin(df)

    logging.info("Step 3: compute YoY features")
    df = _compute_yoy(df)

    logging.info("Step 4: compute QoQ features")
    # period 키 추가 (QoQ 계산에 사용)
    if "quarter_num" not in df.columns:
        df["quarter_num"] = df["quarter"].map(QUARTER_NUM)
    df["period"] = df["fiscal_year"] * 4 + df["quarter_num"]
    df = _compute_qoq(df)

    logging.info("Step 5: compute slowdown counts")
    df = _compute_slowdown_counts(df)

    logging.info("Step 6: add sector exception flags")
    df = _add_sector_exception(df)

    logging.info("Step 7: classify financial momentum phase")
    df = _classify_phase(df)

    logging.info("Step 8: compute scores and risk flags")
    df = _compute_scores(df)

    # 출력 컬럼 선택 및 정리
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out = df[OUTPUT_COLUMNS].copy()
    # Q1 < Q2 < Q3 < Q4 알파벳 순 = 분기 순이므로 quarter 기준 정렬 가능
    out = out.sort_values(["stock_code", "fiscal_year", "quarter"]).reset_index(drop=True)

    # 위상 통계 요약 로그
    phase_counts = out["financial_momentum_phase"].value_counts()
    logging.info("Phase distribution:\n%s", phase_counts.to_string())
    logging.info(
        "hard_fundamental_risk: %d / fundamental_decline_flag: %d",
        out["hard_fundamental_risk"].sum(),
        out["fundamental_decline_flag"].sum(),
    )

    return out


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def save_momentum(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    # CSV 저장
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logging.info("Saved CSV: %s (rows=%d)", OUTPUT_CSV.resolve(), len(out))

    # Postgres 저장
    try:
        if replace_table_rows_pg is not None and get_engine is not None:
            eng = get_engine()
            with eng.begin() as conn:
                from sqlalchemy import text
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS financial_momentum_quarterly (
                        stock_code                  TEXT NOT NULL,
                        fiscal_year                 INTEGER NOT NULL,
                        quarter                     TEXT NOT NULL,
                        source_report_date          DATE,
                        disclosed_at                DATE,
                        fs_div                      TEXT,
                        true_revenue                DOUBLE PRECISION,
                        true_op_income              DOUBLE PRECISION,
                        true_net_income             DOUBLE PRECISION,
                        true_ocf                    DOUBLE PRECISION,
                        assets                      DOUBLE PRECISION,
                        liabilities                 DOUBLE PRECISION,
                        equity                      DOUBLE PRECISION,
                        op_margin                   DOUBLE PRECISION,
                        revenue_yoy                 DOUBLE PRECISION,
                        op_profit_yoy               DOUBLE PRECISION,
                        op_margin_change_yoy        DOUBLE PRECISION,
                        revenue_qoq                 DOUBLE PRECISION,
                        op_profit_qoq               DOUBLE PRECISION,
                        op_margin_change_qoq        DOUBLE PRECISION,
                        revenue_yoy_slowdown        BOOLEAN,
                        op_profit_yoy_slowdown      BOOLEAN,
                        revenue_slowdown_count      INTEGER,
                        op_profit_slowdown_count    INTEGER,
                        revenue_negative_turn       BOOLEAN,
                        op_profit_negative_turn     BOOLEAN,
                        op_profit_turnaround_flag   BOOLEAN,
                        op_profit_loss_expansion_flag BOOLEAN,
                        op_profit_loss_reduction_flag BOOLEAN,
                        financial_momentum_phase    TEXT,
                        financial_momentum_score    DOUBLE PRECISION,
                        financial_risk_score        DOUBLE PRECISION,
                        turnaround_score            DOUBLE PRECISION,
                        fundamental_decline_flag    BOOLEAN,
                        hard_fundamental_risk       BOOLEAN,
                        is_sector_exception         BOOLEAN,
                        created_at                  TIMESTAMP DEFAULT now(),
                        updated_at                  TIMESTAMP DEFAULT now(),
                        PRIMARY KEY (stock_code, fiscal_year, quarter)
                    )
                """))
            replace_table_rows_pg(OUTPUT_DB_TABLE, out, columns=OUTPUT_COLUMNS)
            logging.info("Saved to Postgres: %s (rows=%d)", OUTPUT_DB_TABLE, len(out))
            return
    except Exception:
        logging.exception("Postgres save failed, fallback to SQLite")

    if not use_sqlite_fallback_writes():
        logging.info("SQLite fallback disabled (USE_SQLITE_FALLBACK_WRITES=0)")
        return

    _save_sqlite(out)


def _save_sqlite(df: pd.DataFrame) -> None:
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_momentum_quarterly (
                stock_code                  TEXT NOT NULL,
                fiscal_year                 INTEGER NOT NULL,
                quarter                     TEXT NOT NULL,
                source_report_date          TEXT,
                disclosed_at                TEXT,
                fs_div                      TEXT,
                true_revenue                REAL,
                true_op_income              REAL,
                true_net_income             REAL,
                true_ocf                    REAL,
                assets                      REAL,
                liabilities                 REAL,
                equity                      REAL,
                op_margin                   REAL,
                revenue_yoy                 REAL,
                op_profit_yoy               REAL,
                op_margin_change_yoy        REAL,
                revenue_qoq                 REAL,
                op_profit_qoq               REAL,
                op_margin_change_qoq        REAL,
                revenue_yoy_slowdown        INTEGER,
                op_profit_yoy_slowdown      INTEGER,
                revenue_slowdown_count      INTEGER,
                op_profit_slowdown_count    INTEGER,
                revenue_negative_turn       INTEGER,
                op_profit_negative_turn     INTEGER,
                op_profit_turnaround_flag   INTEGER,
                op_profit_loss_expansion_flag INTEGER,
                op_profit_loss_reduction_flag INTEGER,
                financial_momentum_phase    TEXT,
                financial_momentum_score    REAL,
                financial_risk_score        REAL,
                turnaround_score            REAL,
                fundamental_decline_flag    INTEGER,
                hard_fundamental_risk       INTEGER,
                is_sector_exception         INTEGER,
                PRIMARY KEY (stock_code, fiscal_year, quarter)
            )
        """)

        # 기존 테이블에 누락 컬럼 추가 (migration)
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(financial_momentum_quarterly)").fetchall()}
        new_col_defs = {
            "true_revenue": "REAL", "true_op_income": "REAL", "true_net_income": "REAL", "true_ocf": "REAL",
            "assets": "REAL", "liabilities": "REAL", "equity": "REAL",
            "op_margin": "REAL",
            "revenue_yoy": "REAL", "op_profit_yoy": "REAL", "op_margin_change_yoy": "REAL",
            "revenue_qoq": "REAL", "op_profit_qoq": "REAL", "op_margin_change_qoq": "REAL",
            "revenue_yoy_slowdown": "INTEGER", "op_profit_yoy_slowdown": "INTEGER",
            "revenue_slowdown_count": "INTEGER", "op_profit_slowdown_count": "INTEGER",
            "revenue_negative_turn": "INTEGER", "op_profit_negative_turn": "INTEGER",
            "op_profit_turnaround_flag": "INTEGER", "op_profit_loss_expansion_flag": "INTEGER",
            "op_profit_loss_reduction_flag": "INTEGER",
            "financial_momentum_phase": "TEXT",
            "financial_momentum_score": "REAL", "financial_risk_score": "REAL", "turnaround_score": "REAL",
            "fundamental_decline_flag": "INTEGER", "hard_fundamental_risk": "INTEGER",
            "is_sector_exception": "INTEGER",
        }
        for col, col_type in new_col_defs.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE financial_momentum_quarterly ADD COLUMN {col} {col_type}")

        placeholders = ", ".join(f":{c}" for c in OUTPUT_COLUMNS)
        col_names = ", ".join(OUTPUT_COLUMNS)
        records = df[OUTPUT_COLUMNS].where(pd.notnull(df[OUTPUT_COLUMNS]), None).to_dict(orient="records")
        conn.executemany(
            f"INSERT OR REPLACE INTO financial_momentum_quarterly ({col_names}) VALUES ({placeholders})",
            records,
        )
        conn.commit()
        logging.info("Saved to SQLite: financial_momentum_quarterly (rows=%d)", len(df))
    except Exception:
        logging.exception("SQLite save failed")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Financial Momentum Feature Builder (Phase 2)")
    p.add_argument(
        "--codes", type=str, default="",
        help="쉼표 구분 종목코드 필터 (미지정 시 전체)",
    )
    p.add_argument(
        "--min-year", type=int, default=0,
        help="처리할 최소 fiscal_year (0=전체)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="계산만 하고 저장하지 않음",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    load_dotenv()
    args = parse_args()

    df = load_quarterly()

    if args.codes:
        target = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        df = df[df["stock_code"].isin(target)]
        logging.info("Filtered to %d codes: %d rows", len(target), len(df))

    if args.min_year > 0:
        df = df[pd.to_numeric(df["fiscal_year"], errors="coerce") >= args.min_year]
        logging.info("Filtered fiscal_year >= %d: %d rows", args.min_year, len(df))

    if df.empty:
        logging.warning("No data to process. 종료.")
        return

    result = build_features(df)

    if args.dry_run:
        logging.info("Dry-run mode: 저장 생략. rows=%d", len(result))
        print(result[["stock_code", "fiscal_year", "quarter",
                       "financial_momentum_phase", "financial_momentum_score",
                       "financial_risk_score", "revenue_yoy", "op_profit_yoy"]].to_string(max_rows=30))
        return

    save_momentum(result)
    logging.info("완료: financial_momentum_quarterly rows=%d", len(result))


if __name__ == "__main__":
    main()
