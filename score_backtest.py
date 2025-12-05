# analysis/score_backtest.py

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


# 1) 경로 설정
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # lee_trader 루트
# DATA_DIR = os.path.join(BASE_DIR, "data")

# PRICES_PATH = os.path.join(DATA_DIR, "prices", "prices_daily.csv")

BASE_DIR = Path(__file__).resolve().parent  # lee_trader 루트
DATA_DIR = BASE_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"
PRICES_PATH = PRICES_DIR / "prices_daily.csv"
HISTORY_DIR = os.path.join(DATA_DIR, "history")

HORIZON_DAYS = 60       # 60일 수익률
TOP_N = 5               # 상위 N개 종목만 매수했다고 가정


# 2) 가격 데이터 로드
prices = pd.read_csv(PRICES_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

# code별로 인덱스 붙이기 (빠르게 D+60날짜 찾으려고)
prices_grouped = prices.groupby("code")


def get_forward_return(code: str, trade_date: pd.Timestamp, horizon_days: int):
  """trade_date 기준 horizon_days 후의 실제 수익률 계산 (단순 종가 기준)."""
  if code not in prices_grouped.groups:
    return np.nan

  g = prices_grouped.get_group(code)
  # trade_date 당일 또는 직후 거래일 찾기
  idxs = g.index[g["date"] >= trade_date]
  if len(idxs) == 0:
    return np.nan
  start_idx = idxs[0]
  start_price = g.loc[start_idx, "close"]

  # horizon_days 후 날짜
  target_date = trade_date + timedelta(days=horizon_days)
  idxs2 = g.index[g["date"] >= target_date]
  if len(idxs2) == 0:
    return np.nan
  end_idx = idxs2[0]
  end_price = g.loc[end_idx, "close"]

  if start_price <= 0 or end_price <= 0:
    return np.nan

  return (end_price - start_price) / start_price


# 3) ranking_history 읽어서 date별 TopN 전략 수익률 계산
records = []

for path in sorted(glob.glob(os.path.join(HISTORY_DIR, "ranking_final_*.csv"))):
  fname = os.path.basename(path)
  # 파일명에서 날짜 추출: ranking_final_YYYY-MM-DD.csv
  dt_str = fname.replace("ranking_final_", "").replace(".csv", "")
  try:
    trade_date = datetime.strptime(dt_str, "%Y-%m-%d").date()
  except ValueError:
    print("skip file (date parse error):", fname)
    continue

  df = pd.read_csv(path)

  # date 컬럼이 있으면 필터 한 번 더
  if "date" in df.columns:
    df = df[df["date"] == dt_str]

  # 기본적으로 숫자 컬럼 처리
  for col in ["final_score", "pred_return_60d", "pred_mdd_60d", "prob_top20_60d"]:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # TopN by final_score
  df_score = df.dropna(subset=["final_score"]).sort_values("final_score", ascending=False).head(TOP_N)
  # TopN by predicted return
  df_ret = df.dropna(subset=["pred_return_60d"]).sort_values("pred_return_60d", ascending=False).head(TOP_N)

  # 각 전략별 실제 수익률 계산
  for label, subset in [("score", df_score), ("pred_ret", df_ret)]:
    for _, row in subset.iterrows():
      code = str(row["code"]).zfill(6)  # 6자리 형식 맞추기 (필요하면)

      fwd = get_forward_return(code, pd.to_datetime(trade_date), HORIZON_DAYS)
      if pd.isna(fwd):
        continue

      records.append({
        "trade_date": trade_date,
        "strategy": label,
        "code": code,
        "name": row.get("name", ""),
        "final_score": row.get("final_score", np.nan),
        "pred_return_60d": row.get("pred_return_60d", np.nan),
        "realized_return": fwd,
      })

backtest = pd.DataFrame(records)

if backtest.empty:
    print("백테스트 대상 샘플이 없습니다. (records 가 0개)")
    print(" - history 폴더에 충분한 과거 ranking_final_*.csv 가 있는지,")
    print(" - HORIZON_DAYS 에 맞는 만큼 미래 가격이 prices_daily.csv 에 있는지 확인해 주세요.")
    raise SystemExit(0)

backtest.to_csv(os.path.join(BASE_DIR, "data", "analysis", "score_backtest_result.csv"), index=False)

print("총 샘플 수:", len(backtest))
print("\n전략별 평균 60일 수익률:")
print(backtest.groupby("strategy")["realized_return"].mean())

print("\n전략별 승률 (수익 > 0):")
print(backtest.groupby("strategy")["realized_return"].apply(lambda x: (x > 0).mean()))

# analysis/score_backtest_quantile.py 혹은 같은 파일 마지막에 추가

import pandas as pd
import numpy as np

# 만약 별도 파일에서 실행한다면:
# backtest = pd.read_csv("score_backtest_result.csv")
# backtest["realized_return"] = pd.to_numeric(backtest["realized_return"], errors="coerce")

def decile_analysis(df, col):
  tmp = df[[col, "realized_return"]].dropna().copy()
  tmp["quantile"] = pd.qcut(tmp[col], 5, labels=False) + 1  # 1~5 분위
  return tmp.groupby("quantile")["realized_return"].mean()

print("\n[final_score 분위수별 평균 수익률]")
print(decile_analysis(backtest, "final_score"))

print("\n[pred_return_60d 분위수별 평균 수익률]")
print(decile_analysis(backtest, "pred_return_60d"))
