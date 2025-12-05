# score_backtest_from_labels.py (v2 - 확정본)
#
# labels.csv + scores_final.csv 기반 점수 백테스트
# - ranking_final, predictions 없이도 동작
# - final_score = scores_final.score (기술 점수 v2)
#
# 사용 전제:
#   data/labels.csv      : 라벨 (target_60d 등)
#   data/scores_final.csv: scoring.py에서 만든 기술 점수
#
# 출력:
#   data/analysis/score_backtest_from_labels.csv
#   콘솔에 전략별 평균 수익률 / 승률 / 분위수별 수익률 출력

from pathlib import Path
import numpy as np
import pandas as pd

# ===== 1) 경로 설정 =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SCORES_PATH = DATA_DIR / "scores_final.csv"
LABELS_PATH = DATA_DIR / "labels.csv"

OUT_DIR = DATA_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5  # 매일 상위 N개 종목 매수


# ===== 2) 데이터 로드 =====
print(f"[INFO] scores_final load: {SCORES_PATH}")
scores = pd.read_csv(SCORES_PATH, dtype={"code": str})

print(f"[INFO] labels load: {LABELS_PATH}")
labels = pd.read_csv(LABELS_PATH)


# ===== 3) 기본 전처리 (date, code 통일) =====
def normalize_date_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("DataFrame에 'date' 컬럼이 없습니다.")
    if "code" not in df.columns:
        raise ValueError("DataFrame에 'code' 컬럼이 없습니다.")

    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)

    return df


scores = normalize_date_code(scores)
labels = normalize_date_code(labels)


# ===== 4) labels.csv 에서 60일 수익률 컬럼 찾기 =====
LABEL_CANDIDATES = [
    "target_60d",
    "realized_return_60d",
    "y_return_60d",
    "target_return_60d",
]

label_col = None
for c in LABEL_CANDIDATES:
    if c in labels.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError(
        "labels.csv 에서 60일 수익률 컬럼을 찾지 못했습니다.\n"
        f"다음 이름 중 하나로 컬럼을 만들어 주세요: {LABEL_CANDIDATES}"
    )

print(f"[INFO] label column detected: {label_col}")

labels = labels.rename(columns={label_col: "realized_return"})


# ===== 5) scores + labels merge (date, code 기준) =====
data = pd.merge(
    scores,
    labels[["date", "code", "realized_return"]],
    on=["date", "code"],
    how="inner",
)

print(f"[INFO] merged rows: {len(data)}")

# 숫자 컬럼 정리
data["score"] = pd.to_numeric(data["score"], errors="coerce")
data["realized_return"] = pd.to_numeric(data["realized_return"], errors="coerce")


# ===== 6) 날짜별 TopN 전략 수익률 계산 =====
records = []

for trade_date, g in data.groupby("date"):
    g = g.dropna(subset=["score", "realized_return"]).copy()
    if g.empty:
        continue

    # score(=기술점수) 상위 TOP_N
    g_top = g.sort_values("score", ascending=False).head(TOP_N)

    for _, row in g_top.iterrows():
        records.append(
            {
                "trade_date": trade_date.date(),
                "strategy": "score",          # 전략 이름
                "code": row["code"],
                "name": row.get("name", ""),  # 없으면 빈값
                "final_score": row["score"],  # 여기서는 final_score = score
                "pred_return_60d": np.nan,    # 이 스크립트에서는 예측값 없음
                "realized_return": row["realized_return"],
            }
        )

backtest = pd.DataFrame(records)

out_path = OUT_DIR / "score_backtest_from_labels.csv"
backtest.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[INFO] backtest result saved -> {out_path}")
print(f"[INFO] total samples: {len(backtest)} → Top{TOP_N} 전략으로 백테스트 적용된 종목 수")


# ===== 7) 요약 통계 출력 =====
if backtest.empty:
    print("⚠️ backtest 레코드가 없습니다. TOP_N, merge 키(date, code), 컬럼명을 확인해 주세요.")
else:
    # 전략별 평균 수익률
    print("\n[전략별 평균 60일 수익률] (60일 단위 수익률 평균)")
    print(backtest.groupby("strategy")["realized_return"].mean())

    # 전략별 승률 (수익률 > 0 비율)
    print("\n[전략별 승률 (수익률 > 0 비율)] (예: 0.67 = 67%)")
    print(backtest.groupby("strategy")["realized_return"].apply(lambda x: (x > 0).mean()))

    # 분위(quantile) 분석 함수
    def quantile_analysis(df: pd.DataFrame, col: str, q: int = 5):
        sub = df[[col, "realized_return"]].dropna().copy()
        if sub.empty:
            print(f"{col}: 데이터 없음")
            return None
        sub["quantile"] = pd.qcut(sub[col], q, labels=False) + 1  # 1~q
        return sub.groupby("quantile")["realized_return"].mean()

    if "final_score" in backtest.columns:
        print("\n[final_score 분위수별 평균 수익률]")
        print(quantile_analysis(backtest, "final_score", q=5))

    if "pred_return_60d" in backtest.columns:
        print("\n[pred_return_60d 분위수별 평균 수익률]")
        print(quantile_analysis(backtest, "pred_return_60d", q=5))
