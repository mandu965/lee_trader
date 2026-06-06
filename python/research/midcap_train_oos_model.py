"""
midcap_train_oos_model.py  —  Phase 2b: out-of-sample 모델 학습 (격리 wrapper)

운영 model_train.py를 재사용하되:
  - 학습 데이터: 운영 features.csv + labels.csv (top-100), date <= 2024-09-30 만
    → 2025~2026을 한 번도 보지 않은 모델 (시간 OOS)
  - 출력: data/research_midcap/model_oos.pkl  (운영 model.pkl 미접촉)
  - feature_importance 글로벌 경로를 연구 dir로 override (운영 산출물·web json 미접촉)
  - DB 쓰기 없음 (model_train.main은 DB 미사용 확인)

이 모델을 midcap 2025~2026에 적용하면 시간·유니버스 동시 out-of-sample 전이 테스트가 된다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

FI_DIR = ROOT / "data" / "research_midcap" / "fi_oos"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2024-09-30")
    ap.add_argument("--out", default=str(ROOT / "data" / "research_midcap" / "model_oos.pkl"))
    args = ap.parse_args()

    import model_train as mt
    # 격리: feature_importance 출력 글로벌을 연구 dir로 (운영 data/model_feature_importance* 미접촉)
    mt.MODEL_FEATURE_IMPORTANCE_DIR = FI_DIR
    FI_DIR.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "model_train",
        "--features-csv", str(ROOT / "data" / "features.csv"),
        "--labels-csv", str(ROOT / "data" / "labels.csv"),
        "--train-end-date", args.train_end,
        "--horizons", "60",         # target_log_60d / target_mdd_60d (랭킹은 log_60d)
        "--cls-horizons", "60",     # target_60d_top20
        "--output-pkl", args.out,
        "--model-version", f"oos_{args.train_end}",
    ]
    print(f"[OOS] train_end={args.train_end} → {Path(args.out).name}")
    mt.main()
    print(f"[OOS] done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
