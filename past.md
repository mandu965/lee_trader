## 워크포워드 v5 실험 방법

목적: 중기(60일) vs 단기~중기(30일) horizon 비교. 공통 파라미터는 train_years=1.0, valid_months=3, rebalance_mode=biweekly, top_k=10이며 label_max_date=2025-09-01 기준으로 valid_end를 제한합니다.

### 커맨드 예시

- 메인(60d)
  - Windows (PowerShell)
    python walkforward_backtest.py ^
      --holding-days 60 ^
      --train-years 1.0 ^
      --valid-months 3 ^
      --rebalance-mode biweekly ^
      --top-k 10 ^
      --label-max-date 2025-09-01

  - Linux/macOS
    python walkforward_backtest.py \
      --holding-days 60 \
      --train-years 1.0 \
      --valid-months 3 \
      --rebalance-mode biweekly \
      --top-k 10 \
      --label-max-date 2025-09-01

- 보조(30d)
  - Windows (PowerShell)
    python walkforward_backtest.py ^
      --holding-days 30 ^
      --train-years 1.0 ^
      --valid-months 3 ^
      --rebalance-mode biweekly ^
      --top-k 10 ^
      --label-max-date 2025-09-01

  - Linux/macOS
    python walkforward_backtest.py \
      --holding-days 30 \
      --train-years 1.0 \
      --valid-months 3 \
      --rebalance-mode biweekly \
      --top-k 10 \
      --label-max-date 2025-09-01

### 생성 결과 파일
- prediction_history_{horizon}d_{rebalance_mode}.csv
- backtest_walkforward_{horizon}d_{rebalance_mode}_top10_trades.csv
- backtest_walkforward_{horizon}d_{rebalance_mode}_top10_summary.csv

### 샘플 사이즈 안내
- 전체 trades 수가 200개 이상이면 통계적으로 의미 있는 샘플로 간주하고, 100개 미만이면 신뢰도가 낮음을 참고합니다.
