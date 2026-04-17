# Manual Close Batch

수동 마감 배치는 아래 명령을 기본으로 사용한다.

```powershell
python python\run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

실행 순서:

1. `docker compose build --no-cache --progress=plain python-pipeline`
2. `docker compose run --rm python-pipeline`
3. `data/market_status.csv`, `data/features.csv`, `data/predictions.csv`, `data/ranking_final.csv` 최신 날짜 일치 검증
4. 검증된 최신 날짜를 `MARKET_DATE`로 고정해서 `python python\run_operational_refresh.py` 실행
5. `serving/daily_recommendations.json`, `serving/buy_gate_status.json`, `serving/model_portfolio.json` `asof_date` 일치 검증
6. `python python\sync_web_display_data.py`
7. `docker compose up -d --build node-api`

옵션 예시:

```powershell
python python\run_manual_close_batch.py --skip-build
python python\run_manual_close_batch.py --skip-node-api
python python\run_manual_close_batch.py --skip-web-sync
python python\run_manual_close_batch.py --web-sync-reset-first
```

주의:

- `docker compose run --rm python-pipeline`와 `python python\run_operational_refresh.py`를 따로 실행하면 중간 상태를 놓칠 수 있다.
- 수동 close 배치는 반드시 래퍼 스크립트로 실행한다.
