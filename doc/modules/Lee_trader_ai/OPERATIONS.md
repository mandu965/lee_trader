# AI Operations

## Purpose

이 문서는 AI 선별/자동매매 모듈의 기본 운영 명령과 점검 순서를 정리합니다.

## Main Commands

### Pipeline

```powershell
docker compose run --rm python-pipeline python python/run_pipeline.py
```

### Operational Refresh

```powershell
docker compose run --rm scheduler-auto-buy python python/run_operational_refresh.py
```

### Live Auto Trade Cycle

```powershell
docker compose run --rm scheduler-auto-buy python python/run_live_auto_trade_cycle.py
```

### Web Sync

```powershell
docker compose run --rm scheduler-live-account-sync python python/sync_web_display_data.py
```

## Key Outputs

- `data/predictions.csv`
- `data/ranking_final.csv`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`
