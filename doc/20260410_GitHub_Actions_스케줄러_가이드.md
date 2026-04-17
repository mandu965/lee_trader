# GitHub Actions 스케줄러 가이드

기준일: 2026-04-10 KST

이 문서는 `Render 웹앱 + Supabase DB + GitHub Actions 정시 배치` 조합으로 Lee Trader를 운영하는 기준입니다.

## 1. 왜 이 방식으로 바꾸는가

- 현재 [render.yaml](/d:/ai/Lee_trader/render.yaml)은 Render `web` 서비스만 정의합니다.
- 기존 `scheduler`, `scheduler-recovery`는 [docker-compose.yml](/d:/ai/Lee_trader/docker-compose.yml)의 별도 컨테이너입니다.
- 따라서 Render에 웹앱만 배포한 상태에서는 Docker scheduler가 자동 실행되지 않습니다.

운영 역할은 아래처럼 나눕니다.

- Render: 웹 서비스 응답
- Supabase: 운영 DB 저장소
- GitHub Actions: 정시 Python 실행

## 2. 추가된 파일

- 장중 refresh: [.github/workflows/intraday-refresh.yml](/d:/ai/Lee_trader/.github/workflows/intraday-refresh.yml)
- 마감 batch: [.github/workflows/close-batch.yml](/d:/ai/Lee_trader/.github/workflows/close-batch.yml)
- 1회 실행 래퍼: [run_scheduled_job.py](/d:/ai/Lee_trader/python/run_scheduled_job.py)

`run_scheduled_job.py`는 기존 파이프라인을 1회 실행하고, 아래 상태 payload도 같이 갱신합니다.

- `auto_ops_scheduler_status`
- `auto_ops_recovery_scheduler_status`

즉 운영 화면의 scheduler 카드가 계속 살아 있습니다.

## 3. 실행 시각

GitHub Actions `cron`은 UTC 기준입니다.

- `03:00 UTC` = `12:00 KST` 장중 refresh
- `07:00 UTC` = `16:00 KST` 마감 batch

현재 workflow는 평일 기준 `1-5`로 설정했습니다.

## 4. GitHub Secrets 등록

GitHub 저장소의 `Settings > Secrets and variables > Actions`에 아래 값을 등록합니다.

필수:

- `DATABASE_URL`
- `KIS_BASE_URL`
- `KIS_APP_KEY`
- `KIS_APP_SECRET`

설명:

- `DATABASE_URL`: Supabase Postgres 연결 문자열
- `KIS_*`: 시세 수집과 장중 refresh용 자격 증명

## 5. 적용 절차

1. 현재 변경을 GitHub 원격 저장소에 push 합니다.
2. GitHub 저장소에 Secrets 4개를 등록합니다.
3. Actions 탭에서 `Intraday Refresh`, `Close Batch` workflow가 보이는지 확인합니다.
4. 각 workflow를 `Run workflow`로 1회 수동 실행합니다.
5. Render 웹앱의 `/ops-readiness.html`에서 scheduler 상태가 반영되는지 확인합니다.

## 6. 확인 포인트

성공 기준:

- `research.app_payload_store`에 `auto_ops_scheduler_status` 또는 `auto_ops_recovery_scheduler_status`가 갱신됨
- `daily_recommendations`, `operational_buy_gate`, `score_kpi_monitor` 등 주요 payload가 최신 시각으로 갱신됨
- Render 웹앱에서 최신 데이터가 보임

로컬에서 확인할 SQL 예시:

```sql
select payload_key, updated_at
from research.app_payload_store
where payload_key in (
  'auto_ops_scheduler_status',
  'auto_ops_recovery_scheduler_status',
  'daily_recommendations',
  'operational_buy_gate',
  'score_kpi_monitor'
)
order by updated_at desc;
```

## 7. 상태 파일과 로그

GitHub Actions 실행도 기존 파일명을 그대로 사용합니다.

- [auto_ops_scheduler.log](/d:/ai/Lee_trader/logs/auto_ops_scheduler.log)
- [auto_ops_recovery_scheduler.log](/d:/ai/Lee_trader/logs/auto_ops_recovery_scheduler.log)
- [auto_ops_scheduler_status.json](/d:/ai/Lee_trader/outputs/auto_ops_scheduler_status.json)
- [auto_ops_recovery_scheduler_status.json](/d:/ai/Lee_trader/outputs/auto_ops_recovery_scheduler_status.json)

다만 GitHub Actions 러너는 매 실행마다 새 환경이므로, 로컬 디스크처럼 항상 같은 파일이 남는 구조는 아닙니다. 운영 기준 진실 원천은 Supabase의 `research.app_payload_store`로 봅니다.

## 8. 남는 제약

- GitHub Actions schedule은 몇 분 지연될 수 있습니다.
- 러너는 영구 디스크가 아니므로 CSV history 누적은 제한적입니다.
- 이를 완화하려고 workflow에 `data/history`, `data/ranking_snapshot_archive.csv`, `serving`, `outputs` 캐시를 넣어 두었습니다.
- 그래도 절대적인 영속 저장소는 아니므로, 장기적으로는 history도 DB 중심으로 옮기는 것이 더 안전합니다.
