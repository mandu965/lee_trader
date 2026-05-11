# 5차 과제: scheduler health ledger 구축

> 상태: ✅ 완료
> 작성일: 2026-05-07
> 의존성: 없음 (1차와 독립적으로 진행 가능)
> 다음 과제: 없음 (독립 완결)

---

## 목적

현재 "오늘 자동매매가 왜 실행되지 않았는지" 확인하려면
`auto_ops_scheduler_status.json`과 로그 파일을 직접 뒤져야 한다.

현재 `run_daily_scheduler.py`의 `run_daily_cycle()`은
전체 사이클의 성공/실패만 기록하고 (status: idle / error)
각 step의 개별 결과(소요 시간, 산출물 rows, 실패 사유)는 남기지 않는다.

이 과제에서 각 job의 실행 결과를 구조화하여
운영자가 한 파일에서 오늘 전체 흐름을 파악할 수 있게 한다.

---

## 현재 구조 파악 (소스 기준)

`run_daily_scheduler.py`의 핵심 구조:

```python
def _run_step(name: str, command: list[str]) -> None:
    logging.info("START %s", name)
    subprocess.run(command, cwd=ROOT, check=True)
    logging.info("OK %s", name)

def run_daily_cycle(now, tz_name, status):
    for name, command in run_steps:
        _run_step(name, command)  # 실패 시 CalledProcessError 발생
    # → 전체 성공/실패만 status에 기록
    # → 각 step별 소요시간, 결과, 산출물 정보 없음
```

실행 step 목록 (SCHEDULER_COMMAND_SET 기준):

- `close`: run_manual_close_batch
- `auto_buy`: run_operational_refresh → submit_live_orders
- `live_sync`: sync_live_account_holdings → 여러 리포트 단계
- `rule_after_close`: run_rule_after_close_cycle
- `rule_before_open`: run_rule_before_open_cycle
- `rule_after_open`: run_rule_after_open_cycle
- `intraday`: run_intraday_refresh

---

## 변경 방향

### Step 1. python/scheduler_health.py 신설

각 job의 실행 결과를 기록하는 유틸리티 모듈.

```python
class JobResult:
    job_name: str
    command_set: str
    started_at: str          # ISO timestamp
    finished_at: str         # ISO timestamp
    duration_seconds: float
    status: str              # success / failed / skipped
    exit_code: int | None
    error_message: str | None
    output_rows: int | None  # 산출물 행 수 (파악 가능한 경우)
    output_file: str | None  # 주요 산출물 파일 경로
    notes: str | None        # 추가 컨텍스트

def record_job(result: JobResult) -> None:
    """scheduler_health.json에 해당 job 결과를 추가/갱신한다."""

def write_health_summary(date: str, command_set: str) -> None:
    """scheduler_health_report.md를 생성한다."""
```

### Step 2. run_daily_scheduler.py 수정

`_run_step()`을 확장하여 각 step의 결과를 기록한다.

```python
def _run_step_with_health(name, command, command_set, health_path):
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = subprocess.run(command, cwd=ROOT, check=True,
                                capture_output=True, text=True)
        finished_at = datetime.now().isoformat(timespec="seconds")
        record_job(JobResult(
            job_name=name,
            command_set=command_set,
            started_at=started_at,
            finished_at=finished_at,
            status="success",
            exit_code=0,
            ...
        ))
    except subprocess.CalledProcessError as exc:
        finished_at = datetime.now().isoformat(timespec="seconds")
        record_job(JobResult(
            job_name=name,
            status="failed",
            exit_code=exc.returncode,
            error_message=str(exc),
            ...
        ))
        raise
```

### Step 3. 산출물 row 수 파악

각 step의 주요 산출물 파일과 파악 방법:

| job_name | 산출물 파일 | row 수 파악 방법 |
|---|---|---|
| run_operational_refresh | data/ranking_final.csv | pd.read_csv(nrows=0) + wc -l |
| submit_live_orders | outputs/order_requests_execution.json | len(items) |
| sync_live_account_holdings | data/live_account_holdings.csv | wc -l |
| run_rule_after_close_cycle | data/rule_signals.csv | wc -l |
| run_rule_before_open_cycle | outputs/rule_execution_results.json | len(items) |
| run_rule_after_open_cycle | outputs/rule_execution_fill_sync.json | len(items) |

row 수 파악이 불가능한 step은 `output_rows=null`로 저장.

---

## 출력 파일

### outputs/scheduler_health.json

```json
{
  "date": "2026-05-07",
  "generated_at": "2026-05-07T09:10:00",
  "command_set": "auto_buy",
  "jobs": [
    {
      "job_name": "run_operational_refresh",
      "command_set": "auto_buy",
      "started_at": "2026-05-07T09:01:00",
      "finished_at": "2026-05-07T09:01:45",
      "duration_seconds": 45.2,
      "status": "success",
      "exit_code": 0,
      "error_message": null,
      "output_rows": 150,
      "output_file": "data/ranking_final.csv",
      "notes": null
    },
    {
      "job_name": "submit_live_orders",
      "command_set": "auto_buy",
      "started_at": "2026-05-07T09:01:50",
      "finished_at": "2026-05-07T09:02:10",
      "duration_seconds": 20.1,
      "status": "success",
      "exit_code": 0,
      "error_message": null,
      "output_rows": 2,
      "output_file": "outputs/order_requests_execution.json",
      "notes": "submitted=2, blocked=1"
    },
    {
      "job_name": "sync_live_account_holdings",
      "command_set": "auto_buy",
      "started_at": "2026-05-07T09:02:15",
      "finished_at": "2026-05-07T09:02:47",
      "duration_seconds": 32.0,
      "status": "failed",
      "exit_code": 1,
      "error_message": "ConnectionError: KIS API timeout",
      "output_rows": null,
      "output_file": null,
      "notes": null
    }
  ],
  "summary": {
    "total_jobs": 8,
    "success": 6,
    "failed": 1,
    "skipped": 1,
    "total_duration_seconds": 312.4,
    "first_started_at": "2026-05-07T09:01:00",
    "last_finished_at": "2026-05-07T09:10:00"
  }
}
```

### outputs/scheduler_health_report.md

운영자가 바로 읽을 수 있는 마크다운 리포트.

```markdown
# Scheduler Health Report — 2026-05-07

command_set: auto_buy | 총 소요: 312초

| job | 상태 | 소요(초) | 산출물 |
|---|---|---|---|
| run_operational_refresh | ✅ 성공 | 45 | ranking 150종목 |
| submit_live_orders | ✅ 성공 | 20 | 주문 2건 제출 |
| sync_live_account_holdings | ❌ 실패 | 32 | ConnectionError |
| sync_live_order_fills | ⏭ 건너뜀 | - | 이전 step 실패로 중단 |

## 실패 상세
- sync_live_account_holdings: ConnectionError: KIS API timeout
```

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 신설 | python/scheduler_health.py | health ledger 기록 유틸리티 |
| 수정 | python/run_daily_scheduler.py | _run_step 확장, health 기록 연결 |

### 수정 금지 파일

- python/run_pipeline.py
- python/run_rule_after_close_cycle.py
- python/run_rule_before_open_cycle.py
- python/run_rule_after_open_cycle.py
- (각 step 내부 스크립트는 수정하지 않음)

---

## 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| SCHEDULER_HEALTH_ENABLED | true | health ledger 기록 활성화 |
| SCHEDULER_HEALTH_JSON | outputs/scheduler_health.json | 출력 파일 경로 |
| SCHEDULER_HEALTH_REPORT_MD | outputs/scheduler_health_report.md | MD 리포트 경로 |
| SCHEDULER_HEALTH_NOTIFY_ON_FAILURE | true | job 실패 시 notifier 알림 |

---

## 검증 케이스

| # | 조건 | 기대 결과 |
|---|---|---|
| 1 | step 1개 성공 | health.json에 status=success 기록 |
| 2 | step 1개 실패 | health.json에 status=failed, error_message 기록 |
| 3 | step 실패 후 다음 step | status=skipped 또는 미실행으로 기록 |
| 4 | 전체 사이클 완료 | summary.total_jobs, success, failed 정확히 집계 |
| 5 | output_rows 파악 가능한 step | output_rows 값 기록 |
| 6 | output_rows 파악 불가능한 step | output_rows=null 기록 |
| 7 | MD 리포트 생성 | scheduler_health_report.md 가독성 확인 |
| 8 | SCHEDULER_HEALTH_NOTIFY_ON_FAILURE=true + 실패 | notifier WARNING 발송 확인 |

---

## 주의사항

- 기존 auto_ops_scheduler_status.json 출력은 그대로 유지할 것
  (scheduler_health.json은 추가 파일이며 기존 파일을 대체하지 않음)
- subprocess capture_output=True 추가 시
  기존 로그 출력이 사라지지 않도록 주의할 것
  (stderr는 별도로 logging에 전달)
- 각 step의 산출물 파일 row 수 파악은 best-effort로 처리하고
  실패 시 null로 저장할 것 (row 파악 실패가 health 기록을 막으면 안 됨)

---

## 완료 후 기록

완료일: 2026-05-12

변경 파일:
- python/scheduler_health.py (신설)
- python/run_daily_scheduler.py (수정: _run_step_tracked, _record_remaining_skipped, _finalize_health, run_daily_cycle)
- .env.example (SCHEDULER_HEALTH_ENABLED 등 4개 환경변수 추가)
- doc/improvement/05_health_ledger.md (이 파일)
- doc/improvement/ROADMAP.md (상태 갱신)

검증 결과:
- Case 1 (success): PASS
- Case 2 (failed + exit_code + error_message): PASS
- Case 3 (skipped): PASS
- Case 4 (summary 집계): PASS
- Case 5 (CSV output_rows): PASS
- Case 6 (JSON output_rows): PASS
- Case 7 (unmapped → output_rows=null): PASS
- Case 8 (markdown report): PASS
- Case 9 (ENABLED=false → 파일 미생성): PASS
- Case 10 (NOTIFY_ON_FAILURE + 실패 → notifier 호출): 설계 반영 (notifier 미구성 시 warning 로그)
- Case 11 (health 예외 swallow): PASS
- Case 12 (auto_ops_scheduler_status.json 호환): PASS (_write_status 호출 횟수 3개 유지)

주요 결정 사항:
- emoji 상태 아이콘(✅❌⏭)을 ASCII([OK][FAIL][SKIP])로 변경 (Windows cp949 인코딩 대응)
- em dash(—) 대신 하이픈(-) 사용 (동일 이유)
- _run_step 자체는 수정하지 않고 _run_step_tracked wrapper로 분리 (기존 흐름 보존)
- health 기록 실패는 예외를 swallow하고 warning 로그만 남김 (scheduler 중단 방지)
- run_daily_cycle 내 nested function(_record_remaining_skipped, _finalize_health)으로 skipped 기록 및 summary 생성

다음 과제 연결 포인트:
- scheduler_health.json을 활용한 KPI 모니터링 (score_kpi_monitor.py 연동)
- notifier.py 알림 채널 구성 후 SCHEDULER_HEALTH_NOTIFY_ON_FAILURE=true 활성화
