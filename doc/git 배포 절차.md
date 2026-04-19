# Git 배포 절차

## 문서 역할

- 이 문서는 `배포 실행 절차` 기준 문서입니다.
- 자동매매 정책 해석은 `doc/lee 운영.md`와 `doc/20260417_자동매매 설계.md`를 우선 기준으로 봅니다.
- 다만 배포 시 혼선을 막기 위해 현재 실운용 상태 요약은 함께 유지합니다.

## 목적

- 로컬 운영본에서 만든 최신 산출물을 배포용 Git 디렉토리로 옮깁니다.
- GitHub Actions는 로컬 결과를 그대로 복원해 사용합니다.
- 기본 운영 기준은 `runtime_snapshot` 입니다.

## 운영 원칙

- 로컬 운영본이 기준본입니다.
- 원격에서 다시 계산한 결과를 운영 기준으로 삼지 않습니다.
- GitHub Actions `close-batch`는 기본적으로 `runtime_snapshot` 복원 기준으로 해석합니다.
- `pipeline` 모드는 참고용 재계산 또는 진단용으로만 사용합니다.

## 1. 로컬 산출물 최신화

권장 기준은 아래 순서입니다.

```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.venv\Scripts\python.exe python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

메모:

- `run_manual_close_batch.py`가 `pipeline -> 산출물 날짜 검증 -> run_operational_refresh -> serving 날짜 검증 -> sync_web_display_data -> node-api 재기동` 흐름을 강제합니다.
- 수동 close 배치는 이 경로를 기본으로 씁니다.
- 자동매매 종목선정의 기준 입력도 이 close batch 이후 갱신된 `ranking_final`, `operational_buy_gate`, `trade_intents` 계열 산출물입니다.

## 2. 거래 CSV 내보내기

실거래 기록을 배포본에 같이 반영하려면 아래를 실행합니다.

```powershell
python python\export_trades_csv.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 3. 백업 ZIP 생성

로컬 학습 이력 복원용과 로컬 운영 결과 재현용 ZIP을 함께 만듭니다.

```powershell
python python\backup_git_restore_zip.py --output backups\git_restore_backup_20260418.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot_20260418.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

역할:

- `git_restore_backup`
  - 장기 이력 복원용
  - `prices_daily_adjusted.csv`, `features.csv`, `labels.csv` 등 학습 이력 보존
- `git_runtime_snapshot`
  - 로컬 운영 결과 재현용
  - `ranking_final.csv`, `predictions.csv`, `model.pkl`, `output/stock_theme_daily.csv` 등 현재 운영 결과 보존

## 4. 배포용 Git 디렉토리 생성

```powershell
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

메모:

- 배포본은 `D:\ai\git\lee_trader` 에 생성합니다.
- 이 디렉토리는 코드 배포용 복사본입니다.

## 5. Git 커밋 및 푸시

```powershell
cd D:\ai\git\lee_trader
git fetch origin
git switch main
git pull --ff-only origin main

git status
git add --all
git commit -m "update deploy package"
git push origin main
```

메모:

- `D:\ai\git\lee_trader`가 이미 `git clone`으로 연결된 디렉토리라면 `git init`, `git remote add origin`은 다시 하지 않습니다.
- 비밀값은 `.env`에 두고, Git에는 예시 파일만 올립니다.

## 6. GitHub Actions 실행

GitHub에서 아래 순서로 실행합니다.

1. `Actions`
2. `Close Batch`
3. `Run workflow`
4. `execution_mode = runtime_snapshot`

해석:

- `runtime_snapshot`
  - 로컬에서 만든 결과를 복원해 사용
  - 기본 운영 기준
- `pipeline`
  - 원격에서 다시 계산
  - 참고용 또는 진단용

## 7. 웹 DB 동기화

로컬에서 직접 웹 DB를 최신 상태로 맞추려면 아래를 실행합니다.

```powershell
python python\sync_web_display_data.py --reset-first
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

메모:

- `WEB_DATABASE_URL`이 설정되어 있어야 합니다.
- `--reset-first`는 대상 테이블을 비우고 로컬 결과 기준으로 다시 적재합니다.

## 8. 권장 전체 순서

```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.venv\Scripts\python.exe python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\export_trades_csv.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_git_restore_zip.py --output backups\git_restore_backup_20260418.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot_20260418.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cd D:\ai\git\lee_trader
git fetch origin
git switch main
git pull --ff-only origin main
git add --all
git commit -m "update deploy package"
git push origin main
```

## 9. 현재 자동매매 운영 상태

- 현재 코드 기준 실운용 상태는 `PILOT 제한 실운용`까지 구현된 상태입니다.
- 자동매매 종목선정은 기본적으로 장마감 이후 close batch에서 확정된 랭킹과 게이트 산출물을 기준으로 이뤄집니다.
- 즉 장중 `intraday`는 보조 참고용이고, 실제 자동주문 대상 선정의 기준본은 close batch 결과입니다.
- `WATCH`에서는 아래 제한 규칙만 허용합니다.
  - 최대 `2종목`
  - 총 신규 노출 `15%`
  - 종목당 신규 진입 상한 `8%`
- `PILOT`에서는 아래 제한 규칙을 허용합니다.
  - 최대 `4종목`
  - 총 신규 노출 `30%`
  - 종목당 신규 진입 상한 `12%`
- `HOLD`, `BLOCK`에서는 신규 진입을 허용하지 않습니다.
- `BUY_ALLOWED`는 정식 자동매수 승인 단계입니다.

## 10. auto_buy 점검 메모

- `auto_buy` 스케줄은 `run_operational_refresh -> submit_live_orders` 순서로 동작합니다.
- 따라서 `Scheduler Runtime`에서 `auto_buy 오류`가 보이면 먼저 `trade_intents.json`, `order_requests_preview.json`, `order_requests_execution.json` 생성 여부를 확인합니다.
- 현재 코드는 주문 산출물 생성 후 웹 DB 동기화가 실패해도 전체 주문 단계가 바로 실패로 끝나지 않도록 보강되어 있습니다.
- BUY 주문은 `order_buy_approvals.json`의 `approved_request_ids`가 비어 있으면 승인 가드에 의해 제출이 보류될 수 있습니다.

## 11. 향후 방향

- 현재 운영 기준은 `WATCH`와 `PILOT`를 함께 사용합니다.
- 문서상 권장 해석은 아래와 같습니다.
  - `WATCH`: 탐색적 소액 진입
  - `PILOT`: 제한적 실운용
  - `BUY_ALLOWED`: 정식 자동매수
- 현재 코드는 `PILOT` 단계를 구현한 상태입니다.
- 향후 검토 대상은 `PILOT` 이후 `BUY_ALLOWED` 승격 조건의 추가 정교화입니다.

관련 문서:

- `doc/lee 운영.md`
- `doc/20260417_자동매매 설계.md`
- `doc/20260418_운영 상태 및 PILOT 방향 메모.md`
