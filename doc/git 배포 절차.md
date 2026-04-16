# Git 배포 절차

## 목적

- 로컬에서 계산한 최신 산출물을 Git 배포 디렉터리로 옮긴다.
- GitHub Actions가 로컬 결과를 그대로 복원해서 사용하게 한다.
- 기본 운영 기준은 `runtime_snapshot` 모드다.

## 1. 로컬 산출물 생성

먼저 로컬에서 파이프라인과 운영 후속 산출물을 최신 상태로 만든다.

```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose run --rm python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\run_operational_refresh.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

```

이 단계가 끝나면 아래 경로가 최신 상태여야 한다.

- `data/*.csv`
- `outputs/*.json`
- `serving/*.json`

## 2. Git 배포용 ZIP 생성

로컬 결과를 GitHub Actions에서 복원할 수 있도록 백업 ZIP을 만든다.

```powershell
python python\backup_git_restore_zip.py --output backups\git_restore_backup.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

역할:

- `git_restore_backup`
  - 학습 이력 복원용
  - `prices_daily_adjusted.csv`, `features.csv`, `labels.csv`, `universe.csv` 등 포함
- `git_runtime_snapshot`
  - 로컬 운영 결과 재현용
  - `ranking_final.csv`, `predictions.csv`, `market_status.csv`, `quality.csv`, `model.pkl`, `stock_theme_daily.csv` 등 포함

## 3. Git 배포 디렉터리 생성

배포용 Git 디렉터리로 파일을 복사한다.

```powershell
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

이 단계가 끝나면 배포 대상은 `D:\ai\git\lee_trader` 아래에 정리된다.

## 4. 선택: 거래기록 CSV 생성

웹 표시용 DB 동기화에 `trades`까지 반영할 경우 `data/trades.csv`를 만든다.

```powershell
python python\export_trades_csv.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

이 파일이 있으면 `git_runtime_snapshot.zip`에도 같이 포함되고, GitHub Actions에서 웹 DB 동기화 시 `trades` 테이블까지 반영된다.

## 5. Git 커밋 및 푸시

```powershell
cd D:\ai\git\lee_trader
git status
git add --all
git commit -m "update deploy package"
git push origin main
```

## 6. GitHub Actions 실행

GitHub 저장소에서 아래 순서로 실행한다.

1. `Actions`
2. `Close Batch`
3. `Run workflow`
4. `execution_mode = runtime_snapshot`

의미:

- `runtime_snapshot`
  - 로컬에서 만든 결과를 복원해서 사용
  - `run_pipeline` 재실행 안 함
  - `ranking_builder` 재실행 안 함
- `pipeline`
  - 원격에서 다시 수집, 학습, 예측, 랭킹 수행
  - 로컬과 결과가 달라질 수 있으므로 기본 운영 기준으로 쓰지 않음

## 7. 웹 표시용 DB 동기화

웹 화면이 로컬 결과와 같아야 할 경우 표시용 데이터만 웹 DB에 반영한다.

로컬에서 웹 DB 통신이 가능하면 직접 실행할 수 있다.

```powershell
python python\sync_web_display_data.py --reset-first
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

로컬에서 웹 DB 통신이 안 되더라도, GitHub Actions의 `close-batch`는 `Run close batch` 뒤에 자동으로 아래를 실행한다.

```powershell
python python\sync_web_display_data.py --reset-first
```

조건:

- GitHub Actions `DATABASE_URL` secret이 설정되어 있어야 한다.
- `runtime_snapshot` 모드에서는 복원된 CSV/JSON 기준으로 웹 DB가 갱신된다.
- `trades`는 `data/trades.csv`가 있을 때만 반영된다.
- `--reset-first`가 적용되므로 표시용 테이블은 먼저 비우고 로컬 기준으로 다시 적재된다.

동기화 대상:

- `stocks`
- `market_status`
- `predictions`
- `daily_ranking`
- `research.app_payload_store`
- `research.paper_trading_*`
- `trades.csv`가 있으면 `trades`

옵션 예시:

```powershell
python python\sync_web_display_data.py --skip-trades
python python\sync_web_display_data.py --skip-paper-trading
```


### CSV 백업 / 복구

powershell
python python\backup_csv_md_zip.py --output backups\csv_backup_20260415.zip --overwrite --keep-latest 1
python python\restore_csv_md_zip.py --zip backups\csv_backup_20260415.zip --overwrite

## 권장 전체 순서

```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose run --rm python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\run_operational_refresh.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\export_trades_csv.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_git_restore_zip.py --output backups\git_restore_backup_20260416.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot_20260416.zip --overwrite --keep-latest 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

```powershell
cd D:\ai\git\lee_trader
git status
git add --all
git commit -m "update deploy package"
git push origin main
```


그 다음 `D:\ai\git\lee_trader`에서 `git add / commit / push`를 수행한다.
