
---

## 1. 권장 데일리 실행 순서

### 자동 스케줄 기준 시각

- 종가 close 배치: `16:00`
- 장중 refresh: `12:00`
- 실자동매매 신규 매수 판단: `09:30`
- 실계좌 동기화: `3시간마다`

메모:

- `09:30`은 전날 종가 기준 `top20` 후보를 바탕으로, 장초반 갭/과열/추격매수 위험을 1차 확인한 뒤 신규 매수 판단을 내리는 운영 기준 시각이다.
- 따라서 실자동매매는 `12:10` 장중 신규 진입보다 `09:30` 장초반 보수 집행이 현재 설계와 더 잘 맞는다.
- 장중 `12:00` refresh는 신규 매수 주 실행 시간이 아니라 관제와 재평가 성격으로 본다.

### 한 번에 처리


```powershell
python python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

메모:

- 수동 close 배치는 `run_manual_close_batch.py`를 기본으로 사용한다.
- 이 스크립트가 `python-pipeline -> 산출물 최신 날짜 검증 -> run_operational_refresh.py -> serving asof_date 검증 -> sync_web_display_data.py -> node-api 재기동` 순서를 강제한다.
- 따라서 `docker compose run --rm python-pipeline`와 `python python/run_operational_refresh.py`를 따로 치는 방식보다 기준일 불일치 위험이 낮다.
- Docker `scheduler`의 close 실행도 이제 같은 데이터 경로를 따르며, 내부적으로 `python python/run_manual_close_batch.py --skip-build --skip-node-api`에 맞춘 흐름으로 동작한다.
- 즉 자동 스케줄러는 데이터 기준일 검증과 웹 동기화까지 `run_manual_close_batch.py`와 동일한 기준을 따르고, 컨테이너 안에서 불가능한 이미지 rebuild와 `node-api` 재기동만 제외한다.

### CSV 백업 / 복구

powershell
python python\backup_csv_md_zip.py --output backups\csv_backup_20260416.zip --overwrite --keep-latest 1
python python\restore_csv_md_zip.py --zip backups\csv_backup_20260416.zip --overwrite

### Git 배포 필수 3단계

`GitHub Actions` 운영 배치는 이제 `runtime_snapshot` 기준이 기본입니다.  
즉 `close-batch`는 로컬에서 만든 산출물을 복원해 사용해야 하며, 로컬 snapshot 없이 원격에서 다시 계산한 결과를 운영 기준으로 쓰지 않습니다.

아래 3개 명령을 순서대로 모두 실행합니다.

python python\backup_git_restore_zip.py --output backups\git_restore_backup_20260416.zip --overwrite --keep-latest 1

python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot_20260416.zip --overwrite --keep-latest 1

python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target

역할:

- `backup_git_restore_zip.py`
  - 장기 이력 복원용
  - `prices_daily_adjusted.csv`, `features.csv`, `labels.csv` 등 학습 이력 보존
- `backup_runtime_snapshot_zip.py`
  - 로컬 운영 결과 재현용
  - `ranking_final.csv`, `predictions.csv`, `model.pkl`, `output/stock_theme_daily.csv` 등 로컬 결과 보존
- `export_git_release.py`
  - 위 두 zip을 포함해 `D:\ai\git\lee_trader` 배포본 생성

운영 원칙:

- 로컬 운영본이 기준본입니다.
- `GitHub Actions` 결과도 로컬과 같은 수준이어야 하므로, 기본 실행은 `runtime_snapshot` 복원 기준으로 운영합니다.
- `pipeline` 모드는 참고용 재계산/진단용으로만 사용합니다.
- `runtime_snapshot` 복원 후 해시 검증이 실패하면 배치를 실패시켜야 합니다.

### GitHub Actions 긴 이력 보호
로컬 저장 
python python\backup_git_restore_zip.py --output backups\git_restore_backup_20260415.zip --overwrite --keep-latest 1

`close-batch`는 이제 기본적으로 `runtime_snapshot` 모드로 실행합니다.

- `backups/git_runtime_snapshot_*.zip` 중 최신 파일을 먼저 복원
- 복원된 파일은 manifest 해시 기준으로 검증
- 검증 실패 또는 snapshot 부재 시 배치를 실패
- 그 다음 후속 운영/배포 단계만 실행

장기 이력 보호용으로는 캐시 복원 뒤 아래 보호도 같이 수행합니다.

- `backups/git_restore_backup_*.zip` 중 최신 파일이 있으면, `prices_daily_adjusted.csv`, `features.csv`, `labels.csv`가 없거나 이력이 너무 짧을 때 자동 복원
- 배치 실행 전 이력 길이 검증
  - `data/prices_daily_adjusted.csv`: 최소 `100000`행, 시작일 `2024-01-01` 이전
  - `data/features.csv`: 최소 `100000`행, 시작일 `2024-01-01` 이전
  - `data/labels.csv`: 최소 `50000`행, 시작일 `2024-01-01` 이전
- 조건을 못 맞추면 GitHub Actions 배치를 실패시켜 짧은 이력 산출물을 막음

운영 원칙:

- 장기 이력이 정상인 날에는 로컬에서 `backups/git_restore_backup_YYYYMMDD.zip`를 갱신하고 `--keep-latest 1`로 최신 1개만 유지
- 로컬 결과를 배포하는 날에는 `backups/git_runtime_snapshot_YYYYMMDD.zip`도 같이 갱신하고 `--keep-latest 1`로 최신 1개만 유지
- GitHub Actions는 캐시 미스가 나더라도 `git_restore_backup`으로 학습 이력을 먼저 복원하고, 운영 실행은 `git_runtime_snapshot` 기준으로 로컬 결과를 재현
- `features.csv`, `labels.csv`, `universe.csv`도 캐시에 포함해 다음 실행부터 누적 상태를 최대한 유지

## DB->CSV
python python\export_db_tables_to_csv.py

### 최신 CSV 기준 웹 DB 동기화

```powershell
$env:DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres"
python python/sync_csv_db_parity.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"
python python/sync_csv_db_parity.py


```

### 웹 DB 학습 이력 복구 순서

Git 배치가 짧은 가격 이력으로 `fact_price_daily`, `features`, `labels`를 전체 교체한 경우 아래 순서로 복구합니다.

```powershell
$env:DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"

python python/create_adjusted_prices.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/feature_builder.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/label_builder.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/model_train.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/model_predict.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/ranking_builder.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/sync_csv_db_parity.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### 로컬 Docker DB vs 웹 DB 학습데이터 비교 한 번에 실행

```powershell
$env:LOCAL_DATABASE_URL="postgresql://lee_trader:lee_trader_pw@localhost:5432/lee_trader"
$env:WEB_DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"

python python/compare_training_data_sources.py --code 096530
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

이 스크립트는 아래 검증을 한 번에 출력합니다.

- `fact_price_daily`, `prices_adjusted`, `features`, `labels`, `predictions`, `daily_ranking`의 row 수 / 시작일 / 최신일
- `features INNER JOIN labels` 기준 실제 학습 가능 row 수
- 지정 종목 기본값 `096530`의 `fact_price_daily`, `predictions`, `daily_ranking` 최신 스냅샷

메모:

- 웹 DB가 로컬보다 row 수는 비슷해도 시작일이 더 늦으면 학습데이터가 짧아진 상태일 수 있습니다.
- 특히 `feature_builder.py`는 이제 코드 누락뿐 아니라 row 수와 시작일도 같이 보고, DB 이력이 짧으면 `prices_daily_adjusted.csv`로 fallback 하도록 보강했습니다.

메모:

- `sync_csv_db_parity.py`는 이제 `prices_adjusted`, `fact_price_daily`, `features`, `labels`, `predictions`, `daily_ranking`까지 같이 동기화합니다.
- `export_git_release.py` 배포본에는 `fundamentals.csv`, `interest_universe.csv`, `data/experiments/theme_weight/best_weight*.json`만 포함됩니다. `features.csv`, `labels.csv`, `prices_daily_adjusted.csv`는 git 배포본에 실리지 않습니다.
- 따라서 웹 DB 학습 이력 복구는 git 푸시가 아니라 로컬 CSV -> 웹 DB 동기화로 처리해야 합니다.
- Git 배포본은 `ranking_final.csv` 재생성용 실행본이 아니라 코드/설정 배포용 복사본으로 운영합니다. ranking 재생성은 로컬 운영본에서 수행합니다.
### 운영 확인용 핵심 명령

```powershell
docker compose logs -f python-pipeline
docker compose logs -f node-api
docker compose ps scheduler scheduler-recovery
docker compose logs -f scheduler
docker compose logs -f scheduler-recovery
Get-Content outputs/auto_ops_scheduler_status.json
Get-Content outputs/auto_ops_recovery_scheduler_status.json
```

---

## 15. Git 메모

powershell
git config --global user.name "mandu965"
git config --global user.email "mandu965@naver.com"

git clone https://github.com/mandu965/lee_trader

git init
git remote add origin https://github.com/mandu965/lee_trader.git

git status
git add --all
git commit -m "lee commit"
git push -u origin main
git branch -m main

git fetch origin
git pull --rebase origin main


### 배포용 디렉토리 기준 정리

powershell
# 1. 배포용 복사본 생성
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target


# 2. 배포용 git 작업 디렉토리로 이동
cd D:\ai\git\lee_trader

# 3. 최초 1회만 git 초기화
git init
git branch -m main
git remote add origin https://github.com/mandu965/lee_trader.git

# 4. 커밋 / 푸시
git status
git add --all
git commit -m "initial deploy package"
git push -u origin main


### 이미 원격 저장소가 연결된 이후

# 작업 전에 최신 원격 반영
git fetch origin
git switch main
git pull --ff-only origin main


# 로컬 배포본 다시 생성 후 커밋
cd D:\ai\Lee_trader

-- 장기 이력 복원용
python python\backup_git_restore_zip.py --output backups\git_restore_backup.zip --overwrite --keep-latest 1

-- 로컬 결과 재현용
python python\backup_runtime_snapshot_zip.py --output backups\git_runtime_snapshot.zip --overwrite --keep-latest 1

-- 배포파일 옮기기
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target

git reset --soft origin/main
cd D:\ai\git\lee_trader
git status
git add --all
git commit -m "update deploy package"

git push origin main


### 메모

- `git clone` 방식과 `git init + remote add origin` 방식은 같이 쓰지 않습니다.
- `D:\ai\git\lee_trader`가 이미 `git clone`으로 만들어진 폴더라면 `git init`, `git remote add origin`은 다시 하지 않습니다.
- 실제 비밀값은 `.env`에 두고, git에는 `.env.example`, `.env.render.example`만 올립니다.
- `git_restore_backup_*.zip`, `git_runtime_snapshot_*.zip`은 운영 복원용이므로 배포본에 포함합니다.
- `runtime_snapshot`이 없는 상태에서 `close-batch`를 돌리면 로컬과 동일한 운영 결과를 보장할 수 없습니다.
- 운영 기준 검증은 `runtime_snapshot` 결과를 우선으로 보고, `pipeline` 재계산 결과는 참고용으로만 해석합니다.


```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
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



# 데이터 동기화 작업
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
 
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
