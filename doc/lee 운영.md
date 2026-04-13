
---

## 1. 권장 데일리 실행 순서

### Render 운영 기준

- Render 배포 기준 상시 서비스는 `node-api`가 아니라 Render 웹앱입니다.
- 정시 Python 배치는 Docker `scheduler` 대신 GitHub Actions로 실행합니다.
- GitHub Actions 기준 문서: [20260410_GitHub_Actions_스케줄러_가이드.md](/d:/ai/Lee_trader/doc/20260410_GitHub_Actions_%EC%8A%A4%EC%BC%80%EC%A4%84%EB%9F%AC_%EA%B0%80%EC%9D%B4%EB%93%9C.md)
- GitHub workflow:
  - [intraday-refresh.yml](/d:/ai/Lee_trader/.github/workflows/intraday-refresh.yml)
  - [close-batch.yml](/d:/ai/Lee_trader/.github/workflows/close-batch.yml)
- `flow_daily` 수집은 현재 운영에서 비활성화 상태입니다.
- 이유: 실운영 점수/랭킹에 직접 연결되지 않고, 데이터 신뢰성과 활용 방식을 추가 검토 중입니다.
- 따라서 현재 close 배치와 로컬 기본 운영 절차에서는 `download_flows_kis.py`, `check_flow_ingestion.py`를 자동 실행하지 않습니다.

### 단계별 실행

GitHub Actions 기준:

```powershell
# 1. GitHub Actions workflow에서 정시 실행
# - Intraday Refresh
# - Close Batch

# 2. 로컬에서 수동 보정이 필요할 때만 운영 산출물 전체 갱신
python python/run_operational_refresh.py

# 3. 화면 반영 확인이 필요하면 Node API 재기동
docker compose up -d --build node-api
```

로컬 Docker 수동 점검 기준:

```powershell
# python-pipeline 컨테이너는 상시 서비스가 아니라 1회 실행 배치입니다.
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose run --rm python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```


### 한 번에 처리

GitHub Actions용 해석:

- GitHub Actions에서는 `docker compose build/up`를 쓰지 않습니다.
- workflow 내부에서 `actions/setup-python` 후 `python python/run_scheduled_job.py ...`를 직접 실행합니다.
- 따라서 이 문서의 Docker 명령은 로컬 수동 점검용으로만 해석합니다.

로컬 수동 점검을 한 번에 처리할 때:

```powershell
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose run --rm python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python python/run_operational_refresh.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

### CSV 백업 / 복구

powershell
python python\backup_csv_md_zip.py --output backups\csv_backup_20260410.zip --overwrite
python python\restore_csv_md_zip.py --zip backups\csv_backup_20260409.zip --overwrite


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

- 현재 로컬 최신 CSV를 기준으로 `stocks`, `market_status`, `fundamentals`, `quality`, `features`, `predictions`, `daily_ranking`를 웹 DB에 직접 반영
- 결과 확인:
  - [csv_db_parity_report.json](/d:/ai/Lee_trader/outputs/csv_db_parity_report.json)
  - [csv_db_parity_report.md](/d:/ai/Lee_trader/outputs/csv_db_parity_report.md)
- payload DB는 별도이므로 필요 시 이후 `python python/run_operational_refresh.py` 실행

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

## git 파일 복사
 python python/export_git_release.py


---
## 14. Node / Pipeline 운영 메모

### Pipeline 컨테이너 재생성

```powershell
docker compose build --no-cache --progress=plain python-pipeline
docker compose run --rm python-pipeline
```

- `python-pipeline`은 백그라운드 상시 서비스가 아니라 1회 실행 배치입니다.
- 따라서 `docker compose up -d python-pipeline`보다 `docker compose run --rm python-pipeline`을 우선 사용합니다.
- GitHub Actions에서는 이 Docker 절차를 쓰지 않고 Python 직접 실행을 사용합니다.


### Node API 재기동

```powershell
docker compose build --no-cache --progress=plain node-api
docker compose up -d node-api
docker compose up -d --build node-api
```


### 컨테이너 내부 파일 확인

```powershell
docker run --rm -it --entrypoint sh lee_trader-python-runtime:latest
sed -n '1,120p' /app/python/run_pipeline.py
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

powershell
cd D:\ai\git\lee_trader

## git 파일 복사
 python python/export_git_release.py

# 작업 전에 최신 원격 반영
git fetch origin
git switch main
git pull --ff-only origin main


# 로컬 배포본 다시 생성 후 커밋
cd D:\ai\Lee_trader
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target


cd D:\ai\git\lee_trader
git status
git add --all
git commit -m "update deploy package"

git push origin main


### 메모

- `git clone` 방식과 `git init + remote add origin` 방식은 같이 쓰지 않습니다.
- `D:\ai\git\lee_trader`가 이미 `git clone`으로 만들어진 폴더라면 `git init`, `git remote add origin`은 다시 하지 않습니다.
- 실제 비밀값은 `.env`에 두고, git에는 `.env.example`, `.env.render.example`만 올립니다.
- `data`, `logs`, `outputs`, `backups` 같은 산출물은 git에 올리지 않습니다.
