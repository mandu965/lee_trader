
---

## 1. 권장 데일리 실행 순서

### 한 번에 처리


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
python python\backup_csv_md_zip.py --output backups\csv_backup_20260413.zip --overwrite
python python\restore_csv_md_zip.py --zip backups\csv_backup_20260409.zip --overwrite


## DB->CSV
python python\export_db_tables_to_csv.py실제

### 최신 CSV 기준 웹 DB 동기화

```powershell
$env:DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres"
python python/sync_csv_db_parity.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:DATABASE_URL="postgresql://postgres.wlkyypcakkrjmscfujdp:!760595leeuser@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"
python python/sync_csv_db_parity.py


```
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
