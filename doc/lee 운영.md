
---

## 1. 권장 데일리 실행 순서

### 문서 역할

- 이 문서는 `운영 기준 + 실제 실행 기준` 문서입니다.
- 배포 자체의 Git 절차는 `doc/git 배포 절차.md`를 우선 참고합니다.
- 자동매매 상태 해석은 이 문서와 `doc/20260417_자동매매 설계.md`를 기준으로 봅니다.

### 자동 스케줄 기준 시각

- 종가 close 배치: `16:00`
- 장중 refresh: `12:00`
- 실자동매매 신규 매수 판단: `09:30` 본 실행, `10:00` 재시도 슬롯
- 실계좌 동기화: `10:00, 14:00, 18:00`

메모:

- `09:30`은 전날 종가 기준 `top20` 후보를 바탕으로, 장초반 갭/과열/추격매수 위험을 1차 확인한 뒤 신규 매수 판단을 내리는 운영 기준 시각이다.
- `10:00`은 추가 매수 전략 시간이 아니라 `09:30` 미성공 시 1회 재시도하는 복구 슬롯이다.
- 자동매수 다중 슬롯은 `SCHEDULER_AUTO_BUY_MULTI_SLOT_SUCCESS_POLICY=once_per_day` 기준으로 운영하며, 당일 `09:30` 실행이 성공하면 `10:00`은 스킵한다.
- 따라서 실자동매매는 `12:10` 장중 신규 진입보다 `09:30` 장초반 보수 집행이 현재 설계와 더 잘 맞는다.
- 장중 `12:00` refresh는 신규 매수 주 실행 시간이 아니라 관제와 재평가 성격으로 본다.
- `09:30` auto-buy 스케줄러는 이제 `run_operational_refresh -> submit_live_orders -> sync_live_account_holdings -> sync_web_display_data` 순서로 동작한다.
- `submit_live_orders.py`는 이제 KIS 접근토큰을 `outputs/kis_access_token_cache.json`에 캐시해, 같은 분 안의 후속 프로세스 재발급으로 `1분당 1회` 제한에 걸리지 않도록 보강했다.
- 따라서 주문 직후 웹 조회에서도 `order execution`뿐 아니라 `실계좌 보유/현금`이 최대한 빠르게 갱신되도록 맞춘다.
- 매도는 `EXIT/TRIM` 실패분 재시도를 허용하되, 매도 성공 자금을 같은 날 10:00 신규 BUY로 자동 재투입하지 않는다.
- 교체매매는 같은 사이클의 명시적 replacement pair 또는 다음 영업일 신규 BUY로만 운영한다.
- 현재 `PILOT`은 실험적 소액 모드가 아니라 제한적 실운용 모드로 해석한다.
- 제한적 실운용 신규 BUY 상한은 최대 4종목, 총 신규 노출 35%, 종목당 신규 진입 12%다.
- 단, `max_holdings=8`과 현재 보유 종목 수를 먼저 적용하므로 보유가 이미 많으면 신규 BUY 수는 줄어든다.
- 가격 리스크 매도 기준은 `-5% REVIEW`, `-8% TRIM`, `-12% EXIT 후보`, `+10% 수익보호 TRIM 후보`다.
- confidence 데이터 누락은 데이터 문제와 종목 문제를 구분하기 위해 자동 EXIT가 아니라 REVIEW로 둔다.
- `PILOT`에서는 점수 기반 교체매매를 자동 실행하지 않고 REVIEW로 보류한다.
- Docker Postgres 호스트 포트는 `15432:5432` 기준으로 사용한다.
- 즉 로컬 PC에서 직접 붙을 때는 `localhost:15432`, 컨테이너 내부에서는 계속 `postgres:5432`를 사용한다.

### 한 번에 처리


```powershell
.venv\Scripts\python.exe python/run_manual_close_batch.py
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

### 로그 보는 방법

운영 로그는 `실시간 Docker 로그`와 `파일 로그`를 같이 봅니다.

실시간 확인:

```powershell
docker compose logs -f scheduler
docker compose logs -f scheduler-recovery
docker compose logs -f scheduler-auto-buy
docker compose logs -f scheduler-live-account-sync
docker compose logs -f node-api
```

최근 100줄만 빠르게 보기:

```powershell
Get-Content .\logs\auto_ops_scheduler.log -Tail 100
Get-Content .\logs\auto_ops_recovery_scheduler.log -Tail 100
Get-Content .\logs\auto_ops_auto_buy_scheduler.log -Tail 100
Get-Content .\logs\auto_ops_live_account_sync_scheduler.log -Tail 100
```

로그 파일 위치:

- 종가 close 배치: `logs/auto_ops_scheduler.log`
- 장중 refresh: `logs/auto_ops_recovery_scheduler.log`
- 자동매수: `logs/auto_ops_auto_buy_scheduler.log`
- 실계좌 동기화: `logs/auto_ops_live_account_sync_scheduler.log`

상태 파일도 같이 확인:

- close 배치 상태: `outputs/auto_ops_scheduler_status.json`
- 장중 refresh 상태: `outputs/auto_ops_recovery_scheduler_status.json`
- 자동매수 상태: `outputs/auto_ops_auto_buy_scheduler_status.json`
- 실계좌 동기화 상태: `outputs/auto_ops_live_account_sync_scheduler_status.json`

운영 해석 기준:

- `logs/*.log`는 실행 상세 로그를 봅니다.
- `outputs/*_status.json`은 최근 성공 시각, 최근 실패 시각, 마지막 에러를 빠르게 확인할 때 봅니다.
- 이상 징후가 있으면 먼저 `status.json`에서 `last_error`, `last_success_at`를 보고, 그 다음 대응되는 `logs/*.log` 마지막 100줄을 확인합니다.

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
## 파일이관 
python python\import_git_release.py --source D:\ai\git\lee_trader --clean-target

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

.venv\Scripts\python.exe python/run_manual_close_batch.py
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

.venv\Scripts\python.exe python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
 
python python\export_git_release.py --target D:\ai\git\lee_trader --clean-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
## 2026-04-18 자동매매 기준 메모

- 현재 자동매매 후보 우주는 `전일 ranking_final 상위 top20`입니다.
- 실제 신규 진입 심사는 `top8` 기본, `top9~10` 조건부 확장 구조입니다.
- 실제 보유 목표 상한은 `8종목` 기준으로 운용합니다.
- `top5`는 폐기한 것이 아니라 최우선 강신호 버킷으로 해석합니다.
- `PILOT` 상태에서는 제한적 실운용 모드가 켜져 있습니다.
  - 최대 `4종목`
  - 총 신규 노출 `30%`
  - 종목당 신규 진입 상한 `12%`
- `WATCH` 상태에서는 소액 실거래 모드가 켜져 있습니다.
  - 최대 `2종목`
  - 총 신규 노출 `15%`
  - 종목당 신규 진입 상한 `8%`
- `HOLD`와 `BLOCK`에서는 신규 진입을 하지 않습니다.

### 2026-04-18 게이트 완화 반영

- `walkforward_acceptance = REJECTED`가 나와도 곧바로 `BLOCK`으로 고정하지 않습니다.
- 아래 조건을 만족하면 `soft rejection`으로 보고 `WATCH`까지는 허용합니다.
  - `top20_excess_return_positive`
  - `execution_evidence_ok_or_unavailable`
  - 실패 사유가 `ordering_not_stable`, `drawdown_too_deep`, `confidence_monotonicity_missing` 중심일 것
- 유동성 block 임계값은 현재 검증된 랭킹 운용 기준에 맞춰 아래처럼 완화했습니다.
  - `max_liquidity_risk_ratio: 0.30 -> 0.40`
  - `max_very_low_liquidity_ratio: 0.20 -> 0.40`
  - 비교 방식도 `>=`가 아니라 `>`로 적용합니다.
- 2026-04-18 `PILOT` 구현 후 재검증 결과 현재 기준일 `2026-04-17`의 게이트는 `PILOT`입니다.
- 따라서 현재 자동매매는 `PILOT 제한 실운용` 규칙으로 동작하며, 현재 산출물 기준 신규 진입 3건까지 생성됩니다.

### PILOT 구현 메모

- 현재 게이트는 `BLOCK -> HOLD -> WATCH -> PILOT -> BUY_ALLOWED` 구조입니다.
- 권장 의미는 아래와 같습니다.
  - `WATCH`: 탐색적 소액 진입
  - `PILOT`: 제한적 실운용
  - `BUY_ALLOWED`: 정식 자동매수
- 현재 코드는 `PILOT` 단계를 실제 구현한 상태입니다.
- `PILOT`에서는 제한적 신규 진입만 허용하고, `BUY_ALLOWED` 수준의 풀 비중 운용과 교체매매는 아직 허용하지 않습니다.
- 관련 메모는 `doc/20260418_운영 상태 및 PILOT 방향 메모.md`를 기준으로 같이 봅니다.
- 자동매매 이력은 `실자동매매` 화면에서만 확인합니다.
- `거래내역` 화면의 `public.trades`는 수동 장부와 확정 거래 기준으로 유지합니다.
- 자동매매 산출물인 `trade_intents`, `order_requests_preview`, `order_requests_execution`, `live_account_holdings`는 `public.trades`와 섞어서 해석하지 않습니다.

### 2026-04-20 장전 체크리스트

- `2026-04-19`는 일요일 휴장입니다. 다음 실질 자동매수 시점은 `2026-04-20 09:30`입니다.
- `2026-04-20 09:00~09:20` 사이에 아래를 먼저 확인합니다.

```powershell
docker compose up -d postgres
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-recovery
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-auto-buy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-live-account-sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose ps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

- 자동매수 스케줄러가 살아 있는지 확인합니다.

```powershell
docker compose logs --tail=100 scheduler-auto-buy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

- 장전 최종 산출물을 확인합니다.

```powershell
Get-Content outputs\operational_buy_gate.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-Content outputs\trade_intents.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-Content outputs\order_requests_preview.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

- 현재 기대 기준은 아래와 같습니다.
  - `operational_buy_gate.json`: `overall_status = PILOT`
  - `trade_intents.json`: `gate_status = PILOT`, `BUY` 3건 포함
  - `order_requests_preview.json`: `gate_status = PILOT`, `BUY 3건`, `SELL 3건`
- `09:30`에 무조건 실제 매수가 체결되는 것은 아닙니다.
- `PILOT` 상태, 계좌 현금, KIS 주문 가능 수량, 차단 사유를 다시 반영한 뒤 실제 주문 여부가 결정됩니다.

### 장중 auto_buy 검증 절차

- 목적: `tokenP 1분당 1회` 오류가 해소됐는지와 장중 주문 API 경로가 끝까지 도는지 확인
- 권장 시각: `09:30` 직후 또는 장중 연속호가 시간
- 비권장 시각: 장 시작 전, 장 종료 후, 점검 시간

사전 확인:

```powershell
Get-Content outputs\operational_buy_gate.json
Get-Content outputs\trade_intents.json
Get-Content outputs\order_requests_preview.json
Get-Content outputs\order_buy_approvals.json
Get-Content outputs\kis_access_token_cache.json
```

기대 포인트:

- `operational_buy_gate.json`: `overall_status`가 현재 운영 기대와 일치할 것
- `trade_intents.json`: `gate_status`, `BUY/SELL` intent 수가 기대와 크게 다르지 않을 것
- `order_requests_preview.json`: `executable_now=true` 항목이 실제 주문 후보로 내려올 것
- `order_buy_approvals.json`: BUY를 실제 제출할 경우 `approved_request_ids`가 비어 있지 않을 것
- `kis_access_token_cache.json`: 최근 시각으로 갱신돼 있을 것

장중 수동 검증 명령:

```powershell
docker compose exec -T scheduler-auto-buy python python/submit_live_orders.py --execute --confirm-text LIVE_ORDER --allow-buy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

실행 직후 확인:

```powershell
Get-Content outputs\order_requests_execution.json
Get-Content outputs\auto_ops_auto_buy_scheduler_status.json
Get-Content .\logs\auto_ops_auto_buy_scheduler.log -Tail 100
```

성공 판정:

- `order_requests_execution.json` 생성됨
- 로그에 `Using cached KIS access token` 또는 토큰 발급 성공 후 정상 진행 로그가 보임
- `submission_status=submitted`가 1건 이상 있거나, 적어도 프로세스 전체가 `exit 1` 없이 끝남
- `auto_ops_auto_buy_scheduler_status.json`의 다음 정상 사이클에서 `status=idle`, `last_success_at` 갱신

실패 해석:

- `tokenP failed ... 1분당 1회`
  - 토큰 재사용 경로 이상 여부 확인
  - `outputs/kis_access_token_cache.json` 생성/갱신 여부 확인
- `장운영시간이 아닙니다`
  - 코드 문제가 아니라 실행 시각 문제
- `buy_approval_required`
  - BUY 승인 목록이 비어 있어서 스킵된 것
- `submission_status=failed`와 KIS business error
  - 토큰 문제가 아니라 주문 API 또는 시장 상태 문제

운영 메모:

- 화면에 `auto_buy 오류`가 남아 있어도, 그 값은 직전 실패 상태를 보여줄 수 있습니다.
- 장중 정상 사이클이 한 번 완료돼야 `Scheduler Runtime`의 `auto_buy` 행도 자동으로 정상화됩니다.
- 상태 파일을 수동으로 성공처럼 고치기보다, 장중 실제 검증으로 갱신하는 쪽이 운영상 안전합니다.


# 수동 배포
docker compose build --no-cache --progress=plain python-pipeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.venv\Scripts\python.exe python/run_manual_close_batch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 컨테이너 역할
scheduler: 16:00 close 배치
scheduler-recovery: 12:00 intraday refresh
scheduler-auto-buy: 09:30 자동매매, 09:30 미성공 시 10:00 재시도
scheduler-live-account-sync: 실계좌 동기화
node-api: 웹 조회 API
postgres: 웹/API/동기화 대상 DB

# 재배포
docker compose up -d --build scheduler-auto-buy
docker compose up -d --build scheduler-live-account-sync
docker compose up -d --build node-api

# 상시운용 컨테이너 
docker compose up -d postgres
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d --build node-api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-recovery
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-auto-buy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-Sleep -Seconds 65

docker compose up -d scheduler-live-account-sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose logs -f scheduler
