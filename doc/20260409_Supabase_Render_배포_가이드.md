# Supabase + Render 배포 가이드

이 문서는 `Supabase DB + Render 웹앱` 조합으로 Lee Trader를 처음 배포하는 기준입니다.

현재 프로젝트 기준으로 가장 안전한 1차 배포 구조는 아래와 같습니다.

- `Supabase`: 운영 DB
- `Render`: Node 웹앱
- `파이프라인`: 당분간 로컬 PC에서 실행

이유는 단순합니다.

- 웹앱은 지금 바로 Render에 올릴 수 있습니다.
- DB는 이미 Postgres 구조라 Supabase와 잘 맞습니다.
- 반면 파이프라인은 아직 `CSV/JSON 산출물 + DB`를 함께 쓰는 부분이 있어, 무료 Render에 바로 모두 올리기에는 운영 복잡도가 큽니다.

즉 1차 목표는 `웹앱 공개 + DB 이관`입니다.

## 1. 먼저 이해할 구조

배포 후 데이터 흐름은 아래처럼 잡는 것이 좋습니다.

1. 내 PC에서 파이프라인 실행
2. 파이프라인이 Supabase DB에 결과 저장
3. Render 웹앱이 Supabase DB를 읽어 화면 제공

이 구조면 처음 배포할 때 가장 덜 헷갈립니다.

## 2. Supabase에서 해야 할 일

중요: 현재 캡처처럼 `Storage` 화면이 아니라 `Database` 기준으로 작업해야 합니다.

### 2-1. Supabase 프로젝트 생성

1. Supabase에서 새 프로젝트를 만듭니다.
2. 프로젝트 생성이 끝나면 상단 `Connect` 버튼을 눌러 접속 문자열을 확인합니다.

공식 문서:

- Supabase DB 연결: https://supabase.com/docs/guides/database/connecting-to-postgres
- Supabase 데이터 가져오기: https://supabase.com/docs/guides/database/import-data

### 2-2. 스키마 먼저 생성

이 저장소의 [schema.sql](/d:/ai/Lee_trader/schema.sql)을 Supabase SQL Editor에서 먼저 실행합니다.

권장 순서:

1. Supabase Dashboard
2. SQL Editor
3. [schema.sql](/d:/ai/Lee_trader/schema.sql) 전체 실행

이 프로젝트는 기본적으로 Postgres 문법을 사용하므로 Supabase에서 그대로 사용할 수 있습니다.

### 2-3. 기존 DB 데이터 이관

가장 쉬운 방식은 `pg_dump -> psql` 입니다.

로컬 DB dump:

```powershell
$dump = docker exec -i lee_trader_pg pg_dump `
  --data-only `
  --inserts `
  --no-owner `
  --no-privileges `
  -U lee_trader `
  -d lee_trader 2>$null

Set-Content -Path 'lee_trader_data.sql' -Value $dump -Encoding UTF8

```

Supabase로 import:

```powershell
$env:PGPASSWORD="YOUR_SUPABASE_DB_PASSWORD"
Get-Content .\lee_trader_data.sql | docker exec -i lee_trader_pg psql `
  -h aws-1-ap-northeast-2.pooler.supabase.com `
  -p 5432 `
  -U postgres.wlkyypcakkrjmscfujdp `
  -d postgres


```

주의:

- 실제 값은 Supabase `Connect`에서 복사한 값을 사용합니다.
- Render 웹앱 용도는 `pooler session mode` 문자열이 가장 무난합니다.
- import 전에 [schema.sql](/d:/ai/Lee_trader/schema.sql)을 먼저 적용해 두는 편이 안전합니다.
- 로컬 PC에 `psql`이 없어도 `lee_trader_pg` 컨테이너 안의 `psql`로 그대로 실행할 수 있습니다.

### 2-4. import 후 확인할 테이블

최소한 아래 테이블은 row 수와 최신일을 확인하는 게 좋습니다.

- `stocks`
- `market_status`
- `fundamentals`
- `quality`
- `features`
- `predictions`
- `daily_ranking`
- `research.app_payload_store`

권장 확인 SQL:

```sql
select count(*) from stocks;
select max(date) from market_status;
select max(date) from features;
select max(date) from predictions;
select max(date) from daily_ranking;
select payload_key, updated_at
from research.app_payload_store
order by updated_at desc
limit 20;
```

### 2-5. CSV로 테이블별 이관

`psql` import가 막히거나 일부 테이블만 먼저 옮길 때는 CSV 방식으로 진행할 수 있습니다.
로컬 DB에서 Supabase Table Editor 업로드용 CSV를 먼저 생성합니다.

CSV export:

```powershell
cd D:\ai\Lee_trader
python python\export_db_tables_to_csv.py
```

생성 경로:

- [exports/db_csv/public](/d:/ai/Lee_trader/exports/db_csv/public)
- [exports/db_csv/research](/d:/ai/Lee_trader/exports/db_csv/research)
- [exports/db_csv/manifest.json](/d:/ai/Lee_trader/exports/db_csv/manifest.json)

권장 업로드 순서:

1. `public`
- `stocks`
- `theme_master`
- `etf_master`
- `prices_raw`
- `prices_clean`
- `prices_adjusted`
- `market_status`
- `fundamentals`
- `quality`
- `features`
- `labels`
- `predictions`
- `daily_scores`
- `daily_ranking`

2. 운영성 테이블
- `pipeline_history`
- `trade_audit_log`
- `page_view_events`
- `backtest_trades`

3. `research`
- `dim_model_run`
- `prediction_history`
- `ranking_history`
- `backtest_outcome`
- `app_payload_store`
- `paper_trading_run`
- `paper_trading_position`
- `paper_trading_nav`

주의:

- `research` 테이블은 Table Editor의 schema를 `research`로 바꿔야 보입니다.
- [exports/db_csv/manifest.json](/d:/ai/Lee_trader/exports/db_csv/manifest.json) 으로 테이블별 row 수를 비교하면 누락 확인이 쉽습니다.
- 우선 확인이 목적이면 `stocks`, `market_status`, `features`, `predictions`, `daily_ranking`, `research.app_payload_store` 부터 올려도 됩니다.

## 3. Render에서 해야 할 일

공식 문서:

- Render Blueprint: https://render.com/docs/blueprint-spec
- Render Docker 배포: https://render.com/docs/docker
- Render 무료 플랜: https://render.com/docs/free

이 저장소에는 이미 [render.yaml](/d:/ai/Lee_trader/render.yaml)이 들어 있습니다.

### 3-1. Render에 GitHub 저장소 연결

1. Render Dashboard 접속
2. `New +`
3. `Blueprint` 선택
4. GitHub 저장소 연결
5. 이 저장소 선택

그러면 Render가 [render.yaml](/d:/ai/Lee_trader/render.yaml)을 읽고 웹 서비스를 생성합니다.

### 3-2. Render 환경변수 설정

기준 파일은 [.env.render.example](/d:/ai/Lee_trader/.env.render.example) 입니다.

최소한 아래 값은 직접 넣어야 합니다.

- `DATABASE_URL`
- `SITE_BASE_URL`
- `OPERATOR_PASSWORD`

Render가 자동 생성하도록 둬도 되는 값:

- `OPERATOR_AUTH_SECRET`
- `ADMIN_TOKEN`

#### DATABASE_URL 권장값

Supabase `Connect`에서 `Session pooler` 문자열을 복사해서 사용하세요.

예:

```text
postgres://postgres.PROJECT_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
```

#### SITE_BASE_URL 예시

```text
https://lee-trader-web.onrender.com
```

#### OPERATOR_PASSWORD

운영자 페이지와 매수/매도 기능에 사용할 비밀번호입니다.

## 4. 배포 후 바로 확인할 것

아래 순서로 확인하면 됩니다.

1. 홈: `/`
2. 공개 앱: `/app`
3. 운영자 로그인: `/operator-login`
4. 운영자 페이지: `/ops-readiness.html`
5. 수동매매: `/manual-trading.html`
6. 리서치 랭킹: `/ranking.html`
7. 상세 페이지 예시: `/detail.html?code=145020`

필수 확인 항목:

- 화면이 열리는지
- 날짜가 최신 기준일과 맞는지
- `manual-trading`, `ops-readiness`, `ranking`이 정상 응답하는지
- 운영자 로그인/로그아웃이 동작하는지

## 5. 로컬 파이프라인을 Supabase로 붙이는 방법

호스팅 후에도 당분간 파이프라인은 로컬에서 돌리는 것이 좋습니다.

즉 로컬 `.env`의 `DATABASE_URL`만 Supabase 쪽으로 바꾸면 됩니다.

예:

```env
DATABASE_URL=postgres://postgres.PROJECT_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
```

그 다음 평소처럼 실행합니다.

```powershell
docker compose up -d python-pipeline
python python/run_operational_refresh.py
docker compose up -d --build node-api
```

이렇게 하면 파이프라인 결과가 Supabase DB에 쌓이고, Render 웹앱이 그 결과를 읽습니다.

## 6. 지금 당장 하지 않는 것이 좋은 것

처음부터 아래까지 한 번에 하지 않는 편이 좋습니다.

- Render에 파이프라인까지 같이 올리기
- Supabase Storage까지 같이 쓰기
- 회원가입/회원관리 붙이기

처음에는 구조를 단순하게 유지해야 문제를 빨리 찾을 수 있습니다.

## 7. 무료 플랜에서 예상해야 할 점

무료 플랜에서는 아래를 감안해야 합니다.

- Render 웹 서비스가 유휴 상태 후 느리게 깨어날 수 있음
- Supabase 무료 플랜은 용량/성능 제한이 있음
- 대량 import 시 시간 제한이나 성능 저하가 생길 수 있음

그래서 첫 배포는 `지인/베타 공개`에 적합하고, 불특정 다수 공개는 실제 사용량을 본 뒤 판단하는 편이 맞습니다.

## 8. 배포 전에 꼭 할 것

현재 로컬 `.env`에 들어 있는 실제 비밀값은 배포 전에 새 값으로 교체하는 것이 맞습니다.

반드시 새로 발급하거나 변경할 것:

- DB 비밀번호
- KIS API 키
- 운영자 비밀번호
- 운영자 인증 시크릿
- 관리자 토큰

## 9. 첫 배포 추천 순서

1. Supabase 프로젝트 생성
2. [schema.sql](/d:/ai/Lee_trader/schema.sql) 실행
3. 로컬 DB dump
4. Supabase import
5. Render Blueprint 생성
6. Render 환경변수 입력
7. 배포
8. `/app`, `/ops-readiness.html`, `/operator-login` 확인
9. 로컬 파이프라인을 Supabase DB로 연결
10. 하루 운영해 보고 날짜/산출물 흐름 점검

## 10. 지금 기준의 권장 결론

지금은 아래 조합이 가장 현실적입니다.

- `Supabase`: 운영 DB
- `Render`: 웹앱
- `로컬 PC`: 파이프라인 실행

이 조합으로 먼저 운영을 안정화한 뒤, 나중에 필요하면 파이프라인 자동화만 별도로 옮기면 됩니다.
