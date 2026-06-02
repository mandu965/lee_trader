# LeeTrader PRD
# SEO Content Automation Module (경량 MVP)

Version: 3.0

---

# 1. 개요

LeeTrader 종가 배치가 완료된 후, 매일 생성되는 데이터를 템플릿에 적용하여 블로그 게시용 글 초안을 **파일로 출력**하는 경량 스크립트를 만든다.

운영자는 출력된 파일을 열어 내용을 확인하고, 복사하여 네이버 블로그 또는 티스토리에 **직접 수동 게시**한다.

본 모듈은 DB·API·웹 UI 없이 동작하는 **단일 생성 스크립트 + 템플릿 + 출력 폴더** 구조의 내부 운영 도구이다.

본 기능의 목적은 다음과 같다.

- 검색 유입 증가
- 블로그 콘텐츠 자산 축적
- LeeTrader 브랜드 노출 증가
- 사이트 방문자 증가
- 애드센스 수익 기반 확보

---

# 2. 문제 정의

현재 LeeTrader는 다음 데이터를 매일 생성한다.

- 추천 종목 (serving/daily_recommendations.json)
- 랭킹 순위 (public.daily_ranking 테이블)
- 시장 상태 (public.market_status 테이블, daily_recommendations.json의 regime)

그러나 해당 데이터가 외부 검색 유입으로 연결되지 않고 있다.

현재 블로그 작성은 수동으로 수행되고 있으며 다음 문제가 존재한다.

- 작성 시간 소요
- 게시 빈도 부족
- SEO 콘텐츠 부족
- 데이터 활용도 부족

---

# 3. 목표

## 단기 목표

매일 데이터 기반 글 초안을 파일로 자동 출력

운영자가 복사하여 즉시 게시 가능한 수준으로 제공

---

## 중기 목표

6개월 내 180개 이상 콘텐츠 누적 게시

---

## 장기 목표

1년 내 365개 이상 콘텐츠 누적 게시

네이버 / 티스토리 검색 유입 확보

LeeTrader 브랜드 구축

---

# 4. 범위

## 포함

데이터 리포트형 콘텐츠 (추천 종목, 랭킹 변화, 시장 분석)

네이버 블로그용 plain text 파일 출력

티스토리용 markdown 파일 출력

Jinja2 템플릿 기반 생성 스크립트

---

## 제외

자동 게시 (블로그 업로드는 운영자가 수동 수행)

SNS 자동 발행

LLM / GPT API 실시간 호출

content_drafts DB 테이블, 콘텐츠 API, 웹 UI (→ §15 향후 확장)

LeeTrader 사이트 내부 `/blog`, `/reports`, `/api/site-library` 연동

`node/content/` Markdown 파일 자동 생성

---

## 기존 사이트 콘텐츠와의 관계

기존 LeeTrader 사이트는 `node/content/blog`, `node/content/reports`의 Markdown 파일을 읽어 `/blog`, `/reports`에 공개한다.

본 모듈이 출력하는 파일은 **네이버/티스토리 외부 블로그 게시용 초안**이며, LeeTrader 사이트에 자동 노출하지 않는다. `node/content/`를 변경하지 않는다.

---

# 5. 시스템 구조

## 전체 흐름

```
종가 배치 완료 (로컬 Windows 머신, 18:10)
  ↓
serving/daily_recommendations.json 생성
public.daily_ranking / public.market_status 갱신
  ↓
[19:30] generate_blog_posts.py 실행 (신규, 동일 로컬 머신)
  ├─ 데이터 읽기: JSON 파일 + DB(db.py)
  └─ Jinja2 템플릿 적용
  ↓
outputs/blog_drafts/ 에 파일 출력
  2026-06-02_typeA_naver.txt
  2026-06-02_typeA_tistory.md
  2026-06-02_typeB_naver.txt
  ...
  ↓
운영자가 폴더에서 파일 열기 → 복사
  ↓
네이버 블로그 또는 티스토리에 수동 게시
```

## 구현 상태 및 실행 위치

기존 종가 배치는 **로컬 Windows 머신의 작업 스케줄러**(`scripts/register_daily_operations_task.ps1`)로 18:10에 실행 중이며, `DATABASE_URL`로 공유 PostgreSQL에 결과를 기록한다.

신규 콘텐츠 생성 스크립트는 **종가 배치가 도는 동일 로컬 머신**에서 종가 배치 완료 이후 실행한다. Render 클라우드에 별도 서비스를 만들지 않는다. 기존 종가 배치 코드는 변경하지 않는다.

신규 구현 대상

- `python/generate_blog_posts.py` (생성 스크립트)
- `templates/blog/` (Jinja2 템플릿)
- `outputs/blog_drafts/` (출력 폴더)
- 스케줄 등록 (선택: 수동 실행으로 시작 가능)

---

# 6. 데이터 소스

## 6.1 daily_recommendations.json

경로: serving/daily_recommendations.json

TYPE_A 생성에 사용

주요 필드

- asof_date
- gate_overall_status
- walkforward_acceptance_status
- items[].security.code / name / market / sector
- items[].scores.final_score / confidence_score
- items[].market_signals.regime / regime_reason / pred_return_60d / pred_mdd_60d
- items[].buy_eligibility.status
- items[].selection.buyability_status

---

## 6.2 daily_ranking (PostgreSQL 테이블)

TYPE_B 생성에 사용 (주 소스). `python/db.py`로 조회

주요 필드

- date
- code
- name
- rank_final
- final_score
- sector

비고

`daily_ranking`에는 `name`, `sector`가 포함되어 별도 조인이 필요 없다.

직전 영업일 대비 순위 변화는 두 날짜(최신 영업일, 직전 영업일)로 조회하여 `rank_final` 차이를 계산한다.

```sql
WITH latest AS (
  SELECT code, name, sector, rank_final, final_score
  FROM public.daily_ranking WHERE date = :today
),
prev AS (
  SELECT code, rank_final AS prev_rank
  FROM public.daily_ranking WHERE date = :prev_business_day
)
SELECT l.code, l.name, l.sector, l.rank_final,
       p.prev_rank, (p.prev_rank - l.rank_final) AS rank_gain, l.final_score
FROM latest l JOIN prev p USING (code)
ORDER BY rank_gain DESC;
```

비고: `research.ranking_history` 테이블도 존재하나 컬럼 구조가 다르고(`rank`, 복합 PK, name/sector 없음) 사용 제약이 있어 MVP에서는 쓰지 않는다.

---

## 6.3 시장 상태 (TYPE_C)

데이터 소스 우선순위 (`python/blog_datasources.get_market_status`):

1. `DATABASE_URL` 있으면 `public.market_status` 테이블 (`python/db.py`)
2. 없으면 `outputs/market_status_validation_report.json`의 `latest` 필드 (종가 배치 산출물, **실데이터 파일**)
3. 둘 다 없으면 `fixtures/blog/market_status_sample.json` (개발용 mock)

DB가 연결되지 않는 개인 PC에서도 (2)의 실데이터 파일로 정상 생성된다.

주요 필드

- date
- kospi_close
- kospi_ma20
- volatility_5d
- foreign_net_5d
- market_up (파일 경로에서는 `close_gt_ma20`으로 파생)

비고

regime(상승/중립/방어) 정보는 daily_recommendations.json의 `items[0].market_signals.regime` 사용

---

# 7. 콘텐츠 유형 (데이터 리포트형)

## TYPE_A — 오늘의 AI 관찰 종목 / 추천 후보

생성 주기: 매일 (영업일)

데이터 소스: daily_recommendations.json

표현 규칙

- `gate_overall_status=BUY_ALLOWED`이고 `walkforward_acceptance_status=ACCEPTED`인 경우에만 "추천 후보" 표현 사용
- 그 외 상태에서는 "관찰 종목" 표현 사용
- `buy_eligibility.status` / `selection.buyability_status`가 `WATCH` 또는 `BLOCK`인 종목은 매수 가능한 종목처럼 표현하지 않음

---

## TYPE_B — 랭킹 급상승 종목

생성 주기: 매일 (영업일)

데이터 소스: daily_ranking (직전 영업일 대비 rank_final 변화)

---

## TYPE_C — 시장 분석 리포트

생성 주기: 매일 (영업일)

데이터 소스: market_status + daily_recommendations.json (regime)

---

비고: 자동매매 성과 리포트(주간/월간)는 현재 매매 데이터가 사실상 비어 있어(`live_account.available=false`, 보유 0건) MVP 범위에서 제외한다. 매매 데이터 축적 후 §15 확장 단계에서 검토한다.

---

# 8. 출력 포맷 및 파일 규칙

## 8.1 출력 위치 및 파일명

출력 폴더: `outputs/blog_drafts/`

파일명 규칙: `{source_date}_{type}_{platform}.{ext}`

```
2026-06-02_typeA_naver.txt
2026-06-02_typeA_tistory.md
2026-06-02_typeB_naver.txt
2026-06-02_typeB_tistory.md
2026-06-02_typeC_naver.txt
2026-06-02_typeC_tistory.md
```

동일 파일이 이미 존재하면 덮어쓰지 않고 건너뛴다(중복 생성 방지). 재생성이 필요하면 기존 파일 삭제 후 실행한다.

---

## 8.2 네이버 블로그 (plain text)

```
[제목]
{날짜} AI 분석 기반 오늘의 관찰 종목 TOP5

[본문 구성]
1. 오늘의 시장 상태
2. TOP5 종목별 분석 (관찰 이유 / 점수 / 리스크)
3. 관찰 포인트
4. 면책 고지
```

---

## 8.3 티스토리 (markdown)

```
---
title: {제목}
category: AI 자동매매
tags: [AI주식, 자동매매, 한국주식]
---

# {제목}

## 오늘의 시장 상태

## AI 관찰 종목 분석

## 관찰 포인트

> 면책 고지
```

---

# 9. 템플릿 구조

위치: `templates/blog/`

```
templates/blog/
  type_a_naver.txt.j2
  type_a_tistory.md.j2
  type_b_naver.txt.j2
  type_b_tistory.md.j2
  type_c_naver.txt.j2
  type_c_tistory.md.j2
```

- Jinja2 문법 사용
- 템플릿은 데이터를 받아 본문 문자열을 렌더링
- 면책 고지·금지 표현 규칙(§10)을 모든 템플릿에 내장

---

# 10. 콘텐츠 생성 규칙

## 반드시 포함

- 왜 이 종목이 선정/주목되었는가
- 현재 시장 상태 (regime)
- 점수 기준 설명
- 리스크 요인
- 관찰 포인트

---

## 면책 고지 — 모든 콘텐츠에 필수 포함

```
본 글은 투자 참고용 정보이며 투자 판단과 책임은 투자자 본인에게 있습니다.
```

---

## 금지 표현

- 무조건 상승
- 확정 수익
- 투자 보장
- 수익률 보장
- 매수 추천
- 지금 사세요

---

## 운영 상태 기반 표현 규칙

- 생성 전 `asof_date`, `gate_overall_status`, `walkforward_acceptance_status`를 검증
- `gate_overall_status=BUY_ALLOWED` + `walkforward_acceptance_status=ACCEPTED`인 경우에만 제한적으로 "추천 후보" 표현 허용
- `WATCH` / `BLOCK` / `REJECTED` 상태에서는 "관찰 종목", "검토 후보", "보수적 접근" 표현 사용
- 모델 예상 수익률은 보장 수익률처럼 표현하지 않으며 반드시 예상치임을 명시
- 휴장일 또는 최신 산출물 미생성 상태에서는 신규 콘텐츠를 생성하지 않는다. 별도 휴장일 캘린더 없이 freshness 검증을 재사용한다(예: `daily_recommendations.json`의 `asof_date`가 대상 영업일과 일치하고 `source_status=current`). 산출물이 직전 영업일자에 머물러 있으면 생성을 건너뛰고 로그에 사유를 기록한다.

---

## 10.5 저품질(유사문서) 회피 전략

매일 비슷한 형식·문구의 글을 올리면 네이버 등에서 유사문서(저품질)로 분류될 위험이 있다. 이를 시스템(자동)과 운영(수동) 양쪽에서 완화한다.

### 시스템 측 (변형 엔진 — 구현 완료)

`python/blog_variation.py` + `templates/blog/variations.json`로 다음을 자동 변형한다.

- **시드 기반 결정적 변형**: 시드 = `날짜 + 유형 + 플랫폼(+종목코드)`. 같은 날 같은 입력이면 동일 결과(멱등·재현), 날짜가 바뀌면 제목·도입·연결어·종목 수식어가 자동으로 달라진다.
- **제목/도입/연결어/마무리 문구 풀**: 슬롯별 다중 후보에서 시드로 선택.
- **플랫폼 교차 중복 방지**: 네이버/티스토리는 시드 살트가 달라 같은 날이라도 제목·문구·구조가 서로 다르게 생성된다. (네이버=대화체 plain text, 티스토리=표·헤더 포함 markdown)
- **데이터 기반 본문 차별화**: 종목·점수·국면·사유가 매일 바뀌므로, 이를 자연어로 풀어내면 본문 텍스트 자체가 매일 달라진다.
- **면책 고지·금지 표현은 변형 제외**: 법적 일관성 유지.

### 운영 측 (운영자 수동 — 권장)

코드로 막을 수 없는 부분은 게시 운영에서 보완한다.

- **게시 시각 분산**: 매일 같은 시각에 기계적으로 올리지 않는다.
- **이미지 차별화**: 매번 동일 이미지/캡처 재사용을 피한다. (템플릿에 이미지 삽입 위치를 안내)
- **플랫폼별 분리 게시 권장**: 동일 글을 네이버·티스토리에 그대로 동시 게시하지 않는다. 변형 엔진이 플랫폼별로 다른 결과를 주지만, 가능하면 한쪽은 직접 한두 문장 가필한다.
- **주기적 수동 가필**: 주 1~2회는 도입/마무리에 운영자 코멘트를 직접 추가해 자동 생성 패턴을 흐린다.
- **게시 이력 관리**: 어떤 글을 어디에 올렸는지 별도 기록(스프레드시트 등)하여 중복 게시를 방지한다.

### 향후 확장

- 문구 풀 확대 및 구조 스킨(섹션 순서) 다중화
- §15.5 LLM 엔진 도입 시 문장 자연스러움·다양성 대폭 향상

---

# 11. 생성 스크립트

## python/generate_blog_posts.py

실행 예시

```bash
# 최신 영업일 기준 전체 유형 생성
python python/generate_blog_posts.py

# 특정 날짜 / 특정 유형
python python/generate_blog_posts.py --date 2026-06-02 --type A
```

동작

1. 대상 날짜 결정 (인자 없으면 최신 완료 영업일)
2. freshness 검증 (§10) — 실패 시 생성 중단, 로그 기록
3. 데이터 로드 (JSON 파일 + db.py로 DB 조회)
4. 유형별 Jinja2 템플릿 렌더링 (네이버 / 티스토리)
5. `outputs/blog_drafts/`에 파일 출력 (기존 파일 있으면 건너뜀)
6. 생성 결과 요약을 stdout 및 로그에 출력

DB 접속: 기존 `python/db.py`의 `DATABASE_URL` 기반 커넥션 재사용

---

# 12. 운영 절차 (수동 업로드 동선)

1. 종가 배치 완료 후 `generate_blog_posts.py` 실행 (수동 또는 19:30 스케줄)
2. `outputs/blog_drafts/` 폴더에서 당일 파일 확인
3. 네이버용 `.txt` 또는 티스토리용 `.md` 파일 열기
4. 내용 검토 (이상 표현·데이터 오류 확인)
5. 복사하여 네이버 블로그 / 티스토리에 붙여넣기 후 게시

게시 이력은 운영자가 별도로 관리(예: 스프레드시트). MVP에서는 시스템이 게시 상태를 추적하지 않는다.

---

# 13. 비기능 요구사항

전체 생성 시간: 30초 이하 (3유형 × 2플랫폼)

생성 실패 시: `logs/` 디렉토리에 에러 로그 저장, 수동 재실행 가능

재실행 시 기존 출력 파일 덮어쓰지 않음 (중복 방지)

외부 의존 최소화: DB 조회 실패 시 가능한 유형(TYPE_A 등 JSON 기반)만 생성하고 사유 기록

---

# 14. 성공 기준

매일 TYPE_A / TYPE_B / TYPE_C 파일 출력 성공

운영자가 복사하여 즉시 게시 가능한 품질

생성 글에 면책 고지 포함, 금지 표현 미포함

6개월 내 180건 이상 콘텐츠 누적 게시

1년 내 365건 이상 콘텐츠 누적 게시

검색 유입 및 LeeTrader 브랜드 노출 증가

---

# 15. 향후 확장 (MVP 이후 검토)

아래는 글 누적량이 늘고 검토·이력 관리 필요성이 생길 때 도입을 검토하는 항목이다. MVP 범위에 포함하지 않는다.

## 15.1 콘텐츠 보관 DB (content_drafts)

파일 대신 PostgreSQL에 초안을 저장하여 상태·이력 추적

```sql
CREATE TABLE public.content_drafts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_type TEXT NOT NULL
        CHECK (content_type IN ('TYPE_A','TYPE_B','TYPE_C','TYPE_D','TYPE_E','TYPE_F')),
    title TEXT NOT NULL,
    summary TEXT,
    content_naver TEXT NOT NULL,
    content_tistory TEXT NOT NULL,
    source_date DATE NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'DAILY_CLOSE',
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    template_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'DRAFT'
        CHECK (status IN ('DRAFT','REVIEWED','PUBLISHED','ARCHIVED')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE,
    UNIQUE (content_type, source_date, template_version)
);
```

## 15.2 콘텐츠 API (Render web)

- `GET /api/content` (목록, 필터: type/status/date)
- `GET /api/content/{id}` (상세)
- `POST /api/content/generate` (생성)
- `PUT /api/content/{id}/status` (상태 변경)
- 인증: 기존 `node/operatorAccess.js`의 `apiGuard` 재사용

## 15.3 운영자 UI

콘텐츠 목록/상세 화면, 네이버·티스토리 복사 버튼, 상태 변경

## 15.4 매매 성과 콘텐츠 (TYPE_D/E/F)

자동매매 운영·주간·월간 성과 리포트. `research.live_order_fill` / `public.trades`에 매매 데이터가 축적된 후 활성화

## 15.5 LLM 엔진 (V2)

GPT / Gemini / Claude로 문장 자연스러움 및 SEO 품질 향상. 비용·결과 편차 고려하여 검토

---

# 16. 의존성

- Python: `Jinja2` — requirements에 추가
- DB 접속: 기존 `python/db.py` (`DATABASE_URL` 기반) 재사용
- 스케줄(선택): 기존 `scripts/register_daily_operations_task.ps1` 패턴 재사용
- 출력 폴더: `outputs/blog_drafts/` 신규 생성
