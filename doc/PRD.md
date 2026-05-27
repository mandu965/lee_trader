# Lee Trader — 시스템 PRD (Product Requirements Document)

> **기준일: 2026-05-27**
> 이전 세션 문서들과 달리 현재 운영 실태를 기준으로 재작성된 단일 진입점 문서입니다.

---

## 1. 서비스 현황 요약

| 서비스 | 상태 | 비고 |
|---|---|---|
| **KR AI 자동매매** | LIVE | 주력 서비스. 204종목, 09:30 실행 |
| **수동매매 서비스** | 운영 중 | RULE 종료 후 AI 추천 기반 수동 의사결정 |
| **Web 대시보드** | 운영 중 | Node.js API (port 3400) |
| ~~RULE 자동매매~~ | **종료** (2026-05-21) | `RULE_LIVE_ENABLED=0`, KR AI 주력 집중 결정 |
| ~~US 주식 서비스~~ | **미운영** | 코드는 존재, 스케줄러 비활성화 상태 |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     Lee Trader v1 (현재 운영)                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  국내 주식 (KR)                            │   │
│  │                                                          │   │
│  │  [18:10 종가배치]                                         │   │
│  │  fetch_market_data → fetch_top_universe → prices/flows   │   │
│  │  → quality_builder → feature_builder → model_train       │   │
│  │  → ranking_builder → ranking_final.csv + DB              │   │
│  │                                                          │   │
│  │  [09:30 AI 자동매매]                                      │   │
│  │  run_live_auto_trade_cycle → submit_live_orders → KIS    │   │
│  │                                                          │   │
│  │  [수동매매 서비스]                                         │   │
│  │  AI 추천 + manual-trading.html + KIS 계좌 동기화         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL (공유 DB)                                     │   │
│  │  fact_price_daily · features · flow_daily                │   │
│  │  fundamentals · stocks · daily_ranking                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Node.js Web API (port 3400)                             │   │
│  │  /api/ranking · /api/top20 · /api/live-account/*         │   │
│  │  /api/flow-history · /api/daily-ranking                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. KR AI 자동매매 파이프라인

### 3-1. 일별 배치 (18:10)

```
Step 1  fetch_market_data       → market_status.csv (레짐: bull/neutral/defensive)
Step 2  fetch_top_universe      → universe.csv (204종목 KOSPI/KOSDAQ)
Step 3  download_prices_kis     → fact_price_daily (일별 OHLCV)
Step 4  download_flows_kis      → flow_daily (외국인·기관 순매수)
Step 5  fetch_fundamentals_dart → fundamentals.csv (DART 연간 재무)
Step 6  quality_builder         → quality.csv (재무 품질 점수)
Step 7  feature_builder         → features.csv (43개+ 피처)
Step 8  label_builder           → labels.csv (지도학습 정답)
Step 9  model_train             → model.pkl (LightGBM 재학습)
Step 10 model_predict           → predictions.csv (60d/90d 수익률·MDD·확률)
Step 11 ranking_builder         → ranking_final.csv + daily_ranking DB 누적
```

수동 실행:
```powershell
.venv\Scripts\python.exe python/run_manual_close_batch.py --local
```

### 3-2. 랭킹 점수 공식 (v9_flow)

**6개 컴포넌트 가중 합산:**

| 컴포넌트 | Bull | Neutral | Defensive | 설명 |
|---|---:|---:|---:|---|
| `ret_score` | 0.33 | 0.28 | 0.23 | 예측 수익률 점수 |
| `prob_score` | 0.24 | 0.23 | 0.19 | Top20 진입 확률 |
| `tech_score` | 0.23 | 0.21 | 0.16 | 기술적 점수 |
| `qual_score` | 0.08 | 0.18 | 0.34 | 재무 품질 점수 |
| `flow_score` | 0.12 | 0.10 | 0.08 | 수급 점수 |
| `risk_penalty` | ×0.40 | ×0.65 | ×0.80 | 리스크 차감 (계수) |

**flow_score:**
```
flow_score = 0.6 × percentile(flow_foreign_net_5d)
           + 0.4 × percentile(flow_inst_net_5d)
(동일 날짜 내 상대 순위, 데이터 없으면 50.0)
```

### 3-3. AI 자동매매 사이클 (09:30)

```
run_live_auto_trade_cycle.py
  ├─ run_operational_refresh   (최신 평가 생성)
  │     ├─ buy_candidate_builder     → 매수 후보 선정
  │     ├─ build_operational_buy_gate → BUY 게이트 판단
  │     └─ apply_execution_policy   → 실행 정책 적용
  ├─ submit_live_orders         (KIS API 실주문)
  ├─ sync_live_account_holdings
  ├─ sync_live_order_fills
  └─ build_live_trade_review
```

### 3-4. BUY 게이트 현황

| 상태 | 의미 |
|---|---|
| `BUY_ALLOWED` | 즉시 매수 허용 |
| **`PILOT`** ← 현재 | 파일럿 모드 (제한 수량) |
| `WATCH` | 관찰 중 |
| `BLOCK` | 완전 차단 |

**현재 상태 (2026-05-27)**: PILOT
→ 운영 시작 2026-03-29, 60일 만기 2026-05-28부터 BUY_ALLOWED 자동 전환 예상

Walkforward REJECTED · Gate PILOT은 **데이터 부족**이 원인이며 모델 고장이 아닙니다.

### 3-5. 진입 정책

| 항목 | 값 |
|---|---|
| 검토 대상 | Top 8 종목 |
| 최대 보유 종목 | 5개 |
| 최대 단일 비중 | 24% |
| 섹터·테마 집중도 한도 | 35% |
| 최소 현금 비율 | 5% |
| 최대 보유 기간 | 30일 (하드캡 45일) |

---

## 4. 수동매매 서비스

RULE 자동매매 종료(2026-05-21) 이후 AI 추천 기반 수동 의사결정 서비스로 전환했습니다.

### 4-1. 운영 방식

1. 아침 `run_operational_refresh.py` 실행 → 최신 랭킹·게이트 갱신
2. `manual-trading.html`에서 후보 종목 확인
   - Gate 상태, Walkforward 상태
   - PROMOTION_READY · WATCHLIST 후보 목록
   - 종목별 추천 근거 (top_driver, risk_factor, action_note)
3. HTS/MTS에서 차트·호가 직접 확인 후 수동 주문
4. `real_trade_log.csv`에 체결 기록

### 4-2. 수동매매 화면 (`manual-trading.html`)

| 탭 | 내용 |
|---|---|
| 추천 종목 | AI 점수 상위 + 매수 적합도 판단 |
| 보유 현황 | KIS 계좌(44\*\*\*\*02) 실시간 잔고 동기화 |
| 체결 이력 | 최근 거래 내역 조회 |

### 4-3. 수동 주문 명령

```powershell
# 주문 미리보기
python python/build_live_order_preview.py

# 수동 단건 주문 (실제 실행 시 --execute 추가)
python python/kis_manual_order.py --side buy --code 005930 --qty 1 --price 0 --ord-dvsn 01 --execute --confirm-text LIVE_ORDER
```

---

## 5. Web 대시보드

### 5-1. 주요 화면

| 화면 | 파일 | 설명 |
|---|---|---|
| 메인 랭킹 | `index.html` | Top 20 종목 카드 + 점수 설명 |
| 종목 상세 | `detail.html` | 개별 종목 분석 · 차트 |
| 수동매매 | `manual-trading.html` | AI 추천 + KIS 계좌 연동 |
| 보유종목 | `holdings.html` | 실계좌 잔고 |
| 모의투자 | `paper-trading.html` | Paper trading 현황 |
| 알림 | `alerts.html` | Score KPI 알림 내역 |

### 5-2. 주요 API

| 엔드포인트 | 설명 |
|---|---|
| `GET /api/ranking` | 전체 종목 랭킹 |
| `GET /api/top20` | 상위 20종목 |
| `GET /api/daily-ranking` | 누적 일별 랭킹 히스토리 |
| `GET /api/live-account/holdings` | 실계좌 잔고 |
| `GET /api/live-account/order-preview` | 주문 프리뷰 |
| `GET /api/flow-history` | 종목별 수급 히스토리 |

### 5-3. 메인 화면 상단 카드

| 카드 | ID | 설명 |
|---|---|---|
| 수급 긍정 종목 | `heroFlowPositiveCount` | 외국인·기관 동시 순매수 방향 종목 수 |
| 점수 우수 후보 | `heroHighScoreCount` | 최종 점수 70점 이상 종목 수 |
| 리스크 주의 | `heroRiskCount` | 리스크 페널티 높은 종목 수 |

---

## 6. 피처 체계

### 6-1. 기술적 피처

| 피처 그룹 | 주요 피처 |
|---|---|
| 수익률 | `ret_1d`, `ret_5d`, `ret_10d`, `ret_60d`, `ret_120d` |
| 모멘텀·이동평균 | `mom_20`, `ma_5`, `ma_20`, `ma_60`, `close_over_ma20` |
| 변동성·RSI | `vol_20`, `vol_60`, `rsi_14` |
| 거래량·거래대금 | `volume_ratio_5d/20d`, `value_ratio_5d/20d`, `liquidity_score` |
| 중장기 | `high_52w_ratio` (52주 신고가 비율) |

### 6-2. 수급 피처

| 피처 | 설명 |
|---|---|
| `flow_foreign_net_5d` | 외국인 순매수 5영업일 누적 |
| `flow_foreign_net_20d` | 외국인 순매수 20영업일 누적 |
| `flow_inst_net_5d` | 기관 순매수 5영업일 누적 |
| `flow_inst_net_20d` | 기관 순매수 20영업일 누적 |

수집 경로: KIS API (`FHKST01010200`) → `flow_daily` 테이블
백필: 138,844행 | 216종목 | 330 영업일 (2025-01-02~2026-05-13)

### 6-3. 재무·품질 피처

| 피처 | 출처 | 설명 |
|---|---|---|
| `quality_score` | DART 연간 재무 | ROE·영업이익률·순이익률·부채비율·OCF 종합 (0~100) |
| `roe_yoy` | quality_builder | ROE YoY 변화 |
| `fin_momentum_score` | DART 분기 재무 | 분기 재무 모멘텀 점수 |
| `fin_turnaround_score` | DART 분기 재무 | 실적 턴어라운드 점수 |
| `short_ratio` | pykrx | 공매도 잔고 비율 |
| `short_ratio_20d_avg` | pykrx | 20일 평균 공매도 비율 |
| `sector_rel_momentum_20d` | 파생 | 섹터 대비 20일 상대 수익률 |

### 6-4. 피처 우선순위 (모델 중요도 기준)

| 순위 | 피처 | 중요도 |
|---|---|---|
| 2 | `roe_yoy` | — |
| 8 | `fin_momentum_score` | 46% |
| 9 | `short_ratio_20d_avg` | 43% |
| 10 | `fin_turnaround_score` | 41% |
| 15 | `short_ratio` | 28% |

---

## 7. 환경변수 핵심 체계

### 7-1. 절대 안전 플래그

```bash
AUTO_TRADE_EXECUTE=1           # 실주문 제출 활성화 (현재 LIVE)
AUTO_TRADE_ALLOW_BUY=1         # 매수 허용 (현재 LIVE)
AUTO_TRADE_BUY_APPROVAL_REQUIRED=0  # 완전 자동 승인 (수동 승인 불필요)
RULE_LIVE_ENABLED=0            # RULE 자동매매 비활성화 (종료)
```

> 코드 변경·검증 시: `AUTO_TRADE_EXECUTE=0`으로 임시 변경 후 작업.
> `ranking_builder.py` 수정 시 반드시 shadow 비교 먼저 실행.

### 7-2. 주요 운영 플래그

| 변수 | 현재값 | 설명 |
|---|---|---|
| `SCORE_FORMULA_VERSION` | `v9_flow` | 랭킹 점수 공식 |
| `HORIZON_DAYS` | `60` | 예측 기간 |
| `TOP_N` | `20` | 상위 추천 종목 수 |
| `AI_MAX_HOLDING_DAYS` | `30` | 최대 보유일 |
| `AI_MAX_HOLDING_DAYS_HARD_CAP` | `45` | 하드캡 |
| `FINANCIAL_FEATURE_ENABLED` | `1` | 재무 모멘텀 shadow 계산 |
| `FINANCIAL_SCORE_OVERLAY_ENABLED` | `0` | shadow → live 실반영 (미활성) |

---

## 8. Docker 컨테이너 구성

| 컨테이너 | 역할 | 실행 시각 |
|---|---|---|
| `postgres` | 공유 데이터베이스 | 상시 |
| `node-api` | Web API (port 3400) | 상시 |
| `scheduler` | 종가 배치 (`run_pipeline.py`) | 18:10 |
| `scheduler-recovery` | 장중 refresh | 12:00 |
| `scheduler-auto-buy` | AI 자동매매 사이클 | 09:30, 10:00 |
| `scheduler-live-account-sync` | KIS 계좌 동기화 | 10:00, 14:00, 18:00 |

> **비활성화된 컨테이너** (코드 존재, 미실행):
> `scheduler-rule-*`, `scheduler-us-*` — RULE 종료 및 US 미운영으로 중단

---

## 9. 운영 명령 레퍼런스

### 9-1. 종가 배치 수동 실행

```powershell
.venv\Scripts\python.exe python/run_manual_close_batch.py --local
```

### 9-2. Operational Refresh (자동매매 전 갱신)

```powershell
python python/run_operational_refresh.py
python python/run_operational_refresh.py --with-live-account
python python/run_operational_refresh.py --skip-theme-shadow
```

### 9-3. Web DB 동기화

```powershell
python python/sync_web_display_data.py
$env:DATABASE_URL="웹DB연결문자열"
python python/sync_csv_db_parity.py
```

### 9-4. Node API 재기동

```powershell
docker compose up -d --build node-api
docker compose logs -f --tail=500 node-api
```

### 9-5. 핵심 산출물 확인

| 파일 | 내용 |
|---|---|
| `outputs/operational_buy_gate.json` | BUY 게이트 상태 |
| `outputs/walkforward_acceptance.json` | Walkforward 검증 결과 |
| `data/ranking_final.csv` | 최신 랭킹 + 점수 |
| `data/buy_candidates_top5.csv` | 최우선 매수 후보 |
| `serving/daily_recommendations.json` | 웹 API용 추천 데이터 |

---

## 10. 모니터링 — 2026-05-28 전후

60일 만기 시작 시점부터 아래 지표가 채워지기 시작합니다:

- `operational_buy_gate.json` → `benchmark.matured_dates_max` 값 증가
- `walkforward_acceptance.json` → `status`: REJECTED → CONDITIONAL → ACCEPTED
- Gate → PILOT → BUY_ALLOWED 자동 전환

**코드 수정 없이 자동 처리됩니다.**

---

## 11. 점수 설명 체계 (score_explainer)

AI 추천 종목마다 자동 생성되는 설명 컬럼:

| 컬럼 | 예시 |
|---|---|
| `score_explain_summary` | "60일 기대수익 상위권과 재무 품질 양호가 추천 점수를 견인했습니다." |
| `top_driver_1` / `top_driver_2` | "60일 기대수익 상위권" · "재무 품질 양호" |
| `risk_factor_1` | "최근 변동성 확대 주의" |
| `action_note` | "우선 검토 후보" · "관심 종목으로 추적" |
| `score_explain_json` | JSON 전체 구조 (drivers, drags, component_snapshot) |

파일: `python/score_explainer.py`

---

## 12. 잔여 개선 과제

| 과제 | 우선순위 | 상태 |
|---|---|---|
| A-3: revenue/op_income YoY features 반영 | 낮음 | merge 버그, features 0행 |
| B-1: DART 분기 재무 수집 | 중간 | 미착수 |
| B-2: 외국인 지분율 피처 | 중간 | 미착수 |
| C-2: 배당수익률 피처 | 중간 | 미착수 |

> 현재 모델은 43개 피처로 안정 가동 중. 위 과제는 다음 재학습 시 일괄 반영 권장.

---

## 13. 종료·미운영 서비스 이력 (참고용)

### RULE 자동매매 (2026-05-21 종료)

- **종료 사유**: KR AI 주력 집중 결정, 백테스트 동형성 재구축 전까지 보류
- **최종 성과 (2026-05-14 백테스트)**: entry 신호 +188.34% / MDD -19.36%
- **환경변수**: `RULE_LIVE_ENABLED=0` (영구 비활성)
- **코드 위치**: `python/rule_signal_builder.py`, `python/rule_portfolio_manager.py` 등
- **관련 문서**: `doc/modules/Lee_trader_rule/` (이력 보관용)

### US 주식 서비스 (미운영)

- **현재 상태**: 코드 구현 완료, 스케줄러 미실행
- **코드 위치**: `python/us/` 디렉토리
- **재개 시 필요**: `US_PAPER_TRADING_ENABLED=1`, `US_ML_RANKING_ENABLED=1`
- **관련 문서**: `doc/modules/Lee_trader_us/` (설계 문서 보관용)
- **US 매크로 오버레이**: RULE 종료로 사용처 없음 (코드 유지)

---

*Lee Trader PRD v1.0 | 2026-05-27 | KR AI LIVE 운영 기준, RULE 종료·US 미운영 반영*
