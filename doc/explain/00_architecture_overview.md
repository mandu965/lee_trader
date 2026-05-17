# 시스템 전체 아키텍처 개요

*작성 기준일: 2026-05-17*

---

## 1. 시스템 구성 한눈에 보기

Lee Trader는 **국내 주식 자동매매**와 **미국 주식 연구·Paper Trading** 두 축으로 구성된 퀀트 트레이딩 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Lee Trader v1                            │
│                                                                 │
│  ┌────────────────────────┐   ┌──────────────────────────────┐  │
│  │    국내 주식 (KR)       │   │      미국 주식 (US)            │  │
│  │                        │   │                              │  │
│  │  랭킹 파이프라인         │   │  US 매크로 오버레이 +         │  │
│  │  AI 자동매매            │   │  US 주식 랭킹 + Paper         │  │
│  │  Rule 자동매매          │   │  Trading (Phase 8)           │  │
│  └────────────┬───────────┘   └──────────────┬───────────────┘  │
│               │                              │                  │
│               ▼                              ▼                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            PostgreSQL (공유 데이터베이스)                    │  │
│  │  fact_price_daily · features · flow_daily · fundamentals  │  │
│  │  stocks · us_etf_daily_price · us_macro_feature_daily     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                  │
│               ┌──────────────┘                                  │
│               ▼                                                 │
│  ┌────────────────────────┐                                     │
│  │    Node.js Web API     │  ← 포트폴리오 현황 대시보드          │
│  │    (port 3400)         │                                     │
│  └────────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 컨테이너 구성 (docker-compose.yml)

시스템 전체는 Docker Compose로 구동됩니다. 각 컨테이너의 역할은 아래와 같습니다.

### 2-1. 데이터베이스

| 컨테이너 | 이미지 | 포트 | 역할 |
|---|---|---|---|
| `postgres` | postgres:15 | 15432 | 전체 시스템 공유 데이터베이스 |

- 헬스 체크: `pg_isready` — 정상 응답 시에만 의존 서비스 기동
- 볼륨: 호스트 마운트로 데이터 영속 보장

---

### 2-2. 국내 주식 파이프라인 스케줄러

| 컨테이너 | 실행 스크립트 | 기본 시각 | 역할 |
|---|---|---|---|
| `scheduler` | `run_pipeline.py` | **18:10** | 종가 배치: 데이터 수집 → 피처 → 모델 → 랭킹 전체 파이프라인 |
| `scheduler-recovery` | `run_pipeline.py` | **12:00** | 당일 주 스케줄러 미실행 시 복구 배치 |
| `scheduler-auto-buy` | `run_live_auto_trade_cycle.py` | **09:30, 10:00** | AI 자동매매 사이클 (1일 1회 성공 정책) |
| `scheduler-live-account-sync` | `sync_live_account_holdings.py` | **10:00, 14:00, 18:00** | KIS 계좌 잔고·체결 동기화 |

---

### 2-3. Rule 자동매매 스케줄러

| 컨테이너 | 실행 스크립트 | 기본 시각 | 역할 |
|---|---|---|---|
| `scheduler-rule-before-open` | `run_rule_before_open_cycle.py` | **08:55** | 장 전: Rule 신호 확인 + 주문 프리뷰 생성 |
| `scheduler-rule-after-open` | `run_rule_after_open_cycle.py` | **09:10** | 장 시작 직후: 미체결 주문 재시도 |
| `scheduler-rule-after-close` | `run_rule_after_close_cycle.py` | **18:00** | 장 마감 후: 신호 재계산 + 다음 날 주문 준비 |

---

### 2-4. 미국 주식 스케줄러

| 컨테이너 | 실행 스크립트 | 기본 시각 | 역할 |
|---|---|---|---|
| `scheduler-us-macro` | `run_us_macro_overlay_scheduler.py` | **07:30** | US ETF·지수 일별 수집 + 매크로 피처 계산 |
| `scheduler-us-macro-shadow` | Shadow mode | **08:50** | 매크로 오버레이 Shadow 비교 실행 |
| `scheduler-us-pipeline` | US 파이프라인 | **06:30** | US 주식 유니버스 → 랭킹 → Paper Trading 검토 |

---

### 2-5. 웹 서버

| 컨테이너 | 이미지 | 포트 | 역할 |
|---|---|---|---|
| `node-api` | Node.js 20 | 3400 | 포트폴리오 현황·랭킹 조회 REST API |

---

## 3. 전체 데이터 흐름

```
[외부 데이터 소스]
  네이버 금융 ─────→ KOSPI 지수, 종목 유니버스
  KIS API ──────────→ 일별 가격, 외국인·기관 순매수, 계좌 잔고·주문
  DART API ─────────→ 연간 재무제표 (매출, 영업이익, ROE)
  yfinance ─────────→ US ETF / 지수 OHLCV

          ↓
[일별 배치 파이프라인 — 18:10]
  1. fetch_market_data      → market_status.csv  (시장 레짐: bull/neutral/defensive)
  2. fetch_top_universe     → universe.csv        (상위 N개 종목 메타)
  3. download_prices_kis    → fact_price_daily    (일별 OHLCV)
  4. download_flows_kis     → flow_daily          (외국인·기관 순매수)
  5. fetch_fundamentals_dart→ fundamentals.csv    (재무지표)
  6. quality_builder        → quality.csv         (재무 품질 점수)
  7. feature_builder        → features.csv        (80개+ 피처)
  8. label_builder          → labels.csv          (지도학습 정답)
  9. model_train            → model.pkl           (LightGBM 재학습)
 10. model_predict          → predictions.csv     (60d/90d 수익률·MDD·확률 예측)
 11. ranking_builder        → ranking_final.csv   (최종 순위 + 점수)

          ↓
[자동매매 사이클 — 09:30]
  AI 경로: operational_refresh → build_trade_intents → submit_live_orders → KIS
  Rule 경로: rule_signal_builder → rule_portfolio_manager → rule_order_submitter → KIS

          ↓
[동기화 및 리포트 — 10:00, 14:00, 18:00]
  sync_live_account_holdings → live_account_holdings.csv
  sync_live_order_fills      → 체결 내역 DB 저장
  build_live_trade_review    → 거래 리뷰 리포트
```

---

## 4. 주요 파일·데이터 저장 위치

| 경로 | 내용 |
|---|---|
| `data/features.csv` | 일별 종목별 피처 (80개+ 컬럼) |
| `data/model.pkl` | 학습된 LightGBM 모델 패키지 |
| `data/predictions.csv` | 모델 예측값 |
| `data/ranking_final.csv` | 최종 순위 및 점수 |
| `data/rule_signals.csv` | Rule 신호 |
| `data/live_account_holdings.csv` | KIS 실계좌 잔고 |
| `outputs/operational_buy_gate.json` | AI BUY 게이트 상태 |
| `outputs/rule_portfolio_plan.json` | Rule 포트폴리오 계획 |
| `outputs/rule_trade_intents.json` | Rule 매매 의도 |
| `outputs/order_requests_preview.json` | 주문 검토용 프리뷰 |
| `config/production_v1.yaml` | 운영 정책 설정 |
| `.env` | API 키, 실행 모드 플래그 |

---

## 5. 핵심 환경변수 (안전 플래그)

```bash
AUTO_TRADE_EXECUTE=0           # 1로 설정해야 실주문 제출 (기본 비활성)
AUTO_TRADE_ALLOW_BUY=0         # 1로 설정해야 매수 허용
AUTO_TRADE_CONFIRM_TEXT=LIVE_ORDER  # 실행 승인 확인 문구
RULE_TRADING_RUN_MODE=paper    # paper | pilot | live
```

> **절대 원칙**: `AUTO_TRADE_EXECUTE=0` 상태에서 모든 코드 변경 및 검증 진행.
> 실주문 관련 변경은 paper trading 3일 이상 검증 후 전환.
