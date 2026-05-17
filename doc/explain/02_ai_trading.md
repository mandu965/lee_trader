# AI 기반 자동매매 모듈

*작성 기준일: 2026-05-17*

---

## 개요

랭킹 파이프라인이 생성한 `ranking_final.csv`를 기반으로,  
매일 오전 09:30에 AI 모델 예측값·신뢰도·리스크 게이트를 종합하여 실계좌 주문을 자동으로 제출합니다.

```
run_live_auto_trade_cycle.py
        │
        ├─ run_operational_refresh   (최신 평가·게이트 생성)
        │       ├─ build_operational_buy_gate   → BUY 게이트 판단
        │       ├─ buy_candidate_builder        → 매수 후보 선정
        │       └─ apply_execution_policy       → 실행 정책 적용
        │
        ├─ submit_live_orders        (KIS API 실주문 제출)
        │
        ├─ sync_live_account_holdings
        ├─ update_ai_position_state
        ├─ sync_live_order_fills
        └─ build_live_trade_review
```

---

## 1. 운영 갱신 (Operational Refresh)

**파일**: `python/run_operational_refresh.py`

매매 사이클 전에 최신 평가 데이터를 생성합니다.

**실행 단계**:

| 순서 | 스크립트 | 역할 |
|---|---|---|
| 1 | `run_theme_shadow_daily.py` | 테마 오버레이 계산 |
| 2 | `buy_candidate_builder.py` | 매수 후보 종목 선정 |
| 3 | `build_buy_candidate_comparison.py` | 후보 비교 리포트 |
| 4 | `build_operational_buy_gate.py` | BUY 게이트 상태 판단 |
| 5 | `run_paper_trading_ledger.py` | Paper Trading 장부 갱신 |
| 6 | `sync_paper_trading_db.py` | DB 동기화 |
| 7 | `export_serving_payloads.py` | 웹 API용 데이터 내보내기 |
| 8 | (선택) `sync_live_account_holdings.py` | 실계좌 잔고 동기화 |
| 9 | (선택) `build_live_order_preview.py` | 주문 프리뷰 생성 |

각 단계는 타임아웃(필수 20분 / 선택 10분)이 설정되어 있어 한 단계 실패가 전체를 중단시키지 않습니다.

---

## 2. BUY 게이트

**파일**: `python/build_operational_buy_gate.py`  
**출력**: `outputs/operational_buy_gate.json`

매수를 허용할지 판단하는 다단계 안전 장치입니다.

### 2-1. 게이트 상태

| 상태 | 코드 | 의미 |
|---|---|---|
| `BUY_ALLOWED` | 0 | 즉시 매수 허용 |
| `PILOT` | 1 | 파일럿 모드 (제한된 수량) |
| `WATCH` | 2 | 관찰 중 (매수 보류) |
| `HOLD` | 3 | 홀드 |
| `BLOCK` | 4 | 완전 차단 |

**현재 운영 상태 (2026-05-17)**: PILOT  
→ Walkforward 60일치 실거래 데이터 축적 중 (2026-05-28 이후 BUY_ALLOWED 전환 예정).

### 2-2. 전환 기준 (신뢰도 기반)

| 조건 | 임계값 |
|---|---|
| BUY_ALLOWED 진입 | confidence_score ≥ 82.0 |
| PILOT 진입 | confidence_score ≥ 80.0 |
| 완전 차단 | confidence_score < 55.0 |
| 수량 축소 시작 | confidence_score < 70.0 |

### 2-3. 추가 리스크 필터

| 필터 | 기준 | 의미 |
|---|---|---|
| `max_liquidity_risk_ratio` | 0.30 | 유동성 위험 종목 비율 30% 초과 시 차단 |
| `max_sector_top_share` | 0.40 | 특정 섹터 집중도 40% 초과 시 차단 |
| `max_overheat_ratio` | 0.20 | 과열 종목 비율 20% 초과 시 차단 |

### 2-4. 재무 모멘텀 게이트 (`FINANCIAL_BUY_GATE_ENABLED=1`)

`fin_momentum_phase`에 따라 수량을 자동으로 축소합니다.

| 단계 | 조치 |
|---|---|
| SLOWING | 수량 70%로 축소 |
| WEAKENING | 수량 50%로 축소 |
| DECLINING | 수량 30%로 축소 |
| hard_risk 감지 | 완전 차단 |

---

## 3. 실행 정책 적용

**파일**: `python/apply_execution_policy.py`

신뢰도 밴드에 따라 포지션 크기와 진입 여부를 결정합니다.

### 3-1. 신뢰도 밴드

| 밴드 | 범위 | 가중치 | 포지션 크기 |
|---|---|---|---|
| BLOCK | < 55% | 진입 불가 | 강제 청산 |
| REDUCED | 55~70% | 45% | 50% |
| STANDARD | 70~85% | 100% | 100% |
| EXPANDED | ≥ 85% | 115% | 115% |

### 3-2. 진입 기준

| 항목 | 기본값 |
|---|---|
| 기본 검토 대상 | Top 8 종목 (`entry_review_top_n`) |
| 확장 검토 대상 | Top 10 종목 (`entry_review_extended_top_n`) |
| 표준 진입 신뢰도 임계값 | 76% |
| 확장 진입 신뢰도 임계값 | 70% |
| 최대 보유 종목 수 | 5개 |
| 최대 단일 종목 비중 | 24% |
| 섹터 집중도 한도 | 35% |
| 테마 집중도 한도 | 35% |
| 최소 현금 비율 | 5% |

---

## 4. 주문 제출

**파일**: `python/submit_live_orders.py`

### 4-1. 입력 데이터

| 파일 | 내용 |
|---|---|
| `outputs/trade_intents.json` | 매매 의도 (종목, 방향, 수량) |
| `data/live_account_holdings.csv` | 현재 계좌 보유 현황 |
| `data/ranking_final.csv` | 최신 랭킹 점수 |

### 4-2. 실행 모드

```bash
# 프리뷰만 (실주문 없음)
python submit_live_orders.py

# 실제 주문 제출
python submit_live_orders.py --execute --allow-buy --confirm-text "LIVE_ORDER"
```

### 4-3. 출력

| 파일 | 내용 |
|---|---|
| `outputs/order_requests_preview.json` | 주문 검토용 프리뷰 |
| `outputs/order_requests_execution.json` | 실제 제출된 주문 목록 |
| `outputs/order_buy_approvals.json` | 매수 승인 내역 |
| `outputs/live_auto_trade_run_log.jsonl` | 실행 로그 (append) |

---

## 5. 자동매매 사이클 전체 흐름

**파일**: `python/run_live_auto_trade_cycle.py`  
**실행 시각**: 09:30, 10:00 (1일 1회 성공 정책 — 첫 번째 성공 후 두 번째는 스킵)

```
1.  run_operational_refresh (--with-live-account)
2.  submit_live_orders (--execute)          ← 실주문 제출
3.  sync_live_account_holdings              ← KIS 잔고 동기화
4.  update_ai_position_state                ← AI 포지션 상태 갱신
5.  sync_live_order_fills                   ← 체결 내역 동기화
6.  build_live_trade_consistency_report     ← 일관성 검증
7.  build_live_trade_review                 ← 거래 리뷰 생성
8.  build_live_trade_review_summary         ← 요약 리포트
9.  (선택) build_live_kpi_daily_report
10. (선택) build_live_closed_trade_report
11. (선택) sync_web_display_data            ← 웹 대시보드 동기화
```

---

## 6. Walkforward 검증

**파일**: `python/run_walkforward_backtest.py`

실거래 수익률과 모델 예측 수익률을 비교하여 모델 유효성을 자동으로 검증합니다.

| 지표 | 수집에 필요한 실거래 기간 |
|---|---|
| 벤치마크 비교 | 60일 이상 |
| 순방향 수익률 평가 | 60~90일 |
| Confidence calibration | 90일 이상 |

**운영 시작**: 2026-03-29  
**60일 만기**: 2026-05-28 → 이후 PILOT → BUY_ALLOWED 자동 전환 예정  
**결과 파일**: `outputs/walkforward_acceptance.json`

---

## 7. 핵심 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `AUTO_TRADE_EXECUTE` | `0` | `1` 설정 시 실주문 제출 활성화 |
| `AUTO_TRADE_ALLOW_BUY` | `0` | `1` 설정 시 매수 허용 |
| `AUTO_TRADE_BUY_APPROVAL_REQUIRED` | `0` | `0` = 완전 자동 승인 |
| `AUTO_TRADE_CONFIRM_TEXT` | — | `"LIVE_ORDER"` 입력 필수 |
| `AUTO_TRADE_FORCE_RESUBMIT` | `0` | 이전 성공 무시하고 재제출 |
| `FINANCIAL_BUY_GATE_ENABLED` | `0` | 재무 모멘텀 게이트 활성화 |
| `SCORE_FORMULA_VERSION` | `v9_flow` | 랭킹 점수 공식 버전 |
| `HORIZON_DAYS` | `60` | 예측 기간 (일) |
| `TOP_N` | `20` | 상위 추천 종목 수 |
