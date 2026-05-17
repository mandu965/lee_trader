# Rule 기반 자동매매 모듈

*작성 기준일: 2026-05-17*

---

## 개요

AI 모델 없이 **규칙 기반 신호**로 매매를 실행하는 모듈입니다.  
추세·유동성·수급·안정성·레짐 5개 컴포넌트를 결합한 `rule_score_v3`를 진입 신호로 사용합니다.

2026년 5월 초 Paper → Live 전환 완료.

```
[18:00 장 마감 후]
  run_rule_after_close_cycle.py
        ├─ rule_signal_builder   → rule_signals.csv
        └─ rule_portfolio_manager → rule_portfolio_plan.json

[08:55 장 전]
  run_rule_before_open_cycle.py
        ├─ rule_order_preview_builder → rule_order_preview.json
        └─ rule_order_submitter       → KIS 실주문 제출

[09:10 장 시작 후]
  run_rule_after_open_cycle.py
        └─ rule_order_submitter (재시도)  ← 미체결 주문 처리
```

---

## 1. 신호 생성

**파일**: `python/rule_signal_builder.py`  
**전략명**: `RULE_TREND_LIQUIDITY_V1`

### 1-1. rule_score_v3 계산

5개 컴포넌트를 가중합산합니다.

| 컴포넌트 | 가중치 | 핵심 입력 |
|---|---|---|
| `trend_component` | 0.25 | MA 배열, 이동평균 기울기 |
| `liquidity_component` | 0.20 | 거래대금, volume_ratio |
| `stability_component` | 0.15 | 변동성, RSI 안정성 |
| `regime_component` | 0.15 | KOSPI 레짐, 시장 방향성 |
| `flow_component` | 0.25 | 외국인·기관 순매수 수급 |

**flow_component 계산식**:
```
flow_component = clip(
    0.6 × percentile(flow_foreign_net_5d) / 100
  + 0.4 × percentile(flow_inst_net_5d)   / 100,
  0, 1
)
데이터 미보유 시: 0.5 (neutral)
```

### 1-2. 진입 신호 조건

```python
# 공통 기본 조건 (base_conditions)
trading_value_ma20 > 최소거래대금 AND NOT 거래정지

# entry_signal (일반 진입)
base_conditions AND NOT overheated AND (rule_score >= 70 OR rule_score_v3 >= 65)

# strong_entry_signal (강력 진입)
base_conditions AND NOT overheated AND (rule_score >= 75 AND rule_score_v3 >= 70)
```

### 1-3. 최소 거래대금 기준

| 모드 | 기준값 |
|---|---|
| Paper | 5억 원 → (2026-05-14 개선) 10억 원 |
| Live | 20억 원 → (2026-05-14 개선) 30억 원 |

### 1-4. 출력

- `data/rule_signals.csv` — 종목별 신호 및 점수
- 컬럼: `code, rule_score, rule_score_v3, entry_signal, strong_entry_signal`

---

## 2. 포트폴리오 관리

**파일**: `python/rule_portfolio_manager.py`

신호를 기반으로 진입·청산 계획을 수립합니다.

### 2-1. 포지션 크기 결정

`rule_score_v3` 점수에 비례하여 포지션 크기를 동적으로 배분합니다.

```
기준 비중: RULE_NEW_ENTRY_WEIGHT (= 0.16, 16%)

strong 진입 (rule_score_v3 70~100):
  w_min = 0.16 × 0.85 = 13.6%
  w_max = min(0.16 × 1.25, max_position_weight) = 20%
  → 점수에 비례 보간

entry 진입 (RULE_ALLOW_ENTRY_SIGNAL=1, 현재 활성화):
  고정 비중 = 0.16 × 0.75 = 12%
```

### 2-2. 레짐 필터

| 레짐 | 조치 |
|---|---|
| Defensive | max_positions를 5 → 3으로 축소 (strong 신호만 허용) |
| Defensive | entry_only 신호 진입 차단 |
| Normal | 최대 5개 포지션까지 허용 |

### 2-3. 청산 조건

| 조건 | 기준 |
|---|---|
| 손절 (trailing stop) | 고점 대비 2.5% 하락 |
| 익절 (trailing stop) | 이익 실현 후 5% 되돌림 |
| 보유 기간 초과 | Normal: 20일 / Defensive: 14일 |
| 쿨다운 | 청산 후 `RULE_COOLDOWN_DAYS`(5일)간 재진입 금지 |

> **트레일링 스탑 교정 이력**: 2026-05-14 `0.04/0.03` → `0.025/0.05` 조정.

### 2-4. US 매크로 오버레이 연동

`US_MACRO_ENABLED=1` 시 미국 시장 매크로 상태에 따라 진입을 추가 필터링합니다.

- `US_MACRO_STALE_DAYS_LIMIT`: 데이터 신선도 허용 기간 (기본 3일)
- `fin_momentum_phase` 기반으로 진입 수량 추가 조정 가능

### 2-5. 출력

- `outputs/rule_portfolio_plan.json` — 종목별 진입·청산 계획
- `outputs/rule_trade_intents.json` — 주문 제출용 매매 의도

---

## 3. 주문 생성 및 제출

### 3-1. 주문 프리뷰 빌더

**파일**: `python/rule_order_preview_builder.py`

`rule_trade_intents.json`을 기반으로 실제 제출할 주문 상세를 생성합니다.

| 설정 | 기본값 | 설명 |
|---|---|---|
| `RULE_MIN_ORDER_AMOUNT` | 100,000원 | 최소 주문 금액 |
| `RULE_MAX_ORDER_AMOUNT` | 1,000,000원 | 최대 주문 금액 |
| `RULE_MINIMUM_SHARES` | 1주 | 최소 주문 수량 |
| `RULE_AUTO_ADJUST_MINIMUM_SHARES` | — | 최소 수량 자동 조정 |
| `MARKET_ORDER_ENABLED` | — | 시장가 주문 허용 |

**출력**: `outputs/rule_order_preview.json`

---

### 3-2. 주문 제출자

**파일**: `python/rule_order_submitter.py`

KIS API를 통해 실제 주문을 제출합니다.

**주문 구분**:
- 시장가: `ord_dvsn = "01"`, `ord_unpr = "0"`
- 지정가: `ord_dvsn = "00"`, `ord_unpr = 정규화된_가격`

**오류 처리 정책**:

| 오류 유형 | 처리 방식 |
|---|---|
| `KISAuthError` | 토큰 갱신 후 1회 재시도 |
| HTTP 429 (Rate Limit) | 요청 미처리 → 안전하게 재시도 가능 |
| Timeout / 5xx | **재시도 불가** (중복 주문 방지) |

> **중요**: Timeout/5xx 오류 시 주문이 실제로 제출됐을 수 있어 재시도하면 중복 주문이 발생합니다.  
> 이 경우 KIS 계좌에서 직접 확인이 필요합니다.

**출력**:

| 파일 | 내용 |
|---|---|
| `outputs/rule_execution_results.json` | 종목별 주문 실행 결과 |
| `outputs/rule_execution_reconciliation_report.md` | 미체결·실패 항목 리포트 |
| `outputs/rule_execution_history.jsonl` | 전체 실행 이력 (append) |

---

## 4. 스케줄 및 실행 타이밍

Rule 자동매매는 하루 3번 실행됩니다.

| 시각 | 컨테이너 | 스크립트 | 역할 |
|---|---|---|---|
| **18:00** | `scheduler-rule-after-close` | `run_rule_after_close_cycle.py` | 장 마감 후 신호 계산 + 다음 날 계획 수립 |
| **08:55** | `scheduler-rule-before-open` | `run_rule_before_open_cycle.py` | 장 전 주문 생성 및 제출 |
| **09:10** | `scheduler-rule-after-open` | `run_rule_after_open_cycle.py` | 미체결 주문 재시도 |

> **09:05 타이밍 이력**: 2026-05-12 갭 타이밍 이슈로 08:55 변경 완료.

---

## 5. 백테스트 결과 요약 (2026-05-14 기준)

| 지표 | 원본 (2026-04-29) | 최종 (2026-05-14) |
|---|---:|---:|
| 포트폴리오 수익 (entry 신호) | +103.49% | **+188.34%** |
| 포트폴리오 수익 (strong 신호) | +140.37% | +133.86% |
| 포트폴리오 MDD (entry 신호) | -29.11% | **-19.36%** |
| 2024+ D+20 수익률 (entry) | — | **+5.05%** |
| strong 거래 건수 | 1,267건 | 7,177건 |

> **요약**: entry_signal 포트폴리오(+188.34%)가 strong(+133.86%) 대비 54.48%p 우수.  
> flow 가중치 25% 적용 후 수급 좋은 종목 선별 효과 입증.

---

## 6. 2026-05-14 주요 개선 사항

| 항목 | 변경 내용 |
|---|---|
| RSI dead zone 수정 | `45~75` → `45~80` (버그 수정) |
| 트레일링 스탑 | `0.04/0.03` → `0.025/0.05` |
| flow 가중치 | 15% → **25%** (trend 30→25%, liquidity 25→20%) |
| 보유 기간 기본값 | 10일 → **20일** / Defensive: 7→14일 |
| trading_value 기준 | paper 5억→10억 / live 20억→30억 |
| 포지션 크기 | 고정 5% → `rule_score_v3` 기반 동적 배분 |
| 레짐 필터 | 완전 차단 → Defensive 시 max_positions 3으로 축소 |
| `RULE_ALLOW_ENTRY_SIGNAL` | `1` 활성화 (entry 신호로도 진입 허용) |

---

## 7. 검증 미완 사항

**RULE SELL 실거래 미검증**

RULE이 Live 전환 이후 SELL 조건이 아직 발동된 적 없습니다.  
첫 포지션 진입 후 EXIT 조건 발동 시 KIS 실주문 제출까지 정상 동작하는지 확인이 필요합니다.

- 코드 경로: `rule_portfolio_manager.py` → `rule_order_preview_builder.py` → `rule_order_submitter.py` SELL 경로
- 리스크 수준: 낮음 (발동 전까지 영향 없음)

---

## 8. 핵심 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `RULE_TRADING_RUN_MODE` | `paper` | `paper` \| `pilot` \| `live` |
| `RULE_MAX_POSITIONS` | `5` | 최대 보유 종목 수 |
| `RULE_MAX_POSITIONS_DEFENSIVE` | `3` | Defensive 모드 최대 종목 수 |
| `RULE_NEW_ENTRY_WEIGHT` | `0.16` | 신규 진입 기본 비중 (16%) |
| `RULE_ALLOW_ENTRY_SIGNAL` | `1` | entry 신호 진입 허용 |
| `RULE_COOLDOWN_DAYS` | `5` | 청산 후 재진입 금지 기간 |
| `RULE_MAX_HOLDING_DAYS` | `20` | 최대 보유 기간 |
| `RULE_ENTRY_RULE_SCORE_MIN` | — | entry 진입 최소 rule_score |
| `RULE_STRONG_RULE_SCORE_MIN` | — | strong 진입 최소 rule_score |
| `US_MACRO_ENABLED` | `1` | US 매크로 오버레이 활성화 |
