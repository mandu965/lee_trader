# Score Column Definitions

> 운영 기준 단일화를 위한 컬럼 정의서.
> 실주문 판단 기준 컬럼과 research 전용 컬럼을 명확히 구분합니다.
> 코드 출처: `python/ranking_builder.py`, `python/scoring/final_score.py`

---

## 1. 운영 기준 컬럼 (실주문 판단에 사용)

| 컬럼명 | 역할 | 계산 방식 | 비고 |
|---|---|---|---|
| `final_score` | 베이스라인 운영 점수 | regime-aware weighted sum (아래 가중치 표 참조) | theme OFF 시 정렬 기준 |
| `live_score` | **실제 정렬 기준 점수** | theme ON → `final_score_v3`, theme OFF → `final_score` (**현재: `final_score`**) | UI·주문 순위의 실질 입력값 |
| `live_score_source` | live_score의 출처 컬럼명 | `"final_score"` 또는 `"final_score_v3"` | 디버깅용 |
| `live_rank` | live_score 기준 날짜별 순위 | per-date rank by `live_score_col` | 실주문 종목 선택 기준 |
| `rank_final` | live_rank와 동일 | `= live_rank` | UI 표시용 alias |

### `final_score` 가중치 (regime별)

| Regime | ret_score | prob_score | tech_score | qual_score | risk_penalty 배율 |
|---|---|---|---|---|---|
| `bull` | 0.38 | 0.27 | 0.27 | 0.08 | 0.40 |
| `neutral` | 0.32 | 0.26 | 0.24 | 0.18 | 0.65 |
| `defensive` | 0.26 | 0.22 | 0.18 | 0.34 | 0.80 |

출처: `python/scoring/final_score.py` — `BULL_WEIGHT_PROFILE`, `NEUTRAL_WEIGHT_PROFILE`, `DEFENSIVE_WEIGHT_PROFILE`

---

## 2. final_score 구성 컴포넌트 (운영 축)

| 컬럼명 | 역할 | 계산 방식 |
|---|---|---|
| `ret_score` | **1차 예측 수익률 점수** | `100 × (0.7 × ret_rank_60 + 0.3 × ret_rank_90)` — 날짜별 백분위 blend |
| `prob_score` | **Top20 확률 점수** | `prob_top20_60d`의 날짜별 백분위 순위 × 100 (상대 점수) |
| `tech_score` | **기술적 흐름 점수** | vol, momentum, MA, RSI 지표 composite |
| `qual_score` | **재무 품질 점수** | financial quality composite (안정성·수익성 지표) |
| `risk_penalty` | **손실위험 차감** | `pred_mdd_*` 기반 soft penalty (weighted score에서 차감) |

---

## 3. Research 전용 컬럼 (실주문 판단 제외)

> 아래 컬럼들은 모델 비교·실험·진단 목적으로 저장됩니다.
> **실주문 로직(`submit_live_orders.py`, `build_trade_intents.py`)에서 참조하면 안 됩니다.**

| 컬럼명 | 분류 | 계산 방식 | 현황 |
|---|---|---|---|
| `pred_score` | 레거시 모델 백분위 | `0.6 × pred_score_60 + 0.4 × pred_score_90` | research 비교용 |
| `pred_score_60` | 60d 모델 백분위 | `pred_return_60d`의 날짜별 백분위 × 100 | `pred_score` 구성요소 |
| `pred_score_90` | 90d 모델 백분위 | `pred_return_90d`의 날짜별 백분위 × 100 | `pred_score` 구성요소 |
| `ret_score_v11` | ret_score 레거시명 | `= ret_score` (exact alias) | 호환성 유지 |
| `return_score` | ret_score UI alias | `= ret_score` | UI 호환성 |
| `probability_score` | prob_score UI alias | `= prob_score` | UI 호환성 |
| `prob_score_raw` | 절대값 확률 변환 | `prob_top20_60d × 100` (날짜별 상대화 없음) | 진단용 |
| `prob_top20_90d` | 90d 확률 (저장만) | 모델 출력값 그대로 | research 보조 신호 |
| `final_score_v2` | 고정가중치 레퍼런스 | WEIGHT_PRED=0.30, WEIGHT_PROB=0.25, WEIGHT_TECH=0.15, WEIGHT_QUAL=0.10, WEIGHT_SAFETY=0.15, WEIGHT_LIQUIDITY=0.05 | research 비교용 |
| `rank_v2` | final_score_v2 순위 | per-date rank by `final_score_v2` | research 비교용 |
| `valuation_score` | 밸류에이션 진단 | PER/PBR/PSR 등 — 직접 운영 축 아님 | 호환성 유지 |
| `safety_score` | 안정성 진단 | 부채비율·이자보상배율 등 | 호환성 유지 |
| `liquidity_score` | 유동성 진단 | 거래대금 기반 | 호환성 유지 (liquidity_gate는 별도) |

---

## 4. Financial Momentum Overlay 컬럼

### Shadow 컬럼 (Phase 4 — 항상 계산, live_score 미반영)

> `FINANCIAL_SCORE_OVERLAY_ENABLED=0` (기본) 상태에서는 아래 shadow 컬럼만 생성됩니다.

| 컬럼명 | 역할 |
|---|---|
| `fin_momentum_phase` | 매출·영업이익 추세 구간 (ACCELERATING~DECLINING 등) |
| `fin_hard_risk` | 실적 훼손 위험 flag (0.0 / 1.0) |
| `shadow_fin_momentum_adj` | 재무 모멘텀 가감점 (ACCELERATING +5 ~ DECLINING -10, hard_risk -15) |
| `shadow_fin_final_score` | overlay 반영 가상 점수 (`final_score + shadow_fin_momentum_adj`) |
| `shadow_fin_rank` | 가상 점수 기준 순위 |
| `shadow_fin_rank_diff` | `rank_final - shadow_fin_rank` (양수 = 재무 반영 시 순위 상승) |
| `shadow_fin_hard_risk_triggered` | hard_fundamental_risk 발동 여부 |

### 운영 컬럼 (Phase 7 활성화 시 — `FINANCIAL_SCORE_OVERLAY_ENABLED=1`)

> Phase 6 백테스트 통과 후 활성화. `final_score`와 `live_score`에 실반영됩니다.

| 컬럼명 | 역할 |
|---|---|
| `fin_momentum_adj` | 실제 적용된 가감점 (shadow와 동일 계산, Phase 7 전에는 0.0) |
| `fin_hard_risk_triggered` | hard_risk 발동 여부 (live 버전) |
| `fin_overlay_applied` | Phase 7 overlay 적용 여부 flag |

### BUY gate 규칙 (Phase 8 — `FINANCIAL_BUY_GATE_ENABLED=1`)

| fin_momentum_phase | qty_scale | 효과 |
|---|---|---|
| ACCELERATING / GROWING / UNKNOWN | 1.0 | 정상 |
| SLOWING | 0.7 | 목표수량 70% |
| WEAKENING | 0.5 | 목표수량 50% |
| DECLINING | 0.3 | 목표수량 30% |
| hard_fundamental_risk=1 | 0.0 | BUY 완전 차단 |

---

## 6. Theme Overlay 컬럼 (현재 비활성 — 향후 전환 가능)

> `ENABLE_THEME_OVERLAY=0` / `production_v1.yaml: theme_overlay.enabled: false` 상태.
> theme overlay가 켜지면 `final_score_v3`가 `live_score`의 실질 기준으로 승격됩니다.

| 컬럼명 | 역할 | 운영 전환 조건 |
|---|---|---|
| `final_score_v3` | theme confidence 반영 점수 | `live_uses_theme=True` 시 live_score 기준으로 승격 |
| `final_score_before_theme` | theme 적용 전 baseline | 중간 계산값 |
| `score_diff_v3` | `final_score_v3 - final_score` 차이 | theme 영향 진단 |
| `score_diff_v2` | `final_score_v3 - final_score_v2` 차이 | theme 영향 진단 |
| `shadow_final_score_v3` | shadow 비교용 v3 | debug 전용, 운영 rank 미사용 |
| `shadow_rank_v3` | shadow v3 순위 | debug 전용 |

---

## 7. 진단·내부 계산 컬럼

| 컬럼명 | 역할 |
|---|---|
| `ret_rank_60`, `ret_rank_90` | ret_score 계산용 중간값 (0~1 scale) |
| `pred_return_60d_pct01`, `pred_return_90d_pct01` | ret_rank_60/90의 alias |
| `final_score_raw` | risk penalty 적용 전 intermediate |
| `final_score_baseline` | theme overlay 계산 중간값 |
| `ret_score_missing`, `prob_score_missing` 등 | 컴포넌트 결측 여부 flag |
| `ret_score_fallback_used`, `prob_score_fallback_used` 등 | fallback 적용 여부 flag |
| `confidence_score` | 모델 메타 점수 (calibration 완성 전) — live_score 미반영, research 용도 |

---

## 8. 운영 판단 요약

```
실주문 종목 선택 순서
──────────────────────────────────────────────
1. live_rank (= rank_final)   ← 가장 중요
2. live_score                 ← live_rank의 기준값
3. live_score_source          ← 현재 "final_score" (theme OFF)
4. final_score 구성: ret_score·prob_score·tech_score·qual_score·risk_penalty
──────────────────────────────────────────────
실주문 로직에서 절대 참조하면 안 되는 컬럼
  pred_score, final_score_v2, final_score_v3 (theme OFF 동안), confidence_score
```

---

## 9. 관련 문서

- `doc/modules/Lee_trader_score/RUNTIME_SORTING.md` — theme flag 기준 live_rank 분기 상세
- `doc/shadow_promotion_criteria.md` — `SCORE_FORMULA_VERSION` 전환 조건
- `python/ranking_builder.py` — 전체 score 계산 파이프라인 (모듈 헤더 참조)
- `python/scoring/final_score.py` — `final_score` 가중치 프로파일 정의

---

*2026-05-13 작성 | 2026-05-15 업데이트 — Financial Momentum Phase 4~8, C-1 공매도 잔고 추가*
