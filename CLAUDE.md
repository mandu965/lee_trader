# Lee Trader — 운영 현황

> Claude Code 전용 문서. 마지막 전체 감사: 2026-05-29

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 활성 모듈 | `Lee_trader_ai` · `Lee_trader_score` · `Lee_trader_backTest` |
| 종료 모듈 | `Lee_trader_rule` (2026-05-21), `Lee_trader_us` (2026-05-29) |
| 운영 시스템 | **KR AI 자동매매 단독** |
| 작업 환경 | VS Code + Claude Code |
| 언어/런타임 | Python 3.11+ · Node.js 20+ · TypeScript |
| DB | PostgreSQL (`DATABASE_URL`) — Docker `lee_trader_pg:15432` |
| 환경변수 | [.env](.env) (로컬) · [config/production_v1.yaml](config/production_v1.yaml) (운영) |
| 웹 운영자 페이지 | <http://localhost:3400/ops-readiness.html> |

---

## 2. 🔴 절대 원칙 — 작업 전 반드시 확인

```
1. AUTO_TRADE_EXECUTE=0 상태인지 확인 (.env)
2. .env 파일 존재 확인
3. DATABASE_URL 연결 가능 여부 확인
4. 실주문 파일(submit_live_orders.py, run_live_auto_trade_cycle.py) 수정 시
   → paper trading 환경에서 3일 이상 검증 먼저
5. ranking_builder.py 수정 시
   → 점수·순위·UI·주문 동시 영향 → shadow 비교 먼저 실행
6. 종료된 모듈(rule_*, us_*) 관련 신규 작업·제안 금지
```

---

## 3. 현재 운영 상태 (2026-05-29 기준)

### KR AI 자동매매

| 항목 | 상태 |
|---|---|
| 실행 모드 | LIVE (`AUTO_TRADE_EXECUTE=1`, `AUTO_TRADE_ALLOW_BUY=1`) |
| BUY 승인 | 완전 자동 (`AUTO_TRADE_BUY_APPROVAL_REQUIRED=0`) |
| Gate 상태 | **PILOT** — 2026-06-10 전후 BUY_ALLOWED 자동 전환 예상 |
| Walkforward | REJECTED (만기 표본 1일 누적, 정상) |
| Universe | 204종목 (KOSPI 100 + KOSDAQ 100 + α), 최신일 2026-05-28 |
| daily_ranking | 누적 저장 정상 |
| 모델 | LightGBM, 43 피처, 6 reg + 1 cls 타겟, halflife 3년 sample_weight |
| 보유 한도 | `AI_MAX_HOLDING_DAYS=30`, `AI_MAX_HOLDING_DAYS_HARD_CAP=45` |
| 점수 공식 | `ranking_builder_v9_flow` (수급 피처 포함) |

### 상태 해석 — 중요

**Walkforward REJECTED · Gate PILOT은 모두 정상입니다.**

시스템은 2026-03-29 라이브 운영 시작. 벤치마크 비교, 순방향 수익률, confidence calibration은 **60~90일치 실거래 데이터**가 쌓여야 판단 가능.

```
운영 시작:        2026-03-29
60d 만기 시작:    2026-05-28 ✓ (도달)
90d 만기 시작:    2026-06-27
BUY_ALLOWED 예상: 2026-06-10 전후
```

2026-05-28부터 benchmark matured_dates가 쌓이기 시작하면 walkforward 통과 및 PILOT → BUY_ALLOWED 자동 전환.

---

## 4. 모니터링 — 운영자 페이지

**접속**: <http://localhost:3400/ops-readiness.html>

### Gate 전환 핵심 지표 (이 카드 4개만 봐도 충분)

| 항목 | 현재 | 목표 | 의미 |
|---|---|---|---|
| 첫 배지 (overall_status) | `PILOT` | `BUY_ALLOWED` | 신규 진입 검토 가능 |
| 둘째 배지 (walkforward) | `REJECTED` | `ACCEPTED` | 5개 sub-condition 모두 통과 |
| matured benchmark dates | 0 | **≥ 3** | 60일 만기 도달 날짜 수 |
| trusted ratio top20 | 35% | ≥ 30% (PASS) | 이미 통과 |

### 운영 전환 체크리스트 6개 ([node/index.js:2296](node/index.js#L2296))

운영자 페이지가 자동으로 통과 건수를 카운트. BUY_ALLOWED는 6/6 통과 시점에 자동 승격.

### 매일 확인 절차

매일 18:30 이후 (종가 배치 완료) 운영자 페이지 "매수 gate" 카드에서:
1. `PILOT` → `BUY_ALLOWED` 전환 여부
2. `REJECTED` → `CONDITIONAL` → `ACCEPTED` 진행 여부
3. matured benchmark dates 0 → 1 → 2 → 3 증가 추세

---

## 5. 활성 점수 공식 — ranking_builder_v9_flow

[scoring/final_score.py](python/scoring/final_score.py) · [production_config.py](python/production_config.py) `_SCORE_FORMULA_VERSION_DEFAULT = "ranking_builder_v9_flow"`

```
final_score = w_ret   * ret_score
            + w_prob  * prob_score
            + w_tech  * tech_score
            + w_qual  * qual_score
            + w_flow  * flow_score
            - w_risk  * risk_penalty

flow_score = 0.6 * percentile(flow_foreign_net_5d)
           + 0.4 * percentile(flow_inst_net_5d)
           (0~100, 동일 날짜 내 상대 순위; 데이터 없으면 50.0 neutral)
```

### 레짐별 가중치

| 레짐 | ret | prob | tech | qual | flow |
|---|---:|---:|---:|---:|---:|
| Bull | 0.33 | 0.24 | 0.23 | 0.08 | 0.12 |
| Neutral (현재) | 0.28 | 0.23 | 0.21 | 0.18 | 0.10 |
| Defensive | 0.23 | 0.19 | 0.16 | 0.34 | 0.08 |

상세 정의: [doc/score_column_definitions.md](doc/score_column_definitions.md)

---

## 6. 활성 피처 (43개)

핵심 그룹:
- **수익률**: `ret_5d`, `ret_20d`, `ret_60d`, `ret_120d`, `high_52w_ratio`
- **수급**: `flow_foreign_net_5d/20d`, `flow_inst_net_5d/20d` (FIN 데이터 138K행 백필 완료)
- **공매도**: `short_ratio`, `short_ratio_5d_chg`, `short_ratio_20d_avg` (피처 중요도 9위·15위)
- **재무 모멘텀**: `fin_momentum_score` (8위), `fin_turnaround_score` (10위), `fin_risk_score`, `fin_momentum_phase`
- **재무 YoY**: `roe_yoy`, `revenue_growth_yoy`, `op_income_growth_yoy` (2026-05-29 merge 버그 수정 완료)
- **섹터 상대강도**: `sector_rel_momentum_20d`
- **기술/품질**: tech_*, quality_*, volume_*, value_*, liquidity_*

피처 사전: [doc/feature_dictionary.md](doc/feature_dictionary.md)

---

## 7. 잔여 과제 — 모두 2026-08 이후 재검토 보류

| 과제 | 사유 |
|---|---|
| B-1 DART 분기/반기 재무 | 연간 데이터로 `fin_momentum_score`가 이미 피처 중요도 8위로 작동. 분기 추가는 base period 불일치·look-ahead 리스크 |
| B-2 외국인 지분율 | `flow_foreign_net_5d/20d`와 상관도 높아 marginal. 크롤링 의존성만 늘어남 |
| C-2 배당수익률 | 시스템 목표(월 2% 성장형)와 부합하지 않음. defensive regime만 활용 가능 |

**재검토 시점**: 2026-08~09 (live 운영 2~3개월 누적 후 실제 알파 누수 패턴 관찰 → 데이터 기반 추가 결정)

**원칙**: 가설부터 세우지 말고, 운영 데이터가 보여주는 약점에 맞춰 추가. 현재 KR AI는 Gate 전환 직전이라 신규 피처 추가 시 인과 해석 불가능해짐.

---

## 8. 종료된 서비스 (참고용)

| 서비스 | 종료일 | 사유 |
|---|---|---|
| KR Rule 자동매매 | 2026-05-21 | KR AI 주력 집중, 백테스트 동형성 재구축 전까지 보류 (`RULE_LIVE_ENABLED=0`) |
| US AI 자동매매 | 2026-05-29 | KR AI 단독 운영으로 전환 결정 |

> 위 서비스 관련 신규 작업·제안 금지 (절대 원칙 6번).

---

## 작업 완료 기준 (Definition of Done)

- [ ] 코드 변경 단위 실행 검증 완료
- [ ] 관련 산출물(`data/` · `outputs/`) 정상 생성 확인
- [ ] 영향받는 모듈 문서 갱신
- [ ] 환경변수 추가 시 `.env.example` 갱신
- [ ] `AUTO_TRADE_EXECUTE=0` 상태 유지 확인 (테스트 환경)
- [ ] 실주문 관련 변경 시 paper trading 3일 이상 검증

---

*Lee Trader 운영 현황 v4.0 | 2026-05-29 | KR AI 단독 운영 체제로 재편*
