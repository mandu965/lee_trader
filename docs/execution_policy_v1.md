## 2026-04-18 Authoritative Note

- The current live implementation uses `data/ranking_final.csv` as the candidate source.
- Candidate universe is `top20`.
- Standard entry review range is `top8`.
- Extended review range is `top9~10`.
- `top5` is retained only as a priority interpretation bucket, not as the execution input file.
- Max holdings is `8`.
- `WATCH` allows limited auto-buy with max `2` new names, max `15%` total new exposure, and max `8%` per new position.
- If older sections below still mention `buy_candidates_top5.csv` or `top5` as the primary execution input, treat this note as the source of truth.

# Execution Policy V1

## 목적

Execution Policy V1은 추천 종목을 실제 계좌 운용 행동으로 연결하는 운영 규칙이다. 목표는 추천 결과를 실제 매매 행동으로 일관되게 해석하고, 과도한 turnover와 concentration을 억제하는 것이다.

이 정책은 `production_v1` 운영 설정을 기준으로 고정되며 아래 버전을 따른다.

- `score_formula_version`: `ranking_builder_v8_return_prob_tech_regime`
- `gate_version`: `operational_buy_gate_v1`
- `portfolio_version`: `model_portfolio_constructor_v1`
- `execution_policy_version`: `execution_policy_v1`

## 입력 원칙

- 기본 추천 입력은 `data/buy_candidates_top5.csv`를 사용한다.
- 운영 위험 상태는 `outputs/operational_buy_gate.json`의 `overall_status`를 사용한다.
- 보유 종목 파일이 있으면 해당 포지션을 기준으로 행동을 분류한다.
- 보유 파일이 없으면 빈 계좌 기준 시뮬레이션만 생성한다.
- 추천 산출일이 최신 ranking snapshot보다 오래된 경우 실제 주문 전 추천 파이프라인을 재생성해야 한다.

## 정책 항목

### 1. 신규 진입 조건

- `buy gate` 상태가 `BUY_ALLOWED`일 때만 신규 진입을 허용한다.
- 종목은 공식 추천 버킷 `top5` 안에 있어야 한다.
- `confidence_score >= 80`
- `liquidity_score >= 15`
- `trading_value >= 5,000,000,000`
- 최근 청산 종목은 `10` 영업일 cooldown 안에서는 재진입하지 않는다.
- 신규 진입 후에도 `cash minimum`, `종목당 최대 비중`, `sector/theme cap`을 모두 만족해야 한다.

### 2. 기존 보유 유지 조건

- 현재 보유 종목이 공식 추천 `top5`에 남아 있으면 기본 유지한다.
- 공식 추천에서는 이탈했더라도 최신 snapshot 기준 `top8` 이내이고 `confidence_score >= 76`이면 유지 가능하다.
- `buy gate`가 `HOLD` 또는 `BLOCK`일 때는 신규 매수보다 기존 보유 유지 판단을 우선한다.

### 3. 비중 축소 조건

- 종목 비중이 `24%`를 초과하면 초과분을 축소한다.
- 동일 sector 합산 비중이 `35%`를 초과하면 해당 sector 내 하위 우선순위 종목부터 축소한다.
- 동일 theme 합산 비중이 `35%`를 초과하면 해당 theme 내 하위 우선순위 종목부터 축소한다.
- 종목이 공식 추천 `top5` 밖으로 밀리고 최신 snapshot 기준 `top8` 이내인 경우 유지하되 비중은 축소 가능 상태로 본다.
- `buy gate`가 `HOLD` 또는 `BLOCK`이면 신규 배치 대신 축소와 현금 비중 확보를 우선한다.

### 4. 교체 조건

- 교체는 기본적으로 `BUY_ALLOWED` 상태에서만 허용한다.
- 다만 `WATCH` 상태에서는 위험 축소 목적의 방어적 교체만 허용한다.
- 교체 대상 보유 종목은 다음 중 하나를 만족해야 한다.
  - 최신 snapshot 기준 `top8` 밖으로 이탈
  - `confidence_score < 72`
  - 집중도 cap 위반 해소가 필요
- 신규 후보는 교체 대상보다 `final_score`가 최소 `3.0`점 높아야 한다.
- 한 번의 검토 사이클에서 교체 종목 수는 최대 `2`개로 제한한다.

### 5. 최대 종목 수

- 최대 보유 종목 수는 `5`개다.

### 6. 종목당 최대 비중

- 종목당 최대 비중은 `24%`다.

### 7. Sector/Theme Cap

- 동일 sector 합산 비중은 `35%`를 상한으로 둔다.
- 동일 theme 합산 비중은 `35%`를 상한으로 둔다.
- `dominant_theme`가 `(none)`인 종목은 theme cap 계산에서 제외하고 sector cap만 적용한다.

### 8. Cash Minimum

- 기본 현금 최소 비중은 `5%`다.
- `HOLD` 또는 `BLOCK` 상태에서는 현금 비중이 `5%`보다 높아져도 이를 강제로 재투자하지 않는다.

### 9. Re-entry Cooldown

- 청산 후 `10` 영업일 동안은 동일 종목 재진입을 금지한다.
- cooldown 예외는 두지 않는다. 운영 일관성을 우선한다.

### 10. Buy Gate 상태별 행동 지침

| buy_gate 상태 | 행동 지침 |
| --- | --- |
| `BUY_ALLOWED` | 신규 진입, 교체, 목표 비중 재조정 허용 |
| `WATCH` | 신규 진입은 기본 보류, 위험 축소 목적 교체만 허용 |
| `HOLD` | 신규 진입 금지, 기존 보유 유지 또는 비중 축소만 허용 |
| `BLOCK` | 신규 진입 금지, cap 위반 및 약한 종목부터 축소 또는 청산 |

## 실행 순서

1. `buy gate` 상태를 확인한다.
2. 현재 보유 종목을 공식 추천 `top5`와 최신 snapshot에 대조한다.
3. 집중도 cap과 현금 비중을 점검한다.
4. cooldown 제약을 확인한다.
5. `신규 진입`, `유지`, `축소`, `교체`, `청산 보류` 중 하나로 분류한다.
6. `BUY_ALLOWED`가 아닐 경우 신규 매수는 생성하지 않는다.

## 운영 해석 주의사항

- 이 정책은 연구용 최적화가 아니라 운영 일관성을 위한 보수적 규칙이다.
- 추천 파일이 stale이면 정책 적용 결과도 참고용으로만 해석해야 한다.
- 실제 주문 전에는 최신 추천 산출물과 gate 상태를 같은 날짜 기준으로 맞춰야 한다.
## 11. WATCH Limited Auto-Buy

- `WATCH` 상태에서도 제한적 신규 진입을 허용하는 소액 실거래 모드를 둔다.
- 이 모드는 `BUY_ALLOWED` 전면 자동매매 전 단계에서 실제 체결 데이터를 축적하기 위한 운영 레이어다.
- 구현 위치는 `buy gate`가 아니라 `execution policy`다.

### Rules

- gate status가 `WATCH`일 때만 동작한다.
- 신규 진입 종목 수는 최대 `2`개다.
- 신규 진입 총 비중은 최대 `15%`다.
- 종목당 신규 진입 비중은 최대 `8%`다.
- 기존 `entry_eligible` 조건은 그대로 유지한다.
- 기존 보유 종목은 다시 진입하지 않는다.
- `BUY_ALLOWED`가 되면 기존 full auto 정책이 우선한다.
- `HOLD`와 `BLOCK`에서는 이 모드를 사용하지 않는다.

### Config

- `execution_policy.watch_limited_auto_buy_enabled: true`
- `execution_policy.watch_limited_max_entries: 2`
- `execution_policy.watch_limited_total_exposure: 0.15`
- `execution_policy.watch_limited_position_cap: 0.08`

### Intent

- `top20 진단 -> top5 후보 -> WATCH 소액 실거래 -> BUY_ALLOWED 전면 자동매매` 순서로 운영 증거를 쌓는다.
## 2026-04-18 Current Implementation Note

- 후보 우주는 `data/ranking_final.csv` 기준 `top20`이다.
- 기본 진입 심사군은 `top8`이다.
- 조건부 확장 심사군은 `top9~10`이다.
- `top5`는 더 이상 실행 입력 파일 자체가 아니라, 강신호 해석용 우선 버킷으로 본다.
- 실제 보유 상한은 `8`종목이다.
- `WATCH` 상태에서는 제한적 신규 진입을 허용한다.
  - 최대 `2`종목
  - 총 신규 노출 `15%`
  - 종목당 신규 진입 상한 `8%`
- `HOLD`와 `BLOCK`에서는 신규 진입을 허용하지 않는다.
