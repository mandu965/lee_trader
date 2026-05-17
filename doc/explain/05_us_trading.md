# US 주식 자동매매 모듈

*작성 기준일: 2026-05-17*

---

## 개요

US 주식 자동매매는 현재 **Paper Trading 단계**입니다.  
실계좌 주문 자동 제출은 비활성화 상태이며, Shadow → Paper 검토 레이어를 통해 의사결정을 검증하고 있습니다.

**현재 단계**: Phase 8-6 (제한적 BUY + SELL 설계 완료, 실구현 진행 예정)

---

## 1. 전체 구조

```
[06:30 — US 파이프라인]
  데이터 수집 → 피처 → 랭킹 → Paper Trading 검토

[07:30 — US 매크로 수집]
  collect_us_macro_etf_daily → DB 저장

[08:50 — US 매크로 Shadow]
  compute_us_macro_overlay_shadow → Shadow 비교

[매일]
  Limited BUY 자동화 (Shadow/Paper 결정 파이프라인)
  Limited SELL 자동화 (Shadow/Paper 아티팩트 기록)
```

---

## 2. Paper Trading 계좌 및 주문 흐름 (Phase 5)

### 2-1. Paper Trading 구성

- 실계좌와 **완전히 분리된** 가상 계좌
- 초기 잔고: 설정값 기준 (별도 환경변수)
- 주문 생성·체결·스냅샷·리밸런스를 실제 주문과 동일한 로직으로 처리

### 2-2. 주문 흐름

```
랭킹 결과 (ranking_final)
        ↓
BUY 후보 선정 (buy_candidate_builder)
        ↓
Fail-safe 리스크 가드 (pre-trade check)
        ↓
SHADOW 결정 파이프라인
  └─ SHADOW 모드: 의사결정만 기록 (실제 주문 없음)
        ↓
PAPER 결정 파이프라인
  └─ PAPER 모드: 가상 계좌에 주문 기록
        ↓
[미래] LIVE 결정 파이프라인
  └─ LIVE 모드: 실계좌 KIS API 제출 (현재 비활성)
```

---

## 3. 제한적 BUY 자동화 (Phase 8-1 ~ 8-5)

### 3-1. 단계별 내용

| Phase | 구현 내용 |
|---|---|
| 8-2 | 후보 로딩, fail-safe 리스크 가드, Shadow/Paper 결정 파이프라인 |
| 8-3 | 일별 리포트, 검증 요약, Paper 성과 추적 |
| 8-4 | 스케줄러 래퍼, 일별 파이프라인 훅, 실패 격리 |
| 8-5 | 누적 Paper 성과 평가, Live 준비도 스코어링, 승격 정책 검토 레이어 |

### 3-2. 안전 장치 (Fail-safe Risk Guard)

BUY 의사결정 전 반드시 통과해야 하는 체크:

| 체크 항목 | 내용 |
|---|---|
| 데이터 신선도 | 최신 랭킹·피처가 당일 기준인지 확인 |
| 최소 신뢰도 | 신뢰도 점수 임계값 미달 시 차단 |
| 포지션 한도 | 최대 보유 종목 수 초과 시 차단 |
| 중복 진입 방지 | 쿨다운 기간 내 동일 종목 재진입 차단 |
| Kill Switch | 긴급 차단 플래그 설정 시 모든 BUY 차단 |

### 3-3. Live 준비도 평가 기준 (Phase 8-5)

Paper Trading 성과가 일정 기준을 충족해야 Live 전환 검토가 가능합니다.

- 최소 Paper Trading 운영 기간
- 누적 수익률 벤치마크 초과 여부
- 최대 낙폭(MDD) 허용 범위 이내
- Sharpe Ratio 임계값 이상

---

## 4. 제한적 SELL/Exit 자동화 (Phase 8-6)

### 4-1. SELL 의사결정 유형

| 결정 유형 | 내용 |
|---|---|
| `SELL` | 전량 청산 |
| `PARTIAL_SELL` | 일부 청산 (수익 일부 실현) |
| `HOLD` | 보유 유지 |
| `REVIEW_REQUIRED` | 수동 검토 필요 |

### 4-2. SELL 조건

- 손절선 도달 (trailing stop)
- 최대 보유 기간 초과
- 목표 수익률 달성
- 랭킹 점수 급락 (재평가 후 하위 이탈)

### 4-3. 현재 구현 상태

- SELL 의사결정 파이프라인: 설계 완료 (Phase 8-6)
- Paper SELL 아티팩트 기록: 설계 완료
- **실계좌 SELL 실행**: 미구현 (Phase 8-7 이후 예정)

---

## 5. 안전 경계 (Safety Boundary)

현재 US 모듈에서 **비활성화**된 기능:

| 기능 | 상태 |
|---|---|
| 실계좌 BUY 자동 실행 | ❌ 비활성 |
| 실계좌 SELL 자동 실행 | ❌ 비활성 |
| 주문 자동 재제출 | ❌ 비활성 |
| 브로커 상태 자동 수정 | ❌ 비활성 |
| 국내 매매 로직과 계좌 공유 | ❌ 비활성 |

> **핵심 원칙**: US 모듈의 모든 코드 변경은 국내 실매매 로직에 영향을 주어선 안 됩니다.  
> DB 테이블도 `market.*`, `signal.*` 네임스페이스로 국내 테이블과 분리되어 있습니다.

---

## 6. DB 스키마 (US 전용 테이블)

| 테이블 | 내용 |
|---|---|
| `market.us_etf_daily_price` | US ETF·지수 일별 OHLCV |
| `signal.us_macro_feature_daily` | 매크로 피처 (SPY 추세, VIX 등) |
| (계획) US 주식 가격, 피처, 랭킹, Paper 포지션 테이블 | Phase별 추가 예정 |

---

## 7. 다음 단계

| Phase | 목표 |
|---|---|
| **Phase 8-7** | Paper 포지션 재구성, SELL/PARTIAL_SELL/HOLD 의사결정 구현, Paper SELL 아티팩트 기록 |
| **Phase 8-8** | SELL-first 오케스트레이션, BUY/SELL 충돌 방지, 통합 일별 거래 리포트 |
| **Phase 8-9** | 오케스트레이션 스케줄러, 중복 실행 방지 Lock, Post-run 헬스 체크 |
| **Phase 8-10** | Paper 포트폴리오 모니터링 대시보드, 벤치마크 비교 |

---

## 8. 관련 문서

- [BUY_AUTOMATION_DESIGN.md](../modules/Lee_trader_us/BUY_AUTOMATION_DESIGN.md) — Phase 8 BUY 설계
- [SELL_AUTOMATION_DESIGN.md](../modules/Lee_trader_us/SELL_AUTOMATION_DESIGN.md) — Phase 8-6 SELL 설계
- [PAPER_TRADING_QUALITY_GATE_DESIGN.md](../modules/Lee_trader_us/PAPER_TRADING_QUALITY_GATE_DESIGN.md) — Paper 품질 게이트
- [US_STOCK_LIVE_TRADING_POLICY.md](../modules/Lee_trader_us/US_STOCK_LIVE_TRADING_POLICY.md) — Live 거래 정책
- [US_STOCK_LIVE_RISK_POLICY.md](../modules/Lee_trader_us/US_STOCK_LIVE_RISK_POLICY.md) — Live 리스크 정책
- [US_STOCK_LIVE_OPERATION_RUNBOOK.md](../modules/Lee_trader_us/US_STOCK_LIVE_OPERATION_RUNBOOK.md) — 운영 런북
- [US_STOCK_PAPER_TRADING.md](../modules/Lee_trader_us/US_STOCK_PAPER_TRADING.md) — Paper Trading 상세
