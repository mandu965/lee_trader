# US 주식 랭킹 산정 파이프라인

*작성 기준일: 2026-05-17*

---

## 개요

Lee Trader의 US 모듈(`Lee_trader_us`)은 크게 두 레이어로 구성됩니다.

1. **US 매크로 오버레이** — 미국 ETF·지수 데이터를 국내 Rule 매매의 시장 레짐 보정 신호로 활용
2. **US 주식 랭킹 파이프라인** — 미국 종목을 직접 분석하여 Paper Trading용 추천 순위 산출

현재 상태: Phase 8-6 (제한적 BUY·SELL 자동화 설계 완료, Paper Trading 단계)

---

## 1. US 매크로 오버레이

### 1-1. 목적

S&P500, NASDAQ, VIX, 달러 지수 등 미국 거시 지표를 수집·계산하여  
국내 Rule 자동매매의 진입 조건을 추가로 필터링합니다.

### 1-2. 데이터 수집

**파일**: `python/collect_us_macro_etf_daily.py`  
**실행 시각**: 07:30 (US 장 마감 후)

**수집 티커 목록**:

| 티커 | 설명 |
|---|---|
| `SPY` | S&P 500 ETF |
| `QQQ` | NASDAQ 100 ETF |
| `DIA` | Dow Jones ETF |
| `XLK` | 기술 섹터 ETF |
| `XLF` | 금융 섹터 ETF |
| `XLE` | 에너지 섹터 ETF |
| `XLV` | 헬스케어 섹터 ETF |
| `XLI` | 산업재 섹터 ETF |
| `XLY` | 소비재(경기민감) 섹터 ETF |
| `XLP` | 소비재(필수) 섹터 ETF |
| `SMH` | 반도체 ETF |
| `^VIX` | VIX 변동성 지수 |
| `DX-Y.NYB` | US 달러 인덱스 |
| `^TNX` | 10년물 국채 수익률 |

**데이터 소스**: yfinance (기본) — 추후 Polygon / Alpaca로 교체 가능한 어댑터 구조  
**저장**: DB `market.us_etf_daily_price` 테이블

### 1-3. 매크로 피처 계산

**파일**: `python/compute_us_macro_feature_daily.py`

SPY·QQQ 필수 티커가 있는 경우에 한해 피처를 계산합니다.  
(Missing → `DATA_INCOMPLETE` 상태로 오버레이 비활성)

**계산 피처 예시**:
- SPY/QQQ 추세 (MA 배열, 모멘텀)
- VIX 수준 및 변화
- 달러 강도
- 섹터 로테이션 신호 (XLK vs XLP 강도 비교 등)

**저장**: DB `signal.us_macro_feature_daily` 테이블

### 1-4. 국내 Rule 매매 연동

`rule_portfolio_manager.py`에서 `_load_macro_row()`로 매크로 피처를 로드하여  
진입 필터 및 포지션 크기 조정에 활용합니다.

```python
US_MACRO_ENABLED=1          # 오버레이 활성화
US_MACRO_STALE_DAYS_LIMIT=3 # 3일 이상 오래된 데이터면 오버레이 비활성
```

---

## 2. US 주식 랭킹 파이프라인

### 2-1. 유니버스

S&P 500 구성 종목 중 유동성·거래대금 기준을 충족하는 상위 종목을 대상으로 합니다.

**데이터 소스**:
- 가격·OHLCV: yfinance
- 재무 데이터: 별도 수집 스크립트 (Phase 2)

### 2-2. 피처 생성 (Phase 1~2)

국내 파이프라인의 `feature_builder.py`와 구조가 유사하며, US 전용 테이블에 저장됩니다.

**생성 피처 계열**:
- 가격 수익률 (ret_1d~ret_120d)
- 모멘텀·이동평균
- 섹터 상대 강도 (Relative Strength vs. S&P 500)
- 재무 품질 점수

### 2-3. 랭킹 산출 (Phase 3)

**파일**: 관련 스크립트 (`python/` 하위 `us_` 접두 파일 포함)  
**문서**: [US_STOCK_RANKING_V1.md](../modules/Lee_trader_us/US_STOCK_RANKING_V1.md)

Rule 기반 랭킹 공식을 적용하며, 국내 `rule_score_v3`와 유사한 구조입니다.

### 2-4. 백테스트 (Phase 4)

**파일**: `python/backtest_us_macro_overlay.py`, `python/backtest_us_macro_overlay_rule.py`

- 매크로 오버레이 유무에 따른 수익률 비교
- 레짐별(강세/약세) 성과 분석
- 문서: [US_STOCK_BACKTEST_V1.md](../modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md)

---

## 3. 단계별 구현 현황

| Phase | 내용 | 상태 |
|---|---|---|
| Phase 1 | 유니버스, 가격 수집, 품질 검증, 기본 피처 | ✅ 완료 |
| Phase 2 | 재무 데이터, 피처 엔지니어링, 섹터 상대 강도, 레이블 | ✅ 완료 |
| Phase 3 | Rule 기반 랭킹, 운영 리포트 | ✅ 완료 |
| Phase 4 | 백테스트, 레짐 분석, 가중치 실험, Forward Test | ✅ 완료 |
| Phase 5 | Paper Trading 계좌, 주문, 체결, 스냅샷, 리밸런스, 검증 | ✅ 완료 |
| Phase 6 | Live 안전 정책, 리스크 정책, Pre-trade 체크, Kill Switch | ✅ 완료 |
| Phase 7 | Micro Live 주문 검토, 상태 동기화, Reconciliation, 운영 리포트 | ✅ 완료 |
| Phase 8-1~8-5 | 제한적 BUY 자동화 설계, Shadow/Paper 평가, 스케줄러 연결 | ✅ 완료 |
| Phase 8-6 | 제한적 SELL/Exit 자동화 설계 | ✅ 완료 |
| Phase 8-7+ | SELL/HOLD 실제 구현, 통합 오케스트레이션 | 🔲 진행 예정 |

---

## 4. 주요 파일 위치

| 파일 | 역할 |
|---|---|
| `python/collect_us_macro_etf_daily.py` | US ETF·지수 일별 OHLCV 수집 |
| `python/compute_us_macro_feature_daily.py` | 매크로 피처 계산 |
| `python/compute_us_macro_overlay_shadow.py` | Shadow 비교 실행 |
| `python/run_us_macro_overlay_shadow.py` | Shadow 스케줄러 래퍼 |
| `python/backfill_us_macro_features.py` | 매크로 피처 백필 |
| `python/backtest_us_macro_overlay.py` | 매크로 오버레이 백테스트 |
| `python/backtest_us_macro_overlay_rule.py` | Rule 매매 오버레이 백테스트 |
| `python/run_us_macro_overlay_scheduler.py` | 일일 수집·계산 스케줄러 진입점 |
| `config/us_stock_live_risk_policy.yaml` | US 주식 Live 리스크 정책 |
| `config/us_stock_paper_trading.yaml` | US 주식 Paper Trading 설정 |

---

## 5. 안전 경계

현재 US 모듈은 다음 기능이 **비활성** 상태입니다:

- 실계좌 BUY 자동 실행
- 실계좌 SELL 자동 실행
- 브로커 주문 자동 재제출
- 브로커 상태 자동 덮어쓰기
- 국내 실매매 로직과의 공유 계좌 접근

모든 US 거래 의사결정은 현재 **Shadow → Paper 검토 레이어**에서만 이루어집니다.

---

## 6. 관련 문서

- [ARCHITECTURE.md](../modules/Lee_trader_us/ARCHITECTURE.md) — 전체 실행 경계
- [CONTEXT.md](../modules/Lee_trader_us/CONTEXT.md) — 아키텍처 경계 및 단계별 이력
- [ENV.md](../modules/Lee_trader_us/ENV.md) — 환경변수 전체 목록
- [DB_SCHEMA.md](../modules/Lee_trader_us/DB_SCHEMA.md) — DB 스키마 상세
- [US_STOCK_RANKING_V1.md](../modules/Lee_trader_us/US_STOCK_RANKING_V1.md) — 랭킹 공식 v1
- [US_STOCK_BACKTEST_V1.md](../modules/Lee_trader_us/US_STOCK_BACKTEST_V1.md) — 백테스트 결과
