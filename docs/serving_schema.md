# Serving Schema

## 목적

이 문서는 운영 산출물을 웹서비스에서 바로 사용할 수 있는 JSON payload로 정규화한 serving schema를 정의한다. 기본 원칙은 다음과 같다.

- 프론트엔드가 별도 CSV 파싱 없이 바로 `fetch` 가능한 JSON을 사용한다.
- 운영 데이터가 비어 있어도 엔티티 구조는 유지한다.
- 버전 정보와 source 상태를 payload 최상단에 함께 둔다.

## 엔티티

### 1. `daily_recommendations`

파일: `serving/daily_recommendations.json`

핵심 필드:

- `entity`: 고정값 `daily_recommendations`
- `generated_at`: exporter 생성 시각
- `asof_date`: 공식 추천 산출일
- `source_status`: `current` 또는 `stale`
- `source_detail.candidates_path`: 원본 후보 CSV 경로
- `source_detail.latest_snapshot_date`: 최신 snapshot 기준일
- `versions`: 운영 freeze 버전 묶음
- `gate_overall_status`: 현재 buy gate 전체 상태
- `count`: 추천 종목 수
- `items`: 추천 종목 배열

`items[]` 필드:

- `recommendation_id`: 서비스용 안정 식별자
- `asof_date`: 추천 기준일
- `target_bucket`: 추천 버킷 크기
- `buy_rank`: 추천 순위
- `rank_source`: 원본 rank source
- `security.code`: 6자리 종목코드
- `security.name`: 종목명
- `security.market`: 시장 구분
- `security.sector`: 섹터명
- `security.dominant_theme`: 대표 theme
- `scores.final_score`: 현재 추천 점수
- `scores.confidence_score`: confidence score
- `scores.liquidity_score`: liquidity score
- `scores.theme_score`: theme score
- `scores.latest_snapshot_final_score`: 최신 snapshot 기준 점수
- `scores.latest_snapshot_confidence_score`: 최신 snapshot 기준 confidence
- `scores.score_drift_vs_latest_snapshot`: 최신 snapshot 대비 점수 차이
- `market_signals.*`: 거래대금, 단기 수익률, 모멘텀, RSI
- `selection.selection_stage`: strict / relaxed 여부
- `selection.selection_notes`: 선발 메모
- `selection.recent_surge_soft_flag`: 과열 soft flag
- `selection.entry_rule_pass`: 기본 entry threshold 통과 여부
- `selection.latest_snapshot_rank`: 최신 snapshot 순위
- `score_explanations.summary_text`: 설명 요약
- `score_explanations.highlights[]`: 프론트 노출용 핵심 문장 배열

### 2. `buy_gate_status`

파일: `serving/buy_gate_status.json`

핵심 필드:

- `entity`: 고정값 `buy_gate_status`
- `generated_at`: gate 생성 시각
- `asof_date`: gate 기준일
- `versions`: 운영 freeze 버전 묶음
- `primary_bucket`: 운영 기준 버킷
- `overall_status`: `BUY_ALLOWED` / `WATCH` / `HOLD` / `BLOCK`
- `theme_churn_status`: theme churn gate 상태
- `daily_cycle_status`: 일일 사이클 상태
- `decisions[]`: bucket별 상세 판단

`decisions[]` 필드:

- `bucket`: 대상 버킷
- `status`: 해당 버킷 gate 상태
- `reason_summary`: 상태 요약 이유
- `candidate_diagnostics`: 후보 정적 품질 진단
- `benchmark_diagnostics`: benchmark 비교 진단
- `forward_diagnostics`: 성과 maturity 진단
- `confidence_diagnostics`: confidence 안정성 진단
- `comparison_diagnostics`: raw 대비 변화 진단

### 3. `model_portfolio`

파일: `serving/model_portfolio.json`

핵심 필드:

- `entity`: 고정값 `model_portfolio`
- `generated_at`: exporter 생성 시각
- `asof_date`: 추천 기준일
- `versions`: 운영 freeze 버전 묶음
- `source_status`: `model_portfolio_top5` 또는 `derived_preview`
- `source_path`: 원본 포트폴리오 소스 경로
- `constraints.*`: 현금 버퍼, 종목 cap, sector/theme cap
- `cash_target`: 목표 현금 비중
- `holding_count`: 모델 포트폴리오 종목 수
- `holdings[]`: 종목별 목표 비중

`holdings[]` 필드:

- `code`, `name`: 종목 식별
- `target_weight`: 목표 비중
- `buy_rank`: 추천 순위
- `sector`, `dominant_theme`: 집중도 관리용 분류
- `final_score`, `confidence_score`, `liquidity_score`: 핵심 스코어
- `selection_stage`: strict / relaxed 여부

주의:

- 현재 `data/model_portfolio_top5.csv`가 없으면 exporter가 후보 CSV를 기반으로 preview weight를 계산하고 `source_status=derived_preview`로 표기한다.

### 4. `performance_summary`

파일: `serving/performance_summary.json`

핵심 필드:

- `entity`: 고정값 `performance_summary`
- `generated_at`: exporter 생성 시각
- `versions`: 운영 freeze 버전 묶음
- `paper_trading`: paper trading NAV 요약
- `benchmark_summary.items[]`: horizon별 benchmark 성과 요약
- `weekly_review`: 최신 weekly review row
- `score_kpi_monitor`: 점수 체계 KPI 요약
- `confidence_calibration`: confidence calibration 요약

`paper_trading` 필드:

- `available`: NAV 파일 존재 여부
- `latest_date`: 최신 NAV 일자
- `latest_nav`: 최신 NAV
- `cumulative_return`: 누적 수익률
- `drawdown`: 드로다운

`benchmark_summary.items[]` 필드:

- `top_n`: 포트폴리오 크기
- `horizon_days`: 평가 horizon
- `benchmark_name`: 비교 benchmark
- `dates_total`: 총 관측일 수
- `dates_matured`: maturity 충족 일 수
- `avg_portfolio_return`: 포트폴리오 평균 수익률
- `avg_benchmark_return`: benchmark 평균 수익률
- `avg_excess_return`: 초과 수익률
- `excess_hit_ratio`: 초과 성과 hit ratio
- `benchmark_available`: benchmark 유효 여부

## 변환 규칙

- CSV/JSON 원본에서 `NaN`은 모두 JSON `null`로 변환한다.
- 종목코드는 항상 6자리 문자열로 맞춘다.
- theme 공란은 `(none)`으로 통일한다.
- 성과/포트폴리오 원본이 없어도 엔티티 자체는 빈 구조로 생성한다.

## 현재 운영 주의사항

- `daily_recommendations`는 현재 공식 후보 파일이 최신 snapshot보다 오래되면 `source_status=stale`로 내려간다.
- `model_portfolio`는 실제 top5 포트폴리오 CSV가 아직 없어서 preview 계산값을 사용할 수 있다.
- `paper_trading_nav.csv`가 없으면 `performance_summary.paper_trading.available=false`로 내려간다.
