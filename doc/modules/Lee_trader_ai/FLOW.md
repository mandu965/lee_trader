# Lee_trader_ai Flow

## Execution Flow

### 1. Daily Pipeline

실행 파일:

- `python/run_pipeline.py`

대표 단계:

1. 시장/유니버스 수집
2. 가격 정리 및 수정주가 생성
3. 재무/quality/feature 생성
4. label 생성
5. 모델 학습
6. 모델 예측
7. ranking 생성
8. 보조 진단/리포트 생성

### 2. Operational Refresh

실행 파일:

- `python/run_operational_refresh.py`

주요 역할:

- 현재 ranking 기준 후보 재정리
- trade intents 생성
- 운영자 해석용 payload 보강

### 3. Live Auto Trade

실행 파일:

- `python/run_live_auto_trade_cycle.py`

내부 역할:

1. live 계좌 동기화
2. 체결 동기화
3. `submit_live_orders.py` 호출
4. 리뷰/리포트 생성
5. web payload sync

### 4. Web Sync

실행 파일:

- `python/sync_web_display_data.py`

주요 역할:

- core table sync
- `research.app_payload_store` 갱신
- web API가 읽는 payload 최신화

## Data Flow

### Input

- `data/features.csv`
- `data/labels.csv`
- `data/universe.csv`
- `data/market_status.csv`
- `data/live_account_holdings.csv`
- KIS 실계좌/시세 응답

### Transform

1. feature/label 생성
2. 모델 학습 및 예측
3. final score 계산
4. ranking 정렬
5. trade intents 계산
6. order preview 생성
7. 주문 제출/체결 동기화
8. 화면 payload 반영

### Output

- `data/predictions.csv`
- `data/ranking_final.csv`
- `outputs/trade_intents.json`
- `outputs/order_requests_preview.json`
- `outputs/order_requests_execution.json`
- `outputs/live_account_balance_summary.json`
- `serving/daily_recommendations.json`

## Main Services

- Postgres
- KIS API
- Node API / web payload store

## Notes

- AI 자동매매와 RULE 자동매매는 별도 경로입니다.
- AI 쪽 실주문 여부는 `AUTO_TRADE_EXECUTE`, `AUTO_TRADE_ALLOW_BUY` 영향을 직접 받습니다.
