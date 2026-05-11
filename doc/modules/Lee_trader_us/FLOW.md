# Lee_trader_us Flow

## Target Flow

`Universe 관리 -> OHLCV 수집 -> 데이터 품질 검증 -> feature 생성 -> score/ranking -> backtest -> paper trading -> live trading 검토`

## Phase 1 Scope

Phase 1 prepares only the early data foundation:

1. Universe management
2. OHLCV collection
3. data quality validation
4. baseline feature generation

The following stages are explicitly out of scope in this phase:

1. AI/ML model training
2. score/ranking implementation
3. backtest implementation
4. paper trading
5. live trading review and execution

## Phase 1-5 Note

`build_us_features.py` calculates only simple price-based baseline features.

- It is not an AI model.
- It does not create final recommendation scores.
- It does not connect to Korean auto-trading flows.

## Phase 1-6 Note

`run_us_daily_pipeline.py` runs the standalone Phase 1 sequence:

1. universe load
2. price collection
3. quality validation
4. baseline feature generation

It stays isolated from Korean schedulers and does not execute paper trading, live trading, or AI model training.
