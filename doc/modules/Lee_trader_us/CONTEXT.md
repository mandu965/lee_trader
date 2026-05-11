# Lee_trader_us Context

## Relationship To Korean System

- Project C is separated from the Korean auto-trading pipeline.
- It does not modify Korean score calculation, order creation, or KIS execution flow.
- Failure in the US pipeline must not propagate to the Korean daily trading pipeline.

## Relationship To Project B

- Project B handled US macro overlay data for the Korean system.
- Project C is a separate long-term track for direct US stock recommendation.
- Project B and Project C may share some operational ideas, but their runtime paths remain separate.

## Long-Term Goal

The long-term goal is a US stock recommendation system that can evolve through:

- universe management
- market data collection
- feature generation
- score and ranking
- backtest
- paper trading
- live trading review

## Phase 1 Limits

Phase 1 is limited to data foundation and standalone validation.

- universe loading is implemented
- yfinance OHLCV collection is implemented
- quality validation is implemented
- baseline feature computation is implemented
- standalone US daily pipeline is implemented
- ranking logic is not implemented
- paper trading is not implemented
- live trading is not implemented
- AI/ML training is not implemented
