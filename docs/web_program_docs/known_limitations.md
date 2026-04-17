# Known Limitations

- generated_at: 2026-03-29 23:49:48
- current_buy_gate_status: HOLD

## Current Limitations

- Operational forward-return history is still too short. The current archive has only one latest snapshot date for the newest cycle, so all main horizons are still immature.
- Benchmark outperformance is not yet proven. The buy gate remains on HOLD because mature benchmark dates are still below the required threshold.
- Confidence calibration is not yet trustworthy enough for live use. On the 5-day horizon, higher confidence has not shown a clean monotonic improvement in realized hit rate.
- Theme diversification is currently weak. The latest model portfolio still allocates about 95.00% to `(none)` theme names because alternative theme coverage is limited.
- Top8 and Top10 lists carry more liquidity risk than Top5. The current operational gate blocks them because the very-low-liquidity ratio is too high.
- KOSDAQ benchmark comparison is not available in the local dataset, so cross-market benchmark review is incomplete.
- Paper trading is still in the early stage. Current portfolios have no closed 20-day trades yet, so realized win rate is not meaningful.

## Supporting Detail

Stable 5d confidence buckets:

| bucket_label | rows | hit_rate |
| ------------ | ---- | -------- |
| 60-80        | 154  | 29.22%   |
| 80-100       | 49   | 20.41%   |

These limitations do not mean the system is unusable. They mean the current state is appropriate for monitored operation, paper trading, and continued evidence collection rather than automatic live deployment.
