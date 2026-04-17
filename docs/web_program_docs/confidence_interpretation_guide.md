# Confidence Interpretation Guide

- generated_at: 2026-03-29 23:49:48
- source: operational confidence calibration table
- 5d_monotonicity_check: fail

## What Confidence Means

Confidence is not expected return. It is a reliability signal. It describes how much trust the system has in the current recommendation, based on data quality, model reliability, signal agreement, and fit with the current market regime.

## Practical Interpretation

- `80-100`: strong internal conviction, but still not a guarantee.
- `60-80`: usable signal, but current evidence should be checked against liquidity and market conditions.
- Below `60`: usually not suitable for operational buy lists unless there is a special reason.

## Current Operational Calibration Reality

The live calibration dataset is still small. On the 5-day horizon, the current operational history does not show a clean monotonic relationship between higher confidence and better realized hit rate.

| bucket | sample_rows | avg_raw_confidence | realized_hit_rate | operational_calibrated | bucket_status |
| ------ | ----------- | ------------------ | ----------------- | ---------------------- | ------------- |
| 0-20   | 0           | NA                 | NA                | NA                     | empty         |
| 20-40  | 0           | NA                 | NA                | NA                     | empty         |
| 40-60  | 0           | NA                 | NA                | NA                     | empty         |
| 60-80  | 154         | 73.81              | 29.22%            | 29.22                  | stable        |
| 80-100 | 49          | 84.22              | 20.41%            | 20.41                  | stable        |

## Current User Guidance

- Treat confidence as a supporting indicator, not as standalone buy permission.
- High confidence still requires buy-gate approval and acceptable liquidity.
- Because operational monotonicity is not stable yet, confidence is currently best read as provisional rather than fully calibrated.
