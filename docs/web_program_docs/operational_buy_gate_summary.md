# Operational Buy Gate Summary

- generated_at: 2026-03-29 23:49:48
- asof_date: 2026-03-27
- overall_status: HOLD
- primary_bucket: top5
- daily_cycle_status: WAIT

## What This Gate Does

The buy gate is the final operational safety layer. It prevents the system from automatically approving a live buy list when benchmark outperformance has not been confirmed, confidence calibration is still weak, or liquidity risk is too high.

## Latest Decision Snapshot

| bucket | status | avg_final_score | avg_confidence_score | liquidity_risk_ratio | matured_benchmark_dates | confidence_reliable | reason                                                                                                        |
| ------ | ------ | --------------- | -------------------- | -------------------- | ----------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------- |
| top5   | HOLD   | 71.20           | 88.06                | 0.00%                | 0                       | False               | matured benchmark dates 0 are below required 3; operational confidence calibration is not reliable enough yet |
| top8   | BLOCK  | 67.94           | 86.81                | 25.00%               | 0                       | False               | very low liquidity ratio 25.00% exceeds 20.00%                                                                |
| top10  | BLOCK  | 66.58           | 86.01                | 20.00%               | 0                       | False               | very low liquidity ratio 20.00% exceeds 20.00%                                                                |

## How To Read The Status

- `BUY_ALLOWED`: live evidence is strong enough to support buying.
- `WATCH`: some positive evidence exists, but it is not strong enough for automatic approval.
- `HOLD`: the current list may still be interesting, but the system does not yet have enough mature evidence.
- `BLOCK`: risk conditions are strong enough that the list should not be used for buying.

## Current Message

The current system status is `HOLD`. For the primary `top5` bucket, the main blocker is lack of mature benchmark evidence and insufficiently reliable operational confidence calibration.
