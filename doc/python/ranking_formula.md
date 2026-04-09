# Ranking Formula

## Purpose

This document explains the current ranking score construction used by:

- [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)
- [`python/scoring/final_score.py`](/d:/ai/Lee_trader/python/scoring/final_score.py)

It is intended as the human-readable summary of the live production logic.

## What `final_score` Means

`final_score` is not a direct expected return percentage.

It is:

- a same-date cross-sectional ranking score
- a weighted combination of production operating signals
- a score used to compare candidates against each other on the same date

## Production Operating Axes

The current production operating `final_score` uses:

- `ret_score`
- `prob_score`
- `tech_score`
- `qual_score`
- `risk_penalty`

`risk_penalty` is a deduction term.

## Production Inputs

| file | role | main fields |
| --- | --- | --- |
| `data/predictions.csv` | model outputs | `pred_return_60d`, `pred_return_90d`, `pred_mdd_60d`, `pred_mdd_90d`, `prob_top20_60d`, `prob_top20_90d` |
| `data/scores_final.csv` | technical score source | `composite`, `score_score` |
| `data/features.csv` | quality, volatility, liquidity, technical inputs | `quality_score`, `vol_20`, `vol_60`, `vol_ma_20`, `volume`, `mom_20`, `close_over_ma20`, `rsi_14` |
| `data/universe.csv` | metadata | `code`, `name`, `market`, `sector` |
| `data/market_status.csv` | market regime metadata | `market_up`, `kospi_close`, `kospi_ma20`, `volatility_5d`, `foreign_net_5d` |

## Component Definitions

### 1. `ret_score`

- source: `pred_return_60d`, `pred_return_90d`
- purpose: primary production prediction score
- operating definition:

```text
ret_rank_60 = percentile01(pred_return_60d by date)
ret_rank_90 = percentile01(pred_return_90d by date)
ret_score   = 100 * (0.7 * ret_rank_60 + 0.3 * ret_rank_90)
```

### 2. `prob_score`

- source: `prob_top20_60d`
- purpose: operating probability score
- policy:
  - `prob_top20_60d` is the only direct production probability input
  - `prob_top20_90d` is stored for research / diagnostics only

operating fields:

```text
prob_score_raw = clip(prob_top20_60d * 100, 0, 100)
prob_score     = percentile01(prob_top20_60d by date) * 100
```

### 3. `tech_score`

- source: `composite` or `score_score`
- fallback: feature-based technical score if legacy technical source is weak or missing
- purpose: technical / chart strength

### 4. `qual_score`

- source: `quality_score`
- purpose: financial quality rank

```text
qual_score = percentile(quality_score by date)
```

Current quality factor definition:

| factor | direction | weight |
| --- | ---: | ---: |
| `roe` | `+1.0` | `0.25` |
| `op_margin` | `+1.0` | `0.20` |
| `net_margin` | `+1.0` | `0.20` |
| `debt_ratio` | `-1.0` | `0.15` |
| `ocf_to_assets` | `+1.0` | `0.20` |

Notes:

- `current_ratio` is not part of the current production quality factor set.
- `quality_factor_count`, `quality_missing_ratio`, and `quality_score_confidence` remain active metadata fields.

### 5. `risk_penalty`

- source: `pred_mdd_60d`, `pred_mdd_90d`
- purpose: drawdown deduction

```text
pred_mdd_mix = 0.6 * abs(pred_mdd_60d) + 0.4 * abs(pred_mdd_90d)
```

Then a capped piecewise penalty is applied.

## Regime Weighting

The production score is regime-aware.

| regime | ret | prob | tech | qual | risk_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bull` | `0.38` | `0.27` | `0.27` | `0.08` | `0.40` |
| `neutral` | `0.32` | `0.26` | `0.24` | `0.18` | `0.65` |
| `defensive` | `0.26` | `0.22` | `0.18` | `0.34` | `0.80` |

## Final Formula

```text
final_score =
    w_ret  * ret_score
  + w_prob * prob_score
  + w_tech * tech_score
  + w_qual * qual_score
  - w_risk_penalty * risk_penalty
```

Then:

```text
final_score = clip(final_score, 0, 100)
```

## Directly Computed But Not Direct Operating Axes

The following may still exist as compatibility or diagnostic fields, but they are not direct positive axes in the current production operating `final_score`:

- `valuation_score`
- `safety_score`
- `liquidity_score`

Interpretation:

- they may still be useful for monitoring, explain output, or future experimentation
- they do not directly raise the current production `final_score`

## Market Fallback Policy

`market_status.csv` is used to attach market-state metadata and regime context.

If market status data is missing or broken:

- `market_up` falls back to `False`
- fallback reason is recorded
- downstream handling remains conservative

This is intentionally safer than allowing missing market-state data to drift toward an aggressive fallback.

## Separate Note: Future / Research Topics

The following are not part of the current production operating formula:

- direct use of `prob_top20_90d`
- direct promotion of `valuation_score` into the operating score
- direct promotion of `safety_score`
- direct promotion of `liquidity_score`
