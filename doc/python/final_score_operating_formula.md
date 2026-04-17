# Final Score Operating Formula

## Purpose

This document defines the current production operating `final_score` formula.
It reflects the live code path in:

- [`python/scoring/final_score.py`](/d:/ai/Lee_trader/python/scoring/final_score.py)
- [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)

## Official Policy

- `final_score` is a cross-sectional operating ranking score, not an expected return percentage.
- The production positive axes are:
  - `ret_score`
  - `prob_score`
  - `tech_score`
  - `qual_score`
- The production deduction axis is:
  - `risk_penalty`

## Final Formula

### Final score raw

```text
final_score_raw =
    w_ret  * ret_score
  + w_prob * prob_score
  + w_tech * tech_score
  + w_qual * qual_score
```

### Final score

```text
final_score = clip(final_score_raw - w_risk_penalty * risk_penalty, 0, 100)
```

## Regime Weight Profiles

| regime | ret | prob | tech | qual | risk_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bull` | `0.38` | `0.27` | `0.27` | `0.08` | `0.40` |
| `neutral` | `0.32` | `0.26` | `0.24` | `0.18` | `0.65` |
| `defensive` | `0.26` | `0.22` | `0.18` | `0.34` | `0.80` |

Notes:

- Positive-axis weights sum to `1.00` within each regime.
- `risk_penalty` is a deduction term, not a positive component.

## Input Components

| component | source columns | role | operating use |
| --- | --- | --- | --- |
| `ret_score` | `pred_return_60d`, `pred_return_90d` | primary return signal | direct |
| `prob_score` | `prob_top20_60d` | operating probability signal | direct |
| `tech_score` | `composite` / `score_score` fallback or feature-based tech | technical strength | direct |
| `qual_score` | `quality_score` | financial quality | direct |
| `risk_penalty` | `pred_mdd_60d`, `pred_mdd_90d` | drawdown deduction | direct |
| `valuation_score` | valuation inputs if present | diagnostic / compatibility | not direct |
| `safety_score` | `vol_20`, `vol_60` | diagnostic / compatibility | not direct |
| `liquidity_score` | `vol_ma_20` or `volume` | diagnostic / compatibility | not direct |

## Probability Score Policy

- The operating `prob_score` uses `prob_top20_60d` only.
- `prob_score_raw` is the absolute 60-day probability converted to `0~100`.
- `prob_score` is the same-date relative rank of `prob_top20_60d`, also converted to `0~100`.
- `prob_top20_90d` may be stored and exposed for research or diagnostics, but it is not blended into the production operating probability axis.

## Quality Factor Policy

The current quality factor definition is:

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

## Risk Penalty Rule

### Mix

```text
pred_mdd_mix = 0.6 * abs(pred_mdd_60d) + 0.4 * abs(pred_mdd_90d)
```

### Deduction

```text
if pred_mdd_mix <= 0.10:
    risk_penalty = 0.0
elif pred_mdd_mix <= 0.15:
    risk_penalty = (pred_mdd_mix - 0.10) * 40.0
elif pred_mdd_mix <= 0.20:
    risk_penalty = 2.0 + (pred_mdd_mix - 0.15) * 80.0
else:
    risk_penalty = 6.0 + (pred_mdd_mix - 0.20) * 120.0
```

Final constraint:

```text
risk_penalty = clip(risk_penalty, 0, 18)
```

## Market Fallback Policy

`market_status.csv` is used for regime metadata. If the file is missing, unreadable, empty, or missing `market_up`, the production fallback is conservative:

- `market_up = False`
- fallback reason is recorded in `market_info`
- downstream regime handling remains conservative rather than aggressive

This policy prevents missing market-state data from accidentally biasing the operating score toward a bullish interpretation.

## Separate Note: Future Work

The following are not part of the current production formula and should be treated as future or research-only ideas:

- blending `prob_top20_60d` and `prob_top20_90d`
- promoting `valuation_score` into a direct production axis
- promoting `safety_score` or `liquidity_score` into direct production axes
