# Final Score Operating Formula

## Purpose

This document defines the current production `final_score` formula and the inputs that feed into it.

Live code paths:

- [`python/scoring/final_score.py`](/d:/ai/Lee_trader/python/scoring/final_score.py)
- [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)

## What `final_score` Means

`final_score` is not a direct expected return percentage. It is:

- a same-date cross-sectional ranking score
- a weighted combination of production operating signals
- used to compare candidates against each other on the same date

## Production Inputs

| File | Role | Main Fields |
| --- | --- | --- |
| `data/predictions.csv` | model outputs (required) | `pred_return_60d`, `pred_return_90d`, `pred_mdd_60d`, `pred_mdd_90d`, `prob_top20_60d`, `prob_top20_90d` |
| `data/predictions.csv` | model outputs (optional, 2026-05-22 추가) | `pred_return_30d`, `pred_mdd_30d`, `prob_top20_30d` — 모델 재학습 후 공급됨, 공식 미반영 시 null 허용 |
| `data/scores_final.csv` | technical score source | `composite`, `score_score` |
| `data/features.csv` | quality, volatility, liquidity, technical inputs | `quality_score`, `vol_20`, `vol_60`, `vol_ma_20`, `volume`, `mom_20`, `close_over_ma20`, `rsi_14` |
| `data/universe.csv` | metadata | `code`, `name`, `market`, `sector` |
| `data/market_status.csv` | market regime metadata | `market_up`, `kospi_close`, `kospi_ma20`, `volatility_5d`, `foreign_net_5d` |

> `pred_return_30d`는 `AI_MAX_HOLDING_DAYS=30` 정합을 위해 학습 타겟에 추가됐다 (KR-A, 2026-05-22).  
> `ret_score` 공식은 현재 60d/90d 기반을 유지하며, 30d 반영은 서버 재학습 + OOS 검증 후 별도 결정한다.

## Production Operating Axes

The current `final_score` uses five components:

| Component | Role |
| --- | --- |
| `ret_score` | primary return signal (positive) |
| `prob_score` | operating probability signal (positive) |
| `tech_score` | technical / chart strength (positive) |
| `qual_score` | financial quality rank (positive) |
| `risk_penalty` | drawdown deduction (negative) |

`valuation_score`, `safety_score`, `liquidity_score` — these exist as compatibility / diagnostic fields but are **not** direct positive axes in the current production formula.

## Official Policy

- `final_score` is a cross-sectional operating ranking score, not an expected return percentage.
- Positive axes: `ret_score`, `prob_score`, `tech_score`, `qual_score`
- Deduction axis: `risk_penalty`

---

## Component Definitions

### 1. `ret_score`

```text
ret_rank_60 = percentile01(pred_return_60d by date)
ret_rank_90 = percentile01(pred_return_90d by date)
ret_score   = 100 × (0.7 × ret_rank_60 + 0.3 × ret_rank_90)
```

### 2. `prob_score`

```text
prob_score_raw = clip(prob_top20_60d × 100, 0, 100)
prob_score     = percentile01(prob_top20_60d by date) × 100
```

- `prob_top20_60d`가 현재 운영 확률 입력의 기준이다.
- `prob_top20_90d`는 운영 점수에서 제외됐으며 연구/진단 보조 컬럼으로만 유지 (KR-B, 2026-05-22).
- `prob_top20_30d`는 옵셔널 입력으로 준비됐으나 현재 공식에는 미반영.

### 3. `tech_score`

- Source: `composite` or `score_score`
- Fallback: feature-based technical score if legacy source is weak or missing.

### 4. `qual_score`

```text
qual_score = percentile(quality_score by date)
```

Current quality factor definition:

| Factor | Direction | Weight |
| --- | ---: | ---: |
| `roe` | +1.0 | 0.25 |
| `op_margin` | +1.0 | 0.20 |
| `net_margin` | +1.0 | 0.20 |
| `debt_ratio` | −1.0 | 0.15 |
| `ocf_to_assets` | +1.0 | 0.20 |

Note: `current_ratio` is not part of the current production quality factor set.

### 5. `risk_penalty`

```text
pred_mdd_mix = 0.6 × |pred_mdd_60d| + 0.4 × |pred_mdd_90d|
```

Capped piecewise deduction:

```text
if pred_mdd_mix ≤ 0.10:
    risk_penalty = 0.0
elif pred_mdd_mix ≤ 0.15:
    risk_penalty = (pred_mdd_mix - 0.10) × 40.0
elif pred_mdd_mix ≤ 0.20:
    risk_penalty = 2.0 + (pred_mdd_mix - 0.15) × 80.0
else:
    risk_penalty = 6.0 + (pred_mdd_mix - 0.20) × 120.0

risk_penalty = clip(risk_penalty, 0, 18)
```

---

## Final Formula

### Intermediate

```text
final_score_raw =
    w_ret  × ret_score
  + w_prob × prob_score
  + w_tech × tech_score
  + w_qual × qual_score
```

### Final

```text
final_score = clip(final_score_raw - w_risk_penalty × risk_penalty, 0, 100)
```

---

## Regime Weight Profiles

| Regime | ret | prob | tech | qual | risk_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bull` | 0.38 | 0.27 | 0.27 | 0.08 | 0.40 |
| `neutral` | 0.32 | 0.26 | 0.24 | 0.18 | 0.65 |
| `defensive` | 0.26 | 0.22 | 0.18 | 0.34 | 0.80 |

- Positive-axis weights sum to `1.00` within each regime.
- `risk_penalty` is a deduction term, not a positive component.

---

## Market Fallback Policy

`market_status.csv` is used for regime metadata.  
If the file is missing, unreadable, empty, or missing `market_up`:

- `market_up` falls back to `False`
- Fallback reason is recorded in `market_info`
- Downstream regime handling remains conservative rather than aggressive

This prevents missing market-state data from accidentally biasing the score toward a bullish interpretation.

---

## Financial Momentum Overlay (Phase 7)

When `FINANCIAL_SCORE_OVERLAY_ENABLED=1`:

- `final_score` is replaced by `shadow_fin_final_score` (= `final_score + fin_momentum_adj`)
- `fin_momentum_adj`: ACCELERATING +5 ~ DECLINING −10, hard_risk −15
- See [`score_column_definitions.md`](/d:/ai/Lee_trader/doc/score_column_definitions.md) for full overlay column list.

---

## Future / Research Topics (Not Current Production)

- Blending `prob_top20_60d` and `prob_top20_90d`
- Promoting `valuation_score` into a direct production axis
- Promoting `safety_score` or `liquidity_score` into direct production axes
