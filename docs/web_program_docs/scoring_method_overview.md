# Scoring Method Overview

- generated_at: 2026-03-29 23:49:48
- latest_snapshot_date: 2026-03-27
- latest_detected_regime: defensive

## What The Score Means

The system builds one final recommendation score for each stock. This score is not a single forecast. It combines expected return, the chance of ranking near the top, technical trend quality, business quality, valuation support, and a risk penalty for names expected to suffer larger drawdowns.

In plain language, a higher score means the stock looks better on several dimensions at the same time, not just on one strong signal.

## Main Inputs

- `ret_score`: expected return signal from the production model.
- `prob_score`: how often the stock looks like a likely top-ranked candidate versus peers on the same date.
- `tech_score`: trend, momentum, stability, and volume-based technical quality.
- `quality_score`: financial quality inputs such as profitability and balance-sheet strength.
- `valuation_score`: valuation support when cheaper fundamentals improve the setup.
- `risk_penalty`: a deduction when predicted drawdown risk is high.

## Regime-Based Weights

The system changes weights depending on market regime. In stronger markets it leans more on return and technical momentum. In defensive markets it gives more room to quality, valuation, and drawdown control.

| regime    | ret    | prob   | tech   | quality | valuation | risk_penalty_strength |
| --------- | ------ | ------ | ------ | ------- | --------- | --------------------- |
| bull      | 35.00% | 26.00% | 29.00% | 6.00%   | 4.00%     | 0.40                  |
| neutral   | 30.00% | 26.00% | 26.00% | 10.00%  | 8.00%     | 0.65                  |
| defensive | 27.00% | 24.00% | 15.00% | 19.00%  | 15.00%    | 0.80                  |

## How To Read It

- A high score with high confidence is stronger than a high score with weak confidence.
- A stock can still rank well in a defensive regime even if its momentum is not the strongest, as long as quality and risk are better.
- The score is relative within each date. It is most useful for comparing candidates against each other, not as a standalone promise of future return.
