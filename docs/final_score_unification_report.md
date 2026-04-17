# final_score Unification Report

## Summary

- Created shared scorer module: `python/scoring/final_score.py`.
- `ranking_builder.py` baseline `final_score` now reuses the shared scorer.
- `build_backtest_predictions.py` now merges predictions with feature/market inputs and reuses the same shared scorer.
- Theme overlays remain in `ranking_builder.py`; only the baseline production `final_score` path was unified.

## Shared Input Schema

- Required core inputs: `date`, `code`, `pred_return_60d`, `pred_return_90d`, `prob_top20_60d`, `pred_mdd_60d`, `pred_mdd_90d`.
- Optional feature inputs: `quality_score`, technical features (`close`, `ma_*`, `ret_*`, `mom_20`, `rsi_14`, `vol_*`, `volume`, `composite`, `score_score`), valuation features (`per`, `pbr`, `earnings_yield`, etc.), and `regime`.
- Explain outputs: contribution columns (`contrib_ret`, `contrib_prob`, `contrib_tech`, `contrib_qual`, `contrib_valuation`, `contrib_penalty`) plus `score_contribution_*` aliases and `final_score_raw`.

## Validation

- `py_compile`: passed for `python/scoring/final_score.py`, `python/build_backtest_predictions.py`, `python/ranking_builder.py`.
- Sample size: 100 rows from `data/predictions.csv` joined with `data/features.csv`.

### Production Legacy vs Shared

- mean abs diff: 0.0000000000
- max abs diff: 0.0000000000
- non-zero diff count (>1e-9): 0

Interpretation: shared scorer preserves the previous production baseline formula on the sampled rows.

### Backtest Legacy vs Shared

- mean abs diff: 19.945548
- median abs diff: 18.593715
- max abs diff: 45.186525
- changed row count (>1e-9): 100 / 100
- new regime distribution: {'defensive': 100}
- rows with non-zero `qual_score` under shared scorer: 96
- rows with non-zero `tech_score` under shared scorer: 100
- shared scorer mean `valuation_score`: 50.0000

## Diff Cause Summary

- Backtest diffs are expected and intentional. The old backtest scorer was not production-parity logic.
- `ret_score` changed from a 30/60/90 return plus MDD z-score formula to the production 60/90 percentile blend.
- `prob_score` changed from absolute probability scaling to production same-date percentile scaling on `prob_top20_60d`.
- `risk_penalty` changed from multiplicative `0.5~1.0` scaling to the production piecewise absolute deduction curve.
- Shared scoring now incorporates production `tech_score`, `qual_score`, `valuation_score`, and regime weight logic when the inputs exist.
- The old backtest bias constant (`0.10 * 60`) was removed because it does not exist in production baseline scoring.

## Largest Backtest Diffs

| date | code | legacy_final | shared_final | diff | ret_diff | prob_diff | qual_diff | tech_diff | risk_penalty_diff | regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-27 | 003550 | 26.223800 | 71.410325 | 45.186525 | 25.874028 | 59.388046 | 94.791667 | 53.370437 | 5.101489 | defensive |
| 2026-03-27 | 030200 | 33.688803 | 74.346220 | 40.657417 | 18.272524 | 64.963977 | 62.500000 | 71.284207 | 5.421459 | defensive |
| 2026-03-27 | 006730 | 35.134795 | 75.627304 | 40.492509 | 25.746264 | 50.108601 | 84.375000 | 45.068532 | 5.079089 | defensive |
| 2026-03-27 | 051910 | 25.270044 | 62.321082 | 37.051038 | 22.073671 | 66.434584 | 55.208333 | 47.418425 | 6.426386 | defensive |
| 2026-03-27 | 016360 | 21.985457 | 58.858600 | 36.873143 | 3.753093 | 54.046771 | 85.416667 | 63.288643 | 6.846476 | defensive |

## Notes

- `ranking_builder.py` still owns theme overlay outputs such as `final_score_v2` and `final_score_v3`.
- The shared module is now the single source of truth for baseline production-style component scoring, regime-weighted baseline `final_score`, and explain contributions.