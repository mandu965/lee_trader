# Production V1 Release Note

- release_package: `production_v1`
- released_at: `2026-03-31`
- config: [`config/production_v1.yaml`](/d:/ai/Lee_trader/config/production_v1.yaml)
- manifest: [`outputs/production_v1_manifest.json`](/d:/ai/Lee_trader/outputs/production_v1_manifest.json)

## Freeze Scope

This release freezes the current operational scoring and approval stack so future operational performance accumulates under one stable logic baseline.

- `score_formula_version`: `ranking_builder_v8_return_prob_tech_regime`
- `candidate_selection_version`: `buy_candidate_builder_v1`
- `gate_version`: `operational_buy_gate_v1`
- `portfolio_version`: `model_portfolio_constructor_v1`
- `confidence_calibration_version`: `confidence_four_axis_v1`

## Identified Operational Paths

- Score formula: [`python/ranking_builder.py`](/d:/ai/Lee_trader/python/ranking_builder.py)
  Baseline regime-aware `final_score` with `ret_score`, `prob_score`, `tech_score`, `qual_score`, `valuation_score`, and `risk_penalty`.
- Buy gate: [`python/build_operational_buy_gate.py`](/d:/ai/Lee_trader/python/build_operational_buy_gate.py)
  Final operational approval rule over benchmark maturity, calibration reliability, liquidity, concentration, and overheat checks.
- Portfolio construction: [`python/portfolio_constructor.py`](/d:/ai/Lee_trader/python/portfolio_constructor.py)
  Top5/Top8/Top10 allocation with confidence-weighted sizing, liquidity haircut, turnover keep-slots, sector/theme caps, and cash buffer.
- Confidence calibration:
  - [`python/calibrate_operational_confidence.py`](/d:/ai/Lee_trader/python/calibrate_operational_confidence.py)
  - [`python/build_confidence_calibration_map.py`](/d:/ai/Lee_trader/python/build_confidence_calibration_map.py)

## What Changed

- Added a production-only config snapshot in [`config/production_v1.yaml`](/d:/ai/Lee_trader/config/production_v1.yaml).
- Added shared config loader [`python/production_config.py`](/d:/ai/Lee_trader/python/production_config.py).
- Wired operational defaults for:
  - ranking formula and confidence version
  - buy candidate thresholds
  - portfolio construction thresholds
  - buy gate thresholds
  - confidence calibration sample thresholds
- Forced experimental ranking sidecars off in operational runtime:
  - theme overlay operational defaults off
  - theme risk soft experiment off
  - risk curve experiment off
  - feature-candidate experiment off

## Recording Policy

- `ranking_builder.py` writes `score_formula_version` and `confidence_version` into ranking output.
- `buy_candidate_builder.py` writes `score_formula_version` and `candidate_selection_version` into candidate outputs.
- `portfolio_constructor.py` writes `portfolio_version` and `score_formula_version` into model portfolio outputs.
- `build_operational_buy_gate.py` writes `gate_version`, `score_formula_version`, and `portfolio_version` into gate JSON/report output.
- Confidence calibration outputs now include `confidence_calibration_version`.

## Operational / Research Separation

- Default runtime mode is `operational` when `production_v1.yaml` is present.
- In operational mode, frozen production versions are used as authority.
- Experimental ranking features are disabled by default in operational mode.
- Research runs can still opt into alternate behavior by switching runtime mode with `LEE_TRADER_RUNTIME_MODE=research`.

## Remaining Boundary

- This release freezes the currently identified operational logic. It does not remove research scripts or experimental codepaths.
- If the production formula, gate, or allocation logic changes in the future, a new release package and version bump are required.
