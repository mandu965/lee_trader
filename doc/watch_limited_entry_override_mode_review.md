# watch_limited_entry_override_mode Review

## Current Setting

- Source: [config/production_v1.yaml](/d:/ai/lee_trader/config/production_v1.yaml:1)
- Current value: `execution_policy.watch_limited_entry_override_mode: pilot`

## Current Behavior

- Official gate source can remain `WATCH` in `outputs/operational_buy_gate.json`.
- Runtime execution policy upgrades that `WATCH` state to `PILOT` for trade intent generation.
- Result:
  - `outputs/operational_buy_gate.json` may say `WATCH`
  - `outputs/trade_intents.json` may say `PILOT`
  - `outputs/live_order_preview.json` then needs both source and runtime status to explain the mismatch

## Why This Is Confusing

- Operators may think the pipeline is inconsistent or broken.
- Web pages can show a single gate label without explaining that runtime override logic changed the effective trading mode.
- Incident review becomes harder because the source policy and runtime policy are mixed together.

## If We Keep It

- Pros:
  - Preserves current live execution behavior
  - Keeps limited-entry pilot interpretation during `WATCH`
  - No strategy behavior change
- Cons:
  - `WATCH` and `PILOT` continue to coexist across artifacts
  - Requires explicit UI/report explanation everywhere operators look

## If We Remove It

- Expected change:
  - Runtime gate would remain `WATCH`
  - `trade_intents.json` and preview/runtime displays would align more directly with `operational_buy_gate.json`
- Operational impact:
  - Restricted BUY behavior under current `WATCH -> PILOT` interpretation could become more conservative
  - `gate_guidance`, trade intent labeling, and operator interpretation would change
  - Any path depending on `limited_entry_mode == "pilot"` should be rechecked before rollout

## Recommendation

1. Keep current runtime behavior for now.
2. Expose both `gate_source_status` and `gate_runtime_status` in web/report outputs.
3. Review a few recent production days and compare:
   - source gate
   - runtime gate
   - BUY intents created
   - preview BUY count
   - actual submitted BUY count
4. Only then decide whether the override should remain policy, be renamed for clarity, or be removed.
