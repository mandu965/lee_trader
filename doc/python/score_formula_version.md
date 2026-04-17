# Score Formula Version

## Purpose

`score_formula_version` identifies which `final_score` formula generated a ranking result.

It is used for:

- ranking output traceability
- `research.dim_model_run.config_json` metadata
- walk-forward comparison grouping
- safe comparison after formula changes

## Default

- Default value: `ranking_builder_v1`
- Environment variable override: `SCORE_FORMULA_VERSION`

If the value is missing in older data, the project must tolerate `NULL` or an absent key.

## Naming Rule

Use a stable lowercase identifier with a clear intent and version suffix.

Recommended pattern:

```text
<family>_<focus>_v<major>
```

Examples:

- `ranking_builder_v1`
- `ranking_formula_pred_v1`
- `ranking_formula_pred_quality_v1`
- `ranking_formula_pred_quality_tech_v1`

## Version Bump Rule

Increase the version when the effective ranking result can change because of formula logic.

Version bump required:

- component added or removed
- weight changed
- penalty rule changed
- regime branch logic changed
- normalization rule changed

Version bump not required:

- comments only
- refactor with identical output
- logging changes

## Storage Rule

- Daily ranking output: store `score_formula_version` as a nullable column
- `research.dim_model_run.config_json`: store `"score_formula_version": "<value>"`
- walk-forward reports: expose the value from `config_json`

## Compatibility Rule

Older rows may not have the value.

Handling:

- database column may be nullable
- `config_json` key may be absent
- reports must treat missing values as blank or `NA`
