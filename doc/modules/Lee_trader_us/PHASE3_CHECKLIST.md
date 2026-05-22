# Phase 3 Checklist

> 상태 메모: 2026-05-22 기준 이 문서는 ranking v1 구축 당시의 short-form checklist다.  
> 현재는 closure checklist 자체보다 ranking / backtest / paper lifecycle 검증 문맥에서 참고용으로 사용한다.

This is the short-form Phase 3 closure checklist for Project C US stock ranking v1.

## Checklist

```text
DB
- [ ] meta.us_stock_universe exists
- [ ] recommend.us_stock_rank_daily exists
- [ ] ranking indexes exist

Data
- [ ] active universe rows exist
- [ ] effective-date price features exist
- [ ] effective-date financial features exist
- [ ] effective-date relative strength features exist

Calculation
- [ ] Rule scorer runs
- [ ] total_score stays within 0..100
- [ ] risk_score stays within -10..0
- [ ] rank_no is assigned
- [ ] recommend_grade is assigned

Reporting
- [ ] Top 20 console output runs
- [ ] markdown report runs
- [ ] csv report runs
- [ ] symbol detail output runs
- [ ] excluded-row output runs

Validation
- [ ] validate script runs
- [ ] score_detail_json parses
- [ ] EXCLUDE rows have exclude_reason
- [ ] warning/error summary renders

Operations
- [ ] ENV doc updated
- [ ] execution order documented
- [ ] troubleshooting documented
- [ ] separation from auto-trading documented
```

For detailed guidance, use [US_STOCK_RANKING_V1.md](/d:/ai/lee_trader/doc/modules/Lee_trader_us/US_STOCK_RANKING_V1.md).
