# Lee_trader_rule

## Purpose

이 모듈은 RULE 기반 자동매매 운영 흐름을 정리합니다.

범위:
- RULE 신호 생성
- RULE 백테스트 요약
- RULE 포트폴리오 계획
- RULE 주문 preview
- paper / pilot / live 실행
- RULE 계좌 동기화
- RULE 운영 화면 payload 생성

## Main Files

- `python/run_rule_after_close_cycle.py`
- `python/run_rule_before_open_cycle.py`
- `python/run_rule_after_open_cycle.py`
- `python/rule_signal_builder.py`
- `python/rule_portfolio_manager.py`
- `python/rule_order_preview_builder.py`
- `python/rule_order_submitter.py`
- `python/rule_execution_simulator.py`
- `python/rule_live_account_snapshot.py`
- `python/build_rule_web_payloads.py`

## Main Outputs

- `data/rule_signals.csv`
- `outputs/rule_strategy_backtest_report.json`
- `outputs/rule_portfolio_plan.json`
- `outputs/rule_order_preview.json`
- `outputs/rule_execution_results.json`
- `outputs/rule_account_paper_state.json`
- `outputs/rule_account_live_state.json`
- `outputs/rule_dashboard_summary.json`

## Read First

- [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/CONTEXT.md>)
- [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/FLOW.md>)
- [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/FILE_INDEX.md>)
- [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/ENV.md>)
- [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/OPERATIONS.md>)
