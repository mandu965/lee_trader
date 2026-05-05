# RULE Backtest Comparison

- compared_at: `2026-05-06`
- baseline_source: `user-provided Claude comparison summary`
- repo_current_source_json: [outputs/rule_portfolio_backtest_report.json](/d:/ai/lee_trader/outputs/rule_portfolio_backtest_report.json)
- repo_current_source_md: [outputs/rule_portfolio_backtest_report.md](/d:/ai/lee_trader/outputs/rule_portfolio_backtest_report.md)
- repo_trade_log: [outputs/rule_portfolio_backtest_trades.csv](/d:/ai/lee_trader/outputs/rule_portfolio_backtest_trades.csv)
- repo_equity_curve: [outputs/rule_portfolio_backtest_equity.csv](/d:/ai/lee_trader/outputs/rule_portfolio_backtest_equity.csv)

## Summary

이 문서는 RULE 기반 자동매매 3년 포트폴리오 백테스트의 `수정 전`과 `수정 후`를 비교하기 위한 운영 메모입니다.

중요:
- `수정 후` 수치는 현재 저장소 산출물에서 직접 확인됩니다.
- `수정 전` 수치는 현재 저장소 산출물에서 재생성한 값이 아니라, 사용자께서 제공한 Claude 비교표를 기준으로 기록합니다.
- 따라서 이 문서는 "운영 비교 기록"이며, 두 버전 모두를 같은 코드로 재현한 증적 문서는 아닙니다.

## Config Delta

| 항목 | 수정 전 | 수정 후 |
| --- | ---: | ---: |
| max_holding_days | 20일 | 10일 |
| stop_loss | 7% | 5% |
| trailing_stop | 5% | 4% |
| reduce 로직 | 없음 | 구현 |
| cascade reduce | 없음 | 방지 |

## Performance Delta

| 항목 | 수정 전 | 수정 후 |
| --- | ---: | ---: |
| 총 거래 수 | 235건 | 462건 |
| 총 수익률 | 46.54% | 29.46% |
| CAGR | 13.32% | 8.82% |
| MDD | -15.46% | -18.80% |
| Sharpe | 1.012 | 0.623 |
| 승률 | 43.4% | 51.7% |

## Current Repo Metrics

현재 저장소 기준 최신 산출물 [rule_portfolio_backtest_report.json](/d:/ai/lee_trader/outputs/rule_portfolio_backtest_report.json) 내용:

- 기간: `2023-04-14` ~ `2026-05-04`
- 초기자본: `10,000,000`
- 최종자본: `12,945,657`
- 총수익률: `29.46%`
- CAGR: `8.82%`
- MDD: `-18.80%`
- Sharpe: `0.6232`
- 총 거래 수: `462`
- 승률: `51.73%`
- 평균 수익률: `3.13%`
- 평균 이익: `10.08%`
- 평균 손실: `-4.37%`
- Payoff ratio: `2.3067`
- 평균 보유일: `15.4일`

연도별 수익률:

- `2023`: `-2.88%`
- `2024`: `-2.29%`
- `2025`: `+23.91%`
- `2026`: `+6.40%`

종료 사유 분포:

- `stop_loss`: `116`
- `max_holding_days_exit`: `73`
- `trailing_stop_reduce`: `99`
- `trailing_stop_exit`: `140`
- `max_holding_days_reduce`: `34`

## Interpretation

- 수정 후 결과는 운영 설정과 코드 로직이 일치하는 상태에 더 가깝습니다.
- 총수익률, CAGR, Sharpe는 낮아졌지만, 이전 결과의 낙관 편향이 제거되었을 가능성이 높습니다.
- 승률은 개선되었지만 MDD가 더 커졌기 때문에, 전략의 안정성이 개선되었다고 단정할 수는 없습니다.
- `2023~2024` 약세 구간에서 손절 빈도가 높고, `2025` 회복이 성과 대부분을 만회하는 구조입니다.
- 손실은 짧게 끊고 이익은 길게 가져가는 구조는 유지됩니다.

## Observations

- `stop_loss`가 `116건`으로 가장 많고, 약세장 구간 성과를 가장 크게 누르고 있습니다.
- `trailing_stop_exit`와 `trailing_stop_reduce`가 모두 존재하므로, 부분 축소 로직은 현재 백테스트에 반영된 상태입니다.
- payoff ratio가 `2.31` 수준이므로, 승률보다 손익비가 전략 유지의 핵심입니다.
- 전략의 실질 경쟁력은 `2023~2024` 약세장 필터 개선 여부에 크게 좌우됩니다.

## Follow-up

- `stop_loss` 116건 구간을 별도 샘플링해 진입 필터 개선 가능성을 검토합니다.
- `2023`, `2024` 손실 구간에서 sector/cooldown/gap/trading value 필터 강화 실험을 분리합니다.
- `reduce` 이후 잔여 포지션 성과와 `trailing_stop_exit` 성과를 분리 검증합니다.
- 새 비교가 생기면 이 문서에 `baseline_source`와 `repo_current_source_*`를 함께 갱신합니다.
