---
title: walk-forward acceptance는 무엇을 말해 주나
slug: what-walkforward-acceptance-means
category: 검증 지표
excerpt: backtest 숫자 하나보다, 앞으로도 비슷하게 작동할 가능성을 보는 검증 상태가 walk-forward acceptance입니다.
date: 2026-05-23
featured: true
---
walk-forward acceptance는 "과거 한 번 잘됐다"보다 "앞으로도 비슷한 구조가 반복될 수 있나"를 확인하는 검증 단계입니다. 점수 모델이 좋아 보여도 이 검증이 약하면 실전 승격을 서두르지 않습니다.

## 무엇을 보나

- top20이 top50, 전체 유니버스보다 일관되게 더 나은가
- 평균 수익만이 아니라 MDD도 감당 가능한가
- confidence가 높을수록 성과가 더 좋아지는 단조성이 있는가
- 체결 증거가 있거나, 없어도 차단 사유가 없는가

## 현재처럼 REJECTED일 때 의미

REJECTED는 "모델이 쓸모없다"는 뜻이 아닙니다. 다만 아직 BUY_ALLOWED 승격을 뒷받침할 만큼 정렬력과 검증 강도가 충분하지 않다는 뜻입니다.

이 상태에서는 상위 후보를 아예 버리는 것이 아니라, WATCH 단계에서 관찰과 비교 기록을 더 쌓는 쪽이 맞습니다. 즉 해석은 가능하지만 행동 강도는 낮춰야 합니다.

방법론 전체 흐름은 [방법론](/methodology), 실제 행동 차이는 [WATCHLIST와 BUY_ALLOWED의 차이](/blog/difference-between-watchlist-and-buy-allowed)에서 이어서 볼 수 있습니다.
