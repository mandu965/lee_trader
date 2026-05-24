---
title: 점수는 좋아도 진입하지 않는 경우
slug: case-study-good-score-but-no-entry
category: 운영 사례
excerpt: 상위권 후보라도 확률, 변동성, 검증 상태가 약하면 관찰만 하고 즉시 진입하지 않는 것이 더 합리적일 수 있습니다.
date: 2026-05-23
featured: true
---
랭킹 상위에 오른 종목이 항상 "지금 사도 되는 종목"은 아닙니다. 실제 운영에서는 final score가 높더라도 확률, 예상 MDD, market regime, walk-forward 상태를 함께 봅니다.

## 왜 상위권인데도 멈추나

- final score는 상대 비교이지만 행동 판단은 절대 기준에 가깝습니다.
- 기대수익이 높아 보여도 상위권 진입 확률이 낮으면 보수적으로 해석합니다.
- 예상 MDD가 크면 수익보다 버티기 어려운 구간일 수 있습니다.
- walk-forward acceptance가 약하면 운영 승격을 늦춥니다.

## 실제 읽는 순서

1. 종목 점수로 후보군을 좁힙니다.
2. buy eligibility와 watchlist 상태를 확인합니다.
3. market regime와 gate를 봅니다.
4. 마지막으로 walk-forward와 confidence를 확인합니다.

이 과정을 거치면 "좋아 보이지만 지금은 아니다"라는 결론이 자주 나옵니다. 그 자체가 보수적인 운영의 핵심입니다.

행동 기준 차이는 [WATCHLIST와 BUY_ALLOWED의 차이](/blog/difference-between-watchlist-and-buy-allowed)와 함께 읽으면 더 잘 연결됩니다.
