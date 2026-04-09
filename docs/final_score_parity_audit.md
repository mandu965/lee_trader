# final_score Parity Audit

## Scope

- Backtest scorer: `python/build_backtest_predictions.py`
- Production scorer: `python/ranking_builder.py`
- Audit target: `final_score` calculation parity, including upstream component construction actually feeding `final_score`

## Executive Verdict

결론은 **불일치**다.

- 두 파일 모두 `ret_score`, `prob_score`, `qual_score`, `tech_score`, `risk_penalty`, `final_score`라는 유사한 컬럼명을 사용한다.
- 하지만 실제 계산식은 입력 컬럼, 스케일링, 가중치, penalty 적용 방식, regime 분기, 보조 항목에서 구조적으로 다르다.
- 특히 backtest 경로는 `predict_for_date()` 결과를 바로 `compute_scores()`에 넣기 때문에 `quality_score`, `vol_ma_20`, `vol_ratio_20`, `vol_20`, `valuation_score`, `regime`가 공급되지 않는다. 실제 실행 시 backtest `final_score`는 대부분 `ret_score + prob_score + bias` 중심의 축약 점수로 동작한다 ([build_backtest_predictions.py:343](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L343), [build_backtest_predictions.py:344](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L344)).

## Code Paths

### Backtest path

- 예측 생성: [build_backtest_predictions.py:132-180](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L132)
- 점수 계산: [build_backtest_predictions.py:192-265](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L192)
- 호출 경로: [build_backtest_predictions.py:334-345](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L334)

### Production path

- component score 생성: [ranking_builder.py:1742-1947](/d:/ai/Lee_trader/python/ranking_builder.py#L1742)
- risk penalty 생성: [ranking_builder.py:2010-2032](/d:/ai/Lee_trader/python/ranking_builder.py#L2010)
- regime weight 결정: [ranking_builder.py:411-464](/d:/ai/Lee_trader/python/ranking_builder.py#L411), [ranking_builder.py:2035-2080](/d:/ai/Lee_trader/python/ranking_builder.py#L2035)
- regime 판정: [ranking_builder.py:1561-1633](/d:/ai/Lee_trader/python/ranking_builder.py#L1561)
- baseline `final_score` 계산: [ranking_builder.py:3070-3153](/d:/ai/Lee_trader/python/ranking_builder.py#L3070)
- theme-aware side scores: [ranking_builder.py:2294-2310](/d:/ai/Lee_trader/python/ranking_builder.py#L2294), [ranking_builder.py:2569-2576](/d:/ai/Lee_trader/python/ranking_builder.py#L2569)

## final_score Formula Extraction

### Backtest

`compute_scores()`의 실질 수식은 아래다.

```text
r_comb = mean(pred_return_30d, pred_return_60d, pred_return_90d)
pred_mdd_comb = mean(pred_mdd_30d, pred_mdd_60d, pred_mdd_90d)
r_adj = r_comb / (1 + 3 * abs(pred_mdd_comb))
ret_score = clip(50 + 10 * zscore(r_adj), 0, 100)

prob_score = clip(first_non_null(prob_top20_30d, prob_top20_60d, prob_top20_90d) * 100, 0, 100)
qual_score = percentile_by_date(quality_score) or NaN
tech_score = percentile_by_date(vol_ma_20 | vol_ratio_20 | vol_20) or NaN

risk_penalty = clip(1 - 0.5 * clip(mean(abs(pred_mdd_30d, pred_mdd_60d, pred_mdd_90d)) / 0.4, 0, 1), 0.5, 1.0)

base =
  0.40 * ret_score +
  0.25 * prob_score +
  0.15 * qual_score +
  0.10 * tech_score +
  0.10 * 60

final_score = clip(base * risk_penalty, 0, 100)
```

근거:

- `ret_score`: [build_backtest_predictions.py:218-228](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L218)
- `prob_score`: [build_backtest_predictions.py:230-232](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L230)
- `qual_score`: [build_backtest_predictions.py:234-238](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L234)
- `tech_score`: [build_backtest_predictions.py:240-248](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L240)
- `risk_penalty`: [build_backtest_predictions.py:250-253](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L250)
- `final_score`: [build_backtest_predictions.py:255-265](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L255)

### Production baseline

`apply_default_ranking_scores()`의 baseline `final_score`는 아래다.

```text
ret_score = 100 * (0.7 * percentile01(pred_return_60d) + 0.3 * percentile01(pred_return_90d))

prob_score =
  percentile01(prob_top20_60d) * 100
  with same-date median fill for missing rows
  and 50 fallback if the whole column is absent

qual_score = percentile_by_date(quality_score) or NaN
tech_score = legacy percentile(composite|score_score) or feature_v1 tech composite
valuation_score = percentile-based valuation sleeve or neutral 50

pred_mdd_mix = 0.6 * abs(pred_mdd_60d) + 0.4 * abs(pred_mdd_90d)
risk_penalty = piecewise absolute deduction in [0, 18]

final_score_before_theme =
  w_ret(regime) * ret_score +
  w_prob(regime) * prob_score +
  w_tech(regime) * tech_score +
  w_qual(regime) * qual_score +
  w_valuation(regime) * valuation_score -
  w_risk_penalty(regime) * risk_penalty

final_score = clip(final_score_before_theme, 0, 100)
```

근거:

- `ret_score`: [ranking_builder.py:1762-1784](/d:/ai/Lee_trader/python/ranking_builder.py#L1762)
- `prob_score`: [ranking_builder.py:1817-1833](/d:/ai/Lee_trader/python/ranking_builder.py#L1817)
- `qual_score`: [ranking_builder.py:1845-1850](/d:/ai/Lee_trader/python/ranking_builder.py#L1845)
- `tech_score`: [ranking_builder.py:1708-1740](/d:/ai/Lee_trader/python/ranking_builder.py#L1708), [ranking_builder.py:1393-1483](/d:/ai/Lee_trader/python/ranking_builder.py#L1393)
- `valuation_score`: [ranking_builder.py:1898-1943](/d:/ai/Lee_trader/python/ranking_builder.py#L1898)
- `risk_penalty`: [ranking_builder.py:2010-2032](/d:/ai/Lee_trader/python/ranking_builder.py#L2010)
- regime weights: [ranking_builder.py:424-464](/d:/ai/Lee_trader/python/ranking_builder.py#L424), [ranking_builder.py:2035-2080](/d:/ai/Lee_trader/python/ranking_builder.py#L2035)
- baseline `final_score`: [ranking_builder.py:3142-3153](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)

## Difference Matrix

| Aspect | Backtest | Production | Parity |
| --- | --- | --- | --- |
| `ret_score` input horizon | `pred_return_30d/60d/90d`와 `pred_mdd_30d/60d/90d`를 함께 사용 ([build_backtest_predictions.py:219-225](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L219)) | `pred_return_60d/90d`만 사용, `pred_mdd`는 `ret_score`에 직접 반영하지 않음 ([ranking_builder.py:1762-1784](/d:/ai/Lee_trader/python/ranking_builder.py#L1762)) | 불일치 |
| `ret_score` scaling | 전 종목 `r_adj`를 z-score 후 `50 + 10*z` ([build_backtest_predictions.py:225-228](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L225)) | 날짜별 percentile blend 0~100 ([ranking_builder.py:1765-1783](/d:/ai/Lee_trader/python/ranking_builder.py#L1765)) | 불일치 |
| `prob_score` source column | `prob_top20_30d` 우선, 없으면 `60d`, `90d` ([build_backtest_predictions.py:230-232](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L230)) | `prob_top20_60d`만 운영 점수에 사용 ([ranking_builder.py:1802-1808](/d:/ai/Lee_trader/python/ranking_builder.py#L1802), [ranking_builder.py:1817-1827](/d:/ai/Lee_trader/python/ranking_builder.py#L1817)) | 불일치 |
| `prob_score` scaling | 절대확률 `p*100` ([build_backtest_predictions.py:231-232](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L231)) | same-date percentile rank `*100` ([ranking_builder.py:1821-1827](/d:/ai/Lee_trader/python/ranking_builder.py#L1821)) | 불일치 |
| `prob_score` missing 처리 | 누락 시 0점으로 수렴 ([build_backtest_predictions.py:231](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L231)) | row 누락은 날짜별 median, 컬럼 부재는 50점 ([ranking_builder.py:1822-1833](/d:/ai/Lee_trader/python/ranking_builder.py#L1822)) | 불일치 |
| `qual_score` 계산 | `quality_score` 날짜별 percentile ([build_backtest_predictions.py:234-238](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L234)) | `quality_score` 날짜별 percentile ([ranking_builder.py:1837-1850](/d:/ai/Lee_trader/python/ranking_builder.py#L1837)) | 완전 동일 |
| `tech_score` source | `vol_ma_20` percentile, fallback `vol_ratio_20`, `vol_20` ([build_backtest_predictions.py:240-248](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L240)) | `composite`/`score_score` percentile 또는 feature-based tech composite + liquidity guard ([ranking_builder.py:1708-1740](/d:/ai/Lee_trader/python/ranking_builder.py#L1708), [ranking_builder.py:1393-1483](/d:/ai/Lee_trader/python/ranking_builder.py#L1393)) | 불일치 |
| 추가 positive component | `0.10 * bias_pred_score(60)` 상수항 존재 ([build_backtest_predictions.py:256-264](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L256)) | 상수항 없음, 대신 `valuation_score` sleeve 존재 ([ranking_builder.py:3142-3148](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)) | 불일치 |
| `valuation_score` | 없음 | 존재. explicit valuation metrics percentile 평균, 없으면 50 ([ranking_builder.py:1898-1943](/d:/ai/Lee_trader/python/ranking_builder.py#L1898)) | 불일치 |
| risk input | `mean(abs(pred_mdd_30d,60d,90d))` ([build_backtest_predictions.py:251](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L251)) | `0.6*abs(pred_mdd_60d)+0.4*abs(pred_mdd_90d)` ([ranking_builder.py:2024-2028](/d:/ai/Lee_trader/python/ranking_builder.py#L2024)) | 불일치 |
| penalty shape | multiplier `0.5~1.0` ([build_backtest_predictions.py:252-253](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L252)) | absolute deduction `0~18` piecewise ([ranking_builder.py:2017-2021](/d:/ai/Lee_trader/python/ranking_builder.py#L2017)) | 불일치 |
| penalty application | `base * risk_penalty` ([build_backtest_predictions.py:265](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L265)) | `base - w_risk_penalty * risk_penalty` ([ranking_builder.py:3142-3148](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)) | 불일치 |
| weight policy | 고정 `ret 0.40, prob 0.25, qual 0.15, tech 0.10, bias 0.10` ([build_backtest_predictions.py:256-264](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L256)) | regime별 가변 weight ([ranking_builder.py:424-464](/d:/ai/Lee_trader/python/ranking_builder.py#L424)) | 불일치 |
| regime branching | 없음 | bull / neutral / defensive 분기 ([ranking_builder.py:1561-1633](/d:/ai/Lee_trader/python/ranking_builder.py#L1561), [ranking_builder.py:2035-2080](/d:/ai/Lee_trader/python/ranking_builder.py#L2035)) | 불일치 |
| clipping | 최종 1회 clip, 중간 일부 `ret_score/prob_score/risk_penalty` clip ([build_backtest_predictions.py:228](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L228), [build_backtest_predictions.py:232](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L232), [build_backtest_predictions.py:253](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L253), [build_backtest_predictions.py:265](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L265)) | component별 clip, baseline clip, theme side-score clip 다단계 ([ranking_builder.py:1819-1827](/d:/ai/Lee_trader/python/ranking_builder.py#L1819), [ranking_builder.py:3152-3159](/d:/ai/Lee_trader/python/ranking_builder.py#L3152)) | 부분 동일 |

## Classification Summary

### 완전 동일

| Item | Notes |
| --- | --- |
| `qual_score`가 `quality_score`를 날짜별 percentile로 계산하는 방식 | 두 구현 모두 본질적으로 동일하다. backtest는 `rank(pct=True)` 기본값을 쓰고 production은 `method="average"`를 명시한다. pandas 기본이 `average`라 의미 차이는 없다. |

### 부분 동일

| Item | Notes |
| --- | --- |
| 최종 점수가 0~100으로 clip됨 | 두 구현 모두 최종 clip은 있다. 다만 clip 직전 식이 완전히 다르므로 결과 면에서는 parity를 주지 못한다. |
| `ret/prob/qual/tech/risk_penalty`라는 공통 컴포넌트 이름 사용 | 이름만 유사하고 계산 정의는 대부분 다르다. |

### 불일치

| Item | Notes |
| --- | --- |
| `ret_score` 정의 | backtest는 return과 MDD를 섞은 z-score, production은 60/90일 predicted return percentile blend다. |
| `prob_score` 정의 | backtest는 absolute probability, production은 same-date relative percentile이다. |
| `tech_score` 정의 | backtest는 단순 거래량 percentile, production은 composite 또는 feature-based multi-factor tech score다. |
| `valuation_score` 반영 여부 | production만 반영한다. |
| `bias` 상수항 | backtest만 반영한다. |
| risk penalty 공식과 적용 방식 | backtest는 multiplier, production은 absolute deduction이다. |
| regime 분기 | production만 반영한다. |
| 실제 입력 공급 구조 | backtest 호출 경로에는 `quality_score`/`tech` 입력이 전달되지 않아 해당 항목이 대부분 0 처리된다. |

## Input Supply Gap

backtest 구현은 함수 정의만 보면 `quality_score`와 `tech` 관련 컬럼을 받을 수 있게 작성돼 있다. 하지만 실제 main 경로는 아래처럼 동작한다.

1. `df_day`는 feature 행이다 ([build_backtest_predictions.py:334](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L334)).
2. `predict_for_date(df_day)`는 반환값으로 `date`, `code`, `pred_return_*`, `pred_mdd_*`, `prob_top20_*`만 만든다 ([build_backtest_predictions.py:132-180](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L132)).
3. 그 결과 `preds`를 그대로 `compute_scores()`에 넣는다 ([build_backtest_predictions.py:343-344](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L343)).

즉 실제 backtest `final_score`에는 다음 production 입력이 구조적으로 빠진다.

- `quality_score`
- `composite` / `score_score`
- `vol_ma_20` / `vol_ratio_20` / `vol_20`
- valuation metrics
- `regime`

이 때문에 backtest 쪽 `qual_score`, `tech_score`는 거의 항상 `NaN -> fillna(0)` 경로를 타고, production과 달리 `valuation_score`와 regime weight도 전혀 반영되지 않는다.

## Performance Distortion Analysis

### 1. 확률축 왜곡

production은 `prob_top20_60d`를 날짜별 상대순위로 써서 cross-sectional ranking에 맞춘다 ([ranking_builder.py:1821-1827](/d:/ai/Lee_trader/python/ranking_builder.py#L1821)). backtest는 절대확률을 그대로 쓰므로 모델 calibration drift가 있으면 전체 분포가 눌리거나 부풀어도 순위 점수에 직접 반영된다 ([build_backtest_predictions.py:231-232](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L231)).  
이 차이는 동일한 cross-section에서 실제 운영보다 backtest가 특정 날짜의 확률 스케일 변화에 더 민감해지는 왜곡을 만든다.

### 2. 리스크 축 왜곡

backtest는 risk를 `base * multiplier`로 적용해서 고득점 종목일수록 penalty 절대액이 더 커진다 ([build_backtest_predictions.py:253-265](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L253)). production은 `w_risk_penalty * deduction`을 빼므로 penalty 절대액은 base score와 독립적이다 ([ranking_builder.py:3142-3148](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)).  
이 차이는 고예상수익 고변동 종목의 순위를 backtest에서 과도하게 깎거나, 반대로 저점수 종목 penalty를 과소평가하게 만든다.

### 3. regime 민감도 누락

production은 시장 상태에 따라 bull / neutral / defensive weight를 바꾼다 ([ranking_builder.py:1561-1633](/d:/ai/Lee_trader/python/ranking_builder.py#L1561), [ranking_builder.py:424-464](/d:/ai/Lee_trader/python/ranking_builder.py#L424)). backtest는 고정 weight다.  
따라서 defensive 구간에서 production이 quality/valuation 비중을 높일 때 backtest는 여전히 return/probability 중심으로 평가한다. regime 전환기에 전략 성과를 과대평가할 가능성이 크다.

### 4. tech/quality/valuation 정보 손실

실행 경로상 backtest는 `quality_score`와 tech 입력이 제공되지 않고, valuation도 아예 없다 ([build_backtest_predictions.py:343-344](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L343)). production은 tech와 valuation을 baseline 점수에 직접 넣는다 ([ranking_builder.py:3142-3148](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)).  
즉 backtest는 production보다 훨씬 prediction-heavy한 순위가 된다. 이건 실운영보다 신호 분산이 작고 리스크 제어가 약한 포트폴리오를 선택하게 만들 수 있다.

### 5. bias 상수항의 인위적 바닥 형성

backtest는 `0.10 * 60 = 6점`의 상수항을 넣는다 ([build_backtest_predictions.py:257-264](/d:/ai/Lee_trader/python/build_backtest_predictions.py#L257)). production baseline에는 이런 항이 없다 ([ranking_builder.py:3142-3148](/d:/ai/Lee_trader/python/ranking_builder.py#L3142)).  
이 상수항은 backtest 점수 분포를 위로 밀어 risk multiplier와 결합된 비선형 효과를 만든다. 특히 중하위권 종목 간 점수 간격을 압축해 rank turnover를 바꿀 수 있다.

### 6. 실제 운영 랭킹과의 추가 괴리

`ranking_builder.py`의 baseline `final_score`는 theme를 직접 반영하지 않지만, 실제 최종 순위는 런타임 플래그에 따라 `final_score_v3`를 쓸 수 있다 ([ranking_builder.py:2569-2576](/d:/ai/Lee_trader/python/ranking_builder.py#L2569), [ranking_builder.py:3210-3212](/d:/ai/Lee_trader/python/ranking_builder.py#L3210)).  
따라서 backtest가 baseline `final_score`와도 불일치하고, 실운영 랭킹 컬럼과는 더 큰 괴리가 생길 수 있다.

## Bottom Line

현재 상태의 parity 판정은 아래와 같다.

| Target | Verdict | Reason |
| --- | --- | --- |
| backtest `final_score` vs production baseline `final_score` | 불일치 | 수식 구조, 컴포넌트 정의, penalty shape, weight 체계가 다름 |
| backtest `final_score` vs production live ranking (`final_score_v3` 가능) | 불일치 | baseline 불일치에 더해 theme overlay까지 추가 차이 존재 |
| `qual_score` 단일 컴포넌트 | 완전 동일 | `quality_score` percentile 로직 동일 |

## Reusable Scorer Integration Design

목표는 backtest와 production이 같은 scorer를 호출하도록 만드는 것이다. 설계 원칙은 아래가 적절하다.

### 1. `final_score`를 단일 모듈로 승격

신규 파일 예시:

- `python/scoring/final_score.py`

핵심 API 예시:

```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class FinalScoreConfig:
    use_regime_weights: bool = True
    include_theme_overlay: bool = False
    theme_mode: str = "v3"
    neutral_missing_prob: bool = True
    neutral_missing_valuation: bool = True

def build_component_scores(df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_risk_penalty(df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_final_score(df: pd.DataFrame, config: FinalScoreConfig) -> pd.DataFrame:
    ...
```

### 2. component builder도 공유

단순히 마지막 가중합만 공유하면 부족하다. 다음 함수들을 `ranking_builder.py`에서 분리해 shared module로 이동해야 한다.

- `_compute_ret_and_pred_scores`
- `_compute_prob_score`
- `_compute_qual_score`
- `_compute_tech_score`
- `_compute_valuation_score`
- `_compute_risk_penalty`
- `_resolve_component_weights`
- `detect_market_regime`

이유는 현재 괴리의 대부분이 최종 한 줄이 아니라 component 정의에서 발생하기 때문이다.

### 3. backtest 입력 스키마를 production 스키마로 맞춤

`build_backtest_predictions.py`는 `predict_for_date()` 결과만 넘기지 말고, 최소한 production scorer가 기대하는 컬럼을 함께 넘겨야 한다.

필수:

- `quality_score`
- tech source 컬럼: `composite` 또는 feature-based tech 입력들
- valuation source 컬럼들
- regime 계산용 시장 상태 입력 또는 사전 계산된 `regime`

권장 방식:

1. `df_day` feature row에 prediction 결과를 merge한다.
2. merged frame을 shared scorer에 넣는다.
3. backtest 결과 저장 시 `final_score`, `final_score_v3`, component score를 함께 저장한다.

### 4. parity test 추가

신규 테스트 예시:

- `tests/test_final_score_parity.py`

테스트 내용:

1. 고정 fixture DataFrame 생성
2. shared scorer output과 `ranking_builder` pipeline output 비교
3. `final_score`, `ret_score`, `prob_score`, `tech_score`, `qual_score`, `valuation_score`, `risk_penalty`, `regime`, `weight_profile`까지 column parity assert
4. tolerance는 `1e-9`

### 5. migration 순서

1. shared scorer 모듈 추출
2. `ranking_builder.py`가 shared scorer를 사용하도록 교체
3. `build_backtest_predictions.py`가 feature merge 후 shared scorer를 사용하도록 교체
4. parity tests 추가
5. 기존 `compute_scores()`는 deprecated 후 제거

## Recommended Decision

현재 코드 기준으로는 backtest 결과를 production `final_score`의 대리값으로 해석하면 안 된다.  
우선순위는 `build_backtest_predictions.py`의 `compute_scores()`를 유지보수 대상으로 보지 말고, production과 동일한 shared scorer로 교체하는 것이다.
