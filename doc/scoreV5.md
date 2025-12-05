
## 3. 점수 공식 v5 설계

이제 핵심인 **스코어링 v5**를 설계해보자.
목표는:

* 모든 점수를 **0~100 스케일**로 통일.
* 최종 `final_score_v5`는 **리스크 반영 + 시장 레짐 반영**을 포함.
* 구성 요소는 다음 7가지:

  * `ret_score` (수익률 기대)
  * `prob_score` (상위 20% 확률)
  * `qual_score` (펀더멘털 퀄리티)
  * `tech_score` (기술적 흐름)
  * `pred_score` (모델 신뢰도/일관성)
  * `risk_penalty` (리스크 패널티, 0~1)
  * `market_regime_score` (시장 레짐 영향, 0~1)

---

### 3.1 입력값 정의

하루 기준 `(date, code)`에 대해, 파이프라인에서 이미 가진 값들:

* 예측/확률:

  * `pred_return_60d`, `pred_return_90d`
  * `pred_mdd_60d`, `pred_mdd_90d`
  * `prob_top20_60d`, `prob_top20_90d`
* 펀더멘털:

  * `quality_score` (0~1 혹은 임의 스케일)
* 기술:

  * 모멘텀/변동성 지표들 (features에서 가져옴)
* 시장:

  * `market_vol_5d`, `market_foreign_5d`, `market_up`, `market_regime`
* 라벨/실현값:

  * (백테스트/검증 시에 사용)

---

### 3.2 정규화 전략

**1단계: Winsorize (극단치 잘라내기)**

* 각 스코어의 “원재료 값”에 대해,

  * 2.5% ~ 97.5% 구간으로 클리핑.

**2단계: Z-score → 0~100 변환**

예: `x`가 특정 factor 의 값이고,
`μ` = 종목 전체 평균, `σ` = 표준편차라면:

* `z = (x - μ) / σ`
* `score = 50 + 10 * z`

  * 보통 `|z| <= 2`면 30~70 구간.
* 이후 0~100으로 **클리핑**:

  * `score = max(0, min(100, score))`

이 패턴을 `ret_score_raw`, `qual_score_raw`, `tech_score_raw` 등에 공통 적용.

---

### 3.3 각 컴포넌트 공식

#### 3.3.1 ret_score – 수익률 기대 점수

1. **복합 예측 수익률**

   * 단기/중기를 합치는 복합 수익률:

[
r_{\text{comb}} = 0.6 \cdot \text{pred_return_60d} + 0.4 \cdot \text{pred_return_90d}
]

2. **리스크 조정 수익률**

   * 예측 MDD를 이용해 리스크 조정:

[
\text{pred_mdd_comb} = 0.6 \cdot \text{pred_mdd_60d} + 0.4 \cdot \text{pred_mdd_90d}
]

* 리스크 조정 수익률:

[
r_{\text{adj}} = \frac{r_{\text{comb}}}{1 + \alpha \cdot |\text{pred_mdd_comb}|}
]

* 여기서 `α`는 리스크 민감도 (예: 2.0 ~ 4.0 사이).

3. **정규화 → 0~100**

* 하루 유니버스 전체에 대해:

  * `r_adj` 리스트의 평균 `μ_r`, 표준편차 `σ_r` 계산.
  * 각 종목에 대해:

[
z_r = \frac{r_{\text{adj}} - \mu_r}{\sigma_r}
]
[
\text{ret_score} = \text{clip}(50 + 10 \cdot z_r, 0, 100)
]

---

#### 3.3.2 prob_score – 상위 20% 확률 점수

* 입력: `prob_top20_60d`, `prob_top20_90d` (0~1).
* 결합 확률:

[
p_{\text{comb}} = 0.7 \cdot \text{prob_top20_60d} + 0.3 \cdot \text{prob_top20_90d}
]

* 이를 0~100으로 직선 변환:

[
\text{prob_score} = 100 \cdot p_{\text{comb}}
]

* (원하면 전체 분포 기준 z-score를 한 번 더 걸 수도 있음.)

---

#### 3.3.3 qual_score – 펀더멘털 퀄리티 점수

* 입력: 기존 `quality_score` (예: 0~1 스케일 혹은 임의 값).
* 먼저 유니버스 기준 정규화:

[
z_q = \frac{\text{quality_score} - \mu_q}{\sigma_q}
]
[
\text{qual_score} = \text{clip}(50 + 10 \cdot z_q, 0, 100)
]

* 여기서 `μ_q`, `σ_q`는 당일 유니버스의 quality_score 통계.

---

#### 3.3.4 tech_score – 기술적 흐름 점수

예: 모멘텀/추세/거래량 요약:

* 예시 구성:

[
\text{score_mom} = f(\text{mom_60d})
]
[
\text{score_vol} = f(\text{vol_20d})
]
[
\text{score_turnover} = f(\text{turnover_20d})
]

실제 구현은:

1. `mom_20d`, `mom_60d`, `rsi_14`, `bb_pos_20d` 등을 각각 z-score → 0~100 스케일로 변환.
2. 가중합:

[
\text{tech_score} = 0.5 \cdot \text{score_mom} + 0.3 \cdot \text{score_rsi} + 0.2 \cdot \text{score_turnover}
]

* 최종: 0~100으로 클리핑.

---

#### 3.3.5 pred_score – 모델 신뢰도 점수 (옵션)

이건 **모델 진단 결과**를 반영하는 점수야. 예를 들어:

* 최근 3개월 동안:

  * 예측 수익률 vs 실제 수익률의 상관계수(`corr_pred_real`)
  * Brier score, log loss 등.

전략:

[
\text{pred_score} = \text{clip}(100 \cdot \text{perf_ratio}, 0, 100)
]

* `perf_ratio` : 0.0~1.0 범위의 모델 성능 요약 (예: 상관계수 0.3~0.6 → 0.5 정도).
* 당장은 간단하게 **상수 50~60**으로 두고, 나중에 진짜 모니터링 값 연결해도 됨.

---

#### 3.3.6 risk_penalty – 리스크 패널티

입력:

* `pred_mdd_60d`, `pred_mdd_90d`
* 변동성 피처 `vol_20d`
* 시장 레짐 (`market_regime`)

1. 복합 예측 MDD:

[
\text{pred_mdd_comb} = 0.6 \cdot \text{pred_mdd_60d} + 0.4 \cdot \text{pred_mdd_90d}
]

2. MDD 기반 리스크 레벨:

* `risk_mdd = max(0, min(1, |pred_mdd_comb| / MDD_ref))`

  * 예: `MDD_ref = 0.4` (–40% 기준)
  * MDD가 –40%면 risk_mdd = 1

3. 변동성 기반 리스크 레벨:

* 유니버스 기준 `vol_20d` 정규화 후 0~1로 스케일링 → `risk_vol`.

4. 종합 리스크 레벨:

[
\text{risk_level} = 0.7 \cdot \text{risk_mdd} + 0.3 \cdot \text{risk_vol}
]

5. 리스크 패널티:

[
\text{risk_penalty} = 1 - \beta \cdot \text{risk_level}
]

* `β`는 0~1 사이 (예: 0.5)

  * risk_level = 1 → penalty = 0.5
  * risk_level = 0 → penalty = 1.0
* **최종 범위**: `risk_penalty ∈ [0.5, 1.0]` 정도가 되도록 조정.

---

#### 3.3.7 market_regime_score – 시장 레짐 보정

입력: `market_regime` or `market_up`, `market_vol_5d`, `market_foreign_5d`.

예시:

* 베이스:

```text
if market_regime == "bull":    base = 1.0
elif market_regime == "sideways": base = 0.9
elif market_regime == "bear":  base = 0.8
```

* 외국인 수급/변동성 보정:

[
\text{adj} =
\begin{cases}
+0.05 & \text{if foreign_flow 강한 순매수} \
-0.05 & \text{if foreign_flow 강한 순매도} \
0 & \text{otherwise}
\end{cases}
]

[
\text{market_regime_score} = \text{clip}(base + adj, 0.7, 1.1)
]

---

### 3.4 최종 final_score_v5 공식

1. **Base Score (리스크/시장 적용 전)**

각 스코어는 0~100이라고 가정.

가중치 예시:

* `w_ret = 0.40`
* `w_prob = 0.25`
* `w_qual = 0.15`
* `w_tech = 0.10`
* `w_pred = 0.10`

[
\text{base_score} =
w_{ret}\cdot\text{ret_score} +
w_{prob}\cdot\text{prob_score} +
w_{qual}\cdot\text{qual_score} +
w_{tech}\cdot\text{tech_score} +
w_{pred}\cdot\text{pred_score}
]

2. **리스크 패널티 적용**

[
\text{risk_adjusted_score} = \text{base_score} \cdot \text{risk_penalty}
]

3. **시장 레짐 보정 적용**

[
\text{final_score_v5} = \text{risk_adjusted_score} \cdot \text{market_regime_score}
]

4. 마지막으로 0~100으로 클리핑:

[
\text{final_score_v5} = \text{clip}(\text{final_score_v5}, 0, 100)
]

---

### 3.5 Python 의사코드 예시

나중에 `scoring_v5.py`로 바로 만들 수 있게 의사코드 느낌으로:

```python
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def compute_ret_score(df):
    r_comb = 0.6 * df.pred_return_60d + 0.4 * df.pred_return_90d
    pred_mdd_comb = 0.6 * df.pred_mdd_60d + 0.4 * df.pred_mdd_90d
    alpha = 3.0
    r_adj = r_comb / (1 + alpha * abs(pred_mdd_comb))

    mu = r_adj.mean()
    sigma = r_adj.std() or 1e-6
    z = (r_adj - mu) / sigma
    ret_score = 50 + 10 * z
    return ret_score.clip(0, 100)

def compute_prob_score(df):
    p_comb = 0.7 * df.prob_top20_60d + 0.3 * df.prob_top20_90d
    return (100 * p_comb).clip(0, 100)

def compute_qual_score(df):
    mu = df.quality_score.mean()
    sigma = df.quality_score.std() or 1e-6
    z = (df.quality_score - mu) / sigma
    return (50 + 10 * z).clip(0, 100)

def compute_tech_score(df):
    # 예시: mom_60d, rsi_14, turnover_20d
    def z_to_score(series):
        mu = series.mean()
        sigma = series.std() or 1e-6
        z = (series - mu) / sigma
        return (50 + 10 * z).clip(0, 100)

    score_mom = z_to_score(df.mom_60d)
    score_rsi = z_to_score(df.rsi_14)
    score_turnover = z_to_score(df.turnover_20d)

    tech_score = 0.5 * score_mom + 0.3 * score_rsi + 0.2 * score_turnover
    return tech_score.clip(0, 100)

def compute_risk_penalty(df):
    pred_mdd_comb = 0.6 * df.pred_mdd_60d + 0.4 * df.pred_mdd_90d
    MDD_REF = 0.4
    risk_mdd = (abs(pred_mdd_comb) / MDD_REF).clip(0, 1)

    # vol_20d 는 features에서 가져왔다고 가정
    vol = df.vol_20d
    mu_vol = vol.mean()
    sigma_vol = vol.std() or 1e-6
    z_vol = (vol - mu_vol) / sigma_vol
    risk_vol = ((z_vol - z_vol.min()) / (z_vol.max() - z_vol.min() + 1e-6)).clip(0, 1)

    risk_level = 0.7 * risk_mdd + 0.3 * risk_vol
    beta = 0.5
    risk_penalty = 1 - beta * risk_level
    return risk_penalty.clip(0.5, 1.0)

def compute_market_regime_score(market_row):
    regime = market_row.market_regime
    base = 0.9
    if regime == "bull":
        base = 1.0
    elif regime == "sideways":
        base = 0.9
    elif regime == "bear":
        base = 0.8

    adj = 0.0
    if market_row.market_foreign_5d > 0:
        adj += 0.05
    elif market_row.market_foreign_5d < 0:
        adj -= 0.05

    return clip(base + adj, 0.7, 1.1)

def compute_final_score_v5(df, market_row, pred_score_default=60):
    df = df.copy()

    df["ret_score"]  = compute_ret_score(df)
    df["prob_score"] = compute_prob_score(df)
    df["qual_score"] = compute_qual_score(df)
    df["tech_score"] = compute_tech_score(df)
    df["pred_score"] = pred_score_default  # TODO: 모델 모니터링 연동
    df["risk_penalty"] = compute_risk_penalty(df)

    market_reg_score = compute_market_regime_score(market_row)
    df["market_regime_score"] = market_reg_score

    w_ret, w_prob, w_qual, w_tech, w_pred = 0.40, 0.25, 0.15, 0.10, 0.10

    base_score = (
        w_ret  * df["ret_score"]  +
        w_prob * df["prob_score"] +
        w_qual * df["qual_score"] +
        w_tech * df["tech_score"] +
        w_pred * df["pred_score"]
    )

    df["final_score_v5"] = (base_score * df["risk_penalty"] * df["market_regime_score"]).clip(0, 100)
    return df
```

이렇게 `compute_final_score_v5()`에서 나온 컬럼들을
그대로 `fact_score` 테이블에 `to_sql()`로 넣으면 돼.

---

## 4. 정리 + 다음 스텝 제안

* **ERD**:

  * `dim_stock`, `fact_price_daily`, `fact_fundamental`, `fact_market_status`
  * `fact_feature`, `fact_label`, `fact_prediction`, `fact_score`, `fact_ranking`
  * `fact_trade`, `fact_portfolio_snapshot`, `fact_backtest_result`
    이런 구조로 잡으면,
  * 데이터 분석 / 백테스트 / 실매매 / 리포트까지 한 DB에서 깔끔하게 가능.

* **점수 v5**:

  * 5개 구성 점수(`ret/prob/qual/tech/pred`) → 0~100
  * 리스크 패널티(`risk_penalty`) : 0.5~1.0
  * 시장 레짐 (`market_regime_score`) : 0.7~1.1
  * **최종**: `final_score_v5 = base_score * risk_penalty * market_regime_score`

---

이 다음 단계로는:

1. **`schema.sql` 파일 생성**

   * 방금 정의한 테이블들 기준으로 실제 CREATE TABLE 스크립트.
2. **`scoring_v5.py` 구현**

   * 위 의사코드를 실제 파이프라인에 붙이기.
3. **`migrate_to_sqlite_v5.py`**

   * 기존 CSV → 새 ERD 구조로 마이그레이션하는 스크립트.

원하면 **`schema.sql` 풀 버전**이랑
**기존 `scoring.py`를 v5 구조로 리팩토링하는 실제 코드**까지 쭉 짜줄게.
