````markdown
# Lee_trader 파이프라인 계산 이론 설명서 (v1 기준)

> 이 문서는 **Lee_trader 프로젝트 v1 / v4.1 파이프라인에 포함된 `.py` 파일들이  
> 어떤 계산 이론과 원리로 결과물을 만들어내는지**를 정리한 설명서입니다.  
> 다른 사람이 코드를 안 열어봐도 “어떤 생각으로 이런 값이 나오는지” 이해하는 것을 목표로 합니다. :contentReference[oaicite:0]{index=0}  

---

## 0. 공통 개념 정리

먼저, 거의 모든 파일에서 공통으로 쓰이는 기본 개념부터 정리하고 갈게.

### 0.1 가격 → 수익률

- **종가(close)**
  - 하루의 마감 가격.
- **단기 수익률**
  - 예: 1일 수익률
  - 수식:  
    \[
      \text{ret\_1d}(t) = \frac{\text{close}(t)}{\text{close}(t-1)} - 1
    \]
- **장기 수익률 (예: 60일, 90일)**
  - 타깃(라벨) 계산에도 쓰임.
  - 수식:
    \[
      \text{ret\_60d}(t) = \frac{\text{close}(t+60)}{\text{close}(t)} - 1
    \]
    \[
      \text{ret\_90d}(t) = \frac{\text{close}(t+90)}{\text{close}(t)} - 1
    \]

### 0.2 이동평균(MA)와 모멘텀

- **단순 이동평균 (Simple Moving Average, SMA)**
  - 예: 20일 이동평균
  - 수식:
    \[
      \text{MA}_{20}(t) = \frac{1}{20} \sum_{i=0}^{19} \text{close}(t-i)
    \]
- **모멘텀(mom_20 등)**
  - “얼마나 올랐는가/내렸는가”를 단순 비교
  - 예: 20일 모멘텀
    \[
      \text{mom\_20}(t) = \frac{\text{close}(t)}{\text{close}(t-20)} - 1
    \]

### 0.3 변동성(Volatility) & 롤링 지표

- **변동성(vol_20)**
  - 특정 기간 수익률의 표준편차
  - 예:
    \[
      \text{vol\_20}(t) = \text{stdev}\left(\text{ret\_1d}(t-19), \dots, \text{ret\_1d}(t)\right)
    \]
- 롤링(rolling) 개념:
  - “과거 N일 창(window)을 계속 한 칸씩 옮기면서 통계값 계산”

### 0.4 백분위(Percentile) 점수

- 같은 날짜에 여러 종목이 있을 때,
  - 예측 수익률이 상위 몇 %인지 표현하기 위해 **백분위 점수** 사용.
- 예:
  - 그날 모든 종목의 pred_return_60d를 정렬했을 때,
  - 어떤 종목이 상위 10%면 → `90점`, 하위 10%면 → `10점`처럼.

### 0.5 Winsorization & Hard Clipping (타깃 안정화)

**타깃(60일/90일 수익률)** 에서 너무 극단적인 값(상한가 연속, 폭락 등)을  
그대로 모델에 넣으면, 모델이 “괴물 구간”에 끌려다님.

그래서 다음 두 단계로 정리:

1. **윈저라이징(Winsorization, 퍼센타일 기반)**
   - 예: target_60d에서 train 데이터 기준 2~98퍼센타일 범위를 계산.
   - 그 범위 밖의 값은 가장자리 값으로 잘라냄.
     - 2%보다 작으면 → 2% 값으로
     - 98%보다 크면 → 98% 값으로

2. **하드 클리핑(Hard Clipping)**
   - “아무리 그래도 이 이상/이하는 안 나왔으면 좋겠다”는 절대 범위 적용.
   - 예:
     - target_60d: [-0.5, 0.8]
     - target_90d: [-0.7, 1.0]

최종 bound는  
> “퍼센타일 bound ∩ 하드 bound”  
교집합 영역으로 결정해서 적용. :contentReference[oaicite:1]{index=1}  

---

## 1. `fetch_top_universe.py` — 종목 유니버스 정의

### 1.1 목적

- **어떤 종목들을 “분석 대상”으로 삼을지** 정의하는 단계.
- 너무 잡주/거래 거의 없는 종목까지 모두 포함하면
  - 데이터 노이즈 ↑
  - 실제 매매 가능성 ↓
- 그래서 **시장(KOSPI/KOSDAQ) + 최소 요건 충족 종목만 모아서 `universe.csv`로 저장**.

### 1.2 입력/출력

- 입력
  - KRX 상장 종목 리스트 (pykrx 또는 KIS/기타 데이터 소스)
- 출력
  - `data/universe.csv`
    - `code`, `name`, `market`, `sector` 등의 기본 정보

### 1.3 이론/원리

- **유니버스 제한의 의미**
  - (1) 데이터 품질: 상장폐지/거래중단/죽은 종목 제거
  - (2) 유동성: 지나치게 거래가 없는 종목 제외
  - (3) 현실성: 실제 투자 가능한 종목군으로 제한

- 예시 필터링 원칙 (코드에선 구체 조건으로 구현)
  - 상장일이 너무 최근인 종목 제외 (히스토리 부족)
  - 투자주의/관리종목/스팩 등 필요시 제외
  - 시장 구분: KOSPI, KOSDAQ, ETF, ETN 등을 구분해 특정 시장만 선택  

---

## 2. `download_prices_kis.py` — 일별 가격 데이터 수집

### 2.1 목적

- 유니버스에 포함된 종목들의 **과거 일별 OHLCV(시가/고가/저가/종가/거래량)** 를  
  한국투자증권(KIS) API와 pykrx를 이용해 수집.

### 2.2 입력/출력

- 입력
  - `universe.csv`
  - `.env`에 설정된 KIS API 키
- 출력
  - `data/prices_daily_raw.csv`
    - `date`, `code`, `open`, `high`, `low`, `close`, `volume`, `value` 등

### 2.3 이론/원리

- **기본 데이터의 역할**
  - 이후 모든 피처(모멘텀, 이동평균, 변동성, 타깃 라벨 등)는
    - 이 가격 데이터에서 파생됨.
- **폴백 전략**
  - KIS API 요청 실패 시 pykrx로 대체
    - 실전 운영에서 외부 API 장애 대비.

---

## 3. `clean_prices.py` — 가격 데이터 정제

### 3.1 목적

- 원시 데이터(`prices_daily_raw.csv`)에는 다음과 같은 문제가 있을 수 있음:
  - 휴장일/상장 전 기간의 NaN
  - 상장폐지 직전 비정상 데이터
  - 잘못된 값 (0 가격, 음수, 비정상 급등락 등)
- 이를 정제하여 **모델이 믿을 수 있는 기본 가격열**을 생성.

### 3.2 입력/출력

- 입력
  - `prices_daily_raw.csv`
- 출력
  - `prices_daily_clean.csv`

### 3.3 이론/원리

- **결측치 처리**
  - 상장 전/후 구간 구분
  - 상장 전 기간은 아예 제외
  - 일부 결측은 휴장일로 보고 스킵/보간 등 처리

- **비정상값 처리**
  - 0원 가격, 음수, 말도 안 되는 단일 일간 변동(예: ±90% 이상)을 검출,
  - 필요 시 이전값/이후값 기반 보정 또는 해당 날짜 제거.

---

## 4. `create_adjusted_prices.py` — 조정 주가 생성

### 4.1 목적

- 배당, 액면분할, 유·무상증자 등 **기업 액션으로 인한 단순 가격 갭을 제거**하고  
  순수한 “시장 관점의 가격 흐름”을 반영하는 **조정 주가(adjusted price)** 를 만들기 위함.

### 4.2 입력/출력

- 입력
  - `prices_daily_clean.csv`
  - (필요시) 배당/액면/증자 정보
- 출력
  - `prices_daily_adjusted.csv`
    - 실제 피처/라벨 계산에 주로 사용하는 종가열

### 4.3 이론/원리

- 예:
  - 액면분할로 2:1 분할이 일어나면  
    - 실제 기업 가치 변화 없이 단순히 주식 수만 2배가 됨.
  - 이런 경우 생 주가를 그대로 쓰면  
    - 모멘텀/수익률 지표에 인위적인 -50% 갭이 생김.

- 조정 주가 개념:
  - “액면분할/배당 등을 감안한 가상의 연속 가격 시계열”
  - 이를 기준으로 수익률, 이동평균 등을 계산해야  
    - 모델이 “진짜 상승/하락”을 제대로 학습 가능.

---

## 5. `fetch_fundamentals_dart.py` — 재무 데이터 수집

### 5.1 목적

- OpenDART API를 통해 **재무제표(ROE, 영업이익률, 부채비율 등)** 를 수집하여  
  추후 **quality_score** 계산에 활용. :contentReference[oaicite:2]{index=2}  

### 5.2 입력/출력

- 입력
  - `universe.csv` (종목별 사업자번호/회사코드 매핑에 활용)
  - `.env`의 `DART_API_KEY`
- 출력
  - `fundamentals.csv`
    - `date` (주로 YYYY-12-31)
    - `code`
    - `roe`, `op_margin`, `debt_ratio`, `ocf_to_assets`, `net_margin` 등

### 5.3 이론/원리

- 사용 지표 예:
  - **ROE (Return on Equity)**: 자기자본 대비 순이익, 수익성 지표
  - **영업이익률 (op_margin)**: 매출 대비 영업이익 비율
  - **부채비율 (debt_ratio)**: 재무 레버리지, 지나치게 높으면 위험
  - **OCF/자산 (ocf_to_assets)**: 현금창출능력
  - **순이익률 (net_margin)**: 최종 수익성

- **연간 단일 시점 기준**
  - 사업보고서(11014) 기준 연말(12-31) 데이터 사용.
  - CFS(연결) 우선, 없으면 OFS(별도)로 폴백.

---

## 6. `quality_builder.py` — Quality Score 계산

### 6.1 목적

- 재무 데이터를 기반으로 **기업의 “질(퀄리티)”을 하나의 점수로 요약**하여  
  나중에 랭킹 계산 시 **저품질/좀비기업을 거르는 데 사용**.

### 6.2 입력/출력

- 입력
  - `fundamentals.csv`
- 출력
  - `quality.csv`
    - `date`, `code`, `quality_score`  

### 6.3 이론/원리

핵심 아이디어:  
> “좋은 기업은 수익성이 높고, 부채는 과하지 않고, 현금 창출력이 괜찮다.”

1. **각 재무 지표를 z-score로 표준화**
   - 예: ROE z-score
     \[
       z_{\text{roe}} = \frac{\text{roe} - \mu_{\text{roe}}}{\sigma_{\text{roe}}}
     \]
   - 여기서 μ, σ는 같은 연도 전체 종목 기준 평균/표준편차.

2. **부채비율은 반대로 작을수록 좋으니 부호/변환으로 뒤집음**
   - 예:
     \[
       z_{\text{debt}} = -\frac{\text{debt\_ratio} - \mu_{\text{debt}}}{\sigma_{\text{debt}}}
     \]

3. **각 z-score를 일정 범위로 클리핑**
   - 예: [-3, 3]
   - 극단값이 한 지표를 지배하지 않도록.

4. **가중합으로 quality_score 계산**
   - 예시 형태:
     \[
       \text{quality\_score} = w_1 \cdot z_{\text{roe}}
                             + w_2 \cdot z_{\text{op\_margin}}
                             + w_3 \cdot z_{\text{net\_margin}}
                             + w_4 \cdot z_{\text{ocf\_to\_assets}}
                             + w_5 \cdot z_{\text{debt}}
     \]
   - 구체 가중치는 코드에 정의되어 있고,  
     “수익성/현금창출에 더 높은 가중치, 부채에 페널티” 형태로 설계.

---

## 7. `feature_builder.py` — ML용 피처 생성

### 7.1 목적

- **가격 데이터 + quality_score** 를 조합하여  
  모델 학습에 사용할 **features.csv**를 생성.

### 7.2 입력/출력

- 입력
  - `prices_daily_adjusted.csv`
  - `quality.csv`
- 출력
  - `features.csv`
    - `date`, `code`, 각종 기술적 지표, `quality_score`

### 7.3 이론/원리

**가격 기반 피처 예시** (프로젝트 문서 기준): :contentReference[oaicite:3]{index=3}  

- `close`: 조정 종가
- `ret_1d`: 1일 수익률
- `mom_20`: 20일 모멘텀
- `ma_5`, `ma_20`, `ma_60`: 이동평균
- `close_over_ma20`: (현재가 / 20일 MA) - 1
- `vol_20`: 20일 변동성
- `rsi_14`: RSI(상대강도지수)
- `volume`: 거래량

**Quality 결합**

- (date, code)를 기준으로
  - 과거 가장 최근 연간 재무 데이터의 `quality_score`를 **as-of 방식으로 매핑**
  - 예: 2025-02-15 일자의 quality_score는 2024-12-31 재무 기준 값.

→ 이렇게 해서 “가격/기술 + 재무 퀄리티”가 동시에 들어간 feature 벡터를 구성.

---

## 8. `scoring.py` — 기술적 점수(tech_score) 계산

### 8.1 목적

- 가격 흐름만 봤을 때
  - “차트가 괜찮은지”
  - “추세/모멘텀이 우상향인지”
- 등을 단일 점수(0~100) `tech_score`로 요약하여 `scores_final.csv`로 저장.

### 8.2 입력/출력

- 입력
  - `features.csv` (모멘텀, 이동평균, 변동성 등 포함)
- 출력
  - `scores_final.csv`
    - `date`, `code`, `score` (→ tech_score로 사용)

### 8.3 이론/원리

기본 아이디어:

- **상승 모멘텀**이 강할수록 +
- **우상향 추세**에 가까울수록 +
- **변동성이 극단적이지 않을수록** +
- **과도한 과열 구간은 약간 페널티**

구체 구현은 프로젝트 코드에 있지만, 개념적으로는:

1. 각 기술지표를 개별 점수로 환산
   - 예:
     - mom_20, close_over_ma20 → 모멘텀 점수
     - vol_20 → 안정성 점수
     - rsi_14 → 과열/과매도 보정 점수

2. 일정 스케일(0~100)로 변환 후 가중합

3. 결과를 `score` 컬럼으로 저장

이 `score`가 이후 `ranking_builder.py`에서 **tech_score**로 쓰임.

4. 버전2 추가
그날 전체 종목 중에서

**중기 모멘텀(20일)**이 좋고

최근 10일 수익률도 괜찮고

20일선 위에 있고,

평소보다 거래량이 붙었고,

RSI가 너무 과열/침체가 아니면서,

변동성은 적당한 종목이
→ composite가 높아지고,
→ score(0~100)도 상위권으로 올라가게 됨.

---

## 9. `label_builder.py` — 타깃/라벨 생성

### 9.1 목적

- 모델이 학습할 **정답(y)** 을 만드는 단계.
- 회귀용(수익률 예측)과 분류용(상위 20% 여부)을 모두 생성. :contentReference[oaicite:4]{index=4}  

### 9.2 입력/출력

- 입력
  - `prices_daily_adjusted.csv`
- 출력
  - `labels.csv`
    - `date`, `code`
    - `target_60d`, `target_90d`
    - `target_60d_top20`, `target_90d_top20` (0/1)

### 9.3 이론/원리

1. **회귀 타깃: 60/90일 수익률**
   - 앞에서 설명한 대로:
     \[
       \text{target\_60d}(t) = \frac{\text{close}(t+60)}{\text{close}(t)} - 1
     \]
     \[
       \text{target\_90d}(t) = \frac{\text{close}(t+90)}{\text{close}(t)} - 1
     \]

2. **분류 타깃: 같은 날짜 내 상위 20% 여부**
   - 특정 날짜 t에서, 모든 종목의 `target_60d(t)`를 계산한 뒤,
     - 상위 20%에 속하는 종목 → `target_60d_top20 = 1`
     - 나머지 → `0`
   - `target_90d_top20`도 동일한 방식.

이렇게 하면:

- 회귀 모델은 “정확한 수익률 예측”
- 분류 모델은 “이 종목이 상위 20% 안에 들어갈 확률”을 각각 학습하게 됨.

---

## 10. `model_train.py` — 회귀/분류 모델 학습

### 10.1 목적

- `features.csv` + `labels.csv`를 합쳐
  - **LightGBM 회귀/분류 모델**을 학습하고
  - 최종 결과를 `model.pkl`에 패키징.

### 10.2 입력/출력

- 입력
  - `features.csv`
  - `labels.csv`
- 출력
  - `model.pkl` (여러 모델 + 메타데이터 포함)

### 10.3 이론/원리

1. **타깃 안정화 (윈저라이징 + 클리핑)** :contentReference[oaicite:5]{index=5}  
   - `target_60d`, `target_90d` 각각에 대하여:
     - 퍼센타일 기반 bounds (ex: 2~98%, 5~95%)
     - 하드 bounds (ex: [-0.5,0.8], [-0.7,1.0])
   - 두 범위의 교집합으로 최종 범위 결정 후,
     - 타깃 값을 그 범위로 잘라냄.

2. **TimeSeriesSplit 기반 교차검증**
   - 날짜 기준으로 순서를 유지하는 시계열 CV 사용.
   - 최대 5폴드, 데이터가 부족하면 80/20 단일 분할.
   - 회귀:
     - RMSE, MAE 모니터링
   - 분류:
     - Accuracy, ROC AUC (필요시 LogLoss 등)

3. **LightGBM 모델 구조** :contentReference[oaicite:6]{index=6}  
   - 회귀:
     - `LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=64, ...)`
   - 분류:
     - `LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.03, ...)`

4. **최종 패키징 (model.pkl)**  
   - 여러 모델, 사용 피처 목록, 타깃 이름, 클리핑 정보 등을 딕셔너리로 묶어서 저장:
     ```python
     pack = {
       "reg_models": { "target_60d": reg60, "target_90d": reg90 },
       "cls_models": { "target_60d_top20": cls60, "target_90d_top20": cls90 },
       "features": feature_cols_used,
       "targets_reg": [...],
       "targets_cls": [...],
       # 레거시 호환용
       "models": {...},
       "targets": [...],
       # 타깃 클리핑 관련 메타데이터
       "winsor_percentiles": {...},
       "hard_clip_bounds": {...},
       "winsor_bounds": {...},
     }
     ```

---

## 11. `model_predict.py` — 예측값 생성

### 11.1 목적

- 최신 날짜 기준 feature를 이용해  
  - **60일/90일 수익률 예측 (회귀)**
  - **상위 20% 가능성(확률) 예측 (분류)**
- 결과를 `predictions.csv`에 저장. :contentReference[oaicite:7]{index=7}  

### 11.2 입력/출력

- 입력
  - `features.csv`
  - `model.pkl`
- 출력
  - `predictions.csv`
    - `date`, `code`
    - `pred_return_60d`, `pred_return_90d`
    - `prob_top20_60d`, `prob_top20_90d`

### 11.3 이론/원리

1. **최신 시점의 feature만 추출**
   - 종목별로 가장 마지막 날짜의 feature 1행을 선택.
   - 그 시점에서 “앞으로 60일/90일”을 예측한다는 의미.

2. **회귀 예측**
   - `reg_models["target_60d"].predict(features)` → pred_return_60d
   - `reg_models["target_90d"].predict(features)` → pred_return_90d

3. **분류 예측 (확률)**
   - `cls_models["target_60d_top20"].predict_proba(features)[:,1]` → prob_top20_60d
   - `cls_models["target_90d_top20"].predict_proba(features)[:,1]` → prob_top20_90d

4. **해석**
   - `pred_return_60d = 0.10` → “60일 후 +10% 수익률 기대”
   - `prob_top20_60d = 0.25` → “60일 후 전체 종목 중 상위20% 안 들어갈 확률이 25%”

---

## 12. `ranking_builder.py` — 최종 점수 & 랭킹

### 12.1 목적

- 지금까지 만들어진 모든 정보를 모아:
  - 기술 점수(tech_score)
  - 예측 점수(pred_score)
  - 확률 점수(prob_score)
  - 퀄리티 점수(qual_score)
- 를 계산하고 **최종 종합 점수(final_score)** 로 랭킹을 매김. :contentReference[oaicite:8]{index=8}  

### 12.2 입력/출력

- 입력
  - `predictions.csv`
  - `scores_final.csv`
  - `features.csv` (quality_score, close 등)
  - `universe.csv` (name, market, sector)
- 출력
  - `ranking_final.csv`
    - date, code, name, market, sector
    - close
    - pred_return_60d, pred_return_90d
    - prob_top20_60d, prob_top20_90d
    - tech_score, pred_score, prob_score, qual_score
    - final_score

### 12.3 이론/원리

1. **각 점수의 해석**
   - `tech_score`:
     - `scores_final.csv`의 score 사용 (0~100)
     - “차트/가격 흐름만 봤을 때 기술적으로 얼마나 매력적인지”
   - `pred_score`:
     - 같은 날짜 내 `pred_return_60d` 백분위 (0~100)
     - “수익률 예측이 다른 종목 대비 어느 정도 상위인지”
   - `prob_score`:
     - `prob_top20_60d * 100` (0~100) 클립
     - “상위 20% 안 들어갈 확률이 어느 정도인지”
   - `qual_score`:
     - 같은 날짜 내 `quality_score` 백분위 (0~100)
     - “재무 퀄리티가 다른 종목 대비 어느 정도인지”

2. **최종 점수 산식(v1)** :contentReference[oaicite:9]{index=9}  

```text
final_score = 0.30 * tech_score
             + 0.30 * pred_score
             + 0.25 * prob_score
             + 0.15 * qual_score
````

* Return(예측 수익률)과 Tech(차트)를 합쳐 60%
* 상위20% 확률 25%
* 재무 퀄리티 15%

3. **정렬**

   * (date asc, final_score desc)로 정렬 후
   * 상위 N개를 TOP 리스트로 사용.

---

## 13. `paper_trading_tracker.py` — 페이퍼 트레이딩

### 13.1 목적

* 예측/랭킹이 실제로 의미가 있는지,

  * 백테스트를 통해 **“가상의 매매 기록”을 남기고 성과를 측정**.

### 13.2 입력/출력

* 입력

  * `predictions.csv`
  * `labels.csv`
  * (필요시) `ranking_final.csv`
* 출력

  * `paper_trades.csv`
  * `paper_trading_with_returns.csv` 등 여러 요약 파일

### 13.3 이론/원리

* 예를 들어:

  * 매일 예측과 랭킹을 기준으로 **Top K 종목 매수**,
  * 60일 보유 후 청산하는 전략을 가정.
* 각 가상의 트레이드에 대해:

  * 매수일, 매도가(60일 후), 수익률을 기록.
* 나중에 `paper_trading_report.py`에서 집계/분석.

---

## 14. `paper_trading_report.py` — 페이퍼 트레이딩 리포트

### 14.1 목적

* `paper_trades.csv` 등의 결과를 집계하여,

  * 전략의 **평균 수익률, MDD, 승률, 최근 성과** 등을 표/요약으로 제공.

### 14.2 입력/출력

* 입력

  * `paper_trades.csv`
  * 기타 페이퍼 트레이딩 결과 파일들
* 출력

  * `paper_trading_summary.csv`
  * `paper_trading_by_rank.csv`
  * `paper_trading_by_horizon.csv` 등

### 14.3 이론/원리

* 전략 성능 평가 지표:

  * 평균/중간 수익률
  * MDD(최대 낙폭)
  * 승률(수익 거래 비율)
  * 시간대별 성과 (최근 1년/2년 등)
* “final_score 상위 구간일수록 성과가 좋았는가?” 같은 질문에 답할 수 있음.

---

## 15. (옵션) `model_train_optuna.py` — 하이퍼파라미터 튜닝

### 15.1 목적

* **기본 LightGBM 파라미터를 더 잘 맞추기 위해**
  Optuna를 활용해 하이퍼파라미터를 자동 탐색하고,
  `lgbm_reg_params.json` 같은 파일로 저장.

### 15.2 이론/원리 (간단)

* Optuna:

  * 특정 목적함수(예: 검증 RMSE)를 최소화하는 파라미터(learning_rate, num_leaves 등)를 자동 탐색.
* 이 스크립트는 “연구/개선용 옵션”에 가깝고,

  * 파이프라인 필수 요소는 아님.
  * 한 번 튜닝해서 나온 파라미터를 재활용하는 구조로 쓰는 게 일반적.

---


