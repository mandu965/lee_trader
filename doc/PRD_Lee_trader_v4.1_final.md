````markdown
# Lee_trader 프로젝트 최종 통합 문서 (v4.1 기준)

> 이 문서는 현재 Lee_trader 프로젝트의 **최신 구조(v4.1)** 를 기준으로  
> 기획(PRD) · 모델 상세 · 운영 가이드 · 프로젝트 구조를 모두 통합한 **최종 레퍼런스 문서**입니다.   

---

## 0. 문서 목적

- 현재까지 구축된 **Lee_trader 프로젝트의 전체 그림을 한 번에 파악**할 수 있도록 정리
- 실제로 **주식 매매에 활용 가능한 수준의 시스템**으로 확장하기 위한 기반 문서
- 이후 버전업(v5.x, 실매매/자동화 등) 시, 변경 포인트를 명확하게 파악할 수 있는 기준점 역할

이 문서는 다음 네 개의 내부 문서를 기반으로 작성되었습니다.

- `PRD_Lee_trader_상세_설명서.md` :contentReference[oaicite:1]{index=1}  
- `운영 가이드라인.md` :contentReference[oaicite:2]{index=2}  
- `프로젝트구조.md` :contentReference[oaicite:3]{index=3}  
- `PRD_4.0_모델상세.md (PRD 4.1 – 모델 상세 설계서)` :contentReference[oaicite:4]{index=4}  

---

## 1. 제품 개요 및 목표

### 1.1 제품 정의

**Lee_trader**는 한국 주식시장(KOSPI/KOSDAQ)을 대상으로 하는  
**AI/ML 기반 종목 분석·예측·랭킹 시스템**입니다.  

- 개인 투자자가 **정량적인 근거(퀀트 + 머신러닝)** 를 바탕으로  
  종목을 발굴하고 점수화하여 **의사결정을 돕는 투자 보조 도구**입니다.   

### 1.2 주요 목표

1. **수익률 예측**
   - 60일 / 90일 후 주가 수익률을 ML 모델로 예측 (회귀)   

2. **상위 종목 분류**
   - 같은 날 전체 종목 중 **상위 20% 수익률**에 속할 가능성을 분류 모델로 추정 (확률) :contentReference[oaicite:7]{index=7}  

3. **종합 점수 및 랭킹**
   - 기술적 점수(가격/모멘텀/변동성)  
   + 예측 점수(예상 수익률 백분위)  
   + 확률 점수(Top20 분류 확률)  
   + 품질 점수(재무/펀더멘털 기반)  
   → **복합 가중합으로 final_score 산출 및 TOP 랭킹 제공**   

4. **시장 상태/트렌드 반영**
   - 시장 전체 추세(상승/하락/횡보)에 따른 **위험 필터링 및 경고** :contentReference[oaicite:9]{index=9}  

5. **사용자 친화적 UI**
   - 웹 브라우저에서 **Top 랭킹 / 전체 리스트 / 개별 종목 상세 / 백테스트 결과를 한 번에 확인** 가능   

---

## 2. 전체 시스템 아키텍처

### 2.1 컴포넌트 개요

시스템은 크게 세 부분으로 나뉩니다.   

1. **Python 파이프라인 (python-pipeline 컨테이너)**  
   - 데이터 수집 (KIS, pykrx, OpenDART)  
   - 정제 및 피처/라벨/품질 점수 생성  
   - ML 모델 학습 및 예측  
   - 점수·랭킹·백테스트/페이퍼 트레이딩 산출  

2. **Node.js API + Web (node-api 컨테이너)**  
   - `/api/ranking`, `/api/stocks`, `/api/stocks/:code`, `/api/backtest` REST API 제공  
   - 정적 웹(리스트/상세) 제공 및 시장 상태/알림 표시  

3. **데이터 저장소 (data 디렉토리)**  
   - CSV, 모델 파일(model.pkl), 재무/특징/라벨/랭킹/백테스트 결과 등 모든 산출물 저장  

### 2.2 아키텍처 흐름 (텍스트 다이어그램)

```text
[외부 데이터 소스]
 ├─ 한국투자증권(KIS) API         (일별 가격 데이터)
 ├─ pykrx                         (KIS 실패 시 가격 데이터 폴백)
 └─ OpenDART API                  (재무제표/펀더멘털)

         │
         ▼
[python/run_pipeline.py]
  1) fetch_top_universe.py          → universe.csv (종목 리스트)
  2) download_prices_kis.py         → prices_daily_raw.csv
  3) clean_prices.py                → prices_daily_clean.csv
  4) create_adjusted_prices.py      → prices_daily_adjusted.csv
  5) fetch_fundamentals_dart.py     → fundamentals.csv
  6) quality_builder.py             → quality.csv (quality_score)
  7) feature_builder.py             → features.csv (기술 + 품질 피처)
  8) scoring.py                     → scores_final.csv (기술 점수)
  9) label_builder.py               → labels.csv (타깃/라벨)
 10) model_train.py                 → model.pkl (회귀+분류 모델)
 11) model_predict.py               → predictions.csv (예측 + 확률)
 12) ranking_builder.py             → ranking_final.csv (최종 랭킹)
         │
         ▼
[data/*.csv, model.pkl]

         │
         ▼
[node/index.js (node-api)]
 - /api/ranking: ranking_final.csv + 보조 정보 제공
 - /api/stocks, /api/stocks/:code: 개별/리스트 제공
 - /api/backtest: 백테스트 결과 제공
 - 웹 페이지(public/*.html) 렌더링

         │
         ▼
[사용자(투자자)]
 - 브라우저로 Top 랭킹/상세/백테스트 확인
 - 시장 상태 및 경고 메시지 확인
````

---

## 3. 프로젝트 구조

### 3.1 최상위 구조

````text
lee_trader_project/
├─ docker-compose.yml
├─ bootstrap.ps1
├─ .env
├─ .env.example
├─ PRD_Lee_trader_상세_설명서.md
├─ PRD_4.0_모델상세.md
├─ 운영 가이드라인.md
├─ 프로젝트구조.md
│
├─ data/          # 결과물 및 중간 산출물 (CSV, 모델, 로그성 데이터 등)
├─ python/        # 데이터 파이프라인 및 ML 모델 코드
├─ node/          # Node.js API 서버 및 웹 UI
├─ logs/          # 파이프라인 실행 로그
└─ scripts/       # 윈도우용 유틸/실행 스크립트
``` :contentReference[oaicite:13]{index=13}  

### 3.2 주요 디렉토리 별 역할

#### 3.2.1 `python/` – 데이터 & 모델 & 파이프라인

- **Dockerfile**: python-pipeline 컨테이너 빌드 정의  
- **requirements.txt**: 파이썬 의존성 패키지 목록  
- 핵심 스크립트 (일부):   

  - `run_pipeline.py` : 전체 파이프라인 오케스트레이션 (1번 실행으로 전체 단계 수행)  
  - `fetch_top_universe.py` : KOSPI/KOSDAQ 상위/필터링된 종목 유니버스 수집  
  - `download_prices_kis.py` : KIS API로 일별 가격데이터 수집 (실패 시 pykrx 대체 로직)  
  - `clean_prices.py` / `create_adjusted_prices.py` : 가격 데이터 정제 및 액면/배당 반영 조정  
  - `fetch_fundamentals_dart.py` : OpenDART로 연간 재무제표 수집 (corp_codes 캐시 포함)  
  - `quality_builder.py` : 재무 지표에서 quality_score 산출  
  - `feature_builder.py` : 기술적 지표 + 품질 지표 합쳐 features.csv 생성  
  - `scoring.py` : 가격 기반 기술점수(모멘텀/변동성 등) 계산  
  - `label_builder.py` : 60d/90d 수익률 및 상위20% 분류 라벨 생성  
  - `model_train.py` : 회귀/분류 LightGBM 모델 학습 및 model.pkl 패키징  
  - `model_predict.py` : model.pkl 을 이용하여 predictions.csv 생성  
  - `ranking_builder.py` : predictions + scores + quality 를 합쳐 ranking_final.csv 생성  
  - `paper_trading_tracker.py` / `paper_trading_report.py` : 페이퍼 트레이딩 + 리포트  

#### 3.2.2 `data/` – 산출물 저장소

대표 파일 예시:   

- `prices_daily_raw.csv`, `prices_daily_clean.csv`, `prices_daily_adjusted.csv`  
- `fundamentals.csv`, `quality.csv`, `features.csv`, `labels.csv`  
- `scores_final.csv`, `predictions.csv`, `ranking_final.csv`  
- `backtest_results_60d_top10.csv`  
- `paper_trades.csv`, `paper_trading_summary.csv`, `paper_trading_by_rank.csv`, `paper_trading_by_horizon.csv`  
- `universe.csv`, `sectors_template.csv`  
- `model.pkl`  
- `dart/corp_codes.xml` (OpenDART 기업 코드 캐시)  

#### 3.2.3 `node/` – Node.js API & Web

- **Dockerfile**: node-api 컨테이너 빌드 정의  
- **index.js**: 메인 서버 코드, Express 기반 REST API 제공  
  - `/api/health`, `/api/ranking`, `/api/stocks`, `/api/stocks/:code`, `/api/backtest` 구현 :contentReference[oaicite:16]{index=16}  
- **public/**  
  - `index.html` : 메인 리스트 UI (랭킹 및 필터/검색)  
  - `detail.html` : 개별 종목 상세 페이지  
  - 필요 시 추가적인 study/분석용 페이지 확장 가능  

#### 3.2.4 `logs/`

- `pipeline_YYYYMMDD_HHMMSS.log`  
  - 파이프라인 실행 당시 각 단계 진행 상황 및 에러/성능 정보를 저장  
  - 운영 중 트러블슈팅의 1차 확인 지점   

#### 3.2.5 `scripts/`

- `run_pipeline.cmd` : Windows에서 더블클릭으로 파이프라인 실행  
- `pack_for_home.ps1` : 집/다른 환경으로 프로젝트 이동 시 패킹 스크립트 (.env 제외 ZIP)   

---

## 4. 데이터 파이프라인 (최신 실행 순서)

파이프라인의 최신 정식 순서는 **운영 가이드라인 v4.1** 및 **PRD 4.1 – 모델 상세 설계서** 기준입니다.   

```text
run_pipeline.py
  1) fetch_top_universe.py       → data/universe.csv(code,name,market,sector)
  2) download_prices_kis.py      → data/prices_daily_raw.csv
  3) clean_prices.py             → data/prices_daily_clean.csv
  4) create_adjusted_prices.py   → data/prices_daily_adjusted.csv
  5) fetch_fundamentals_dart.py  → data/fundamentals.csv
  6) quality_builder.py          → data/quality.csv
  7) feature_builder.py          → data/features.csv
  8) scoring.py                  → data/scores_final.csv
  9) label_builder.py            → data/labels.csv
 10) model_train.py              → data/model.pkl
 11) model_predict.py            → data/predictions.csv
 12) ranking_builder.py          → data/ranking_final.csv
````

각 단계 요약:

1. **Universe 수집 (`fetch_top_universe.py`)**

   * KOSPI/KOSDAQ 전체 혹은 상위(시총, 거래대금 등) 중심 유니버스 구성
   * `universe.csv` : code, name, market, sector 포함

2. **가격 데이터 수집 (`download_prices_kis.py`)**

   * 기본: **KIS API** 활용
   * 실패 시: **pykrx**로 대체 수집
   * 결과: `prices_daily_raw.csv`

3. **가격 데이터 정제/보정**

   * `clean_prices.py` → 이상치/결측/비상장 기간 제거
   * `create_adjusted_prices.py` → 액면분할, 배당, 유상증자 등 반영한 adjusted price 생성

4. **재무 데이터 수집 (`fetch_fundamentals_dart.py`)**

   * OpenDART `fnlttSinglAcntAll` API 활용
   * 사업보고서(11014) 기준, CFS 우선/OFS 폴백
   * 연간 기준(YYYY-12-31)으로 재무 지표 산출
   * `fundamentals.csv` 생성

5. **품질 스코어 생성 (`quality_builder.py`)**

   * 재무 지표(ROE, op_margin, debt_ratio, ocf_to_assets, net_margin)를 z-score로 변환 후 가중합
   * clipping [-3, 3], inverse debt_ratio 등 적용
   * `quality.csv` : date,code,quality_score 

6. **피처 생성 (`feature_builder.py`)**

   * 가격 기반 피처:

     * close, ret_1d, mom_20, ma_5, ma_20, ma_60, close_over_ma20, vol_20, rsi_14, volume 등
   * 품질 결합:

     * (date, code) 기준 backward asof + ffill로 `quality_score` 병합
   * 결과: `features.csv` 

7. **기술 점수 (`scoring.py`)**

   * 모멘텀/변동성/추세 등 기술적 지표들을 점수화
   * 0 ~ 100 점수로 스케일링
   * `scores_final.csv`로 저장

8. **라벨 생성 (`label_builder.py`)**

   * 회귀 타깃:

     * target_60d = (close[t+60]/close[t]) - 1
     * target_90d = (close[t+90]/close[t]) - 1
   * 분류 타깃:

     * 같은 날짜 내 상위 20% 수익률을 1로 라벨링 (target_60d_top20 / target_90d_top20)
   * 결과: `labels.csv` 

9. **모델 학습 (`model_train.py`)**

   * features + labels를 결합하여 LightGBM 회귀/분류 모델 학습
   * 타깃 안정화를 위한 winsorization + hard clipping 수행
   * 회귀(60d/90d) + 분류(top20 60d/90d) 동시 학습
   * 결과를 `model.pkl`에 패키징 (모델/피처목록/타깃/클리핑 정보 포함) 

10. **예측 (`model_predict.py`)**

    * 각 종목마다 최신 날짜 1행을 사용
    * pred_return_60d, pred_return_90d
    * prob_top20_60d, prob_top20_90d 생성
    * `predictions.csv`에 저장 

11. **랭킹 생성 (`ranking_builder.py`)**

    * predictions + scores_final + quality + universe 정보를 조합
    * tech_score, pred_score, prob_score, qual_score 개별 점수 생성
    * 가중합으로 final_score 산출 후 정렬
    * `ranking_final.csv` 생성

---

## 5. 데이터 스키마 상세

여기서는 **핵심 CSV 파일들의 컬럼 구조**만 요약합니다.

### 5.1 `fundamentals.csv`

* **컬럼**

  * date (YYYY-12-31)
  * code (6자리 문자열)
  * roe
  * op_margin
  * debt_ratio
  * ocf_to_assets
  * net_margin

### 5.2 `quality.csv`

* **컬럼**

  * date
  * code
  * quality_score

### 5.3 `features.csv`

* **키**: (date, code)
* **대표 컬럼 예시**

  * date, code
  * close
  * ret_1d
  * mom_20
  * ma_5, ma_20, ma_60
  * close_over_ma20
  * vol_20
  * rsi_14
  * volume
  * quality_score

### 5.4 `labels.csv`

* **키**: (date, code)
* **컬럼**

  * target_60d
  * target_90d
  * target_60d_top20 (0/1)
  * target_90d_top20 (0/1)

### 5.5 `predictions.csv`

* **컬럼**

  * date, code
  * pred_return_60d
  * pred_return_90d
  * prob_top20_60d
  * prob_top20_90d

### 5.6 `ranking_final.csv`

* **컬럼 예시**

  * date, code, name, market, sector
  * close
  * pred_return_60d, pred_return_90d
  * prob_top20_60d, prob_top20_90d
  * tech_score, pred_score, prob_score, qual_score
  * final_score

---

## 6. 모델 상세 설계

(요약은 PRD 4.1 – 모델 상세 설계서 기준) 

### 6.1 타깃 안정화 (Winsorization & Hard Clipping)

* target_60d:

  * 퍼센타일: (2, 98)
  * 하드 클립: [-0.5, 0.8]
* target_90d:

  * 퍼센타일: (5, 95)
  * 하드 클립: [-0.7, 1.0]

퍼센타일 기반 bounds ∩ 하드 bounds 교집합으로 최종 클리핑 범위를 결정.
날짜 기반 TimeSeriesSplit 환경에서 각 fold의 train 데이터 분포로 bounds 계산 후 train/val 모두에 동일 적용(데이터 누설 방지).

### 6.2 교차 검증

* **전략**: 고유 날짜 기준 TimeSeriesSplit (최대 5분할)

* 데이터 부족 시:

  * 80/20 단일 분할
  * 극단적으로 부족하면 CV 생략하고 전체 학습만 수행

* **회귀 메트릭**

  * RMSE, MAE

* **분류 메트릭**

  * Accuracy, ROC AUC (단, 단일 클래스 폴드 발생 시 AUC NaN 허용)
  * LogLoss 등 추가 가능

### 6.3 모델 구조

#### 6.3.1 회귀 (Regressor)

* LightGBM Regressor

  * n_estimators=800
  * learning_rate=0.03
  * num_leaves=64
  * subsample=0.8
  * colsample_bytree=0.8
  * random_state=42, n_jobs=-1

#### 6.3.2 분류 (Classifier)

* LightGBM Classifier

  * objective="binary"
  * n_estimators=600
  * learning_rate=0.03
  * num_leaves=64
  * subsample=0.8
  * colsample_bytree=0.8
  * random_state=42, n_jobs=-1
* 향후 옵션:

  * class_weight="balanced"
  * 임계치 최적화/캘리브레이션 가능

### 6.4 최종 패키징 (`model.pkl`)

```python
pack = {
  "reg_models": { "target_60d": reg60, "target_90d": reg90 },
  "cls_models": { "target_60d_top20": cls60, "target_90d_top20": cls90 },
  "features": feature_cols_used,
  "targets_reg": ["target_60d", "target_90d"],
  "targets_cls": ["target_60d_top20", "target_90d_top20"],
  # 레거시 호환용
  "models": { "target_60d": reg60, "target_90d": reg90 },
  "targets": ["target_60d", "target_90d"],
  # 메타데이터
  "winsor_percentiles": { ... },
  "hard_clip_bounds": { ... },
  "winsor_bounds": { ... },
}
```

* model_predict.py에서 **구버전 패키지와도 호환 가능**하도록 설계 

---

## 7. 랭킹 및 점수 산식

### 7.1 개별 컴포넌트

입력:

* `predictions.csv` : pred_return_60d/90d, prob_top20_60d/90d
* `scores_final.csv` : 기술 점수
* `features.csv` : quality_score, close
* `universe.csv` : name, market, sector

산출 점수:

* **tech_score**: scores_final.score (0~100 clip)
* **pred_score**: pred_return_60d의 날짜 내 백분위(0~100)
* **prob_score**: prob_top20_60d * 100 (0~100 clip)
* **qual_score**: quality_score의 날짜 내 백분위(0~100)

### 7.2 최종 산식

```text
final_score = 0.30 * tech_score
             + 0.30 * pred_score
             + 0.25 * prob_score
             + 0.15 * qual_score
```

* 정렬: (date asc, final_score desc)
* Node API에서는 final_score를 `score`라는 이름으로도 제공하여 레거시 호환

---

## 8. Node.js API & Web

### 8.1 API 엔드포인트

운영 가이드라인 기준: 

1. `GET /api/health`

   * 시스템 상태 점검용 (status, message, demo 여부)

2. `GET /api/ranking?market=KOSPI|KOSDAQ|ALL&sector=...`

   * 반환 예시 컬럼:

     * date, code, name, market, sector
     * close
     * pred_return_60d
     * prob_top20_60d
     * tech_score, pred_score, prob_score, qual_score, final_score
   * 기본 정렬: `final_score desc`

3. `GET /api/stocks?market=...&sector=...`

   * 주식 리스트 기반:

     * date, code, name, market, sector
     * close, ret_3m, pred_return_60d, pred_return_90d, score

4. `GET /api/stocks/:code?limit=90`

   * 개별 종목 최근 N일(기본 90일) 데이터
   * 예:

     * code, name, count, latest, pred_return_60d, pred_return_90d, rows[...]

5. `GET /api/backtest?file=backtest_results_60d_top10.csv`

   * 백테스트 결과 파일을 읽어 테이블 형태로 반환

### 8.2 Web UI 기능

* **리스트 페이지 (`/`, `index.html`)**

  * `/api/ranking` 기반
  * 최종 score 기준 정렬
  * 검색/시장/섹터 필터 제공
  * 컬럼: 순위, 시장, 섹터, 종목명, 코드, 현재가, 3개월 수익률, 예측 수익률(60d/90d), 점수
  * 색상 규칙: 양수(red), 음수(blue)

* **상세 페이지 (`/detail.html?code=005930`)**

  * `/api/stocks/:code` 기반
  * 최근 90일 가격/지표/예측 수익률을 차트/테이블로 표시

---

## 9. 실행 및 운영 가이드

### 9.1 환경 준비

* OS: Windows 11 (WSL2 권장)
* 필수 도구:

  * Docker Desktop
  * Python 3.10+ (로컬 파이프라인 실행 시)
  * VS Code / Git (선택)
* `.env` 설정:

  * KIS 관련 API 키 (KIS_APP_KEY, KIS_APP_SECRET 등)
  * DART_API_KEY
  * SYMBOLS (fallback용, 기본은 universe.csv 사용) 

### 9.2 파이프라인 실행

* **로컬 (Python 직접 실행)**

  ```bash
  python python/run_pipeline.py
  ```

* **Docker 컨테이너에서 실행**

  ```bash
  docker compose build python-pipeline
  docker compose run --rm python-pipeline
  ```
  * 파이프라인 실행 후 CSV 결과를 SQLite DB(`data/lee_trader.db`)로 반영하려면:
  
    ```bash
    python migrate_to_sqlite.py
    ```

### 9.3 Node API/웹 서버 실행

```bash
# 빌드 및 백그라운드 실행
docker compose up -d --build node-api
docker compose build --no-cache --progress=plain node-api
docker compose up -d node-api


# 컨테이너/이미지 정리(선택)
docker compose down node-api

# 캐시 없이 리빌드
docker compose build --no-cache node-api

# 재기동
docker compose up -d node-api

# 도커 관련 명령어
실시간 로그 보기
docker compose logs -f node-api

# 접속
http://localhost:3000


```

* 컨테이너 접속:

  ````bash
  docker exec -it lee_trader_app bash
  ``` :contentReference[oaicite:34]{index=34}  
  ````



### 9.4 스케줄링 (선택)

* Windows 작업 스케줄러에 등록 예시:

  * 매일 07:00 실행:

    * `python python/run_pipeline.py`
    * 또는 `docker compose run --rm python-pipeline`

---

## 10. 페이퍼 트레이딩 및 백테스트

### 10.1 전제 조건

* `predictions.csv`, `labels.csv` 생성 완료
* `universe.csv` 가 있으면 이름/시장/섹터 병합 가능 

### 10.2 실행 예시

* 도커 빌드:

  ```bash
  docker compose build python-pipeline
  ```
  * 파이프라인 실행 후 CSV → SQLite DB 반영:

    ```bash
    python migrate_to_sqlite.py
    ```

* 트래커 실행 (60일, Top10):

  ```bash
  docker compose run --rm python-pipeline \
    python python/paper_trading_tracker.py --horizon 60 --top-k 10
  ```

* 트래커 실행 (90일, Top10):

  ```bash
  docker compose run --rm python-pipeline \
    python python/paper_trading_tracker.py --horizon 90 --top-k 10
  ```

* 리포트 생성:

  ```bash
  docker compose run --rm python-pipeline \
    python python/paper_trading_report.py
  ```

### 10.3 산출 파일

* `data/paper_trades.csv`
* `data/paper_trading_with_returns.csv`
* `data/paper_trading_summary.csv`
* `data/paper_trading_by_rank.csv`
* `data/paper_trading_by_horizon.csv` 

---

## 11. 보안 및 컴플라이언스

* `.env` 파일(각종 API 키 포함)은 **절대 Git에 커밋 금지**
* 로그에 API 키/민감 정보가 출력되지 않도록 주의
* 외부 API (KIS, DART)의 **호출 제한/약관 준수**
* 프로젝트 이관 시:

  * `scripts/pack_for_home.ps1` 사용
  * 항상 `.env` 제외하고 전달

---

## 12. 향후 확장 방향 (요약 로드맵)

> 아래는 현재 PRD와 모델 설계를 기반으로 한 **향후 확장 아이디어**입니다.
> 구현 여부는 추후 버전에서 선택적으로 진행.

1. **스타일 팩터 추가**

   * Value, Growth, Momentum, Quality, Size 등 팩터 스코어 도입
   * 포트폴리오 스타일 노출 및 전략 세분화

2. **포트폴리오 자동 구성**

   * 리스크(변동성, 상관관계, MDD) 기반 비중 최적화
   * TopN + 분산 전략 자동 포트 구성

3. **실매매 모드 (Live Trading Engine)**

   * 증권사 API(실거래) 연결
   * 매수/매도 알고리즘, 손절/익절, 포지션 관리 로직 추가

4. **실시간(WebSocket) 데이터 반영**

   * 장중 가격/호가/체결 정보 반영
   * 실시간 신호/경고 알림 UI

5. **모델 고도화**

   * Optuna 기반 하이퍼파라미터 튜닝 자동화
   * 시계열 전용 모델(LSTM, Temporal Fusion Transformer 등) 시범 도입
   * 뉴스/공시/섹터/거시 데이터 추가

6. **리포트/설명 자동화**

   * LLM 활용 종목 리포트 자동 생성
   * “이 종목이 점수가 높은 이유 / 위험 요소” 설명 텍스트 자동 생성

---

이 문서는 **지금 네 프로젝트의 실제 코드/구조/문서들을 4.1 기준으로 통합한 최종 스냅샷**이야.
앞으로 버전업을 할 때는,

* 이 문서의 **섹션 단위(파이프라인/모델/랭킹/API/운영)** 를 기준으로
* “어디를 바꿨는지” 를 주석처럼 남겨가면,
  나중에 돌아봐도 헷갈리지 않을 거야.

필요하면 이걸 **`PRD_Lee_trader_v4.1_final.md`** 같은 이름으로 프로젝트 루트에 바로 넣고 써도 된다.

```
```
