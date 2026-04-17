# Ranking Snapshot Archive Design

## Goal

운영 파이프라인이 매일 생성하는 `data/ranking_final.csv`에서 최신 추천 결과를 장기 추적 가능한 형태로 고정 저장한다.

핵심 목적:

- 최신 운영 추천 Top20을 일자별로 누적 저장
- `rank <= 10`, `rank <= 5`를 같은 archive에서 직접 재구성 가능하게 유지
- forward return 추적, churn 분석, confidence calibration, 의미성 리포트에 재사용

## Output Structure

### 1. Full daily snapshot

- Path: `data/history/ranking/YYYYMMDD_ranking_final.csv`
- 역할:
  - 당일 전체 랭킹의 원본 보존
  - 추후 운영 상태 재현
  - 기존 snapshot 기반 리포트와 호환 유지

### 2. Compact archive table

- Path: `data/ranking_snapshot_archive.csv`
- 역할:
  - 최신 날짜 기준 Top20 추천만 누적 저장
  - 운영 추천 성과 추적용 장기 archive
  - Top10/Top5는 `rank <= 10`, `rank <= 5` 조건으로 복원

## Stored Columns

`data/ranking_snapshot_archive.csv`는 아래 고정 스키마를 사용한다.

| Column | Meaning |
| --- | --- |
| `asof_date` | 추천 기준일 |
| `rank` | 당일 추천 순위 |
| `code` | 종목 코드 |
| `name` | 종목명 |
| `final_score` | 운영 baseline 추천 점수 |
| `confidence_score` | confidence meta score |
| `ret_score` | return component |
| `prob_score` | probability component |
| `tech_score` | technical component |
| `quality_score` | quality component. 내부적으로 `qual_score`를 우선 사용 |
| `safety_score` | safety component |
| `risk_penalty` | risk deduction component |
| `dominant_theme` | 대표 theme |
| `theme_score` | theme raw score |
| `explain_text` | 운영 explain text |

## Top20 / Top10 / Top5 Policy

archive 본체는 `rank <= 20`까지만 저장한다.

이유:

- Top10과 Top5는 Top20의 strict subset이다.
- 한 archive에서 `WHERE rank <= 10`, `WHERE rank <= 5`로 바로 복원 가능하다.
- 같은 날짜를 Top20/Top10/Top5로 중복 저장하면 중복 row와 유지보수 비용만 늘어난다.

정의:

- Top20: `rank <= 20`
- Top10: `rank <= 10`
- Top5: `rank <= 5`

## Dedup / Upsert Policy

중복 방지 키는 아래 3개다.

- `asof_date`
- `rank`
- `code`

동작 원칙:

1. 같은 날짜 archive가 이미 있고 row 내용도 같으면 `skipped_existing`
2. 같은 날짜 archive가 있지만 재생성 내용이 다르면 해당 날짜 row를 교체
3. 다른 날짜는 append

즉 archive CSV는 append-only 성격을 유지하되, 동일 날짜에 대해서는 idempotent upsert처럼 동작한다.

## Pipeline Integration

운영 파이프라인은 이미 마지막 단계에서 `python/archive_ranking_snapshot.py`를 호출한다.

현재 연결점:

- `python/run_pipeline.py`
- `python/run_theme_shadow_daily.py`

따라서 별도 파이프라인 추가 없이 아래 흐름으로 자동 실행된다.

1. `ranking_builder.py`가 `data/ranking_final.csv` 생성
2. 마지막 archive step이 `python/archive_ranking_snapshot.py` 실행
3. full dated snapshot 저장
4. compact Top20 archive CSV upsert
5. metadata JSON 갱신

## Date Resolution

latest snapshot date는 아래 우선순위로 결정한다.

1. `as_of_date`
2. `date`
3. `trade_date`
4. `snapshot_date`
5. CLI `--as-of-date`

그 날짜에 해당하는 row만 골라 Top20 archive를 만든다.

## Compatibility Notes

- 기존 `data/history/ranking/*.csv` snapshot 소비 로직은 유지된다.
- compact archive는 신규 산출물이라 기존 리포트를 깨지 않는다.
- meta JSON에는 `top20_tickers`, `top10_tickers`, `top5_tickers`, archive path, archive row count를 추가 기록한다.

## Operational Usage

예시:

```powershell
python python/archive_ranking_snapshot.py --skip-if-exists
```

재생성:

```powershell
python python/archive_ranking_snapshot.py --overwrite
```

Top10 조회 예시:

```python
df = pd.read_csv("data/ranking_snapshot_archive.csv")
top10 = df[df["rank"] <= 10]
```

Top5 조회 예시:

```python
df = pd.read_csv("data/ranking_snapshot_archive.csv")
top5 = df[df["rank"] <= 5]
```
