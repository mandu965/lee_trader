아래 내용을 그대로 Claude/Codex에 전달하면 됩니다.
이번 4단계는 **실반영이 아니라 검증 단계**입니다. 핵심은 “미국 Macro Overlay가 실제로 손실을 줄였는지 / 수익률을 개선했는지 / 기회손실만 만든 건 아닌지”를 백테스트로 확인하는 것입니다.

````markdown
# 작업명: 미국 매크로 신호 기반 국내 RULE 강화 - Phase 4 백테스트 검증

이전 Phase에서 아래 작업이 완료되었습니다.

1. 미국 ETF / 지수 데이터 수집
2. `us_macro_feature_daily` 생성
3. 국내 추천 / RULE 후보에 대한 Shadow Mode overlay 로그 생성
4. `kr_macro_overlay_log` 저장

이번 Phase 4의 목표는 **미국 macro overlay를 실제로 적용했다고 가정했을 때 기존 국내 전략 대비 성과가 개선되는지 백테스트로 검증하는 것**입니다.

중요:
이번 Phase에서도 실제 매수 / 매도 주문에는 영향을 주면 안 됩니다.
기존 추천 점수, 주문 후보, 주문 실행 로직을 변경하면 안 됩니다.
이번 작업은 백테스트 / 리포트 생성 전용입니다.

---

## 1. 반드시 먼저 확인할 파일

아래 파일을 먼저 확인해주세요.

- doc/modules/Lee_trader_ai/CONTEXT.md
- doc/modules/Lee_trader_ai/OPERATIONS.md
- doc/modules/Lee_trader_ai/ENV.md
- Phase 1~2에서 추가한 미국 macro 관련 파일
- Phase 3에서 추가한 Shadow Mode overlay 관련 파일
- signal_builder.py
- run_daily_scheduler.py
- rule 관련 매수 후보 생성 파일
- 추천 랭킹 생성 파일
- 기존 백테스트 관련 파일
- 기존 성과 분석 / KPI / 리포트 관련 파일
- schema.sql
- .env.example

경로가 다르면 프로젝트 내에서 유사 역할 파일을 찾아서 확인해주세요.

---

## 2. 이번 작업 목표

이번 Phase의 목표는 다음입니다.

1. 기존 전략 성과를 기준선으로 계산한다.
2. 미국 macro overlay를 적용했다고 가정한 전략 성과를 계산한다.
3. 두 전략을 비교한다.
4. Risk-Off 차단 효과를 검증한다.
5. 반도체 가산 / 감점 효과를 검증한다.
6. 매매 횟수 감소, 기회손실, MDD 개선 여부를 확인한다.
7. 결과를 DB 또는 리포트 파일로 저장한다.
8. 실제 주문 / 추천 / RULE 실행 로직에는 영향을 주지 않는다.

---

## 3. 절대 제외 범위

아래 작업은 이번 Phase에서 절대 하지 마세요.

1. 실제 매수 / 매도 주문 로직 변경
2. 실제 주문 후보 변경
3. 기존 final_score 변경
4. 기존 추천 랭킹 테이블 update
5. 기존 RULE BUY 후보 테이블 update
6. order 관련 테이블 insert / update / delete
7. 실제 order 로직에서 adjusted_score 사용
8. `US_MACRO_ALLOW_REAL_APPLY=true` 전제 구현
9. 자동매매 스케줄러에 강제 편입
10. 실매매 반영 로직 구현
11. 실시간 자동매매 판단 흐름 변경

이번 Phase는 **백테스트 계산과 리포트 생성만** 허용합니다.

---

## 4. 백테스트 비교 대상

아래 두 전략을 비교해주세요.

### A. 기존 전략

기존 국내 추천 / RULE 후보 기준으로 성과를 계산합니다.

예:

```text
기존 final_score 기준 TopN 매수
기존 RULE BUY 후보 기준 매수
기존 전략의 매수 / 매도 기준 유지
````

### B. Macro Overlay 적용 가정 전략

Phase 3에서 생성한 `kr_macro_overlay_log`를 활용하여 overlay가 적용되었다고 가정한 성과를 계산합니다.

예:

```text
adjusted_score 기준 TopN 재정렬
buy_blocked_flag = 'Y' 후보 제외 가정
macro_adjustment 반영 후 성과 계산
```

중요:
이 계산은 백테스트 전용입니다.
실제 추천 테이블이나 주문 후보 테이블을 변경하면 안 됩니다.

---

## 5. 백테스트 범위

가능하면 아래 기간을 지원해주세요.

```text
기본: 최근 1년
옵션: 최근 3개월, 6개월, 1년, 전체 가능 기간
```

실행 옵션 예시:

```bash
python backtest/backtest_us_macro_overlay.py --start-date 2025-01-01 --end-date 2026-05-08
```

또는 프로젝트 구조에 맞게:

```bash
python -m backtest.backtest_us_macro_overlay --start-date 2025-01-01 --end-date 2026-05-08
```

날짜 옵션을 지정하지 않으면 최근 1년 기준으로 실행되게 해주세요.

---

## 6. 백테스트 기준

기존 시스템의 매수 / 매도 기준이 있으면 우선 그대로 사용해주세요.

기존 백테스트 엔진이 있다면 이를 재사용하세요.

기존 백테스트 엔진이 없다면 최소 기준으로 아래 방식을 구현해주세요.

### 기본 매수 기준

```text
매일 국내 추천 후보 또는 RULE BUY 후보를 기준으로 TopN 선정
TopN 기본값: 5 또는 10
동일 비중 매수 가정
```

### Overlay 적용 전략 매수 기준

```text
buy_blocked_flag = 'Y' 후보 제외
adjusted_score 기준 재정렬
TopN 선정
동일 비중 매수 가정
```

### 보유 기간 기준

다음 보유 기간별 성과를 계산해주세요.

```text
1D
5D
20D
60D
```

매도 로직이 기존 시스템에 있으면 기존 매도 로직도 함께 비교할 수 있게 해주세요.

초기에는 보유기간 고정 방식으로 먼저 검증해도 됩니다.

---

## 7. 검증 지표

아래 지표를 반드시 계산해주세요.

### 수익률 관련

```text
누적 수익률
평균 거래 수익률
중앙값 거래 수익률
1D 평균 수익률
5D 평균 수익률
20D 평균 수익률
60D 평균 수익률
Top5 수익률
Top10 수익률
```

### 리스크 관련

```text
MDD
변동성
손실 거래 비율
최대 손실 거래
평균 손실 거래
```

### 승률 관련

```text
전체 승률
1D 승률
5D 승률
20D 승률
60D 승률
```

### 거래 빈도 관련

```text
매매 횟수
매수 차단 횟수
Risk-Off 차단 횟수
반도체 가산 적용 횟수
반도체 감점 적용 횟수
기회손실 발생 횟수
```

### Overlay 효과 관련

```text
기존 전략 대비 수익률 차이
기존 전략 대비 MDD 차이
기존 전략 대비 승률 차이
기존 전략 대비 매매 횟수 차이
손실 회피 효과
기회 손실 효과
```

---

## 8. 핵심 검증 질문

리포트에는 반드시 아래 질문에 대한 답을 포함해주세요.

```text
1. Risk-Off 차단이 실제 손실을 줄였는가?
2. Risk-Off 차단으로 인해 좋은 매수 기회를 놓친 경우는 얼마나 되는가?
3. Macro Overlay 적용 후 MDD가 개선되었는가?
4. Macro Overlay 적용 후 누적 수익률이 개선되었는가?
5. Macro Overlay 적용 후 승률이 개선되었는가?
6. 반도체 가산 룰이 실제 수익률 개선으로 이어졌는가?
7. 반도체 감점 룰이 실제 손실 회피에 도움이 되었는가?
8. 매매 횟수만 줄고 수익도 같이 줄어든 것은 아닌가?
9. Overlay가 기존 추천 랭킹을 과도하게 왜곡하지 않았는가?
10. 실반영해도 될 만큼 통계적으로 의미 있는 결과인가?
```

---

## 9. Risk-Off 차단 효과 분석

`buy_blocked_flag = 'Y'`였던 후보를 별도로 분석해주세요.

분석 항목:

```text
차단 후보 수
차단 후보의 실제 이후 1D / 5D / 20D / 60D 수익률
차단했을 때 피한 손실
차단했기 때문에 놓친 수익
차단 후보 중 실제 하락한 비율
차단 후보 중 실제 상승한 비율
```

결론은 다음 형태로 정리해주세요.

```text
Risk-Off 차단은 손실 회피에 도움이 되었는가?
차단 기준이 너무 강한가?
차단 기준이 너무 약한가?
실반영 후보로 적합한가?
```

---

## 10. 반도체 룰 효과 분석

반도체 관련 overlay를 별도로 분석해주세요.

분석 대상:

```text
semiconductor_strength_flag = 'Y'
semiconductor_ret_1d >= US_MACRO_SEMI_POSITIVE_RET
semiconductor_ret_1d <= US_MACRO_SEMI_NEGATIVE_RET
반도체 섹터 후보
```

분석 항목:

```text
반도체 가산 적용 후보 수
반도체 감점 적용 후보 수
가산 후보의 이후 1D / 5D / 20D / 60D 수익률
감점 후보의 이후 1D / 5D / 20D / 60D 수익률
가산 룰이 수익률 개선에 기여했는지
감점 룰이 손실 회피에 기여했는지
```

---

## 11. 데이터 정합성 체크

백테스트 전 반드시 데이터 정합성을 확인해주세요.

확인 항목:

```text
us_macro_feature_daily에 대상 기간 데이터가 충분히 있는가?
kr_macro_overlay_log에 대상 기간 데이터가 충분히 있는가?
국내 추천 / RULE 후보 데이터가 대상 기간에 존재하는가?
국내 종목 가격 데이터가 대상 기간에 존재하는가?
us_trade_date와 kr_apply_date가 정상 매핑되어 있는가?
DATA_INCOMPLETE / STALE_DATA / ERROR 상태가 얼마나 많은가?
```

정합성 문제가 있으면 백테스트를 무리하게 진행하지 말고, 경고를 출력하고 리포트에 기록해주세요.

---

## 12. 결과 저장 방식

가능하면 결과를 DB와 파일 둘 다 저장해주세요.

### 12-1. DB 저장

기존 스타일에 맞춰 신규 테이블을 추가하거나 기존 성과 테이블을 재사용해주세요.

권장 테이블 예시:

```sql
CREATE TABLE IF NOT EXISTS research.us_macro_overlay_backtest_result (
    run_id             VARCHAR(100) NOT NULL,
    start_date         DATE NOT NULL,
    end_date           DATE NOT NULL,
    strategy_name      VARCHAR(100) NOT NULL,

    top_n              INTEGER,
    holding_days       INTEGER,

    total_return       NUMERIC(18,6),
    avg_return         NUMERIC(18,6),
    median_return      NUMERIC(18,6),
    win_rate           NUMERIC(10,4),
    mdd                NUMERIC(18,6),
    volatility         NUMERIC(18,6),

    trade_count        INTEGER,
    blocked_count      INTEGER,
    avoided_loss       NUMERIC(18,6),
    missed_gain        NUMERIC(18,6),

    summary            TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (run_id, strategy_name, top_n, holding_days)
);
```

DB schema를 사용하지 않는 프로젝트라면 기존 규칙에 맞게 prefix 방식으로 조정해주세요.

예:

```text
research_us_macro_overlay_backtest_result
```

### 12-2. 파일 저장

아래 형태로 결과 파일을 생성해주세요.

```text
reports/us_macro_overlay_backtest_YYYYMMDD_HHMMSS.md
reports/us_macro_overlay_backtest_YYYYMMDD_HHMMSS.csv
```

가능하면 Markdown 리포트를 우선 생성해주세요.

---

## 13. 리포트 형식

Markdown 리포트에는 최소 아래 내용을 포함해주세요.

```markdown
# US Macro Overlay Backtest Report

## 1. 실행 정보
- 실행일
- 분석 기간
- 사용 데이터
- TopN 기준
- 보유 기간 기준

## 2. 데이터 정합성 점검
- us_macro_feature_daily 건수
- kr_macro_overlay_log 건수
- 국내 후보 데이터 건수
- 가격 데이터 건수
- 누락 데이터 여부

## 3. 기존 전략 성과
- 누적 수익률
- 평균 수익률
- 승률
- MDD
- 매매 횟수

## 4. Overlay 적용 가정 전략 성과
- 누적 수익률
- 평균 수익률
- 승률
- MDD
- 매매 횟수

## 5. 기존 전략 vs Overlay 전략 비교
- 수익률 차이
- MDD 차이
- 승률 차이
- 매매 횟수 차이

## 6. Risk-Off 차단 효과
- 차단 횟수
- 피한 손실
- 놓친 수익
- 실제 하락 비율
- 실제 상승 비율

## 7. 반도체 가산 / 감점 효과
- 가산 후보 성과
- 감점 후보 성과
- 효과 판단

## 8. 결론
- 실반영 가능 여부
- 추가 검증 필요 여부
- 권장 조정 사항

## 9. 다음 단계
- Phase 5 실반영 전 해야 할 작업
```

---

## 14. 환경변수 추가

필요하면 `.env.example`에 아래 값을 추가해주세요.

```env
# US Macro Overlay Backtest
US_MACRO_BACKTEST_ENABLED=true
US_MACRO_BACKTEST_DEFAULT_LOOKBACK_DAYS=365
US_MACRO_BACKTEST_TOP_N=5,10
US_MACRO_BACKTEST_HOLDING_DAYS=1,5,20,60
US_MACRO_BACKTEST_REPORT_DIR=reports
```

단, 이 환경변수들은 백테스트 전용입니다.
실제 자동매매에는 영향을 주면 안 됩니다.

---

## 15. 실행 방식

단독 실행 가능해야 합니다.

예시:

```bash
python backtest/backtest_us_macro_overlay.py --start-date 2025-01-01 --end-date 2026-05-08 --top-n 5,10 --holding-days 1,5,20,60
```

또는 프로젝트 구조에 맞게:

```bash
python -m backtest.backtest_us_macro_overlay --start-date 2025-01-01 --end-date 2026-05-08 --top-n 5,10 --holding-days 1,5,20,60
```

옵션을 지정하지 않으면 기본값으로 실행되게 해주세요.

기본값:

```text
lookback_days = 365
top_n = 5,10
holding_days = 1,5,20,60
```

---

## 16. 테스트 조건

최소 다음 테스트를 포함해주세요.

1. 대상 기간에 대해 백테스트가 정상 실행되는지 확인
2. 기존 전략 성과가 계산되는지 확인
3. Overlay 적용 가정 전략 성과가 계산되는지 확인
4. Top5 / Top10 각각 결과가 생성되는지 확인
5. 1D / 5D / 20D / 60D 각각 결과가 생성되는지 확인
6. Risk-Off 차단 후보 성과가 별도로 계산되는지 확인
7. 반도체 가산 / 감점 후보 성과가 별도로 계산되는지 확인
8. 데이터 누락 시 경고를 출력하고 안전하게 종료하는지 확인
9. 결과 Markdown 리포트가 생성되는지 확인
10. 결과 CSV가 생성되는지 확인
11. DB 저장을 구현한 경우 결과 테이블에 정상 저장되는지 확인
12. 실제 주문 / 추천 / RULE 테이블이 변경되지 않는지 확인

---

## 17. 완료 조건

작업 완료 후 다음을 정리해주세요.

1. 추가 / 수정된 파일 목록
2. 추가된 DB 테이블 DDL
3. 추가된 환경변수
4. 백테스트 실행 방법
5. 생성되는 리포트 예시
6. 계산되는 지표 목록
7. 기존 전략 vs Overlay 전략 비교 결과 예시
8. Risk-Off 차단 효과 분석 예시
9. 반도체 가산 / 감점 효과 분석 예시
10. 실제 주문 영향 여부
11. Phase 5 실반영 전 검토해야 할 항목

특히 실제 주문 영향 여부는 명확히 적어주세요.

최종 보고에는 반드시 아래 문장을 포함해주세요.

```text
이번 Phase 4 작업은 미국 macro overlay의 효과를 백테스트로 검증하는 작업이며, 실제 주문 생성/실행 로직과 기존 추천 점수에는 영향을 주지 않습니다.
```

---

## 18. 최종 안전 원칙

이번 Phase의 핵심 원칙은 다음입니다.

```text
1. 읽기는 허용한다.
2. 백테스트 결과 저장은 허용한다.
3. 리포트 생성은 허용한다.
4. 기존 추천 점수 변경은 금지한다.
5. 기존 주문 후보 변경은 금지한다.
6. 실제 주문 생성 / 실행 로직 변경은 금지한다.
7. adjusted_score는 백테스트 계산용으로만 사용한다.
8. buy_blocked_flag는 백테스트 가정용으로만 사용한다.
9. Phase 5 실반영 여부는 백테스트 결과를 보고 별도로 결정한다.
10. 이번 Phase만으로 실매매 반영을 진행하지 않는다.
```

```

이 프롬프트로 진행하면 됩니다.  
다만 Claude가 작업 완료 후 결과를 주면, 바로 Phase 5로 가지 말고 **백테스트 리포트 결과를 먼저 검토**해야 합니다. 특히 `수익률은 개선됐지만 매매 횟수가 과도하게 줄었는지`, `MDD가 실제로 개선됐는지`, `Risk-Off 차단이 손실 회피인지 기회손실인지`를 먼저 봐야 합니다.
```
