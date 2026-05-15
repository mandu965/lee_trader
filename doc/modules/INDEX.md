# Module Index

`doc/modules` 아래 기능 문서의 최상위 진입점입니다.
작업자는 여기서 어떤 모듈 문서를 먼저 읽어야 하는지 판단하고, 변경 범위에 맞는 문서를 같이 갱신해야 합니다.

## 목적

- 변경 대상이 어느 모듈에 속하는지 빠르게 분류
- 각 모듈의 책임 범위와 핵심 문서 확인
- 코드 변경 시 같이 갱신해야 할 문서 기준 제공

## Modules

### Lee_trader_rule

- 목적: RULE 기반 자동매매와 `after-close -> before-open -> after-open` 운영 흐름 정리
- 위치: [Lee_trader_rule](</d:/ai/lee_trader/doc/modules/Lee_trader_rule>)
- 우선 문서:
  - [README.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/README.md>)
  - [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/CONTEXT.md>)
  - [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/FLOW.md>)
  - [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/FILE_INDEX.md>)
  - [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/ENV.md>)
  - [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_rule/OPERATIONS.md>)
- 운영 중요도: 매우 높음
- 갱신 트리거:
  - RULE 주문 가드 변경
  - scheduler-rule-* 흐름 변경
  - RULE 계좌 동기화, preview, execution 결과 구조 변경
  - RULE 화면/API payload 변경

### Lee_trader_ai

- 목적: AI 예측, 랭킹, 주문 preview, 실자동매매 흐름 정리
- 위치: [Lee_trader_ai](</d:/ai/lee_trader/doc/modules/Lee_trader_ai>)
- 우선 문서:
  - [README.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/README.md>)
  - [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/CONTEXT.md>)
  - [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FLOW.md>)
  - [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FILE_INDEX.md>)
  - [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/ENV.md>)
  - [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/OPERATIONS.md>)
  - [FINANCIAL_MOMENTUM_DESIGN.md](</d:/ai/lee_trader/doc/modules/Lee_trader_ai/FINANCIAL_MOMENTUM_DESIGN.md>) — 재무 모멘텀 설계 (2026-05-15 확정)
- 운영 중요도: 높음
- 갱신 트리거:
  - ranking/selection/order preview 기준 변경
  - AI 실주문 제출 흐름 변경
  - live account/fill sync 구조 변경
  - 메인 랭킹/자동매매 화면 변경
  - 재무 모멘텀 phase 분류 기준 변경
  - financial_momentum_score overlay 공식 변경

### Lee_trader_backTest

- 목적: walk-forward, prediction history, outcome, RULE 백테스트 검증 흐름 정리
- 위치: [Lee_trader_backTest](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest>)
- 우선 문서:
  - [README.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/README.md>)
  - [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/CONTEXT.md>)
  - [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/FLOW.md>)
  - [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/FILE_INDEX.md>)
  - [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/ENV.md>)
  - [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_backTest/OPERATIONS.md>)
- 운영 중요도: 높음
- 갱신 트리거:
  - walk-forward split 기준 변경
  - prediction/outcome 적재 구조 변경
  - RULE 포트폴리오 백테스트 규칙 변경
  - 비교 리포트 스키마 변경

### Lee_trader_score

- 목적: 최종 점수, 파생 점수, 운영 정렬 기준 정리
- 위치: [Lee_trader_score](</d:/ai/lee_trader/doc/modules/Lee_trader_score>)
- 우선 문서:
  - [README.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/README.md>)
  - [CONTEXT.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/CONTEXT.md>)
  - [FLOW.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/FLOW.md>)
  - [FILE_INDEX.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/FILE_INDEX.md>)
  - [RUNTIME_SORTING.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/RUNTIME_SORTING.md>)
  - [ENV.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/ENV.md>)
  - [OPERATIONS.md](</d:/ai/lee_trader/doc/modules/Lee_trader_score/OPERATIONS.md>)
- 운영 중요도: 높음
- 갱신 트리거:
  - `final_score` 수식 변경
  - `live_rank`, `rank_final` 기준 변경
  - confidence/theme overlay 해석 변경
  - 점수 설명 컬럼 또는 점수 검증 화면 변경

## Work Rules

기본 순서:

1. 관련 모듈의 `README.md`, `CONTEXT.md`, `FLOW.md`, `FILE_INDEX.md`를 먼저 읽습니다.
2. 입력, 처리, 출력, 운영 가드가 어디서 결정되는지 확인합니다.
3. 코드 수정 후 실제 실행이나 산출물 검증을 수행합니다.
4. 흐름, 환경변수, 출력, 운영 절차가 바뀌면 해당 모듈 문서를 같이 갱신합니다.
5. 모듈을 가로지르는 변경이면 `docs/` 아래 별도 문서를 추가합니다.

## Reference

- 공통 작성 템플릿: [MODULE_TEMPLATE.md](</d:/ai/lee_trader/doc/modules/MODULE_TEMPLATE.md>)
- 공통 문서 규칙: [DOCUMENTATION_RULES.md](</d:/ai/lee_trader/doc/modules/DOCUMENTATION_RULES.md>)
