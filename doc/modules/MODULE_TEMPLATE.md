# Module Template

이 문서는 새 모듈 문서를 만들 때 사용하는 표준 템플릿입니다.

## Required Files

각 모듈은 최소 아래 4개 파일을 가집니다.

- `README.md`
- `CONTEXT.md`
- `FLOW.md`
- `FILE_INDEX.md`

필요 시 아래 파일을 추가합니다.

- `ENV.md`
- `OPERATIONS.md`
- `RUNTIME.md`
- `CHANGELOG.md`

## README.md

권장 섹션:

- 모듈 목적
- 포함 범위
- 관련 산출물
- 읽는 순서
- 연관 문서 링크

예시:

```md
# Module Name

## Purpose

이 모듈이 무엇을 담당하는지 설명합니다.

## Scope

- 포함 기능
- 제외 기능

## Outputs

- 주요 JSON/CSV/MD 산출물

## Read First

- CONTEXT.md
- FLOW.md
- FILE_INDEX.md
```

## CONTEXT.md

권장 섹션:

- 왜 이 모듈이 존재하는가
- 운영상 제약
- 안전장치
- 다른 모듈과의 경계
- 실수하기 쉬운 포인트

## FLOW.md

권장 섹션:

- 입력 데이터
- 실행 순서
- 주요 스크립트
- 출력 산출물
- 예외/차단 조건

흐름은 가능하면 다음 형태로 씁니다.

```md
입력 -> 전처리 -> 핵심 판단 -> 주문/저장/리포트 -> 화면 반영
```

## FILE_INDEX.md

권장 섹션:

- 핵심 Python 파일
- 관련 Node/프론트 파일
- 관련 설정 파일
- 관련 outputs
- 관련 docs

파일 목록은 설명과 함께 유지합니다.

예시:

```md
- python/example.py: 주문 가드 계산
- node/public/example.js: 화면 렌더링
- outputs/example.json: 최신 산출물
```

## ENV.md

환경변수가 5개 이상이거나, 안전 관련 변수가 있으면 추가합니다.

권장 컬럼:

| 변수명 | 기본값 | 설명 | 영향 범위 | 안전 주의 |
| --- | --- | --- | --- | --- |

## OPERATIONS.md

운영 절차가 있으면 추가합니다.

권장 섹션:

- 실행 명령어
- 점검 순서
- 장애 시 복구
- 롤백 방법

## CHANGELOG.md

사람이 읽는 운영 변경 기록이 필요하면 추가합니다.

권장 포맷:

```md
## 2026-05-06

- before-open guard 강화
- live account sync payload 반영
```
