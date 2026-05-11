# 10차 과제: theme overlay 활성화 시 알림/로그 추가

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 없음 (독립 진행 가능)
> 다음 과제: 없음 (독립 완결)

---

## 목적

현재 `ENABLE_THEME_OVERLAY` 환경변수 또는
`config/production_v1.yaml`의 `ranking.theme_overlay.enabled` 설정에 따라
운영 랭킹 정렬 기준이 `final_score`에서 `final_score_v3`로 조용히 바뀐다.

이 변경이 알림 없이 발생하면 운영자가 모르는 채로
다른 기준의 자동매매가 실행된다.

---

## 현재 구조 상세 파악 (소스 기준)

### ranking_builder.py — theme overlay 분기

```python
# _resolve_theme_overlay_runtime_flags()
live_uses_theme = operational and applied
# operational: LEE_TRADER_RUNTIME_MODE == "operational"
# applied: theme overlay mode가 "applied" 상태

# build_ranking() 내부
live_score_col = "final_score_v3" if live_uses_theme else "final_score"
live_rank = rank by live_score_col
rank_final = live_rank  # 동일값
```

### 활성화 조건

```python
# ENABLE_THEME_OVERLAY 환경변수
# config/production_v1.yaml: ranking.theme_overlay.enabled: false (현재 기본)
# THEME_OVERLAY_MODE 환경변수
```

현재 기본 설정: `live_uses_theme = false` → `final_score` 기준

### 현재 로그 상태

`ranking_builder.py`에 `logging.info()`는 많지만
theme overlay 활성화/비활성화 시점에
**명시적인 WARNING 레벨 로그나 알림이 없다.**

`live_uses_theme = True`가 되는 순간에 대한 별도 알림 코드가 없다.

---

## 변경 방향

### Step 1. ranking_builder.py 수정 — 구조화 로그 추가

`_resolve_theme_overlay_runtime_flags()` 또는
`build_ranking()` 내에서 `live_uses_theme` 결정 직후 로그를 추가한다.

```python
# live_uses_theme가 결정된 직후

if live_uses_theme:
    logging.warning(
        "THEME_OVERLAY_ACTIVATED | date=%s | live_score_col=final_score_v3 | "
        "rank_final=final_score_v3 기준 | trigger=%s",
        as_of_date,
        theme_overlay_trigger_info  # 어떤 설정이 활성화를 유발했는지
    )
else:
    logging.info(
        "THEME_OVERLAY_INACTIVE | date=%s | live_score_col=final_score | "
        "rank_final=final_score 기준",
        as_of_date,
    )
```

활성화 시 WARNING, 비활성화 시 INFO로 레벨을 구분한다.

### Step 2. ranking_builder.py 수정 — notifier 알림 추가

`live_uses_theme = True`가 되는 경우 notifier 경고 발송.

```python
from notifier import notify_warning

if live_uses_theme:
    notify_warning(
        title="[랭킹] theme overlay 활성화 — 정렬 기준 변경됨",
        message=(
            f"{as_of_date} 랭킹 정렬 기준이 "
            f"final_score → final_score_v3로 변경되었습니다."
        ),
        details={
            "as_of_date": as_of_date,
            "live_score_col": "final_score_v3",
            "rank_final_basis": "final_score_v3",
            "theme_overlay_mode": resolved_mode,
            "enable_theme_overlay": enable_theme_overlay_raw,
            "config_enabled": config_enabled,
            "action_note": (
                "자동매매 BUY 후보 정렬이 final_score_v3 기준으로 실행됩니다. "
                "의도한 변경인지 확인하세요."
            )
        }
    )
```

### Step 3. ranking_trust_summary.json에 기준 컬럼 추가

1차 과제에서 생성되는 `outputs/ranking_trust_summary.json`에
현재 적용 중인 정렬 기준 컬럼을 포함한다.

```json
{
  ...
  "ranking_sort_basis": {
    "live_score_col": "final_score",
    "live_uses_theme": false,
    "theme_overlay_mode": "off",
    "rank_final_col": "final_score"
  }
}
```

1차 과제가 완료된 경우 해당 파일을 수정하고,
미완료인 경우 별도 파일 `outputs/ranking_sort_basis.json`에 저장한다.

### Step 4. 이전 기준과 비교 로그 추가

직전 실행의 `live_score_col`을 읽어,
현재와 다르면 추가 WARNING 로그를 출력한다.

```python
# previous_sort_basis = 직전 ranking_sort_basis.json 읽기
if previous_live_score_col and previous_live_score_col != current_live_score_col:
    logging.warning(
        "THEME_OVERLAY_CHANGED | date=%s | "
        "previous=%s | current=%s | 정렬 기준이 변경되었습니다",
        as_of_date,
        previous_live_score_col,
        current_live_score_col,
    )
    notify_warning(
        title="[랭킹] 정렬 기준 변경 감지",
        message=(
            f"이전: {previous_live_score_col} → 현재: {current_live_score_col}"
        ),
        details={...}
    )
```

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 수정 | python/ranking_builder.py | 구조화 로그, notifier 알림, sort_basis 저장 |
| 수정 | doc/modules/Lee_trader_score/RUNTIME_SORTING.md | 알림 정책 설명 추가 |

### 수정 금지 파일

- python/notifier.py (기존 notify_warning 그대로 사용)
- python/scoring/final_score.py
- config/production_v1.yaml

---

## 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| THEME_OVERLAY_NOTIFY_ON_ACTIVATE | true | 활성화 시 알림 발송 여부 |
| THEME_OVERLAY_NOTIFY_ON_CHANGE | true | 기준 변경 감지 시 알림 발송 여부 |
| THEME_OVERLAY_SORT_BASIS_JSON | outputs/ranking_sort_basis.json | 기준 저장 경로 |

기존 환경변수 (변경 없음):

| 변수명 | 기본값 | 설명 |
|---|---|---|
| ENABLE_THEME_OVERLAY | 0 | theme overlay 활성화 여부 |
| THEME_OVERLAY_MODE | off | theme overlay 모드 |

---

## 로그 형식 정리

```
# 활성화 시 (WARNING)
THEME_OVERLAY_ACTIVATED | date=2026-05-07 | live_score_col=final_score_v3 |
rank_final=final_score_v3 기준 | trigger=ENABLE_THEME_OVERLAY=1

# 비활성화 시 (INFO)
THEME_OVERLAY_INACTIVE | date=2026-05-07 | live_score_col=final_score |
rank_final=final_score 기준

# 기준 변경 감지 시 (WARNING)
THEME_OVERLAY_CHANGED | date=2026-05-07 |
previous=final_score | current=final_score_v3 | 정렬 기준이 변경되었습니다
```

---

## 검증 케이스

| # | 조건 | 기대 결과 |
|---|---|---|
| 1 | ENABLE_THEME_OVERLAY=0 (기본) | INFO 로그 출력, 알림 없음 |
| 2 | ENABLE_THEME_OVERLAY=1 + 활성화 조건 충족 | WARNING 로그 + notifier WARNING 발송 |
| 3 | 이전 실행 final_score → 현재 final_score_v3 | THEME_OVERLAY_CHANGED WARNING 로그 |
| 4 | 이전과 동일한 기준 | CHANGED 로그 없음 |
| 5 | ranking_sort_basis.json 생성 | live_score_col, live_uses_theme 포함 확인 |
| 6 | THEME_OVERLAY_NOTIFY_ON_ACTIVATE=false | 활성화 시 알림 미발송 확인 |

---

## 주의사항

- theme overlay 활성화/비활성화 조건 로직은 변경하지 말 것
  (`_resolve_theme_overlay_runtime_flags()` 함수 내부 변경 금지)
- 알림 발송 실패가 ranking 생성을 중단시키면 안 됨
  (try/except로 감싸서 best-effort 처리)
- `ranking_sort_basis.json` 읽기 실패 시 이전 비교 로그는 생략하고
  정상 흐름 계속 진행할 것

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
