# Lee_trader_score Runtime Sorting

## 목적
- 이 문서는 실제 운영 시 어떤 점수 컬럼이 정렬과 `rank_final`에 쓰이는지 따로 설명한다.
- 핵심은 `final_score`, `final_score_v3`, `live_rank`, `rank_final`의 관계와 theme overlay runtime flag다.

## 기준 코드
- `python/ranking_builder.py`
  - `_resolve_theme_overlay_runtime_flags()`
  - `build_ranking()`
- 핵심 분기:
  - `live_score_col = "final_score_v3" if live_uses_theme else "final_score"`
  - `live_rank`는 `live_score_col` 기준
  - `rank_final`은 현재 구현상 `live_rank`와 같은 값으로 저장

## runtime flag 해석
- `live_uses_theme`는 다음 조건이 모두 맞을 때만 `true`가 된다.
  - runtime mode가 operational
  - theme overlay가 applied 상태
  - resolved mode가 operational 계열
- 반대로 아래 중 하나면 `false`다.
  - `ENABLE_THEME_OVERLAY != 1`
  - `THEME_OVERLAY_MODE=off`
  - requested mode가 invalid
  - production config에서 theme overlay disabled

## 점수 컬럼별 역할
- `final_score`
  - baseline 운영 점수
  - ret/prob/tech/qual/risk penalty만 직접 반영
  - theme overlay가 꺼져 있으면 실운영 정렬 기준
- `final_score_v2`
  - theme score 직접 혼합 비교용 점수
  - 운영 baseline이 아니라 comparison/reference 성격
- `final_score_v3`
  - `theme_score_effective`를 써서 theme confidence를 반영한 점수
  - runtime에서 theme live 사용이 켜진 경우 실운영 정렬 기준이 될 수 있음
- `shadow_final_score_v3`
  - shadow/debug 비교용 점수
  - 직접 운영 rank 기준으로 쓰이지 않음

## 현재 저장소 설정 기준
- `config/production_v1.yaml`
  - `ranking.theme_overlay.enabled: false`
  - `ranking.theme_overlay.mode: off`
  - `ranking.theme_overlay.validation_enabled: false`
- 현재 `.env`에서 `ENABLE_THEME_OVERLAY`, `THEME_OVERLAY_MODE`는 별도 설정이 확인되지 않았다.
- 따라서 현재 기본 해석은:
  - `live_uses_theme = false`
  - `live_rank`는 `final_score` 기준
  - `rank_final`도 `final_score` 기준

## 운영 해석 요약
- 지금 기본 운영에서 종목 최종 정렬은 `final_score`를 보면 된다.
- `final_score_v2`, `final_score_v3`는 theme 비교/실험 또는 향후 operational theme 전환 대비용으로 보는 것이 맞다.
- 만약 나중에 theme overlay operational을 켜면 그 시점부터는 `live_rank`, `rank_final`이 `final_score_v3` 기준으로 바뀔 수 있다.

## 나중에 확인할 파일
- `data/theme_overlay_mode_resolution.md` 또는 관련 debug output
- `data/ranking_final.csv`
  - `final_score`
  - `final_score_v3`
  - `live_rank`
  - `rank_final`
  - `theme_overlay_mode`
