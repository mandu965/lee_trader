# 9차 과제: 섹터 집중 hard cap 명시화

> 상태: ⬜ 대기
> 작성일: 2026-05-07
> 의존성: 없음 (독립 진행 가능)
> 다음 과제: 없음 (독립 완결)

---

## 목적

한국 시장 특성상 반도체·이차전지·AI 등 특정 테마로
포트폴리오 쏠림이 빠르게 발생할 수 있다.

현재 섹터 비중 관련 구조:
- Rule 자동매매: `RULE_MAX_SECTOR_WEIGHT` 환경변수 존재 (기본 0.35)
- AI 자동매매: `--sector-cap` 인수 존재 (기본 production_config 값 또는 0.35)
- 그러나 **매일 실제 섹터 노출 현황을 기록하는 파일이 없다**
- 섹터 비중 한계 근접 시 **경고 알림이 없다**

이 과제의 목표:
1. 섹터 비중 상한 환경변수를 명시적으로 통일한다
2. 매일 섹터별 현재 노출 현황을 파일로 기록한다
3. 상한 근접 시 경고 알림을 추가한다

---

## 현재 구조 상세 파악

### Rule 자동매매 (rule_portfolio_manager.py)

```python
max_sector_weight = cfg_float("RULE_MAX_SECTOR_WEIGHT", 0.35)

# 포트폴리오 편입 시 섹터 비중 체크
sector_exposure: dict[str, float] = {}
for sector, group in positions.groupby("sector"):
    sector_exposure[sector] = float(group["weight"].sum())

sector_after = sector_exposure.get(sector, 0.0) + new_entry_weight
sector_limit_pass = sector_after <= max_sector_weight or held
```

`sector_limit_pass=False`이면 `rule_account_guard.py`에서
`"sector_limit_failed"` 사유로 BUY 차단된다.

### AI 자동매매 (submit_live_orders.py / portfolio_constructor.py)

```python
# submit_live_orders.py
parser.add_argument("--sector-cap", type=float,
    default=float(get_production_config_value(
        ["execution_policy", "sector_cap"],
        get_production_config_value(["portfolio", "sector_cap"], 0.35)
    ))
)

# portfolio_constructor.py
sector_cap_slots = max(1, int(math.ceil(slot_count * args.sector_cap)))
if sector_counts.get(sector, 0) >= sector_cap_slots:
    # 제외 처리
```

AI 쪽은 `--sector-cap` 인수 또는 production_config YAML 값을 사용.
환경변수로 직접 제어하는 경로가 없다.

### 공통 문제점

- Rule/AI 섹터 상한이 별도 설정 경로를 가져 불일치 가능성 있음
- 매일 실제 섹터별 비중이 어떻게 구성됐는지 기록 파일 없음
- 섹터 비중 80% 이상 도달 시 경고 없음
- 운영자가 "지금 IT 섹터가 몇 %인지" 바로 확인할 파일 없음

---

## 변경 방향

### Step 1. 환경변수 통일

```
# Rule 자동매매 (기존 유지)
RULE_MAX_SECTOR_WEIGHT=0.35

# AI 자동매매 (신규)
AI_MAX_SECTOR_WEIGHT=0.35          # submit_live_orders의 --sector-cap 기본값 덮어쓰기
AI_MAX_SECTOR_COUNT=5              # 동시 보유 최대 섹터 수

# 공통 경고 기준
SECTOR_WARN_THRESHOLD_RATIO=0.80   # 상한의 80% 도달 시 경고
                                   # 예: 상한 35%이면 28% 이상 시 경고
```

`submit_live_orders.py`에서 `AI_MAX_SECTOR_WEIGHT` 환경변수를
`--sector-cap` 기본값으로 읽도록 수정한다.

### Step 2. 섹터 노출 현황 파일 생성

매일 포트폴리오 구성 후 `outputs/sector_exposure_summary.json`을 저장한다.

```json
{
  "generated_at": "2026-05-07T09:05:00",
  "as_of_date": "2026-05-07",
  "run_mode": "live",
  "engine": "rule",
  "sector_cap": 0.35,
  "warn_threshold": 0.28,
  "total_positions": 8,
  "sectors": [
    {
      "sector": "반도체",
      "position_count": 3,
      "total_weight": 0.32,
      "weight_pct": 32.0,
      "cap_usage_pct": 91.4,
      "warn_triggered": true,
      "codes": ["005930", "000660", "058470"]
    },
    {
      "sector": "이차전지",
      "position_count": 2,
      "total_weight": 0.18,
      "weight_pct": 18.0,
      "cap_usage_pct": 51.4,
      "warn_triggered": false,
      "codes": ["006400", "247540"]
    }
  ],
  "warn_triggered_sectors": ["반도체"],
  "cap_exceeded_sectors": []
}
```

### Step 3. 경고 알림 추가

`cap_usage_pct >= SECTOR_WARN_THRESHOLD_RATIO * 100`인 섹터 발생 시:

```python
notify_warning(
    title="[섹터 집중] 비중 경고",
    message=f"반도체 섹터 비중 32.0% — 상한(35%)의 91% 도달",
    details={
        "sector": "반도체",
        "current_weight": 0.32,
        "cap": 0.35,
        "cap_usage_pct": 91.4,
        "codes": ["005930", "000660", "058470"],
        "as_of_date": "2026-05-07"
    }
)
```

`cap_exceeded_sectors`가 발생하면 CRITICAL 알림.

---

## 수정 대상 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 수정 | python/rule_portfolio_manager.py | sector_exposure_summary.json 저장, 경고 알림 추가 |
| 수정 | python/submit_live_orders.py | AI_MAX_SECTOR_WEIGHT 환경변수 연결 |
| 수정 | .env.example | 신규 환경변수 추가 |

### 수정 금지 파일

- python/portfolio_constructor.py (내부 로직 변경 금지)
- python/rule_account_guard.py (sector_limit_failed guard 로직 변경 금지)
- python/rule_signal_builder.py

---

## 환경변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| RULE_MAX_SECTOR_WEIGHT | 0.35 | Rule 단일 섹터 최대 비중 (기존 유지) |
| AI_MAX_SECTOR_WEIGHT | 0.35 | AI 자동매매 단일 섹터 최대 비중 |
| AI_MAX_SECTOR_COUNT | 5 | AI 자동매매 동시 보유 최대 섹터 수 |
| SECTOR_WARN_THRESHOLD_RATIO | 0.80 | 상한 대비 경고 발동 비율 |
| SECTOR_EXPOSURE_NOTIFY_ENABLED | true | 섹터 경고 알림 활성화 |

---

## 검증 케이스

| # | 조건 | 기대 결과 |
|---|---|---|
| 1 | 섹터 비중 20% (상한 35%) | warn_triggered=false |
| 2 | 섹터 비중 29% (상한 35%, 임계 28%) | warn_triggered=true, WARNING 알림 |
| 3 | 섹터 비중 36% (상한 35%) | cap_exceeded=true, CRITICAL 알림 |
| 4 | 포트폴리오 구성 후 | sector_exposure_summary.json 생성 확인 |
| 5 | AI_MAX_SECTOR_WEIGHT=0.30 | submit_live_orders --sector-cap에 0.30 반영 |
| 6 | 섹터 없는 종목 | sector=(none)으로 그룹화 |

---

## 주의사항

- 기존 `sector_limit_pass` / `sector_limit_failed` guard 로직은 변경하지 말 것
- `sector_exposure_summary.json`은 포트폴리오 계획 완료 후 best-effort로 저장하며
  이 파일 저장 실패가 포트폴리오 계획 자체를 막으면 안 됨
- 섹터 정보가 없는 종목(sector=None)은 "(none)" 또는 "기타"로 그룹화할 것
- AI와 Rule의 섹터 cap 값이 다를 수 있으며 각 엔진 기준을 별도로 기록할 것

---

## 완료 후 기록

완료일:
변경 파일:
검증 결과:
주요 결정 사항:
다음 과제 연결 포인트:
