# 시스템 전체 아키텍처 개요

> **최신화 기준일: 2026-05-27**
> 전체 PRD는 [doc/PRD.md](../PRD.md)를 참조합니다.

---

## 현재 운영 중인 서비스

| 서비스 | 상태 |
|---|---|
| KR AI 자동매매 | **LIVE** (주력) |
| 수동매매 서비스 | 운영 중 (RULE 종료 후 전환) |
| Web 대시보드 | 운영 중 (Node.js, port 3400) |
| RULE 자동매매 | **종료** (2026-05-21) |
| US 주식 서비스 | **미운영** (코드 존재, 스케줄러 비활성) |

---

## 시스템 구성

```
┌──────────────────────────────────────────────────────────────┐
│                  Lee Trader (현재 운영)                       │
│                                                              │
│  [18:10 종가배치]                                             │
│  fetch_market_data → fetch_top_universe (204종목)            │
│  → download_prices/flows → quality/feature/label builder    │
│  → model_train (LightGBM) → model_predict                   │
│  → ranking_builder → ranking_final.csv + DB                 │
│                                                              │
│  [09:30 AI 자동매매]                                          │
│  run_live_auto_trade_cycle → submit_live_orders → KIS API   │
│                                                              │
│  [수동매매]                                                   │
│  manual-trading.html + KIS 계좌(44****02) 동기화             │
│                                                              │
│  ──────────────────────────────────────────────────────     │
│  PostgreSQL · Node.js API (port 3400) · Web 대시보드         │
└──────────────────────────────────────────────────────────────┘
```

---

## 상세 문서

- 전체 PRD: [doc/PRD.md](../PRD.md)
- AI 파이프라인: [01_kr_ranking_pipeline.md](01_kr_ranking_pipeline.md)
- AI 자동매매: [02_ai_trading.md](02_ai_trading.md)
- RULE 자동매매 이력: [03_rule_trading.md](03_rule_trading.md) *(종료 서비스, 이력 보관)*
- 모듈 인덱스: [doc/modules/INDEX.md](../modules/INDEX.md)
