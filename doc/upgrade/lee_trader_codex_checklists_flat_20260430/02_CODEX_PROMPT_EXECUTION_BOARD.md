# Lee Trader Codex Prompt Execution Board

Purpose: track which prompt is being worked, locally verified, applied to server, and observed after release.

---

## Status Codes

| Status | Meaning |
|---|---|
| TODO | not started |
| IN_PROGRESS | currently being edited |
| LOCAL_TESTED | local or preview verification completed |
| SERVER_APPLIED | applied to server |
| MONITORING | observing after server apply |
| DONE | finished |
| BLOCKED | stopped by an issue |
| ROLLED_BACK | reverted |

---

## Execution Board

| Prompt | Priority | Work Item | Status | Local Test | Server Applied | Observation Done | Notes |
|---:|---|---|---|---|---|---|---|
| 1 | P0 | add `common_live_risk_guard.py` | LOCAL_TESTED | [x] | [ ] | [ ] | self-test passed + integration preview verified |
| 2 | P0 | add AI entry price gate to `submit_live_orders.py` | LOCAL_TESTED | [x] | [ ] | [ ] | entry gate + BUY common guard preview verified |
| 3 | P0 | connect RULE order guard to common guard | LOCAL_TESTED | [x] | [ ] | [ ] | BUY-only common layer + preview verified |
| 4 | P1 | expand live trade ledger fields | TODO | [ ] | [ ] | [ ] | DB migration caution |
| 5 | P2 | introduce AI `live_confidence_grade` | TODO | [ ] | [ ] | [ ] | affects live sizing |
| 6 | P3 | add RULE max holding / stop loss / trailing stop | TODO | [ ] | [ ] | [ ] | affects exit logic |
| 7 | P4 | add operations dashboard payload | LOCAL_TESTED | [x] | [ ] | [ ] | payload + API + UI fallback verified |
| 8 | P5 | master risk manager preview integration | LOCAL_TESTED | [x] | [ ] | [ ] | preview-only master approval outputs verified |

---

## Recommended Sequence

```text
Day 1: Prompt 1
Day 2: Prompt 1 verification + Prompt 2
Day 3: Prompt 2 verification + Prompt 3
Day 4: Prompt 3 verification + monitoring
Day 5: Prompt 4 DB / ledger work
Day 6: Prompt 4 verification
After that: Prompt 5, 6, 7, 8 in order
```

---

## Stop Criteria

### Prompt 1

```text
[ ] common guard crashes on missing files or exceptions
[ ] SELL / EXIT logic is affected
[ ] block reasons are unclear
```

### Prompt 2

```text
[ ] live price lookup fails but orders still submit
[ ] AUTO_TRADE_EXECUTE safety is weakened
[ ] preview JSON breaks existing UI
```

### Prompt 3

```text
[ ] RULE SELL / EXIT is incorrectly blocked by BUY guard
[ ] existing RULE block reasons disappear
[ ] preview and submit results diverge
```

### Prompt 4

```text
[ ] migration damages existing DB
[ ] existing live trade review breaks
[ ] existing column meanings change
```

### Prompt 5

```text
[ ] existing confidence outputs disappear
[ ] weak sample buckets get A / B grades
[ ] D grade still produces BUY
```

### Prompt 6

```text
[ ] new BUY logic is unintentionally changed
[ ] missing entry data still triggers forced exits
[ ] hold action reasons disappear
```

### Prompt 7

```text
[ ] existing UI breaks
[ ] scheduler failure is shown as success
[ ] kill switch state is shown incorrectly
```

### Prompt 8

```text
[ ] direct order submission gets wired by mistake
[ ] existing AI / RULE flow breaks
[ ] duplicate BUY blocking is unclear
```
