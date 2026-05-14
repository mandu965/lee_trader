# Notification Adapter Design

## Purpose

Phase 8-13 defines a safe notification-adapter structure for the US Paper Trading dashboard flow.

The goal is to make future notification delivery extensible without turning notification generation into an execution path.

Important boundary:

- this phase is design only
- no SMTP delivery is implemented
- no Slack webhook delivery is implemented
- no external API call is implemented
- no broker API is called
- no real account balance or position is read
- all notifications must clearly state that they are based on `Paper Trading`

## Core Principles

- notification failure must not change BUY, SELL, HOLD, or REVIEW_REQUIRED results
- notification failure must not rewrite orchestration results
- notification generation is an operator-review layer only
- notification delivery approval is not trading approval
- `LIVE` notification mode must remain blocked in this phase
- sensitive fields must be redacted before any channel-specific payload is assembled

## Adapter Scope

Input sources:

- dashboard summary payload from Phase 8-12
- dashboard JSON / Markdown artifact paths
- scheduler / health result summary

Output targets:

- local file artifacts
- console/stdout rendering
- dry-run email message body
- dry-run Slack message body

Not in scope:

- actual email sending
- actual Slack sending
- webhook retry logic
- approval UI
- approval workflow automation

## Channel Design

### 1. FILE

Purpose:

- persist notification artifacts for audit and manual review

Input payload:

- standard notification JSON payload
- text summary payload

Output format:

- `.json`
- `.txt`

External call:

- none

Default enabled:

- yes, as the safest baseline channel

Required ENV:

- `US_NOTIFICATION_FILE_ENABLED`
- `US_NOTIFICATION_CHANNELS`

Failure policy:

- file write failure becomes `FILE_WRITE_FAILED`
- default `pipeline_should_fail=false`

Security notes:

- payload must already be redacted
- avoid embedding secrets, tokens, or broker/account identifiers

### 2. CONSOLE

Purpose:

- let the operator see the summary immediately in scheduler or manual runs

Input payload:

- text summary payload

Output format:

- stdout text block

External call:

- none

Default enabled:

- yes, as a local-only review channel

Required ENV:

- `US_NOTIFICATION_CONSOLE_ENABLED`
- `US_NOTIFICATION_CHANNELS`

Failure policy:

- render failure becomes `MESSAGE_RENDER_FAILED`
- default `pipeline_should_fail=false`

Security notes:

- console output must remain summary-only
- avoid dumping large raw JSON with hidden or future-sensitive fields

### 3. EMAIL_DRY_RUN

Purpose:

- prepare email-ready content without sending it

Input payload:

- standard notification JSON payload
- text summary payload
- optional Markdown body derived from dashboard

Output format:

- subject
- plain-text body
- optional Markdown body
- optional attachment candidate paths

External call:

- none

Default enabled:

- no

Required ENV:

- `US_NOTIFICATION_EMAIL_DRY_RUN_ENABLED`
- `US_NOTIFICATION_EMAIL_RECIPIENTS`
- `US_NOTIFICATION_EMAIL_SUBJECT_PREFIX`

Failure policy:

- recipient validation failure becomes `PAYLOAD_INVALID`
- render failure becomes `MESSAGE_RENDER_FAILED`
- default `pipeline_should_fail=false`

Security notes:

- recipients must be reviewed manually
- generated content must not contain tokens, webhook URLs, or real-account data

### 4. SLACK_DRY_RUN

Purpose:

- prepare Slack-ready content without posting it

Input payload:

- standard notification JSON payload
- concise text summary

Output format:

- channel
- username
- text
- optional block-style structure

External call:

- none

Default enabled:

- no

Required ENV:

- `US_NOTIFICATION_SLACK_DRY_RUN_ENABLED`
- `US_NOTIFICATION_SLACK_CHANNEL`
- `US_NOTIFICATION_SLACK_USERNAME`

Failure policy:

- invalid channel or render failure becomes `PAYLOAD_INVALID` or `MESSAGE_RENDER_FAILED`
- default `pipeline_should_fail=false`

Security notes:

- do not store webhook URL in this phase
- generated text must include explicit Paper-only notice

### 5. EMAIL_LIVE

Purpose:

- future actual email delivery channel

Input payload:

- standard notification JSON payload
- rendered email subject/body

Output format:

- SMTP-ready message object

External call:

- yes in a future phase only

Default enabled:

- no

Required ENV:

- `US_NOTIFICATION_EMAIL_LIVE_ENABLED`

Failure policy:

- blocked in this phase as `LIVE_NOTIFICATION_NOT_IMPLEMENTED`

Security notes:

- future implementation must use secret management, not `.env.example`

### 6. SLACK_LIVE

Purpose:

- future actual Slack delivery channel

Input payload:

- standard notification JSON payload
- rendered Slack text/blocks

Output format:

- webhook-ready or API-ready request body

External call:

- yes in a future phase only

Default enabled:

- no

Required ENV:

- `US_NOTIFICATION_SLACK_LIVE_ENABLED`

Failure policy:

- blocked in this phase as `LIVE_NOTIFICATION_NOT_IMPLEMENTED`

Security notes:

- future implementation must keep webhook URL outside plain repo config

## Notification Mode Policy

### DISABLED

- adapter does not generate channel output
- scheduler records the adapter as skipped
- `pipeline_should_fail=false`

### DRY_RUN

- adapter renders channel-specific artifacts only
- no external send is allowed
- output may include file artifacts and console text

### MANUAL_APPROVAL

- adapter renders channel-specific artifacts
- approval record is created conceptually with `approval_status=PENDING`
- no delivery occurs until a later, separate manual step
- approval is for notification delivery only

### LIVE

- reserved for a future phase only
- if configured in this phase, return `LIVE_NOTIFICATION_NOT_IMPLEMENTED`
- no external send occurs

## Standard Notification Payload

Recommended standard payload:

```json
{
  "message_type": "US_PAPER_TRADING_DASHBOARD_SUMMARY",
  "trade_date": "2026-05-14",
  "generated_at": "2026-05-14T20:30:00+09:00",
  "mode": "SHADOW",
  "status": "WARNING",
  "severity": "WARNING",
  "paper_trading_only": true,
  "live_orders_executed": false,
  "buy": {
    "candidates": 5,
    "final_allowed": 1,
    "conflict_blocked": 2
  },
  "sell": {
    "positions": 4,
    "sell_signals": 1,
    "review_required": 1
  },
  "risk": {
    "data_missing_rate": 8.5,
    "fail_safe_triggered": true,
    "top_warning_reason": "DATA_MISSING"
  },
  "health": {
    "scheduler_status": "PASS",
    "dashboard_status": "PASS"
  },
  "readiness": {
    "live_ready": false,
    "readiness_score": 62,
    "manual_approval_required": true
  },
  "links": {
    "dashboard_json": "reports/lee_trader_us/dashboard/latest_dashboard.json",
    "dashboard_markdown": "reports/lee_trader_us/dashboard/latest_dashboard.md"
  },
  "notice": "Paper Trading only. No live orders were executed."
}
```

Required fields:

- `message_type`
- `trade_date`
- `generated_at`
- `mode`
- `status`
- `severity`
- `paper_trading_only`
- `live_orders_executed`
- `notice`

## Severity Policy

### INFO

Use when:

- orchestration succeeded
- no major warning exists
- dashboard and health artifacts exist

### WARNING

Use when:

- `REVIEW_REQUIRED` exists
- data-missing rate exceeds warning threshold
- conflict blocks are elevated
- dashboard Markdown is missing but JSON exists

### ERROR

Use when:

- orchestration failed
- dashboard JSON is missing
- health check fails
- invalid decision log is present

### CRITICAL

Use when:

- `live_trading_enabled=true` appears in a payload or dashboard
- `live_orders_executed=true` appears
- LIVE-mode block fails
- actual order API activity is detected
- portfolio state is severely inconsistent

Policy notes:

- `CRITICAL` never triggers automatic LIVE transition
- `CRITICAL` means manual operator review is required

## Channel-Specific Message Format

### Console / File Text

```text
[US Paper Trading Dashboard] WARNING

date: 2026-05-14
mode: SHADOW
status: WARNING

BUY:
- candidates: 5
- final allowed: 1
- conflict blocked: 2

SELL:
- positions: 4
- sell signals: 1
- review required: 1

Risk/Data:
- data missing rate: 8.5%
- fail-safe: YES
- top warning: DATA_MISSING

Health:
- scheduler: PASS
- dashboard: PASS

Readiness:
- live_ready: false
- readiness_score: 62
- manual approval required: true

Notice:
Paper Trading only. No live orders were executed.
```

### Email Dry Run

Fields:

- `subject`
- `plain_text_body`
- optional `markdown_body`
- `attachment_candidate_paths`
- `recipient_validation`

Example subject:

```text
[US Paper Trading][WARNING] 2026-05-14 Dashboard Summary
```

### Slack Dry Run

Fields:

- `channel`
- `username`
- `text`
- optional `blocks`
- `severity_emoji`
- `dashboard_path`

Example text:

```text
:warning: US Paper Trading Dashboard - WARNING
Date: 2026-05-14
BUY allowed: 1 / SELL signals: 1 / Review required: 1
Paper Trading only. No live orders were executed.
```

## Manual Approval Policy

Approval states:

- `PENDING`
- `APPROVED`
- `REJECTED`
- `EXPIRED`
- `SENT`
- `FAILED`

Rules:

- `MANUAL_APPROVAL` mode stores the generated message in `PENDING` state
- operator approval is required before any future live delivery channel can send
- approval record should include `approver`, `approved_at`, and `comment`
- expired approvals move to `EXPIRED`
- delivery attempt result becomes `SENT` or `FAILED`

Important clarification:

- notification approval is not trading approval
- notification approval must not be confused with LIVE trading approval

## Security And Sensitive-Field Policy

Must not appear in payloads or rendered messages:

- API keys
- access tokens
- webhook URLs
- broker account numbers
- real account balances
- real account positions
- personal identifying information

Allowed content:

- Paper Trading symbols
- Paper Trading order counts
- Paper Trading PnL and returns
- dashboard file paths
- scheduler and health status

Redaction policy:

- enable `US_NOTIFICATION_REDACT_SENSITIVE_FIELDS=1` by default
- redact unknown fields conservatively if a future payload loader sees suspicious keys

## Failure Policy

| Failure code | Severity | Retry needed | pipeline_should_fail | Operator action |
| --- | --- | --- | --- | --- |
| `PAYLOAD_MISSING` | `ERROR` | no | `false` | inspect dashboard/notification payload generation |
| `PAYLOAD_INVALID` | `ERROR` | no | `false` | inspect formatter and required fields |
| `CHANNEL_DISABLED` | `INFO` | no | `false` | none |
| `MANUAL_APPROVAL_REQUIRED` | `INFO` | no | `false` | review and approve later if needed |
| `LIVE_NOTIFICATION_NOT_IMPLEMENTED` | `ERROR` | no | `false` | reset mode away from `LIVE` |
| `DELIVERY_DRY_RUN_ONLY` | `INFO` | no | `false` | none |
| `MESSAGE_RENDER_FAILED` | `ERROR` | maybe | `false` | inspect renderer and payload completeness |
| `FILE_WRITE_FAILED` | `ERROR` | maybe | `false` | inspect path and filesystem permissions |
| `UNKNOWN_ERROR` | `ERROR` | maybe | `false` | inspect adapter logs |

Critical override:

- if payload shows `live_orders_executed=true` or `live_trading_enabled=true`, severity must be escalated to `CRITICAL`

## Future Module Structure Proposal

Actual repository-aligned proposal:

```text
python/us/notification/
  __init__.py
  config.py
  notification_payload_loader.py
  severity_policy.py
  channel_router.py
  console_adapter.py
  file_adapter.py
  email_dry_run_adapter.py
  slack_dry_run_adapter.py
  manual_approval.py
  notification_logger.py
  run_us_notification_adapter.py
```

Prompt-aligned alternative path:

```text
src/modules/lee_trader_us/notification/
```

Suggested responsibilities:

- `config.py`: load adapter ENV, channel enablement, safety mode
- `notification_payload_loader.py`: load standardized payload from dashboard scheduler artifacts
- `severity_policy.py`: assign `INFO/WARNING/ERROR/CRITICAL`
- `channel_router.py`: decide which adapters run for the current mode and enabled channels
- `console_adapter.py`: render text summary to stdout
- `file_adapter.py`: persist text/json delivery artifacts
- `email_dry_run_adapter.py`: create subject/body/attachments without SMTP send
- `slack_dry_run_adapter.py`: create Slack-ready text/blocks without webhook send
- `manual_approval.py`: manage approval-status state transitions and retention rules
- `notification_logger.py`: write event, delivery, approval logs
- `run_us_notification_adapter.py`: CLI wrapper for manual dry-run and review

## Scheduler Integration Design

Recommended order:

1. trade orchestration execution
2. integrated report generation
3. dashboard report generation
4. dashboard health check
5. notification payload generation
6. notification adapter execution
7. scheduler final result recording

Rules:

- notification adapter runs last
- notification failure must not alter orchestration output
- notification payload should include dashboard artifact paths
- actual delivery remains disabled by default

## ENV Design

Recommended ENV:

```env
# US Dashboard Notification Adapter
US_NOTIFICATION_ADAPTER_ENABLED=0
US_NOTIFICATION_ADAPTER_MODE=DRY_RUN

US_NOTIFICATION_CHANNELS=FILE,CONSOLE
US_NOTIFICATION_REQUIRE_MANUAL_APPROVAL=1
US_NOTIFICATION_FAIL_PIPELINE_ON_ERROR=0

# File / Console
US_NOTIFICATION_FILE_ENABLED=1
US_NOTIFICATION_CONSOLE_ENABLED=1

# Email Dry Run
US_NOTIFICATION_EMAIL_DRY_RUN_ENABLED=0
US_NOTIFICATION_EMAIL_RECIPIENTS=
US_NOTIFICATION_EMAIL_SUBJECT_PREFIX=[US Paper Trading]

# Slack Dry Run
US_NOTIFICATION_SLACK_DRY_RUN_ENABLED=0
US_NOTIFICATION_SLACK_CHANNEL=
US_NOTIFICATION_SLACK_USERNAME=LeeTraderBot

# Live channels are disabled by default
US_NOTIFICATION_EMAIL_LIVE_ENABLED=0
US_NOTIFICATION_SLACK_LIVE_ENABLED=0

# Safety
US_NOTIFICATION_INCLUDE_PAPER_TRADING_NOTICE=1
US_NOTIFICATION_INCLUDE_LIVE_DISABLED_NOTICE=1
US_NOTIFICATION_MAX_SYMBOLS=10
US_NOTIFICATION_REDACT_SENSITIVE_FIELDS=1
```

## Table Design

### `trade.us_notification_event_log`

Purpose:

- store one notification-generation event per trade date / payload type

Suggested fields:

- `notification_event_id`
- `trade_date`
- `message_type`
- `severity`
- `mode`
- `paper_trading_only`
- `approval_required`
- `approval_status`
- `payload_json`
- `message_text`
- `error_message`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(trade_date, message_type, mode)`

Stored when:

- the standardized payload is finalized before channel routing

Retention:

- at least 1 year

Related modules:

- `notification_payload_loader.py`
- `notification_logger.py`

### `trade.us_notification_delivery_log`

Purpose:

- store one delivery attempt or dry-run result per channel

Suggested fields:

- `delivery_log_id`
- `notification_event_id`
- `trade_date`
- `channel`
- `delivery_mode`
- `delivery_status`
- `severity`
- `message_text`
- `payload_json`
- `error_message`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(notification_event_id, channel, delivery_mode)`

Stored when:

- each channel adapter finishes dry-run rendering or future send attempt

Retention:

- at least 1 year

Related modules:

- `channel_router.py`
- `file_adapter.py`
- `console_adapter.py`
- `email_dry_run_adapter.py`
- `slack_dry_run_adapter.py`

### `trade.us_notification_approval_log`

Purpose:

- track manual approval lifecycle for notification delivery

Suggested fields:

- `approval_log_id`
- `notification_event_id`
- `trade_date`
- `approval_required`
- `approval_status`
- `approver`
- `approved_at`
- `comment`
- `expires_at`
- `created_at`
- `updated_at`

Suggested uniqueness:

- `(notification_event_id, approval_status, created_at)`

Stored when:

- a manual-approval artifact is created or its state changes

Retention:

- at least 2 years for auditability

Related modules:

- `manual_approval.py`
- `notification_logger.py`

## Operations Notes

- operators should treat notification output as a review artifact, not a trading action
- file and console channels are the only safe-default channels in this phase
- email/slack dry-run output must still carry the Paper-only notice
- any future live-delivery phase must be reviewed separately after dry-run stability is proven

## Known Limitations

- no actual delivery exists in this phase
- no approval UI exists in this phase
- no DB persistence is required in this phase
- severity assignment is policy-driven and may need later tuning
- future channel adapters must re-validate redaction rules before any external send path is opened

## Phase 8-14 Implementation Notes

Implemented modules in this repository:

- `python/us/notification/config.py`
- `python/us/notification/notification_payload_loader.py`
- `python/us/notification/severity_policy.py`
- `python/us/notification/channel_router.py`
- `python/us/notification/console_adapter.py`
- `python/us/notification/file_adapter.py`
- `python/us/notification/email_dry_run_adapter.py`
- `python/us/notification/slack_dry_run_adapter.py`
- `python/us/notification/manual_approval.py`
- `python/us/notification/notification_logger.py`
- `python/us/notification/run_us_notification_adapter.py`
- `python/us/run_us_notification_adapter.py`
- `scripts/run_us_notification_adapter.py`

Supported channels:

- `FILE`
- `CONSOLE`
- `EMAIL_DRY_RUN`
- `SLACK_DRY_RUN`

Blocked channels:

- `EMAIL_LIVE`
- `SLACK_LIVE`

Execution examples:

```powershell
python -m python.us.notification.run_us_notification_adapter --force
python -m python.us.notification.run_us_notification_adapter --trade-date 2026-05-15 --force
python scripts/run_us_notification_adapter.py --channels FILE,CONSOLE --force
```

Primary output files:

- `reports/lee_trader_us/notification/YYYY-MM-DD_notification_adapter.txt`
- `reports/lee_trader_us/notification/YYYY-MM-DD_notification_adapter.json`
- `reports/lee_trader_us/notification/latest_notification_adapter.txt`
- `reports/lee_trader_us/notification/latest_notification_adapter.json`

Manual approval storage:

- `reports/lee_trader_us/notification/approvals/YYYY-MM-DD_approval_pending.json`
- `reports/lee_trader_us/notification/approvals/latest_approval_pending.json`

Implementation boundary:

- no external delivery is performed
- dry-run rendering and file persistence only
- notification failure does not change trading decisions

## Phase 8-15 Quality Gate Relationship

In the future quality-gate layer, notification artifacts should be treated as one of the review inputs, not as the source of trading truth.

The notification-related gate should verify:

- `paper_trading_only=true`
- `live_orders_executed=false`
- sensitive-field redaction
- live-channel blocking integrity
- manual-approval pending artifact presence when required

Even if the notification-safety gate passes:

- LIVE notification delivery must remain unimplemented
- LIVE trading approval must remain separate from notification approval
