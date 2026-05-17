"""
compute_us_macro_overlay_shadow.py

Phase 3 - US Macro Overlay: Shadow Mode.

Reads US macro features and Korean buy candidates, applies overlay rules,
and writes hypothetical adjusted results to signal.kr_macro_overlay_log.

⚠ SHADOW MODE ONLY (Phase 3):
   - Reads from: signal.us_macro_feature_daily (US macro features)
   - Reads from: public.daily_ranking (AI top candidates)
   - Reads from: data/rule_signals.csv (RULE buy candidates)
   - Writes to:  signal.kr_macro_overlay_log ONLY
   - Does NOT touch: final_score, rule_score, daily_ranking, any order tables
   - Does NOT create or trigger any orders

Overlay rules (ref: RRD_US_Macro_Overlay.md §24):
  1. Risk-Off block    : QQQ drop + VIX spike → buy_blocked + MAX_NEGATIVE_ADJUST
  2. Semiconductor +   : SMH strong + QQQ >= 0 + stock in semi sector → +3
  3. Semiconductor -   : SMH weak + stock in semi sector → -5
  4. Market-wide -     : SPY drop → -5
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

LOGGER = logging.getLogger("compute_us_macro_overlay_shadow")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_SIGNALS_CSV = ROOT / "data" / "rule_signals.csv"

# ── Semiconductor sector keyword matching ─────────────────────────────────────

_SEMI_KEYWORDS = ["반도체", "semiconductor"]


def _is_semiconductor(sector: str | None) -> bool:
    if not sector:
        return False
    s = str(sector).lower()
    return any(k in s for k in _SEMI_KEYWORDS)


# ── Config from env ────────────────────────────────────────────────────────────

class _Cfg:
    def __init__(self) -> None:
        self.risk_off_qqq_ret     = float(os.environ.get("US_MACRO_RISK_OFF_QQQ_RET", "-0.015"))
        self.risk_off_spy_ret     = float(os.environ.get("US_MACRO_RISK_OFF_SPY_RET", "-0.012"))
        self.vix_spike_ret        = float(os.environ.get("US_MACRO_VIX_SPIKE_RET", "0.10"))
        self.semi_positive_ret    = float(os.environ.get("US_MACRO_SEMI_POSITIVE_RET", "0.01"))
        self.semi_negative_ret    = float(os.environ.get("US_MACRO_SEMI_NEGATIVE_RET", "-0.01"))
        self.max_positive_adjust  = float(os.environ.get("US_MACRO_MAX_POSITIVE_ADJUST", "5.0"))
        self.max_negative_adjust  = float(os.environ.get("US_MACRO_MAX_NEGATIVE_ADJUST", "-10.0"))
        self.ai_top_n             = int(os.environ.get("US_MACRO_AI_TOP_N", "30"))


# ── DB read helpers ────────────────────────────────────────────────────────────

_MACRO_SQL = text("""
SELECT
    us_trade_date, kr_apply_date,
    spy_ret_1d, qqq_ret_1d, semiconductor_ret_1d,
    vix_ret_1d, dxy_ret_1d, tnx_ret_1d,
    sector_breadth, top_sector, bottom_sector,
    risk_on_flag, risk_off_flag, vix_spike_flag,
    semiconductor_strength_flag,
    macro_status, macro_summary
FROM signal.us_macro_feature_daily
WHERE kr_apply_date = :kr_apply_date
ORDER BY us_trade_date DESC
LIMIT 1
""")

_LATEST_MACRO_APPLY_DATE_SQL = text("""
SELECT kr_apply_date
FROM signal.us_macro_feature_daily
ORDER BY kr_apply_date DESC, us_trade_date DESC
LIMIT 1
""")

_RANKING_SQL = text("""
SELECT date, code, name, sector, final_score, rank_final
FROM public.daily_ranking
WHERE date = (
    SELECT MAX(date) FROM public.daily_ranking
)
ORDER BY rank_final
LIMIT :top_n
""")

_INSERT_LOG_SQL = text("""
INSERT INTO signal.kr_macro_overlay_log (
    run_date, us_trade_date, kr_apply_date,
    macro_status, engine_type,
    code, name, sector,
    original_score, macro_adjustment, adjusted_score,
    buy_blocked_flag, is_buy_candidate,
    overlay_reason, shadow_mode
) VALUES (
    :run_date, :us_trade_date, :kr_apply_date,
    :macro_status, :engine_type,
    :code, :name, :sector,
    :original_score, :macro_adjustment, :adjusted_score,
    :buy_blocked_flag, :is_buy_candidate,
    :overlay_reason, TRUE
)
ON CONFLICT (run_date, engine_type, code) DO UPDATE SET
    us_trade_date    = EXCLUDED.us_trade_date,
    kr_apply_date    = EXCLUDED.kr_apply_date,
    macro_status     = EXCLUDED.macro_status,
    original_score   = EXCLUDED.original_score,
    macro_adjustment = EXCLUDED.macro_adjustment,
    adjusted_score   = EXCLUDED.adjusted_score,
    buy_blocked_flag = EXCLUDED.buy_blocked_flag,
    is_buy_candidate = EXCLUDED.is_buy_candidate,
    overlay_reason   = EXCLUDED.overlay_reason,
    shadow_mode      = TRUE
""")


def _fetch_macro_row(engine: Any, kr_apply_date: date) -> dict | None:
    with engine.connect() as conn:
        result = conn.execute(_MACRO_SQL, {"kr_apply_date": kr_apply_date})
        row = result.fetchone()
    if row is None:
        return None
    return dict(row._mapping)


def fetch_latest_macro_apply_date(engine: Any) -> date | None:
    with engine.connect() as conn:
        row = conn.execute(_LATEST_MACRO_APPLY_DATE_SQL).fetchone()
    if row is None:
        return None
    value = row[0]
    return value if isinstance(value, date) else None


def _fetch_ai_candidates(engine: Any, top_n: int) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            result = conn.execute(_RANKING_SQL, {"top_n": top_n})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as exc:
        LOGGER.warning("AI ranking read failed: %s", exc)
        return pd.DataFrame()


def _fetch_rule_candidates(csv_path: Path = DEFAULT_RULE_SIGNALS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        LOGGER.warning("rule_signals.csv not found at %s - skipping RULE candidates.", csv_path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, dtype={"code": str}, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest = df["date"].max()
            df = df[df["date"] == latest]
        if "entry_signal" in df.columns:
            df = df[df["entry_signal"].fillna(False).astype(bool)]
        keep = [c for c in ["code", "name", "sector", "rule_score_v2", "rule_score",
                             "entry_signal", "strong_entry_signal", "date"] if c in df.columns]
        return df[keep].reset_index(drop=True)
    except Exception as exc:
        LOGGER.warning("rule_signals.csv read failed: %s", exc)
        return pd.DataFrame()


# ── Overlay rule engine ────────────────────────────────────────────────────────

def _apply_overlay_rules(
    code: str,
    name: str | None,
    sector: str | None,
    original_score: float | None,
    is_buy_candidate: bool,
    macro: dict,
    cfg: _Cfg,
) -> dict:
    """Apply all overlay rules and return a result dict for one stock."""

    qqq_1d        = macro.get("qqq_ret_1d")
    spy_1d        = macro.get("spy_ret_1d")
    vix_1d        = macro.get("vix_ret_1d")
    semi_1d       = macro.get("semiconductor_ret_1d")
    macro_status  = macro.get("macro_status", "UNKNOWN")

    reasons: list[str] = []
    total_adj: float = 0.0
    buy_blocked = False
    is_semi = _is_semiconductor(sector)

    # ── Rule 1: Risk-Off block (QQQ drop + VIX spike) ──────────────────────────
    qqq_drop = qqq_1d is not None and qqq_1d <= cfg.risk_off_qqq_ret
    vix_spike = vix_1d is not None and vix_1d >= cfg.vix_spike_ret

    if qqq_drop and vix_spike:
        buy_blocked = True
        total_adj = cfg.max_negative_adjust  # e.g. -10
        reasons.append(
            f"[RISK_OFF_BLOCK] QQQ={qqq_1d:+.2%} ≤ {cfg.risk_off_qqq_ret:.2%},"
            f" VIX={vix_1d:+.2%} ≥ {cfg.vix_spike_ret:.2%}"
        )

    else:
        # ── Rule 2: Semiconductor bonus ─────────────────────────────────────────
        if is_semi and semi_1d is not None and semi_1d >= cfg.semi_positive_ret \
                and qqq_1d is not None and qqq_1d >= 0:
            total_adj += 3.0
            reasons.append(
                f"[SEMI_POSITIVE] 반도체섹터 + SMH={semi_1d:+.2%} ≥ {cfg.semi_positive_ret:.2%},"
                f" QQQ={qqq_1d:+.2%} ≥ 0"
            )

        # ── Rule 3: Semiconductor penalty ───────────────────────────────────────
        if is_semi and semi_1d is not None and semi_1d <= cfg.semi_negative_ret:
            total_adj -= 5.0
            reasons.append(
                f"[SEMI_NEGATIVE] 반도체섹터 + SMH={semi_1d:+.2%} ≤ {cfg.semi_negative_ret:.2%}"
            )

        # ── Rule 4: Market-wide penalty ──────────────────────────────────────────
        if spy_1d is not None and spy_1d <= cfg.risk_off_spy_ret:
            total_adj -= 5.0
            reasons.append(
                f"[MARKET_WEAK] SPY={spy_1d:+.2%} ≤ {cfg.risk_off_spy_ret:.2%}"
            )

        # Clamp total adjustment to configured bounds
        total_adj = max(cfg.max_negative_adjust, min(cfg.max_positive_adjust, total_adj))

    if not reasons:
        reasons.append(f"[NEUTRAL] macro_status={macro_status}")

    adj_score: float | None = None
    if original_score is not None:
        adj_score = round(max(0.0, min(100.0, float(original_score) + total_adj)), 2)

    return {
        "code":             code,
        "name":             name,
        "sector":           sector,
        "original_score":   round(float(original_score), 2) if original_score is not None else None,
        "macro_adjustment": round(total_adj, 2),
        "adjusted_score":   adj_score,
        "buy_blocked_flag": buy_blocked,
        "is_buy_candidate": is_buy_candidate,
        "overlay_reason":   " | ".join(reasons),
    }


# ── Orchestration ──────────────────────────────────────────────────────────────

def run(
    kr_apply_date: date,
    engine: Any,
    rule_signals_csv: Path = DEFAULT_RULE_SIGNALS_CSV,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compute shadow overlay for all buy candidates and write to log table.

    Returns a summary dict. Raises nothing — all errors are logged.
    """
    cfg = _Cfg()

    # 1. Fetch macro features
    macro = _fetch_macro_row(engine, kr_apply_date)
    if macro is None:
        LOGGER.warning(
            "[Phase 3] No macro data for kr_apply_date=%s. "
            "Run Phase 1~2 first: python run_us_macro_overlay_scheduler.py",
            kr_apply_date,
        )
        return {
            "kr_apply_date": kr_apply_date,
            "macro_status":  "NO_DATA",
            "rule_rows":     0,
            "ai_rows":       0,
            "blocked_count": 0,
        }

    macro_status = macro.get("macro_status", "UNKNOWN")
    us_trade_date = macro.get("us_trade_date")

    LOGGER.info(
        "[Phase 3] Macro status for kr_apply_date=%s: %s (us_trade=%s)",
        kr_apply_date, macro_status, us_trade_date,
    )
    LOGGER.info("[Phase 3] Summary: %s", macro.get("macro_summary", ""))

    run_date = date.today()
    base_row = {
        "run_date":      run_date,
        "us_trade_date": us_trade_date,
        "kr_apply_date": kr_apply_date,
        "macro_status":  macro_status,
    }

    # 2. Fetch candidates
    ai_df = _fetch_ai_candidates(engine, cfg.ai_top_n)
    rule_df = _fetch_rule_candidates(rule_signals_csv)

    LOGGER.info("[Phase 3] Candidates - AI: %d, RULE: %d", len(ai_df), len(rule_df))

    all_rows: list[dict] = []

    # 3. Apply overlay to RULE candidates
    for _, row in rule_df.iterrows():
        code = str(row.get("code", "")).zfill(6)
        score_col = "rule_score_v2" if "rule_score_v2" in row else "rule_score"
        original_score = float(row[score_col]) if score_col in row and pd.notna(row[score_col]) else None
        is_buy = bool(row.get("entry_signal", True))

        result = _apply_overlay_rules(
            code=code,
            name=str(row.get("name", "")) or None,
            sector=str(row.get("sector", "")) or None,
            original_score=original_score,
            is_buy_candidate=is_buy,
            macro=macro,
            cfg=cfg,
        )
        all_rows.append({**base_row, "engine_type": "rule", **result})

    # 4. Apply overlay to AI candidates
    for _, row in ai_df.iterrows():
        code = str(row.get("code", "")).zfill(6)
        original_score = float(row["final_score"]) if "final_score" in row and pd.notna(row["final_score"]) else None

        result = _apply_overlay_rules(
            code=code,
            name=str(row.get("name", "")) or None,
            sector=str(row.get("sector", "")) or None,
            original_score=original_score,
            is_buy_candidate=True,
            macro=macro,
            cfg=cfg,
        )
        all_rows.append({**base_row, "engine_type": "ai", **result})

    blocked_count = sum(1 for r in all_rows if r.get("buy_blocked_flag"))
    LOGGER.info(
        "[Phase 3] Overlay applied: %d rows | blocked: %d",
        len(all_rows), blocked_count,
    )

    # 5. Write to log table (shadow only, never touches orders)
    if dry_run:
        LOGGER.info("[Phase 3] DRY RUN - %d rows NOT written to DB.", len(all_rows))
        _log_shadow_summary(all_rows, macro)
        return {
            "kr_apply_date": kr_apply_date,
            "macro_status":  macro_status,
            "rule_rows":     sum(1 for r in all_rows if r["engine_type"] == "rule"),
            "ai_rows":       sum(1 for r in all_rows if r["engine_type"] == "ai"),
            "blocked_count": blocked_count,
        }

    if all_rows:
        with engine.begin() as conn:
            conn.execute(_INSERT_LOG_SQL, all_rows)
        LOGGER.info("[Phase 3] Inserted/updated %d rows in signal.kr_macro_overlay_log.", len(all_rows))
    else:
        LOGGER.warning("[Phase 3] No candidate rows - nothing written to log.")

    _log_shadow_summary(all_rows, macro)

    return {
        "kr_apply_date": kr_apply_date,
        "macro_status":  macro_status,
        "rule_rows":     sum(1 for r in all_rows if r["engine_type"] == "rule"),
        "ai_rows":       sum(1 for r in all_rows if r["engine_type"] == "ai"),
        "blocked_count": blocked_count,
    }


def _log_shadow_summary(rows: list[dict], macro: dict) -> None:
    """Print a readable Shadow Mode summary to the log."""
    blocked = [r for r in rows if r.get("buy_blocked_flag")]
    adjusted_up = [r for r in rows if (r.get("macro_adjustment") or 0) > 0]
    adjusted_down = [r for r in rows if not r.get("buy_blocked_flag") and (r.get("macro_adjustment") or 0) < 0]

    LOGGER.info("=" * 60)
    LOGGER.info("[SHADOW MODE] 이 결과는 실제 주문에 영향을 주지 않습니다.")
    LOGGER.info("-" * 60)
    LOGGER.info("US Macro Status  : %s", macro.get("macro_status"))
    LOGGER.info("Summary          : %s", macro.get("macro_summary", ""))
    LOGGER.info("-" * 60)
    LOGGER.info("Total candidates : %d", len(rows))
    LOGGER.info("Buy blocked      : %d", len(blocked))
    LOGGER.info("Score adjusted + : %d", len(adjusted_up))
    LOGGER.info("Score adjusted - : %d", len(adjusted_down))

    if blocked:
        LOGGER.info("--- Blocked candidates ---")
        for r in blocked[:10]:
            LOGGER.info(
                "  [%s] %s (%s) | original=%.1f → blocked | %s",
                r["engine_type"].upper(),
                r.get("name", r["code"]),
                r["code"],
                r.get("original_score") or 0,
                r.get("overlay_reason", ""),
            )

    if adjusted_up:
        LOGGER.info("--- Score increased ---")
        for r in adjusted_up[:5]:
            LOGGER.info(
                "  [%s] %s (%s) | %.1f → %.1f (%+.1f)",
                r["engine_type"].upper(),
                r.get("name", r["code"]),
                r["code"],
                r.get("original_score") or 0,
                r.get("adjusted_score") or 0,
                r.get("macro_adjustment") or 0,
            )

    if adjusted_down:
        LOGGER.info("--- Score decreased ---")
        for r in adjusted_down[:5]:
            LOGGER.info(
                "  [%s] %s (%s) | %.1f → %.1f (%+.1f) | %s",
                r["engine_type"].upper(),
                r.get("name", r["code"]),
                r["code"],
                r.get("original_score") or 0,
                r.get("adjusted_score") or 0,
                r.get("macro_adjustment") or 0,
                r.get("overlay_reason", ""),
            )

    LOGGER.info("=" * 60)
    LOGGER.info("[SHADOW MODE] 실제 주문에 반영되지 않음 - Phase 3 로그 전용.")
    LOGGER.info("=" * 60)
