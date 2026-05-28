"""ranking_final.csv 상위 10개 종목의 진입 시점 안전성을 평가하여 점수와 등급 부여.

[Stage 2] 점수 산출 체인의 진입 위험 평가 단계.

이 스크립트는 모델 ranking과 무관한 "진입 시점 위험" 신호만 평가합니다.
ranking_builder가 "좋은 종목"을 고르고, 이 스크립트는 "지금 진입해도 안전한가"를 판단합니다.

입력:
- data/ranking_final.csv: 상위 10개 종목과 관련 지표
  (liquidity_score, risk_penalty, ret_5d, ret_10d, rsi_14, vol_20_pct 등)

출력:
- data/ai_entry_quality_score.csv: 종목별 entry_quality_score (가중평균) + grade + status
- outputs/ai_entry_quality_score.json
- outputs/ai_entry_quality_score_report.md

entry_quality_score 산출:
    entry_quality_score = sum(component_score × weight) / sum(weights)
    components: 모멘텀(_momentum_score), RSI(_rsi_score), 변동성(_volatility_score), entry gate(_entry_gate_score)

entry_quality_status 결정 (BLOCK/WATCH/PASS):
    BLOCK 조건 (가장 강한 차단):
        - liquidity_score < 30
        - risk_penalty >= 18
        - ret_10d >= 20% (이미 과열)
        - rsi_14 >= 80 (과매수 극단)
    WATCH 조건 (보유 가능하나 주의):
        - liquidity_score < 50
        - risk_penalty >= 17
        - ret_10d >= 12% or ret_5d >= 12%
        - rsi_14 >= 75
    PASS: 위 조건 모두 미해당

소비처:
- build_ai_filtered_top_candidates.py가 entry_quality_score를 가산점으로 사용 (가중치 0.20).
- build_trade_intents.py가 entry_quality_status를 매수/보유 의사결정에 참고.

설계 메모 (2026-05-29 추가):
- 임계값(30/50, 17/18, 12%/20%, 75/80)의 출처는 추적 불가능 — 공식 설계 문서 없음.
- 운영 데이터로 false positive(불필요 차단) / false negative(놓친 위험) 측정 후 튜닝 권장.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_RANKING_CSV = DATA_DIR / "ranking_final.csv"
DEFAULT_CANDIDATES_CSV = DATA_DIR / "ai_filtered_top_candidates.csv"
DEFAULT_OUT_CSV = DATA_DIR / "ai_entry_quality_score.csv"
DEFAULT_OUT_JSON = OUTPUT_DIR / "ai_entry_quality_score.json"
DEFAULT_OUT_MD = OUTPUT_DIR / "ai_entry_quality_score_report.md"

REQUIRED_COLUMNS = ["code", "final_score", "liquidity_score", "risk_penalty", "ret_5d", "ret_10d"]
OPTIONAL_ENTRY_GATE_COLUMNS = [
    "entry_price_gate_status",
    "entry_price_gate_reason",
    "entry_price_gap_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI entry quality score layer for AI top candidates.")
    parser.add_argument("--ranking-csv", type=Path, default=DEFAULT_RANKING_CSV)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _to_float(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _fmt_num(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric) * 100:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    work = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(work.columns.tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in work.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


def load_csv(path: Path, *, label: str) -> pd.DataFrame:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    df = pd.read_csv(resolved, dtype={"code": str}, low_memory=False)
    if df.empty:
        raise ValueError(f"{label} is empty: {resolved}")
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def prepare_base(ranking: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in candidates.columns and col not in ranking.columns]
    if missing:
        raise ValueError(f"required columns missing from inputs: {', '.join(missing)}")

    ranking_cols = {col for col in ranking.columns}
    candidate_cols = {col for col in candidates.columns}
    missing_in_candidates = sorted([col for col in REQUIRED_COLUMNS if col not in candidate_cols and col in ranking_cols])

    base = candidates.copy()
    base["code"] = base["code"].astype(str).str.zfill(6)
    if "date" in base.columns:
        base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        base["date"] = ""

    merge_cols = [col for col in ranking.columns if col not in base.columns or col in missing_in_candidates]
    merge_cols = ["code", *[col for col in merge_cols if col != "code"]]
    ranking_slice = ranking.loc[:, merge_cols].copy()
    ranking_slice = ranking_slice.drop_duplicates("code", keep="first")
    base = base.merge(ranking_slice, on="code", how="left", suffixes=("", "_ranking"))

    for col in set(base.columns):
        if not col.endswith("_ranking"):
            continue
        raw_col = col[:-8]
        if raw_col not in base.columns:
            base[raw_col] = base[col]
        else:
            current = base[raw_col]
            base[raw_col] = current.where(current.notna(), base[col])
        base = base.drop(columns=[col])

    numeric_cols = [
        "final_score",
        "liquidity_score",
        "risk_penalty",
        "ret_5d",
        "ret_10d",
        "rsi_14",
        "vol_20",
        "vol_60",
        "vol_20_pct",
        "vol_60_pct",
        "entry_price_gap_pct",
        "ai_top10_rank",
        "ai_filtered_rank",
    ]
    for col in numeric_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if "name" in base.columns:
        base["name"] = base["name"].fillna("").astype(str)
    else:
        base["name"] = ""

    meta = {
        "candidate_row_count": int(len(base)),
        "missing_in_candidates_filled_from_ranking": missing_in_candidates,
        "available_entry_gate_columns": [col for col in OPTIONAL_ENTRY_GATE_COLUMNS if col in base.columns],
        "missing_entry_gate_columns": [col for col in OPTIONAL_ENTRY_GATE_COLUMNS if col not in base.columns],
        "available_rsi_column": "rsi_14" if "rsi_14" in base.columns else None,
        "available_volatility_columns": [col for col in ["vol_20_pct", "vol_60_pct", "vol_20", "vol_60"] if col in base.columns],
    }
    return base, meta


def _momentum_score(value: float | None, *, warn_level: float, block_level: float | None = None) -> float | None:
    if value is None:
        return None
    if value <= 0.0:
        return 94.0
    if value <= warn_level:
        return _clamp(100.0 - (value / warn_level) * 35.0)
    if block_level is None:
        return _clamp(65.0 - ((value - warn_level) / max(warn_level, 1e-8)) * 50.0)
    if value <= block_level:
        span = max(block_level - warn_level, 1e-8)
        return _clamp(65.0 - ((value - warn_level) / span) * 50.0)
    return _clamp(15.0 - ((value - block_level) / max(block_level, 1e-8)) * 20.0)


def _rsi_score(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 60.0:
        return 92.0
    if value <= 70.0:
        return _clamp(92.0 - (value - 60.0) * 2.8)
    if value <= 75.0:
        return _clamp(64.0 - (value - 70.0) * 5.0)
    return _clamp(35.0 - (value - 75.0) * 4.0)


def _volatility_score(row: pd.Series) -> tuple[float | None, str | None]:
    for col in ["vol_20_pct", "vol_60_pct"]:
        if col in row.index:
            value = _to_float(row.get(col))
            if value is None:
                continue
            if value <= 0.20:
                return 92.0, col
            if value <= 0.35:
                return _clamp(92.0 - ((value - 0.20) / 0.15) * 30.0), col
            if value <= 0.60:
                return _clamp(62.0 - ((value - 0.35) / 0.25) * 42.0), col
            return _clamp(20.0 - ((value - 0.60) / 0.20) * 10.0), col
    for col in ["vol_20", "vol_60"]:
        if col in row.index:
            value = _to_float(row.get(col))
            if value is None:
                continue
            if value <= 0.25:
                return 90.0, col
            if value <= 0.40:
                return _clamp(90.0 - ((value - 0.25) / 0.15) * 28.0), col
            if value <= 0.65:
                return _clamp(62.0 - ((value - 0.40) / 0.25) * 42.0), col
            return _clamp(18.0 - ((value - 0.65) / 0.20) * 10.0), col
    return None, None


def _entry_gate_score(row: pd.Series) -> tuple[float | None, list[str]]:
    reasons: list[str] = []
    status = str(row.get("entry_price_gate_status") or "").strip().lower()
    reason = str(row.get("entry_price_gate_reason") or "").strip()
    gap_pct = _to_float(row.get("entry_price_gap_pct"))

    if not status and not reason and gap_pct is None:
        return None, ["entry_price_gate columns unavailable in input data"]

    if status in {"blocked", "block"}:
        reasons.append(f"entry_price_gate_status={status}")
        if reason:
            reasons.append(f"entry_price_gate_reason={reason}")
        return 0.0, reasons
    if reason in {"entry_gap_up_hard_blocked", "entry_gap_up_blocked", "entry_gap_down_blocked"}:
        reasons.append(f"entry_price_gate_reason={reason}")
        return 0.0, reasons
    if reason == "entry_gap_ok" or status == "ok":
        return 100.0, ["entry_price_gate indicates ok"]
    if gap_pct is not None:
        if gap_pct >= 0.05:
            reasons.append(f"entry_price_gap_pct {_fmt_pct(gap_pct)} indicates stretched entry")
            return 35.0, reasons
        if gap_pct <= -0.03:
            reasons.append(f"entry_price_gap_pct {_fmt_pct(gap_pct)} indicates weak open")
            return 55.0, reasons
        reasons.append(f"entry_price_gap_pct {_fmt_pct(gap_pct)} inside neutral range")
        return 88.0, reasons
    if reason:
        reasons.append(f"entry_price_gate_reason={reason}")
    return 70.0, reasons or ["entry_price_gate data partially available"]


def _grade_from_score(score: float | None, *, unknown: bool) -> str:
    if unknown or score is None:
        return "UNKNOWN"
    if score >= 85.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 55.0:
        return "C"
    return "D"


def evaluate_row(row: pd.Series) -> dict[str, object]:
    reasons: list[str] = []
    status = "PASS"
    unknown = False

    liquidity = _to_float(row.get("liquidity_score"))
    risk_penalty = _to_float(row.get("risk_penalty"))
    ret_5d = _to_float(row.get("ret_5d"))
    ret_10d = _to_float(row.get("ret_10d"))
    rsi = _to_float(row.get("rsi_14"))
    gate_score, gate_reasons = _entry_gate_score(row)
    vol_score, vol_col = _volatility_score(row)

    component_scores: list[tuple[str, float, float]] = []

    if liquidity is None:
        unknown = True
        reasons.append("liquidity_score missing")
    else:
        component_scores.append(("liquidity_score", _clamp(liquidity), 0.24))
        if liquidity < 30.0:
            status = "BLOCK"
            reasons.append(f"liquidity_score {_fmt_num(liquidity)} < 30 -> BLOCK")
        elif liquidity < 50.0:
            if status != "BLOCK":
                status = "WATCH"
            reasons.append(f"liquidity_score {_fmt_num(liquidity)} < 50 -> WATCH")

    if risk_penalty is None:
        unknown = True
        reasons.append("risk_penalty missing")
    else:
        risk_score = _clamp(100.0 - risk_penalty * 5.0)
        component_scores.append(("risk_penalty", risk_score, 0.22))
        if risk_penalty >= 18.0:
            status = "BLOCK"
            reasons.append(f"risk_penalty {_fmt_num(risk_penalty)} >= 18 -> BLOCK")
        elif risk_penalty >= 17.0 and status != "BLOCK":
            status = "WATCH"
            reasons.append(f"risk_penalty {_fmt_num(risk_penalty)} >= 17 -> WATCH")

    if ret_10d is None:
        unknown = True
        reasons.append("ret_10d missing")
    else:
        score = _momentum_score(ret_10d, warn_level=0.12, block_level=0.20)
        if score is not None:
            component_scores.append(("ret_10d", score, 0.14))
        if ret_10d >= 0.20:
            status = "BLOCK"
            reasons.append(f"ret_10d {_fmt_pct(ret_10d)} >= 20.00% -> BLOCK")
        elif ret_10d >= 0.12 and status != "BLOCK":
            status = "WATCH"
            reasons.append(f"ret_10d {_fmt_pct(ret_10d)} >= 12.00% -> WATCH")

    if ret_5d is None:
        unknown = True
        reasons.append("ret_5d missing")
    else:
        score = _momentum_score(ret_5d, warn_level=0.08, block_level=0.12)
        if score is not None:
            component_scores.append(("ret_5d", score, 0.12))
        if ret_5d >= 0.12 and status != "BLOCK":
            status = "WATCH"
            reasons.append(f"ret_5d {_fmt_pct(ret_5d)} >= 12.00% -> WATCH")

    if rsi is None:
        unknown = True
        reasons.append("rsi_14 missing")
    else:
        score = _rsi_score(rsi)
        if score is not None:
            component_scores.append(("rsi_14", score, 0.10))
        if rsi >= 80.0:
            status = "BLOCK"
            reasons.append(f"rsi_14 {_fmt_num(rsi)} >= 80 -> BLOCK")
        elif rsi >= 75.0 and status != "BLOCK":
            status = "WATCH"
            reasons.append(f"rsi_14 {_fmt_num(rsi)} >= 75 -> WATCH")

    if vol_score is None:
        unknown = True
        reasons.append("volatility columns missing or empty")
    else:
        component_scores.append((vol_col or "volatility", vol_score, 0.10))
        if vol_score < 35.0 and status != "BLOCK":
            status = "WATCH"
            reasons.append(f"{vol_col} implies elevated volatility -> WATCH")

    if gate_score is not None:
        component_scores.append(("entry_price_gate", gate_score, 0.08))
        reasons.extend(gate_reasons)
        if gate_score <= 0.0:
            status = "BLOCK"
            reasons.append("entry_price_gate layer suggests blocked entry context")
        elif gate_score < 45.0 and status != "BLOCK":
            status = "WATCH"
            reasons.append("entry_price_gate layer suggests stretched entry context")
    else:
        reasons.extend(gate_reasons)

    if not reasons:
        reasons.append("entry context looks acceptable on available data")

    total_weight = sum(weight for _, _, weight in component_scores)
    if total_weight > 0:
        entry_quality_score = round(sum(score * weight for _, score, weight in component_scores) / total_weight, 2)
    else:
        entry_quality_score = None

    if unknown and status == "PASS":
        status = "UNKNOWN"

    grade = _grade_from_score(entry_quality_score, unknown=unknown and status == "UNKNOWN")
    return {
        "entry_quality_score": entry_quality_score,
        "entry_quality_grade": grade,
        "entry_quality_status": status,
        "entry_quality_reasons": "; ".join(dict.fromkeys(reasons)),
        "entry_quality_unknown_data": unknown,
        "entry_quality_component_count": len(component_scores),
    }


def evaluate_candidates(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        result = dict(row)
        result.update(evaluate_row(row))
        rows.append(result)
    out = pd.DataFrame(rows)
    status_order = {"BLOCK": 0, "WATCH": 1, "UNKNOWN": 2, "PASS": 3}
    out["status_sort_key"] = out["entry_quality_status"].map(status_order).fillna(9)
    out = out.sort_values(
        ["status_sort_key", "ai_filtered_rank", "ai_top10_rank", "final_score", "code"],
        ascending=[True, True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out.drop(columns=["status_sort_key"], errors="ignore")


def build_json_payload(result: pd.DataFrame, meta: dict[str, object]) -> dict[str, object]:
    status_counts = result["entry_quality_status"].fillna("UNKNOWN").astype(str).value_counts().to_dict()

    def _row_payload(row: pd.Series) -> dict[str, object]:
        return {
            "code": str(row.get("code") or "").zfill(6),
            "name": str(row.get("name") or ""),
            "ai_top10_rank": None if pd.isna(row.get("ai_top10_rank")) else int(row.get("ai_top10_rank")),
            "ai_filtered_rank": None if pd.isna(row.get("ai_filtered_rank")) else int(row.get("ai_filtered_rank")),
            "selected_for_ai_top5": bool(row.get("selected_for_ai_top5")) if pd.notna(row.get("selected_for_ai_top5")) else False,
            "fallback_selected": bool(row.get("fallback_selected")) if pd.notna(row.get("fallback_selected")) else False,
            "entry_quality_score": None if pd.isna(row.get("entry_quality_score")) else float(row.get("entry_quality_score")),
            "entry_quality_grade": str(row.get("entry_quality_grade") or ""),
            "entry_quality_status": str(row.get("entry_quality_status") or ""),
            "entry_quality_reasons": str(row.get("entry_quality_reasons") or ""),
            "liquidity_score": None if pd.isna(row.get("liquidity_score")) else float(row.get("liquidity_score")),
            "risk_penalty": None if pd.isna(row.get("risk_penalty")) else float(row.get("risk_penalty")),
            "ret_5d": None if pd.isna(row.get("ret_5d")) else float(row.get("ret_5d")),
            "ret_10d": None if pd.isna(row.get("ret_10d")) else float(row.get("ret_10d")),
            "rsi_14": None if pd.isna(row.get("rsi_14")) else float(row.get("rsi_14")),
        }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": {
            "ranking_csv": str(DEFAULT_RANKING_CSV),
            "candidates_csv": str(DEFAULT_CANDIDATES_CSV),
        },
        "score_formula": {
            "range": "0-100",
            "components": {
                "liquidity_score": {"weight": 0.24, "direction": "higher_better"},
                "risk_penalty": {"weight": 0.22, "direction": "lower_better"},
                "ret_5d": {"weight": 0.12, "direction": "lower_heat_better"},
                "ret_10d": {"weight": 0.14, "direction": "lower_heat_better"},
                "rsi_14": {"weight": 0.10, "direction": "lower_heat_better"},
                "volatility": {"weight": 0.10, "direction": "lower_better"},
                "entry_price_gate": {"weight": 0.08, "direction": "entry_context_better"},
            },
            "status_rules": {
                "block": [
                    "liquidity_score < 30",
                    "risk_penalty >= 18",
                    "ret_10d >= 20%",
                    "rsi_14 >= 80",
                    "entry_price_gate indicates blocked context when available",
                ],
                "watch": [
                    "ret_5d >= 12%",
                    "ret_10d >= 12%",
                    "rsi_14 >= 75",
                    "liquidity_score < 50",
                    "risk_penalty >= 17",
                ],
                "unknown": [
                    "required price/technical data missing and no stronger WATCH/BLOCK signal",
                ],
            },
        },
        "summary": {
            "row_count": int(len(result)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "watch_count": int((result["entry_quality_status"] == "WATCH").sum()),
            "block_count": int((result["entry_quality_status"] == "BLOCK").sum()),
            "unknown_count": int((result["entry_quality_status"] == "UNKNOWN").sum()),
            "pass_count": int((result["entry_quality_status"] == "PASS").sum()),
        },
        "data_availability": meta,
        "rows": [_row_payload(row) for _, row in result.iterrows()],
    }


def build_markdown_report(result: pd.DataFrame, payload: dict[str, object], meta: dict[str, object]) -> str:
    display = result.copy()
    for col in ["final_score", "liquidity_score", "risk_penalty", "entry_quality_score"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: _fmt_num(x, 2))
    for col in ["ret_5d", "ret_10d", "entry_price_gap_pct"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: _fmt_pct(x, 2))
    if "rsi_14" in display.columns:
        display["rsi_14"] = display["rsi_14"].map(lambda x: _fmt_num(x, 2))
    for col in ["selected_for_ai_top5", "fallback_selected", "filter_passed"]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "Y" if bool(x) else "N")
    if "ai_filtered_rank" in display.columns:
        display["ai_filtered_rank"] = display["ai_filtered_rank"].map(lambda x: "" if pd.isna(x) else str(int(x)))

    watch_block = result.loc[result["entry_quality_status"].isin(["WATCH", "BLOCK"])].copy()
    status_counts = payload.get("summary", {}).get("status_counts", {})

    lines: list[str] = [
        "# AI Entry Quality Score Report",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- ranking_csv: {payload.get('source_files', {}).get('ranking_csv')}",
        f"- candidates_csv: {payload.get('source_files', {}).get('candidates_csv')}",
        f"- row_count: {len(result)}",
        f"- status_counts: {status_counts}",
        "",
        "## Score Formula",
        "",
        "- liquidity_score 24%",
        "- risk_penalty 22%",
        "- ret_5d 12%",
        "- ret_10d 14%",
        "- rsi_14 10% when available",
        "- volatility 10% when available",
        "- entry_price_gate layer 8% when available",
        "- available components are re-normalized when some optional inputs are missing",
        "",
        "## Data Availability",
        "",
        f"- missing_in_candidates_filled_from_ranking: {meta.get('missing_in_candidates_filled_from_ranking')}",
        f"- available_entry_gate_columns: {meta.get('available_entry_gate_columns')}",
        f"- missing_entry_gate_columns: {meta.get('missing_entry_gate_columns')}",
        f"- available_rsi_column: {meta.get('available_rsi_column')}",
        f"- available_volatility_columns: {meta.get('available_volatility_columns')}",
        "",
        "## Status Summary",
        "",
        f"- PASS: {int((result['entry_quality_status'] == 'PASS').sum())}",
        f"- WATCH: {int((result['entry_quality_status'] == 'WATCH').sum())}",
        f"- BLOCK: {int((result['entry_quality_status'] == 'BLOCK').sum())}",
        f"- UNKNOWN: {int((result['entry_quality_status'] == 'UNKNOWN').sum())}",
        "",
        "## Candidate Evaluation",
        "",
        _markdown_table(
            display,
            [
                "ai_top10_rank",
                "ai_filtered_rank",
                "code",
                "name",
                "selected_for_ai_top5",
                "fallback_selected",
                "final_score",
                "liquidity_score",
                "risk_penalty",
                "ret_5d",
                "ret_10d",
                "rsi_14",
                "entry_quality_score",
                "entry_quality_grade",
                "entry_quality_status",
                "entry_quality_reasons",
            ],
        ),
        "",
        "## WATCH/BLOCK Candidates",
        "",
    ]
    if watch_block.empty:
        lines.extend(["- none", ""])
    else:
        for _, row in watch_block.iterrows():
            lines.append(
                f"- {row['entry_quality_status']} top10_rank={_fmt_num(row.get('ai_top10_rank'), 0)} "
                f"{str(row.get('code') or '').zfill(6)} {row.get('name') or ''}: "
                f"{row.get('entry_quality_reasons') or ''}"
            )
        lines.append("")
    return "\n".join(lines)


def save_outputs(result: pd.DataFrame, payload: dict[str, object], out_csv: Path, out_json: Path, out_md: Path, meta: dict[str, object]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown_report(result, payload, meta), encoding="utf-8")


def print_summary(result: pd.DataFrame, meta: dict[str, object]) -> None:
    print(f"row_count={len(result)}")
    for status in ["PASS", "WATCH", "BLOCK", "UNKNOWN"]:
        print(f"{status.lower()}_count={int((result['entry_quality_status'] == status).sum())}")

    watch_block = result.loc[result["entry_quality_status"].isin(["WATCH", "BLOCK"])].copy()
    if watch_block.empty:
        print("watch_block_candidates=none")
    else:
        print("watch_block_candidates:")
        for _, row in watch_block.iterrows():
            print(
                f" - {row['entry_quality_status']} "
                f"top10_rank={_fmt_num(row.get('ai_top10_rank'), 0)} "
                f"{str(row.get('code') or '').zfill(6)} {row.get('name') or ''}: "
                f"{row.get('entry_quality_reasons') or ''}"
            )

    print(f"missing_entry_gate_columns={meta.get('missing_entry_gate_columns')}")
    print(f"available_volatility_columns={meta.get('available_volatility_columns')}")
    print(f"csv={DEFAULT_OUT_CSV}")
    print(f"json={DEFAULT_OUT_JSON}")
    print(f"md={DEFAULT_OUT_MD}")


def main() -> None:
    args = parse_args()
    ranking = load_csv(args.ranking_csv, label="ranking csv")
    candidates = load_csv(args.candidates_csv, label="ai filtered candidates csv")
    base, meta = prepare_base(ranking, candidates)
    result = evaluate_candidates(base)
    payload = build_json_payload(result, meta)
    save_outputs(result, payload, _resolve(args.out_csv), _resolve(args.out_json), _resolve(args.out_md), meta)
    print_summary(result, meta)


if __name__ == "__main__":
    main()
