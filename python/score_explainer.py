import json
from typing import Any

import pandas as pd


VERY_STRONG_THRESHOLD = 80.0
STRONG_THRESHOLD = 65.0
NEUTRAL_THRESHOLD = 45.0
WEAK_THRESHOLD = 30.0


DRIVER_TEXT = {
    "high_ret_score": "60일 기대수익 상위권",
    "high_probability_score": "상위권 진입 확률 우수",
    "high_tech_score": "추세 구조 양호",
    "strong_quality_profile": "재무 품질 양호",
    "strong_safety_profile": "변동성 관리 양호",
    "healthy_liquidity_profile": "거래 유동성 양호",
    "bull_regime_fit": "상승 국면 적합",
    "defensive_regime_fit": "방어형 국면 적합",
    "neutral_regime_fit": "중립 국면 적합",
}

RISK_TEXT = {
    "very_high_risk_penalty": "예상 손실 폭 부담 큼",
    "elevated_risk_penalty": "최근 변동성 확대 주의",
    "low_probability_score": "상위권 진입 확률 약함",
    "weak_quality_score": "재무 품질 보완 필요",
    "weak_tech_score": "추세 탄력 약함",
    "weak_safety_score": "방어력 약함",
    "low_liquidity_score": "거래 유동성 부족",
    "weak_ret_score": "기대수익 매력 낮음",
    "low_confidence": "신호 신뢰도 낮음",
    "multiple_component_fallbacks": "대체 데이터 사용 비중 높음",
    "partial_quality_coverage": "재무 데이터 일부 부족",
}


def _num(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def classify_component_score(value: Any) -> str:
    score = _num(value)
    if score is None:
        return "missing"
    if score >= VERY_STRONG_THRESHOLD:
        return "very_strong"
    if score >= STRONG_THRESHOLD:
        return "strong"
    if score >= NEUTRAL_THRESHOLD:
        return "neutral"
    if score >= WEAK_THRESHOLD:
        return "weak"
    return "very_weak"


def classify_risk_penalty(value: Any) -> str:
    score = _num(value)
    if score is None:
        return "unknown"
    if score < 4.0:
        return "low"
    if score < 8.0:
        return "moderate"
    if score < 12.0:
        return "high"
    return "very_high"


def _strength_candidates(row: pd.Series) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for col, code, priority_boost in [
        ("ret_score", "high_ret_score", 35.0),
        ("prob_score", "high_probability_score", 25.0),
        ("tech_score", "high_tech_score", 20.0),
        ("qual_score", "strong_quality_profile", 8.0),
        ("safety_score", "strong_safety_profile", 4.0),
        ("liquidity_score", "healthy_liquidity_profile", 2.0),
    ]:
        value = _num(row.get(col))
        band = classify_component_score(value)
        if value is None or band not in {"strong", "very_strong"}:
            continue
        priority = value + priority_boost + (10.0 if band == "very_strong" else 0.0)
        candidates.append({"code": code, "text": DRIVER_TEXT[code], "priority": priority})

    regime = str(row.get("regime") or "defensive").strip().lower()
    regime_fit_score = _num(row.get("regime_fitness_score"))
    if regime_fit_score is not None and regime_fit_score >= 70.0:
        regime_code = {
            "bull": "bull_regime_fit",
            "neutral": "neutral_regime_fit",
        }.get(regime, "defensive_regime_fit")
        candidates.append({"code": regime_code, "text": DRIVER_TEXT[regime_code], "priority": regime_fit_score + 5.0})

    return sorted(candidates, key=lambda item: item["priority"], reverse=True)


def _drag_candidates(row: pd.Series) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    risk_penalty = _num(row.get("risk_penalty"))
    risk_band = classify_risk_penalty(risk_penalty)
    if risk_penalty is not None and risk_band in {"high", "very_high"}:
        code = "elevated_risk_penalty" if risk_band == "high" else "very_high_risk_penalty"
        candidates.append({"code": code, "text": RISK_TEXT[code], "priority": risk_penalty + 60.0})

    for col, code in [
        ("prob_score", "low_probability_score"),
        ("qual_score", "weak_quality_score"),
        ("tech_score", "weak_tech_score"),
        ("safety_score", "weak_safety_score"),
        ("liquidity_score", "low_liquidity_score"),
        ("ret_score", "weak_ret_score"),
    ]:
        value = _num(row.get(col))
        band = classify_component_score(value)
        if value is None or band not in {"weak", "very_weak"}:
            continue
        candidates.append({"code": code, "text": RISK_TEXT[code], "priority": 100.0 - value})

    confidence_score = _num(row.get("confidence_score"))
    confidence_label = str(row.get("confidence_label") or "").strip()
    if confidence_score is not None and (confidence_score < 60.0 or confidence_label in {"Low", "Experimental"}):
        code = "low_confidence"
        candidates.append({"code": code, "text": RISK_TEXT[code], "priority": 160.0 - confidence_score})

    fallback_count = _num(row.get("fallback_count"))
    if fallback_count is not None and fallback_count >= 2:
        code = "multiple_component_fallbacks"
        candidates.append({"code": code, "text": RISK_TEXT[code], "priority": 125.0 + fallback_count})

    quality_conf = _num(row.get("quality_score_confidence"))
    if quality_conf is not None and quality_conf < 70.0:
        code = "partial_quality_coverage"
        candidates.append({"code": code, "text": RISK_TEXT[code], "priority": 145.0 - quality_conf})

    return sorted(candidates, key=lambda item: item["priority"], reverse=True)


def _confidence_text(row: pd.Series) -> str:
    label = str(row.get("confidence_label") or "Experimental").strip()
    reason = str(row.get("confidence_reason") or "").strip()
    if reason:
        return f"신뢰도 {label}: {reason}"
    return f"신뢰도 {label}"


def _regime_text(row: pd.Series) -> str:
    regime = str(row.get("regime") or "defensive").strip().lower()
    if regime == "bull":
        return "현재 bull 체계로 수익·확률·기술 비중이 확대 적용됐습니다."
    if regime == "neutral":
        return "현재 neutral 체계로 균형형 가중치가 적용됐습니다."
    return "현재 defensive 체계로 품질 비중과 리스크 통제가 강화 적용됐습니다."


_DEFAULT_STRENGTH_CODES = {"high_ret_score", "high_probability_score"}


def _ko_conj(word: str) -> str:
    """받침 있으면 '과', 없으면 '와'."""
    if not word:
        return "와"
    cp = ord(word[-1])
    if 0xAC00 <= cp <= 0xD7A3 and (cp - 0xAC00) % 28 != 0:
        return "과"
    return "와"


def _ko_subj(word: str) -> str:
    """받침 있으면 '이', 없으면 '가'."""
    if not word:
        return "이"
    cp = ord(word[-1])
    if 0xAC00 <= cp <= 0xD7A3 and (cp - 0xAC00) % 28 != 0:
        return "이"
    return "가"


def _summary_text(row: pd.Series, strengths: list[dict[str, Any]], drags: list[dict[str, Any]]) -> str:
    # 기본 강점(수익·확률)을 제외한 차별화 강점
    distinctive = [s for s in strengths if s["code"] not in _DEFAULT_STRENGTH_CODES]

    if strengths:
        # strengths[0]과 중복되지 않는 차별화 강점
        secondary = [s for s in distinctive if s["code"] != strengths[0]["code"]]
        if secondary:
            a = strengths[0]["text"]
            b = secondary[0]["text"]
            strength_part = f"{a}{_ko_conj(a)} {b}{_ko_subj(b)} 추천 점수를 견인했습니다."
        else:
            a = strengths[0]["text"]
            strength_part = f"{a}{_ko_subj(a)} 추천 점수를 견인했습니다."
    else:
        strength_part = "뚜렷한 추천 강점은 제한적입니다."

    risk_part = f"{drags[0]['text']} 요인은 함께 점검이 필요합니다." if drags else ""
    return f"{strength_part} {risk_part}".strip() if risk_part else strength_part


def _build_action_note(row: pd.Series, strengths: list[dict[str, Any]], drags: list[dict[str, Any]]) -> str:
    live_score = _num(row.get("live_score"))
    final_score = live_score if live_score is not None else (_num(row.get("final_score")) or 0.0)
    confidence_score = _num(row.get("confidence_score")) or 0.0
    risk_penalty = _num(row.get("risk_penalty")) or 0.0

    if final_score >= 65.0 and confidence_score >= 70.0 and risk_penalty < 8.0:
        return "우선 검토 후보"
    if final_score >= 55.0 and confidence_score >= 60.0:
        return "관심 종목으로 추적"
    if drags and drags[0]["code"] in {"very_high_risk_penalty", "low_confidence", "multiple_component_fallbacks"}:
        return "보수적 접근 권장"
    return "추가 확인 후 판단"


def _component_snapshot(row: pd.Series) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for col in [
        "ret_score",
        "prob_score",
        "qual_score",
        "tech_score",
        "valuation_score",
        "risk_penalty",
        "live_score",
        "final_score",
        "confidence_score",
    ]:
        value = _num(row.get(col))
        snapshot[col] = None if value is None else round(value, 2)
    snapshot["live_rank"] = _num(row.get("live_rank"))
    snapshot["live_score_source"] = str(row.get("live_score_source") or "") or None
    return snapshot


def _build_row_payload(row: pd.Series) -> pd.Series:
    strengths = _strength_candidates(row)[:3]
    drags = _drag_candidates(row)[:2]

    strength_text = " / ".join(item["text"] for item in strengths) if strengths else "뚜렷한 강점 제한적"
    risk_text = " / ".join(item["text"] for item in drags) if drags else "주요 리스크 제한적"
    confidence_text = _confidence_text(row)
    regime_text = _regime_text(row)
    summary_text = _summary_text(row, strengths, drags)
    action_note = _build_action_note(row, strengths, drags)

    payload = {
        "drivers": [item["code"] for item in strengths],
        "driver_texts": [item["text"] for item in strengths],
        "drags": [item["code"] for item in drags],
        "drag_texts": [item["text"] for item in drags],
        "confidence_label": str(row.get("confidence_label") or ""),
        "confidence_reason": str(row.get("confidence_reason") or ""),
        "regime": str(row.get("regime") or ""),
        "weight_profile": str(row.get("weight_profile") or ""),
        "regime_reason": str(row.get("regime_reason") or ""),
        "action_note": action_note,
        "component_snapshot": _component_snapshot(row),
    }
    explain_json = json.dumps(payload, ensure_ascii=False)

    return pd.Series(
        {
            "score_explain_summary": summary_text,
            "score_explain_strengths": strength_text,
            "score_explain_risks": risk_text,
            "score_explain_confidence": confidence_text,
            "score_explain_regime": regime_text,
            "score_driver_1": strengths[0]["code"] if len(strengths) >= 1 else None,
            "score_driver_2": strengths[1]["code"] if len(strengths) >= 2 else None,
            "score_driver_3": strengths[2]["code"] if len(strengths) >= 3 else None,
            "score_drag_1": drags[0]["code"] if len(drags) >= 1 else None,
            "score_drag_2": drags[1]["code"] if len(drags) >= 2 else None,
            "top_driver_1": strengths[0]["text"] if len(strengths) >= 1 else None,
            "top_driver_2": strengths[1]["text"] if len(strengths) >= 2 else None,
            "top_driver_3": strengths[2]["text"] if len(strengths) >= 3 else None,
            "risk_factor_1": drags[0]["text"] if len(drags) >= 1 else None,
            "risk_factor_2": drags[1]["text"] if len(drags) >= 2 else None,
            "confidence_reason": str(row.get("confidence_reason") or ""),
            "action_note": action_note,
            "score_explain_json": explain_json,
        }
    )


def attach_score_explain_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    explain = out.apply(_build_row_payload, axis=1)
    for col in explain.columns:
        out[col] = explain[col]
    return out
