"""
블로그 게시용 글 초안 생성 스크립트 (경량 MVP).

종가 데이터를 Jinja2 템플릿에 적용해 네이버/티스토리용 글 파일을 출력한다.
출력 파일은 운영자가 직접 복사해 외부 블로그에 수동 게시한다.

핵심 설계:
- 저품질(유사문서) 회피를 위해 blog_variation.VariationPicker로 제목/도입/문구/구조를 날짜별로 변형.
- 같은 날 같은 입력이면 동일 결과(멱등) → 이미 존재하는 파일은 덮어쓰지 않음(--force로 강제).
- 네이버/티스토리는 시드 살트가 달라 같은 날이라도 표현이 달라짐(교차 중복 방지).

사용 예:
    python python/generate_blog_posts.py
    python python/generate_blog_posts.py --type A --date 2026-05-28
    python python/generate_blog_posts.py --type A --platform naver --force
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader, select_autoescape
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blog_variation import VariationPicker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVING_DIR = ROOT / "serving"
TEMPLATE_DIR = ROOT / "templates" / "blog"
OUTPUT_DIR = ROOT / "outputs" / "blog_drafts"
POLISHED_DIR = ROOT / "outputs" / "blog_polished"
LOG_DIR = ROOT / "logs"

PLATFORM_EXT = {"naver": "txt", "tistory": "md"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("blog_gen")


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def load_recommendations() -> dict:
    path = SERVING_DIR / "daily_recommendations.json"
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def korean_date(source_date: str) -> str:
    """'2026-05-28' -> '2026년 5월 28일'"""
    y, m, d = source_date.split("-")
    return f"{int(y)}년 {int(m)}월 {int(d)}일"


def parse_regime_reason(reason: str | None) -> str:
    """regime_reason 문자열에서 사람이 읽을 한 줄 요약 생성."""
    if not reason:
        return ""
    kv: dict[str, str] = {}
    for seg in reason.split(";"):
        seg = seg.strip()
        if "=" in seg:
            k, v = seg.split("=", 1)
            kv[k.strip()] = v.strip()
    bits: list[str] = []
    if "breadth_20d" in kv:
        try:
            bits.append(f"상승 종목 비율 약 {round(float(kv['breadth_20d']) * 100)}%")
        except ValueError:
            pass
    if "recent_20d_return" in kv:
        try:
            r = float(kv["recent_20d_return"]) * 100
            bits.append(f"최근 20일 수익률 {r:+.1f}%")
        except ValueError:
            pass
    base = ", ".join(bits)
    if str(kv.get("volatility_risk_flag", "")).lower() == "true":
        base = (base + " 수준이며, 변동성 주의 신호가 켜져 있습니다.") if base else "변동성 주의 신호가 켜져 있습니다."
    elif base:
        base += " 수준입니다."
    return base


def _has_batchim(ch: str):
    """한글 음절의 받침 유무. 한글이 아니면 None."""
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return None


def josa(word: str, has: str, no: str) -> str:
    """단어 끝 받침에 맞는 조사 부착(은/는, 이/가, 을/를 등)."""
    for ch in reversed(word):
        b = _has_batchim(ch)
        if b is not None:
            return word + (has if b else no)
        if ch.isdigit():
            return word + (no if ch in "2459" else has)
    return word + no  # 영문·기호로 끝나면 받침 없는 형태(는/가/를)로


def _score_band(score: float) -> str:
    if score >= 75:
        return "최상위권"
    if score >= 70:
        return "상위권"
    if score >= 60:
        return "중상위권"
    return "중위권"


def _rsi_band(rsi: float) -> str:
    if rsi < 30:
        return "과매도"
    if rsi < 45:
        return "약세"
    if rsi < 55:
        return "중립"
    if rsi < 70:
        return "강세"
    return "과매수"


def _risk_sentence(ret_pct: float, mdd_pct: float, formal: bool) -> str:
    big_mdd = abs(mdd_pct) >= 20
    if ret_pct >= 30 and big_mdd:
        return "기대 수익률이 높은 만큼 예상 낙폭도 커, 변동성이 큰 구간으로 " + ("해석할 수 있습니다" if formal else "해석할 수 있다")
    if ret_pct >= 20 and big_mdd:
        return "기대 수익률은 양호하지만 예상 낙폭이 작지 않아 " + ("균형 있게 볼 필요가 있습니다" if formal else "균형 있게 볼 필요가 있다")
    if ret_pct >= 20:
        return "기대 수익률 신호가 살아 있는 편이라 흐름을 " + ("지켜볼 만합니다" if formal else "지켜볼 만하다")
    return "기대 수익률과 낙폭이 모두 제한적이라 보수적으로 " + ("관찰하는 편이 낫습니다" if formal else "관찰하는 편이 낫다")


def build_stock_commentary(s: dict, picker: VariationPicker, formal: bool) -> str:
    """종목 지표를 한두 문장의 해석형 산문으로 변환.

    formal=True  → 네이버용 존댓말(~합니다)
    formal=False → 티스토리용 평어(~한다)
    조사는 josa()로 받침에 맞춰 처리한다.
    """
    name = s["name"]
    band = _score_band(s["final_score"])
    rsi_band = _rsi_band(s["rsi"])
    ret, mdd, score, rsi = s["pred_return_pct"], s["pred_mdd_pct"], s["final_score"], s["rsi"]
    name_eunneun = josa(name, "은", "는")
    name_ui = name + "의"

    e = {  # 어미
        "locate": "자리합니다" if formal else "자리한다",
        "flow": "흐름입니다" if formal else "흐름이다",
        "expect": "예상합니다" if formal else "예상한다",
        "forecast": "전망됩니다" if formal else "전망된다",
        "present": "제시합니다" if formal else "제시한다",
        "phase": "국면입니다" if formal else "국면이다",
        "level": "수준입니다" if formal else "수준이다",
    }
    variant = picker.choice_index(5, "stock_comment", s["code"])
    if variant == 0:
        lead = (f"{name_eunneun} AI 종합점수 {score}점으로 {band}에 {e['locate']}. "
                f"RSI는 {rsi}로 {rsi_band} 구간이며, 모델은 60일 기준 약 {ret}%의 "
                f"수익과 {mdd}%의 낙폭을 {e['expect']}.")
    elif variant == 1:
        nm = josa(name, "은", "는")
        lead = (f"점수 {score}점({band})을 받은 {nm} 현재 RSI {rsi}, "
                f"즉 {rsi_band} {e['flow']}. 모델 추정으로는 60일간 {ret}% 수익, "
                f"{mdd}% 낙폭이 {e['forecast']}.")
    elif variant == 2:
        lead = (f"{name_ui} AI 점수는 {score}점으로 {band} {e['level']}. "
                f"기술적으로는 RSI {rsi}의 {rsi_band} {e['phase']}. "
                f"모델은 60일 수익 {ret}%·낙폭 {mdd}%를 {e['present']}.")
    elif variant == 3:
        nm = josa(name, "은", "는")
        lead = (f"{nm} 기술적으로 RSI {rsi}의 {rsi_band} 흐름 속에서 AI 종합점수 {score}점({band})을 "
                f"기록했습니다." if formal else
                f"{nm} 기술적으로 RSI {rsi}의 {rsi_band} 흐름 속에서 AI 종합점수 {score}점({band})을 기록했다.")
        lead += (f" 모델이 보는 60일 그림은 수익 {ret}%, 낙폭 {mdd}%"
                 + ("입니다." if formal else "다."))
    else:
        nm = josa(name, "은", "는")
        lead = (f"{band} 점수({score}점)를 받은 {nm}, 60일 기준 수익 {ret}%·낙폭 {mdd}%가 "
                f"{e['forecast']}. RSI {rsi}로 {rsi_band} {e['phase']}.")
    risk = _risk_sentence(ret, mdd, formal)
    return lead + " " + risk + "."


def build_market_commentary(regime_word: str, regime_reason_human: str,
                            picker: VariationPicker, formal: bool) -> str:
    """시장 국면을 2~3문장 산문으로 확장(문체 분기)."""
    parts: list[str] = []
    head = {
        "상승": "전반적으로 매수 우위가 읽히는 국면",
        "중립": "뚜렷한 방향성보다 종목별 차별화가 나타나는 국면",
        "방어": "위험 관리가 우선되는 보수적 국면",
    }.get(regime_word, "방향성을 탐색하는 국면")
    cls = "분류됐습니다" if formal else "분류됐다"
    safe = "해석하는 편이 안전합니다" if formal else "해석하는 편이 안전하다"
    parts.append(f"오늘 시장은 '{regime_word}'으로 {cls}. {head}{'입니다' if formal else '이다'}.")
    if regime_reason_human:
        parts.append("세부 지표로는 " + regime_reason_human)
    parts.append(f"이런 국면에서는 개별 종목의 점수만 보기보다 전체 흐름과 함께 {safe}.")
    return " ".join(parts)


def prepare_items(data: dict, picker: VariationPicker, limit: int) -> list[dict]:
    items: list[dict] = []
    for it in data.get("items", [])[:limit]:
        sec = it.get("security", {})
        sc = it.get("scores", {})
        ms = it.get("market_signals", {})
        be = it.get("buy_eligibility", {})
        sel = it.get("selection", {})

        support = picker.reasons_kr(sel.get("buyability_supporting_reasons", []))
        block_codes = list(sel.get("buyability_blocking_reasons", [])) + list(be.get("caution_reasons", []))
        block = picker.reasons_kr(block_codes)  # 부분매핑·영문제거·중복제거 포함

        # 업종/테마 정리: (unknown)/(none)/빈값은 노출하지 않음
        sector = sec.get("sector") or ""
        if sector in ("(unknown)", "-"):
            sector = "기타"
        theme = sec.get("dominant_theme") or ""
        if theme in ("(none)", "-"):
            theme = ""
        # 업종에 테마가 이미 포함되면(예: 업종 '바이오/제약', 테마 '바이오') 중복 제거
        if theme and theme in sector:
            theme = ""
        sector_theme = sector + (f" / {theme}" if theme else "")

        items.append({
            "rank": it.get("buy_rank"),
            "name": sec.get("name", "-"),
            "code": str(sec.get("code", "")),
            "market": sec.get("market", ""),
            "sector": sector,
            "theme": theme,
            "sector_theme": sector_theme,
            "final_score": round(float(sc.get("final_score", 0)), 1),
            "confidence_score": round(float(sc.get("confidence_score", 0)), 1),
            "pred_return_pct": round(float(ms.get("pred_return_60d", 0)) * 100, 1),
            "pred_mdd_pct": round(float(ms.get("pred_mdd_60d", 0)) * 100, 1),
            "rsi": round(float(ms.get("rsi_14", 0)), 1),
            "status": be.get("status", "-"),
            "support_reasons_kr": support,
            "block_reasons_kr": block,
        })
    # commentary는 위에서 채운 수치 기반이므로 별도 루프로 부여
    formal = picker.platform == "naver"
    for s in items:
        s["commentary"] = build_stock_commentary(s, picker, formal)
    return items


def regime_word_from_recommendations(data: dict, picker: VariationPicker) -> tuple[str, str]:
    """recommendations에서 raw regime과 한글 regime_word 반환."""
    regime = ""
    if data.get("items"):
        regime = data["items"][0].get("market_signals", {}).get("regime", "")
    return regime, picker.map_value("regime_word", regime)


def build_riser_commentary(r: dict, picker: VariationPicker, formal: bool) -> str:
    """순위 상승 종목을 한두 문장 산문으로 — 5가지 variant."""
    name = r["name"]
    nm = josa(name, "은", "는")
    e = "입니다" if formal else "이다"
    v = picker.choice_index(5, "riser_comment", r["code"])
    if v == 0:
        body = (f"{nm} 전일 {r['prev_rank']}위에서 오늘 {r['rank']}위로 {r['rank_gain']}계단 "
                + ("올라섰습니다." if formal else "올라섰다."))
        body += f" AI 종합점수는 {r['final_score']}점{e}."
    elif v == 1:
        body = (f"{r['prev_rank']}위였던 {nm} 하루 만에 {r['rank']}위까지 {r['rank_gain']}계단 "
                + ("뛰었습니다." if formal else "뛰었다."))
        body += f" 현재 AI 점수는 {r['final_score']}점{e}."
    elif v == 2:
        body = (f"오늘 {nm} {r['rank_gain']}계단 올라 {r['rank']}위를 "
                + ("기록했습니다." if formal else "기록했다."))
        body += (f" AI 점수 {r['final_score']}점으로 전일 대비 상승 모멘텀이 "
                 + ("확인됩니다." if formal else "확인된다."))
    elif v == 3:
        body = f"AI 랭킹 {r['prev_rank']}위에서 {r['rank']}위로, {r['rank_gain']}계단 급등{e}."
        body += (f" {r['final_score']}점의 AI 종합점수로 오늘 상위권에 "
                 + ("진입했습니다." if formal else "진입했다."))
    else:
        body = (f"전일 대비 {r['rank_gain']}계단 상승하며 {r['rank']}위에 "
                + ("오른 종목입니다." if formal else "오른 종목이다."))
        body += (f" AI 종합점수는 {r['final_score']}점으로 단기 상승 흐름이 "
                 + ("감지됩니다." if formal else "감지된다."))
    return body


def build_riser_commentary_5d(r: dict, picker: VariationPicker, formal: bool) -> str:
    """화면의 5영업일 전 비교 기준과 맞춘 TYPE_B 요약 문장."""
    name = r["name"]
    nm = josa(name, "는", "은")
    ending = "입니다" if formal else "이다"
    v = picker.choice_index(5, "riser_comment_5d", r["code"])
    if v == 0:
        body = (
            f"{nm} 5영업일 전 {r['prev_rank']}위에서 오늘 {r['rank']}위로 {r['rank_gain']}계단 "
            + ("올라섰습니다." if formal else "올라섰다.")
        )
        body += f" AI 종합점수는 {r['final_score']}점{ending}."
    elif v == 1:
        body = (
            f"{r['prev_rank']}위였던 {nm} 최근 5영업일 동안 {r['rank']}위까지 {r['rank_gain']}계단 "
            + ("뛰었습니다." if formal else "뛰었다.")
        )
        body += f" 현재 AI 점수는 {r['final_score']}점{ending}."
    elif v == 2:
        body = (
            f"오늘 {nm} {r['rank_gain']}계단 올라 {r['rank']}위를 "
            + ("기록했습니다." if formal else "기록했다.")
        )
        body += (
            f" AI 점수 {r['final_score']}점으로 최근 5영업일 기준 상승 모멘텀이 "
            + ("확인됩니다." if formal else "확인된다.")
        )
    elif v == 3:
        body = f"AI 랭킹 {r['prev_rank']}위에서 {r['rank']}위로, {r['rank_gain']}계단 급등{ending}."
        body += (
            f" {r['final_score']}점의 AI 종합점수로 오늘 상위권에 "
            + ("진입했습니다." if formal else "진입했다.")
        )
    else:
        body = (
            f"5영업일 전 대비 {r['rank_gain']}계단 상승하며 {r['rank']}위에 "
            + ("오른 종목입니다." if formal else "오른 종목이다.")
        )
        body += (
            f" AI 종합점수는 {r['final_score']}점으로 단기 상승 흐름이 "
            + ("감지됩니다." if formal else "감지된다.")
        )
    return body


def build_market_commentary_c(ms: dict, regime_word: str, picker: VariationPicker, formal: bool) -> str:
    """TYPE_C용 시장 지표 산문 해설."""
    s = []
    close, ma20 = ms.get("kospi_close"), ms.get("kospi_ma20")
    if close and ma20:
        above = close >= ma20
        rel = "위" if above else "아래"
        e = "있습니다" if formal else "있다"
        gap = (close - ma20) / ma20 * 100
        s.append(f"코스피는 {close:,.2f}로 20일 이동평균선({ma20:,.2f}) {rel}에 {e}.")
        s.append(f"이동평균선과의 이격은 약 {gap:+.1f}%" + ("입니다." if formal else "이다."))
        trend = "단기 흐름은 우호적인 편" if above else "단기 흐름은 다소 눌린 모습"
        s.append((trend + ("입니다." if formal else "이다.")))
    vol = ms.get("volatility_5d")
    if vol is not None:
        lvl = "낮은" if vol < 0.015 else ("보통" if vol < 0.025 else "높은")
        s.append(f"5일 변동성은 약 {vol*100:.1f}%로 {lvl} 수준" + ("입니다." if formal else "이다."))
    fnet = ms.get("foreign_net_5d")
    if fnet is not None:
        if fnet >= 0:
            s.append(f"최근 5일 외국인은 순매수({fnet:,.0f}) 흐름" + ("입니다." if formal else "이다."))
        else:
            s.append(f"최근 5일 외국인은 순매도({fnet:,.0f}) 흐름" + ("입니다." if formal else "이다."))
    s.append(f"종합하면 오늘은 '{regime_word}' 국면으로 해석" + ("됩니다." if formal else "된다."))
    return " ".join(s)


def build_context_b(rc: dict, source_date: str, picker: VariationPicker) -> dict:
    formal = picker.platform == "naver"
    risers = rc.get("risers", [])
    for r in risers:
        r["commentary"] = build_riser_commentary_5d(r, picker, formal)
    title_text = f"{korean_date(source_date)} 최근 5영업일 랭킹 급상승 종목 — 변화 분석"
    intro_text = "5영업일 전 대비 AI 랭킹이 크게 상승한 종목들을 정리했습니다. 화면의 랭킹 변화 기준과 같은 비교 구간을 사용합니다."
    riser_intro_text = "5영업일 전 대비 상승폭이 크고, 현재 50위 이내에 들어온 종목들입니다."
    ctx = {
        "source_date": source_date,
        "date_display": korean_date(source_date),
        "prev_display": korean_date(rc.get("prev", source_date)),
        "count": len(risers),
        "risers": risers,
        "skin": picker.choice_index(2, "section_skin"),
        "title_text": title_text,
        "intro_text": intro_text,
        "riser_intro_text": riser_intro_text,
    }
    ctx.update(picker.as_jinja_globals())
    return ctx


def _fmt_market(ms: dict) -> dict:
    """시장 지표를 표시용 문자열로 미리 포맷(템플릿 조건문 회피)."""
    close, ma20 = ms.get("kospi_close"), ms.get("kospi_ma20")
    vol, fnet = ms.get("volatility_5d"), ms.get("foreign_net_5d")
    return {
        "close": f"{close:,.2f}" if close else "-",
        "ma20": f"{ma20:,.2f}" if ma20 else "-",
        "vol": f"{vol * 100:.1f}%" if vol is not None else "-",
        "fnet": f"{fnet:,.0f}" if fnet is not None else "-",
    }


def build_context_c(ms: dict, regime: str, regime_word: str, source_date: str,
                    picker: VariationPicker) -> dict:
    formal = picker.platform == "naver"
    ctx = {
        "source_date": source_date,
        "date_display": korean_date(source_date),
        "regime": regime,
        "regime_word": regime_word,
        "mkt": _fmt_market(ms),
        "market_commentary_c": build_market_commentary_c(ms, regime_word, picker, formal),
        "regime_note": picker.pick_group("regime_note", regime_word),
        "faqs": picker.pick_n("faq_c", 2),
        "skin": picker.choice_index(2, "section_skin"),
    }
    ctx.update(picker.as_jinja_globals())
    return ctx


def build_context(data: dict, source_date: str, picker: VariationPicker, limit: int) -> dict:
    items = prepare_items(data, picker, limit)
    regime = ""
    if data.get("items"):
        regime = data["items"][0].get("market_signals", {}).get("regime", "")
    regime_reason = ""
    if data.get("items"):
        regime_reason = data["items"][0].get("market_signals", {}).get("regime_reason", "")

    gate = data.get("gate_overall_status", "")
    wf = data.get("walkforward_acceptance_status", "")
    mode = "recommend" if (gate == "BUY_ALLOWED" and wf == "ACCEPTED") else "watch"

    ctx = {
        "source_date": source_date,
        "date_display": korean_date(source_date),
        "regime": regime,
        "regime_word": picker.map_value("regime_word", regime),
        "regime_reason_human": parse_regime_reason(regime_reason),
        "market_commentary": build_market_commentary(
            picker.map_value("regime_word", regime),
            parse_regime_reason(regime_reason),
            picker,
            picker.platform == "naver",
        ),
        "count": len(items),
        # 섹션 순서 스킨: 0=관찰포인트 먼저, 1='점수 읽는 법' 먼저 (날짜별로 달라짐)
        "skin": picker.choice_index(2, "section_skin"),
        "mode": mode,
        "gate_overall_status": gate,
        "walkforward_acceptance_status": wf,
        "items": items,
    }
    ctx.update(picker.as_jinja_globals())
    return ctx


# ---------------------------------------------------------------------------
# 렌더링 / 출력
# ---------------------------------------------------------------------------
def render(content_type: str, platform: str, data: dict, source_date: str, limit: int) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    picker = VariationPicker(source_date, content_type, platform)
    if content_type == "A":
        ctx = build_context(data, source_date, picker, limit)
    elif content_type == "B":
        import blog_datasources as ds
        rc = ds.get_ranking_change(source_date, limit)
        ctx = build_context_b(rc, source_date, picker)
    elif content_type == "C":
        import blog_datasources as ds
        ms = ds.get_market_status(source_date)
        regime, regime_word = regime_word_from_recommendations(data, picker)
        ctx = build_context_c(ms, regime, regime_word, source_date, picker)
    else:
        raise ValueError(f"미지원 유형: {content_type}")
    template_name = f"type_{content_type.lower()}_{platform}.{PLATFORM_EXT[platform]}.j2"
    template = env.get_template(template_name)
    return template.render(**ctx)


def write_output(content_type: str, platform: str, source_date: str, text: str, force: bool) -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = PLATFORM_EXT[platform]
    out = OUTPUT_DIR / f"{source_date}_type{content_type.upper()}_{platform}.{ext}"
    if out.exists() and not force:
        log.info("이미 존재하여 건너뜀: %s (재생성하려면 --force)", out.name)
        return None
    out.write_text(text, encoding="utf-8")
    log.info("생성 완료: %s", out.name)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def generate_for_type(content_type: str, data: dict, source_date: str,
                      platforms: list[str], limit: int, force: bool) -> int:
    """한 유형(A/B/C)을 지정 플랫폼들에 대해 생성. 생성 파일 수 반환."""
    generated = 0
    for platform in platforms:
        text = render(content_type, platform, data, source_date, limit)
        if write_output(content_type, platform, source_date, text, force):
            generated += 1
    return generated


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(description="블로그 글 초안 생성")
    ap.add_argument("--type", default="A", choices=["A", "B", "C", "all"],
                    help="콘텐츠 유형 (all = A·B·C 일괄)")
    ap.add_argument("--platform", choices=["naver", "tistory"], help="미지정 시 둘 다 생성")
    ap.add_argument("--date", help="데이터 기준일 YYYY-MM-DD (미지정 시 데이터의 asof_date 사용)")
    ap.add_argument("--limit", type=int, default=5, help="종목 수 (기본 5)")
    ap.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    ap.add_argument("--strict", action="store_true", help="요청 날짜와 데이터 asof_date 불일치 시 중단")
    ap.add_argument(
        "--polish-ollama",
        action="store_true",
        help="초안 생성 후 로컬 Ollama로 본문 문단을 다듬은 최종본도 생성",
    )
    ap.add_argument(
        "--polish-model",
        default=os.environ.get("BLOG_POLISH_OLLAMA_MODEL", "qwen2.5:7b"),
        help="Ollama polishing 모델명",
    )
    ap.add_argument(
        "--keep-latest-only",
        action="store_true",
        help="생성 완료 후 outputs/blog_drafts, outputs/blog_polished에서 기준일 외 파일 삭제",
    )
    args = ap.parse_args()

    data = load_recommendations()
    asof = data.get("asof_date")
    source_status = data.get("source_status", "")
    source_date = args.date or asof
    if not source_date:
        log.error("기준일을 결정할 수 없습니다(asof_date 없음).")
        return 2

    # freshness 검증 (§10)
    if args.date and asof and args.date != asof:
        msg = f"요청 날짜({args.date})와 데이터 asof_date({asof})가 다릅니다."
        if args.strict:
            log.error("%s --strict 이므로 중단.", msg)
            return 3
        log.warning("%s 데이터의 asof_date 기준으로 진행합니다.", msg)
        source_date = asof
    if source_status and source_status != "current":
        log.warning("source_status=%s (current 아님). 데이터 신선도에 유의하세요.", source_status)

    platforms = [args.platform] if args.platform else ["naver", "tistory"]
    types = ["A", "B", "C"] if args.type == "all" else [args.type]

    total = 0
    failures: list[str] = []
    for ct in types:
        try:
            total += generate_for_type(ct, data, source_date, platforms, args.limit, args.force)
        except Exception as exc:  # noqa: BLE001 — 한 유형 실패가 다른 유형을 막지 않도록
            failures.append(ct)
            log.error("TYPE_%s 생성 실패(계속 진행): %s", ct, exc)

    log.info("요약: types=%s, source_date=%s, 생성=%d건, 실패=%s",
             ",".join(types), source_date, total, ",".join(failures) or "없음")
    polish_enabled = args.polish_ollama or str(os.environ.get("BLOG_POLISH_OLLAMA_ENABLED", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if polish_enabled:
        polish_drafts_with_ollama(source_date, types, platforms, args.force, args.polish_model)
    save_drafts_to_db(source_date, types, platforms)
    keep_latest_only = args.keep_latest_only or str(os.environ.get("BLOG_OUTPUT_KEEP_LATEST_ONLY", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if keep_latest_only:
        prune_blog_outputs(source_date)
        prune_blog_payloads_from_db(source_date)
    # all 모드에서 일부만 실패하면 0(성공), 전부 실패하면 1
    if failures and len(failures) == len(types):
        return 1
    return 0


def prune_blog_outputs(source_date: str) -> None:
    """기준일 외 블로그 산출물을 삭제한다.

    명시 옵션/환경변수로만 실행된다. 파일명 앞 10자리가 YYYY-MM-DD인 블로그 산출물만
    대상으로 삼아 다른 outputs 파일을 건드리지 않는다.
    """
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_type[A-Z]_.+")
    for directory in (OUTPUT_DIR, POLISHED_DIR):
        if not directory.exists():
            continue
        removed = 0
        for path in directory.iterdir():
            if not path.is_file() or not pattern.match(path.name):
                continue
            if path.name.startswith(source_date):
                continue
            try:
                path.unlink()
                removed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("오래된 블로그 산출물 삭제 실패(%s): %s", path, exc)
        if removed:
            log.info("오래된 블로그 산출물 삭제: %s (%d개)", directory, removed)


def prune_blog_payloads_from_db(source_date: str) -> None:
    """DB의 과거 블로그 payload 키를 삭제한다.

    --keep-latest-only 또는 BLOG_OUTPUT_KEEP_LATEST_ONLY=1일 때만 호출된다.
    """
    try:
        import psycopg2  # noqa: PLC0415
    except ImportError:
        log.warning("psycopg2 없음 — DB 블로그 payload 정리 건너뜀")
        return

    targets = _resolve_draft_db_targets()
    if not targets:
        return

    keep_keys = [f"blog_drafts_{source_date}", f"blog_polished_{source_date}"]
    for label, db_url in targets:
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM research.app_payload_store
                 WHERE (payload_key LIKE 'blog_drafts_%' OR payload_key LIKE 'blog_polished_%')
                   AND payload_key <> ALL(%s)
                """,
                [keep_keys],
            )
            removed = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            if removed:
                log.info("오래된 블로그 DB payload 삭제(%s): %d개", label, removed)
        except Exception as exc:  # noqa: BLE001
            log.warning("DB 블로그 payload 정리 실패(%s, 계속 진행): %s", label, exc)


def polish_drafts_with_ollama(
    source_date: str,
    types: list[str],
    platforms: list[str],
    force: bool,
    model: str,
) -> int:
    """생성된 초안을 로컬 Ollama 후처리기로 다듬어 outputs/blog_polished에 저장한다."""
    try:
        import polish_blog_drafts_ollama as polisher  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.warning("Ollama 후처리 모듈 로드 실패 — 최종본 생성 건너뜀: %s", exc)
        return 0

    generated = 0
    for ct in types:
        for platform in platforms:
            ext = PLATFORM_EXT[platform]
            input_path = OUTPUT_DIR / f"{source_date}_type{ct.upper()}_{platform}.{ext}"
            output_path = POLISHED_DIR / f"{source_date}_type{ct.upper()}_{platform}_ollama_polished.{ext}"
            if not input_path.exists():
                continue
            if output_path.exists() and not force:
                log.info("Ollama 최종본 이미 존재 — 건너뜀: %s", output_path)
                continue
            try:
                ns = SimpleNamespace(
                    input=str(input_path),
                    output=str(output_path),
                    date=source_date,
                    type=ct.upper(),
                    platform=platform,
                    model=model,
                    ollama_url=os.environ.get("BLOG_POLISH_OLLAMA_URL", "http://localhost:11434"),
                    temperature=float(os.environ.get("BLOG_POLISH_TEMPERATURE", "0.55")),
                    timeout=int(os.environ.get("BLOG_POLISH_TIMEOUT_SEC", "120")),
                    style=os.environ.get(
                        "BLOG_POLISH_STYLE",
                        "템플릿 문장처럼 보이지 않게, 설명은 조금 더 부드럽고 구체적으로 쓴다.",
                    ),
                    max_paragraphs=0,
                    force=force,
                )
                text = input_path.read_text(encoding="utf-8")
                polished, results = polisher.polish_text(text, ns)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(polished, encoding="utf-8")
                report_path = polisher.write_report(output_path, results)
                changed = sum(1 for r in results if r.changed)
                fallback = sum(1 for r in results if not r.changed)
                generated += 1
                log.info(
                    "Ollama 최종본 생성: %s (changed=%d fallback=%d report=%s)",
                    output_path,
                    changed,
                    fallback,
                    report_path,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Ollama 최종본 생성 실패(%s/%s, 계속 진행): %s", ct, platform, exc)
    return generated


def _resolve_draft_db_targets() -> list[tuple[str, str]]:
    """초안 저장 대상 DB 목록을 반환한다.

    - local/runtime DB: DATABASE_URL
    - web DB: WEB_DATABASE_URL
    같은 URL이면 중복 저장하지 않는다.
    """
    seen: set[str] = set()
    targets: list[tuple[str, str]] = []
    for label, env_name in (("local", "DATABASE_URL"), ("web", "WEB_DATABASE_URL")):
        url = str(os.environ.get(env_name, "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        targets.append((label, url))
    return targets


def _read_blog_payload_item(path: Path) -> dict | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    title = next((l.lstrip("# ").strip() for l in lines if l.strip()), "")
    return {"title": title, "char_count": len(content), "content": content}


def _polished_path(source_date: str, content_type: str, platform: str) -> Path:
    ext = PLATFORM_EXT[platform]
    return POLISHED_DIR / f"{source_date}_type{content_type.upper()}_{platform}_ollama_polished.{ext}"


def save_drafts_to_db(source_date: str, types: list[str], platforms: list[str]) -> None:
    """생성된 초안을 research.app_payload_store에 upsert (실패 시 로그만 출력)."""
    try:
        import psycopg2  # noqa: PLC0415
    except ImportError:
        log.warning("psycopg2 없음 — DB 저장 건너뜀")
        return
    targets = _resolve_draft_db_targets()
    if not targets:
        log.warning("DATABASE_URL/WEB_DATABASE_URL 없음 — DB 저장 건너뜀")
        return

    drafts = []
    polished_drafts = []
    for ct in types:
        entry: dict = {"type": ct}
        polished_entry: dict = {"type": ct}
        for platform in platforms:
            ext = PLATFORM_EXT[platform]
            fp = OUTPUT_DIR / f"{source_date}_type{ct.upper()}_{platform}.{ext}"
            original_item = _read_blog_payload_item(fp)
            if original_item:
                final_item = _read_blog_payload_item(_polished_path(source_date, ct, platform))
                if final_item:
                    original_item["polished"] = final_item
                    polished_entry[platform] = final_item
                entry[platform] = original_item
        if len(entry) > 1:
            drafts.append(entry)
        if len(polished_entry) > 1:
            polished_drafts.append(polished_entry)

    if not drafts:
        return

    payload_key = f"blog_drafts_{source_date}"
    payload = {"asof_date": source_date, "drafts": drafts}
    polished_payload_key = f"blog_polished_{source_date}"
    polished_payload = {"asof_date": source_date, "drafts": polished_drafts, "source": payload_key}
    payload_json = json.dumps(payload, ensure_ascii=False)
    polished_payload_json = json.dumps(polished_payload, ensure_ascii=False)
    for label, db_url in targets:
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO research.app_payload_store
                  (payload_key, payload_json, asof_date, generated_at, updated_at)
                VALUES (%s, %s::jsonb, %s, now(), now())
                ON CONFLICT (payload_key) DO UPDATE
                SET payload_json = EXCLUDED.payload_json, updated_at = now()
                """,
                [payload_key, payload_json, source_date],
            )
            if polished_drafts:
                cur.execute(
                    """
                    INSERT INTO research.app_payload_store
                      (payload_key, payload_json, asof_date, generated_at, updated_at)
                    VALUES (%s, %s::jsonb, %s, now(), now())
                    ON CONFLICT (payload_key) DO UPDATE
                    SET payload_json = EXCLUDED.payload_json, updated_at = now()
                    """,
                    [polished_payload_key, polished_payload_json, source_date],
                )
            conn.commit()
            cur.close()
            conn.close()
            log.info("DB 저장 완료(%s): %s (%d건)", label, payload_key, len(drafts))
            if polished_drafts:
                log.info("DB 저장 완료(%s): %s (%d건)", label, polished_payload_key, len(polished_drafts))
        except Exception as exc:  # noqa: BLE001
            log.warning("DB 저장 실패(%s, 계속 진행): %s", label, exc)


if __name__ == "__main__":
    raise SystemExit(main())
