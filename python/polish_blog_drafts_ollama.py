"""
Polish generated blog drafts with a local Ollama model.

This script keeps structural/data-heavy lines intact and only asks the model to
rewrite normal prose paragraphs. If a rewritten paragraph drops required tokens
or contains risky investment wording, the original paragraph is kept.

Examples:
    python python/polish_blog_drafts_ollama.py --input outputs/blog_drafts/2026-06-02_typeA_tistory.md --force
    python python/polish_blog_drafts_ollama.py --date 2026-06-02 --type A --platform tistory --force
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DRAFT_DIR = ROOT / "outputs" / "blog_drafts"
POLISHED_DIR = ROOT / "outputs" / "blog_polished"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

PLATFORM_EXT = {"naver": "txt", "tistory": "md"}

FORBIDDEN_PHRASES = [
    "매수 추천",
    "매수 기회",
    "강력 추천",
    "사야 한다",
    "사야 합니다",
    "급등 가능",
    "급등할",
    "수익 보장",
    "확실한 수익",
    "확실하다",
    "지난 60일",
    "최근 60일",
    "평균",
    "증가",
    "높았으나",
    "였음을",
]

TOKEN_RE = re.compile(
    r"""
    [+-]?\d+(?:,\d{3})*(?:\.\d+)?%?
    |[A-Z]{2,}(?:_[A-Z]+)?
    |LightGBM
    |WATCH
    |REJECTED
    |PILOT
    |BUY_ALLOWED
    |BLOCK
    """,
    re.VERBOSE,
)

INFORMAL_ENDING_RE = re.compile(r"(이다|한다|된다|있다|없다|보인다|달한다|나타낸다|의미한다)\.")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class ParagraphResult:
    index: int
    original: str
    polished: str
    changed: bool
    status: str


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input).resolve()
    if not args.date or not args.type or not args.platform:
        raise SystemExit("--input 또는 --date/--type/--platform 조합이 필요합니다.")
    ext = PLATFORM_EXT[args.platform]
    return (DRAFT_DIR / f"{args.date}_type{args.type.upper()}_{args.platform}.{ext}").resolve()


def resolve_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).resolve()
    suffix = "_ollama_polished"
    return (POLISHED_DIR / f"{input_path.stem}{suffix}{input_path.suffix}").resolve()


def is_protected_line(line: str, *, in_frontmatter: bool) -> bool:
    stripped = line.strip()
    if in_frontmatter:
        return True
    if not stripped:
        return False
    if stripped == "---":
        return True
    if stripped.startswith(("#", ">", "|", "-", "▶")):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if "면책 고지" in stripped:
        return True
    if "https://" in stripped or "http://" in stripped:
        return True
    return False


def iter_mutable_paragraphs(lines: list[str]) -> Iterable[tuple[int, int, str]]:
    """Yield (start_line, end_line_exclusive, paragraph_text)."""
    in_frontmatter = False
    if lines and lines[0].strip() == "---":
        in_frontmatter = True

    start: int | None = None
    buf: list[str] = []

    def flush(end: int):
        nonlocal start, buf
        if start is not None and buf:
            paragraph = "\n".join(buf).strip()
            if paragraph:
                yield (start, end, paragraph)
        start = None
        buf = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx > 0 and in_frontmatter and stripped == "---":
            in_frontmatter = False
            yield from flush(idx)
            continue

        protected = is_protected_line(line, in_frontmatter=in_frontmatter)
        if protected or not stripped:
            yield from flush(idx)
            continue

        if start is None:
            start = idx
        buf.append(line)

    yield from flush(len(lines))


def extract_required_tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text))


def validate_polished(original: str, polished: str) -> tuple[bool, str]:
    normalized = polished.strip()
    if not normalized:
        return False, "empty"
    if "\n\n" in normalized:
        return False, "multi_paragraph"
    missing = sorted(tok for tok in extract_required_tokens(original) if tok not in normalized)
    if missing:
        return False, "missing_tokens:" + ",".join(missing)
    bad = sorted(phrase for phrase in FORBIDDEN_PHRASES if phrase in normalized)
    if bad:
        return False, "forbidden:" + ",".join(bad)
    if CHINESE_CHAR_RE.search(normalized):
        return False, "unexpected_chinese_chars"
    if not re.search(r"[.!?。…]$", normalized):
        return False, "incomplete_sentence"
    extra_upper = sorted(
        tok
        for tok in extract_required_tokens(normalized)
        if re.fullmatch(r"[A-Z]{2,}(?:_[A-Z]+)?", tok) and tok not in extract_required_tokens(original)
    )
    if extra_upper:
        return False, "extra_upper_tokens:" + ",".join(extra_upper)
    if INFORMAL_ENDING_RE.search(normalized):
        return False, "informal_ending"
    if len(normalized) > max(120, int(len(original) * 2.4)):
        return False, "too_long"
    return True, "ok"


def build_prompt(paragraph: str, *, platform: str, style: str) -> str:
    platform_hint = "티스토리 마크다운 글" if platform == "tistory" else "네이버 블로그 일반 텍스트 글"
    return f"""
너는 한국어 금융 블로그 편집자다.
아래 한 문단만 {platform_hint}에 맞게 더 자연스럽고 덜 템플릿처럼 다시 써라.

스타일:
- {style}
- 존댓말을 기본으로 한다.
- 모든 문장은 '~입니다', '~합니다', '~보입니다', '~필요가 있습니다'처럼 존댓말로 끝낸다.
- 데이터 기반 관찰 글의 차분한 톤을 유지한다.
- 문장 구조를 원문과 다르게 만들어도 된다.
- 한국어만 사용한다. 중국어, 일본어, 불필요한 영어 단어를 섞지 않는다.

절대 변경 금지:
- 날짜, 종목명, 시장명, 모델명, 상태값
- 모든 숫자, 점수, 퍼센트, RSI, +/− 부호
- WATCH, REJECTED, PILOT, LightGBM 같은 영문 토큰
- 의미상 매수 추천이나 투자 권유처럼 보이면 안 된다.

금지 표현:
매수 추천, 매수 기회, 강력 추천, 사야 한다, 급등 가능, 수익 보장, 확실한 수익

출력:
- 설명 없이 다듬은 한 문단만 출력한다.
- 표, 제목, bullet은 만들지 않는다.

원문 문단:
{paragraph}
""".strip()


def call_ollama(prompt: str, *, base_url: str, model: str, temperature: float, timeout: int) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ollama_http_{exc.code}:{detail}") from exc
    return str(data.get("response", "")).strip()


def polish_text(text: str, args: argparse.Namespace) -> tuple[str, list[ParagraphResult]]:
    lines = text.splitlines()
    replacements: dict[tuple[int, int], str] = {}
    results: list[ParagraphResult] = []
    paragraphs = list(iter_mutable_paragraphs(lines))

    if args.max_paragraphs:
        paragraphs = paragraphs[: args.max_paragraphs]

    for idx, (start, end, paragraph) in enumerate(paragraphs, start=1):
        prompt = build_prompt(paragraph, platform=args.platform or "tistory", style=args.style)
        try:
            candidate = call_ollama(
                prompt,
                base_url=args.ollama_url,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except Exception as exc:
            results.append(ParagraphResult(idx, paragraph, paragraph, False, f"ollama_error:{exc}"))
            continue

        ok, status = validate_polished(paragraph, candidate)
        if ok and candidate.strip() != paragraph.strip():
            replacements[(start, end)] = candidate.strip()
            results.append(ParagraphResult(idx, paragraph, candidate.strip(), True, status))
        else:
            results.append(ParagraphResult(idx, paragraph, paragraph, False, status))

    out_lines: list[str] = []
    idx = 0
    while idx < len(lines):
        match = next((((s, e), v) for (s, e), v in replacements.items() if s == idx), None)
        if match:
            (start, end), value = match
            out_lines.extend(value.splitlines())
            idx = end
        else:
            out_lines.append(lines[idx])
            idx += 1

    return "\n".join(out_lines).rstrip() + "\n", results


def write_report(output_path: Path, results: list[ParagraphResult]) -> Path:
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    payload = {
        "output": str(output_path),
        "paragraphs": [
            {
                "index": r.index,
                "changed": r.changed,
                "status": r.status,
                "original_len": len(r.original),
                "polished_len": len(r.polished),
            }
            for r in results
        ],
        "changed_count": sum(1 for r in results if r.changed),
        "fallback_count": sum(1 for r in results if not r.changed),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polish blog draft prose with local Ollama.")
    parser.add_argument("--input", help="초안 파일 경로")
    parser.add_argument("--output", help="결과 파일 경로")
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--type", choices=["A", "B", "C", "a", "b", "c"], help="블로그 유형")
    parser.add_argument("--platform", choices=["naver", "tistory"], default="tistory")
    parser.add_argument("--model", default=os.getenv("BLOG_POLISH_OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.getenv("BLOG_POLISH_OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("BLOG_POLISH_TEMPERATURE", "0.55")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("BLOG_POLISH_TIMEOUT_SEC", "120")))
    parser.add_argument("--style", default="템플릿 문장처럼 보이지 않게, 설명은 조금 더 부드럽고 구체적으로 쓴다.")
    parser.add_argument("--max-paragraphs", type=int, default=0, help="테스트용: 앞 N개 문단만 처리")
    parser.add_argument("--force", action="store_true", help="기존 결과 파일 덮어쓰기")
    return parser.parse_args()


def main() -> int:
    _load_env()
    args = parse_args()
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(input_path, args)

    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")
    if output_path.exists() and not args.force:
        raise SystemExit(f"결과 파일이 이미 있습니다. 덮어쓰려면 --force 사용: {output_path}")

    text = input_path.read_text(encoding="utf-8")
    polished, results = polish_text(text, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(polished, encoding="utf-8")
    report_path = write_report(output_path, results)

    changed = sum(1 for r in results if r.changed)
    fallback = sum(1 for r in results if not r.changed)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"report={report_path}")
    print(f"paragraphs={len(results)} changed={changed} fallback={fallback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
