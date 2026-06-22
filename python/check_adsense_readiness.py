from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "node" / "public"
CONTENT_DIRS = [ROOT / "node" / "content" / "blog", ROOT / "node" / "content" / "reports"]
MIN_INDEXABLE_WORDS = 500
MIN_INDEXABLE_CONTENT_FILES = 10

REQUIRED_PUBLIC_FILES = [
    "landing.html",
    "about.html",
    "editorial-policy.html",
    "contact.html",
    "privacy.html",
    "terms.html",
    "disclaimer.html",
    "methodology.html",
    "glossary.html",
    "robots.txt",
    "ads.txt",
]

MOJIBAKE_PATTERNS = re.compile(r"(?:먯|섏|쒖|꾨|湲|醫|媛|釉|吏|蹂|쨌|룻|퀎)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") and "\n---\n" in text[4:]


def frontmatter_value(text: str, key: str) -> str | None:
    if not has_frontmatter(text):
        return None
    header = text.split("\n---\n", 1)[0][4:]
    match = re.search(rf"(?mi)^{re.escape(key)}:\s*(.+?)\s*$", header)
    return match.group(1).strip().strip("\"'") if match else None


def content_body(text: str) -> str:
    return text.split("\n---\n", 1)[1] if has_frontmatter(text) else text


def main() -> int:
    problems: list[str] = []

    for filename in REQUIRED_PUBLIC_FILES:
      path = PUBLIC_DIR / filename
      if not path.exists():
          problems.append(f"missing public file: {path.relative_to(ROOT)}")
          continue
      if filename.endswith(".html"):
          text = read(path)
          if "<title>" not in text.lower():
              problems.append(f"missing title: {path.relative_to(ROOT)}")
          if 'name="description"' not in text.lower():
              problems.append(f"missing meta description: {path.relative_to(ROOT)}")

    content_files = [path for folder in CONTENT_DIRS for path in folder.glob("*.md")]
    if len(content_files) < 20:
        problems.append(f"too few public content files: {len(content_files)}")

    indexable_content_count = 0
    for path in content_files:
        text = read(path)
        if not has_frontmatter(text):
            problems.append(f"missing frontmatter: {path.relative_to(ROOT)}")
        suspicious = len(MOJIBAKE_PATTERNS.findall(text))
        replacement = text.count("�")
        if suspicious > 12 or replacement:
            problems.append(f"possible mojibake: {path.relative_to(ROOT)}")
        indexable = str(frontmatter_value(text, "indexable") or "true").lower() != "false"
        word_count = len(re.findall(r"\S+", content_body(text)))
        if indexable:
            indexable_content_count += 1
        if indexable and word_count < MIN_INDEXABLE_WORDS:
            problems.append(
                f"thin indexable content ({word_count} words < {MIN_INDEXABLE_WORDS}): "
                f"{path.relative_to(ROOT)}"
            )

    if indexable_content_count < MIN_INDEXABLE_CONTENT_FILES:
        problems.append(
            f"too few indexable content files: {indexable_content_count} "
            f"< {MIN_INDEXABLE_CONTENT_FILES}"
        )

    ads_txt = read(PUBLIC_DIR / "ads.txt") if (PUBLIC_DIR / "ads.txt").exists() else ""
    if "google.com, pub-" not in ads_txt:
        problems.append("ads.txt does not contain a Google publisher line")

    robots = read(PUBLIC_DIR / "robots.txt") if (PUBLIC_DIR / "robots.txt").exists() else ""
    if "Sitemap:" not in robots:
        problems.append("robots.txt does not advertise sitemap")

    if problems:
        print("AdSense readiness check: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("AdSense readiness check: PASS")
    print(f"- content files: {len(content_files)}")
    print(f"- indexable content files: {indexable_content_count}")
    print(f"- required public files: {len(REQUIRED_PUBLIC_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
