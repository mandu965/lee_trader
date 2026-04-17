from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
REQUIRED_FILES = [
    DATA_DIR / "top20_before_after_compare_v3.csv",
    DATA_DIR / "theme_overlay_acceptance_summary.md",
    DATA_DIR / "no_theme_displacement_report.md",
    DATA_DIR / "theme_concentration_report.csv",
    DATA_DIR / "near_top20_theme_lift_report.csv",
]
CHECK_MD = DATA_DIR / "theme_overlay_validation_check.md"

LOGGER = logging.getLogger("check_theme_overlay_validation")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_check_markdown() -> str:
    rows = []
    for path in REQUIRED_FILES:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
        )
    status_df = pd.DataFrame(rows)
    missing = status_df.loc[~status_df["exists"], "path"].tolist()

    lines = [
        "# Theme Overlay Validation Check",
        "",
        "## Required Files",
    ]
    for row in status_df.itertuples(index=False):
        lines.append(f"- {row.path}: exists={row.exists}, size={int(row.size)}")
    lines.extend([
        "",
        "## Result",
        "- PASS" if not missing else f"- FAIL: missing={missing}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    text = build_check_markdown()
    CHECK_MD.write_text(text, encoding="utf-8")
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required validation files: {missing}")
    LOGGER.info("Saved %s", CHECK_MD.resolve())
    print(f"generated_files={[str(CHECK_MD), *[str(path) for path in REQUIRED_FILES]]}")


if __name__ == "__main__":
    main()
