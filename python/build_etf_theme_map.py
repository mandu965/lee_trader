import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import MetaData, Table

from db import get_engine


LOGGER = logging.getLogger("build_etf_theme_map")
DEFAULT_OVERRIDE_PATH = Path("data") / "etf_theme_overrides.json"
DEFAULT_AS_OF_DATE = date.today()
DEFAULT_MAPPING_WEIGHT = 1.0


@dataclass(frozen=True)
class ThemeRule:
    theme_code: str
    theme_name: str
    keywords: tuple[str, ...]
    theme_group: str = "thematic"
    description: str = ""
    display_order: int = 0


THEME_RULES: tuple[ThemeRule, ...] = (
    ThemeRule(
        theme_code="semiconductor",
        theme_name="\ubc18\ub3c4\uccb4",
        keywords=("\ubc18\ub3c4\uccb4", "semiconductor", "hbm"),
        description="Semiconductor and memory supply-chain theme.",
        display_order=10,
    ),
    ThemeRule(
        theme_code="secondary_battery",
        theme_name="2\ucc28\uc804\uc9c0",
        keywords=("2\ucc28\uc804\uc9c0", "\ubc30\ud130\ub9ac"),
        description="Rechargeable battery and EV battery theme.",
        display_order=20,
    ),
    ThemeRule(
        theme_code="defense",
        theme_name="\ubc29\uc0b0",
        keywords=("\ubc29\uc0b0", "defense"),
        description="Defense and aerospace manufacturing theme.",
        display_order=30,
    ),
    ThemeRule(
        theme_code="power_grid",
        theme_name="\uc804\ub825",
        keywords=("\uc804\ub825", "\uc804\uc120", "grid"),
        description="Power infrastructure and grid theme.",
        display_order=40,
    ),
    ThemeRule(
        theme_code="shipbuilding",
        theme_name="\uc870\uc120",
        keywords=("\uc870\uc120", "ship"),
        description="Shipbuilding and marine engineering theme.",
        display_order=50,
    ),
    ThemeRule(
        theme_code="nuclear",
        theme_name="\uc6d0\uc804",
        keywords=("\uc6d0\uc804", "nuclear"),
        description="Nuclear generation and equipment theme.",
        display_order=60,
    ),
    ThemeRule(
        theme_code="bio_healthcare",
        theme_name="\ubc14\uc774\uc624",
        keywords=("\ubc14\uc774\uc624", "\ud5ec\uc2a4\ucf00\uc5b4"),
        description="Biotech and healthcare theme.",
        display_order=70,
    ),
    ThemeRule(
        theme_code="robotics",
        theme_name="\ub85c\ubd07",
        keywords=("\ub85c\ubd07", "robot"),
        description="Robotics and factory automation theme.",
        display_order=80,
    ),
    ThemeRule(
        theme_code="ai",
        theme_name="AI",
        keywords=("ai", "\uc778\uacf5\uc9c0\ub2a5"),
        description="Artificial intelligence theme.",
        display_order=90,
    ),
    ThemeRule(
        theme_code="brokerage_markets",
        theme_name="Brokerage Markets",
        keywords=("증권", "brokerage", "capital markets"),
        description="Brokerage, wealth management, and capital-markets earnings sensitivity theme.",
        display_order=100,
    ),
    ThemeRule(
        theme_code="retail_finance_return",
        theme_name="Retail Finance Return",
        keywords=("은행", "bank", "dividend", "return", "finance"),
        description="Bank earnings durability plus dividend and capital-return theme.",
        display_order=110,
    ),
    ThemeRule(
        theme_code="digital_finance_platform",
        theme_name="Digital Finance Platform",
        keywords=("fintech", "finance", "platform", "digital"),
        description="Mobile banking, payments, wallet activity, and platform-based financial distribution theme.",
        display_order=120,
    ),
    ThemeRule(
        theme_code="digital_platform_ecosystem",
        theme_name="Digital Platform Ecosystem",
        keywords=("platform", "internet", "commerce", "content"),
        description="Traffic, advertising, commerce, and content monetization theme.",
        display_order=130,
    ),
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETF to theme mapping from ETF names.")
    parser.add_argument(
        "--as-of-date",
        default=DEFAULT_AS_OF_DATE.isoformat(),
        help="Validity start date for etf_theme_map rows (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--override-file",
        default=str(DEFAULT_OVERRIDE_PATH),
        help="Optional JSON file for manual override rules.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to the database. Print summary and sample rows only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for the number of ETF master rows to inspect.",
    )
    return parser.parse_args()


def parse_as_of_date(raw_value: str) -> date:
    return datetime.strptime(raw_value, "%Y-%m-%d").date()


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def is_explicit_keyword_match(name: str, keyword: str) -> bool:
    lower_name = name.lower()
    lower_keyword = keyword.lower()
    if lower_keyword in lower_name:
        return True
    compact_name = re.sub(r"[\s_\-/]+", "", lower_name)
    compact_keyword = re.sub(r"[\s_\-/]+", "", lower_keyword)
    return bool(compact_keyword) and compact_keyword in compact_name


def get_table(name: str) -> Table:
    metadata = MetaData()
    return Table(name, metadata, autoload_with=get_engine())


def ensure_theme_master_table() -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS theme_master (
                    theme_code TEXT PRIMARY KEY,
                    theme_name TEXT NOT NULL,
                    theme_group TEXT NULL,
                    theme_description TEXT NULL,
                    display_order INTEGER NOT NULL DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS etf_theme_map (
                    etf_code TEXT NOT NULL,
                    theme_code TEXT NOT NULL,
                    mapping_source TEXT NOT NULL,
                    mapping_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    is_primary BOOLEAN NOT NULL DEFAULT false,
                    valid_from DATE NOT NULL,
                    valid_to DATE NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (etf_code, theme_code, valid_from)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_etf_theme_map_theme_code
                ON etf_theme_map (theme_code)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_etf_theme_map_etf_code
                ON etf_theme_map (etf_code)
                """
            )
        )


def load_etf_master(limit: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT etf_code, etf_name, issuer_name, asset_class, market, is_active
        FROM etf_master
        ORDER BY etf_code
    """
    params: dict[str, Any] = {}
    if limit is not None and limit > 0:
        sql += " LIMIT :limit"
        params["limit"] = limit

    with get_engine().begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(row) for row in rows]


def load_override_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        LOGGER.info("Override file not found -> skip manual override path=%s", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("override file must contain a JSON object")
    LOGGER.info("Loaded manual override config path=%s keys=%s", path, len(payload))
    return payload


def ensure_theme_master(theme_rules: tuple[ThemeRule, ...], dry_run: bool) -> int:
    rows = [
        {
            "theme_code": rule.theme_code,
            "theme_name": rule.theme_name,
            "theme_group": rule.theme_group,
            "theme_description": rule.description,
            "display_order": rule.display_order,
            "is_active": True,
        }
        for rule in theme_rules
    ]
    if dry_run:
        LOGGER.info("Dry-run -> skip theme_master upsert rows=%s", len(rows))
        return len(rows)

    ensure_theme_master_table()
    theme_master = get_table("theme_master")
    stmt = insert(theme_master).values(rows)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["theme_code"],
        set_={
            "theme_name": stmt.excluded.theme_name,
            "theme_group": stmt.excluded.theme_group,
            "theme_description": stmt.excluded.theme_description,
            "display_order": stmt.excluded.display_order,
            "is_active": stmt.excluded.is_active,
            "updated_at": text("now()"),
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def calculate_confidence(match_type: str) -> float:
    if match_type == "explicit":
        return 0.9
    return 0.6


def rule_match_etf(etf_name: str) -> list[dict[str, Any]]:
    normalized_name = normalize_name(etf_name)
    normalized_lower = normalized_name.lower()
    matches: list[dict[str, Any]] = []

    for rule in THEME_RULES:
        matched_keywords: list[str] = []
        match_type = "partial"
        for keyword in rule.keywords:
            keyword_text = keyword.strip()
            if not keyword_text:
                continue
            if keyword_text.lower() in normalized_lower:
                matched_keywords.append(keyword_text.lower())
                if is_explicit_keyword_match(normalized_name, keyword_text):
                    match_type = "explicit"
        if not matched_keywords:
            continue
        matches.append(
            {
                "theme_code": rule.theme_code,
                "theme_name": rule.theme_name,
                "matched_keywords": matched_keywords,
                "match_type": match_type,
                "mapping_weight": DEFAULT_MAPPING_WEIGHT,
                "mapping_confidence": calculate_confidence(match_type),
            }
        )

    matches.sort(key=lambda item: (-float(item["mapping_confidence"]), str(item["theme_code"])))
    for idx, item in enumerate(matches):
        item["is_primary"] = idx == 0
        item["mapping_source"] = "rule"
    return matches


def normalize_override_entry(raw_entry: Any) -> dict[str, Any]:
    if raw_entry is None:
        return {}
    if isinstance(raw_entry, list):
        return {"replace": [str(item) for item in raw_entry]}
    if isinstance(raw_entry, dict):
        normalized: dict[str, Any] = {}
        for key in ("replace", "add", "remove"):
            value = raw_entry.get(key)
            if value is None:
                continue
            if not isinstance(value, list):
                raise ValueError(f"override field '{key}' must be a list")
            normalized[key] = [str(item) for item in value]
        confidence = raw_entry.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, dict):
                raise ValueError("override field 'confidence' must be an object")
            normalized["confidence"] = {str(k): float(v) for k, v in confidence.items()}
        primary_theme_code = raw_entry.get("primary_theme_code")
        if primary_theme_code is not None:
            normalized["primary_theme_code"] = str(primary_theme_code)
        return normalized
    raise ValueError("override entry must be either a list or an object")


def apply_override(
    *,
    etf_code: str,
    base_matches: list[dict[str, Any]],
    override_entry: dict[str, Any],
    rule_lookup: dict[str, ThemeRule],
) -> list[dict[str, Any]]:
    normalized_override = normalize_override_entry(override_entry)
    if not normalized_override:
        return base_matches

    by_theme_code: dict[str, dict[str, Any]] = {
        item["theme_code"]: dict(item)
        for item in base_matches
    }

    replace = normalized_override.get("replace")
    if replace:
        by_theme_code = {}
        for theme_code in replace:
            if theme_code not in rule_lookup:
                LOGGER.warning("Unknown theme_code in replace override etf_code=%s theme_code=%s", etf_code, theme_code)
                continue
            by_theme_code[theme_code] = {
                "theme_code": theme_code,
                "theme_name": rule_lookup[theme_code].theme_name,
                "matched_keywords": ["manual_override"],
                "match_type": "manual_override",
                "mapping_weight": DEFAULT_MAPPING_WEIGHT,
                "mapping_confidence": 0.99,
                "mapping_source": "manual_override",
                "is_primary": False,
            }

    for theme_code in normalized_override.get("add", []):
        if theme_code not in rule_lookup:
            LOGGER.warning("Unknown theme_code in add override etf_code=%s theme_code=%s", etf_code, theme_code)
            continue
        existing = by_theme_code.get(theme_code, {})
        by_theme_code[theme_code] = {
            "theme_code": theme_code,
            "theme_name": rule_lookup[theme_code].theme_name,
            "matched_keywords": existing.get("matched_keywords", ["manual_override"]),
            "match_type": "manual_override",
            "mapping_weight": DEFAULT_MAPPING_WEIGHT,
            "mapping_confidence": max(float(existing.get("mapping_confidence", 0.0)), 0.95),
            "mapping_source": "manual_override",
            "is_primary": False,
        }

    for theme_code in normalized_override.get("remove", []):
        by_theme_code.pop(theme_code, None)

    for theme_code, confidence in normalized_override.get("confidence", {}).items():
        if theme_code not in by_theme_code:
            if theme_code not in rule_lookup:
                LOGGER.warning("Unknown theme_code in confidence override etf_code=%s theme_code=%s", etf_code, theme_code)
                continue
            by_theme_code[theme_code] = {
                "theme_code": theme_code,
                "theme_name": rule_lookup[theme_code].theme_name,
                "matched_keywords": ["manual_override"],
                "match_type": "manual_override",
                "mapping_weight": DEFAULT_MAPPING_WEIGHT,
                "mapping_confidence": float(confidence),
                "mapping_source": "manual_override",
                "is_primary": False,
            }
        else:
            by_theme_code[theme_code]["mapping_confidence"] = float(confidence)
            by_theme_code[theme_code]["mapping_source"] = "manual_override"

    items = sorted(
        by_theme_code.values(),
        key=lambda item: (-float(item["mapping_confidence"]), str(item["theme_code"])),
    )
    primary_theme_code = normalized_override.get("primary_theme_code")
    for item in items:
        item["is_primary"] = False
    if primary_theme_code and primary_theme_code in by_theme_code:
        by_theme_code[primary_theme_code]["is_primary"] = True
    elif items:
        items[0]["is_primary"] = True
    return items


def build_mapping_rows(
    etf_rows: list[dict[str, Any]],
    *,
    as_of_date: date,
    override_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule_lookup = {rule.theme_code: rule for rule in THEME_RULES}
    mapping_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for row in etf_rows:
        etf_code = str(row["etf_code"])
        etf_name = normalize_name(row.get("etf_name", ""))
        if not etf_name:
            unmatched.append({"etf_code": etf_code, "etf_name": etf_name, "reason": "empty_name"})
            continue

        matches = rule_match_etf(etf_name)
        override_entry = override_config.get(etf_code) or override_config.get(etf_name)
        if override_entry is not None:
            matches = apply_override(
                etf_code=etf_code,
                base_matches=matches,
                override_entry=override_entry,
                rule_lookup=rule_lookup,
            )

        if not matches:
            unmatched.append({"etf_code": etf_code, "etf_name": etf_name, "reason": "no_rule_match"})
            continue

        for item in matches:
            mapping_rows.append(
                {
                    "etf_code": etf_code,
                    "theme_code": item["theme_code"],
                    "mapping_source": item["mapping_source"],
                    "mapping_confidence": float(item["mapping_confidence"]),
                    "is_primary": bool(item["is_primary"]),
                    "valid_from": as_of_date,
                    "valid_to": None,
                }
            )

    return mapping_rows, unmatched


def upsert_etf_theme_map(rows: list[dict[str, Any]], dry_run: bool) -> int:
    if not rows:
        return 0
    if dry_run:
        LOGGER.info("Dry-run -> skip etf_theme_map upsert rows=%s", len(rows))
        return len(rows)

    ensure_theme_master_table()
    etf_theme_map = get_table("etf_theme_map")
    stmt = insert(etf_theme_map).values(rows)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["etf_code", "theme_code", "valid_from"],
        set_={
            "mapping_source": stmt.excluded.mapping_source,
            "mapping_confidence": stmt.excluded.mapping_confidence,
            "is_primary": stmt.excluded.is_primary,
            "valid_to": stmt.excluded.valid_to,
            "updated_at": text("now()"),
        },
    )
    with get_engine().begin() as conn:
        result = conn.execute(upsert_stmt)
    return int(result.rowcount or 0)


def summarize_rows(rows: list[dict[str, Any]], unmatched: list[dict[str, Any]], dry_run: bool) -> None:
    matched_etf_count = len({row["etf_code"] for row in rows})
    theme_counts: dict[str, int] = {}
    for row in rows:
        theme_counts[row["theme_code"]] = theme_counts.get(row["theme_code"], 0) + 1

    top_theme_counts = ", ".join(
        f"{theme_code}:{count}"
        for theme_code, count in sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    )
    print(
        "ETF theme map build completed "
        f"dry_run={dry_run} matched_etfs={matched_etf_count} rows={len(rows)} unmatched={len(unmatched)} "
        f"top_themes=[{top_theme_counts}]"
    )

    if rows:
        preview = rows[:10]
        LOGGER.info("Mapping preview: %s", json.dumps(preview, ensure_ascii=False, default=str))
    if unmatched:
        LOGGER.info("Unmatched ETF preview: %s", json.dumps(unmatched[:10], ensure_ascii=False, default=str))


def main() -> int:
    setup_logging()
    args = parse_args()
    as_of_date = parse_as_of_date(args.as_of_date)
    override_path = Path(args.override_file)

    LOGGER.info(
        "Starting ETF theme map build as_of_date=%s dry_run=%s override_file=%s",
        as_of_date,
        args.dry_run,
        override_path,
    )

    try:
        ensure_theme_master(THEME_RULES, dry_run=args.dry_run)
        override_config = load_override_config(override_path)
        etf_rows = load_etf_master(limit=args.limit)
        mapping_rows, unmatched = build_mapping_rows(
            etf_rows,
            as_of_date=as_of_date,
            override_config=override_config,
        )
        affected_count = upsert_etf_theme_map(mapping_rows, dry_run=args.dry_run)

        LOGGER.info(
            "ETF theme map build finished etfs=%s rows=%s affected=%s unmatched=%s",
            len(etf_rows),
            len(mapping_rows),
            affected_count,
            len(unmatched),
        )
        summarize_rows(mapping_rows, unmatched, dry_run=args.dry_run)
        return 0
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while building ETF theme map: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("ETF theme map build failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
