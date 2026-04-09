from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DECISION_JSON = DATA_DIR / "theme_promotion_decision.json"
LOG_PATH = DATA_DIR / "theme_operational_mode_apply_log.txt"
ENV_PATH = BASE_DIR / ".env"

TARGET_ENV = {
    "ENABLE_THEME_OVERLAY": "1",
    "THEME_OVERLAY_MODE": "operational",
    "ENABLE_THEME_VALIDATION": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply theme overlay operational mode from promotion decision.")
    parser.add_argument("--env-file", default=".env", help="Target .env file path relative to project root or absolute path.")
    parser.add_argument("--dry-run", action="store_true", help="Preview .env changes without writing them.")
    return parser.parse_args()


def load_promotion_decision(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required decision file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid decision payload: {path}")
    return payload


def _split_lines_with_endings(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines(keepends=True)


def update_env_text(original_text: str, updates: dict[str, str]) -> tuple[str, list[str]]:
    lines = _split_lines_with_endings(original_text)
    changed_keys: list[str] = []
    seen_keys: set[str] = set()

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in updates:
            continue

        existing_value = value.rstrip("\r\n")
        desired_value = updates[key]
        seen_keys.add(key)
        if existing_value != desired_value:
            newline = "\n"
            if line.endswith("\r\n"):
                newline = "\r\n"
            lines[index] = f"{key}={desired_value}{newline}"
            changed_keys.append(key)

    if lines and not lines[-1].endswith(("\n", "\r\n")):
        lines[-1] = lines[-1] + "\n"

    for key, desired_value in updates.items():
        if key in seen_keys:
            continue
        lines.append(f"{key}={desired_value}\n")
        changed_keys.append(key)

    return "".join(lines), changed_keys


def write_apply_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def update_env_file(env_path: Path, updates: dict[str, str]) -> list[str]:
    original_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    updated_text, changed_keys = update_env_text(original_text, updates)
    env_path.write_text(updated_text, encoding="utf-8")
    return changed_keys


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = BASE_DIR / env_path

    try:
        decision_payload = load_promotion_decision(DECISION_JSON)
    except Exception as exc:
        message = f"PROMOTION NOT APPLIED: failed_to_load_decision ({exc})"
        print(message)
        write_apply_log(LOG_PATH, message)
        return 1

    decision = str(decision_payload.get("decision", "")).strip().upper()
    if decision != "PROMOTE":
        message = f"PROMOTION NOT APPLIED: decision={decision or 'UNKNOWN'}"
        print(message)
        write_apply_log(LOG_PATH, message)
        return 0

    try:
        original_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        _, changed_keys = update_env_text(original_text, TARGET_ENV)
    except Exception as exc:
        message = f"PROMOTION NOT APPLIED: env_update_failed ({exc})"
        print(message)
        write_apply_log(LOG_PATH, message)
        return 1

    if args.dry_run:
        message = f"THEME OVERLAY PROMOTION DRY RUN: keys={','.join(changed_keys) if changed_keys else 'none'}"
        print(message)
        write_apply_log(LOG_PATH, f"{message} | env={env_path}")
        return 0

    try:
        changed_keys = update_env_file(env_path, TARGET_ENV)
    except Exception as exc:
        message = f"PROMOTION NOT APPLIED: env_write_failed ({exc})"
        print(message)
        write_apply_log(LOG_PATH, message)
        return 1

    message = "THEME OVERLAY PROMOTED TO OPERATIONAL MODE"
    print(message)
    write_apply_log(
        LOG_PATH,
        f"{message} | keys={','.join(changed_keys) if changed_keys else 'none'} | env={env_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
