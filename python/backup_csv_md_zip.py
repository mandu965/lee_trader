"""Backup CSV files under the project root into a ZIP archive.

This script scans the current project tree, collects `.csv` files,
and stores them in a ZIP while preserving relative paths. A small manifest is
also embedded so the restore script can validate the archive before extraction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "backups"
ALLOWED_SUFFIXES = {".csv"}
MANIFEST_PATH = "_backup_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup CSV files into a ZIP archive."
    )
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help="Root directory to scan. Defaults to the current project root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output ZIP path. If omitted, a dated file is created under backups/.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Relative directory to exclude from backup. Can be provided multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output ZIP if it already exists.",
    )
    return parser.parse_args()


def normalize_relative_dir(value: str) -> str:
    return value.replace("\\", "/").strip("/").lower()


def collect_target_files(root: Path, excluded_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts[:-1]
        rel_dir = "/".join(rel_parts).lower()
        if any(rel_dir == ex or rel_dir.startswith(f"{ex}/") for ex in excluded_dirs):
            continue
        files.append(path)
    return sorted(files)


def build_output_path(root: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = root / output_path
        return output_path

    DEFAULT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_BACKUP_DIR / f"csv_md_backup_{timestamp}.zip"


def build_manifest(root: Path, files: list[Path], excluded_dirs: set[str]) -> dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "file_count": len(files),
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "excluded_dirs": sorted(excluded_dirs),
        "files": [str(path.relative_to(root)).replace("\\", "/") for path in files],
    }


def write_zip(root: Path, files: list[Path], output_path: Path, manifest: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in files:
            arcname = str(path.relative_to(root)).replace("\\", "/")
            zf.write(path, arcname=arcname)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    if not root.exists():
        print(f"ROOT_ERROR: root directory not found: {root}")
        return 1

    excluded_dirs = {normalize_relative_dir(item) for item in args.exclude_dir if item}
    files = collect_target_files(root, excluded_dirs)
    output_path = build_output_path(root, args.output)

    if output_path.exists() and not args.overwrite:
        print(f"WRITE_ERROR: output already exists: {output_path}")
        print("Use --overwrite or provide a different --output path.")
        return 1

    manifest = build_manifest(root, files, excluded_dirs)
    write_zip(root, files, output_path, manifest)

    print(f"backup root: {root}")
    print(f"backup zip: {output_path}")
    print(f"file count: {len(files)}")
    print("included suffixes: .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
