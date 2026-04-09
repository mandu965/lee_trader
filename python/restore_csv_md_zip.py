"""Restore CSV files from a ZIP archive into the project tree.

The archive is expected to be created by `backup_csv_md_zip.py` and to contain
relative paths plus an embedded manifest. Files are extracted back into the
target root while preserving the original directory layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MANIFEST_PATH = "_backup_manifest.json"
ALLOWED_SUFFIXES = {".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore CSV files from a ZIP archive."
    )
    parser.add_argument(
        "--zip",
        required=True,
        help="ZIP archive path created by backup_csv_md_zip.py",
    )
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help="Root directory to restore into. Defaults to the current project root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files during restore.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be restored without writing files.",
    )
    return parser.parse_args()


def load_manifest(zip_path: Path) -> dict[str, object]:
    with ZipFile(zip_path, "r") as zf:
        if MANIFEST_PATH not in zf.namelist():
            raise ValueError(f"manifest not found in archive: {MANIFEST_PATH}")
        with zf.open(MANIFEST_PATH) as fh:
            return json.loads(fh.read().decode("utf-8"))


def resolve_zip_members(zip_path: Path) -> list[str]:
    with ZipFile(zip_path, "r") as zf:
        members = []
        for name in zf.namelist():
            if name == MANIFEST_PATH or name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix in ALLOWED_SUFFIXES:
                members.append(name)
        return sorted(members)


def restore_archive(zip_path: Path, root: Path, overwrite: bool, dry_run: bool) -> tuple[int, int]:
    restored_count = 0
    skipped_count = 0

    with ZipFile(zip_path, "r") as zf:
        for member in resolve_zip_members(zip_path):
            target = root / Path(member)
            if target.exists() and not overwrite:
                skipped_count += 1
                continue
            if dry_run:
                restored_count += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            restored_count += 1

    return restored_count, skipped_count


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.is_absolute():
        zip_path = PROJECT_ROOT / zip_path
    zip_path = zip_path.resolve()

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    if not zip_path.exists():
        print(f"FILE_ERROR: zip archive not found: {zip_path}")
        return 1
    if not root.exists():
        print(f"ROOT_ERROR: restore root not found: {root}")
        return 1

    try:
        manifest = load_manifest(zip_path)
        member_count = len(resolve_zip_members(zip_path))
        restored_count, skipped_count = restore_archive(
            zip_path=zip_path,
            root=root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"RESTORE_ERROR: {exc}")
        return 1

    print(f"restore zip: {zip_path}")
    print(f"restore root: {root}")
    print(f"manifest file count: {manifest.get('file_count')}")
    print(f"archive member count: {member_count}")
    print(f"restored count: {restored_count}")
    print(f"skipped count: {skipped_count}")
    print(f"dry run: {args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
