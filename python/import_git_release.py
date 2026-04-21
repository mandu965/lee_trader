from __future__ import annotations

import argparse
import os
import stat
import shutil
from pathlib import Path

from export_git_release import (
    DEFAULT_TARGET,
    ROOT_DIRS,
    ROOT_FILES,
    CopyStats,
    _remove_file,
    iter_copyable_files,
)


DEFAULT_SOURCE = DEFAULT_TARGET
DEFAULT_TARGET_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy git-deployable project files from the deploy clone back into the local project tree."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Directory to import from.")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Project root to receive the imported files.",
    )
    parser.add_argument(
        "--clean-target",
        action="store_true",
        help="Delete only managed deploy paths in target before copy.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing files.")
    return parser.parse_args()


def ensure_parent(path: Path, dry_run: bool, stats: CopyStats) -> None:
    parent = path.parent
    if parent.exists():
        return
    if dry_run:
        return
    parent.mkdir(parents=True, exist_ok=True)
    stats.created_dirs += 1


def copy_file(src: Path, dst: Path, dry_run: bool, stats: CopyStats) -> None:
    ensure_parent(dst, dry_run, stats)
    if dry_run:
        print(f"COPY {src} -> {dst}")
        stats.copied_files += 1
        return
    shutil.copy2(src, dst)
    stats.copied_files += 1


def _handle_remove_readonly(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise exc_info[1]


def clean_managed_target(source: Path, target: Path, dry_run: bool) -> None:
    managed_roots = []
    for rel in ROOT_FILES + ROOT_DIRS:
        source_candidate = source / rel
        if source_candidate.exists():
            managed_roots.append(target / rel)

    for path in managed_roots:
        if not path.exists():
            continue
        if dry_run:
            print(f"REMOVE {path}")
            continue
        if path.is_dir():
            shutil.rmtree(path, onerror=_handle_remove_readonly)
        else:
            _remove_file(path)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    stats = CopyStats()

    if not source.exists():
        raise FileNotFoundError(f"source root not found: {source}")
    if not target.exists() and not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    if args.clean_target:
        clean_managed_target(source, target, args.dry_run)

    selected_roots: list[Path] = []
    for rel in ROOT_FILES + ROOT_DIRS:
        candidate = source / rel
        if candidate.exists():
            selected_roots.append(candidate)
        else:
            print(f"SKIP missing: {candidate}")
            stats.skipped_paths += 1

    for selected in selected_roots:
        for src_path in iter_copyable_files(source, selected):
            dst_path = target / src_path.relative_to(source)
            copy_file(src_path, dst_path, args.dry_run, stats)

    print("")
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"copied_files: {stats.copied_files}")
    print(f"created_dirs: {stats.created_dirs}")
    print(f"skipped_paths: {stats.skipped_paths}")
    print(f"dry_run: {args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
