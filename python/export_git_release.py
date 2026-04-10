from __future__ import annotations

import argparse
import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGET = Path(r"D:\ai\git\lee_trader")

ROOT_FILES = [
    ".dockerignore",
    ".env.example",
    ".env.render.example",
    "base_weights.json",
    "bootstrap.ps1",
    "docker-compose.yml",
    "render.yaml",
    "schema.sql",
    "ScoreContributionBar.tsx",
]

ROOT_DIRS = [
    ".github",
    "config",
    "doc",
    "docs",
    "node",
    "postgres",
    "python",
    "scripts",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "backups",
    "data",
    "logs",
    "models",
    "node_modules",
    "output",
    "outputs",
    "serving",
    "tmp",
    "venv",
    "_tmp_kis_openapi",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".DS_Store",
    "Thumbs.db",
    "python.zip",
}

EXCLUDED_GLOBS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.log",
    "*.zip",
]

GENERATED_GITIGNORE = """# Runtime secrets
.env
.env.local
.env.production
.env.development

# Generated data
data/
logs/
output/
outputs/
backups/
artifacts/
models/
serving/

# Local dependencies / caches
node_modules/
node/node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.venv/
venv/

# OS / editor noise
.DS_Store
Thumbs.db
"""


@dataclass
class CopyStats:
    copied_files: int = 0
    created_dirs: int = 0
    skipped_paths: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy only git-deployable project files to a clean target directory.")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1], help="Project root to export from.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Directory to receive the export.")
    parser.add_argument("--clean-target", action="store_true", help="Delete target contents before copy, while preserving .git.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without changing files.")
    return parser.parse_args()


def should_exclude(path: Path) -> bool:
    name = path.name
    if path.is_dir() and name in EXCLUDED_DIR_NAMES:
        return True
    if path.is_file() and name in EXCLUDED_FILE_NAMES:
        return True
    if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_GLOBS):
        return True
    return False


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


def iter_copyable_files(root: Path, base: Path):
    if base.is_file():
        if not should_exclude(base):
            yield base
        return

    for path in sorted(base.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if should_exclude(path):
            continue
        if path.is_file():
            yield path


def clean_target(target: Path, dry_run: bool) -> None:
    if not target.exists():
        return
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if dry_run:
            print(f"REMOVE {child}")
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_gitignore(target: Path, dry_run: bool, stats: CopyStats) -> None:
    gitignore_path = target / ".gitignore"
    if dry_run:
        print(f"WRITE {gitignore_path}")
        return
    target.mkdir(parents=True, exist_ok=True)
    gitignore_path.write_text(GENERATED_GITIGNORE, encoding="utf-8")
    stats.copied_files += 1


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    stats = CopyStats()

    if not source.exists():
        raise FileNotFoundError(f"source root not found: {source}")

    if args.clean_target:
        clean_target(target, args.dry_run)

    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

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

    write_gitignore(target, args.dry_run, stats)

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
