"""Build a git-safe runtime snapshot ZIP for reproducing local ranking outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "backups"
MANIFEST_PATH = "_backup_manifest.json"
BACKUP_GLOB = "git_runtime_snapshot_*.zip"
DEFAULT_FILES = [
    "data/market_status.csv",
    "data/quality.csv",
    "data/predictions.csv",
    "data/ranking_final.csv",
    "data/model.pkl",
    "data/paper_trading_positions.csv",
    "data/paper_trading_nav.csv",
    "data/trades.csv",
    "data/score_kpi_monitor.json",
    "data/confidence_score_v2.json",
    "data/theme_overlay_acceptance_report.md",
    "data/ranking_builder_theme_guard_report.md",
    "data/theme_lift_analysis.csv",
    "data/theme_overlay_gate_debug.json",
    "data/theme_overlay_gate_debug.md",
    "data/theme_overlay_mode_resolution.md",
    "data/theme_overlay_shadow_preview.csv",
    "data/theme_overlay_shadow_summary.json",
    "output/stock_theme_daily.csv",
    "outputs/score_kpi_monitor.md",
    "outputs/paper_trading_report.md",
    "outputs/top20_meaningfulness_report.json",
    "outputs/top20_buyability_report.json",
    "outputs/operational_buy_gate.json",
    "serving/daily_recommendations.json",
    "serving/model_portfolio.json",
    "serving/buy_gate_status.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a runtime snapshot ZIP for GitHub Actions ranking reproduction.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root. Defaults to current project root.")
    parser.add_argument("--output", default=None, help="Optional output ZIP path under backups/.")
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=1,
        help="Keep only the newest N git_runtime_snapshot_*.zip files in the output directory. Default: 1",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional relative file path to include. Can be provided multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output ZIP if it already exists.")
    return parser.parse_args()


def build_output_path(root: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = root / output_path
        return output_path

    DEFAULT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_BACKUP_DIR / f"git_runtime_snapshot_{timestamp}.zip"


def prune_old_backups(output_path: Path, keep_latest: int) -> list[Path]:
    if keep_latest < 1:
        raise ValueError("--keep-latest must be at least 1")
    backups = sorted(
        output_path.parent.glob(BACKUP_GLOB),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    to_delete = backups[keep_latest:]
    for path in to_delete:
        path.unlink(missing_ok=True)
    return to_delete


def build_manifest(root: Path, files: list[Path]) -> dict[str, object]:
    file_entries = []
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_entries.append({"path": rel, "sha256": digest, "size": path.stat().st_size})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "file_count": len(files),
        "files": [entry["path"] for entry in file_entries],
        "file_entries": file_entries,
        "backup_type": "git_runtime_snapshot",
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()

    requested = []
    seen = set()
    for rel in DEFAULT_FILES + list(args.include or []):
        norm = rel.replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)
        requested.append(norm)

    files: list[Path] = []
    missing: list[str] = []
    for rel in requested:
        path = root / rel
        if path.exists() and path.is_file():
            files.append(path)
        else:
            missing.append(rel)

    if missing:
        print("MISSING_FILES:")
        for rel in missing:
            print(f" - {rel}")
        return 1

    output_path = build_output_path(root, args.output)
    if output_path.exists() and not args.overwrite:
        print(f"WRITE_ERROR: output already exists: {output_path}")
        print("Use --overwrite or provide a different --output path.")
        return 1

    manifest = build_manifest(root, files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in files:
            zf.write(path, arcname=str(path.relative_to(root)).replace("\\", "/"))

    removed = prune_old_backups(output_path, args.keep_latest)
    print(f"snapshot root: {root}")
    print(f"snapshot zip: {output_path}")
    print(f"file count: {len(files)}")
    print(f"keep latest: {args.keep_latest}")
    print(f"pruned snapshots: {len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
