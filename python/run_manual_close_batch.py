from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SERVING_DIR = ROOT / "serving"
OUTPUTS_DIR = ROOT / "outputs"

REQUIRED_DATE_FILES: list[tuple[str, Path, str]] = [
    ("market_status", DATA_DIR / "market_status.csv", "date"),
    ("features", DATA_DIR / "features.csv", "date"),
    ("predictions", DATA_DIR / "predictions.csv", "date"),
    ("ranking_final", DATA_DIR / "ranking_final.csv", "date"),
]

SERVING_JSON_CHECKS: list[tuple[str, Path, str]] = [
    ("daily_recommendations", SERVING_DIR / "daily_recommendations.json", "asof_date"),
    ("buy_gate_status", SERVING_DIR / "buy_gate_status.json", "asof_date"),
    ("model_portfolio", SERVING_DIR / "model_portfolio.json", "asof_date"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual close batch in strict sequence.")
    parser.add_argument("--skip-build", action="store_true", help="Skip docker compose build for python-pipeline.")
    parser.add_argument("--skip-node-api", action="store_true", help="Skip docker compose up -d --build node-api.")
    parser.add_argument("--skip-web-sync", action="store_true", help="Skip sync_web_display_data.py after refresh.")
    parser.add_argument("--skip-live-account", action="store_true", help="Skip read-only live-account sync and preview during operational refresh.")
    parser.add_argument("--web-sync-reset-first", action="store_true", help="Pass --reset-first to sync_web_display_data.py.")
    parser.add_argument("--refresh-arg", action="append", default=[], help="Extra arg forwarded to run_operational_refresh.py.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run pipeline directly via local .venv Python (skip docker). Implies --skip-build and --skip-node-api.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    print(f"[START] {name}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(command, cwd=ROOT, check=True, env=env)
    print(f"[OK] {name}")


def docker_available() -> bool:
    return shutil.which("docker") is not None


def latest_csv_date(path: Path, date_col: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"required csv missing: {path}")
    df = pd.read_csv(path, usecols=[date_col], low_memory=False)
    if df.empty:
        raise ValueError(f"required csv empty: {path}")
    parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if parsed.empty:
        raise ValueError(f"date column '{date_col}' has no valid values: {path}")
    return parsed.max().strftime("%Y-%m-%d")


def verify_pipeline_outputs() -> str:
    resolved: dict[str, str] = {}
    for label, path, date_col in REQUIRED_DATE_FILES:
        resolved[label] = latest_csv_date(path, date_col)

    expected = resolved["ranking_final"]
    mismatches = {label: value for label, value in resolved.items() if value != expected}
    if mismatches:
        details = ", ".join(f"{label}={value}" for label, value in sorted(resolved.items()))
        raise RuntimeError(
            "pipeline outputs are not aligned on the same latest date; "
            f"expected ranking_final={expected}; resolved: {details}"
        )

    print("[OK] pipeline output dates aligned:", ", ".join(f"{label}={value}" for label, value in sorted(resolved.items())))
    return expected


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required json missing: {path}")
    raw = path.read_text(encoding="utf-8-sig")
    normalized = raw.replace("NaN", "null").replace("Infinity", "null").replace("-null", "null")
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError(f"json payload is not an object: {path}")
    return payload


def verify_serving_outputs(expected_asof_date: str) -> None:
    resolved: dict[str, str | None] = {}
    for label, path, field in SERVING_JSON_CHECKS:
        payload = read_json(path)
        value = payload.get(field)
        resolved[label] = str(value).strip() if value is not None else None

    mismatches = {label: value for label, value in resolved.items() if value != expected_asof_date}
    if mismatches:
        details = ", ".join(f"{label}={value}" for label, value in sorted(resolved.items()))
        raise RuntimeError(
            "operational refresh outputs are not aligned to the pipeline date; "
            f"expected asof_date={expected_asof_date}; resolved: {details}"
        )

    print("[OK] refresh output dates aligned:", ", ".join(f"{label}={value}" for label, value in sorted(resolved.items())))


def main() -> int:
    args = parse_args()

    # --local implies skip-build and skip-node-api (no docker involvement)
    if args.local:
        args.skip_build = True
        args.skip_node_api = True

    has_docker = docker_available() and not args.local
    pipeline_env = {
        "RUN_PIPELINE_SKIP_FLOW_INGESTION": os.environ.get("RUN_PIPELINE_SKIP_FLOW_INGESTION", "0"),
    }

    if not args.skip_build and not has_docker:
        raise RuntimeError("docker is required for python-pipeline build, but the docker CLI is not available")

    if not args.skip_build:
        run_step(
            "docker compose build python-pipeline",
            ["docker", "compose", "build", "--no-cache", "--progress=plain", "python-pipeline"],
        )

    pipeline_command = (
        ["docker", "compose", "run", "--rm", "python-pipeline"]
        if has_docker
        else [sys.executable, str(ROOT / "python" / "run_pipeline.py")]
    )
    pipeline_name = "docker compose run --rm python-pipeline" if has_docker else "run_pipeline.py (local)"
    run_step(pipeline_name, pipeline_command, extra_env=pipeline_env)

    market_date = verify_pipeline_outputs()

    refresh_args = list(args.refresh_arg)
    if not args.skip_live_account and "--with-live-account" not in refresh_args:
        refresh_args.append("--with-live-account")

    refresh_command = [sys.executable, str(ROOT / "python" / "run_operational_refresh.py"), *refresh_args]
    run_step(
        "run_operational_refresh",
        refresh_command,
        extra_env={"MARKET_DATE": market_date},
    )

    verify_serving_outputs(market_date)

    if not args.skip_web_sync:
        sync_command = [sys.executable, str(ROOT / "python" / "sync_web_display_data.py")]
        if args.web_sync_reset_first:
            sync_command.append("--reset-first")
        run_step("sync_web_display_data", sync_command, extra_env={"MARKET_DATE": market_date})

    if not args.skip_node_api:
        if not docker_available():
            raise RuntimeError("docker is required to restart node-api, but the docker CLI is not available")
        run_step(
            "docker compose up -d --build node-api",
            ["docker", "compose", "up", "-d", "--build", "node-api"],
        )

    print(f"[DONE] manual close batch completed market_date={market_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
