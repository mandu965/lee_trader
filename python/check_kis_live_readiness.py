from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KIS account configuration and optionally perform a read-only balance probe."
    )
    parser.add_argument(
        "--probe-balance",
        action="store_true",
        help="Issue a token and fetch account balance summary. No orders are sent.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write readiness results as JSON.",
    )
    return parser.parse_args()


def _mask(value: str, *, head: int = 2, tail: int = 2) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if tail <= 0:
        return f"{text[:head]}{'*' * max(len(text) - head, 0)}"
    if len(text) <= head + tail:
        return "*" * len(text)
    return f"{text[:head]}{'*' * (len(text) - head - tail)}{text[-tail:]}"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_env() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


def _infer_env_dv(base_url: str | None) -> str:
    text = str(base_url or "").lower()
    return "demo" if "openapivts" in text else "real"


def _env_status() -> dict[str, object]:
    base_url = (os.getenv("KIS_BASE_URL") or "").strip()
    env_dv = (os.getenv("KIS_ENV_DV") or _infer_env_dv(base_url)).strip() or "unknown"
    app_key = (os.getenv("KIS_APP_KEY") or "").strip()
    app_secret = (os.getenv("KIS_APP_SECRET") or "").strip()
    cano = (os.getenv("KIS_CANO") or "").strip()
    acnt_prdt_cd = (os.getenv("KIS_ACNT_PRDT_CD") or "").strip()
    account_no = (os.getenv("KIS_ACCOUNT_NO") or "").strip()
    app_id = (os.getenv("KIS_APP_ID") or "").strip()
    app_password = (os.getenv("KIS_APP_PASSWORD") or "").strip()

    checks = {
        "has_base_url": bool(base_url),
        "has_app_key": bool(app_key),
        "has_app_secret": bool(app_secret),
        "has_cano": bool(cano),
        "has_acnt_prdt_cd": bool(acnt_prdt_cd),
        "has_account_no": bool(account_no),
        "has_app_id": bool(app_id),
        "has_app_password": bool(app_password),
        "is_real_domain": "openapi.koreainvestment.com:9443" in base_url.lower(),
        "is_demo_domain": "openapivts.koreainvestment.com:29443" in base_url.lower(),
    }
    ready_for_balance_probe = all(
        [
            checks["has_base_url"],
            checks["has_app_key"],
            checks["has_app_secret"],
            checks["has_cano"],
            checks["has_acnt_prdt_cd"],
        ]
    )
    ready_for_real_trading = bool(ready_for_balance_probe and checks["is_real_domain"] and env_dv == "real")

    warnings: list[str] = []
    if checks["is_demo_domain"]:
        warnings.append("KIS_BASE_URL is set to the demo domain. Real account orders are not ready.")
    if checks["has_app_id"]:
        warnings.append("KIS_APP_ID is present but is not used for broker authentication or account resolution.")
    if checks["has_app_password"]:
        warnings.append("KIS_APP_PASSWORD is present but is not used by the current code path.")
    if not checks["has_cano"] or not checks["has_acnt_prdt_cd"]:
        warnings.append("KIS_CANO and KIS_ACNT_PRDT_CD are required for account queries and orders.")

    return {
        "env_dv": env_dv,
        "base_url": base_url,
        "base_url_masked": base_url,
        "checks": checks,
        "resolved_account": {
            "cano_masked": _mask(cano, head=2, tail=0),
            "acnt_prdt_cd": acnt_prdt_cd or None,
            "account_no_masked": _mask(account_no, head=2, tail=2) if account_no else None,
        },
        "ready_for_balance_probe": ready_for_balance_probe,
        "ready_for_real_trading": ready_for_real_trading,
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    _load_env()
    status = _env_status()

    balance_probe: dict[str, object] | None = None
    if args.probe_balance:
        from kis_client import KISClient
        from kis_live_account import inquire_balance, resolve_account_env, summarize_cash

        client = KISClient.from_env()
        client.issue_access_token()
        account = resolve_account_env()
        _, summary_df = inquire_balance(client, account)
        cash_summary = summarize_cash(summary_df)
        balance_probe = {
            "probe_ok": True,
            "env_dv": account.env_dv,
            "cano_masked": _mask(account.cano, head=2, tail=0),
            "account_product_code": account.acnt_prdt_cd,
            "cash_summary": cash_summary,
        }

    payload = {
        "status": status,
        "balance_probe": balance_probe,
    }

    if args.out_json:
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
