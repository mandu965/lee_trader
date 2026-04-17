from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kis_client import KISClient
from kis_live_account import order_cash, resolve_account_env


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUT_CSV = OUTPUT_DIR / "kis_manual_order_result.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a single KIS cash order with explicit manual confirmation.")
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--code", required=True)
    parser.add_argument("--qty", required=True, type=int)
    parser.add_argument("--price", required=True, type=int)
    parser.add_argument("--ord-dvsn", default="00", help="00 limit, 01 market")
    parser.add_argument("--exchange", default="KRX")
    parser.add_argument("--execute", action="store_true", help="Actually place the order. Without this flag, only validate and print.")
    parser.add_argument("--confirm-text", default="", help="Must be LIVE_ORDER to execute.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    if args.qty <= 0:
        raise ValueError("qty must be > 0")
    if args.price < 0:
        raise ValueError("price must be >= 0")

    if not args.execute:
        print("dry_run_only: add --execute --confirm-text LIVE_ORDER to place a real/demo order")
        return 0
    if args.confirm_text != "LIVE_ORDER":
        raise ValueError("execution blocked: --confirm-text LIVE_ORDER is required")

    client = KISClient.from_env()
    client.issue_access_token()
    account = resolve_account_env()
    result = order_cash(
        client,
        account,
        side=args.side,
        pdno=str(args.code).zfill(6),
        ord_dvsn=args.ord_dvsn,
        ord_qty=str(args.qty),
        ord_unpr=str(args.price),
        excg_id_dvsn_cd=args.exchange,
    )
    out = result.copy()
    out["requested_side"] = args.side
    out["requested_code"] = str(args.code).zfill(6)
    out["requested_qty"] = args.qty
    out["requested_price"] = args.price
    out["env_dv"] = account.env_dv

    out_csv = _resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"order_result_csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
