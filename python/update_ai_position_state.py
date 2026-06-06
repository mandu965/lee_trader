"""AI 포지션 상태 파일 갱신 스크립트.

sync_live_account_holdings 완료 후 자동매매/계좌 동기화 흐름에서 호출된다.
live_account_holdings.csv를 읽어 outputs/ai_position_state.json을 갱신한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except Exception:
    pass

from ai_position_risk import update_position_state_from_holdings

ROOT = Path(__file__).resolve().parents[1]
HOLDINGS_CSV = ROOT / "data" / "live_account_holdings.csv"


def main() -> int:
    if not HOLDINGS_CSV.exists():
        print(f"[SKIP] {HOLDINGS_CSV} not found — ai_position_state not updated")
        return 0

    try:
        holdings = pd.read_csv(HOLDINGS_CSV, dtype={"code": str})
    except Exception as exc:
        print(f"[ERROR] failed to read {HOLDINGS_CSV}: {exc}", file=sys.stderr)
        return 1

    if holdings.empty:
        print("[SKIP] holdings empty — ai_position_state not updated")
        return 0

    state = update_position_state_from_holdings(holdings)
    positions = state.get("positions") or {}
    print(f"[OK] ai_position_state updated: {len(positions)} active positions")
    for code, pos in positions.items():
        print(
            f"  {code} {pos.get('name','')} "
            f"entry={pos.get('entry_date')} "
            f"peak={pos.get('peak_price'):.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
