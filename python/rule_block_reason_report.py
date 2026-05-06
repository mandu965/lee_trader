from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rule_signal_builder import ROOT, resolve


OUTPUT_DIR = ROOT / "outputs"
DEFAULT_PREVIEW = OUTPUT_DIR / "rule_order_preview.json"
DEFAULT_OUT = OUTPUT_DIR / "rule_block_reason_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate order block reason statistics from rule_order_preview.json.")
    parser.add_argument("--preview-json", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_preview(path: Path) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"rule order preview not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _item_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": item.get("code"),
        "name": item.get("name"),
        "side": item.get("side"),
        "portfolio_action": item.get("portfolio_action"),
        "portfolio_action_reason": item.get("portfolio_action_reason"),
        "rule_score": item.get("rule_score"),
        "rule_score_v2": item.get("rule_score_v2"),
        "signal_strength": item.get("signal_strength"),
        "order_amount": item.get("order_amount"),
        "current_return_pct": item.get("current_return_pct"),
        "holding_days": item.get("holding_days"),
        "entry_signal": item.get("entry_signal"),
        "strong_entry_signal": item.get("strong_entry_signal"),
    }


def build_block_reason_stats(preview: dict[str, Any]) -> dict[str, Any]:
    items = preview.get("items") or []
    as_of_date = preview.get("as_of_date")

    order_items = [i for i in items if i.get("side") in {"BUY", "SELL"}]
    skip_items = [i for i in items if i.get("side") == "NONE"]
    blocked_order_items = [i for i in order_items if not i.get("order_allowed")]
    allowed_order_items = [i for i in order_items if i.get("order_allowed")]

    # --- order-level block reason aggregation ---
    reason_counter: Counter[str] = Counter()
    reason_items: dict[str, list[dict[str, Any]]] = {}
    for item in blocked_order_items:
        raw = str(item.get("order_block_reason") or "none")
        for r in raw.split(";"):
            r = r.strip()
            if r and r != "none":
                reason_counter[r] += 1
                reason_items.setdefault(r, []).append(_item_snapshot(item))

    order_block_stats = [
        {
            "block_reason": reason,
            "count": count,
            "items": reason_items.get(reason, []),
        }
        for reason, count in reason_counter.most_common()
    ]

    # --- portfolio-level skip reason aggregation (not_strong_entry_signal etc.) ---
    portfolio_reason_counter: Counter[str] = Counter()
    portfolio_reason_items: dict[str, list[dict[str, Any]]] = {}
    for item in skip_items:
        reason = str(item.get("portfolio_action_reason") or "no_action")
        if reason not in {"no_action", "none", ""}:
            portfolio_reason_counter[reason] += 1
            portfolio_reason_items.setdefault(reason, []).append(_item_snapshot(item))

    portfolio_block_stats = [
        {
            "portfolio_block_reason": reason,
            "count": count,
            "top_items": portfolio_reason_items.get(reason, [])[:10],
        }
        for reason, count in portfolio_reason_counter.most_common()
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "summary": {
            "total_items": len(items),
            "order_items": len(order_items),
            "blocked_order_count": len(blocked_order_items),
            "allowed_order_count": len(allowed_order_items),
            "skip_count": len(skip_items),
            "distinct_order_block_reasons": len(reason_counter),
            "distinct_portfolio_block_reasons": len(portfolio_reason_counter),
        },
        "order_block_reason_stats": order_block_stats,
        "portfolio_block_reason_stats": portfolio_block_stats,
    }


def main() -> None:
    args = parse_args()
    preview = load_preview(args.preview_json)
    stats = build_block_reason_stats(preview)
    out = resolve(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
