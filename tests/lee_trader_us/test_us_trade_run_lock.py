from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from python.us.trade_orchestration.config import load_trade_orchestration_config
from python.us.trade_orchestration.run_lock import acquire_run_lock, inspect_run_lock, release_run_lock


class TradeRunLockTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_LOCK_TTL_SECONDS": "1",
        },
        clear=False,
    )
    def test_duplicate_lock_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_TRADE_LOCK_DIR": tmpdir}, clear=False):
                cfg = load_trade_orchestration_config()
                first = acquire_run_lock(cfg, trade_date="2026-05-14", owner="test")
                second = acquire_run_lock(cfg, trade_date="2026-05-14", owner="test")
                self.assertTrue(first["lock_acquired"])
                self.assertFalse(second["lock_acquired"])
                self.assertEqual(second["reason"], "DUPLICATE_RUN_DETECTED")

    @patch.dict(
        os.environ,
        {
            "US_TRADE_SCHEDULER_LOCK_TTL_SECONDS": "1",
        },
        clear=False,
    )
    def test_stale_lock_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_TRADE_LOCK_DIR": tmpdir}, clear=False):
                cfg = load_trade_orchestration_config()
                path = Path(tmpdir) / "2026-05-14.lock"
                payload = {
                    "created_at": (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
                    "created_ts": time.time() - 5,
                    "trade_date": "2026-05-14",
                    "owner": "old",
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                result = acquire_run_lock(cfg, trade_date="2026-05-14", owner="new")
                self.assertTrue(result["lock_acquired"])
                self.assertTrue(result["stale_lock_removed"])

    def test_release_missing_lock_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"US_TRADE_LOCK_DIR": tmpdir}, clear=False):
                cfg = load_trade_orchestration_config()
                result = release_run_lock(cfg, trade_date="2026-05-14")
                self.assertTrue(result["released"])


if __name__ == "__main__":
    unittest.main()
