from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

payload_store_stub = types.ModuleType("payload_store")
payload_store_stub.upsert_json_payload = lambda *args, **kwargs: None
sys.modules.setdefault("payload_store", payload_store_stub)

import run_daily_scheduler


class RunDailySchedulerTests(unittest.TestCase):
    def test_live_sync_updates_ai_position_state_after_holdings_sync(self) -> None:
        with patch.dict(os.environ, {"SCHEDULER_COMMAND_SET": "live_sync"}, clear=False):
            steps = run_daily_scheduler._resolve_run_steps()

        step_names = [name for name, _ in steps]
        self.assertIn("sync_live_account_holdings", step_names)
        self.assertIn("update_ai_position_state", step_names)
        self.assertIn("sync_live_order_fills", step_names)
        self.assertLess(
            step_names.index("sync_live_account_holdings"),
            step_names.index("update_ai_position_state"),
        )
        self.assertLess(
            step_names.index("update_ai_position_state"),
            step_names.index("sync_live_order_fills"),
        )

        command = dict(steps)["update_ai_position_state"]
        self.assertTrue(command[-1].endswith(os.path.join("python", "update_ai_position_state.py")))


if __name__ == "__main__":
    unittest.main()
