from __future__ import annotations

import unittest

from utils.us_order_status_mapper import (
    is_fill_status,
    is_terminal_order_status,
    map_broker_order_status,
    normalize_fill_payload,
)


class USOrderStatusMapperTests(unittest.TestCase):
    def test_map_open_status(self) -> None:
        self.assertEqual(map_broker_order_status("MOCK", "accepted"), "ORDER_OPEN")

    def test_map_partial_fill_status(self) -> None:
        self.assertEqual(map_broker_order_status("MOCK", "partially_filled"), "ORDER_PARTIALLY_FILLED")

    def test_map_filled_status(self) -> None:
        self.assertEqual(map_broker_order_status("MOCK", "filled", {"filled_qty": 1, "order_qty": 1}), "ORDER_FILLED")

    def test_unknown_status_maps_to_unknown(self) -> None:
        self.assertEqual(map_broker_order_status("MOCK", "weird_state"), "ORDER_UNKNOWN")

    def test_helpers(self) -> None:
        self.assertTrue(is_fill_status("ORDER_FILLED"))
        self.assertTrue(is_terminal_order_status("ORDER_REJECTED"))

    def test_normalize_fill_payload(self) -> None:
        result = normalize_fill_payload(
            "MOCK",
            {
                "broker_fill_id": "FILL1",
                "filled_qty": 0.4,
                "filled_price": 900,
                "fill_time": "2026-05-15T15:30:00+00:00",
            },
        )
        self.assertEqual(result["broker_fill_id"], "FILL1")
        self.assertEqual(result["filled_amount_usd"], 360.0)


if __name__ == "__main__":
    unittest.main()
