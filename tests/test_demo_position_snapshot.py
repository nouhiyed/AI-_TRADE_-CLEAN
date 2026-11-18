import threading
import unittest

from core.coordinator import TradingCoordinator


class TrackingLock:
    def __init__(self):
        self.lock = threading.RLock()
        self.held = False

    def __enter__(self):
        self.lock.acquire()
        self.held = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.held = False
        self.lock.release()


class PositionMap(dict):
    def __init__(self, lock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracking_lock = lock

    def items(self):
        if not self.tracking_lock.held:
            raise AssertionError("DEMO positions must be snapshotted under _demo_lock")
        return super().items()


class EmptyJournal:
    @staticmethod
    def closed_trades():
        return []


class DemoPositionSnapshotTests(unittest.TestCase):
    def test_status_snapshots_positions_under_demo_lock(self):
        coordinator = object.__new__(TradingCoordinator)
        lock = TrackingLock()
        coordinator._demo_lock = lock
        coordinator.demo_positions = PositionMap(lock, {
            "open-1": {
                "status": "DEMO_OPEN",
                "symbol": "BTCUSDT",
                "domain": "futures",
            },
            "closed-1": {
                "status": "DEMO_CLOSED",
                "symbol": "ETHUSDT",
                "domain": "spot",
            },
        })
        coordinator.journal = EmptyJournal()
        coordinator.mode = "DEMO"
        coordinator.demo_execution = None
        coordinator.demo_account_id = None

        result = coordinator.get_demo_positions()

        self.assertEqual([p["id"] for p in result["open_positions"]], ["open-1"])
        self.assertEqual([p["id"] for p in result["closed_positions"]], ["closed-1"])
        self.assertFalse(lock.held)


if __name__ == "__main__":
    unittest.main()
