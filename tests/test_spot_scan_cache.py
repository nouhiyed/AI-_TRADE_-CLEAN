import unittest

from core.coordinator import TradingCoordinator


class FakeSpotUniverse:
    def __init__(self):
        self.calls = []

    def fetch_top_symbols(self, market, limit):
        self.calls.append((market, limit))
        return [
            {
                "symbol": "COIN%02dUSDT" % index,
                "quote_volume": float(1000 - index),
                "last_price": 10.0,
                "book_liquidity_usdt": 100000.0 - index,
                "spread_bps": 1.0,
            }
            for index in range(30)
        ]


class FakeSpotCache:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = []

    def _snapshot(self, market, symbol, max_age_seconds=None):
        self.calls.append((market, symbol, max_age_seconds))
        return {"5m": "5m", "15m": "15m", "1h": "1h"} if self.ready else None


class SpotScanCacheOnlyTests(unittest.TestCase):
    def make_coordinator(self, ready=True):
        coordinator = object.__new__(TradingCoordinator)
        coordinator._agent_enabled = lambda _domain: True
        coordinator.market_feed = FakeSpotUniverse()
        coordinator.market_stream = FakeSpotCache(ready=ready)
        coordinator.agents = {"spot": object()}
        coordinator.last_signals = {}
        coordinator.analyzed = []

        def analyze(symbol, domain, frames, agent=None):
            coordinator.analyzed.append((symbol, domain, frames, agent))
            return {
                "signals": {
                    "spot": {
                        "symbol": symbol,
                        "direction": "BUY",
                        "strategy_version": "v3",
                        "confidence": 80,
                        "trade_plan": {
                            "success": True,
                            "strategy_version": "v3",
                            "direction": "LONG",
                            "timeframe": "5m",
                            "entry": 10.0,
                            "stop_loss": 9.9,
                            "take_profit": 10.2,
                        },
                    }
                },
                "data_errors": {},
            }

        coordinator.analyze_live_from_frames = analyze
        coordinator.analyze_live = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Spot scan must not fall back to REST analysis")
        )
        return coordinator

    def test_spot_scan_caps_universe_at_30_and_uses_only_cached_frames(self):
        coordinator = self.make_coordinator()
        result = coordinator.scan_spot(limit=999)

        self.assertEqual(coordinator.market_feed.calls, [("spot", 30)])
        self.assertEqual(result["scanned"], 30)
        self.assertEqual(result["actionable"], 30)
        self.assertEqual(len(coordinator.market_stream.calls), 30)
        self.assertEqual(len(coordinator.analyzed), 30)
        self.assertEqual(coordinator.analyzed[0][1], "spot")
        self.assertEqual(set(coordinator.analyzed[0][2]), {"5m", "15m", "1h"})

    def test_spot_scan_reports_cache_warmup_without_rest_fallback(self):
        coordinator = self.make_coordinator(ready=False)
        result = coordinator.scan_spot()

        self.assertEqual(coordinator.market_feed.calls, [("spot", 30)])
        self.assertEqual(result["scanned"], 30)
        self.assertEqual(result["actionable"], 0)
        self.assertEqual(len(coordinator.analyzed), 0)
        self.assertEqual(len(result["errors"]), 30)
        self.assertTrue(all(
            "cache" in reason.lower() or "warming" in reason.lower()
            for reason in result["errors"].values()
        ))


if __name__ == "__main__":
    unittest.main()
