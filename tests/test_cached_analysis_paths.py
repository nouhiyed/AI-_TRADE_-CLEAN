import unittest

from core.coordinator import TradingCoordinator


class FakeFuturesScanner:
    def __init__(self):
        self.calls = []

    def scan_universe(self, limit=None, min_quote_volume=None):
        self.calls.append((limit, min_quote_volume))
        return [
            {
                "symbol": "FUT%02dUSDT" % index,
                "quote_volume": float(1000 - index),
                "book_liquidity_usdt": 100000.0 - index,
                "spread_bps": 1.0,
            }
            for index in range(30)
        ]


class FakeCache:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = []

    def _snapshot(self, market, symbol, max_age_seconds=None):
        self.calls.append((market, symbol, max_age_seconds))
        if not self.ready:
            return None
        return {"5m": "5m", "15m": "15m", "1h": "1h"}


class CachedAnalysisPathTests(unittest.TestCase):
    def test_futures_scan_uses_cached_frames_not_rest_analysis(self):
        coordinator = object.__new__(TradingCoordinator)
        coordinator._agent_enabled = lambda _domain: True
        coordinator.market_scanner = FakeFuturesScanner()
        coordinator.market_stream = FakeCache()
        coordinator.agents = {"futures": object()}
        coordinator.last_signals = {}
        coordinator.analyzed = []

        def analyze(symbol, domain, frames, agent=None):
            coordinator.analyzed.append((symbol, domain, frames, agent))
            return {
                "signals": {
                    "futures": {
                        "symbol": symbol,
                        "direction": "LONG",
                        "strategy_version": "v3",
                        "confidence": 85,
                        "trade_plan": {
                            "success": True,
                            "strategy_version": "v3",
                            "direction": "LONG",
                            "timeframe": "5m",
                            "entry": 100.0,
                            "stop_loss": 99.0,
                            "take_profit": 102.0,
                        },
                    }
                },
                "data_errors": {},
            }

        coordinator.analyze_live_from_frames = analyze
        coordinator.analyze_live = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Futures scan must not fall back to REST analysis")
        )
        result = coordinator.scan_futures()

        self.assertEqual(result["scanned"], 30)
        self.assertEqual(result["actionable"], 30)
        self.assertEqual(len(coordinator.market_stream.calls), 30)
        self.assertEqual(len(coordinator.analyzed), 30)
        self.assertEqual(coordinator.analyzed[0][1], "futures")

    def test_chart_submission_waits_for_cache_instead_of_rest_fallback(self):
        coordinator = object.__new__(TradingCoordinator)
        coordinator.agents = {"spot": object()}
        coordinator._agent_enabled = lambda _domain: True
        coordinator.market_stream = FakeCache(ready=False)
        coordinator.analyze_live = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Chart submission must not fall back to REST analysis")
        )

        result = coordinator.submit_chart_signal(
            "spot", "BTCUSDT", "5m",
            entry_price=100.0, take_profit=102.0, stop_loss=99.0,
        )

        self.assertEqual(result["decision_status"], "REJECTED")
        self.assertIn("cache is still warming", result["reason"].lower())
        self.assertEqual(coordinator.market_stream.calls, [("spot", "BTCUSDT", 90)])


if __name__ == "__main__":
    unittest.main()
