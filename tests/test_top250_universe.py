import unittest
from unittest.mock import patch

from core.binance_market_data import BinanceMarketData
from core.binance_websocket import BinanceKlineStream
from core.market_scanner import FuturesMarketScanner


class RankedMarketFeed(BinanceMarketData):
    def __init__(self):
        super().__init__(timeout=0.2, total_timeout=0.2)
        self.requests = []
        self.symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        self.tickers = [
            {"symbol": "AAAUSDT", "quoteVolume": "1000000000", "lastPrice": "10", "priceChangePercent": "1"},
            {"symbol": "BBBUSDT", "quoteVolume": "800000000", "lastPrice": "10", "priceChangePercent": "2"},
            {"symbol": "CCCUSDT", "quoteVolume": "500000000", "lastPrice": "10", "priceChangePercent": "-1"},
        ]
        self.books = [
            {"symbol": "AAAUSDT", "bidPrice": "9.99", "askPrice": "10.01", "bidQty": "10", "askQty": "10"},
            {"symbol": "BBBUSDT", "bidPrice": "9.99", "askPrice": "10.00", "bidQty": "1000", "askQty": "1000"},
            {"symbol": "CCCUSDT", "bidPrice": "9.90", "askPrice": "10.10", "bidQty": "100", "askQty": "100"},
        ]

    def fetch_symbols(self, market):
        self.requested_market = market
        return list(self.symbols)

    def _request(self, market, endpoint, params):
        self.requests.append((market, endpoint, dict(params)))
        if endpoint.endswith("/ticker/24hr"):
            return list(self.tickers)
        if endpoint.endswith("/ticker/bookTicker"):
            return list(self.books)
        raise AssertionError("Unexpected endpoint: " + endpoint)


class Top250UniverseTests(unittest.TestCase):
    def test_spot_market_data_uses_official_market_data_only_rest_endpoint(self):
        self.assertEqual(
            BinanceMarketData.BASE_URLS["spot"],
            "https://data-api.binance.vision",
        )
        self.assertEqual(
            BinanceMarketData.FALLBACK_URLS["spot"][0],
            "https://data-api.binance.vision",
        )


    def test_429_without_retry_after_uses_bounded_backoff_then_retries_once(self):
        class Response:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self.headers = {}
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.responses = [
                    Response(429, {"code": -1003, "msg": "Too many requests"}),
                    Response(200, {"symbols": []}),
                ]
                self.calls = 0

            def get(self, *_args, **_kwargs):
                response = self.responses[self.calls]
                self.calls += 1
                return response

        feed = BinanceMarketData(timeout=6, total_timeout=6)
        feed.session = Session()
        with patch("core.binance_market_data.time.sleep") as sleep:
            result = feed._request("futures", "/fapi/v1/exchangeInfo", {})
        self.assertEqual(result, {"symbols": []})
        self.assertEqual(feed.session.calls, 2)
        self.assertEqual(sleep.call_args_list[0].args, (3.0,))
        self.assertIn(3.0, [call.args[0] for call in sleep.call_args_list])


    def test_repeated_429_uses_bounded_backoff_without_fallback_burst(self):
        class Response:
            status_code = 429
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {"code": -1003, "msg": "Too many requests"}

        class Session:
            def __init__(self):
                self.calls = 0

            def get(self, *_args, **_kwargs):
                self.calls += 1
                return Response()

        feed = BinanceMarketData(timeout=30, total_timeout=30)
        feed.session = Session()
        with patch("core.binance_market_data.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                feed._request("futures", "/fapi/v1/exchangeInfo", {})

        self.assertEqual(feed.session.calls, 4)
        delays = [call.args[0] for call in sleep.call_args_list]
        self.assertTrue(all(delay in delays for delay in (3.0, 6.0, 12.0)))

    def test_ranking_combines_24h_volume_book_liquidity_and_spread(self):
        feed = RankedMarketFeed()

        ranked = feed.fetch_top_symbols("spot", limit=3)

        # BBB wins on balanced top-of-book depth and spread while AAA's
        # much larger 24h turnover keeps it ahead of CCC.
        self.assertEqual([item["symbol"] for item in ranked], [
            "BBBUSDT", "AAAUSDT", "CCCUSDT"
        ])
        self.assertGreater(ranked[0]["book_liquidity_usdt"], ranked[1]["book_liquidity_usdt"])
        self.assertGreater(ranked[0]["volume_percentile"], 0)
        self.assertIn("ranking_day", ranked[0])
        self.assertEqual(len(feed.requests), 2)

    def test_daily_ranking_is_cached_and_force_refresh_rebuilds_it(self):
        feed = RankedMarketFeed()

        first = feed.fetch_top_symbols("futures", limit=2)
        second = feed.fetch_top_symbols("futures", limit=2)
        self.assertEqual(first, second)
        self.assertEqual(feed.requested_market, "futures")
        self.assertEqual(len(feed.requests), 2)

        feed.fetch_top_symbols("futures", limit=2, force_refresh=True)
        self.assertEqual(len(feed.requests), 4)

    def test_incomplete_book_snapshot_fails_closed(self):
        feed = RankedMarketFeed()
        feed.books = feed.books[:1]

        with self.assertRaisesRegex(RuntimeError, "Incomplete Binance liquidity snapshot"):
            feed.fetch_top_symbols("spot", limit=3)

    def test_websocket_loads_exactly_the_configured_top_n(self):
        class Feed:
            def __init__(self):
                self.requested = []

            def fetch_top_symbols(self, market, limit):
                self.requested.append((market, limit))
                return [
                    {"symbol": "COIN%03dUSDT" % index, "ranking_day": "2026-11-17"}
                    for index in range(limit)
                ]

        feed = Feed()
        stream = BinanceKlineStream(
            feed, lambda _market, _event: None, max_symbols=250
        )
        symbols = stream._load_symbols("spot")

        self.assertEqual(len(symbols), 250)
        self.assertEqual(len(set(symbols)), 250)
        self.assertEqual(feed.requested, [("spot", 250)])
        self.assertEqual(stream.status()["spot"]["ranking_day"], "2026-11-17")
        self.assertTrue(stream.status()["fast_scan_mode"])
        self.assertEqual(stream._active_intervals("spot"), ["5m"])
        stream.stop()

    def test_websocket_rejects_a_partial_top_n_selection(self):
        class Feed:
            def fetch_top_symbols(self, market, limit):
                return [
                    {"symbol": "COIN%03dUSDT" % index}
                    for index in range(limit - 1)
                ]

        stream = BinanceKlineStream(
            Feed(), lambda _market, _event: None, max_symbols=250
        )
        with self.assertRaisesRegex(RuntimeError, "exactly 250"):
            stream._load_symbols("futures")
        stream.stop()

    def test_daily_universe_prunes_removed_symbols_from_live_cache(self):
        stream = BinanceKlineStream(
            object(), lambda _market, _event: None, max_symbols=250
        )
        stream._cache = {
            ("spot", "KEEPUSDT", "5m"): "current",
            ("spot", "REMOVEUSDT", "5m"): "stale",
        }
        stream._cache_ready = {
            ("spot", "KEEPUSDT"),
            ("spot", "REMOVEUSDT"),
        }
        stream._latest_candles = {
            ("spot", "KEEPUSDT"): {"symbol": "KEEPUSDT"},
            ("spot", "REMOVEUSDT"): {"symbol": "REMOVEUSDT"},
        }
        stream._last_open = {
            ("spot", "KEEPUSDT"): 1,
            ("spot", "REMOVEUSDT"): 2,
        }
        stream._closed_seen = {
            ("spot", "KEEPUSDT", "5m", 1),
            ("spot", "REMOVEUSDT", "5m", 2),
        }
        stream._warm_errors = {
            ("spot", "REMOVEUSDT", "5m"): "old error",
        }

        stream._prune_market_state("spot", {"KEEPUSDT"})

        self.assertEqual(set(stream._cache), {("spot", "KEEPUSDT", "5m")})
        self.assertEqual(stream._cache_ready, {("spot", "KEEPUSDT")})
        self.assertEqual(set(stream._latest_candles), {("spot", "KEEPUSDT")})
        self.assertEqual(set(stream._last_open), {("spot", "KEEPUSDT")})
        self.assertEqual(stream._closed_seen, {("spot", "KEEPUSDT", "5m", 1)})
        self.assertEqual(stream._warm_errors, {})
        self.assertEqual(stream.status()["latest_candle_symbols"]["spot"], 1)
        stream.stop()

    def test_futures_scanner_never_exceeds_250(self):
        class Feed:
            def __init__(self):
                self.requested = None

            def fetch_top_symbols(self, market, limit):
                self.requested = (market, limit)
                return [
                    {"symbol": "COIN%03dUSDT" % index, "quote_volume": 1000 - index}
                    for index in range(250)
                ]

        feed = Feed()
        scanner = FuturesMarketScanner(market=feed)

        universe = scanner.scan_universe(limit=999)

        self.assertEqual(feed.requested, ("futures", 250))
        self.assertEqual(len(universe), 250)
        self.assertEqual(universe[0]["symbol"], "COIN000USDT")


if __name__ == "__main__":
    unittest.main()
