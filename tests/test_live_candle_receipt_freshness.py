import json
import time
import unittest

from core.binance_websocket import BinanceKlineStream


class LiveCandleReceiptFreshnessTests(unittest.TestCase):
    def test_latest_candle_tracks_local_receipt_independently_of_exchange_time(self):
        class Feed:
            pass

        stream = BinanceKlineStream(Feed(), lambda *_args: None, max_symbols=1)
        now_ms = int(time.time() * 1000)
        packet = {
            "e": "kline",
            "s": "BTCUSDT",
            # Simulate an exchange/device clock offset greater than 30 seconds.
            "E": now_ms - 120_000,
            "k": {
                "i": "5m", "t": now_ms - 60_000,
                "T": now_ms + 239_999,
                "o": "100", "h": "101", "l": "99",
                "c": "100.5", "v": "10", "x": False,
            },
        }
        stream._handle_message(
            "futures", 0, 1, type("App", (), {})(), json.dumps(packet)
        )
        candle = stream.latest_candle("futures", "BTCUSDT")
        self.assertIsNotNone(candle)
        self.assertEqual(candle["event_time"], now_ms - 120_000)
        self.assertGreaterEqual(candle["received_at_ms"], now_ms)
        stream.stop()


if __name__ == "__main__":
    unittest.main()
