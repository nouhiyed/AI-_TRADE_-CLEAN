import unittest
from unittest.mock import call, patch

from core.binance_websocket import BinanceKlineStream


class FakeFeed:
    pass


class StopAfterPass:
    def is_set(self):
        return False

    def wait(self, _seconds):
        return True


class SpotOptionalRefreshSchedulingTests(unittest.TestCase):
    def test_optional_four_hour_refresh_waits_for_all_required_caches(self):
        stream = BinanceKlineStream(FakeFeed(), lambda *_args: None)
        stream._stop = StopAfterPass()
        stream._market_symbols["spot"] = {"BTCUSDT", "ETHUSDT"}

        with patch.object(stream, "_snapshot", return_value=None):
            with patch.object(stream, "_warm_symbol", return_value=False):
                with patch.object(stream, "_refresh_spot_4h_if_due") as refresh:
                    stream._warm_market("spot")

        refresh.assert_not_called()

    def test_optional_four_hour_refresh_runs_after_all_required_caches_are_ready(self):
        stream = BinanceKlineStream(FakeFeed(), lambda *_args: None)
        stream._stop = StopAfterPass()
        stream._market_symbols["spot"] = {"BTCUSDT", "ETHUSDT"}
        healthy = {"5m": object(), "15m": object(), "1h": object()}

        with patch.object(stream, "_snapshot", return_value=healthy):
            with patch.object(stream, "_refresh_spot_4h_if_due") as refresh:
                stream._warm_market("spot")

        self.assertEqual(
            refresh.call_args_list,
            [call("BTCUSDT"), call("ETHUSDT")],
        )


if __name__ == "__main__":
    unittest.main()
