import unittest

from api.main import build_cycle_coverage


class CycleCoverageThresholdTests(unittest.TestCase):
    def test_cycle_succeeds_at_95_percent_successful_processing(self):
        universe = {"S%02dUSDT" % index for index in range(30)}
        coverage = build_cycle_coverage(
            universe, universe, universe, set(), set(), set(),
            analysis_errors=1, cycle_open_time_ms=1_800_000_000_000,
        )
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["successful_symbols"], 29)
        self.assertEqual(coverage["analysis_errors"], 1)
        self.assertAlmostEqual(coverage["processed_pct"], 96.67, places=2)

    def test_cycle_fails_below_95_percent_and_keeps_errors_separate(self):
        universe = {"S%02dUSDT" % index for index in range(30)}
        coverage = build_cycle_coverage(
            universe, universe, universe, set(), set(), set(),
            analysis_errors=2, cycle_open_time_ms=1_800_000_000_000,
        )
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["successful_symbols"], 28)
        self.assertEqual(coverage["analysis_errors"], 2)


if __name__ == "__main__":
    unittest.main()
