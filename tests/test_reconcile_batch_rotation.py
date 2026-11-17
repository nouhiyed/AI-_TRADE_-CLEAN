import unittest

from api.main import select_reconcile_batch


class ReconcileBatchRotationTests(unittest.TestCase):
    def test_shrinking_pending_set_does_not_skip_symbols(self):
        universe = {"PAIR%03dUSDT" % index for index in range(250)}
        pending = set(universe)
        cursor = 0
        visited = set()

        for _ in range(10):
            batch, cursor = select_reconcile_batch(
                universe, pending, cursor, 96
            )
            self.assertTrue(batch)
            visited.update(batch)
            pending.difference_update(batch)
            if not pending:
                break

        self.assertFalse(pending)
        self.assertEqual(visited, universe)

    def test_symbols_without_cache_are_retried_after_full_rotation(self):
        universe = {"PAIR%03dUSDT" % index for index in range(250)}
        pending = set(universe)
        unavailable = {"PAIR000USDT", "PAIR111USDT"}
        cursor = 0
        first_pass = set()

        for _ in range(3):
            batch, cursor = select_reconcile_batch(
                universe, pending, cursor, 96
            )
            first_pass.update(batch)
            pending.difference_update(set(batch) - unavailable)

        self.assertEqual(pending, unavailable)
        batch, cursor = select_reconcile_batch(
            universe, pending, cursor, 96
        )
        self.assertEqual(set(batch), unavailable)


if __name__ == "__main__":
    unittest.main()
