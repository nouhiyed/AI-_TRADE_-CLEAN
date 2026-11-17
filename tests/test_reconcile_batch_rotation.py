import unittest

from api.main import reconcile_completed_symbols, select_reconcile_batch


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

    def test_precompleted_symbols_are_removed_from_all_pending_sets(self):
        universe = {"PAIR%03dUSDT" % index for index in range(250)}
        precompleted = set(sorted(universe)[:62])
        state = {
            "symbols": set(universe),
            "pending": set(universe),
            "accepted": set(),
            "completed": set(),
            "missing_cache": set(precompleted),
            "queue_rejected": set(precompleted),
        }

        reconcile_completed_symbols(state, precompleted)

        self.assertEqual(state["completed"], precompleted)
        self.assertEqual(state["accepted"], precompleted)
        self.assertEqual(state["pending"], universe - precompleted)
        self.assertFalse(state["pending"].intersection(state["completed"]))
        self.assertFalse(state["missing_cache"])
        self.assertFalse(state["queue_rejected"])

        # A later audit completion must also clear symbols that were pending
        # when the reconciliation worker first started.
        remaining = universe - precompleted
        reconcile_completed_symbols(state, remaining)
        self.assertEqual(state["completed"], universe)
        self.assertEqual(state["accepted"], universe)
        self.assertFalse(state["pending"])
        self.assertFalse(state["missing_cache"])
        self.assertFalse(state["queue_rejected"])


    def test_symbols_without_cache_are_retried_after_full_rotation(self):
        universe = {"PAIR%03dUSDT" % index for index in range(250)}
        pending = set(universe)
        unavailable = set(sorted(universe)[62:])
        cursor = 0
        first_pass = set()

        for _ in range(3):
            batch, cursor = select_reconcile_batch(
                universe, pending, cursor, 96
            )
            first_pass.update(batch)
            pending.difference_update(set(batch) - unavailable)

        self.assertEqual(pending, unavailable)
        retried = set()
        for _ in range(3):
            batch, cursor = select_reconcile_batch(
                universe, pending, cursor, 96
            )
            retried.update(batch)

        self.assertEqual(retried, unavailable)


if __name__ == "__main__":
    unittest.main()
