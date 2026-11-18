import unittest

from api.main import (build_cycle_coverage, reconcile_completed_symbols,
                      select_reconcile_batch)


class ReconcileBatchRotationTests(unittest.TestCase):
    def test_coverage_uses_disjoint_exact_cycle_buckets(self):
        universe = {"PAIR%03dUSDT" % index for index in range(250)}
        ordered = sorted(universe)
        completed = set(ordered[:38])
        pending_analysis = set(ordered[38:50])
        waiting_cache = set(ordered[50:55])
        missing_cache = set(ordered[55:138])
        accepted = completed | pending_analysis | waiting_cache
        queue_rejected = set(ordered[138:145])

        coverage = build_cycle_coverage(
            universe, accepted, completed, missing_cache, waiting_cache,
            queue_rejected, analysis_errors=0, cycle_open_time_ms=123,
        )

        self.assertEqual(coverage["expected_symbols"], 250)
        self.assertEqual(coverage["accepted_symbols"], 55)
        self.assertEqual(coverage["completed_symbols"], 38)
        self.assertEqual(coverage["pending_cache_symbols"], 88)
        self.assertEqual(coverage["pending_analysis_symbols"], 12)
        self.assertEqual(coverage["pending_reconcile_symbols"], 112)
        self.assertEqual(coverage["queue_rejected_symbols"], 7)
        self.assertEqual(coverage["coverage_overlap_symbols"], 0)
        self.assertEqual(coverage["coverage_unclassified_symbols"], 0)
        self.assertFalse(coverage["complete"])

    def test_coverage_never_counts_completed_symbols_as_pending(self):
        universe = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        accepted = set(universe)
        completed = {"BTCUSDT", "ETHUSDT"}
        # Deliberately stale cache/reconcile bookkeeping must not duplicate
        # completed symbols in the published coverage buckets.
        coverage = build_cycle_coverage(
            universe, accepted, completed,
            missing_cache={"BTCUSDT", "SOLUSDT"},
            waiting_cache={"ETHUSDT"},
            queue_rejected={"BTCUSDT", "SOLUSDT"},
            analysis_errors=0,
        )
        self.assertEqual(coverage["completed_symbols"], 2)
        self.assertEqual(coverage["pending_cache_symbols"], 1)
        self.assertEqual(coverage["pending_analysis_symbols"], 0)
        self.assertEqual(coverage["pending_reconcile_symbols"], 0)
        self.assertEqual(coverage["coverage_overlap_symbols"], 0)
        self.assertEqual(coverage["coverage_unclassified_symbols"], 0)
        self.assertFalse(coverage["complete"])

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
