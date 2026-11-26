from copy import deepcopy
import unittest
from .forward import assert_ready, completed_pairs, comparison_checks, ARMS


class EvaluationTests(unittest.TestCase):
    def test_seal_guard_runs_before_source_or_capture_reads(self):
        with self.assertRaisesRegex(ValueError, 'sealed'):
            assert_ready(dict(outcome_release_unix=1000), now=999)
        with self.assertRaisesRegex(ValueError, 'flushing'):
            assert_ready(dict(outcome_release_unix=1000), now=1010)

    def test_aborted_unhedged_leg_is_not_a_completed_pair(self):
        events = [dict(event=e) for e in ['entry_intent', 'exit_intent', 'flat_confirmed',
            'entry_intent', 'balanced_open', 'exit_intent', 'flat_confirmed',
            'entry_intent', 'balanced_open', 'exit_intent']]
        self.assertEqual(completed_pairs(events), 1)

    def rows(self):
        return [dict(arm=n, scenario=s, mark_net_pnl=100 if n == ARMS[-1] else 10,
            metrics=dict(mark_equity=dict(financial_score=.2 if n == ARMS[-1] else .1, completed_days=7)),
            completed_pairs=2 if n == ARMS[-1] else 0, fatal=[], flat=True, pending_orders=0)
            for n in ARMS for s in ('base', 'stress')]

    def test_cost_stress_and_scaled_control_cannot_be_ignored(self):
        rows = self.rows(); self.assertTrue(all(c['passed'] for c in comparison_checks(rows)))
        next(r for r in rows if r['arm'] == 'second_scaled_control' and r['scenario'] == 'stress')['mark_net_pnl'] = 101
        checks = comparison_checks(rows)
        self.assertFalse(all(c['passed'] for c in checks))
        self.assertFalse(next(c for c in checks if c['scenario'] == 'stress' and c['baseline'] == 'second_scaled_control')['gates']['higher_pnl'])

    def test_positive_profit_with_no_daily_score_or_unclosed_inventory_fails(self):
        for change in (dict(flat=False), dict(pending_orders=1), dict(completed_pairs=1), dict(fatal=['gap'])):
            rows = self.rows()
            next(r for r in rows if r['arm'] == ARMS[-1] and r['scenario'] == 'base').update(change)
            self.assertFalse(all(c['passed'] for c in comparison_checks(rows)))
        rows = self.rows()
        next(r for r in rows if r['arm'] == ARMS[-1] and r['scenario'] == 'base')['metrics']['mark_equity']['financial_score'] = None
        self.assertFalse(all(c['passed'] for c in comparison_checks(rows)))


if __name__ == '__main__': unittest.main()
