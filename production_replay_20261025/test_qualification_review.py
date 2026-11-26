import copy
import unittest

from .qualification_review import qualification, REVIEWS


def fixture():
    group = dict(name='quote_history', primary='quote_inputs', controls=['incumbent','frozen_side_quote'],
                 start=1, replay_start=0, rows=[], checks=[])
    for scenario in ('base','stress'):
        group['rows'] += [dict(arm=a, scenario=scenario) for a in ('quote_inputs','incumbent','frozen_side_quote')]
        group['checks'].append(dict(scenario=scenario, passed=True, comparisons=[
            dict(control=a, advantage_after_exit=10., rounding_bound=.01, passed=True)
            for a in group['controls']]))
    groups = {label:copy.deepcopy(group) for label in REVIEWS}
    capacity = {(label,row['arm'],row['scenario']):dict(complete=True)
                for label,g in groups.items() for row in g['rows']}
    fidelity = {label:dict(valid=True,absolute_exit_pnl_mismatch=1.) for label in REVIEWS}
    return groups,capacity,fidelity


class QualificationTests(unittest.TestCase):
    def test_complete_synthetic_evidence_qualifies_without_live_release(self):
        result=qualification(*fixture())
        self.assertTrue(result['short_conditional_trial_eligible'])
        self.assertFalse(result['live_release_created'])
        self.assertIsNone(result['official_Q'])

    def test_no_selection_of_favorable_timing_phase(self):
        groups,capacity,fidelity=fixture()
        groups['realistic_phase45']['checks'][1]['passed']=False
        self.assertFalse(qualification(groups,capacity,fidelity)['short_conditional_trial_eligible'])

    def test_missing_secondary_control_is_not_ignored(self):
        groups,capacity,fidelity=fixture()
        groups['realistic_phase0']['checks'][0]['comparisons'].pop()
        with self.assertRaisesRegex(ValueError,'control'):
            qualification(groups,capacity,fidelity)

    def test_observed_execution_discrepancy_exceeds_improvement(self):
        groups,capacity,fidelity=fixture()
        fidelity['realistic_phase0']['absolute_exit_pnl_mismatch']=11.
        result=qualification(groups,capacity,fidelity)
        self.assertFalse(result['short_conditional_trial_eligible'])
        self.assertTrue(any('discrepancy' in r for r in result['reasons']))

    def test_missing_capacity_check_does_not_pass(self):
        groups,capacity,fidelity=fixture()
        del capacity[('realistic_phase45','quote_inputs','stress')]
        self.assertFalse(qualification(groups,capacity,fidelity)['short_conditional_trial_eligible'])

    def test_changed_window_is_not_comparable(self):
        groups,capacity,fidelity=fixture()
        groups['realistic_phase0']['start']=2
        with self.assertRaisesRegex(ValueError,'windows'):
            qualification(groups,capacity,fidelity)

    def test_partial_family_is_not_a_result(self):
        groups,capacity,fidelity=fixture()
        del groups['realistic_phase45']
        with self.assertRaisesRegex(ValueError,'timing'):
            qualification(groups,capacity,fidelity)


if __name__=='__main__':unittest.main()
