import unittest
import numpy as np
from .policy import Settings, select_exposure, select_margin


class ExposureTests(unittest.TestCase):
    def data(self):
        return dict(base_qty=[10.], current_qty=[0.], prices=[100.], equity=10000.,
                    forecast=[.05], funding=[0.], hourly_covariance=[[1e-6]],
                    protected=[False], entry_allowed=[True], liquid=[True], decimals=[2],
                    risk=dict(gross_fraction=2., net_fraction=2., asset_fraction=2.,
                              asset_notional=20000., daily_volatility=.1, sizing_headroom=.9))

    def test_positive_edge_increases_exposure(self):
        d = self.data(); r = select_exposure(**d)
        self.assertGreater(r['selected_target_multiplier'], 1)
        self.assertGreater(r['forecast_utility_gain'], 0)

    def test_short_is_symmetric(self):
        d = self.data(); a = select_exposure(**d)
        d.update(base_qty=[-10.], forecast=[-.05])
        b = select_exposure(**d)
        self.assertEqual(a['target_qty'][0], -b['target_qty'][0])

    def test_uncosted_or_negative_edge_cannot_lever_up(self):
        for mu in [.003, 0., -.05]:
            d = self.data(); d['forecast'] = [mu]
            self.assertEqual(select_exposure(**d)['target_qty'], d['base_qty'])

    def test_higher_covariance_reduces_selected_multiplier(self):
        d = self.data(); a = select_exposure(**d)
        d['hourly_covariance'] = [[.0004]]
        b = select_exposure(**d)
        self.assertLess(b['selected_target_multiplier'], a['selected_target_multiplier'])

    def test_protected_timing_and_liquidity_cannot_be_overridden(self):
        for key, val in [('protected', True), ('entry_allowed', False), ('liquid', False)]:
            d = self.data(); d[key] = [val]
            self.assertEqual(select_exposure(**d)['target_qty'], d['base_qty'])

    def test_reduction_or_reversal_not_restored_by_scaling(self):
        for old in [15., -15.]:
            d = self.data(); d['current_qty'] = [old]
            self.assertEqual(select_exposure(**d)['target_qty'], d['base_qty'])

    def test_gross_cap_and_lot_precision(self):
        d = self.data(); d['risk']['gross_fraction'] = .2
        r = select_exposure(**d)
        self.assertLessEqual(r['after']['gross'], .18 + 1e-10)
        self.assertEqual(round(r['target_qty'][0], 2), r['target_qty'][0])

    def test_correlation_is_included(self):
        d = self.data()
        d.update(base_qty=[10., 10.], current_qty=[0., 0.], prices=[100., 100.],
                 forecast=[.05, .05], funding=[0., 0.], protected=[False, False],
                 entry_allowed=[True, True], liquid=[True, True], decimals=[2, 2])
        d['hourly_covariance'] = [[.0001, .0001], [.0001, .0001]]
        correlated = select_exposure(**d)
        d['hourly_covariance'] = [[.0001, 0.], [0., .0001]]
        independent = select_exposure(**d)
        self.assertLess(correlated['selected_target_multiplier'], independent['selected_target_multiplier'])

    def test_bad_covariance_or_forecast_rejected(self):
        d = self.data(); d['hourly_covariance'] = [[-1.]]
        with self.assertRaises(ValueError): select_exposure(**d)
        d = self.data(); d['forecast'] = [float('nan')]
        with self.assertRaises(ValueError): select_exposure(**d)

    def test_costs_never_invent_alpha(self):
        d = self.data()
        r = select_exposure(**d, settings=Settings(entry_cost=.03, exit_cost=.03))
        self.assertEqual(r['target_qty'], d['base_qty'])


class MarginTests(unittest.TestCase):
    def data(self):
        return dict(own_notional=[60000.], teammate_notional=[0.], own_equity=50000.,
                    teammate_equity=50000., current_leverage=[1], exchange_max=[40], leverage_ceiling=3)

    def test_minimum_feasible_integer_not_max_exchange_value(self):
        r = select_margin(**self.data())
        self.assertEqual(r['target_leverage'], [2])
        self.assertEqual(r['own_actual_gross_leverage'], 1.2)

    def test_no_automatic_lending_of_teammate_budget(self):
        d = self.data(); d['teammate_equity'] = 1000000.
        self.assertEqual(select_margin(**d)['target_leverage'], [2])

    def test_shared_position_requires_shared_setting_authorization(self):
        d = self.data(); d['teammate_notional'] = [-60000.]
        r = select_margin(**d)
        self.assertEqual(r['target_leverage'], [2])
        self.assertEqual(r['physical_margin_estimate'], 0.)
        self.assertFalse(r['setting_change_permitted'])
        d['shared_setting_authorized'] = True
        self.assertTrue(select_margin(**d)['setting_change_permitted'])

    def test_exchange_cap_can_make_proposal_infeasible(self):
        d = self.data(); d['exchange_max'] = [1]
        self.assertFalse(select_margin(**d)['feasible'])

    def test_small_position_remains_one_and_inactive_markets_untouched(self):
        d = self.data(); d.update(own_notional=[1000., 0.], teammate_notional=[0., 0.],
                                 current_leverage=[1, 20], exchange_max=[40, 20])
        self.assertEqual(select_margin(**d)['target_leverage'], [1, 20])

    def test_insufficient_equity_rejected(self):
        d = self.data(); d['own_equity'] = 0.
        with self.assertRaises(ValueError): select_margin(**d)


if __name__ == '__main__': unittest.main()
