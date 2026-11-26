import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.dense_history_pairs_20261017 import test_dense as fixture
from experiments.dense_history_pairs_20261017.portfolio import replay as original_replay
from .portfolio import replay
from .policy import renewal_plan,held_state


class RenewalTests(unittest.TestCase):
    def setUp(self):
        f=fixture.DensePortfolioTests();f.setUp();self.d,self.de,self.old,self.cfg,self.p=f.d,f.de,f.old,f.cfg,f.p
        self.d['par_close'][:]=100;self.de['source_prices'][:]=100
        self.cfg['renewal_policy']=json.loads((Path(__file__).resolve().parent/'par_protocol.json').read_text())['renewal_policy']

    def run_arm(self,arm='pair_dense_renew_ridge_24',engine=replay,de=None):
        return engine(self.d,self.de if de is None else de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]))

    def direct_plan(self,*,risk_blocked=False,broken_leg=False):
        t0=int(self.d['times'][0]);i=int(np.searchsorted(self.d['times'],t0+18*3600))
        pair=dict(renewable=True,dense=True,balanced=True,ids=[1,0],pair=['ETH','BTC'],pair_index=0,
            target=[5.,-5.],direction=1,forecast_source=t0,horizon=24,first_fill=t0+300,entry_vol=.002,
            risk_entry_prices=[100.1,99.9],risk_quantities=[5.,-5.])
        q=np.zeros(10);q[[1,0]]=[5.,0. if broken_leg else -5.]
        return renewal_plan(pair,q,i,self.d,self.de,self.cfg,self.p,self.cfg['scenarios']['base'],q,1e6,risk_blocked=risk_blocked)

    def test_disabled_renewal_reproduces_existing_controls_exactly(self):
        for arm in ['first_strategy','second_strategy','first_dense_ridge_24','second_dense_ridge_72']:
            a=self.run_arm(arm);b=self.run_arm(arm,original_replay)
            for key in ['fills','snapshots','metrics','events']:self.assertEqual(a[key],b[key],(arm,key))

    def test_renewal_keeps_actual_units_and_does_not_create_fills_or_completed_trades(self):
        r=self.run_arm();renewals=[e for e in r['events'] if e['event']=='renewed']
        self.assertEqual(len(renewals),2);self.assertEqual(r['metrics']['completed_pairs'],0)
        self.assertEqual(len(r['owner_fills']['pair']),2);self.assertGreater(r['metrics']['residual_gross'],0)
        self.assertEqual(renewals[0]['time'],int(self.d['times'][0])+18*3600)
        for e in renewals:
            self.assertLess(e['time'],e['old_maturity']);self.assertLessEqual(e['new_source'],e['time'])
            self.assertEqual(e['orders_required'],0);self.assertEqual(e['entry_fees_charged_again'],0)
            self.assertEqual(e['owned_qty'],[5.,-5.]);self.assertGreater(e['incremental_advantage'],e['buffer_cash'])

    def test_close_time_future_forecast_does_not_cancel_previous_close_decision(self):
        t=self.d['times'];self.de['mu'][(t>=t[0]+18*3600)&(t<t[0]+24*3600)]=-.02
        r=self.run_arm();first=next(e for e in r['events'] if e['event']=='closed')
        self.assertEqual(first['time'],int(t[0])+24*3600)
        self.assertFalse(any(e['event']=='renewed' and e['time']<first['time'] for e in r['events']))

    def test_original_loss_stop_remains_after_renewal(self):
        t=self.d['times'];self.d['par_close'][t>=t[0]+20*3600,1]=80.;self.de['vol'][t>=t[0]+18*3600]=10.
        r=self.run_arm();self.assertTrue(any(e['event']=='renewed' for e in r['events']))
        exit_=next(e for e in r['events'] if e['event']=='exit_requested')
        self.assertEqual(exit_['reason'],'pair_return_stop');self.assertEqual(exit_['time'],int(t[0])+20*3600)
        self.assertLess(exit_['return_stop_bound'],.03)

    def test_incomplete_paid_exit_is_not_reset_by_new_positive_forecast(self):
        t=self.d['times'];self.d['par_close'][t>=t[0]+20*3600,1]=80.
        self.d['par_volume'][(t>=t[0]+20*3600)&(t<t[0]+30*3600),:2]=0
        r=self.run_arm();closed=next(e for e in r['events'] if e['event']=='closed')
        self.assertGreaterEqual(closed['time'],int(t[0])+30*3600)
        self.assertFalse(any(e['event']=='renewed' and t[0]+20*3600<=e['time']<=closed['time'] for e in r['events']))
        self.assertGreater(r['metrics']['fees'],0)

    def test_new_hedge_that_requires_resizing_is_rejected(self):
        t=self.d['times'];self.de['beta'][t>=t[0]+18*3600]=2.
        r=self.run_arm();rejections=[e for e in r['events'] if e['event']=='renewal_rejected']
        self.assertEqual(rejections[0]['reason'],'new_hedge_requires_resizing')
        self.assertEqual(next(e for e in r['events'] if e['event']=='closed')['time'],int(t[0])+24*3600)

    def test_forecast_beyond_hard_holding_cap_cannot_be_shortened_for_free(self):
        self.cfg['renewal_policy']['maximum_holding_hours']=36
        r=self.run_arm();e=next(e for e in r['events'] if e['event']=='renewal_rejected')
        self.assertEqual(e['reason'],'forecast_exceeds_hard_holding_limit')
        self.assertEqual(next(e for e in r['events'] if e['event']=='closed')['time'],int(self.d['times'][0])+24*3600)

    def test_future_changes_preserve_all_earlier_renewals_and_fills(self):
        a=self.run_arm();cut=int(self.d['times'][0])+24*3600
        self.d['par_close'][self.d['times']>cut]*=1.2;self.de['mu'][self.d['times']>cut]*=-2
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual([e for e in a[key] if e['time']<=cut],[e for e in b[key] if e['time']<=cut],key)

    def test_missing_fresh_signal_keeps_original_exit(self):
        t=self.d['times'];self.de['directions'][t>=t[0]+6*3600]=0
        a=self.run_arm();b=self.run_arm('pair_dense_ridge_24',original_replay)
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])

    def test_zero_forecast_is_not_renewed_just_to_avoid_current_fee(self):
        self.de['mu'][:]=0.;r=self.direct_plan()
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'holding_value_below_incremental_cost_buffer')

    def test_incremental_value_compares_two_paid_exits_without_recharging_entry(self):
        self.de['mu'][:]=.0007;r=self.direct_plan()
        self.assertTrue(r['accepted']);self.assertLess(r['expected_hold_cash'],0.)
        self.assertGreater(r['incremental_advantage'],r['buffer_cash'])
        self.assertAlmostEqual(r['incremental_advantage'],r['expected_hold_cash']-r['immediate_close_cash'])
        self.assertEqual(r['entry_fees_charged_again'],0)

    def test_existing_halt_and_unbalanced_inventory_cannot_renew(self):
        self.assertEqual(self.direct_plan(risk_blocked=True)['reason'],'global_risk_or_halt')
        self.assertEqual(self.direct_plan(broken_leg=True)['reason'],'current_inventory_unbalanced')

    def test_future_dated_signal_cannot_renew(self):
        t0=int(self.d['times'][0]);i=int(np.searchsorted(self.d['times'],t0+18*3600))
        self.de['signal_times'][i,0]=self.d['times'][i]+3600
        self.assertEqual(self.direct_plan()['reason'],'no_new_available_signal')

    def test_renewal_prediction_does_not_credit_cash_and_actual_funding_paid_once(self):
        self.d['funding'][:,1]=.0001;self.d['known_hourly_funding'][:,1]=.0001
        r=self.run_arm();m=r['metrics']
        self.assertEqual(m['renewals'],2);self.assertAlmostEqual(m['funding'],-2.4)
        self.assertAlmostEqual(m['fees'],.9);self.assertAlmostEqual(m['net_pnl'],-4.3,places=7)


if __name__=='__main__':unittest.main()
