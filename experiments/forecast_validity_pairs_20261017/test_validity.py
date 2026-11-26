import copy
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.paid_rehedge_pairs_20261017 import test_rehedge as fixture
from experiments.paid_rehedge_pairs_20261017.portfolio import replay as prior_replay
from .policy import extend_validity, replay


class ValidityTests(unittest.TestCase):
    def setUp(self):
        f=fixture.PaidAdjustmentTests();f.setUp()
        self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        self.t=self.d['times'];self.de['beta'][:]=1.
        self.cfg['source_validity_policy']=json.loads((Path(__file__).parent/'par_protocol.json').read_text())['source_validity_policy']

    def run_arm(self, arm='pair_dense_refreshhedge_ridge_24', de=None, engine=replay):
        return engine(self.d, self.de if de is None else de, None, self.cfg, self.p, arm, 'base', int(self.t[0]), int(self.t[-1]))

    def test_one_hour_mapping_and_all_forecast_values_are_unchanged(self):
        for hours in [1,6]:
            out=extend_validity(self.de,self.t,hours)
            for k in self.de:
                if k!='directions':np.testing.assert_array_equal(out[k],self.de[k])
            original=self.de['directions']!=0
            np.testing.assert_array_equal(out['directions'][original],self.de['directions'][original])
        np.testing.assert_array_equal(extend_validity(self.de,self.t,1)['directions'],self.de['directions'])

    def test_invalid_new_source_cannot_reuse_previous_forecast(self):
        source=self.t[0]+6*3600;self.de['directions'][self.de['signal_times'][:,0]==source]=0
        out=extend_validity(self.de,self.t)
        self.assertTrue(np.all(out['directions'][self.t<source]))
        self.assertTrue(np.all(out['directions'][(self.t>=source)&(self.t<source+6*3600)]==0))

    def test_no_valid_source_seen_does_not_invent_history(self):
        start=int(np.searchsorted(self.t,self.t[0]+2*3600))
        truncated={k:(v[start:].copy() if isinstance(v,np.ndarray) and len(v)==len(self.t) else copy.deepcopy(v)) for k,v in self.de.items()}
        out=extend_validity(truncated,self.t[start:])
        self.assertTrue(np.all(out['directions'][:48]==0))

    def test_future_sources_or_new_validity_beyond_refresh_are_rejected(self):
        with self.assertRaises(ValueError):extend_validity(self.de,self.t,7)
        self.de['signal_times'][0]+=3600
        with self.assertRaises(ValueError):extend_validity(self.de,self.t)

    def test_prefix_invariance_when_future_source_is_changed(self):
        a=extend_validity(self.de,self.t);cut=200
        self.de['mu'][cut:]*=-1;self.de['directions'][cut:]=0
        b=extend_validity(self.de,self.t)
        for k in ['directions','mu','beta','signal_times']:
            np.testing.assert_array_equal(a[k][:cut],b[k][:cut])

    def test_control_account_paths_remain_exact(self):
        for arm in ['first_strategy','second_strategy','first_dense_ridge_24','first_dense_renew_ridge_24','first_dense_renewhedge_ridge_24']:
            a=self.run_arm(arm);b=self.run_arm(arm,engine=prior_replay)
            for key in ['fills','snapshots','events','metrics']:self.assertEqual(a[key],b[key],(arm,key))

    def test_later_economic_entry_retains_original_terminal_and_maturity(self):
        # Current venue is too expensive at source time. It becomes affordable
        # two hours later; no zero-fill attempt has been spent in the meantime.
        before=self.t<self.t[0]+2*3600
        self.d['par_close'][before,1]=102.;self.d['par_close'][before,0]=98.
        extended=extend_validity(self.de,self.t)
        r=self.run_arm(de=extended)
        first=next(e for e in r['events'] if e['event']=='pair_intent')
        self.assertEqual(first['time'],int(self.t[0])+2*3600)
        self.assertEqual(first['forecast_source'],int(self.t[0]))
        self.assertEqual(first['terminal_prices'],[102.,98.])
        old=self.run_arm('pair_dense_renewhedge_ridge_24')
        oldfirst=next(e for e in old['events'] if e['event']=='pair_intent')
        self.assertEqual(oldfirst['time'],int(self.t[0])+6*3600)
        # No future refresh can be used to prolong the entry's original label.
        extended['directions'][self.t>=self.t[0]+6*3600]=0
        fixed=self.run_arm(de=extended)
        closed=next(e for e in fixed['events'] if e['event']=='closed')
        self.assertEqual(closed['time'],int(self.t[0])+24*3600)

    def test_expanded_validity_does_not_make_stale_price_profitable(self):
        self.d['par_close'][:,1]=102.;self.d['par_close'][:,0]=98.
        r=self.run_arm(de=extend_validity(self.de,self.t))
        self.assertEqual(r['fills'],[])


if __name__=='__main__':unittest.main()
