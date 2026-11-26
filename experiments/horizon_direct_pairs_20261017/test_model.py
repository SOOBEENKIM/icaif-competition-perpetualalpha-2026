import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.policy import held_state
from .model import infer

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


class HorizonDirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs=json.loads((HERE/'protocol.json').read_text())['pairs']
        cls.spec=json.loads((ROOT/'experiments/rolling_feature_pairs_20261017/protocol.json').read_text())
        cls.p=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']
        cls.cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
        with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
            cls.reference={k:z[k].copy() for k in ['market_times','close','volume','trades']}
        cls.runs={h:infer(cls.reference['market_times'],cls.reference,cls.p['symbols'],cls.pairs,cls.spec,h) for h in (6,24,72)}

    def test_24h_control_exact_forecast_and_risk_units(self):
        state,_,_=self.runs[24]
        with np.load(ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/reference_direct_24_state.npz') as z:
            for k in ['mu','vol','beta']:
                np.testing.assert_allclose(state[k],z[k],atol=1e-14,rtol=1e-12,equal_nan=True)
            for k in ['directions','signal_times','source_prices']:np.testing.assert_array_equal(state[k],z[k])

    def test_targets_use_own_horizon_and_fixed_source_hedge(self):
        r=self.reference;i=4000
        features=design(r['market_times'],r['close'],r['volume'],r['trades'],self.p['symbols'],self.pairs)
        for h,(_,records,a) in self.runs.items():
            np.testing.assert_array_equal(a['maturity'],r['market_times']+h*3600)
            self.assertTrue(np.isnan(a['target'][-h:]).all())
            for k,(left,right) in enumerate(features['pairs']):
                beta=features['beta'][i,k];vol=features['vol'][i,k]
                returns=r['close'][i+h,[left,right]]/r['close'][i,[left,right]]-1
                target=(returns[0]-beta*returns[1])/(1+beta)/vol/np.sqrt(h)
                self.assertAlmostEqual(a['target'][i,k],target,places=12)
            for record in records:
                self.assertEqual(record['fit_time']%86400,0)
                self.assertLess(record['last_maturity'],record['fit_time'])
                self.assertEqual(record['horizon'],h)
                self.assertGreaterEqual(record['rows'],200)

    def test_future_mutation_cannot_change_forecasts_or_daily_fit(self):
        cutoff=int(np.datetime64('2026-09-01T12:00:00','s').astype('int64'))
        changed=copy.deepcopy(self.reference);after=changed['market_times']>cutoff
        changed['close'][after]*=1.2
        for h in (6,72):
            state,records,_=infer(changed['market_times'],changed,self.p['symbols'],self.pairs,self.spec,h)
            old,old_records,_=self.runs[h]
            np.testing.assert_allclose(state['mu'][~after],old['mu'][~after],atol=1e-14,rtol=1e-12,equal_nan=True)
            self.assertEqual([r for r in records if r['fit_time']<=cutoff],
                             [r for r in old_records if r['fit_time']<=cutoff])

    def test_horizon_propagates_to_close_lead_and_original_stop(self):
        for step in (300,3600):
            times=np.arange(0,80*3600+1,step)
            d={'times':times,'par_close':np.ones((len(times),10))*100}
            for h in (6,24,72):
                pair=dict(dense=True,forecast_source=0,horizon=h,first_fill=step,entry_vol=.001,
                    risk_entry_prices=[100.,100.],risk_quantities=[1.,-1.],ids=[0,1])
                before=(h*3600)//step-2;close=before+1
                self.assertIsNone(held_state(pair,before,d,None,self.cfg)['reason'])
                state=held_state(pair,close,d,None,self.cfg)
                self.assertEqual(state['reason'],'forecast_maturity')
                self.assertEqual(state['maturity'],h*3600)
                self.assertAlmostEqual(state['return_stop_bound'],self.cfg['dense_policy']['stop_sigma']*.001*np.sqrt(h))

    def test_unknown_horizon_rejected_and_spec_not_mutated(self):
        before=copy.deepcopy(self.spec)
        with self.assertRaises(ValueError):
            infer(self.reference['market_times'],self.reference,self.p['symbols'],self.pairs,self.spec,12)
        self.assertEqual(before,self.spec)
        self.assertEqual(self.spec['horizon'],24)


if __name__=='__main__':unittest.main()
