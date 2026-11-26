import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model_v2 import build
from experiments.dense_history_pairs_20261017.model import map_state
from experiments.forecast_validity_pairs_20261017 import test_validity as fixture
from experiments.forecast_validity_pairs_20261017.policy import replay as prior_replay,extend_validity
from .model import infer
from .policy import replay


class HourlyInferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((Path(__file__).parent/'model_protocol.json').read_text())
        n=2400;self.t=np.arange(n)*3600;rng=np.random.default_rng(42)
        self.r=dict(market_times=self.t,close=100*np.exp(rng.normal(0,.001,(n,2)).cumsum(0)))
        self.d=dict(x=rng.normal(size=(n,1,2)),beta=np.ones((n,1)),vol=np.ones((n,1))*.001,
                    fresh=np.ones((n,1),bool),pairs=[[0,1]],names=['f1','f2'])
        self.models={(('BTC','ETH'),24):dict(names=['f1','f2'],center=[0.,0.],scale=[1.,1.],coefficient=[.1,.01,-.02])}

    def test_six_hour_control_exact_and_hourly_coefficients_only_use_mature_rows(self):
        hourly,rebuilt,records=infer(self.t,self.r,self.d,self.models,[['BTC','ETH']],24,self.spec)
        old=copy.deepcopy(self.spec);old['decision_hours']=6
        pp,rr=build(self.r,self.d,self.models,[['BTC','ETH']],24,old)
        for kind in hourly:
            state=map_state(self.t,self.r,self.d,pp[kind],self.d['pairs'])
            for k in state:np.testing.assert_array_equal(state[k],rebuilt[kind][k])
        for row in records:self.assertLess(row['last_maturity'],row['time'])
        self.assertEqual([x for x in records if x['time']%21600==0],rr)
        self.assertTrue(any(x['time']%21600!=0 for x in records))

    def test_new_hour_uses_new_price_and_keeps_its_own_maturity(self):
        hourly,_,_=infer(self.t,self.r,self.d,self.models,[['BTC','ETH']],24,self.spec)
        i=1501;s=hourly['ridge']
        self.assertEqual(s['signal_times'][i,0],self.t[i])
        np.testing.assert_array_equal(s['source_prices'][i],self.r['close'][i])
        self.assertNotEqual(s['signal_times'][i,0],self.t[i]//21600*21600)

    def test_future_prices_do_not_change_earlier_predictions_or_calibration(self):
        before,_,records=infer(self.t,self.r,self.d,self.models,[['BTC','ETH']],24,self.spec)
        cut=1800;self.r['close'][cut+1:]*=2.;self.d['x'][cut+1:]*=-10
        after,_,changed=infer(self.t,self.r,self.d,self.models,[['BTC','ETH']],24,self.spec)
        for kind in before:
            for k in ['mu','beta','vol','source_prices','signal_times','directions']:
                np.testing.assert_array_equal(before[kind][k][:cut+1],after[kind][k][:cut+1])
        self.assertEqual([r for r in records if r['time']<=self.t[cut]], [r for r in changed if r['time']<=self.t[cut]])

    def test_intrahour_mapping_cannot_use_next_complete_bar(self):
        times=np.array([1501*3600+3599,1502*3600])
        hourly,_,_=infer(times,self.r,self.d,self.models,[['BTC','ETH']],24,self.spec)
        np.testing.assert_array_equal(hourly['ridge']['signal_times'][:,0],[1501*3600,1502*3600])
        np.testing.assert_array_equal(hourly['ridge']['source_prices'][0],self.r['close'][1501])

    def test_all_existing_execution_variants_remain_exact(self):
        f=fixture.ValidityTests();f.setUp()
        for arm in ['first_strategy','second_strategy','first_dense_ridge_24','first_dense_renew_ridge_24',
                    'first_dense_renewhedge_ridge_24','first_dense_refreshhedge_ridge_24']:
            state=extend_validity(f.de,f.t) if '_refreshhedge_' in arm else f.de
            a=replay(f.d,state,None,f.cfg,f.p,arm,'base',int(f.t[0]),int(f.t[-1]))
            b=prior_replay(f.d,state,None,f.cfg,f.p,arm,'base',int(f.t[0]),int(f.t[-1]))
            for k in ['fills','snapshots','metrics','events']:self.assertEqual(a[k],b[k],(arm,k))


if __name__=='__main__':unittest.main()
