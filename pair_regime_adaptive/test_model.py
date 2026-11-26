import unittest
from pathlib import Path
import numpy as np
from .model import Spec,fit,stats,select,build
ROOT=Path(__file__).resolve().parents[1]
class AdaptiveRegimeTests(unittest.TestCase):
    def test_known_false_hedges_are_rejected_by_return_covariance(self):
        d=np.load(ROOT/'practice_live/updates/20261023T080007449138Z/inputs.npz');i=int(np.searchsorted(d['market_times'],1792713600));logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0)
        for ids in [(1,6),(5,6)]:
            m,r=fit(logs,active,i,ids);self.assertIsNone(m);self.assertEqual(r['reasons'],['unsupported_return_hedge']);self.assertLess(r['return_statistics']['beta'],0)
    def test_variance_reduction_identity(self):
        rng=np.random.default_rng(12);y=rng.normal(size=500);x=.7*y+rng.normal(size=500);r=stats(x,y);actual=1-np.var(x-r['beta']*y)/np.var(x)
        self.assertAlmostEqual(r['hedged_variance_reduction'],actual,places=12)
    def test_rolling_hedge_recovers_a_stable_synthetic_relation(self):
        rng=np.random.default_rng(9);y=np.cumsum(rng.normal(0,.005,1700));x=1.5*y+np.cumsum(rng.normal(0,.0003,1700));logs=np.c_[x,y];m,r=fit(logs,np.ones_like(logs,bool),1600,(0,1))
        self.assertIsNotNone(m);self.assertTrue(r['eligible']);self.assertAlmostEqual(m['beta'],1.5,delta=.05)
        logs[1601:]=1000.;self.assertEqual((m,r),fit(logs,np.ones_like(logs,bool),1600,(0,1)))
    def fixture(self):
        t=np.arange(0,3000)*3600;prices=np.c_[np.exp(np.arange(3000)*.0002),np.ones(3000)];pred=np.full((3000,2),[-.014,.014]);beta=np.ones(3000);fresh=np.ones(3000,bool)
        return t,pred,beta,prices,fresh,2700*3600
    def test_persistent_trend_can_be_selected_over_reversion(self):
        args=self.fixture();r=select(*args);self.assertFalse(r[0]['eligible']);self.assertTrue(r[1]['eligible']);self.assertLess(r[1]['last_maturity'],args[-1])
    def test_future_or_unmatured_labels_cannot_change_regime_selection(self):
        t,p,b,x,f,source=self.fixture();r=select(t,p,b,x,f,source);x[t>=source]=1e6;p[t>=source]=-1e6
        self.assertEqual(r,select(t,p,b,x,f,source))
    def test_no_past_advantage_does_not_manufacture_an_expert(self):
        t,p,b,x,f,s=self.fixture();x[:]=1.;r=select(t,p,b,x,f,s);self.assertFalse(any(a['eligible'] for a in r))
    def test_insufficient_origins_cannot_be_relabelled_a_validation(self):
        t,p,b,x,f,s=self.fixture();r=select(t,p,b,x,f,500*3600);self.assertTrue(all(a['reason']=='insufficient_matured_predictions' for a in r))
if __name__=='__main__':unittest.main()
