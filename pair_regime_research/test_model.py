import unittest
from pathlib import Path
import numpy as np
from pair_model_research.adaptive_hedge import fit_pair as old_fit
from .model import Spec,formation,fit_ols,trend_guard,build,ARMS
ROOT=Path(__file__).resolve().parents[1]

class RelationshipRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=dict(np.load(ROOT/'practice_live/updates/20261023T080007449138Z/inputs.npz'))
        cls.logs=np.log(cls.d['close']);cls.active=(cls.d['volume']>0)&(cls.d['trades']>0);cls.i=int(np.searchsorted(cls.d['market_times'],1792713600))

    def test_original_failure_is_reproducible_then_rejected_by_actual_fit(self):
        for ids in [(1,6),(5,6)]:
            old=old_fit(self.logs,self.active,self.i,ids)
            self.assertIsNotNone(old);self.assertLess(old['level_correlation'],.05)
            new,audit=fit_ols(self.logs,self.active,self.i,ids)
            self.assertIsNone(new);self.assertEqual(audit['reason'],'unsupported_ols_hedge')

    def test_future_prices_cannot_change_formation_or_regime(self):
        before=formation(self.logs,self.active,self.i)
        x=self.logs.copy();x[self.i+1:]=1000.;a=self.active.copy();a[self.i+1:]=False
        self.assertEqual(before,formation(x,a,self.i))
        old=old_fit(self.logs,self.active,self.i,(0,2))
        self.assertEqual(trend_guard(self.logs,self.active,self.i,(0,2),old),trend_guard(x,a,self.i,(0,2),old))

    def test_true_stable_relationship_is_not_vetoed_unconditionally(self):
        rng=np.random.default_rng(104);n=1700;common=np.cumsum(rng.normal(0,.006,n));noise=np.zeros((n,10))
        for i in range(1,n):noise[i]=.96*noise[i-1]+rng.normal(0,.0007,10)
        logs=common[:,None]*np.linspace(.8,1.2,10)+noise;models,audits=formation(logs,np.ones_like(logs,bool),1600)
        self.assertEqual(len(audits),45);self.assertGreater(sum(r['eligible'] for r in audits),0)
        for r in audits:
            if r['eligible']:self.assertLessEqual(r['holm_pvalue'],.05)

    def test_multiple_testing_and_block_reasons_are_retained_for_all45(self):
        models,audit=formation(self.logs,self.active,self.i)
        self.assertEqual(len(audit),45);self.assertEqual(len({r['pair_index'] for r in audit}),45)
        for r in audit:
            self.assertTrue(0<=r['holm_pvalue']<=1)
            if r['eligible']:self.assertIn(r['pair_index'],models);self.assertEqual(r['reasons'],[])

    def test_diverging_trend_blocks_instead_of_flipping_direction(self):
        x=np.zeros((101,2));x[:,0]=np.arange(101)*.001;active=np.ones_like(x,bool)
        m=dict(beta=1.,equilibrium=0.,rho=.999)
        r=trend_guard(x,active,100,(0,1),m)
        self.assertFalse(r['eligible']);self.assertEqual(r['reason'],'persistent_divergence')
        # A fresh six-hour reversal removes the persistence veto without itself
        # granting formation/cost/risk approval.
        x[95:,0]=x[94,0]-np.arange(1,7)*.001
        self.assertTrue(trend_guard(x,active,100,(0,1),m)['eligible'])

    def test_stationary_or_converging_spread_is_not_called_persistent(self):
        x=np.zeros((101,2));x[:,0]=np.linspace(.2,.1,101)
        r=trend_guard(x,np.ones_like(x,bool),100,(0,1),dict(beta=1.,rho=.99,equilibrium=0.))
        self.assertTrue(r['eligible'])

    def test_stale_regime_prices_fail_closed(self):
        x=np.zeros((101,2));active=np.ones_like(x,bool);active[94,0]=False
        r=trend_guard(x,active,100,(0,1),dict(beta=1.,rho=.99,equilibrium=0.))
        self.assertFalse(r['eligible']);self.assertEqual(r['reason'],'regime_prices_not_fresh')

    def test_completed_prefix_and_all45_gate_preserve_forecasts_for_inventory(self):
        last=self.i+2;d={k:self.d[k][:last+1] for k in ('market_times','close','volume','trades')}
        states,_,counts=build(d,int(d['market_times'][self.i]))
        for arm,s in states.items():
            self.assertEqual(s['mu'].shape,(last+1,45))
            denied=(~s['relationship_allowed']|~s['regime_allowed'])&s['economic_available']
            self.assertTrue(np.all(s['economic_action'][denied]==0))
            self.assertTrue(np.isfinite(s['mu'][denied]).all())
            self.assertTrue(np.all(s['signal_times']<=d['market_times'][:,None]))
        self.assertGreater(counts['stable_regime_72']['relationship_reject'],0)

if __name__=='__main__':unittest.main()
