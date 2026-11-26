import copy,unittest
import numpy as np
from .policy import CashflowController
from pas_performance_research.funding import forecast
from competition_live.profile import load_profile
from replay_lab.common_comparison.engine import risk_breaches

class Contracts(unittest.TestCase):
    def test_future_settlements_do_not_change_previous_predictions(self):
        t=np.arange(36)*3600
        obs=[dict(known=h*3600+60,symbol='BTC',rate=-.0001) for h in range(36)]
        a,n=forecast(t,obs,['BTC'],horizon=72,decay=True)
        altered=copy.deepcopy(obs)
        for r in altered:
            if r['known']>18*3600:r['rate']=.1
        b,_=forecast(t,altered,['BTC'],horizon=72,decay=True)
        np.testing.assert_array_equal(a[:19],b[:19]);self.assertEqual(n[1,0],1)

    def make(self,carry,mode='funding_basket72',fresh=True):
        p=load_profile()['trading'];n=10;t=21600
        c=CashflowController([t],np.zeros((1,n)),np.asarray(carry)[None,:],np.full((1,n),fresh),mode)
        prices=np.full(n,100.);sig=np.eye(n)*1e-6
        result=c.propose(t,prices,np.zeros(n),50000.,np.zeros(n),sig,np.ones(n),np.ones(n),p)
        return c,np.array(result['target_qty']),p,prices,sig

    def test_symmetric_sides_and_final_risk_caps(self):
        carry=np.linspace(-.04,.04,10)
        a,q,p,prices,sig=self.make(carry);b,z,*_=self.make(-carry)
        np.testing.assert_allclose(q,-z,atol=1e-8)
        self.assertTrue(np.any(q>0) and np.any(q<0))
        self.assertFalse(risk_breaches(q,50000.,prices,sig,p['risk']))
        self.assertLessEqual(abs(q*prices).sum()/50000.,p['risk']['normal_turnover_fraction']+1e-8)

    def test_stale_market_cannot_open_and_funding_below_cost_is_rejected(self):
        _,q,*_=self.make(np.ones(10)*.04,fresh=False);self.assertFalse(np.any(q))
        _,q,*_=self.make(np.ones(10)*.0001);self.assertFalse(np.any(q))

    def test_all_45_pairs_considered_without_symbol_whitelist(self):
        c,q,*_=self.make(np.linspace(-.04,.04,10),mode='funding_pairs72')
        self.assertEqual(len(c.trace[-1]['pairs']),45)
        self.assertEqual(len({tuple(r['ids']) for r in c.trace[-1]['pairs']}),45)

if __name__=='__main__':unittest.main()
