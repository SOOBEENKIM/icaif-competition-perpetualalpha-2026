import unittest
import numpy as np
from types import SimpleNamespace
from .models import segmented_area,conditional,stopping_tables,estimate
from .policy import PairsController

class Tests(unittest.TestCase):
    def test_segmented_unsigned_area_not_cancelling(self):
        p=np.array([[0.,0.],[1.,1.],[2.,0.],[3.,-1.],[4.,0.]])
        self.assertAlmostEqual(segmented_area(p),2.)
        self.assertAlmostEqual(segmented_area(p+10),2.)
        self.assertAlmostEqual(segmented_area(2*p),8.)

    def test_copula_independence_and_swap(self):
        c=dict(df=None,rho=0.,margins=[np.arange(100.),np.arange(100.)])
        h=conditional(c,[10.,90.]);self.assertLess(h[0],.15);self.assertGreater(h[1],.85)
        c['df']=5;c['rho']=.7
        np.testing.assert_allclose(conditional(c,[10.,90.]),conditional(c,[90.,10.])[::-1])

    def test_stopping_terminal_cost_and_no_profit_without_motion(self):
        f=dict(rho=.95,sd=.01,beta=1.)
        table=stopping_tables(f,.002,12)
        np.testing.assert_allclose(table['transition'].sum(1),1.)
        np.testing.assert_allclose(table['hold'][0],-.002)
        f['sd']=0.;flat=stopping_tables(f,.002,12)
        np.testing.assert_allclose(flat['flat'],0.)
        self.assertTrue(np.all(flat['entry'][1:]<0))

    def test_cointegrated_synthetic_levels(self):
        rng=np.random.default_rng(12);x=np.cumsum(rng.normal(0,.01,1440));e=np.zeros(1440)
        for i in range(1,len(e)):e[i]=.92*e[i-1]+rng.normal(0,.001)
        f=estimate(3+1.3*x+e,x,np.ones(1440,bool))
        self.assertLess(f['p'],.01);self.assertAlmostEqual(f['beta'],1.3,places=1)
        self.assertGreater(f['half_life'],3);self.assertLess(f['half_life'],30)

    def test_partial_pair_is_closed_not_treated_as_hedged(self):
        import json
        from pathlib import Path
        cfg=json.loads((Path(__file__).parent/'protocol.json').read_text())
        prices=np.tile([100.,50.],(200,1));m=SimpleNamespace(close=prices,funding=np.zeros_like(prices),exact=lambda t:int(t//3600))
        c=PairsController(m,[],cfg,'ou',dict(impact_bps=10.,participation=.01),199)
        f=dict(key='A/B',assets=[0,1],kind='log',alpha=float(np.log(2)),beta=1.,mean=0.,sd=.01,rho=.95,relative_funding=0.)
        c.states['A/B']=dict(model=f,direction=1,entered=170,pending=True,closing=False,target=np.array([10.,-20.]))
        p=dict(risk=dict(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,asset_notional=150000.,daily_volatility=.005),execution=dict(decimals=[4,4]))
        action=c.propose(170*3600,prices[170],np.array([10.,0.]),1e6,np.zeros(2),np.eye(2)*1e-6,np.ones(2),np.ones(2),p)
        np.testing.assert_array_equal(action['target_qty'],[0.,0.])
        self.assertEqual(c.events[0]['event'],'partial_entry_abort')

if __name__=='__main__':unittest.main()
