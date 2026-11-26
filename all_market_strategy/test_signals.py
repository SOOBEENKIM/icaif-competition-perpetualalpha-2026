import json
import unittest
from pathlib import Path

import numpy as np

from .signals import ROOT, SYMBOLS, build


class SignalUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pointer=json.loads((ROOT/'reports/all_markets_extension_20261020/input_pointer.json').read_text())
        run=ROOT/pointer['run']
        with np.load(run/'inputs.npz',allow_pickle=False) as z: cls.data={k:z[k].copy() for k in z.files}
        with np.load(run/'targets.npz',allow_pickle=False) as z:
            cls.forecast=z['chronos_forecast'].copy(); cls.probability=z['frozen_regime_probability'].copy()
            cls.legacy=z['default'].copy()
        cls.weights,_=build(cls.data,cls.forecast,cls.probability)

    def test_old_scope_reproduces_frozen_entire_history(self):
        w,_=build(self.data,self.forecast,self.probability,scope='legacy')
        np.testing.assert_allclose(w,self.legacy,rtol=1e-11,atol=1e-13)

    def test_appending_future_cannot_change_past_targets(self):
        stop=4001
        data={k:v[:stop].copy() for k,v in self.data.items()}
        w,_=build(data,self.forecast[:stop],self.probability[:stop])
        np.testing.assert_allclose(w,self.weights[:stop],rtol=1e-11,atol=1e-13)

    def test_asset_order_does_not_choose_privileged_markets(self):
        order=np.array([7,4,9,0,6,2,8,1,5,3])
        d={k:(v[:,order].copy() if v.ndim==2 and v.shape[1]==10 else v.copy()) for k,v in self.data.items()}
        w,_=build(d,self.forecast[:,order],self.probability[:,order])
        np.testing.assert_allclose(w,self.weights[:,order],rtol=1e-10,atol=1e-12)

    def test_identical_eligible_trends_trade_in_every_column(self):
        n=24*100; t=1775001600+np.arange(n)*3600
        price=100*np.exp(np.arange(n)*.012/24)
        matrix=np.repeat(price[:,None],10,axis=1)
        d=dict(market_times=t,known=t,times=t-3600,close=matrix,open=matrix,
               high=matrix*1.001,low=matrix*.999,volume=np.ones_like(matrix)*1000,
               trades=np.ones_like(matrix),funding=np.zeros_like(matrix),observed=np.ones_like(matrix,dtype=bool))
        w,_=build(d,np.zeros_like(matrix),np.ones_like(matrix))
        self.assertTrue(np.all(w[-1]>0))
        np.testing.assert_allclose(w[-1],np.repeat(w[-1,0],10),atol=1e-12)

    def test_flat_prices_do_not_force_any_purchase(self):
        n=24*100; t=1775001600+np.arange(n)*3600; p=np.full((n,10),100.)
        d=dict(market_times=t,known=t,times=t-3600,close=p,open=p,high=p,low=p,
               volume=np.zeros_like(p),trades=np.zeros_like(p),funding=np.zeros_like(p),observed=np.ones_like(p,dtype=bool))
        w,_=build(d,np.zeros_like(p),np.ones_like(p))
        np.testing.assert_array_equal(w,np.zeros_like(w))


if __name__=='__main__': unittest.main()
