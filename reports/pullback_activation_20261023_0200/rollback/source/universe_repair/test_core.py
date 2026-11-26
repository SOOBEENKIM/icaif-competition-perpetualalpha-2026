import unittest
import numpy as np
from .core import calculate, REVISION
from candidate_handoff.portfolio import target_from_frame
from competition_live.profile import load_profile


def fixture(signs, n=24*105):
    t=1775001600+np.arange(n)*3600
    price=100*np.exp(np.arange(n)[:,None]*.012/24*np.asarray(signs)[None,:])
    return dict(market_times=t,known=t,times=t-3600,close=price,open=price,
        high=price*1.001,low=price*.999,volume=np.full_like(price,1000.),
        trades=np.ones_like(price),funding=np.zeros_like(price),observed=np.ones_like(price,bool))


class UniverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=fixture([1,-1]*5);cls.f=np.zeros_like(cls.d['close']);cls.p=np.tile([1.,0.]*5,(len(cls.f),1))
        cls.w,cls.g,cls.parts=calculate(cls.d,cls.f,cls.p)

    def test_all_ten_assets_have_core_signal_and_matching_signed_targets(self):
        np.testing.assert_array_equal(self.parts['allowed'],True)
        np.testing.assert_array_equal(np.sign(self.w[-1]),[1,-1]*5)

    def test_symbol_permutation_cannot_privilege_crypto(self):
        ix=np.array([9,7,5,3,1,8,6,4,2,0])
        d={k:v[:,ix] if v.ndim==2 else v for k,v in self.d.items()}
        w,_,_=calculate(d,self.f[:,ix],self.p[:,ix])
        np.testing.assert_allclose(w,self.w[:,ix],rtol=1e-10,atol=1e-12)

    def test_appended_future_does_not_change_any_past_target(self):
        stop=24*90+7;d={k:v[:stop] for k,v in self.d.items()}
        w,gate,_=calculate(d,self.f[:stop],self.p[:stop])
        np.testing.assert_allclose(w,self.w[:stop],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(gate['protected'],self.g['protected'][:stop])

    def test_ten_core_targets_reach_live_allocation_without_pair_membership_filter(self):
        f=dict(anchor_weights=self.w[-1].tolist(),protected=self.g['protected'][-1].tolist(),
            price_forecast_168h=[0.]*10,reference_funding_forecast_168h=[0.]*10,
            risk_aversion=5.,sigma=(np.eye(10)*1e-8).tolist(),volume=[1000.]*10,trades=[1.]*10)
        action,_=target_from_frame(f,np.full(10,100.),np.zeros(10),50000.,load_profile()['trading'])
        np.testing.assert_array_equal(np.sign(action['target_qty']),[1,-1]*5)

    def test_zero_signal_does_not_force_a_diversified_portfolio(self):
        d=fixture([0]*10);zero=np.zeros_like(d['close'])
        w,_,_=calculate(d,zero,np.full_like(zero,.5))
        np.testing.assert_array_equal(w,0.)


if __name__=='__main__':unittest.main()
