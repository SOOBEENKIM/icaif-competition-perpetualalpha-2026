import unittest
import numpy as np
from .model import features,rolling_forecast,symmetric_state,fit,targets
from competition_live.profile import load_profile


class CausalAllMarketTests(unittest.TestCase):
    def test_every_asset_can_enter_long_or_short_and_cannot_reverse_same_step(self):
        old=np.zeros(10)
        want=np.array([1,-1]*5)
        opened=symmetric_state(old,want,np.ones(10,bool))
        np.testing.assert_array_equal(opened,want)
        closed=symmetric_state(opened,-want,np.ones(10,bool))
        np.testing.assert_array_equal(closed,0)
        reversed_=symmetric_state(closed,-want,np.ones(10,bool))
        np.testing.assert_array_equal(reversed_,-want)

    def test_freshness_blocks_entry_but_does_not_force_unpriced_reversal(self):
        result=symmetric_state(np.zeros(10),np.ones(10),np.zeros(10,bool))
        np.testing.assert_array_equal(result,0)

    def test_maturity_guard_rejects_current_or_future_label(self):
        spec=dict(minimum_rows=2,half_life_days=45,ridge_penalty=.1,target_clip=5)
        with self.assertRaisesRegex(ValueError,"Unmatured"):
            fit(np.ones((2,3)),np.ones(2),np.array([0,1]),np.array([10,11]),11,spec)

    def test_cost_gate_rejects_forecasts_below_roundtrip_cost_in_all_columns(self):
        n=700;t=1775001600+np.arange(n)*3600
        d=dict(close=np.ones((n,10))*100,market_times=t)
        f=dict(x=np.ones((n,10,8)),unit=np.ones((n,10))*.02,
               fresh=np.ones((n,10),bool),funding=np.zeros((n,10)))
        policy=dict(minimum_warmup_hours=672,decision_hours=6,daily_risk_budget=.0006,
                    entry_signal=.25,exit_signal=.1,roundtrip_cost=.0038)
        sigma=np.repeat((np.eye(10)*1e-6)[None],n,axis=0)
        w,_=targets(d,f,np.ones((n,10))*.003,sigma,load_profile()["trading"],policy,"pooled_ridge")
        np.testing.assert_array_equal(w,0)

    def test_features_are_asset_permutation_equivariant(self):
        rng=np.random.default_rng(191);n=800
        close=100*np.exp(np.cumsum(rng.normal(0,.005,(n,10)),axis=0))
        d=dict(close=close,volume=np.ones_like(close),trades=np.ones_like(close),
               funding=np.zeros_like(close))
        perm=rng.permutation(10)
        a=features(d);b=features({k:v[:,perm] for k,v in d.items()})
        for key in a:np.testing.assert_allclose(a[key][:,perm],b[key],equal_nan=True,atol=1e-12)


if __name__=="__main__":
    unittest.main()
