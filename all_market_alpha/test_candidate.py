import hashlib
import json
import unittest
import numpy as np
from competition_live.profile import load_profile
from .candidate import publication,build_weights
from .model import targets


class CandidateTests(unittest.TestCase):
    def test_high_forecast_can_create_position_in_each_of_ten_assets(self):
        n=700;t=1775001600+np.arange(n)*3600
        d=dict(close=np.ones((n,10))*100,market_times=t)
        f=dict(x=np.ones((n,10,8)),unit=np.ones((n,10))*.02,
               fresh=np.ones((n,10),bool),funding=np.zeros((n,10)))
        policy=dict(minimum_warmup_hours=672,decision_hours=6,daily_risk_budget=.0006,
                    entry_signal=.25,exit_signal=.1,roundtrip_cost=.0038)
        sigma=np.repeat((np.eye(10)*1e-6)[None],n,axis=0)
        signs=np.array([1,-1]*5)
        forecast=np.tile(signs*.05,(n,1))
        w,_=targets(d,f,forecast,sigma,load_profile()["trading"],policy,"pooled_ridge")
        np.testing.assert_array_equal(np.sign(w[-1]),signs)

    def test_publication_preserves_availability_and_updates_digest(self):
        before=dict(received_ms=2000,data=dict(frame=dict(
            source_time=1,published_time=2,weights=[.1]*10),candidate=[.1]*10))
        after=publication(before,np.array([-.01]*10))
        self.assertEqual(after["received_ms"],before["received_ms"])
        self.assertEqual(after["data"]["frame"]["source_time"],1)
        self.assertEqual(before["data"]["frame"]["weights"],[.1]*10)
        expected=hashlib.sha256(json.dumps(after["data"]["frame"],sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()
        self.assertEqual(after["data"]["publication_sha256"],expected)
        self.assertEqual(after["data"]["frame"]["weights"],[-.01]*10)

    def test_risk_match_parameter_cannot_raise_leverage(self):
        d=dict(market_times=np.array([0,3600]),close=np.ones((2,10)))
        with self.assertRaisesRegex(ValueError,"may not increase"):
            build_weights(d,np.zeros((2,10)),np.zeros((2,10,10)),{}, {},
                          dict(agreement_scale=1.1,final_scale=1.))


if __name__=="__main__":
    unittest.main()
