import unittest
import numpy as np
from .model_v2 import map_state
from experiments.event_value_pairs_20261017.mapping_v2 import map_state as earlier_map


class PairIdentityTests(unittest.TestCase):
    def setUp(self):
        self.ref=dict(market_times=np.arange(26)*3600,close=np.tile([100.,200.,50.],(26,1)))
        self.raw=dict(pairs=[[0,1],[0,2]],beta=np.ones((26,2)),sd=np.full((26,2),.01),delayed=np.ones((26,2),int),
            z=np.full((26,2),-2.9),confirmed=np.tile([False,True],(26,1)))
        self.state=dict(pairs=np.array([[0,2],[0,1]]),directions=np.ones((1,2),int),signal_times=np.full((1,2),86400),
            terminal_prices=np.array([[[102.,49.],[102.,196.]]]),beta=np.ones((1,2)))
        self.models={('BTC','HYPE'):dict(fit_time=1,last_maturity=0,events=17,scalar=dict(ratio=.5),ridge=None),
            ('BTC','ETH'):dict(fit_time=1,last_maturity=0,events=50,scalar=dict(ratio=.25),ridge=None)}

    def test_different_venue_order_uses_correct_source_prices_and_features(self):
        d,a=map_state(self.state,np.array([86400]),self.ref,self.raw,self.models,'scalar',[['BTC','HYPE'],['BTC','ETH']])
        np.testing.assert_array_equal(d['terminal_prices'][0],[[101.,49.5],[100.5,199.]])
        self.assertTrue(all(x['full_gap']>0 for x in a))

    def test_source_reordering_is_invariant(self):
        for m in self.models.values():
            m['ridge']=dict(center=[0.]*5,scale=[1.]*5,coefficient=[0.,0.,0.,0.,0.,1.],feature_clip=8.)
        x,_=map_state(self.state,np.array([86400]),self.ref,self.raw,self.models,'ridge',[['BTC','HYPE'],['BTC','ETH']])
        r={k:v[:,::-1] if isinstance(v,np.ndarray) else v[::-1] for k,v in self.raw.items()}
        y,_=map_state(self.state,np.array([86400]),self.ref,r,self.models,'ridge',[['BTC','HYPE'],['BTC','ETH']])
        for k in x:np.testing.assert_array_equal(x[k],y[k])
        np.testing.assert_array_equal(x['directions'],[[1,0]])

    def test_missing_source_pair_is_an_explicit_failure(self):
        raw={k:v[:,:1] if isinstance(v,np.ndarray) else v[:1] for k,v in self.raw.items()}
        with self.assertRaises(ValueError):map_state(self.state,np.array([86400]),self.ref,raw,self.models,'scalar',[['BTC','HYPE'],['BTC','ETH']])

    def test_prior_experiment_confirmation_uses_identity_as_well(self):
        d,a=earlier_map(self.state,np.array([86400]),self.ref,self.raw,[],'confirmed')
        np.testing.assert_array_equal(d['directions'],[[1,0]])
        fits=[dict(time=1,last_maturity=0,groups={'True':dict(ratio=.5),'False':dict(ratio=.25)})]
        d,a=earlier_map(self.state,np.array([86400]),self.ref,self.raw,fits,'calibrated')
        np.testing.assert_array_equal(d['terminal_prices'][0],[[101.,49.5],[100.5,199.]])


if __name__=='__main__':unittest.main()
