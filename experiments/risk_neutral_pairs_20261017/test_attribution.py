import unittest
import numpy as np
from .attribution import episode_attribution


class AttributionTests(unittest.TestCase):
    def test_zero_physical_position_still_attributes_offsetting_owner_funding(self):
        t=np.array([0,3600,7200]);p=np.full((3,2),100.)
        d=dict(times=t,par_close=p,reference_close=p,funding=np.tile([.01,.02],(3,1)))
        fills=[dict(time=now,asset=j,signed_qty=q,price=100.,fee=1.) for now,qs in [(3600,[1.,-1.]),(7200,[-1.,1.])] for j,q in enumerate(qs)]
        run=dict(end=7200,events=[dict(time=0,event='pair_intent',pair=['A','B'],target=[1,-1]),dict(time=7200,event='closed')],
            owner_fills={'pair':fills},metrics={'pair_owned_pnl':-3.},
            snapshots=[dict(time=int(now),pair_qty=q,qty=[0,0]) for now,q in zip(t,[[0,0],[1,-1],[0,0]])])
        a=episode_attribution(run,d,dict(funding_receipt_multiplier=.5));e=a['episodes'][0]
        self.assertEqual(e['funding'],1.);self.assertEqual(e['fees'],4.);self.assertEqual(e['net_pnl'],-3.)
        self.assertTrue(e['closed']);self.assertEqual(a['accounting_error'],0)


if __name__=='__main__':unittest.main()
