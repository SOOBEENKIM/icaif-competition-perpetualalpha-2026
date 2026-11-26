import unittest
import numpy as np
from .lifecycle import LifecycleController
from pas_adaptive_development.study import ts


class CandidateLifecycleTests(unittest.TestCase):
    def fixture(self,forecast,qty=None,protected=None,anchor=None):
        from competition_live.profile import load_profile
        t=ts('2026-08-01T12:00:00Z');n=10;p=load_profile()['trading']
        c=LifecycleController([t],np.asarray(forecast)[None,:],np.zeros((1,n)),
            np.zeros((1,n)) if anchor is None else np.asarray(anchor)[None,:],
            np.zeros((1,n),bool) if protected is None else np.asarray(protected)[None,:],5.)
        args=(t,np.full(n,100.),np.zeros(n) if qty is None else qty,50000.,np.zeros(n),
              np.eye(n)*.000001,np.ones(n),np.ones(n),p)
        return c,args

    def test_candidate_itself_all_ten_markets_both_directions(self):
        for j in range(10):
            for side in (-1,1):
                mu=np.zeros(10);mu[j]=side*.05;c,args=self.fixture(mu)
                self.assertGreater(side*c.propose(*args)['target_qty'][j],0.)

    def test_partial_inventory_held_without_replenishing_old_desired_size(self):
        mu=np.zeros(10);mu[1]=.05;qty=np.zeros(10);qty[1]=2.
        c,args=self.fixture(mu,qty);q=c.propose(*args)['target_qty']
        self.assertEqual(q[1],2.)

    def test_changed_thesis_closes_instead_of_instantly_flipping(self):
        mu=np.zeros(10);mu[1]=-.05;qty=np.zeros(10);qty[1]=2.
        c,args=self.fixture(mu,qty)
        self.assertEqual(c.propose(*args)['target_qty'][1],0.)

    def test_risk_gate_and_hard_caps_override_positive_thesis_hold(self):
        mu=np.full(10,.05);qty=np.zeros(10);qty[1]=50.
        mask=np.zeros(10,bool);mask[1]=True;anchor=np.zeros(10);anchor[1]=.02
        c,args=self.fixture(mu,qty,mask,anchor);q=c.propose(*args)['target_qty']
        self.assertLessEqual(abs(q[1]),10.)
        qty[1]=500.;c,args=self.fixture(mu,qty)
        self.assertLess(abs(c.propose(*args)['target_qty'][1]),500.)


if __name__=='__main__':unittest.main()
