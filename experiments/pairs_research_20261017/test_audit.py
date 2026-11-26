import unittest,copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from replay_lab.common_comparison.engine import Ledger
from .partial import fit_partial,filter_step,PartialController

class AuditTests(unittest.TestCase):
    def test_two_leg_fee_denominator(self):
        ledger=Ledger(10000.,('A','B'))
        ledger.fill(0,1,100.,9,1);ledger.fill(1,-2,50.,9,1)
        ledger.fill(0,-1,100.,9,2);ledger.fill(1,2,50.,9,2)
        self.assertAlmostEqual(ledger.fees,.36)
        self.assertAlmostEqual(ledger.fees/200*10000,18.)
        self.assertAlmostEqual(ledger.reconcile(np.array([100.,50.])),9999.64)

    def test_partial_estimation_on_known_latent_process(self):
        rng=np.random.default_rng(732);r=np.cumsum(rng.normal(0,.0003,1440));v=np.zeros(1440)
        for i in range(1,len(v)):v[i]=.95*v[i-1]+rng.normal(0,.002)
        f=fit_partial(r+v)
        self.assertTrue(f['converged']);self.assertGreater(f['mr_variance_fraction'],.8)
        self.assertLess(abs(f['rho']-.95),.08)
        self.assertLess(f['aic'],f['rw_aic'])
        state,cov=filter_step(f['fitted_state'],f['fitted_cov'],float(r[-1]+v[-1]),f)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
        self.assertAlmostEqual(state.sum(),r[-1]+v[-1],places=6)

    def test_future_changes_cannot_change_filtered_signals(self):
        cfg=json.loads((Path(__file__).parent/'protocol.json').read_text())
        t=np.arange(500);prices=np.c_[100*np.exp(.01*np.sin(t/12)),np.full(500,50.)]
        first=SimpleNamespace(close=prices.copy(),funding=np.zeros_like(prices),volume=np.ones_like(prices),trades=np.ones_like(prices))
        changed=copy.deepcopy(first);changed.close[301:]*=np.array([2.,.3])
        model=dict(key='A/B',assets=[0,1],kind='log',alpha=float(np.log(2)),beta=1.,mean=0.,sd=.01,rho=.95,
            relative_funding=0.,formation_index=200,trained_through=199*3600,q_rw=1e-7,q_mr=1e-5,
            fitted_state=np.zeros(2),fitted_cov=np.eye(2)*1e-5)
        controllers=[PartialController(m,[],cfg,'stopping',dict(impact_bps=10.,participation=.01),499) for m in (first,changed)]
        for i in range(200,301):
            aa,bb=[c.signal(model,i) for c in controllers]
            self.assertEqual(aa['direction'],bb['direction'])
            self.assertEqual(aa['z'],bb['z'])
            self.assertEqual(aa['edge'],bb['edge'])

if __name__=='__main__':unittest.main()
