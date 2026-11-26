import json,unittest
import numpy as np
from .research import data,ts,HERE
from .diversified import paths
from .band import InwardBandController
from experiments.risk_preserving_20261016.policy import RiskCapController
from replay_lab.common_comparison.engine import risk_breaches


class BandInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inputs,cls.a,cls.p,cls.market=data()

    def test_binding_cap_is_never_raised_and_only_reduces_inventory(self):
        t=ts('2026-09-01');i=self.market.exact(t);prices=self.market.close[i]
        raw=np.zeros(10);raw[0]=.03;qty=np.zeros(10);qty[0]=.031*1e6/prices[0]
        retention=np.ones(10);retention[0]=.75
        agent=InwardBandController([t],[retention])
        action=agent.propose(t,prices,qty,1e6,raw,self.a['sigma'][i],self.market.volume[i],self.market.trades[i],self.p)
        self.assertTrue(action['band_binding'][0])
        self.assertLessEqual(action['target_qty'][0],.95*action['risk_cap_qty'][0])
        self.assertLessEqual(action['target_qty'][0],qty[0])
        self.assertFalse(risk_breaches(action['target_qty'],1e6,prices,self.a['sigma'][i],self.p['risk']))

    def test_hard_halt_and_duplicate_round_are_preserved(self):
        t=ts('2026-09-01');i=self.market.exact(t);q=np.ones(10)
        a=InwardBandController([t],[np.ones(10)])
        args=(t,self.market.close[i],q,1e6,np.ones(10)*.03,self.a['sigma'][i],self.market.volume[i],self.market.trades[i],self.p)
        action=a.propose(*args,halted=True)
        np.testing.assert_array_equal(action['target_qty'],np.zeros(10))
        self.assertIsNone(a.propose(*args,halted=True))

    def test_nonbinding_ceiling_matches_retained_controller(self):
        t=ts('2026-09-01');i=self.market.exact(t);q=np.zeros(10);raw=np.zeros(10);raw[0]=.03
        args=(t,self.market.close[i],q,1e6,raw,self.a['sigma'][i],self.market.volume[i],self.market.trades[i],self.p)
        a=InwardBandController([t],[np.ones(10)]).propose(*args)
        b=RiskCapController().propose(*args,retention=np.ones(10))
        np.testing.assert_array_equal(a['target_qty'],b['target_qty'])

    def test_diversified_state_is_causal(self):
        c=json.loads((HERE/'diversified_protocol.json').read_text());n=self.market.exact(ts('2026-08-01'))+1
        for lookbacks in c['lookbacks_hours'].values():
            full,_=paths(self.inputs,lookbacks,c)
            part,_=paths({k:v[:n] for k,v in self.inputs.items()},lookbacks,c)
            np.testing.assert_array_equal(full[:n],part)


if __name__=='__main__':unittest.main()
