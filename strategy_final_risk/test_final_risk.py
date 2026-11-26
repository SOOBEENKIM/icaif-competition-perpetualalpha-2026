import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

# Initialize the same frozen imports as production before candidate modules.
from competition_live.profile import START
from pas_strategy_audit.test_execution import ExecutionTests
from competition_live.live_account import LiveAccount
from experiments.published_pair_lifecycle_20261017.signals import validate_envelope
from .model import final_mix
from .candidate import publication,validate,build


class FinalRiskTests(unittest.TestCase):
    def test_reversion_is_subject_to_final_jm_and_chronos_filters(self):
        n=1450
        market=SimpleNamespace(times=np.arange(n)*3600,close=np.full((n,10),100.),
            funding=np.zeros((n,10)),trades=np.ones((n,10)),volume=np.ones((n,10)))
        trend=np.zeros((n,10));reversion=np.zeros_like(trend);reversion[:,2]=.01
        probability=np.zeros_like(trend);forecast=np.zeros_like(trend)
        weights,trace=final_mix(market,trend,reversion,1.,probability,forecast)
        self.assertAlmostEqual(weights[-1,2],.0075) # JM half + Chronos full, averaged.
        forecast[:,2]=-.01
        weights,trace=final_mix(market,trend,reversion,1.,probability,forecast)
        self.assertAlmostEqual(weights[-1,2],.005)
        self.assertTrue(trace['protected'][-1,2])
        self.assertTrue(np.all(abs(weights)<=abs(trace['pre_gate'])+1e-12))

    def event(self,clock,weights):
        return publication(dict(channel='strategySignal',received_ms=int(clock*1000)+1,
            data=dict(frame=dict(source_time=START,published_time=clock,
                weights=weights,retention=[1.]*10,sigma=(np.eye(10)*1e-8).tolist(),
                volume=[1000.]*10,trades=[10.]*10,source_hash='fixture'))),
            weights,np.array([True]+[False]*9))

    def test_wire_digest_and_execution_targets_are_both_verified(self):
        event=self.event(START+100,[.01]+[0.]*9)
        validate_envelope(event);validate(event)
        for location in ('caps','weights','execution','mask'):
            bad=copy.deepcopy(event)
            if location=='caps':bad['data']['risk_contract']['weight_caps'][0]=.02
            elif location=='weights':bad['data']['frame']['weights'][0]=.02
            elif location=='execution':bad['data']['incumbent'][0]=.02
            else:bad['data']['risk_contract']['protected'][0]=False
            with self.assertRaises(ValueError):validate(bad)

    def test_small_cut_bypasses_buffer_and_rebalance_window_without_repeated_drift(self):
        fixture=ExecutionTests()
        a,_,clock,orders,stream,_=fixture.setup_case(LiveAccount)
        a.refresh_market=stream
        p=build(a.profile,lambda *args:None,a)['portfolio']
        fixture.execute(a,p,clock,stream,[.01]+[0.]*9,'seed')
        equity=a.reconcile(np.full(10,80000.))
        event=self.event(clock[0],[.009*80000/equity]+[0.]*9)
        p.last_decision=START
        clock[0]=START+2000
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            p.step(clock[0],{s:stream(s) for s in a.profile['trading']['symbols']},event,[])
        self.assertTrue(orders[-1].reduce_only)
        self.assertLessEqual(a.physical.qty[0],.009+1e-10)
        self.assertEqual(a.reserved.cash,50000)
        units=p.risk_cap_units.copy()
        p.protected_target(a.owners['base'].qty,np.full(10,90000.),equity+10)
        np.testing.assert_array_equal(p.risk_cap_units,units)


if __name__=='__main__':
    unittest.main()
