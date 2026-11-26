import unittest
from unittest.mock import patch
import numpy as np
from pas_strategy_audit import test_execution as fixture
from .guarded_portfolio import GuardedAccount,adopt,REVISION
from candidate_handoff import test_risk_priority as fixtures
from competition_live.profile import START

class GuardTests(unittest.TestCase):
    def test_price_cap_does_not_follow_a_more_expensive_mark(self):
        helper=fixture.ExecutionTests();a,p,t,orders,stream,_=helper.setup_case(GuardedAccount)
        a.base_entry_anchor=np.full(10,80000.)
        for sign in [1,-1]:
            src=80800. if sign>0 else 79200.
            packet=a.seal(dict(packet_id='base:'+str(sign),source_prices=[src]*10,
                owner_targets={'base':[sign*.01]+[0.]*9,'pair':[0.]*10}),p.p,p.sc)
            self.assertAlmostEqual(packet['sealed_owner_limits'][('base',0)],80000*(1+sign*.001))

    def test_no_decision_anchor_can_never_create_new_exposure(self):
        helper=fixture.ExecutionTests();a,p,*_=helper.setup_case(GuardedAccount)
        with self.assertRaises(ValueError):a.seal(dict(packet_id='base:1',source_prices=[80000.]*10,
            owner_targets={'base':[.01]+[0.]*9,'pair':[0.]*10}),p.p,p.sc)

    def test_risk_exit_needs_no_entry_anchor_and_has_no_price_veto(self):
        helper=fixture.ExecutionTests();a,p,t,*_=helper.setup_case(GuardedAccount)
        a.physical_fill(0,.01,80000.,0.,t[0]);a.owner_fill('base',0,.01,80000.,0.,t[0],'seed')
        r=a.seal(dict(packet_id='base:exit',source_prices=[80000.]*10,
            owner_targets={'base':[0.]*10,'pair':[0.]*10}),p.p,p.sc)
        self.assertIsNone(r['sealed_owner_limits'][('base',0)])

    def test_actual_coordinator_chain_saves_price_and_preserves_remaining_orders(self):
        p,a,t,orders,stream,packet,target,profile=fixtures.RiskPriorityTests().setup_case()
        p=adopt(p);a=p.account;a.refresh_market=stream
        with patch('execution_revision_20261024.portfolio.target_from_frame',return_value=(dict(target_qty=target.tolist()),{})),patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)
            np.testing.assert_array_equal(p.base_task['decision_prices'],np.full(10,80000.))
            t[0]=START+1801
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)
        self.assertEqual([x.coin for x in orders],['pas:BTC','pas:ETH'])
        self.assertAlmostEqual(a.physical.qty[1],-.01)
        self.assertIsNone(a.base_entry_anchor)
        event=next(e for e in p.events if e['kind']=='budget_lifecycle_decision')
        self.assertEqual(event['execution_revision'],REVISION)

if __name__=='__main__':unittest.main()
