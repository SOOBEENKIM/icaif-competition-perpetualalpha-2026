import copy
import unittest
from unittest.mock import patch
import numpy as np
from candidate_handoff.test_risk_priority import RiskPriorityTests
from competition_live.profile import START
from .portfolio import ExecutionPortfolio,adopt,reduction_target

class ExecutionTests(unittest.TestCase):
    def case(self,held=.01,cap=.005):
        case=RiskPriorityTests().setup_case(held,cap)
        case[0].__class__=ExecutionPortfolio
        return case

    def tick(self,case,target=None):
        p,a,clock,orders,stream,packet,desired,profile=case
        with patch('execution_revision_20261024.portfolio.target_from_frame',return_value=(dict(target_qty=(desired if target is None else target).tolist()),{})),patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            p.step_candidate(clock[0],{s:stream(s) for s in p.p['symbols']},packet,profile)

    def test_normal_remainder_can_fill_after_fifteen_minutes(self):
        case=self.case();p,a,t,orders,*_=case
        self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertEqual(p.base_task['expires'],START+3600)
        t[0]=START+1801;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:ETH'])
        self.assertAlmostEqual(a.physical.qty[1],-.01)
        t[0]+=20;self.tick(case);self.assertEqual(len(orders),2)

    def test_no_new_exposure_after_hour(self):
        case=self.case();p,a,t,orders,*_=case
        self.tick(case);t[0]=START+3601;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertEqual(a.physical.qty[1],0.)
        self.assertIsNone(p.base_task)

    def test_only_remaining_own_reduction_survives_deadline(self):
        case=self.case(cap=.01);p,a,t,orders,stream,packet,target,profile=case
        # Suppress a genuinely requested exit using absent executable depth.
        a.refresh_market=lambda s:None if s=='BTC' else stream(s)
        target=target.copy();target[0]=0.;target[1]=0.
        self.tick(case,target);self.assertEqual(orders,[])
        t[0]=START+3601;a.refresh_market=stream;self.tick(case,target)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertAlmostEqual(a.physical.qty[0],0.)
        t[0]+=20;self.tick(case,target);self.assertEqual(len(orders),1)

    def test_expired_reverse_target_closes_without_flipping(self):
        case=self.case(cap=.01);p,a,t,orders,stream,packet,target,profile=case
        p.last_decision=START
        p.base_task=dict(target=np.array([-.02]+[0.]*9),expires=START+3600,entry_expires=START+3600,intent_source=START)
        t[0]=START+3700;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertAlmostEqual(a.physical.qty[0],0.)

    def test_unknown_outcome_still_blocks_all_orders(self):
        case=self.case();p,a,t,orders,*_=case
        self.tick(case);a.unknown.add('BTC');t[0]=START+1800;self.tick(case)
        self.assertEqual(len(orders),1)

    def test_forced_halt_closes_instead_of_continuing(self):
        case=self.case();p,a,t,orders,*_=case
        self.tick(case);p.halted=True;t[0]+=20;self.tick(case)
        self.assertAlmostEqual(a.physical.qty[0],0.)
        self.assertEqual(a.physical.qty[1],0.)
        self.assertIsNone(p.base_task)

    def test_missing_signal_does_not_extend_addition_deadline(self):
        case=self.case();p,a,t,orders,stream,packet,target,profile=case
        self.tick(case);t[0]+=20
        with patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},None,profile)
        self.assertIsNone(p.base_task)
        self.assertAlmostEqual(a.physical.qty[0],0.)

    def test_adoption_preserves_owned_and_reserved_inventory(self):
        case=self.case();p,a,t,orders,*_=case
        a.reserved.qty[0]=-.03
        self.tick(case);saved=copy.deepcopy(p)
        new=adopt(p)
        for field in ['qty','entry','cash','realized','fees','funding']:
            np.testing.assert_equal(getattr(new.account.physical,field),getattr(saved.account.physical,field))
            np.testing.assert_equal(getattr(new.account.reserved,field),getattr(saved.account.reserved,field))
        self.assertEqual(new.last_decision,saved.last_decision)
        self.assertEqual(new.counter,saved.counter)
        self.assertIsNone(new.base_task)

    def test_reduction_projection_is_symmetric_and_never_increases_exposure(self):
        held=np.array([2.,-3.,0.,2.,-3.]);want=np.array([-1.,1.,4.,5.,-7.])
        np.testing.assert_array_equal(reduction_target(held,want),[0,0,0,2,-3])
        np.testing.assert_array_equal(reduction_target(-held,-want),-reduction_target(held,want))

if __name__=='__main__':unittest.main()
