"""Regressions through the real candidate -> coordinator -> fill adapter chain."""
import copy,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import test_handoff
from .portfolio import adopt,CandidatePortfolio
from .risk_priority import RiskPriorityPortfolio

from competition_live.profile import START
from competition_live.runtime import fingerprint

class RiskPriorityTests(unittest.TestCase):
    def setup_case(self,held=.01,cap=.005):
        helper=test_handoff.HandoffTests();helper.setUp()
        f,a,original,t,orders,stream=helper.fixture()
        if held:
            a.physical_fill(0,held,80000.,0.,START+1)
            a.owner_fill('base',0,held,80000.,0.,START+1,'seed')
        p=adopt(original);a=p.account;a.refresh_market=stream
        packet=helper.packet(t[0]);body=packet['body']
        body['frame']['protected'][0]=True
        body['frame']['anchor_weights'][0]=cap*80000/50000
        packet['sha256']=fingerprint(body)
        desired=np.array([cap,-.01]+[0.]*8)
        return p,a,t,orders,stream,packet,desired,helper.p

    def tick(self,case,desired=None):
        p,a,t,orders,stream,packet,target,profile=case
        action={'target_qty':(target if desired is None else desired).tolist()}
        with patch('candidate_handoff.portfolio.target_from_frame',return_value=(action,{})),patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)

    def test_material_risk_cut_retains_other_target_and_resumes_only_remaining_qty(self):
        case=self.setup_case();p,a,t,orders,*_=case
        self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertIsNotNone(p.base_task)
        self.assertEqual(p.base_task['expires'],START+900)
        self.assertEqual(p.last_decision,START)
        t[0]+=20;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:ETH'])
        self.assertAlmostEqual(a.physical.qty[1],-.01)
        t[0]+=20;self.tick(case)
        self.assertEqual(len(orders),2)

    def test_dust_cap_does_not_block_other_target_or_repeat_empty_packets(self):
        case=self.setup_case(cap=.00995);p,a,t,orders,*_=case
        self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:ETH'])
        self.assertAlmostEqual(a.physical.qty[0],.01)
        count=p.counter
        for _ in range(3):t[0]+=20;self.tick(case)
        self.assertEqual(p.counter,count)
        self.assertEqual(len(orders),1)
        self.assertEqual(len([e for e in p.events if e['kind']=='risk_cap_dust_deferred']),1)

    def test_expired_normal_intent_cannot_resume_after_reduction(self):
        case=self.setup_case();p,a,t,orders,*_=case
        self.tick(case);t[0]=START+901;self.tick(case)
        self.assertIsNone(p.base_task)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])

    def test_partial_risk_fill_retries_remainder_before_normal_addition(self):
        case=self.setup_case();p,a,t,orders,*_=case
        send=a.executor.submit;first=[True]
        def partial(order):
            if first[0]:first[0]=False;order=replace(order,size=str(float(order.size)/2))
            return send(order)
        a.executor.submit=partial
        self.tick(case);self.assertAlmostEqual(a.physical.qty[0],.0075)
        t[0]+=20;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:BTC'])
        t[0]+=20;self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:BTC','pas:ETH'])
        self.assertEqual(len({o.cloid for o in orders}),3)

    def test_unknown_outcome_blocks_all_continuation(self):
        case=self.setup_case();p,a,t,orders,*_=case
        self.tick(case);a.unknown.add('BTC');t[0]+=20;self.tick(case)
        self.assertEqual(len(orders),1)
        self.assertIn('unobserved_order_outcome',p.fatal)

    def test_drawdown_halt_still_cancels_intent_and_closes_position(self):
        case=self.setup_case(cap=.00995);p,a,t,orders,*_=case
        p.halted=True;self.tick(case)
        self.assertIsNone(p.base_task)
        self.assertEqual(len(orders),1);self.assertTrue(orders[0].reduce_only)
        self.assertAlmostEqual(a.physical.qty[0],0.)

    def test_dust_does_not_starve_existing_pair_lifecycle(self):
        case=self.setup_case(cap=.00995);p,a,t,orders,*_=case
        calls=[]
        p.pair=SimpleNamespace(state='holding',intent={'pair':['BTC','ETH']},events=[],ready_at=0,
            advance=lambda *args,**kwargs:calls.append(kwargs))
        self.tick(case)
        self.assertEqual(len(calls),1)
        self.assertEqual([o.coin for o in orders],['pas:ETH'])

    def test_genuine_physical_full_close_keeps_below_minimum_exception(self):
        case=self.setup_case(held=.00005,cap=0.);p,a,t,orders,*_=case
        self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])
        self.assertTrue(orders[0].reduce_only);self.assertAlmostEqual(a.physical.qty[0],0.)

    def test_virtual_close_that_increases_team_net_has_no_minimum_exception(self):
        case=self.setup_case(held=.00005,cap=0.);p,a,t,orders,*_=case
        a.reserved.qty[0]=-.0001
        self.tick(case)
        self.assertEqual([o.coin for o in orders],['pas:ETH'])
        self.assertAlmostEqual(a.physical.qty[0],.00005)

    def test_risk_cap_cannot_be_undone_by_saved_normal_target(self):
        case=self.setup_case();p,a,t,orders,*_=case
        target=case[6].copy();target[0]=.02
        self.tick(case,target)
        self.assertAlmostEqual(p.base_task['target'][0],.005)
        t[0]+=20;self.tick(case,target)
        self.assertAlmostEqual(a.physical.qty[0],.005)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:ETH'])

    def test_unfilled_material_reduction_preserves_task_and_blocks_risk_addition(self):
        case=self.setup_case();p,a,t,orders,stream,*_=case
        a.refresh_market=lambda symbol:None if symbol=='BTC' else stream(symbol)
        self.tick(case);t[0]+=20;self.tick(case)
        self.assertIsNotNone(p.base_task);self.assertEqual(orders,[])
        self.assertAlmostEqual(a.physical.qty[0],.01)


    def test_mixed_material_and_dust_targets_are_not_capped_twice(self):
        case=self.setup_case();p,a,t,orders,stream,packet,target,profile=case
        a.physical_fill(1,.00005,80000.,0.,START+1)
        a.owner_fill('base',1,.00005,80000.,0.,START+1,'seed-eth')
        a.reserved.qty[1]=-.0001
        packet['body']['frame']['protected'][1]=True
        packet['body']['frame']['anchor_weights'][1]=0.
        packet['sha256']=fingerprint(packet['body'])
        with patch.object(a,'execute_books',wraps=a.execute_books) as execute:
            self.tick(case)
        intent=execute.call_args.args[0]
        self.assertAlmostEqual(intent['owner_targets']['base'][1],.00005)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])

    def test_global_gross_breach_still_flattens_instead_of_resuming(self):
        case=self.setup_case();p,a,t,orders,*_=case
        p.p['risk']['gross_fraction']=.001
        self.tick(case)
        self.assertIsNone(p.base_task)
        self.assertAlmostEqual(a.physical.qty[0],0.)
        self.assertEqual([o.coin for o in orders],['pas:BTC'])

    def test_missing_signal_clears_pending_and_preserves_stale_signal_exit(self):
        case=self.setup_case();p,a,t,orders,stream,packet,target,profile=case
        self.tick(case);t[0]+=20
        with patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},None,profile)
        self.assertIsNone(p.base_task);self.assertAlmostEqual(a.physical.qty[0],0.)
        self.assertEqual([o.coin for o in orders],['pas:BTC','pas:BTC'])

class DeadlineTests(unittest.TestCase):
    def test_expiry_during_preflight_never_reaches_send_or_leaves_pending(self):
        import tempfile
        from .risk_priority import DeadlineGateway
        from joint_wallet.dispatch import Dispatcher,SubmissionNotAttempted
        from joint_wallet.book import JointBook
        from competition_live.profile import load_profile
        profile=load_profile();clock=[100.];book=JointBook(profile['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        plan=book.plan('ours','BTC',.01,80000.,5,max_notional=5000.,slot='deadline')
        def preflight(*args):clock[0]=201.
        gateway=DeadlineGateway(SimpleNamespace(preflight=preflight,send=lambda *args:self.fail('Expired send')),200.)
        with tempfile.TemporaryDirectory() as d,patch('candidate_handoff.risk_priority.time.time',side_effect=lambda:clock[0]):
            dispatch=Dispatcher(d,profile)
            with self.assertRaises(SubmissionNotAttempted):
                dispatch.execute(plan,np.full(10,80000.),np.eye(10)*1e-8,refresh=lambda:book,
                    preflight=gateway.preflight,send=gateway.send,confirmed_fills=lambda *args:[])
            self.assertIsNone(dispatch.state()['pending']);self.assertFalse(dispatch.state()['completed'])

    def test_expiry_between_preflight_and_transmission_is_known_rejection(self):
        import tempfile
        from .risk_priority import DeadlineGateway
        from joint_wallet.dispatch import Dispatcher
        from joint_wallet.book import JointBook
        from competition_live.profile import load_profile
        from bot.execution import OrderRejected
        profile=load_profile();clock=[100.];book=JointBook(profile['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        plan=book.plan('ours','BTC',.01,80000.,5,max_notional=5000.,slot='late-send')
        gateway=DeadlineGateway(SimpleNamespace(send=lambda *args:self.fail('Expired send')),200.)
        def preflight(*args):clock[0]=201.
        with tempfile.TemporaryDirectory() as d,patch('candidate_handoff.risk_priority.time.time',side_effect=lambda:clock[0]):
            dispatch=Dispatcher(d,profile)
            with self.assertRaises(OrderRejected):
                dispatch.execute(plan,np.full(10,80000.),np.eye(10)*1e-8,refresh=lambda:book,
                    preflight=preflight,send=gateway.send,confirmed_fills=lambda *args:[])
            self.assertIsNone(dispatch.state()['pending']);self.assertFalse(dispatch.state()['completed'])

if __name__=='__main__':unittest.main()
