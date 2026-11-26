import unittest
from unittest.mock import patch
import numpy as np
from pas_strategy_audit import test_execution as execution_fixtures
from pas_strategy_audit.execution import FreshLiveAccount
from competition_live.profile import START
from bot.execution import OutcomeUnknown
from .candidate import build_repaired_candidate,publication_with_repairs


class IntegratedRepairTests(unittest.TestCase):
    def test_missing_receipt_stops_retries_in_fresh_executor(self):
        fixture=execution_fixtures.ExecutionTests()
        account,portfolio,clock,orders,stream,_=fixture.setup_case(FreshLiveAccount)
        account.collect_fills=lambda *args:[]
        with self.assertRaises(OutcomeUnknown):
            fixture.execute(account,portfolio,clock,stream,[.01]+[0.]*9,'missing')
        self.assertEqual(len(orders),1)
        self.assertEqual(account.physical.qty[0],0.)
        self.assertTrue(account.unknown)
        with self.assertRaises(RuntimeError):
            fixture.execute(account,portfolio,clock,stream,[.01]+[0.]*9,'retry')
        self.assertEqual(len(orders),1)

    def test_same_publication_does_not_create_repeated_price_drift_cuts(self):
        fixture=execution_fixtures.ExecutionTests()
        account,_,_,_,_,_=fixture.setup_case(FreshLiveAccount)
        portfolio=build_repaired_candidate(account.profile,lambda *x:None,account=account)['portfolio']
        portfolio.repair_frame=dict(source_time=START,risk_protected=[True]+[False]*9,
                                    risk_weight_caps=[.02]+[0.]*9)
        target=np.array([20.]+[0.]*9)
        first,_=portfolio.protected_target(target,np.full(10,100.),50000.)
        later,_=portfolio.protected_target(target,np.full(10,110.),50000.)
        np.testing.assert_array_equal(first,later)
        portfolio.repair_frame['source_time']+=3600
        refreshed,_=portfolio.protected_target(target,np.full(10,110.),50000.)
        self.assertLess(refreshed[0],first[0])

    def test_risk_reduction_reaches_confirmed_order_outside_rebalance(self):
        fixture=execution_fixtures.ExecutionTests()
        account,_,clock,orders,stream,_=fixture.setup_case(FreshLiveAccount)
        profile=account.profile
        portfolio=build_repaired_candidate(profile,lambda *x:None,account=account)['portfolio']
        fixture.execute(account,portfolio,clock,stream,[.01]+[0.]*9,'seed')
        held=account.physical.qty.copy();equity=account.reconcile(np.full(10,80000.))
        weights=np.array([.009*80000/equity]+[0.]*9)
        publication=dict(received_ms=int(clock[0]*1000),data=dict(frame=dict(
            source_time=START,published_time=clock[0],weights=weights.tolist(),sigma=(np.eye(10)*1e-8).tolist(),
            volume=[1000.]*10,trades=[10.]*10,retention=[1.]*10)))
        publication=publication_with_repairs(publication,weights,[True]+[False]*9)
        portfolio.last_decision=START
        clock[0]=START+2000
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            portfolio.step(int(clock[0]),{s:stream(s) for s in profile['trading']['symbols']},publication,[])
        self.assertTrue(orders[-1].reduce_only)
        self.assertLessEqual(account.physical.qty[0],.009+1e-10)
        self.assertLess(account.physical.qty[0],held[0])
        self.assertTrue(any(e['kind']=='risk_cap_request' for e in portfolio.events))
        self.assertLess(account.maximum_error,1e-6)

    def test_signed_publication_generates_short_order_from_flat(self):
        fixture=execution_fixtures.ExecutionTests()
        account,_,clock,orders,stream,_=fixture.setup_case(FreshLiveAccount)
        portfolio=build_repaired_candidate(account.profile,lambda *x:None,account=account)['portfolio']
        weights=np.array([-.01]+[0.]*9)
        publication=dict(received_ms=int(clock[0]*1000),data=dict(frame=dict(
            source_time=START,published_time=clock[0],weights=weights.tolist(),sigma=(np.eye(10)*1e-8).tolist(),
            volume=[1000.]*10,trades=[10.]*10,retention=[1.]*10)))
        publication=publication_with_repairs(publication,weights,[False]*10)
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            portfolio.step(int(clock[0]),{s:stream(s) for s in account.profile['trading']['symbols']},publication,[])
        self.assertTrue(orders)
        self.assertFalse(orders[-1].is_buy)
        self.assertFalse(orders[-1].reduce_only)
        self.assertLess(account.physical.qty[0],0.)


if __name__=='__main__':
    unittest.main()
