import tempfile
import unittest
import numpy as np
from .book import JointBook,ReconciliationError
from .dispatch import Dispatcher
from .test_book import fill


class DispatcherTests(unittest.TestCase):
    def fixture(self):
        from competition_live.profile import load_profile
        p=load_profile();b=JointBook(p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('teammate',fill(1,'A',2,0))
        plan=b.plan('ours','ETH',1.,100.,3,max_notional=1000.,slot='new1')
        return p,b,plan

    def test_confirmed_partial_fill_keeps_teammate_and_restart_does_not_resend(self):
        p,b,plan=self.fixture();calls=[];rows=[]
        with tempfile.TemporaryDirectory() as d:
            dispatch=Dispatcher(d,p)
            def send(order):
                calls.append(order);self.assertIsNotNone(dispatch.state()['pending'])
                r=fill(2,'B',.4,-2,fee=.036);r['cloid']=plan['cloid'];rows.append(r);b.apply_fill('ours',r)
                return dict(oid=2,filled_size='.4')
            kwargs=dict(refresh=lambda:b,preflight=lambda *a:None,send=send,confirmed_fills=lambda *a:rows)
            result=dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,**kwargs)
            self.assertEqual(result['filled_size'],'0.4');self.assertEqual(b.owners['teammate'].qty[1],-2.)
            second=Dispatcher(d,p).execute(plan,np.full(10,100.),np.eye(10)*1e-6,**kwargs)
            self.assertTrue(second['already_completed']);self.assertEqual(len(calls),1)

    def test_unknown_send_is_durable_and_never_blindly_retried(self):
        p,b,plan=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            dispatch=Dispatcher(d,p)
            def timeout(order):raise TimeoutError('network')
            kwargs=dict(refresh=lambda:b,preflight=lambda *a:None,send=timeout,confirmed_fills=lambda *a:[])
            with self.assertRaises(TimeoutError):dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,**kwargs)
            with self.assertRaises(ReconciliationError):Dispatcher(d,p).execute(plan,np.full(10,100.),np.eye(10)*1e-6,**kwargs)

    def test_physical_reduce_only_does_not_bypass_owner_risk(self):
        p,b,plan=self.fixture();p['trading']['risk']['asset_notional']=20.
        self.assertTrue(plan['reduce_only'])
        with tempfile.TemporaryDirectory() as d:
            dispatch=Dispatcher(d,p)
            def forbidden(*args):self.fail('Must reject before submission')
            with self.assertRaises(ReconciliationError):dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,
                refresh=lambda:b,preflight=forbidden,send=forbidden,confirmed_fills=forbidden)

    def test_external_fill_during_send_changes_only_external_owner_allocation(self):
        p,b,plan=self.fixture();rows=[]
        with tempfile.TemporaryDirectory() as d:
            dispatch=Dispatcher(d,p)
            def send(order):
                b.apply_fill('teammate',fill(2,'A',.3,-2))
                r=fill(3,'B',1.,-2.3);r['cloid']=plan['cloid'];rows.append(r);b.apply_fill('ours',r)
                return dict(oid=3,filled_size='1')
            dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,refresh=lambda:b,
                preflight=lambda *a:None,send=send,confirmed_fills=lambda *a:rows)
            self.assertAlmostEqual(b.owners['ours'].qty[1],1.)
            self.assertAlmostEqual(b.owners['teammate'].qty[1],-2.3)
            self.assertAlmostEqual(b.physical.qty[1],-1.3)
            b.reconcile(np.full(10,100.))

    def test_explicit_venue_rejection_is_recorded_without_unresolved_fill(self):
        from bot.execution import OrderRejected
        p,b,plan=self.fixture();sent=[]
        with tempfile.TemporaryDirectory() as d:
            dispatch=Dispatcher(d,p)
            def send(order):sent.append(order);raise OrderRejected('Order could not immediately match')
            args=dict(refresh=lambda:b,preflight=lambda *a:None,send=send,confirmed_fills=lambda *a:[])
            with self.assertRaises(OrderRejected):dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,**args)
            self.assertIsNone(dispatch.state()['pending']);self.assertIn(plan['cloid'],dispatch.state()['rejected'])
            with self.assertRaises(OrderRejected):dispatch.execute(plan,np.full(10,100.),np.eye(10)*1e-6,**args)
            self.assertEqual(len(sent),1);self.assertEqual(b.owners['ours'].qty[1],0.)


if __name__=='__main__':unittest.main()
