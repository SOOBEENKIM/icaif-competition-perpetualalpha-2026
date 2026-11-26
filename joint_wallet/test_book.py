import copy
import json
import unittest
import numpy as np
from .book import JointBook,ReconciliationError


def fill(tid,side,size,start,price=100.,fee=.1,time=None):
    return dict(coin='pas:ETH',tid=tid,oid=tid,side=side,sz=str(size),startPosition=str(start),
        px=str(price),fee=str(fee),feeToken='USDP',time=tid*1000 if time is None else time)


class JointOwnershipTests(unittest.TestCase):
    def book(self):return JointBook(['ETH'],{'ours':50000.,'teammate':50000.})

    def test_opposite_positions_keep_teammate_units_and_equity_not_cash_identity(self):
        b=self.book();b.apply_fill('teammate',fill(1,'A',2,0))
        b.apply_fill('ours',fill(2,'B',1,-2,price=105.))
        self.assertEqual(b.physical.qty[0],-1.)
        self.assertEqual(b.owners['teammate'].qty[0],-2.)
        self.assertEqual(b.owners['ours'].qty[0],1.)
        self.assertNotEqual(sum(x.cash for x in b.owners.values()),b.physical.cash)
        self.assertLess(max(b.reconcile([110.]).values()),1e-7)
        self.assertAlmostEqual(b.owners['ours'].equity([110.])-50000,4.9)
        self.assertAlmostEqual(b.owners['teammate'].equity([110.])-50000,-20.1)

    def test_signed_funding_and_zero_net_requires_actual_settlement(self):
        b=self.book();b.apply_fill('teammate',fill(1,'A',2,0));b.apply_fill('ours',fill(2,'B',1,-2))
        b.apply_funding('pas:ETH',3000,.2,-1,'h1')
        self.assertAlmostEqual(b.owners['teammate'].funding,.4)
        self.assertAlmostEqual(b.owners['ours'].funding,-.2)
        b.apply_fill('ours',fill(4,'B',1,-1))
        saved=json.dumps(b.serialize(),sort_keys=True)
        with self.assertRaises(ReconciliationError):b.apply_funding('pas:ETH',5000,0,0,'h2')
        self.assertEqual(saved,json.dumps(b.serialize(),sort_keys=True))
        b.apply_funding('pas:ETH',5000,0,0,'h2',cash_per_unit=-.25)
        self.assertAlmostEqual(b.owners['ours'].funding,-.7)
        self.assertAlmostEqual(b.owners['teammate'].funding,.9)
        b.reconcile([99.])

    def test_owner_target_is_delta_and_virtual_exit_can_increase_physical_short(self):
        b=self.book();b.apply_fill('teammate',fill(1,'A',2,0));b.apply_fill('ours',fill(2,'B',1,-2))
        p=b.plan('ours','ETH',0.,101.,3,max_notional=5000,slot='close')
        self.assertEqual(p['delta'],-1.);self.assertFalse(p['reduce_only'])
        self.assertEqual(p['physical_target'],-2.)
        b.validate_plan(p);b.apply_fill('ours',fill(3,'A',1,-1,price=101.))
        self.assertEqual(b.owners['teammate'].qty[0],-2.)
        self.assertEqual(b.owners['ours'].qty[0],0.)
        b.reconcile([101.])

    def test_teammate_fill_between_decision_and_send_invalidates_plan(self):
        b=self.book();b.apply_fill('teammate',fill(1,'A',2,0))
        p=b.plan('ours','ETH',1.,100.,3,max_notional=5000,slot='open')
        b.apply_fill('teammate',fill(2,'B',.5,-2))
        with self.assertRaises(ReconciliationError):b.validate_plan(p)
        fresh=b.plan('ours','ETH',1.,100.,3,max_notional=5000,slot='open2')
        self.assertEqual(fresh['delta'],1.)
        self.assertEqual(fresh['physical_target'],-.5)

    def test_partial_fills_duplicate_restore_and_fees_once(self):
        b=self.book();row=fill(1,'B',.3,0,fee=.03);b.apply_fill('ours',row)
        self.assertFalse(b.apply_fill('ours',row))
        restored=JointBook.restore(json.loads(json.dumps(b.serialize())))
        p=restored.plan('ours','ETH',1.,100.,3,max_notional=5000,slot='remaining')
        self.assertEqual(p['delta'],.7);self.assertAlmostEqual(restored.physical.fees,.03)
        bad=copy.deepcopy(row);bad['fee']='.04'
        with self.assertRaises(ReconciliationError):restored.apply_fill('ours',bad)
        with self.assertRaises(ReconciliationError):restored.apply_fill('teammate',row)

    def test_owner_reversal_closes_before_flip_and_dust_rules_use_exchange(self):
        b=self.book();b.apply_fill('ours',fill(1,'B',1,0))
        p=b.plan('ours','ETH',-1.,100.,3,max_notional=5000,slot='flip')
        self.assertEqual(p['delta'],-1.)
        b.apply_fill('ours',fill(2,'A',1,1))
        self.assertEqual(b.plan('ours','ETH',-1.,100.,3,max_notional=5000,slot='flip2')['delta'],-1.)
        b=self.book();b.apply_fill('teammate',fill(1,'A',.02,0));b.apply_fill('ours',fill(2,'B',.01,-.02))
        self.assertIsNone(b.plan('ours','ETH',0.,100.,3,max_notional=5000,slot='dust'))

    def test_missing_fill_and_ambiguous_settlement_fail_without_mutation(self):
        b=self.book();saved=b.serialize()
        with self.assertRaises(ReconciliationError):b.apply_fill('ours',fill(1,'B',1,2))
        self.assertEqual(saved,b.serialize())
        b.apply_fill('ours',fill(2,'B',1,0))
        with self.assertRaises(ReconciliationError):b.apply_funding('pas:ETH',2000,-.1,1,'ambiguous')

    def test_many_interleaved_reversals_conserve_total_pnl(self):
        rng=np.random.default_rng(42);b=self.book()
        for i in range(1,151):
            side='B' if rng.uniform()>.5 else 'A';qty=float(rng.integers(1,21))/10
            b.apply_fill('ours' if i%3 else 'teammate',fill(i,side,qty,b.physical.qty[0],price=float(rng.uniform(90,110)),fee=.05))
            b.reconcile([float(rng.uniform(90,110))])
        restored=JointBook.restore(b.serialize());restored.reconcile([100.])
        self.assertAlmostEqual(b.physical.equity([100.]),restored.physical.equity([100.]))

    def test_public_history_trade_ids_are_not_sequence_numbers(self):
        from .audit import reconstruct
        a=fill(999,'B',.2,0,time=1000);c=fill(111,'B',.3,.2,time=1000)
        # Funding normally arrives a few milliseconds AFTER the hourly edge.
        f=dict(time=3600014,delta=dict(coin='pas:ETH',usdc='-.05',szi='.5'))
        b=reconstruct(['ETH'],{'ours':50000.,'teammate':50000.},[a,c],[f],set(),{999,111})
        self.assertEqual(b.owners['ours'].qty[0],.5)
        self.assertAlmostEqual(b.owners['ours'].funding,-.05)


if __name__=='__main__':unittest.main()
