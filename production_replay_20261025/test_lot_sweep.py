import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.basis_carry_20261017.probe import sweep as old_sweep
from .lot_sweep import sweep


class LotSweepTests(unittest.TestCase):
    def test_reproduced_overfill_is_exactly_one_requested_quantity(self):
        levels=[dict(px=str(100+i),sz=str(q)) for i,q in enumerate([.005,.085,.140])]
        old,_=old_sweep(levels,.23,1.,1,3)
        self.assertGreater(Decimal(str(old)),Decimal('0.230'))
        actual,price=sweep(levels,.23,1.,1,3)
        self.assertEqual(Decimal(str(actual)),Decimal('0.230'))
        self.assertAlmostEqual(price,(.005*100+.085*101+.140*102)/.23)

    def test_partial_lots_are_not_created_by_depth_haircut(self):
        actual,_=sweep([dict(px='100',sz='0.003')],.003,.5,1,3)
        self.assertEqual(actual,.001)

    def test_limit_and_requested_budget_are_preserved_for_both_sides(self):
        for side in (-1,1):
            levels=[dict(px=str(100+side*i),sz='0.111') for i in range(5)]
            for units in range(1,445):
                wanted=Decimal(units)/1000
                qty,_=sweep(levels,float(wanted),1.,side,3,100+side)
                self.assertLessEqual(Decimal(str(qty)),wanted)
                self.assertLessEqual(qty,.222)
                self.assertEqual(Decimal(str(qty))*1000,(Decimal(str(qty))*1000).to_integral_value())

    def test_strict_live_account_accepts_lot_exact_paper_receipt(self):
        from .test_books import RecordedBookTests
        from . import book_replay
        v,p,t=RecordedBookTests().case()
        for _,_,book in v.streams['SP500']['book']:
            book['levels'][0]=[dict(px=str(99.99-.01*i),sz=str(q)) for i,q in enumerate([.005,.085,.140])]
        target=np.zeros(10);target[3]=-.23
        with patch.object(book_replay,'sweep',sweep),patch('competition_live.live_account.time',SimpleNamespace(time=lambda:v.now)):
            p.submit_baseline(t,target,v.streams,np.full(10,100.),np.eye(10)*1e-6,False)
        self.assertEqual(p.account.physical.qty[3],-.23)
        self.assertFalse(p.account.pending_orders)
        self.assertFalse(p.account.unknown)


if __name__=='__main__':unittest.main()
