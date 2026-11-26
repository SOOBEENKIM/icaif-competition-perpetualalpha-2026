from decimal import Decimal
from types import SimpleNamespace
import unittest
from competition_live.venue import check_oi
from .venue_guard import check_oi_strict,opening_quantity


class VenueGuardTests(unittest.TestCase):
    def case(self,oi='9.5'):
        return ([SimpleNamespace(coin='pas:BTC',is_buy=False,size='2',reduce_only=False)],
            [dict(position=dict(coin='pas:BTC',szi='1'))],dict(universe=[dict(name='pas:BTC')]),
            [dict(oraclePx='10',markPx='10',openInterest=oi)],dict(coinToOiCap=[['pas:BTC','100']],totalOiCap='1000'))

    def test_equal_absolute_reversal_can_still_add_oi(self):
        args=self.case();self.assertEqual(check_oi(*args)['increment_upper_bound'],0.)
        with self.assertRaisesRegex(ValueError,'reversal'):check_oi_strict(*args)

    def test_valid_reversal_reserves_new_side_size(self):
        self.assertEqual(check_oi_strict(*self.case('8'))['increment_upper_bound'],10.)

    def test_plain_reduction_adds_no_oi(self):
        args=self.case();args[0][0].size='.5';args[0][0].reduce_only=True
        self.assertEqual(check_oi_strict(*args)['increment_upper_bound'],0.)

    def test_net_reversal_and_same_direction_increase(self):
        self.assertEqual(opening_quantity(Decimal('1'),Decimal('-3')),Decimal('2'))
        self.assertEqual(opening_quantity(Decimal('-1'),Decimal('-3')),Decimal('3'))


if __name__=='__main__':unittest.main()
