import copy
import unittest
from .control_audit import compare


class ControlAudit(unittest.TestCase):
    def setUp(self):
        self.old = {k: [] for k in ['events', 'fills', 'owner_fills', 'snapshots', 'decisions']}
        self.old.update(metrics={'net_pnl': 12.}, orders=[{'requested_qty': 1.}])
        self.new = copy.deepcopy(self.old)
        self.new['orders'][0]['entry_limit_blocked'] = False

    def test_only_absent_false_annotation_is_compatible(self):
        self.assertEqual(compare(self.new, self.old), 1)
        self.assertIn('entry_limit_blocked', self.new['orders'][0])

    def test_true_block_is_not_compatible(self):
        self.new['orders'][0]['entry_limit_blocked'] = True
        with self.assertRaises(AssertionError):
            compare(self.new, self.old)

    def test_order_or_financial_difference_is_not_compatible(self):
        for field in ['orders', 'metrics']:
            different = copy.deepcopy(self.new)
            if field == 'orders':
                different['orders'][0]['requested_qty'] = 2.
            else:
                different['metrics']['net_pnl'] = 13.
            with self.assertRaises(AssertionError):
                compare(different, self.old)


if __name__ == '__main__':
    unittest.main()
