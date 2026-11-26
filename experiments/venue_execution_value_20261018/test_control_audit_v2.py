import copy
import unittest
from .control_audit_v2 import compare


class ControlAudit(unittest.TestCase):
    def setUp(self):
        self.old = {k: [] for k in ['events', 'fills', 'owner_fills', 'snapshots', 'decisions']}
        self.old.update(metrics={'net_pnl': 12.}, orders=[{'requested_qty': 1., 'filled_qty': 0., 'capacity': 0.}])
        self.new = copy.deepcopy(self.old)
        self.new['orders'][0]['entry_limit_blocked'] = False

    def test_absent_annotation_logged_without_mutation(self):
        self.assertEqual(compare(self.new, self.old), {'false': 1, 'true': 0})
        self.assertIn('entry_limit_blocked', self.new['orders'][0])
        self.new['orders'][0]['entry_limit_blocked'] = True
        self.assertEqual(compare(self.new, self.old), {'false': 0, 'true': 1})

    def test_true_annotation_requires_both_zero_fill_and_capacity(self):
        self.new['orders'][0]['entry_limit_blocked'] = True
        for key in ['filled_qty', 'capacity']:
            old, new = copy.deepcopy(self.old), copy.deepcopy(self.new)
            old['orders'][0][key] = new['orders'][0][key] = 1.
            with self.assertRaises(AssertionError): compare(new, old)

    def test_order_or_financial_difference_is_not_compatible(self):
        for field in ['orders', 'metrics']:
            different = copy.deepcopy(self.new)
            if field == 'orders': different['orders'][0]['requested_qty'] = 2.
            else: different['metrics']['net_pnl'] = 13.
            with self.assertRaises(AssertionError): compare(different, self.old)

    def test_existing_annotation_must_match(self):
        self.old['orders'][0]['entry_limit_blocked'] = True
        with self.assertRaises(AssertionError): compare(self.new, self.old)


if __name__ == '__main__': unittest.main()
