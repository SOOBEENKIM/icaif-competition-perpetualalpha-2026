import unittest
from .peer_prepare import verify_inventory

def fill(start,side,qty,t=1000):
    return dict(coin='pas:BTC',time=t,startPosition=str(start),side=side,sz=str(qty))

class InventoryTests(unittest.TestCase):
    def test_trade_id_order_irrelevant_for_same_timestamp_chain(self):
        rows=[fill(2,'A',1),fill(0,'B',2),fill(1,'B',3)]
        self.assertEqual(verify_inventory(rows,['BTC'])['qty']['BTC'],4.)

    def test_missing_earlier_fill_cannot_be_assumed(self):
        with self.assertRaisesRegex(ValueError,'Incomplete inventory'):
            verify_inventory([fill(2,'B',1)],['BTC'])

    def test_disconnected_offsetting_cycle_is_not_hidden_by_net_zero(self):
        rows=[fill(0,'B',1),fill(3,'B',1),fill(4,'A',1)]
        with self.assertRaisesRegex(ValueError,'Incomplete inventory'):
            verify_inventory(rows,['BTC'])

    def test_later_timestamp_cannot_repair_missing_earlier_inventory(self):
        with self.assertRaisesRegex(ValueError,'Incomplete inventory'):
            verify_inventory([fill(2,'A',2),fill(0,'B',2,t=2000)],['BTC'])

if __name__=='__main__':unittest.main()
