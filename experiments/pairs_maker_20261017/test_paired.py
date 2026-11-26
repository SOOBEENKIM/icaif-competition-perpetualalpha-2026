import json
from pathlib import Path
from unittest.mock import patch
import unittest
from .paired import evaluate


class PairedAccountingTests(unittest.TestCase):
    def test_both_entries_and_both_exits_count_and_short_sign_is_correct(self):
        cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
        execution=json.loads(Path(__file__).with_name('paired_protocol.json').read_text())
        snapshot=dict(oracle=100.,book=dict(levels=[[dict(px='99',sz='5')],[dict(px='101',sz='5')]]))
        fill=dict(coin='par:ETH',qty=1.,side=1,price=100.,time=1000)
        with patch('experiments.pairs_maker_20261017.paired.asof',return_value=snapshot), \
             patch('experiments.pairs_maker_20261017.paired.delayed_snapshot',side_effect=[(2000,snapshot),(33000,snapshot),(33000,snapshot)]):
            r=evaluate(fill,['ETH','BTC'],1.,{'ETH':{},'BTC':{}},{'ETH':3,'BTC':3},{'ETH':{},'BTC':{}},30,cfg,execution)
        self.assertEqual(r['status'],'closed')
        self.assertAlmostEqual(r['gross_price_pnl'],-3.)
        self.assertAlmostEqual(r['fees'],.2991)
        self.assertAlmostEqual(r['net_pnl'],-3.2991)
        self.assertAlmostEqual(r['notional'],199.)

    def test_below_minimum_hedge_leaves_unresolved_inventory(self):
        cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
        execution=json.loads(Path(__file__).with_name('paired_protocol.json').read_text())
        snapshot=dict(oracle=100.,book=dict(levels=[[dict(px='99',sz='5')],[dict(px='101',sz='5')]]))
        fill=dict(coin='par:ETH',qty=.01,side=1,price=100.,time=1000)
        with patch('experiments.pairs_maker_20261017.paired.asof',return_value=snapshot), \
             patch('experiments.pairs_maker_20261017.paired.delayed_snapshot',return_value=(2000,snapshot)):
            r=evaluate(fill,['ETH','BTC'],1.,{'ETH':{},'BTC':{}},{'ETH':3,'BTC':3},{'ETH':{},'BTC':{}},30,cfg,execution)
        self.assertEqual(r['status'],'hedge_below_minimum')
        self.assertAlmostEqual(r['unhedged_notional'],1.)
        self.assertNotIn('net_pnl',r)


if __name__=='__main__':unittest.main()
