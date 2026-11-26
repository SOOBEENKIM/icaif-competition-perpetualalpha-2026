import json
from pathlib import Path
import unittest
from .probe import evaluate_fill


class AccountingTests(unittest.TestCase):
    def test_maker_entry_and_taker_exit_are_both_charged(self):
        cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
        book=dict(time=32000,coin='par:BTC',levels=[[dict(px='101',sz='2')],[dict(px='103',sz='2')]])
        stream=dict(book=[(32000,0,book)],book_times=[32000],ctx=[(32000,0,dict(oraclePx='102',funding='0'))],ctx_times=[32000],resets=[])
        fill=dict(time=1000,side=1,qty=1.,price=100.)
        r=evaluate_fill(fill,stream,30,cfg,{})
        self.assertEqual(r['status'],'closed')
        self.assertAlmostEqual(r['fees'],.1209)
        self.assertAlmostEqual(r['net_pnl'],.8791)
        self.assertAlmostEqual(r['gross_markout_bps'],200.)
        missing=evaluate_fill(fill,stream,300,cfg,{})
        self.assertEqual(missing['status'],'missing_future_exit')
        self.assertNotIn('net_pnl',missing)


if __name__=='__main__':unittest.main()
