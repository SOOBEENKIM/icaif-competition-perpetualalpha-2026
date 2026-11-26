import json
from pathlib import Path
import unittest
import numpy as np
from experiments.cost_horizon_20261016.research import data
from .probe import Probe,settlement_clock,HOUR


class ProbeTests(unittest.TestCase):
    def fixture(self):
        _,_,trading,_=data();start=1792188000000-5000
        cfg=json.loads(Path('experiments/paired_venue_20261017/protocol.json').read_text())
        coins=['par:EWY','par:SP500']
        p=dict(paper_start_unix=start/1000,capture_end_unix=(start+20000)/1000,
            entry_window_seconds=10,end_liquidation_seconds=5,beta=1.,direction=-1,coins=coins,
            symbols=['EWY','SP500'],trading_protocol=trading,paired_protocol=cfg,
            metadata={c:dict(szDecimals=3) for c in coins})
        return p,start

    def snapshot(self,probe,now,funding=.0004):
        books={};contexts={}
        for j,c in enumerate(probe.plan['coins']):
            d=dict(coin=c,time=now,levels=[[dict(px='99.99',sz='10000',n=1)],[dict(px='100.01',sz='10000',n=1)]])
            books[c]=dict(received_ms=now,data=d)
            contexts[c]=dict(received_ms=now,ctx=dict(oraclePx='100',markPx='100',funding=str(funding if j==0 else 0)))
            for arm in probe.arms.values():arm.depth.update(c,d)
        signal=dict(frame=dict(source_time=now//HOUR*3600,sigma=(np.eye(2)*1e-6).tolist()))
        return books,contexts,signal

    def test_non_hour_start_uses_real_hourly_settlements(self):
        start=1792188000000-5000
        self.assertEqual(settlement_clock(start,start+20000),[1792188000000])

    def test_future_books_paid_funding_and_both_exits_reconcile(self):
        plan,start=self.fixture();probe=Probe(plan)
        for second in range(21):
            now=start+second*1000;books,contexts,signal=self.snapshot(probe,now)
            if second==5:
                for arm in probe.arms.values():
                    self.assertLess(arm.qty['par:EWY'],0);self.assertGreater(arm.qty['par:SP500'],0)
                    expected=-arm.qty['par:EWY']*100*.0004
                    arm.charge_funding(now,{'par:EWY':.0004,'par:SP500':0},contexts)
                    self.assertAlmostEqual(arm.funding,expected)
            probe.step(now,books,contexts,signal)
            for arm in probe.arms.values():arm.snapshot(now,contexts)
        for s,arm in probe.arms.items():
            r=arm.result(contexts)
            self.assertTrue(r['flat']);self.assertFalse(r['fatal']);self.assertEqual(r['fill_count'],4)
            self.assertGreater(r['fees'],0);self.assertGreater(r['funding'],0)
            self.assertEqual(sum(e['event']=='balanced_open' for e in probe.managers[s].events),1)
            self.assertEqual(sum(e['event']=='flat_confirmed' for e in probe.managers[s].events),1)
            self.assertTrue(all(f['time']>=start+plan['paired_protocol']['scenarios'][s]['latency_seconds']*1000 for f in arm.fills))

    def test_no_edge_does_not_create_order_or_cost(self):
        plan,start=self.fixture();probe=Probe(plan)
        books,contexts,signal=self.snapshot(probe,start,funding=0)
        probe.step(start,books,contexts,signal)
        self.assertFalse(probe.attempted)
        for arm in probe.arms.values():self.assertFalse(arm.pending);self.assertFalse(arm.fills)

    def test_stale_risk_signal_cannot_start_a_pair(self):
        plan,start=self.fixture();probe=Probe(plan)
        books,contexts,signal=self.snapshot(probe,start)
        signal['frame']['source_time']-=7*3600
        probe.step(start,books,contexts,signal)
        self.assertFalse(probe.attempted)


if __name__=='__main__':unittest.main()
