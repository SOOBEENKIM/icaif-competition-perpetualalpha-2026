"""Synthetic plumbing test only; these prices are not performance evidence."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
from . import forward
from .test_engine import CFG


class IntegrationTests(unittest.TestCase):
    def test_six_accounts_replay_and_flag_incomplete_market_coverage(self):
        from strategy.agent import TAIL
        trading = json.loads((TAIL / 'baseline.json').read_text())['protocol']
        coins = ['par:' + s for s in trading['symbols']]
        metadata = {c: dict(name=c, szDecimals=3) for c in coins}
        start = 1788264000000; end = start + 73*3600000
        def events():
            for hour in range(74):
                base = start + hour*3600000
                for offset in (-1000, 0, 1000, 2000, 3000):
                    now = base + offset
                    if now >= end: continue
                    for c in coins:
                        yield dict(received_ms=now, channel='activeAssetCtx', data=dict(coin=c, ctx=dict(markPx='100', oraclePx='100', funding='0')))
                        # A deliberately large synthetic discount exercises entry.
                        bid = 98. if c == 'par:ETH' and hour < 72 else 99.99
                        yield dict(received_ms=now, channel='l2Book', data=dict(coin=c, time=now,
                            levels=[[dict(px=str(bid), sz='1000')], [dict(px=str(bid+.02), sz='1000')]]))
                    if offset == 0:
                        frame = dict(source_time=base//1000, sigma=(np.eye(10)*1e-8).tolist(), retention=[1.]*10,
                            volume=[1e6]*10, trades=[1000.]*10,
                            paired_metrics=[dict(coins=['par:ETH','par:BTC'],beta=1.,correlation=.9,hourly_vol=.001)])
                        yield dict(received_ms=now, channel='strategySignal', data=dict(frame=frame,
                            incumbent=[.04,.04]+[0.]*8, candidate=[.03,.03]+[0.]*8))
        def info(payload):
            return [dict(time=at, fundingRate='0') for at in range(start+3600000, end+1,3600000)]
        plan = dict(outcome_release_unix=end/1000, sources={}, paper_start_unix=start/1000, capture_end_unix=end/1000,
                    paired_protocol=CFG, trading_protocol=trading, coins=coins, metadata=metadata, end_liquidation_seconds=600)
        with tempfile.TemporaryDirectory(prefix='paired-synthetic-') as directory:
            dest = Path(directory)
            with patch.object(forward, 'DEST', dest), patch.object(forward.capture_module, 'DEST', dest), \
                 patch.object(forward.capture_module, 'events', events), patch.object(forward.capture_module, 'info', info):
                forward.evaluate(plan)
            result = json.loads((dest/'results.json').read_text())
            self.assertEqual(len(result['rows']), 6)
            pair = [r for r in result['rows'] if r['arm']=='basis_carry_pair_72h']
            self.assertTrue(all(r['fill_count'] >= 4 for r in pair))
            self.assertTrue(all(r['completed_pairs'] == 1 for r in pair))
            self.assertTrue(all(r['fatal'] for r in result['rows']))
            self.assertFalse(result['paper_improvement'])
            self.assertEqual(result['live_orders_sent'], 0)


if __name__ == '__main__': unittest.main()
