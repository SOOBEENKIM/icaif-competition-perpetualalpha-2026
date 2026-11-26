import copy,json,unittest
from pathlib import Path
import numpy as np
from .test_strategy import AdaptiveTests


class EconomicLimitTests(unittest.TestCase):
    def fixture(self):
        f=AdaptiveTests();f.setUp()
        return f

    def test_price_movement_within_cash_budget_can_fill_both_legs(self):
        f=self.fixture();f.d['par_close'][1,1]=100.3;f.d['par_close'][1,0]=99.7
        tight=f.run_replay();f.cfg['entry_slippage_cap_bps']=10000
        economic=f.run_replay()
        self.assertEqual(tight['metrics']['completed_pairs'],0)
        self.assertEqual(economic['metrics']['completed_pairs'],1)
        event=next(e for e in economic['events'] if e['event']=='pair_intent')
        self.assertGreaterEqual(event['minimum_expected_cash_at_limits'],event['buffer_cash']-1e-8)

    def test_one_leg_outside_cash_budget_does_not_cancel_other_retroactively(self):
        f=self.fixture();f.cfg['entry_slippage_cap_bps']=10000;f.d['par_close'][1,1]=200
        r=f.run_replay()
        self.assertEqual(r['metrics']['aborted_pairs'],1)
        self.assertTrue(any(e['event']=='one_leg_abort' for e in r['events']))
        at=int(f.d['times'][1]);fills=[x for x in r['fills'] if x['time']==at]
        self.assertEqual(len(fills),1);self.assertEqual(fills[0]['symbol'],'BTC')
        self.assertTrue(any(x['reduce_only'] and x['fee']>0 for x in r['fills']))


if __name__=='__main__':unittest.main()
