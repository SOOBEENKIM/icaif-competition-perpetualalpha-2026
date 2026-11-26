import json,unittest
from pathlib import Path
from experiments.persistent_intent_pairs_20261017.test_intent import IntentTests
from experiments.persistent_intent_pairs_20261017.engine import replay as old_replay
from .portfolio import replay

HERE=Path(__file__).resolve().parent


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        IntentTests.setUp(self)
        self.cfg.update(json.loads((HERE/'par_protocol.json').read_text()))
        self.cfg['pairs']=[['ETH','BTC']];self.cfg['pair_max_gross']=1000

    def run_arm(self,arm):
        return replay(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]))

    def test_native_baseline_and_standalone_pair_reproduce_original(self):
        for new,old in [('first_strategy','first_strategy'),('second_strategy','second_strategy'),('pairs_alone','fixed_delayed_immediate_complete_168')]:
            r=self.run_arm(new);o=old_replay(self.d,self.de,None,self.cfg,self.p,old,'base',int(self.d['times'][0]),int(self.d['times'][-1]))
            self.assertEqual(r['fills'],o['fills'])
            self.assertEqual([{k:v for k,v in s.items() if k not in ('base_qty','pair_qty')} for s in r['snapshots']],o['snapshots'])

    def test_pair_can_trade_symbol_owned_by_baseline(self):
        r=self.run_arm('first_plus_pairs')
        self.assertGreater(r['metrics']['completed_pairs'],0)
        self.assertTrue(any(any(s['base_qty']) and any(s['pair_qty']) for s in r['snapshots']))
        m=r['metrics'];self.assertAlmostEqual(m['base_owned_pnl']+m['pair_owned_pnl'],m['net_pnl'],places=7)
        self.assertEqual(m['risk_breach_bars'],0)

    def test_no_pair_signal_ensemble_is_exactly_baseline(self):
        self.de['directions'][:]=0
        a=self.run_arm('second_strategy');b=self.run_arm('second_plus_pairs')
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])


if __name__=='__main__':unittest.main()
