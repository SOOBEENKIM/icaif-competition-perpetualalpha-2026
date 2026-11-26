import copy,json,unittest
import numpy as np
from experiments.adaptive_mean_pairs_20261017.test_strategy import AdaptiveTests
from replay_lab.common_comparison.engine import Ledger
from .engine import replay
from .completion import completion_plan


class CompletionTests(AdaptiveTests):
    def run_replay(self,arm='adaptive_standard_abort_168',overlay=False):
        self.cfg['hedge_completion_attempts']=1
        return replay(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def test_missing_leg_can_complete_once_without_adding_to_first_leg(self):
        self.d['par_close'][1,1]=100.3
        r=self.run_replay('adaptive_standard_complete_168')
        self.assertEqual(r['metrics']['aborted_pairs'],0);self.assertEqual(r['metrics']['completed_pairs'],1)
        fills=[f for f in r['fills'] if not f['reduce_only']]
        self.assertEqual([f['symbol'] for f in fills],['BTC','ETH'])
        self.assertEqual(fills[1]['time']-fills[0]['time'],300)
        intent=next(e for e in r['events'] if e['event']=='completion_intent')
        self.assertAlmostEqual(intent['sunk_pair_fees'],fills[0]['fee'])
        self.assertGreaterEqual(intent['minimum_expected_cash_at_limits'],intent['buffer_cash']-1e-8)

    def test_fallback_outside_sealed_limit_is_paid_unwind(self):
        self.d['par_close'][1,1]=100.3
        self.d['par_close'][2,1]=200
        r=self.run_replay('adaptive_standard_complete_168')
        self.assertEqual(r['metrics']['aborted_pairs'],1)
        self.assertEqual(sum(e['event']=='completion_intent' for e in r['events']),1)
        self.assertTrue(any(e['event']=='completion_abort' for e in r['events']))
        self.assertTrue(any(f['reduce_only'] and f['fee']>0 for f in r['fills']))

    def test_zero_source_volume_does_not_become_invented_hedge_liquidity(self):
        self.d['par_volume'][1,1]=0
        r=self.run_replay('adaptive_standard_complete_168')
        self.assertEqual(r['metrics']['aborted_pairs'],1)
        self.assertFalse(any(f['symbol']=='ETH' for f in r['fills']))

    def test_other_positions_do_not_subsidize_completion_budget(self):
        at=int(self.d['times'][0]);i=1
        pair=dict(ids=[1,0],target=[5.,-5.],beta=1.,direction=1,
            terminal_prices=[103.,97.],forecast_source=at,intent_time=at,
            first_fill=at+300,completion_deadline=at+600,
            entry_mean=.05,entry_sd=.01,adaptive_mean=False)
        ledger=Ledger(1e6,self.p['symbols']);ledger.fill(0,-5,100,9,at+300)
        enriched=copy.deepcopy(ledger);enriched.fill(7,1,100,10000,at+300)
        rate=np.zeros(10);rate[0]=.001;rate[7]=.9
        for account in (ledger,enriched):account.charge_funding(at+300,np.ones(10)*100,rate,np.ones(10,bool),0)
        sc=self.cfg['scenarios']['base']
        a=completion_plan(pair,ledger,i,self.d,self.de,self.cfg,self.p,sc)
        b=completion_plan(pair,enriched,i,self.d,self.de,self.cfg,self.p,sc)
        self.assertEqual(a,b);self.assertAlmostEqual(a['realized_pair_funding'],.5)
        self.assertIsNone(completion_plan({**pair,'completion_deadline':at+300},ledger,i,self.d,self.de,self.cfg,self.p,sc))

    def test_fallback_partial_fill_is_trimmed_not_called_balanced(self):
        self.d['par_volume'][1,1]=0
        # Missing leg's zero source volume prohibits the next fill. Use a
        # limit rejection to create the first missing leg instead.
        self.d['par_volume'][1,1]=10000
        self.d['par_close'][1,1]=100.3
        self.d['par_volume'][2,1]=25
        r=self.run_replay('adaptive_standard_complete_168')
        self.assertTrue(any(e['event']=='completion_trim_requested' for e in r['events']))
        self.assertGreater(r['metrics']['hedge_repairs'],0)

    def test_expired_completion_and_forecast_hit_are_not_retried(self):
        self.d['par_volume'][1,1]=0
        self.de['mean_log_prices'][1,1]=np.log(100.)
        r=self.run_replay('adaptive_standard_complete_168')
        self.assertTrue(any(e['event']=='completion_rejected_unwind' for e in r['events']))
        self.assertFalse(any(e['event']=='completion_intent' for e in r['events']))


if __name__=='__main__':unittest.main()
