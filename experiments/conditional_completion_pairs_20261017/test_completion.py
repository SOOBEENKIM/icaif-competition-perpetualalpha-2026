import copy,unittest
import numpy as np
from experiments.paid_rehedge_pairs_20261017.test_rehedge import PaidAdjustmentTests as Fixture
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.hourly_refresh_pairs_20261017.policy import replay as previous_replay
from experiments.dense_history_pairs_20261017.completion import completion_plan as original
from .completion import completion_plan
from .portfolio import replay

class ConditionalTests(unittest.TestCase):
    def setUp(self):
        f=Fixture();f.setUp();self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        self.de['beta'][:]=1.;self.t=self.d['times'];self.cfg['conditional_completion']={'risk_aversion':0.}
    def direct(self,have=0):
        a=SharedAccount(1e6,self.p['symbols']);ids=[1,0];target=[5.,-5.]
        j=ids[have];amount=target[have];price=101. if amount>0 else 99.
        a.physical_fill(j,amount,price,9,int(self.t[1]));a.owner_fill('pair',j,amount,price,9,int(self.t[1]),'external_fill')
        pair=dict(dense=True,ids=ids,pair=['ETH','BTC'],pair_index=0,target=target,direction=1,
            terminal_prices=[100.3,99.7],forecast_source=int(self.t[0]),intent_time=int(self.t[0]),
            horizon=24,entry_vol=.002,first_fill=int(self.t[1]),completion_deadline=int(self.t[2]))
        return pair,a
    def plan(self,pair,a,i=1):
        return completion_plan(pair,a.owners['pair'],i,self.d,self.de,self.cfg,self.p,self.cfg['scenarios']['base'],risk_ledger=a.physical)
    def run_arm(self,engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,'pair_dense_hourlyhedge_direct_24','base',int(self.t[0]),int(self.t[-1]))
    def test_remaining_choice_can_improve_without_recovering_sunk_loss(self):
        pair,a=self.direct();new=self.plan(pair,a)
        self.assertIsNotNone(new)
        self.assertIsNone(original(pair,a.owners['pair'],1,self.d,self.de,self.cfg,self.p,self.cfg['scenarios']['base'],risk_ledger=a.physical))
        self.assertGreater(new['cash_advantage'],new['buffer_cash'])
        self.assertEqual(new['sunk_entry_costs_charged_again'],0)
        self.assertEqual(new['target'][0],a.owners['pair'].qty[1])
    def test_both_missing_sides_have_cost_bounded_limits(self):
        for have in [0,1]:
            pair,a=self.direct(have);plan=self.plan(pair,a);self.assertIsNotNone(plan)
            self.assertGreaterEqual(plan['minimum_cash_advantage_at_limit'],plan['buffer_cash']-1e-8)
            self.assertGreaterEqual(plan['minimum_utility_advantage_at_limit'],plan['buffer_cash']-1e-8)
            self.assertEqual(plan['target'][have],pair['target'][have])
    def test_past_fee_does_not_change_cash_choice_but_remains_in_equity(self):
        pair,a=self.direct();old=self.plan(pair,a);eq=a.physical.equity(self.d['par_close'][1])
        for ledger in [a.physical,a.owners['pair']]:
            ledger.cash-=100.;ledger.shadow_cash-=100.;ledger.fees+=100.
            ledger.fills[0]['fee']+=100.
        self.assertEqual(self.plan(pair,a),old)
        self.assertAlmostEqual(a.physical.equity(self.d['par_close'][1]),eq-100.)
    def test_future_source_expiry_and_lost_remaining_value_reject(self):
        pair,a=self.direct()
        self.assertIsNone(self.plan({**pair,'completion_deadline':int(self.t[1])},a))
        self.assertIsNone(self.plan({**pair,'forecast_source':int(self.t[2])},a))
        self.assertIsNone(self.plan({**pair,'terminal_prices':[99.,101.]},a))
    def test_global_risk_and_covariance_charge_can_reject(self):
        pair,a=self.direct();self.assertIsNotNone(self.plan(pair,a))
        self.cfg['conditional_completion']['risk_aversion']=1e8
        self.assertIsNone(self.plan(pair,a))
        self.cfg['conditional_completion']['risk_aversion']=0.
        self.p['risk']['asset_notional']=100.
        self.assertIsNone(self.plan(pair,a))
    def test_disabled_policy_reproduces_old_engine(self):
        self.cfg.pop('conditional_completion')
        a=self.run_arm();b=self.run_arm(previous_replay)
        for k in ['events','fills','metrics','snapshots']:self.assertEqual(a[k],b[k],k)
    def test_one_retry_remains_independent_and_charged(self):
        self.d['par_close'][1,0]=97.
        r=self.run_arm();events=[e for e in r['events'] if e['event']=='completion_intent']
        self.assertTrue(events);self.assertTrue(events[0]['conditional_completion'])
        fills=r['owner_fills']['pair'];self.assertEqual(fills[1]['time']-fills[0]['time'],300)
        self.assertTrue(all(f['fee']>0 for f in fills[:2]))
        self.assertTrue(any(e['event']=='completion_balanced' for e in r['events']))
    def test_zero_liquidity_is_not_a_free_hedge_or_free_unwind(self):
        self.d['par_close'][1,0]=97.;self.d['par_volume'][1,0]=0.
        r=self.run_arm();close=next(e for e in r['events'] if e['event']=='closed')
        self.assertTrue(close['aborted']);self.assertFalse(close['balanced'])
        attempts=[e for e in r['events'] if e['time']<=close['time'] and e['event']=='completion_intent']
        self.assertEqual(len(attempts),1)
        fills=[f for f in r['owner_fills']['pair'] if f['time']<=close['time']]
        self.assertEqual(len(fills),2);self.assertTrue(all(f['fee']>0 for f in fills))
    def test_future_price_and_funding_do_not_change_past_actions(self):
        self.d['par_close'][1,0]=97.;a=self.run_arm();cut=int(self.t[30])
        self.d['par_close'][31:]*=2.;self.d['funding'][31:]=.1;self.d['known_hourly_funding'][31:]=.1
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual([r for r in a[key] if r['time']<=cut],[r for r in b[key] if r['time']<=cut],key)

if __name__=='__main__':unittest.main()
