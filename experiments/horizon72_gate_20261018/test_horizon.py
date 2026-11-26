import copy
import unittest
import numpy as np

from experiments.dense_history_pairs_20261017.model import labels
from experiments.decision_gate_pairs_20261017.model import rolling
from experiments.published_pair_lifecycle_20261017.lifecycle import PairLifecycle
from experiments.published_pair_lifecycle_20261017 import test_lifecycle as fixture
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry
from experiments.basis_carry_20261017.probe import asof


class HorizonTests(unittest.TestCase):
    def test_price_labels_and_strict_maturity_follow_72h(self):
        times=np.arange(400)*3600
        close=np.c_[100*np.exp(.0003*np.arange(400)),100*np.exp(.0001*np.arange(400))]
        features=dict(beta=np.full((400,1),.5),vol=np.full((400,1),.01),pairs=np.array([[0,1]]),
                      x=np.c_[np.sin(np.arange(400)/13),np.cos(np.arange(400)/31)][:,None,:],fresh=np.ones((400,1),bool),
                      names=['synthetic_sine','synthetic_cosine'])
        target,maturity=labels(times,close,features,72)
        expected=((close[72,0]/close[0,0]-1)-.5*(close[72,1]/close[0,1]-1))/1.5/(.01*np.sqrt(72))
        self.assertAlmostEqual(target[0,0],expected)
        np.testing.assert_array_equal(maturity,times+72*3600)
        terminal=np.zeros((400,1),bool);terminal[:-72]=True
        spec=dict(horizon=72,refit_hours=24,lookback_days=90,sample_hours=6,minimum_training_rows=20,
            half_life_days=30,ridge_penalty=1.,intercept_prior_rows=12,target_clip=4.,fee_bps=9,impact_bps=10,
            solver_gap_tolerance=1e-6,solver_max_iterations=100000)
        result=rolling(times,features,target,maturity,np.zeros((400,1)),terminal,spec)
        self.assertTrue(result[-1])
        self.assertTrue(all(row['last_maturity']<row['fit_time'] for row in result[-1]))
        changed=target.copy();cutoff=times[288];changed[maturity>=cutoff]=3.
        other=rolling(times,features,changed,maturity,np.zeros((400,1)),terminal,spec)
        np.testing.assert_array_equal(result[0][times<=cutoff],other[0][times<=cutoff])
        np.testing.assert_array_equal(result[1][times<=cutoff],other[1][times<=cutoff])

    def test_72h_received_book_position_stays_open_at_24h_and_pays_exit_at_72h(self):
        f=fixture.LifecycleTests();f.setUp()
        intent=dict(f.intent,horizon=72)
        session=PairLifecycle(f.account,intent,f.cfg,f.p,f.sc,on_time=f.funding)
        session.enter(0,f.streams,f.marks,f.sigma)
        self.assertEqual(session.state,'open')
        at=24*3600-6;f.window(range(at-1,at+5))
        session.advance(at,f.streams,f.marks,f.sigma)
        self.assertEqual(session.state,'open')
        fees=f.account.physical.fees
        at=72*3600-6;f.window(range(at-1,at+5),eth=102.,btc=98.)
        session.advance(at,f.streams,f.marks,f.sigma)
        self.assertEqual(session.state,'closed')
        self.assertEqual(session.closed_reason,'forecast_maturity')
        self.assertFalse(np.any(f.account.owners['pair'].qty))
        self.assertGreater(f.account.physical.fees,fees)

    def test_entry_funding_uses_remaining_72h_without_tripling_fees(self):
        f=fixture.LifecycleTests();f.setUp()
        for symbol,rate in [('ETH',.00002),('BTC',.000005)]:
            stream=f.streams[symbol]
            stream['ctx']=[(at,sequence,dict(ctx,funding=str(rate))) for at,sequence,ctx in stream['ctx']]
        quotes={s:asof(f.streams[s],0,f.cfg) for s in f.intent['pair']}
        values=[]
        for horizon in [24,72]:
            intent=dict(f.intent,horizon=horizon,account_marks=f.marks.tolist())
            plan=plan_entry(intent,quotes,0,f.cfg,f.p,f.sc,np.zeros(10),1e6,f.sigma)
            self.assertTrue(plan['accepted'],plan)
            values.append(plan)
        self.assertEqual(values[0]['target'],values[1]['target'])
        self.assertAlmostEqual(values[1]['expected_funding_cash'],3*values[0]['expected_funding_cash'])
        self.assertEqual(values[0]['exit_cost_reserve'],values[1]['exit_cost_reserve'])
        self.assertAlmostEqual(values[1]['expected_net_cash']-values[0]['expected_net_cash'],
                               values[1]['expected_funding_cash']-values[0]['expected_funding_cash'])


if __name__=='__main__':unittest.main()
