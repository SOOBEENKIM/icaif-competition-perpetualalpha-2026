import copy
import json
import unittest
import numpy as np
from experiments.shared_execution_pairs_20261017 import test_account as fixture
from experiments.opportunity_switch_pairs_20261018 import test_switch as portfolio_fixture
from experiments.decision_gate_pairs_20261017.portfolio import replay as old_replay
from .account import ResidualAccount,ResidualBookAccount,safe_with_residual
from .portfolio import replay


def plain(value):
    return json.loads(json.dumps(value,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))


class ResidualTests(unittest.TestCase):
    def setUp(self):
        f=fixture.AccountTests();f.setUp();self.f=f
        self.p,self.sc,self.cfg=f.p,f.sc,f.cfg
        self.price,self.sigma,self.market=f.price,f.sigma,f.market
        self.a=ResidualAccount(1e6,self.p['symbols'])

    def seed(self,account=None,qty=(5.,-5.)):
        a=self.a if account is None else account
        pair=dict(ids=[0,1],pair=['BTC','ETH'],closing=True,observed_close_attempt=True,balanced=True)
        a.start_episode(pair,0)
        for j,q in enumerate(qty):
            if q:
                a.physical_fill(j,q,100.,9,0)
                a.owner_fill('pair',j,q,100.,9,0,'external_fill')
        a.reconcile(self.price)
        return pair

    def intent(self,account=None,base=None,pair=None,at=0):
        a=self.a if account is None else account;z=np.zeros(10)
        raw=dict(decision_time=at,source_close=at,source_available=at,execution_time=at+300,
            source_volume=np.full(10,10000.),source_trades=np.full(10,10.),source_prices=self.price.tolist(),
            owner_targets=dict(base=(z if base is None else base).tolist(),pair=(z if pair is None else pair).tolist(),residual=z.tolist()),
            risk_forced=False,sigma=self.sigma)
        return a.seal(raw,self.p,self.sc)

    def test_handoff_conserves_cash_positions_equity_fees_and_episode(self):
        pair=self.seed();self.price[:2]=[110.,95.]
        before=(self.a.physical.cash,self.a.physical.fees,len(self.a.physical.fills),self.a.reconcile(self.price))
        pnl=self.a.episode_books[0].equity(self.price)
        self.a.handoff(pair,self.price,1)
        self.assertEqual(before,(self.a.physical.cash,self.a.physical.fees,len(self.a.physical.fills),self.a.reconcile(self.price)))
        self.assertEqual(pnl,self.a.episode_books[0].equity(self.price))
        np.testing.assert_array_equal(self.a.owners['pair'].qty,np.zeros(10))
        self.assertEqual(self.a.finished_residuals(1),[])

    def test_unobserved_or_unattempted_exit_cannot_handoff(self):
        pair=self.seed();pair['observed_close_attempt']=False
        with self.assertRaises(ValueError):self.a.handoff(pair,self.price,1)
        pair['observed_close_attempt']=True;self.a.unknown.add('BTC')
        with self.assertRaises(ValueError):self.a.handoff(pair,self.price,1)

    def test_residual_cannot_increase_and_active_symbol_cannot_overlap(self):
        pair=self.seed();self.a.handoff(pair,self.price,1)
        with self.assertRaises(ValueError):self.a.owner_fill('residual',0,1.,100.,9,2,'external_fill')
        with self.assertRaises(ValueError):self.a.start_episode(dict(ids=[0,2],pair=['BTC','HYPE']),2)
        raw=self.intent();raw['owner_targets']['residual'][0]=1
        with self.assertRaises(ValueError):self.a.seal(raw,self.p,self.sc)

    def test_reduction_priority_is_allocated_once(self):
        pair=self.seed(qty=(5.,0.));self.a.handoff(pair,self.price,1)
        self.a.physical_fill(0,5.,100.,9,2);self.a.owner_fill('base',0,5.,100.,9,2,'external_fill')
        self.market.volume[1,0]=30
        self.a.execute_candle(self.intent(at=2),self.market,1,self.p,self.sc)
        self.assertEqual(self.a.owners['residual'].qty[0],2.)
        self.assertEqual(self.a.owners['base'].qty[0],5.)
        self.a.reconcile(self.price)

    def test_internal_cross_to_base_does_not_create_external_fee(self):
        pair=self.seed();self.a.handoff(pair,self.price,1)
        before=(len(self.a.physical.fills),self.a.physical.cash,self.a.physical.fees)
        base=np.zeros(10);base[0]=5
        intent=self.intent(base=base,at=2);self.a.cross(intent,0,100.,300)
        self.assertEqual(before,(len(self.a.physical.fills),self.a.physical.cash,self.a.physical.fees))
        self.assertEqual(self.a.owners['base'].qty[0],5)
        self.assertEqual(self.a.owners['residual'].qty[0],0)
        self.a.reconcile(self.price)

    def test_funding_is_conserved_after_handoff_with_opposing_base(self):
        pair=self.seed();self.a.handoff(pair,self.price,1)
        self.a.physical_fill(1,2.,100.,9,2);self.a.owner_fill('base',1,2.,100.,9,2,'external_fill')
        raw=np.full(10,.0002);cash=-self.a.physical.qty*self.price*raw
        rates=np.where(cash>0,raw*.5,raw)
        self.a.charge_funding(3600,self.price,rates,np.ones(10,bool));self.a.reconcile(self.price)
        self.assertAlmostEqual(sum(x.funding for x in self.a.owners.values()),self.a.physical.funding)
        self.assertAlmostEqual(self.a.episode_books[0].funding,self.a.owners['pair'].funding+self.a.owners['residual'].funding)

    def test_completion_requires_observed_zero_inventory(self):
        pair=self.seed();self.a.handoff(pair,self.price,1)
        self.market.volume[1,:2]=10
        self.a.execute_candle(self.intent(at=2),self.market,1,self.p,self.sc)
        self.assertEqual(self.a.finished_residuals(300),[])
        self.a.execute_candle(self.intent(at=300),self.market,2,self.p,self.sc)
        done=self.a.finished_residuals(600)
        self.assertEqual(len(done),1);self.assertEqual(self.a.finished_residuals(600),[])
        self.assertGreater(self.a.episode_books[0].fees,0)

    def test_residual_leg_risk_is_checked_without_assuming_it_fills(self):
        proposal=dict(ids=[0,1],target=[5.,-5.]);res=np.zeros(10);res[2]=5.
        risk=copy.deepcopy(self.p);risk['risk']['net_fraction']=.0006
        # Final net exposure 500 is allowed, but a single new long plus the
        # still-unfilled residual reaches 1000 and must be rejected.
        self.assertFalse(safe_with_residual(proposal,res,np.zeros(10),np.zeros(10),1e6,self.price,self.sigma,risk,self.sc))
        self.assertTrue(safe_with_residual(proposal,res,np.zeros(10),np.zeros(10),1e6,self.price,self.sigma,self.p,self.sc))

    def test_received_book_unknown_leg_blocks_new_submissions(self):
        a=ResidualBookAccount(1e6,self.p['symbols']);pair=self.seed(a);a.handoff(pair,self.price,0)
        result=a.execute_books(self.intent(a),{'BTC':self.f.book()},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertIn('ETH',result['unknown_symbols'])
        self.assertNotEqual(a.owners['residual'].qty[1],0)
        with self.assertRaises(ValueError):a.start_episode(dict(ids=[2,3],pair=['HYPE','SP500']),2)
        with self.assertRaises(RuntimeError):a.execute_books(self.intent(a,at=2),{},self.price,self.sigma,self.p,self.sc,self.cfg)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        f=portfolio_fixture.SwitchTests();f.setUp();self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        self.cfg.pop('opportunity_switch');self.cfg['residual_queue']=True
        self.d['par_close'][3:,1]=110.
        self.d['par_volume'][4,1]=1
        self.t=self.d['times']

    def run_arm(self,engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,'pair_dense_hourlyhedge_queue_24','base',int(self.t[0]),int(self.t[-1]))

    def test_disabled_is_exact_original(self):
        self.cfg.pop('residual_queue');self.assertEqual(plain(self.run_arm()),plain(self.run_arm(old_replay)))

    def test_new_disjoint_entry_can_fill_while_paid_residual_exit_continues(self):
        r=self.run_arm();handoffs=[e for e in r['events'] if e['event']=='residual_handoff']
        self.assertTrue(handoffs)
        concurrent=[s for s in r['snapshots'] if any(s['active_pair_qty']) and any(s['residual_qty'])]
        self.assertTrue(concurrent)
        for s in concurrent:self.assertFalse(np.any((np.array(s['active_pair_qty'])!=0)&(np.array(s['residual_qty'])!=0)))
        self.assertAlmostEqual(sum(e['net_pnl'] for e in r['episodes']),r['metrics']['pair_owned_pnl'])
        self.assertAlmostEqual(sum(e['fees'] for e in r['episodes']),sum(f['fee'] for f in r['owner_fills']['pair']+r['owner_fills']['residual']))
        self.assertGreater(r['metrics']['fees'],0)

    def test_future_market_changes_do_not_change_earlier_handoffs_or_fills(self):
        a=self.run_arm();cut=int(self.t[7])
        self.d['par_close'][8:]*=2;self.d['funding'][8:]=.001;self.d['known_hourly_funding'][8:]=.001
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual(plain([x for x in a[key] if x['time']<=cut]),plain([x for x in b[key] if x['time']<=cut]))


if __name__=='__main__':unittest.main()
