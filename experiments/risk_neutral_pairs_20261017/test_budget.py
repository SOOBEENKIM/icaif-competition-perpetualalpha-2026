import copy
import unittest
import numpy as np
from .budget import maximum_gross,variance,budget,rounded_risk_check
from .portfolio import replay
from .proposals import proposals
from experiments.persistent_intent_pairs_20261017 import test_intent as fixture
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay as old_replay


class RiskBudgetTests(unittest.TestCase):
    def test_quadratic_boundary_includes_percentage_cost_denominator(self):
        s=np.diag([.01,.02]);b=np.array([10000.,0]);u=np.array([-.5,.5]);e=1e6;c=.0064
        cap=maximum_gross(b,u,s,e,c);before=variance(b,s,e)
        self.assertGreater(cap,0)
        self.assertAlmostEqual(variance(b+cap*u,s,e-cap*c),before,places=14)
        self.assertLess(variance(b+.99*cap*u,s,e-.99*cap*c),before)
        self.assertGreater(variance(b+1.01*cap*u,s,e-1.01*cap*c),before)
        self.assertLess(cap,maximum_gross(b,u,s,e,0))

    def test_cash_and_nonhedging_pairs_receive_no_variance_allowance(self):
        s=np.eye(2);u=np.array([-.5,.5])
        self.assertEqual(maximum_gross(np.zeros(2),u,s,1e6,.0038),0)
        self.assertEqual(maximum_gross(np.array([0.,10000.]),u,s,1e6,.0038),0)

    def test_both_current_and_pending_baseline_are_binding(self):
        r=budget(np.array([100.,0]),np.array([20.,0]),np.array([100.,100.]),[0,1],[-.5,.5],np.eye(2),1e6,.0038,.999)
        self.assertLess(r['pending_baseline_cap'],r['current_baseline_cap'])
        self.assertAlmostEqual(r['maximum_gross'],r['pending_baseline_cap']*.999)
        r=budget(np.array([100.,0]),np.zeros(2),np.array([100.,100.]),[0,1],[-.5,.5],np.eye(2),1e6,.0038,.999)
        self.assertEqual(r['maximum_gross'],0)

    def test_rounded_units_are_checked_instead_of_assuming_continuous_hedge(self):
        b=np.array([1.,0.]);s=np.eye(2);prices=np.array([100.,100.])
        ok,detail=rounded_risk_check(b,b,[-1.,.99],prices,[0,1],s,1e6,.0038)
        self.assertTrue(ok);self.assertGreater(detail['estimated_cost_cash'],0)
        ok,_=rounded_risk_check(b,b,[-1.,1.01],prices,[0,1],s,1e6,.0038)
        self.assertFalse(ok)

    def test_invalid_cost_or_equity_is_rejected(self):
        for equity,cost in [(float('nan'),.01),(1e6,float('inf')),(0,.01),(1e6,-1)]:
            with self.assertRaises(ValueError):maximum_gross([1,0],[-.5,.5],np.eye(2),equity,cost)

    def test_correlated_multiasset_quadratic_matches_direct_variance(self):
        rng=np.random.default_rng(2317);positive=0
        for _ in range(100):
            x=rng.normal(size=(4,4));s=x@x.T*.0001;b=rng.normal(size=4)*10000
            u=np.array([-.5,.5,0.,0.]);u*= -1 if b@s@u>0 else 1
            cap=maximum_gross(b,u,s,1e6,.0064)
            if cap>0:
                positive+=1
                self.assertLessEqual(variance(b+.999*cap*u,s,1e6-.999*cap*.0064),variance(b,s,1e6)*(1+1e-12))
        self.assertGreater(positive,90)


class PortfolioBudgetTests(unittest.TestCase):
    def setUp(self):
        f=fixture.IntentTests();f.setUp();self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        self.cfg['continuous_size_margin']=.999

    def run_arm(self,arm,d=None,de=None,engine=replay):
        d=self.d if d is None else d;de=self.de if de is None else de
        return engine(d,de,None,self.cfg,self.p,arm,'base',int(d['times'][0]),int(d['times'][-1]))

    def test_existing_native_pair_and_mixed_controls_are_exact(self):
        for arm in ['first_strategy','second_strategy','pairs_alone','first_plus_pairs','second_plus_pairs']:
            a=self.run_arm(arm);b=self.run_arm(arm,engine=old_replay)
            for key in ['fills','snapshots','events','metrics']:self.assertEqual(a[key],b[key],(arm,key))

    def test_new_pair_waits_for_current_base_and_records_valid_rounded_budget(self):
        r=self.run_arm('first_risk_neutral')
        entries=[e for e in r['events'] if e['event']=='pair_intent']
        self.assertTrue(entries);self.assertGreater(entries[0]['time'],int(self.d['times'][0]))
        self.assertEqual(r['allocation_checks'][0]['maximum_gross'],0)
        accepted=[a for a in r['allocation_checks'] if a['quantity_and_cost_ok']]
        self.assertTrue(accepted)
        for a in accepted:
            x=a['rounded_risk_check']
            for before,after in zip(x['before'],x['after']):self.assertLessEqual(after,before*(1+1e-12))
        self.assertEqual(r['metrics']['completed_pairs'],1)
        self.assertEqual(r['metrics']['risk_breach_bars'],0)
        m=r['metrics'];self.assertAlmostEqual(m['net_pnl'],m['base_owned_pnl']+m['pair_owned_pnl'],places=7)

    def test_pending_baseline_liquidation_blocks_pair_even_with_current_hedge(self):
        current=np.zeros(10);current[0]=10
        result=proposals(0,self.d,self.de,None,168,self.cfg,self.p,self.cfg['scenarios']['base'],np.zeros(10),1e6,np.zeros(10),[1],
            'first_risk_neutral',neutral=True,current_baseline=current,baseline_target=np.zeros(10),risk_qty=current)
        self.assertEqual(result,[])

    def test_future_changes_cannot_change_previous_allocations_and_fills(self):
        a=self.run_arm('first_risk_neutral');d=copy.deepcopy(self.d);de=copy.deepcopy(self.de)
        cutoff=int(d['times'][0])+86400;future=d['times']>cutoff
        d['par_close'][future]*=1.1;d['sigma'][future]*=2;de['terminal_prices'][future]*=1.2
        b=self.run_arm('first_risk_neutral',d,de)
        for key in ['fills','snapshots','allocation_checks','events']:
            self.assertEqual([x for x in a[key] if x['time']<=cutoff],[x for x in b[key] if x['time']<=cutoff],key)


if __name__=='__main__':unittest.main()
