import copy,json,unittest
from pathlib import Path
import numpy as np
from .model import design,labels,fit,map_state
from .policy import proposals,held_state
from .portfolio import replay
from .research import one_pair
from experiments.persistent_intent_pairs_20261017 import test_intent as fixture
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay as previous_replay

HERE=Path(__file__).resolve().parent


class DenseModelTests(unittest.TestCase):
    def test_pair_identity_survives_permuted_universes(self):
        d=dict(names=['a'],pairs=[[0,2],[0,1]],x=np.array([[[20.],[10.]]]),
            beta=np.array([[2.,1.]]),vol=np.array([[.02,.01]]),fresh=np.array([[True,False]]))
        one=one_pair(d,[0,1]);self.assertEqual(one['pairs'],[[0,1]])
        self.assertEqual(one['x'][0,0,0],10.);self.assertEqual(one['beta'][0,0],1.)
        self.assertFalse(one['fresh'][0,0])

    def test_price_label_anchors_to_source_not_future_entry(self):
        t=np.arange(4)*3600;p=np.array([[100.,200.],[110.,190.],[121.,180.],[120.,170.]])
        d=dict(pairs=[[0,1]],beta=np.full((4,1),.5),vol=np.full((4,1),.01))
        y,m=labels(t,p,d,1)
        self.assertAlmostEqual(y[0,0],(.1-.5*(-.05))/1.5/.01);self.assertEqual(m[0],3600)
        self.assertTrue(np.isnan(y[-1,0]))

    def test_features_are_price_only_and_prefix_invariant(self):
        rng=np.random.default_rng(21);t=np.arange(1200)*3600
        r=rng.normal(0,.002,(1200,2));r[:,1]=r[:,0]*.5+r[:,1]*.5;p=100*np.exp(r.cumsum(0));v=np.full_like(p,100)
        a=design(t,p,v,v,['BTC','ETH'],[['BTC','ETH']]);b=design(t[:1000],p[:1000],v[:1000],v[:1000],['BTC','ETH'],[['BTC','ETH']])
        self.assertTrue(all(not n.startswith(('funding_','pair_')) for n in a['names']))
        for k in ['x','beta','vol','fresh']:np.testing.assert_array_equal(a[k][:1000],b[k])

    def test_future_prices_after_calibration_cutoff_do_not_change_fit(self):
        spec=json.loads((HERE/'model_protocol.json').read_text());spec['tree']['num_boost_round']=3
        rng=np.random.default_rng(22);n=4500;t=np.arange(n)*3600;p=100*np.exp(rng.normal(0,.001,(n,2)).cumsum(0))
        d=dict(x=rng.normal(size=(n,1,3)),beta=np.ones((n,1)),vol=np.full((n,1),.001),fresh=np.ones((n,1),bool),pairs=[[0,1]],names=['a','b','c'])
        a,ta,_=fit(t,p,d,24,spec,2500*3600,4000*3600);p[4000:]*=2;d['x'][4000:]*=100
        b,tb,_=fit(t,p,d,24,spec,2500*3600,4000*3600)
        self.assertEqual(a,b);self.assertEqual(ta.save_raw(),tb.save_raw())
        self.assertLess(a['last_training_maturity'],2500*3600);self.assertLess(a['last_calibration_maturity'],4000*3600)

    def test_missing_source_never_uses_a_future_bar_and_ttl_is_separate(self):
        rt=np.arange(14,25)*3600;ref=dict(market_times=rt,close=np.ones((11,2))*100)
        d=dict(beta=np.ones((11,1)),vol=np.ones((11,1))*.001,fresh=np.ones((11,1),bool))
        times=np.array([14*3600,18*3600,18*3600+3500,19*3600])
        out=map_state(times,ref,d,np.ones((11,1))*.01,[[0,1]])
        np.testing.assert_array_equal(out['directions'][:,0],[0,1,1,0]);self.assertTrue(np.isnan(out['mu'][0,0]))
        self.assertEqual(out['forecast_times'][2],18*3600)


class DensePortfolioTests(unittest.TestCase):
    def setUp(self):
        f=fixture.IntentTests();f.setUp();self.d,self.old,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        t=self.d['times'];source=t//21600*21600;ix=np.searchsorted(t,source)
        self.de=dict(dense=True,pairs=np.array([[1,0]]),signal_times=source[:,None],directions=((t-source)<3600)[:,None].astype(int),
            source_prices=self.d['par_close'][ix].copy(),beta=np.ones((len(t),1)),vol=np.full((len(t),1),.002),mu=np.full((len(t),1),.02),forecast_times=source)
        self.cfg['dense_pairs']=[['ETH','BTC']];self.cfg['dense_policy']=dict(stop_sigma=3.,decision_hours=6,source_lifetime_hours=1)

    def run_arm(self,arm='first_dense_ridge_24',de=None,engine=replay):
        return engine(self.d,self.de if de is None else de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]))

    def test_existing_five_control_paths_are_exact(self):
        for arm in ['first_strategy','second_strategy','pairs_alone','first_plus_pairs','second_plus_pairs']:
            a=self.run_arm(arm,self.old);b=self.run_arm(arm,self.old,previous_replay)
            for k in ['fills','snapshots','metrics']:self.assertEqual(a[k],b[k],(arm,k))

    def test_fixed_maturity_matches_label_and_ignores_intervening_forecast_flip(self):
        self.de['mu'][self.d['times']>=self.d['times'][0]+6*3600]=-.02
        r=self.run_arm();closed=[e for e in r['events'] if e['event']=='closed']
        self.assertTrue(closed);self.assertEqual(closed[0]['time'],int(self.d['times'][0])+24*3600)
        exits=[e for e in r['events'] if e['event']=='exit_requested']
        self.assertEqual(exits[0]['reason'],'forecast_maturity');self.assertEqual(exits[0]['time']+300,closed[0]['time'])
        self.assertEqual(r['metrics']['risk_breach_bars'],0)

    def test_single_leg_gets_one_paid_completion_without_moving_maturity(self):
        # First short misses its sealed price; the next observed price is
        # acceptable. Volume stays present at both decision and arrival.
        self.d['par_close'][1,0]=97;r=self.run_arm('pair_dense_ridge_24')
        f=r['owner_fills']['pair'];entries=[a for a in f if not a['reduce_only']]
        self.assertEqual(entries[1]['time']-entries[0]['time'],300);self.assertTrue(all(x['fee']>0 for x in entries[:2]))
        closed=next(e for e in r['events'] if e['event']=='closed');self.assertEqual(closed['time'],int(self.d['times'][0])+24*3600)
        self.assertTrue(any(e['event']=='completion_balanced' for e in r['events']))

    def test_missing_completion_source_volume_forces_paid_unwind(self):
        self.d['par_volume'][1,0]=0;r=self.run_arm('pair_dense_ridge_24')
        events=r['events'];self.assertTrue(any(e['event']=='completion_abort' for e in events))
        first_close=next(e for e in events if e['event']=='closed')
        self.assertTrue(first_close['aborted']);self.assertFalse(first_close['balanced'])
        fills=[f for f in r['owner_fills']['pair'] if f['time']<=first_close['time']]
        self.assertEqual(len(fills),2);self.assertEqual(fills[0]['signed_qty'],-fills[1]['signed_qty'])
        self.assertTrue(all(f['fee']>0 for f in fills))

    def test_current_venue_basis_preserves_source_unit_hedge(self):
        self.de['source_prices'][0,1]=100.;self.de['source_prices'][0,0]=200.;self.de['beta'][0,0]=.5;self.de['mu'][0,0]=0.
        self.d['par_close'][0,1]=110.;self.d['par_close'][0,0]=190.
        choices=proposals(0,self.d,self.de,None,24,self.cfg,self.p,self.cfg['scenarios']['base'],np.zeros(10),1e6,np.zeros(10),[1],'first_dense_zero_24')
        self.assertTrue(choices);a=choices[0];q=np.array(a['target']);self.assertEqual(a['direction'],-1)
        self.assertAlmostEqual(abs(q[1]/q[0]),.25,places=3)
        actual=float(q@(np.array(a['terminal_prices'])-self.d['par_close'][0,a['ids']]))
        self.assertAlmostEqual(a['expected_price_bps'],actual/(abs(q)@self.d['par_close'][0,a['ids']])*10000)

    def test_future_market_changes_cannot_change_earlier_fills(self):
        a=self.run_arm();cut=int(self.d['times'][0])+86400
        self.d['par_close'][self.d['times']>cut]*=1.1;b=self.run_arm()
        for k in ['fills','snapshots','events']:self.assertEqual([x for x in a[k] if x['time']<=cut],[x for x in b[k] if x['time']<=cut],k)


if __name__=='__main__':unittest.main()
