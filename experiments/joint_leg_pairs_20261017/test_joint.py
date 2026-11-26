import copy, json, unittest
from pathlib import Path
import numpy as np
from .model import fit, rolling, common_projection, map_forecasts, terminal_prices
from .portfolio import replay
from experiments.conditional_completion_pairs_20261017.test_completion import ConditionalTests
from experiments.conditional_completion_pairs_20261017.portfolio import replay as original


class JointTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((Path(__file__).parent/'protocol.json').read_text())

    def fixture(self, n=1500):
        rng=np.random.default_rng(51);t=np.arange(n)*3600
        x=rng.normal(size=(n,2,3));y=np.stack([.5*x[...,0]+.1,.3*x[...,1]-.2],axis=-1)
        f=dict(x=x,fresh=np.ones((n,2),bool))
        return t,f,y,t+86400,np.full_like(y,.02),np.ones((n,2),bool)

    def test_two_output_fit_matches_independent_augmented_least_squares(self):
        t,f,y,m,u,tf=self.fixture(300);now=m[-1]+1
        v=fit(f['x'][:,0],y[:,0],m,now,self.spec)
        w=np.exp2(-(now-m)/(30*86400));w/=w.sum()
        z=np.c_[np.ones(len(t)),(f['x'][:,0]-v['center'])/v['scale']]
        penalty=np.diag([12/len(t),1,1,1])
        a=np.r_[np.sqrt(w[:,None])*z,np.sqrt(penalty)]
        b=np.r_[np.sqrt(w[:,None])*y[:,0],np.zeros((4,2))]
        np.testing.assert_allclose(v['coefficient'],np.linalg.lstsq(a,b,rcond=None)[0],atol=1e-12)
        with self.assertRaises(ValueError):fit(f['x'][:,0],y[:,0],m,m[-1],self.spec)

    def test_future_targets_and_terminal_activity_cannot_change_current_prediction(self):
        t,f,y,m,u,tf=self.fixture();cut=1350
        a,st,records=rolling(t,f,y,m,u,tf,self.spec)
        f['x'][cut+1:]*=30;y[m>t[cut]]=100;tf[m>t[cut]]=False;u[cut+1:]*=2
        b,ss,rr=rolling(t,f,y,m,u,tf,self.spec)
        np.testing.assert_array_equal(a[:cut+1],b[:cut+1]);np.testing.assert_array_equal(st[:cut+1],ss[:cut+1])
        self.assertTrue(records);self.assertTrue(all(r['last_maturity']<r['fit_time'] for r in records))

    def test_terminal_inactive_rows_are_not_training_labels(self):
        t,f,y,m,u,tf=self.fixture();tf[1200]=False
        a,_,ra=rolling(t,f,y,m,u,tf,self.spec);y[1200]=100
        b,_,rb=rolling(t,f,y,m,u,tf,self.spec)
        np.testing.assert_array_equal(a,b);self.assertEqual(ra,rb)

    def test_pair_independence_and_missing_warmup(self):
        t,f,y,m,u,tf=self.fixture();a,st,_=rolling(t,f,y,m,u,tf,self.spec)
        f['x'][:,1]*=10;y[:,1]*=-10
        b,_,_=rolling(t,f,y,m,u,tf,self.spec)
        np.testing.assert_array_equal(a[:,0],b[:,0]);self.assertTrue(np.isnan(a[:1100]).all())

    def test_scalar_is_not_an_identified_leg_forecast_and_projection_preserves_pair(self):
        beta=np.array([.25,1.,4.]);mu=np.array([.01,.02,-.01]);legs=np.tile([.08,.06],(3,1))
        r=common_projection(mu,beta,legs)
        np.testing.assert_allclose((r[:,0]-beta*r[:,1])/(1+beta),mu,atol=1e-15)
        np.testing.assert_allclose(r.mean(-1),legs.mean(-1))
        # Identical exact pair forecasts can imply very different unhedged inventory values.
        self.assertGreater(r[1,0],mu[1]);self.assertGreater(r[1,1],-mu[1])

    def test_state_uses_source_clock_and_explicit_prices(self):
        t=np.array([3600,3900,7200]);rt=np.array([3600,7200]);leg=np.array([[[.03,.02]],[[.04,.01]]])
        old=dict(signal_times=np.array([[3600],[3600],[7200]]),mu=np.full((3,1),.003),beta=np.ones((3,1)),directions=np.ones((3,1),int),source_prices=np.array([[100.,50.]]*3),pairs=np.array([[0,1]]))
        s=map_forecasts(t,rt,leg,old,'joint')
        np.testing.assert_allclose(s['terminal_prices'][:,0],[[103,51],[103,51],[104,50.5]])
        np.testing.assert_allclose(s['mu'][:,0],[.005,.005,.015])
        np.testing.assert_array_equal(s['signal_times'],old['signal_times'])
        bad=copy.deepcopy(old);bad['signal_times'][0]=7200
        self.assertEqual(map_forecasts(t,rt,leg,bad,'joint')['directions'][0,0],0)
        np.testing.assert_allclose(terminal_prices(s,1,0,[0,1]),[103,51])

    def test_no_explicit_prices_reproduce_old_account(self):
        f=ConditionalTests();f.setUp();a=f.run_arm(original);b=f.run_arm(replay)
        for key in ['fills','events','metrics','snapshots']:self.assertEqual(a[key],b[key],key)

    def test_explicit_zero_common_prices_reproduce_old_account(self):
        f=ConditionalTests();f.setUp();a=f.run_arm(original)
        f.de['terminal_prices']=np.stack([f.de['source_prices'][:,ids]*np.stack([1+f.de['mu'][:,k],1-f.de['mu'][:,k]],axis=-1) for k,ids in enumerate(f.de['pairs'])],axis=1)
        b=f.run_arm(replay)
        for key in ['fills','events','metrics','snapshots']:self.assertEqual(a[key],b[key],key)

    def test_entry_preserves_explicit_hypothesis_and_future_account_prefix(self):
        f=ConditionalTests();f.setUp()
        f.de['terminal_prices']=np.stack([f.de['source_prices'][:,ids]*np.stack([1+f.de['mu'][:,k],1-f.de['mu'][:,k]],axis=-1) for k,ids in enumerate(f.de['pairs'])],axis=1)
        f.de['terminal_prices']+=1.
        f.d['par_close'][1,0]=97.
        a=f.run_arm(replay);first=next(e for e in a['events'] if e['event']=='pair_intent')
        i=np.searchsorted(f.t,first['time'])
        np.testing.assert_array_equal(first['terminal_prices'],f.de['terminal_prices'][i,first['pair_index']])
        cut=int(f.t[30]);f.de['terminal_prices'][31:]*=2;f.d['par_close'][31:]*=2
        b=f.run_arm(replay)
        for key in ['fills','events','snapshots']:
            self.assertEqual([r for r in a[key] if r['time']<=cut],[r for r in b[key] if r['time']<=cut],key)

    def test_explicit_prices_reach_both_renewal_branches(self):
        from experiments.paid_rehedge_pairs_20261017.test_rehedge import PaidAdjustmentTests
        from .renewal import renewal_plan
        for beta in [1.,2.]:
            f=PaidAdjustmentTests();f.setUp();f.de['beta'][:]=beta
            pair,a=f.direct()
            f.de['terminal_prices']=np.full((*f.de['mu'].shape,2),np.nan)
            f.de['terminal_prices'][f.i,0]=[110.,90.]
            plan=renewal_plan(pair,a.owners['pair'].qty,f.i,f.d,f.de,f.cfg,f.p,
                              f.cfg['scenarios']['base'],a.physical.qty,1e6,ledger=a.owners['pair'])
            self.assertTrue(plan['accepted']);self.assertEqual(plan['terminal_prices'],[110.,90.])
            self.assertEqual(bool(plan.get('requires_adjustment')),beta==2.)
            f.de['terminal_prices'][f.i,0]=[90.,110.]
            plan=renewal_plan(pair,a.owners['pair'].qty,f.i,f.d,f.de,f.cfg,f.p,
                              f.cfg['scenarios']['base'],a.physical.qty,1e6,ledger=a.owners['pair'])
            self.assertFalse(plan['accepted'])

    def test_completion_values_the_frozen_individual_leg_prices(self):
        f=ConditionalTests();f.setUp();pair,a=f.direct();old=f.plan(pair,a)
        pair['terminal_prices']=[110.3,109.7]
        new=f.plan(pair,a)
        # Exact hedge price PnL is the same; future fees can differ, and the
        # single-leg close alternative never receives an invented future fill.
        self.assertIsNotNone(new)
        self.assertEqual(new['frozen_terminal_prices'],[110.3,109.7])
        self.assertEqual(old['immediate_close_cash'],new['immediate_close_cash'])
        self.assertNotEqual(old['exit_reserve'],new['exit_reserve'])


if __name__=='__main__':unittest.main()
