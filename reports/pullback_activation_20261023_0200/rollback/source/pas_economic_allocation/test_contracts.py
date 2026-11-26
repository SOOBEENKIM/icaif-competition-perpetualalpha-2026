import copy
import unittest
import numpy as np
from scipy.optimize import minimize
from . import model
from .policy import solve,utility,EconomicController
from pas_adaptive_development.study import ts


class AllocationTests(unittest.TestCase):
    def test_matches_independent_convex_solver_and_short_symmetry(self):
        rng=np.random.default_rng(27);a=rng.normal(size=(5,5));cov=(a@a.T+np.eye(5))*.001
        old=np.array([.05,-.03,0.,.06,0.]);mu=np.array([.004,-.005,.02,-.008,.004])
        bound=np.full(5,.135);cost=.0019;exit_cost=np.linspace(.0024,.004,5)
        w=solve(old,mu,cov,5.,cost,exit_cost,bound)
        ref=minimize(lambda x:-utility(x,old,mu,cov,5.,cost,exit_cost),w,
                     method='Powell',bounds=list(zip(-bound,bound)),options={'xtol':1e-11,'ftol':1e-11})
        self.assertLessEqual(ref.fun,-utility(w,old,mu,cov,5.,cost,exit_cost)+1e-8)
        self.assertLessEqual(-utility(w,old,mu,cov,5.,cost,exit_cost)-ref.fun,1e-7)
        np.testing.assert_allclose(solve(-old,-mu,cov,5.,cost,exit_cost,bound),-w,atol=1e-8)

    def test_cost_creates_no_trade_region_and_funding_payments_are_full(self):
        old=np.zeros(3);cov=np.eye(3)*.01;bound=np.full(3,.135)
        np.testing.assert_array_equal(solve(old,np.array([.003,-.003,0]),cov,20.,.0019,.0024,bound),0.)
        for w in [-.1,.1]:
            f=.02;actual=-w*f if w*f>0 else -.5*w*f
            self.assertAlmostEqual(-.75*f*w-.25*abs(f*w),actual)

    def test_every_market_can_go_long_or_short_and_zero_liquidity_cannot_open(self):
        from competition_live.profile import load_profile
        p=load_profile()['trading'];time=ts('2026-08-01T12:00:00Z')
        for j in range(10):
            for sign in [-1,1]:
                forecast=np.zeros((1,10));forecast[0,j]=sign*.05
                c=EconomicController([time],forecast,np.zeros_like(forecast),20.)
                args=[time,np.full(10,100.),np.zeros(10),50000.,np.zeros(10),np.eye(10)*.00001,np.ones(10),np.ones(10),p]
                a=c.propose(*args);self.assertGreater(sign*a['target_qty'][j],0.)
                c=EconomicController([time],forecast,np.zeros_like(forecast),20.)
                args[6]=np.zeros(10)
                np.testing.assert_array_equal(c.propose(*args)['target_qty'],0.)

    def test_uses_actual_inventory_and_does_not_restore_quantity_after_cap(self):
        from competition_live.profile import load_profile
        from replay_lab.common_comparison.engine import risk_breaches
        p=load_profile()['trading'];time=ts('2026-08-01T12:00:00Z');mu=np.zeros((1,10))
        sigma=np.eye(10)*.00001;prices=np.full(10,100.);owned=np.zeros(10);owned[3]=10.
        c=EconomicController([time],mu,mu,20.)
        q=c.propose(time,prices,owned,50000.,mu[0],sigma,np.ones(10),np.ones(10),p)['target_qty']
        self.assertLess(abs(q[3]),abs(owned[3]));self.assertFalse(risk_breaches(q,50000.,prices,sigma,p['risk']))

    def test_post_projection_rejects_an_uneconomic_addition(self):
        from .projected import choose
        old=np.array([.1,.02]);feasible=old.copy();bad=np.array([.01,.135])
        mu=np.array([.02,.001]);cov=np.eye(2)*.001
        chosen,gain=choose(bad,feasible,old,mu,cov,5.,.0019,.0024)
        self.assertLess(gain,0.)
        np.testing.assert_array_equal(chosen,feasible)
        better=np.array([.13,.02]);chosen,gain=choose(better,feasible,old,mu,cov,5.,.0019,.0024)
        self.assertGreater(gain,0.);np.testing.assert_array_equal(chosen,better)


class CausalityTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(28);t=ts('2026-03-18T00:00:00Z')+np.arange(150*24)*3600
        shape=(len(t),10);close=100*np.exp(np.cumsum(rng.normal(0,.003,shape),axis=0))
        return dict(market_times=t,close=close,volume=np.ones(shape),trades=np.ones(shape),
                    funding=rng.normal(0,.00002,shape),observed=np.ones(shape,bool))

    def test_future_poison_and_prefix_cannot_change_fitted_model_or_past_features(self):
        d=self.data();p=dict(train_start=ts('2026-03-18T00:00:00Z'),train_end=ts('2026-06-01T00:00:00Z'),
                            calibration_start=ts('2026-06-01T00:00:00Z'),calibration_end=ts('2026-07-01T00:00:00Z'))
        a=model.train(d,p);bad=copy.deepcopy(d);future=d['market_times']>=p['calibration_end']
        bad['close'][future]*=100;bad['funding'][future]=.5
        self.assertEqual(a,model.train(bad,p))
        self.assertLess(a['last_training_maturity'],p['train_end'])
        self.assertLess(a['last_calibration_maturity'],p['calibration_end'])
        prefix={k:v[~future] for k,v in d.items()}
        for full,short,poison in zip(model.features(d),model.features(prefix),model.features(bad)):
            np.testing.assert_allclose(full[~future],short,equal_nan=True)
            np.testing.assert_allclose(full[~future],poison[~future],equal_nan=True)

    def test_current_funding_settlement_is_not_used_at_same_timestamp(self):
        d=self.data();i=2000;original=model.features(d)[2];d['funding'][i]+=1
        changed=model.features(d)[2]
        np.testing.assert_array_equal(original[:i+1],changed[:i+1])
        self.assertGreater(abs(original[i+1]-changed[i+1]).max(),0.)

    def test_slow_forecast_prefix_and_asset_permutation(self):
        from .slow import forecast
        d=self.data();p=d['close'];n=3200;full=forecast(p)
        np.testing.assert_allclose(full[:n],forecast(p[:n]),equal_nan=True)
        perm=np.arange(9,-1,-1)
        np.testing.assert_allclose(full[:,perm],forecast(p[:,perm]),equal_nan=True)

    def test_budget_uses_completed_past_returns_and_keeps_final_caps(self):
        from .adaptive import budget
        from .anchored import AnchoredController
        from competition_live.profile import load_profile
        d=self.data();rng=np.random.default_rng(31);w=rng.uniform(-.02,.02,d['close'].shape)
        f,_=budget(d['market_times'],d['close'],d['funding'],w)
        cut=2400;bad=copy.deepcopy(d);bad['close'][cut:]*=2;bad['funding'][cut:]=.1
        g,_=budget(bad['market_times'],bad['close'],bad['funding'],w)
        np.testing.assert_array_equal(f[:cut],g[:cut])
        h,_=budget(d['market_times'][:cut],d['close'][:cut],d['funding'][:cut],w[:cut])
        np.testing.assert_array_equal(f[:cut],h)
        self.assertGreaterEqual(f.min(),.5);self.assertLessEqual(f.max(),2.)
        p=load_profile()['trading'];t=ts('2026-08-01T12:00:00Z')
        anchor=np.zeros((1,10));anchor[0,0]=.02;protected=np.zeros((1,10),bool);protected[0,0]=True
        forecast=np.full((1,10),.05);carry=np.zeros((1,10));c=AnchoredController([t],forecast,carry,anchor,protected,5.)
        q=np.zeros(10);q[0]=30.;px=np.full(10,100.)
        action=c.propose(t,px,q,50000.,anchor[0],np.eye(10)*1e-6,np.ones(10),np.ones(10),p)
        self.assertLessEqual(action['target_qty'][0]*100/50000,.02+1e-12)
        self.assertGreater(sum(abs(v) for v in action['target_qty'][1:]),0.)


if __name__=='__main__':unittest.main()
