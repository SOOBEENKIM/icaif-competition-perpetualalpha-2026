import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .model import features,labels,fit,load_predict
from .policy import optimize,one_coordinate,utility,CostController
from .research import data,ts,HERE


class CausalityAndForecasts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inputs,cls.arrays,cls.protocol,cls.market=data()

    def test_features_do_not_depend_on_future_suffix(self):
        x,v,_=features(self.inputs);n=self.market.exact(ts('2026-08-01'))+1
        prefix={k:a[:n] for k,a in self.inputs.items()}
        xx,vv,_=features(prefix)
        np.testing.assert_allclose(xx,x[:n],equal_nan=True,rtol=0,atol=0)
        np.testing.assert_allclose(vv,v[:n],equal_nan=True,rtol=0,atol=0)

    def test_labels_start_after_delayed_entry(self):
        _,v,_=features(self.inputs);y,maturity=labels(self.inputs,v,24);i=400
        expected=(np.log(self.inputs['close'][i+25,:3]/self.inputs['close'][i+1,:3])
                 -self.inputs['funding'][i+2:i+26,:3].sum(0))/(v[i]*np.sqrt(24))
        np.testing.assert_array_equal(y[i],expected)
        self.assertEqual(maturity[i],self.market.times[i+25])

    def test_future_data_cannot_change_fit_or_saved_inference(self):
        protocol=json.loads((HERE/'protocol.json').read_text())
        n=self.market.exact(ts(protocol['fit_end_exclusive']))+1
        prefix={k:a[:n] for k,a in self.inputs.items()}
        original=json.loads((HERE/'artifacts/models/fit.json').read_text())
        with tempfile.TemporaryDirectory() as out:
            preds,states=fit(prefix,protocol,out)
            self.assertEqual(original,states)
            for h in protocol['horizons']:
                self.assertLess(states[str(h)]['last_label_time'],ts(protocol['fit_end_exclusive']))
                p=load_predict(prefix,out,h)
                original_prediction=load_predict(prefix,HERE/'artifacts/models',h)
                for kind in ('ridge','tree'):
                    np.testing.assert_array_equal(p[kind],preds[f'{kind}_{h}'])
                    np.testing.assert_array_equal(p[kind],original_prediction[kind])


class Optimization(unittest.TestCase):
    def test_one_dimension_against_dense_independent_grid(self):
        rng=np.random.default_rng(28)
        grid=np.linspace(-.135,.135,100001)
        for _ in range(20):
            a=rng.uniform(.01,4);b=rng.uniform(-.04,.04);old=rng.uniform(-.1,.1)
            cost=rng.uniform(0,.005);reserve=rng.uniform(0,.005)
            w=one_coordinate(a,b,old,cost,reserve,.135)
            actual=.5*a*w*w-b*w+cost*abs(w-old)+reserve*abs(w)
            grid_min=np.min(.5*a*grid**2-b*grid+cost*abs(grid-old)+reserve*abs(grid))
            self.assertLessEqual(actual,grid_min+1e-12)

    def test_correlated_optimizer_dominates_hold_and_coordinate_perturbations(self):
        cov=np.array([[2,1,.4],[1,3,.6],[.4,.6,4]])*1e-5
        old=np.array([.03,-.02,.01]);mu=np.array([.012,-.004,.005]);cost=np.full(3,.0019)
        w,_=optimize(mu,cov,old,100,24,cost)
        q=100*24*cov;score=utility(w,old,mu,q,cost,cost)
        self.assertGreaterEqual(score,utility(old,old,mu,q,cost,cost)-1e-12)
        for j in range(3):
            for shift in (-1e-5,1e-5):
                v=w.copy();v[j]+=shift
                if abs(v[j])>.135:continue
                self.assertGreaterEqual(score+1e-10,utility(v,old,mu,q,cost,cost))

    def test_cost_creates_no_trade_zone(self):
        w,_=optimize(np.full(3,.002),np.eye(3)*1e-5,np.zeros(3),100,24,.0019)
        np.testing.assert_array_equal(w,np.zeros(3))
        w,_=optimize(np.full(3,.006),np.eye(3)*1e-5,np.zeros(3),100,24,.0019)
        self.assertTrue(np.all(w>0))

    def test_rejects_invalid_covariance_or_cost(self):
        for cov,cost in [(np.diag([1,1,-1]),.0019),(np.eye(3),-.001)]:
            with self.assertRaises(ValueError):optimize(np.ones(3),cov,np.zeros(3),10,24,cost)


class AgentLifecycle(unittest.TestCase):
    def setUp(self):
        _,_,self.p,_=data();self.t=ts('2026-09-01');self.prices=np.full(10,100.)
        self.sigma=np.eye(10)*1e-5;self.vol=np.full(10,10000.);self.trades=np.full(10,100.)
        self.mu=np.full((2,3),.02);self.amb=np.zeros_like(self.mu)
    def agent(self,saved=None):return CostController([self.t,self.t+21600],self.mu,self.amb,100,24,saved=saved)
    def call(self,agent,time,qty,**kw):return agent.propose(time,self.prices,qty,1e6,np.zeros(10),self.sigma,self.vol,self.trades,self.p,**kw)

    def test_restart_does_not_repeat_and_next_round_matches(self):
        a=self.agent();q=np.zeros(10);first=self.call(a,self.t,q)
        self.assertIsNotNone(first);b=self.agent(a.snapshot())
        self.assertIsNone(self.call(b,self.t,q))
        q=np.array(first['target_qty'])
        self.assertEqual(self.call(a,self.t+21600,q),self.call(b,self.t+21600,q))

    def test_off_schedule_and_stale_do_not_increase(self):
        a=self.agent();q=np.zeros(10)
        self.assertIsNone(self.call(a,self.t+1,q))
        self.assertIsNone(self.call(a,self.t,q,valid_input=False))
        self.assertEqual(a.last_decision,-1)

    def test_risk_reductions_do_not_require_forecast_timestamp(self):
        a=self.agent();q=np.zeros(10);q[0]=2000
        action=self.call(a,self.t+1,q)
        self.assertTrue(action['risk_forced']);target=np.array(action['target_qty'])
        self.assertTrue(np.all(abs(target)<=abs(q)));self.assertLess(target[0],q[0])

    def test_halt_flattens_despite_positive_alpha(self):
        q=np.zeros(10);q[0]=100
        action=self.call(self.agent(),self.t+1,q,halted=True)
        np.testing.assert_array_equal(action['target_qty'],np.zeros(10))

    def test_missing_volume_does_not_create_new_position(self):
        self.vol[:3]=0;action=self.call(self.agent(),self.t,np.zeros(10))
        np.testing.assert_array_equal(action['target_qty'],np.zeros(10))


if __name__=='__main__':unittest.main()
