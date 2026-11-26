import copy
import json
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from .model import action, loss, solve, fit, rolling, map_gate
from .entry import filter_proposals


class DecisionGateTests(unittest.TestCase):
    def setUp(self):self.spec=json.loads((Path(__file__).parent/'protocol.json').read_text())
    def fixture(self,n=1500):
        rng=np.random.default_rng(171019);t=np.arange(n)*3600;x=rng.normal(size=(n,2,3))
        target=.4*x[...,0]-.2*x[...,1]
        f=dict(x=x,fresh=np.ones((n,2),bool),vol=np.full((n,2),.002),names=['a','b','c'])
        return t,f,target,t+86400,np.zeros_like(target),np.ones_like(target,bool)

    def test_surrogate_matches_enumerated_three_action_oracle(self):
        rng=np.random.default_rng(171020);y=rng.normal(size=100);m=rng.normal(size=100);c=rng.uniform(0,.5,100)
        points=np.array([[-1,1],[0,0],[1,1]])
        actual=np.c_[-y,c];predicted=np.c_[-m,c]
        costs=actual@points.T;ix=np.argmin(costs,axis=1);opt=points[ix]
        expected=np.max((actual-2*predicted)@points.T,axis=1)+2*np.sum(predicted*opt,axis=1)-np.min(costs,axis=1)
        np.testing.assert_allclose(loss(m,y,c),expected,atol=1e-14)
        selected=action(m,c);regret=np.max(y[:,None]*points[:,0]-c[:,None]*points[:,1],axis=1)-(selected*y-c*abs(selected))
        self.assertTrue(np.all(loss(m,y,c)+1e-12>=regret))
        np.testing.assert_array_equal(action(np.array([-.1,.1]),.1),[0,0])

    def test_solver_agrees_with_independent_primal_qp(self):
        rng=np.random.default_rng(171021);n=32;z=np.c_[np.ones(n),rng.normal(size=(n,3))]
        y=.8*z[:,1]+rng.normal(size=n)*.3;c=rng.uniform(.05,.4,n);w=np.full(n,1/n);p=np.array([.2,1,1,1])
        theta,info=solve(z,y,c,w,p,tolerance=1e-7)
        a=action(y,c);dim=z.shape[1];identity=np.eye(n)
        matrix=np.r_[np.c_[-2*z,identity],np.c_[2*z,identity],np.c_[np.zeros_like(z),identity]]
        lower=np.r_[-y-c,y-c,np.zeros(n)]
        def objective(v):return p@(v[:dim]**2)-2*v[:dim]@(z.T@(w*a))+w@v[dim:]+w@(y*a+c*abs(a))
        def gradient(v):return np.r_[2*p*v[:dim]-2*z.T@(w*a),w]
        initial=np.r_[np.zeros(dim),np.maximum(0,abs(y)-c)]
        result=minimize(objective,initial,jac=gradient,constraints=[LinearConstraint(matrix,lower,np.inf)],method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
        self.assertTrue(result.success,result.message)
        self.assertLessEqual(info['gap'],1e-7)
        self.assertAlmostEqual(info['primal'],result.fun,places=6)
        np.testing.assert_allclose(theta,result.x[:dim],atol=6e-4)

    def test_high_cost_selects_flat_without_imagined_return(self):
        z=np.c_[np.ones(30),np.linspace(-1,1,30)];y=np.linspace(-.1,.1,30)
        coef,info=solve(z,y,np.ones(30),np.full(30,1/30),np.ones(2))
        np.testing.assert_array_equal(coef,[0.,0.]);self.assertEqual(info['gap'],0.)

    def test_fit_checks_maturity_and_positive_units(self):
        t,f,y,m,raw,tf=self.fixture(300);unit=f['vol'][:,0]*np.sqrt(24)
        with self.assertRaises(ValueError):fit(f['x'][:,0],y[:,0],unit,m,m[-1],self.spec)
        unit[0]=0
        with self.assertRaises(ValueError):fit(f['x'][:,0],y[:,0],unit,m,m[-1]+1,self.spec)

    def test_future_changes_do_not_change_past_decisions(self):
        t,f,y,m,raw,tf=self.fixture();cut=1350
        a,s,v,st,records=rolling(t,f,y,m,raw,tf,self.spec)
        f['x'][cut+1:]*=10;f['vol'][cut+1:]*=3;y[m>t[cut]]=100;tf[m>t[cut]]=False
        b,ss,vv,tt,rr=rolling(t,f,y,m,raw,tf,self.spec)
        for x1,x2 in [(a,b),(s,ss),(v,vv),(st,tt)]:np.testing.assert_array_equal(x1[:cut+1],x2[:cut+1])
        self.assertTrue(v[1248].all());self.assertTrue(all(r['last_maturity']<r['fit_time'] for r in records))

    def test_terminal_inactivity_only_excludes_training(self):
        t,f,y,m,raw,tf=self.fixture();tf[1248]=False
        a,s,v,st,records=rolling(t,f,y,m,raw,tf,self.spec);y[1248]=100
        b,ss,vv,tt,rr=rolling(t,f,y,m,raw,tf,self.spec)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(s,ss);self.assertEqual(records,rr)
        self.assertTrue(v[1248].all())

    def test_each_pair_fits_independently(self):
        t,f,y,m,raw,tf=self.fixture();a,s,_,_,_=rolling(t,f,y,m,raw,tf,self.spec)
        y[:,1]*=-3;f['x'][:,1]*=2
        b,ss,_,_,_=rolling(t,f,y,m,raw,tf,self.spec)
        np.testing.assert_array_equal(a[:,0],b[:,0]);np.testing.assert_array_equal(s[:,0],ss[:,0])

    def test_mapping_preserves_all_price_and_execution_fields(self):
        t=np.arange(4)*3600;source=np.broadcast_to(t[:,None],(4,2)).copy()
        baseline=dict(signal_times=source,mu=np.full((4,2),.001),beta=np.ones((4,2)),source_prices=np.ones((4,3))*100,directions=np.ones((4,2),int))
        saved=copy.deepcopy(baseline);actions=np.tile([1,-1],(4,1));available=np.ones((4,2),bool);available[-1]=False
        state=map_gate(t,t,actions,available,baseline)
        for key in baseline:np.testing.assert_array_equal(state[key],saved[key]);np.testing.assert_array_equal(baseline[key],saved[key])
        np.testing.assert_array_equal(state['economic_action'][-1],[0,0])

    def test_gate_filters_direction_without_changing_order_payload(self):
        choices=[dict(pair_index=0,direction=1,terminal_prices=[100,99],target=[1,-2]),dict(pair_index=0,direction=-1),dict(pair_index=1,direction=1)]
        kept=filter_proposals(choices,np.array([1,1]),np.array([True,False]))
        self.assertEqual(kept,[choices[0]]);self.assertIs(kept[0],choices[0])
        self.assertEqual(filter_proposals(choices,np.zeros(2),np.ones(2,bool)),[])

    def test_insufficient_data_and_failed_solver_are_not_silent(self):
        t,f,y,m,raw,tf=self.fixture(100);a,s,v,st,records=rolling(t,f,y,m,raw,tf,self.spec)
        self.assertFalse(v.any());self.assertFalse(records)
        with self.assertRaises(ValueError):rolling(t,f,y,m+1,raw,tf,self.spec)
        z=np.c_[np.ones(30),np.linspace(-1,1,30)];y=z[:,1]
        with self.assertRaises(ArithmeticError):solve(z,y,np.full(30,.1),np.full(30,1/30),np.ones(2),maximum=0)


if __name__=='__main__':unittest.main()
