import copy,json,unittest
from pathlib import Path
import numpy as np
from .model import fit,predict,rolling
from experiments.rolling_feature_pairs_20261017.model import rolling as normalized_rolling


class RawTargetTests(unittest.TestCase):
    def setUp(self):self.spec=json.loads((Path(__file__).parent/'protocol.json').read_text())
    def fixture(self,n=1500):
        rng=np.random.default_rng(61017);t=np.arange(n)*3600;x=rng.normal(size=(n,2,3))
        target=.4*x[...,0]+.2*x[...,1]-.1
        f=dict(x=x,fresh=np.ones((n,2),bool),vol=.002*np.exp(.1*x[...,2]),names=['a','b','c'])
        return t,f,target,t+86400,np.zeros_like(target),np.ones_like(target,bool)

    def test_raw_fit_matches_cash_return_augmented_least_squares(self):
        t,f,y,m,r,tf=self.fixture(300);y[10,0]=10.;u=f['vol'][:,0]*np.sqrt(24);now=m[-1]+1
        model=fit(f['x'][:,0],y[:,0],u,m,now,self.spec)
        w=np.exp2(-(now-m)/(30*86400));w/=w.sum()
        z=np.c_[np.ones(len(t)),(f['x'][:,0]-model['center'])/model['scale']]
        penalty=np.diag([12/len(t),1,1,1])
        aa=np.r_[np.sqrt(w[:,None])*z,np.sqrt(penalty)]
        yy=np.r_[np.sqrt(w)*np.clip(y[:,0],-4,4)*u,np.zeros(4)]
        np.testing.assert_allclose(model['coefficient'],np.linalg.lstsq(aa,yy,rcond=None)[0],atol=1e-14)
        self.assertEqual(model['clipped_targets'],1)

    def test_maturity_and_nonpositive_unit_are_rejected(self):
        t,f,y,m,r,tf=self.fixture(300);u=f['vol'][:,0]*np.sqrt(24)
        with self.assertRaises(ValueError):fit(f['x'][:,0],y[:,0],u,m,m[-1],self.spec)
        u[0]=0
        with self.assertRaises(ValueError):fit(f['x'][:,0],y[:,0],u,m,m[-1]+1,self.spec)

    def test_control_is_exact_and_constant_unit_is_equivalent(self):
        t,f,y,m,r,tf=self.fixture();f['vol'][:]=.002
        cash,control,st,records=rolling(t,f,y,m,r,tf,self.spec)
        old_spec={**self.spec,'candidates':['direct'],'diagnostic_variants':[]}
        old,ss,_=normalized_rolling(t,f,y,m,r,tf,old_spec)
        np.testing.assert_array_equal(control,old['direct']);np.testing.assert_array_equal(st,ss)
        np.testing.assert_allclose(cash,control,rtol=1e-12,atol=1e-15)
        self.assertTrue(records)

    def test_current_volatility_does_not_multiply_raw_forecast(self):
        t,f,y,m,r,tf=self.fixture(300);model=fit(f['x'][:,0],y[:,0],f['vol'][:,0]*np.sqrt(24),m,m[-1]+1,self.spec)
        a=predict(model,f['x'][10,0],.1,self.spec);b=predict(model,f['x'][10,0],.2,self.spec)
        self.assertEqual(a,b);self.assertNotEqual(a,0)
        capped=predict(model,f['x'][10,0],abs(a)/8,self.spec)
        self.assertAlmostEqual(abs(capped),abs(a)/2)

    def test_future_features_labels_and_activity_do_not_change_past_forecasts(self):
        t,f,y,m,r,tf=self.fixture();cut=1350
        a,control,st,rec=rolling(t,f,y,m,r,tf,self.spec)
        f['x'][cut+1:]*=10;f['vol'][cut+1:]*=4;y[m>t[cut]]=100;tf[m>t[cut]]=False
        b,cc,ss,rr=rolling(t,f,y,m,r,tf,self.spec)
        for aa,bb in [(a,b),(control,cc),(st,ss)]:np.testing.assert_array_equal(aa[:cut+1],bb[:cut+1])
        self.assertEqual([v for v in rec if v['fit_time']<=t[cut]],[v for v in rr if v['fit_time']<=t[cut]])

    def test_inactive_terminal_rows_excluded_only_when_training(self):
        t,f,y,m,r,tf=self.fixture();tf[1248]=False
        a,_,_,rec=rolling(t,f,y,m,r,tf,self.spec);y[1248]=100
        b,_,_,rr=rolling(t,f,y,m,r,tf,self.spec)
        np.testing.assert_array_equal(a,b);self.assertEqual(rec,rr)
        self.assertTrue(np.isfinite(a[1248]).all())

    def test_pair_fits_independent(self):
        t,f,y,m,r,tf=self.fixture();a,_,_,_=rolling(t,f,y,m,r,tf,self.spec)
        y[:,1]*=-10;f['x'][:,1]*=3
        b,_,_,_=rolling(t,f,y,m,r,tf,self.spec)
        np.testing.assert_array_equal(a[:,0],b[:,0])

    def test_insufficient_training_and_wrong_horizon(self):
        t,f,y,m,r,tf=self.fixture(200);a,c,st,rec=rolling(t,f,y,m,r,tf,self.spec)
        self.assertTrue(np.isnan(a).all());self.assertEqual(rec,[]);self.assertTrue((st==-1).all())
        with self.assertRaises(ValueError):rolling(t,f,y,m+1,r,tf,self.spec)


if __name__=='__main__':unittest.main()
