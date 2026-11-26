import unittest
import numpy as np
from .model import Spec,training_rows,fit_trees,fit_mixture,prepare


class EconomicLearningTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(23);t=np.arange(700)*3600
        shape=(len(t),3);x=rng.normal(size=(*shape,17));y=.4*x[:,:,0]-.2*x[:,:,1]
        return dict(times=t,maturity=t+72*3600,x=x,y=y,ar=x[:,:,0],trend=x[:,:,1],unit=np.ones(shape),valid=np.ones(shape,bool),terminal=np.ones(shape,bool))

    def test_boundary_maturity_and_future_labels_never_enter_fit(self):
        f=self.fixture();source=600*3600;spec=Spec(min_origins=20,tree_rounds=8)
        rows=training_rows(f,source,spec);self.assertTrue(np.all(f['maturity'][rows[0]]<source))
        a,ma=fit_trees(f,source,spec);mix=fit_mixture(f,source,spec)
        f['y'][f['maturity']>=source]=1e9;f['x'][f['times']>=source]=1e9
        b,mb=fit_trees(f,source,spec)
        for k in a:self.assertEqual(bytes(a[k].save_raw()),bytes(b[k].save_raw()))
        self.assertEqual(mix,fit_mixture(f,source,spec));self.assertEqual(ma,mb)

    def test_future_source_beta_does_not_define_labels(self):
        # Explicit physical hedge definition; no differencing of refitted residuals.
        p0=np.array([100.,80.]);p1=np.array([103.,81.]);bt=1.2
        q=np.array([1/(1+bt)/p0[0],-bt/(1+bt)/p0[1]])
        y=((p1[0]/p0[0]-1)-bt*(p1[1]/p0[1]-1))/(1+bt)
        self.assertAlmostEqual(y,float(q@(p1-p0)))

    def test_calibration_weight_is_bounded_and_only_matured(self):
        f=self.fixture();m=fit_mixture(f,600*3600)
        self.assertTrue(0<=m['weight']<=1);self.assertLess(m['last_maturity'],600*3600)

    def test_origin_weights_do_not_count_each_pair_as_independent_day(self):
        f=self.fixture();ii,kk,w=training_rows(f,600*3600,Spec(min_origins=20))
        for i in np.unique(ii):
            expected=2**(-(600*3600-f['maturity'][i])/(86400*30))
            self.assertAlmostEqual(float(w[ii==i].sum()),expected)

    def test_features_and_labels_preserve_prefix(self):
        from pair_value_learning.study import INPUT
        raw=dict(np.load(INPUT));d={k:raw[k][:2200].copy() for k in ['market_times','close','volume','trades']}
        a,_=prepare(d);cut=2100;d['close'][cut+1:]*=np.linspace(2,5,10)
        b,_=prepare(d)
        np.testing.assert_allclose(a['x'][:cut+1],b['x'][:cut+1],atol=1e-11,equal_nan=True)
        np.testing.assert_allclose(a['y'][:cut-72+1],b['y'][:cut-72+1],atol=1e-11,equal_nan=True)


if __name__=='__main__':unittest.main()
