import unittest
from dataclasses import replace
import numpy as np
from .model import Spec,fit_day,predict,features


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(13);n=600;q=45;t=np.arange(n)*3600
        x=rng.normal(size=(n,q,8));y=.12*x[:,:,0]-.15*x[:,:,1]
        return dict(times=t,x=x,y=y,maturity=t+72*3600,fresh=np.ones((n,q),bool),terminal=np.ones((n,q),bool),allowed=np.ones((n,q),bool),vol=np.full((n,q),.001))

    def test_only_strictly_matured_labels_can_change_fit(self):
        f=self.fixture();s=replace(Spec(),source_stride_hours=1);source=500*3600
        m=fit_day(f,source,s);g={k:v.copy() for k,v in f.items()}
        g['y'][g['maturity']>=source]=1e6;g['x'][g['times']>=source]=1e6
        z=fit_day(g,source,s)
        np.testing.assert_array_equal(m['coefficient'],z['coefficient'])
        for k in m['local']:np.testing.assert_array_equal(m['local'][k],z['local'][k])
        self.assertLess(m['last_maturity'],source)

    def test_calibration_corrects_exaggerated_raw_forecast(self):
        f=self.fixture();s=replace(Spec(),source_stride_hours=1);m=fit_day(f,500*3600,s);ii=np.arange(500,600)
        calibrated=predict(f,ii,m,'hierarchical',72,s)/(.001*np.sqrt(72))
        self.assertLess(np.mean(abs(calibrated-f['y'][ii])),np.mean(abs(f['x'][ii,:,0]-f['y'][ii])))

    def test_missing_pair_shrinks_to_shared_model(self):
        f=self.fixture();f['allowed'][:,3]=False;m=fit_day(f,500*3600,replace(Spec(),source_stride_hours=1))
        self.assertNotIn(3,m['local']);ii=np.arange(500,520)
        np.testing.assert_array_equal(predict(f,ii,m,'hierarchical',72)[:,3],predict(f,ii,m,'pooled',72)[:,3])

    def test_labels_hold_original_beta_fixed(self):
        # Identity is the physical two-leg return, rather than the difference
        # between residuals fitted with two different betas.
        a0,b0,a1,b1,beta=100.,50.,110.,49.,.6
        q=np.array([1.,-beta*a0/b0]);gross=abs(q)@np.array([a0,b0])
        measured=q@(np.array([a1,b1])-np.array([a0,b0]))/gross
        label=((a1/a0-1)-beta*(b1/b0-1))/(1+beta)
        self.assertAlmostEqual(measured,label)

    def test_no_training_when_maturities_are_in_future(self):
        f=self.fixture();f['maturity'][:]=10**15
        self.assertIsNone(fit_day(f,500*3600))

    def test_all45_pair_labels_and_features_are_causal(self):
        rng=np.random.default_rng(7);n=1800;t=np.arange(n)*3600
        common=np.cumsum(rng.normal(0,.002,n));x=common[:,None]+np.cumsum(rng.normal(0,.0008,(n,10)),axis=0)
        d=dict(market_times=t,close=np.exp(x+5),volume=np.ones((n,10)),trades=np.ones((n,10)))
        a,_=features(d,72);prefix={k:v[:1700] for k,v in d.items()};b,_=features(prefix,72)
        self.assertEqual(a['x'].shape,(n,45,8))
        for k in ['x','vol','beta','allowed','fresh']:np.testing.assert_array_equal(a[k][:1700],b[k])
        np.testing.assert_array_equal(a['y'][:1628],b['y'][:1628])


if __name__=='__main__':unittest.main()
