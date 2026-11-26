import unittest
import numpy as np
from .sma_calibration import coefficients


class SourceCalibrationTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(180*24)*3600;rng=np.random.default_rng(8);x=rng.normal(0,.01,(len(t),45));unit=np.full_like(x,.02);y=.4*x
        return t,x,unit,y,np.ones_like(x,bool),150*86400

    def test_unmatured_targets_do_not_change_coefficients(self):
        t,x,u,y,f,s=self.fixture();a=coefficients(t,x,u,y,f,s)
        y[t+72*3600>=s]=1e10;x[t>=s]=1e10;b=coefficients(t,x,u,y,f,s)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_invalid_pair_history_cannot_poison_other_pairs(self):
        t,x,u,y,f,s=self.fixture();y[:,4]=np.nan;a=coefficients(t,x,u,y,f,s)
        self.assertTrue(np.isfinite(a[0]));self.assertTrue(np.isfinite(a[1]).all())

    def test_calibrated_size_cannot_exceed_original_forecast(self):
        a=coefficients(*self.fixture());self.assertTrue(0<=a[0]<=1);self.assertTrue(np.all((a[1]>=0)&(a[1]<=1)))


if __name__=='__main__':unittest.main()
