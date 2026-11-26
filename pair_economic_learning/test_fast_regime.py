import unittest
import numpy as np
from .fast_regime import fit


class RegimeTests(unittest.TestCase):
    def test_nonmatured_target_and_invalid_pair_cannot_change_selection(self):
        rng=np.random.default_rng(33);t=np.arange(1000)*3600;ar=rng.normal(size=(1000,4));tr=rng.normal(size=(1000,4));y=rng.normal(size=(1000,4));v=np.ones_like(y);good=np.ones_like(y,bool)
        source=800*3600;a,log=fit(t,ar,tr,y,v,good,source,24)
        y[t+24*3600>=source]=1e9
        b,log2=fit(t,ar,tr,y,v,good,source,24)
        np.testing.assert_array_equal(a,b);self.assertEqual(log,log2);self.assertLess(log['last_maturity'],source)
        good[:,3]=False;ar[:,3]=np.nan;tr[:,3]=np.nan;y[:,3]=np.nan
        c,_=fit(t,ar,tr,y,v,good,source,24);self.assertTrue(np.isfinite(c).all())


if __name__=='__main__':unittest.main()
