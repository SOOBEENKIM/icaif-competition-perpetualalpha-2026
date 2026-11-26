import unittest
import numpy as np
from .opportunities import fit,predict,slope


class OpportunityTests(unittest.TestCase):
    def test_future_poison_does_not_affect_fit(self):
        rng=np.random.default_rng(34);n=250;t=np.arange(n)*21600;m=t+86400;source=t[200]
        x=rng.normal(size=(n,5,3));raw=rng.normal(0,.02,(n,5));unit=np.full_like(raw,.02);y=raw*.5+rng.normal(0,.02,raw.shape)
        spec=dict(lookback_days=90,minimum_rows=50,minimum_sources=30,ridge_penalty=1.)
        a,ta=fit(t,x,raw,unit,y,m,source,spec);ca,na=slope(t,raw,unit,y,m,source)
        future=m>=source;x[future]=np.nan;y[future]=1e9;raw[future]=np.nan
        b,tb=fit(t,x,raw,unit,y,m,source,spec);cb,nb=slope(t,raw,unit,y,m,source)
        self.assertEqual(a,b);self.assertEqual(ta.save_raw(),tb.save_raw());np.testing.assert_equal(ca,cb);self.assertEqual(na,nb)
        self.assertLess(a['last_maturity'],source)
        aa,bb=predict(x[:10],raw[:10],unit[:10],a,ta)
        for p in [aa,bb]:
            self.assertTrue(np.all(abs(p)<=abs(raw[:10])+1e-12));self.assertTrue(np.all(p*raw[:10]>=0))


if __name__=='__main__':unittest.main()
