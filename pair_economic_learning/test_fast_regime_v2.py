import unittest
import numpy as np
from .fast_regime_v2 import training_validity,fit


class RegimeValidityTests(unittest.TestCase):
    def test_future_model_rejection_cannot_censor_source_labels(self):
        d={'volume':np.ones((200,10)),'trades':np.ones((200,10))};f={'valid':np.ones((200,45),bool)}
        a=training_validity(d,f,24);f['valid'][100:]=False;b=training_validity(d,f,24)
        np.testing.assert_array_equal(a[:100],b[:100])
        self.assertTrue(b[99].all())

    def test_nontraded_endpoint_excludes_label(self):
        d={'volume':np.ones((200,10)),'trades':np.ones((200,10))};f={'valid':np.ones((200,45),bool)}
        d['trades'][100]=0;a=training_validity(d,f,24)
        self.assertFalse(a[76].any());self.assertFalse(a[-24:].any())

    def test_only_strictly_matured_labels_enter_selection(self):
        rng=np.random.default_rng(33);t=np.arange(1000)*3600;ar=rng.normal(size=(1000,4));tr=rng.normal(size=(1000,4));y=rng.normal(size=(1000,4));v=np.ones_like(y);good=np.ones_like(y,bool)
        source=800*3600;a,log=fit(t,ar,tr,y,v,good,source,24);y[t+24*3600>=source]=1e9
        b,log2=fit(t,ar,tr,y,v,good,source,24)
        np.testing.assert_array_equal(a,b);self.assertEqual(log,log2);self.assertLess(log['last_maturity'],source)


if __name__=='__main__':unittest.main()
