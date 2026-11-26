import copy
import unittest
import numpy as np
from .models import fit,predict,training_rows
from .study import SPEC


def fixture():
    rng=np.random.default_rng(444);n=24*120;q=45
    t=np.arange(n)*3600;x=rng.normal(size=(n,q,4))
    f=dict(x=x,fresh=np.ones((n,q),bool),vol=np.full((n,q),.003),names=['a','b','c','d'])
    y=.4*x[:,:,0]-.2*x[:,:,1]+rng.normal(0,.2,(n,q))
    return t,f,y,t+72*3600,np.ones((n,q),bool)


class SharedModelsTests(unittest.TestCase):
    def test_future_labels_and_features_do_not_change_fit(self):
        t,f,y,m,terminal=fixture();source=100*86400
        spec=dict(SPEC,tree=dict(SPEC['tree'],rounds=8))
        a,at=fit(t,f,y,m,terminal,source,spec)
        g=copy.deepcopy(f);g['x'][m>=source]*=1000
        poison=y.copy();poison[m>=source]=1e9
        b,bt=fit(t,g,poison,m,terminal,source,spec)
        self.assertEqual(a,b);self.assertEqual(at.save_raw(),bt.save_raw())
        self.assertLess(a['last_maturity'],source)
        self.assertEqual(a['represented_pairs'],list(range(45)))

    def test_timestamp_weights_do_not_count_45_pairs_as45_days(self):
        t,f,y,m,terminal=fixture();f['fresh'][:,10:]=False
        x,yy,w,ii,kk=training_rows(t,f,y,m,terminal,100*86400,SPEC)
        self.assertAlmostEqual(float(w.sum()),len(np.unique(ii)))
        self.assertTrue(all(np.allclose(w[ii==i],w[ii==i][0]) for i in np.unique(ii)))

    def test_both_models_emit45_forecasts_and_stale_pair_is_unavailable(self):
        t,f,y,m,terminal=fixture();spec=dict(SPEC,tree=dict(SPEC['tree'],rounds=8))
        model,tree=fit(t,f,y,m,terminal,100*86400,spec)
        ix=np.array([2400,2401]);f['fresh'][2400,44]=False
        a,b=predict(f,ix,model,tree,72,spec)
        self.assertEqual(a.shape,(2,45));self.assertTrue(np.isnan(a[0,44]));self.assertTrue(np.isnan(b[0,44]))
        self.assertTrue(np.isfinite(a[1]).all());self.assertTrue(np.isfinite(b[1]).all())
        self.assertGreater(float(np.corrcoef(a[1],y[2401])[0,1]),.8)


if __name__=='__main__':unittest.main()
