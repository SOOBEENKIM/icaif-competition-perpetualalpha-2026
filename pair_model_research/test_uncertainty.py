import unittest
import numpy as np
from auto_pair_selection.study import read,ROOT
from experiments.rolling_feature_pairs_20261017.model import fit
from .uncertainty import covariance,conservative


class UncertaintyTests(unittest.TestCase):
    def test_covariance_is_psd_and_rejects_unmatured_labels(self):
        rng=np.random.default_rng(42);n=300;t=np.arange(n)*21600;m=t+86400;source=100*86400
        x=rng.normal(size=(n,4));y=x[:,0]*.5+rng.normal(size=n)
        spec=read(ROOT/'competition_live/strategy_plan.json')['direct_spec']
        model=fit(x,y,np.zeros(n),m,source,spec);c=covariance(x,y,t,m,source,model,spec)
        self.assertGreaterEqual(np.linalg.eigvalsh(c['matrix']).min(),-1e-12)
        self.assertLess(c['last_maturity'],source)
        with self.assertRaises(ValueError):covariance(x,y,t,m,86400,model,spec)
        z=np.array([1.,0.,0.,0.,0.]);far=np.array([1.,20.,0.,0.,0.]);v=np.asarray(c['matrix'])
        self.assertGreater(far@v@far,z@v@z)

    def test_valuation_never_inflates_or_reverses_signal(self):
        raw=np.array([.02,-.02,.001,-.001]);se=np.array([.001,.001,.01,.01])
        a=conservative(raw,se,1.)
        np.testing.assert_allclose(a,[.019,-.019,0,0]);self.assertTrue((abs(a)<=abs(raw)).all())


if __name__=='__main__':unittest.main()
