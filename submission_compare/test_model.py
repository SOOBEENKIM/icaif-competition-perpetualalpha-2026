import unittest
import numpy as np
from .model import candidates


class ModelTests(unittest.TestCase):
    def test_extra_short_needs_all_confirmations_and_keeps_other_assets(self):
        base=np.full((2,10),.02);short=-base;pred=-base;prob=np.full_like(base,.2)
        pred[0,0]=.01;prob[0,1]=.8
        out=candidates(base,short,pred,prob,base*1.1)
        np.testing.assert_array_equal(out['confirmed_short'][0,:3],[.02,.02,-.02])
        np.testing.assert_array_equal(out['confirmed_short'][:,3:],base[:,3:])

    def test_filter_reduces_conflicting_long_and_short_without_rescaling(self):
        base=np.full((1,10),.02);base[0,1]=-.02
        pred=np.full_like(base,-.01);pred[0,1]=.01
        result=candidates(base,-abs(base),pred,np.full_like(base,.5),base)['forecast_filter']
        np.testing.assert_array_equal(result[0,:3],[0,0,0])
        np.testing.assert_array_equal(result[:,3:],base[:,3:])

    def test_future_rows_do_not_change_prior_targets(self):
        rng=np.random.default_rng(21)
        arrays=[rng.normal(0,.02,(120,10)) for _ in range(5)]
        arrays[3]=rng.random((120,10))
        full=candidates(*arrays);prefix=candidates(*(x[:60] for x in arrays))
        for name in full:np.testing.assert_array_equal(full[name][:60],prefix[name])


if __name__=='__main__':unittest.main()
