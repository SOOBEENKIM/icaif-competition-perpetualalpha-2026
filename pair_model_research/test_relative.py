import unittest
import numpy as np
from auto_pair_selection.test_selection import data
from .relative import forecasts,ar_fit


class RelativeTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_signals(self):
        d=data(40);a,fa,ea,_=forecasts(d,72,'regime');cut=34*24
        changed={k:v.copy() for k,v in d.items()};changed['close'][cut+1:]*=np.arange(1,11)*12
        b,fb,eb,_=forecasts(changed,72,'regime')
        np.testing.assert_allclose(a[:cut+1],b[:cut+1],equal_nan=True)
        np.testing.assert_array_equal(ea[:cut+1],eb[:cut+1])
        self.assertEqual(a.shape[1],45)

    def test_pair_order_symmetry_and_both_directions(self):
        d=data(40);a,_,_,_=forecasts(d,72,'trend')
        other={k:v.copy() for k,v in d.items()}
        for k in ['close','volume','trades']:other[k][:,[0,1]]=other[k][:,[1,0]]
        b,_,_,_=forecasts(other,72,'trend')
        np.testing.assert_allclose(a[:,0],-b[:,0],equal_nan=True,atol=1e-12)
        self.assertTrue(np.any(a>0));self.assertTrue(np.any(a<0))

    def test_ar_forecast_uses_no_future_prices(self):
        rng=np.random.default_rng(1);x=np.zeros(1000)
        for i in range(1,len(x)):x[i]=.97*x[i-1]+rng.normal(0,.01)
        fresh=np.ones(1000,bool);a=ar_fit(x,fresh,800)
        x[801:]=1e8;b=ar_fit(x,fresh,800)
        self.assertEqual(a,b);self.assertIsNotNone(a);self.assertLessEqual(a['last_price_index'],800)


if __name__=='__main__':unittest.main()
