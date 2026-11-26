import unittest
import numpy as np
from .model_v2 import fit_predict

class Contracts(unittest.TestCase):
    def test_future_outcomes_cannot_change_past_fits_or_forecasts(self):
        rng=np.random.default_rng(164);t=np.arange(120)*3600
        price=100*np.exp(np.cumsum(rng.normal(0,.002,(120,10)),axis=0))
        fresh=np.ones_like(price,bool);carry=np.full_like(price,.001);basis=np.zeros_like(price)
        cut=90;changed=price.copy();changed[cut+1:]*=np.linspace(1,4,10)
        for h in [6,24]:
            a,ar=fit_predict(t,price,fresh,carry,basis,h);b,br=fit_predict(t,changed,fresh,carry,basis,h)
            np.testing.assert_allclose(a[:cut+1],b[:cut+1],atol=1e-12,rtol=0,equal_nan=True)
            self.assertEqual([x for x in ar if x['fit_time']<=t[cut]],[x for x in br if x['fit_time']<=t[cut]])
            self.assertTrue(all(r['last_maturity']<r['fit_time'] for r in ar))

    def test_zero_price_returns_do_not_create_price_alpha(self):
        t=np.arange(120)*3600;price=np.ones((120,10))*100
        a,rows=fit_predict(t,price,np.ones_like(price,bool),np.zeros_like(price),np.zeros_like(price),6)
        self.assertTrue(rows);self.assertEqual(np.nanmax(abs(a)),0)

if __name__=='__main__':unittest.main()
