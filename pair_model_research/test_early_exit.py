import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from .early_exit import reassess
from .stable_reversion import forecasts


class EarlyExitTests(unittest.TestCase):
    def test_confirmation_and_duplicate_source(self):
        t=np.arange(4,8)*3600;d=dict(times=t,par_close=np.ones((4,2))*100)
        de=dict(signal_times=t[:,None],directions=np.ones((4,1)),source_prices=d['par_close'],mu=np.ones((4,1))*.01)
        pair=dict(pair_index=0,ids=[0,1],first_fill=0,horizon=72);ledger=SimpleNamespace(qty=np.array([1,-1.]));state=dict(reason=None)
        with patch('pair_model_research.early_exit.exit_value',return_value=dict(expected_hold_cash=-20,immediate_close_cash=-1)):
            a=reassess(state,pair,ledger,0,d,de,dict(early_value_exit=True),{},{});self.assertIsNone(a['reason'])
            a=reassess(state,pair,ledger,0,d,de,dict(early_value_exit=True),{},{});self.assertIsNone(a['reason'])
            a=reassess(state,pair,ledger,1,d,de,dict(early_value_exit=True),{},{});self.assertEqual(a['reason'],'fresh_forecast_negative_holding_value')
        self.assertEqual(pair['negative_value_reviews'],2)

    def test_existing_stop_takes_priority(self):
        s=dict(reason='pair_return_stop')
        self.assertIs(reassess(s,{},None,0,{}, {},dict(early_value_exit=True),{},{}),s)

    def test_disjoint_model_price_cutoff(self):
        rng=np.random.default_rng(7);n=1900;t=np.arange(n)*3600;c=100*np.exp(np.cumsum(rng.normal(0,.003,(n,10)),axis=0))
        d=dict(market_times=t,close=c,volume=np.ones_like(c)*1000,trades=np.ones_like(c))
        a,_,e,records=forecasts(d);end=1700;prefix={k:v[:end] for k,v in d.items()};b,_,eb,_=forecasts(prefix)
        np.testing.assert_allclose(a[:end],b,equal_nan=True);np.testing.assert_equal(e[:end],eb)
        for r in records:
            source=r['source']//3600
            if r['current']:self.assertLessEqual(r['current']['last_price_index'],source)
            if r['previous']:self.assertLessEqual(r['previous']['last_price_index'],source-720)


if __name__=='__main__':unittest.main()
