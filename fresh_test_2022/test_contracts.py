import copy,unittest
import numpy as np
from .models import candidate,incumbent
from .data import ROOT
from pair_value_learning.study import INPUT
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount


class FreshTestContracts(unittest.TestCase):
    def test_future_quotes_do_not_change_past_forecasts_or_daily_fits(self):
        # Only previously exposed2026prices. Never inspect2022testreturns here.
        raw=dict(np.load(INPUT));start=1785542400;cut=1785628800
        n=int(np.searchsorted(raw['market_times'],cut+48*3600))+1
        d={k:raw[k][:n,:2].copy() for k in ['close','volume','trades']};d['market_times']=raw['market_times'][:n]
        changed={k:v.copy() for k,v in d.items()};changed['close'][d['market_times']>cut]*=np.array([3.,7.])
        before=d['market_times']<=cut
        for fn in [candidate,incumbent]:
            s,a=fn(d,start);z,b=fn(changed,start)
            for k in ['mu','beta','vol','economic_available','economic_action']:
                np.testing.assert_allclose(s[k][before],z[k][before],atol=1e-12,rtol=0,equal_nan=True)
            aa=[r for r in a if r['source']<=cut];bb=[r for r in b if r['source']<=cut]
            self.assertEqual(aa,bb)
            if fn==incumbent:
                for r in a:
                    for m in ['direct','gate']:
                        if r[m] is not None:self.assertLess(r[m]['last_maturity'],r['source'])

    def test_funding_is_cash_only_at_verified_settlement(self):
        a=SharedAccount(50000.,('BTC','ETH'))
        a.physical.qty[:]=[1.,-2.];a.owners['pair'].qty[:]=[1.,-2.]
        before=a.physical.cash
        for h in range(1,9):
            rate=np.array([.001,.002]) if h==8 else np.zeros(2)
            a.charge_funding(h*3600,np.array([100.,50.]),rate,np.ones(2,bool))
        self.assertAlmostEqual(a.physical.cash-before,.1)
        self.assertAlmostEqual(a.physical.funding,.1)

    def test_isolated_executor_changes_only_array_dimension(self):
        source=(ROOT/'pair_regime_research/replay.py').read_text()
        adapter=(ROOT/'fresh_test_2022/replay.py').read_text()
        self.assertEqual(source.replace('np.zeros(10)','np.zeros(len(prices))'),adapter)


if __name__=='__main__':unittest.main()
