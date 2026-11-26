"""Guard missing observations and boundary semantics in the append adapter."""
import unittest
import numpy as np
from .prepare import candle_values,funding_values,exact_prefix

class DataContract(unittest.TestCase):
    def test_incomplete_or_missing_candle_rejected(self):
        row=dict(t=0,T=299999,s='par:BTC',c='5',v='0',n=0)
        np.testing.assert_array_equal(candle_values([row],[300],'par:BTC'),[[5,0,0]])
        with self.assertRaises(ValueError):candle_values([row],[300,600],'par:BTC')
        with self.assertRaises(ValueError):candle_values([{**row,'T':299998}],[300],'par:BTC')
    def test_duplicate_or_wrong_market_rejected(self):
        row=dict(t=0,T=299999,s='par:BTC',c='5',v='0',n=0)
        with self.assertRaises(ValueError):candle_values([row,row],[300],'par:BTC')
        with self.assertRaises(ValueError):candle_values([row],[300],'par:ETH')
    def test_known_zero_funding_distinct_from_missing(self):
        row=dict(time=3600000,coin='par:BTC',fundingRate='0')
        np.testing.assert_array_equal(funding_values([row],[3600],'par:BTC'),[0])
        with self.assertRaises(ValueError):funding_values([row],[3600,7200],'par:BTC')
        with self.assertRaises(ValueError):funding_values([{**row,'time':3600001}],[3600],'par:BTC')
    def test_revision_rejected_and_nan_prefix_preserved(self):
        old={'x':np.array([1.,np.nan])}
        exact_prefix({'x':np.array([1.,np.nan,3.])},old)
        with self.assertRaises(AssertionError):exact_prefix({'x':np.array([2.,np.nan,3.])},old)

if __name__=='__main__':unittest.main()
