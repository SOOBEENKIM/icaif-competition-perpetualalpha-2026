import unittest
import numpy as np
from .relationships import evaluate
from auto_pair_selection.selection import IDS


class RelationshipsTests(unittest.TestCase):
    def test_causality_and_all_pairs(self):
        rng=np.random.default_rng(123);t=np.arange(100*24+1)*3600
        common=rng.normal(0,.002,(len(t),1));r=common+rng.normal(0,.001,(len(t),10))
        close=100*np.exp(np.cumsum(r,axis=0));d=dict(market_times=t,close=close,volume=np.full_like(close,1000),trades=np.ones_like(close))
        source=95*86400;a=evaluate(d,source)
        self.assertEqual(len(a['pairs']),45);self.assertEqual(len(a['stable']),45)
        future=t>source;d['close'][future]=np.nan;d['volume'][future]=-1
        self.assertEqual(evaluate(d,source),a)
        self.assertTrue(any(a['mutual']))
        self.assertTrue(all(not x or y for x,y in zip(a['mutual'],a['stable'])))

    def test_selection_is_equivariant_to_asset_identity(self):
        rng=np.random.default_rng(3);t=np.arange(92*24+1)*3600
        r=rng.normal(0,.001,(len(t),10))+rng.normal(0,.002,(len(t),1))
        close=100*np.exp(np.cumsum(r,axis=0));d=dict(market_times=t,close=close,volume=np.full_like(close,1000),trades=np.ones_like(close))
        a=evaluate(d,91*86400);perm=np.array([7,9,2,8,4,1,0,3,6,5])
        e={k:(v if k=='market_times' else v[:,perm]) for k,v in d.items()};b=evaluate(e,91*86400)
        before={frozenset(ids):selected for ids,selected in zip(IDS,a['mutual'])}
        for ids,selected in zip(IDS,b['mutual']):self.assertEqual(selected,before[frozenset(perm[list(ids)])])


if __name__=='__main__':unittest.main()
