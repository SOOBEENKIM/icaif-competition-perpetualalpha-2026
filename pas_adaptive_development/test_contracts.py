import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from . import study
from .covariance_v2 import construct, targets
from pas_venue_research.shadow import DisplayedLiquidity


class CovarianceContracts(unittest.TestCase):
    def test_original_size_change_and_exit_are_not_deferred(self):
        n=200; t=np.arange(n)*3600
        sigma=np.tile(np.eye(10),(n,1,1)); base=np.zeros((n,10))
        base[168:,0]=.04; base[174:,0]=.08; base[180:,0]=0
        out=targets(base,sigma,t)
        self.assertEqual(out[168,0],.04)
        self.assertEqual(out[174,0],.08)
        np.testing.assert_array_equal(out[180:],0)

    def test_diagonal_covariance_does_not_invent_hedges(self):
        b=np.array([.08,-.03]+[0.]*8)
        np.testing.assert_array_equal(construct(b,np.diag(np.arange(1,11))),b)

    def test_hedge_is_two_sided_on_every_asset_and_reduces_risk(self):
        sigma=.75*np.eye(10)+.25*np.ones((10,10))
        for asset in range(10):
            b=np.zeros(10);b[asset]=.1
            h=construct(b,sigma);short=construct(-b,sigma)
            self.assertAlmostEqual(h[asset],b[asset])
            self.assertTrue(np.all(np.delete(h,asset)<0))
            np.testing.assert_allclose(short,-h)
            self.assertLess(h@sigma@h,b@sigma@b)
            self.assertLessEqual(abs(h-b).sum(),.5*abs(b).sum()+1e-12)

    def test_future_prices_and_targets_cannot_change_past_allocation(self):
        rng=np.random.default_rng(26);n=350
        base=np.zeros((n,10));base[:,0]=rng.uniform(0,.1,n)
        x=rng.normal(size=(n,10,10));sigma=x@x.transpose(0,2,1)+np.eye(10)
        t=np.arange(n)*3600
        full=targets(base,sigma,t)
        np.testing.assert_array_equal(targets(base[:240],sigma[:240],t[:240]),full[:240])
        base[240:]=100;sigma[240:]=np.eye(10)*1e8
        np.testing.assert_array_equal(targets(base,sigma,t)[:240],full[:240])

    def test_real_history_is_causal_and_respects_caps(self):
        from competition_live.profile import load_profile
        root=study.ROOT/'reports/pas_covariance_v2_20261021'
        z=dict(np.load(root/'targets.npz',allow_pickle=False))
        p=load_profile()['trading'];risk=p['risk']
        # Targets are fractions of the strategy sleeve, not the teammate reserve.
        for name in ['hedged','hedged_restored']:
            w=z[name][168:];s=z['sigma'][168:]
            self.assertTrue(np.all(np.isfinite(w)))
            self.assertLessEqual(abs(w).sum(1).max(),.6+1e-10)
            self.assertLessEqual(abs(w.sum(1)).max(),.4+1e-10)
            self.assertLessEqual(abs(w).max(),.15+1e-10)
            volatility=np.sqrt(np.maximum(24*np.einsum('ni,nij,nj->n',w,s,w),0))
            self.assertLessEqual(volatility.max(),.005+1e-10)


class ForecastContracts(unittest.TestCase):
    def test_purged_fitting_calibration_and_features_ignore_future(self):
        rng=np.random.default_rng(26);n=24*125
        start=study.ts('2026-04-01T00:00:00Z');times=start+np.arange(n)*3600
        close=100*np.exp(np.cumsum(rng.normal(0,.003,(n,10)),axis=0))
        d=dict(market_times=times,close=close,volume=np.ones((n,10)),
            trades=np.ones((n,10)),funding=np.zeros((n,10)),observed=np.ones((n,10),bool))
        p=json.loads((study.OUT/'protocol.json').read_text())
        end=study.ts('2026-07-15T00:00:00Z');poison={k:v.copy() for k,v in d.items()}
        poison['close'][times>=end]*=100
        with tempfile.TemporaryDirectory() as directory,patch.object(study,'OUT',Path(directory)):
            a,ra=study.fit_predict(d,p,24,False)
            b,rb=study.fit_predict(poison,p,24,False)
        self.assertEqual(ra,rb)
        np.testing.assert_allclose(a[times<end],b[times<end],equal_nan=True)
        self.assertLess(ra[0]['training_last_label'],ra[0]['train_cutoff'])
        self.assertLess(ra[0]['calibration_last_label'],ra[0]['fit'])
        prefix={k:v[times<end] for k,v in d.items()}
        x,_,_=study.features(d);xp,_,_=study.features(prefix)
        np.testing.assert_allclose(x[times<end],xp,equal_nan=True)


class ExecutionContracts(unittest.TestCase):
    def test_ioc_reserves_depth_and_pays_spread_on_round_trip(self):
        depth=DisplayedLiquidity(.25)
        book=dict(time=1,levels=[[dict(px='99',sz='4')],[dict(px='101',sz='4')]])
        depth.update(book)
        q,p=depth.consume(1,2,101);self.assertEqual(q,1);self.assertEqual(p,101)
        depth.update(dict(book,time=2))
        self.assertEqual(depth.consume(1,2,101)[0],0)
        q2,p2=depth.consume(-1,q,99)
        pnl=q2*(p2-p)-.0009*q2*(p+p2)
        self.assertLess(pnl,0)

    def test_ioc_limit_rejects_adverse_future_price(self):
        depth=DisplayedLiquidity(.25)
        depth.update(dict(time=1,levels=[[dict(px='100',sz='4')],[dict(px='102',sz='4')]]))
        self.assertEqual(depth.consume(1,1,101)[0],0)


if __name__=='__main__':unittest.main()
