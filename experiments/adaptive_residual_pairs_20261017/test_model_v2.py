import copy,json,unittest
from pathlib import Path
import numpy as np
from .model_v2 import fit_correction,adapt,build
from experiments.dense_history_pairs_20261017.model import design

SPEC=json.loads((Path(__file__).resolve().parent/'model_protocol.json').read_text())


class CalibrationTests(unittest.TestCase):
    def test_bias_correction_has_prior_and_effective_count(self):
        n=20;now=10000000;p=np.full(n,.2);y=p+.5;m=now-np.arange(n,0,-1)*86400
        r=fit_correction(p,y,m,now,SPEC);n_eff=r['effective_rows']
        self.assertAlmostEqual(r['bias'],.5*n_eff/(n_eff+SPEC['prior_rows']))
        self.assertLess(n_eff,n);self.assertLess(r['last_maturity'],now)

    def test_affine_slope_changes_when_past_predictions_have_wrong_direction(self):
        now=10000000;p=np.tile([-2.,-1.,1.,2.],25);m=now-np.arange(100,0,-1)*86400
        r=fit_correction(p,-p,m,now,SPEC)
        self.assertLess(r['affine_slope'],0);self.assertAlmostEqual(r['affine_intercept'],0,delta=.1)

    def test_maturity_boundary_is_rejected(self):
        with self.assertRaises(ValueError):fit_correction(np.ones(12),np.ones(12),np.arange(12),11,SPEC)

    def test_too_few_observations_does_not_invent_forecast(self):
        self.assertIsNone(fit_correction(np.ones(11),np.ones(11),np.arange(11),20,SPEC))

    def test_future_targets_do_not_change_earlier_decisions(self):
        n=2200;h=24;t=np.arange(n)*3600;raw=np.tile(np.sin(t/1e5)[:,None],(1,2));y=raw+.3
        fresh=np.ones(raw.shape,bool);unit=np.full(raw.shape,.01);m=t+h*3600
        a,ar=adapt(t,y,m,raw,fresh,unit,h,SPEC);cut=1600*3600
        y[m>=cut]=100;raw[t>=cut]*=-2;fresh[t>=cut]=False
        b,br=adapt(t,y,m,raw,fresh,unit,h,SPEC)
        for name in a:np.testing.assert_array_equal(a[name][t<cut],b[name][t<cut])
        self.assertEqual([x for x in ar if x['time']<cut],[x for x in br if x['time']<cut])
        for r in ar:self.assertLess(r['last_maturity'],r['time'])

    def test_nonoverlap_and_recent_forecast_ignore_pretrained_values(self):
        n=2000;h=72;t=np.arange(n)*3600;raw=np.ones((n,1));y=raw*.2;fresh=np.ones(raw.shape,bool);unit=raw*.01
        a,records=adapt(t,y,t+h*3600,raw,fresh,unit,h,SPEC)
        b,_=adapt(t,y,t+h*3600,raw*-3,fresh,unit,h,SPEC)
        np.testing.assert_array_equal(a['recent'],b['recent']);self.assertTrue(records)
        for r in records:
            self.assertEqual(r['first_source']%(h*3600),0)
            self.assertEqual((r['last_source']-r['first_source'])//(h*3600)+1,r['rows'])

    def test_real_feature_pipeline_is_prefix_invariant(self):
        rng=np.random.default_rng(1706);n=2600;t=np.arange(n)*3600
        prices=100*np.exp(rng.normal(0,.002,(n,2)).cumsum(0));v=np.full_like(prices,10.)
        ref=dict(market_times=t,close=prices,volume=v,trades=v)
        d=design(t,prices,v,v,['BTC','ETH'],[['BTC','ETH']]);count=len(d['names'])
        record=dict(names=d['names'],center=np.zeros(count),scale=np.ones(count),coefficient=np.r_[0.,np.ones(count)*.01])
        models={(('BTC','ETH'),24):record}
        a,ar=build(ref,d,models,[['BTC','ETH']],24,SPEC);cut=2200
        rr={k:x[:cut] for k,x in ref.items()};dd=design(t[:cut],prices[:cut],v[:cut],v[:cut],['BTC','ETH'],[['BTC','ETH']])
        b,br=build(rr,dd,models,[['BTC','ETH']],24,SPEC)
        for name in a:np.testing.assert_array_equal(a[name][:cut],b[name])
        self.assertEqual([x for x in ar if x['time']<=t[cut-1]],br)


if __name__=='__main__':unittest.main()
