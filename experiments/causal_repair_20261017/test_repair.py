import copy
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from .forecast import features,labels,training_mask,admission_evidence,timestamp
from .policy import AnchoredController,path_calibrate
from .data import save
from experiments.cost_horizon_20261016.research import data,SecondController


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inp,cls.arr,cls.protocol,cls.market=data()

    def test_features_have_no_future_dependency(self):
        a={k:self.inp[k][:,0] for k in ('open','high','low','close','volume','trades')}
        x,v,_=features(a);n=2000
        xp,vp,_=features({k:z[:n] for k,z in a.items()})
        np.testing.assert_allclose(x[:n],xp,atol=0,rtol=0,equal_nan=True)
        np.testing.assert_allclose(v[:n],vp,atol=0,rtol=0,equal_nan=True)

    def test_economic_results_with_numpy_flags_are_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            save(path,{'gates':{'accounting':np.bool_(True)},'equity':np.float64(1000000.),'qty':np.zeros(3)})
            self.assertEqual(json.loads(path.read_text()),{'gates':{'accounting':True},'equity':1000000.,'qty':[0.,0.,0.]})

    def test_training_purges_unmatured_labels(self):
        n=9000;times=np.arange(n)*3600+timestamp('2025-01-01')
        a={'times':times,'close':np.exp(np.arange(n)*.00001)}
        y,mt=labels(a,np.ones(n)*.01)
        cutoff=timestamp('2026-01-01')
        mask=training_mask(a,np.ones((n,2)),y,mt,cutoff)
        self.assertLess(int(mt[mask].max()),cutoff)
        self.assertTrue(np.all(times[mask]+73*3600<cutoff))
        a['times']=a['times']+3600
        self.assertIsNone(training_mask(a,np.ones((n,2)),y,mt,cutoff))

    def test_future_outcomes_cannot_change_asset_gate(self):
        t=np.arange(100)*72*3600;cutoff=int(t[80]);p=np.sin(np.arange(100))*.02;y=p.copy();m=t+73*3600
        cfg={'lookback_days':60,'minimum_nonoverlapping_blocks':16,'round_trip_cost_bps':38}
        old=admission_evidence(t,p,y,m,cutoff,cfg)
        y[m>=cutoff]=100
        self.assertEqual(old,admission_evidence(t,p,y,m,cutoff,cfg))
        self.assertLess(old['last_maturity'],cutoff)

    def test_bad_asset_cannot_borrow_good_asset_evidence(self):
        t=np.arange(100)*72*3600;p=np.sin(np.arange(100))*.02;m=t+73*3600
        cfg={'lookback_days':60,'minimum_nonoverlapping_blocks':16,'round_trip_cost_bps':38}
        self.assertTrue(admission_evidence(t,p,p,m,int(t[80]),cfg)['enabled'])
        self.assertFalse(admission_evidence(t,p,-p,m,int(t[80]),cfg)['enabled'])

    def test_calibration_uses_sequential_risk_not_return_or_monotonicity(self):
        calls=[]
        def evaluate(g):
            calls.append(g)
            return {'mean_expected_daily_volatility':{50:.002,100:.003,200:.0005}[g],
                    'engineering_pass':True,'roi':{50:.2,100:.3,200:-.1}[g]}
        c=path_calibrate(evaluate,.0006,[50,100,200])
        self.assertEqual(c['gamma'],200);self.assertEqual(calls,[50,100,200])

    def test_no_admissible_risk_means_no_candidate(self):
        c=path_calibrate(lambda g:{'mean_expected_daily_volatility':.1,'engineering_pass':True},.0005,[50,100])
        self.assertIsNone(c['gamma']);self.assertTrue(c['fallback'])

    def test_rejected_alpha_preserves_baseline_exactly(self):
        n=len(self.market.times);elig=np.zeros((n,3),bool);mu=np.full((n,3),.1)
        new=AnchoredController(self.market.times,self.arr['retention'],mu,elig,np.full(n,50.))
        old=SecondController(self.market.times,self.arr['retention'])
        ix=np.flatnonzero((self.market.times>=timestamp('2026-08-01'))&(self.market.times%21600==0))[:8]
        qty=np.zeros(10)
        for i in ix:
            args=(int(self.market.times[i]),self.market.close[i],qty,1e6,self.arr['candidate'][i],self.arr['sigma'][i],self.market.volume[i],self.market.trades[i],self.protocol)
            a=old.propose(*args,buffer=.2);b=new.propose(*args,buffer=.2)
            self.assertEqual(a['target_qty'],b['target_qty']);qty=np.array(a['target_qty'])

    def test_overlay_cannot_change_other_markets_or_increase_target_risk(self):
        n=len(self.market.times);rng=np.random.default_rng(10);mu=rng.normal(0,.03,(n,3));elig=np.ones((n,3),bool)
        new=AnchoredController(self.market.times,self.arr['retention'],mu,elig,np.full(n,200.))
        old=SecondController(self.market.times,self.arr['retention']);qty=np.zeros(10)
        ix=np.flatnonzero((self.market.times>=timestamp('2026-08-01'))&(self.market.times%21600==0))[:50]
        for i in ix:
            args=(int(self.market.times[i]),self.market.close[i],qty,1e6,self.arr['candidate'][i],self.arr['sigma'][i],self.market.volume[i],self.market.trades[i],self.protocol)
            a=old.propose(*args,buffer=.2);b=new.propose(*args,buffer=.2)
            np.testing.assert_array_equal(a['target_qty'][3:],b['target_qty'][3:])
            if b['alpha_reason']=='positive_incremental_net_utility':
                self.assertLessEqual(b['candidate_expected_variance'],b['baseline_expected_variance']+1e-14)
                self.assertGreater(b['alpha_utility_improvement'],0)
            qty=np.array(b['target_qty'])

    def test_restart_does_not_repeat_completed_round_and_halt_overrides_alpha(self):
        n=len(self.market.times);elig=np.ones((n,3),bool);mu=np.ones((n,3))*.05
        i=self.market.exact(timestamp('2026-08-01'));t=int(self.market.times[i]);qty=np.array([.3,0,100,0,0,0,20,0,0,0])
        args=(t,self.market.close[i],qty,1e6,self.arr['candidate'][i],self.arr['sigma'][i],self.market.volume[i],self.market.trades[i],self.protocol)
        c=AnchoredController(self.market.times,self.arr['retention'],mu,elig,np.full(n,200.))
        a=c.propose(*args,halted=True);self.assertTrue(a['risk_forced']);self.assertTrue(np.all(np.array(a['target_qty'])==0))
        c2=AnchoredController(self.market.times,self.arr['retention'],mu,elig,np.full(n,200.),saved=c.snapshot())
        self.assertIsNone(c2.propose(*args))


if __name__=='__main__':unittest.main()
