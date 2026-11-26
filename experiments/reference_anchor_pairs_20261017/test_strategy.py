import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import experiments.par_direct_forecast_20261017.test_direct as fixtures
from replay_lab.common_comparison.models import Market
from .research import HERE,ROOT,INPUT
from .engine import replay
from .model import design,labels,fit_predict,convert_to_par
from .quote_probe import decide
from experiments.par_direct_forecast_20261017.data import ts
from experiments.pair_execution_calibration_20261017.test_lifecycle import stream


class StrategyTests(fixtures.DirectTests):
    def setUp(self):
        super().setUp();cfg=json.loads((HERE/'protocol.json').read_text())
        cfg['pairs']=self.cfg['pairs'];cfg['pair_max_gross']=1000;self.cfg=cfg
        self.de['forecast_times']=self.d['times'].copy()

    def run_replay(self,arm='ridge_6',overlay=False):
        return replay(self.d,self.de,self.pred,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def reference(self):
        cfg=json.loads((HERE/'protocol.json').read_text())
        with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:r={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(r['known'],r['market_times'])
        self.assertTrue(np.all(r['market_times']==r['times']+3600))
        return Market(*(r[k] for k in ('market_times','close','volume','trades','funding','observed')),tuple(self.p['symbols']))

    def test_labels_include_price_only_and_source_anchor(self):
        m=self.reference();cfg=json.loads((HERE/'protocol.json').read_text());de=design(m,cfg)
        y,t=labels(m,de,24);changed=copy.deepcopy(m);changed.funding[:]=999.
        yy,tt=labels(changed,de,24);np.testing.assert_allclose(y,yy,equal_nan=True);np.testing.assert_array_equal(t,tt)
        i=2000;a,b=de['pairs'][0];beta=de['beta'][i,0]
        expected=(m.close[i+24,a]/m.close[i,a]-1-beta*(m.close[i+24,b]/m.close[i,b]-1))/(1+beta)
        self.assertAlmostEqual(y[i,0]*de['vol'][i,0]*np.sqrt(24),expected)
        self.assertEqual(t[i],m.times[i]+24*3600)

    def test_change_of_anchor_preserves_predicted_cash_and_units(self):
        m=SimpleNamespace(times=np.array([0,3600]),close=np.array([[100.,50.],[102.,51.]]))
        de=dict(pairs=[[0,1]],beta=np.ones((2,1))*2,correlation=np.ones((2,1)),fresh=np.ones((2,1),bool))
        pred={'ridge_6':np.ones((2,1))*.01};err={'ridge_6':np.ones((2,1))*.02}
        d=dict(times=np.array([900,4500]),par_close=np.array([[105.,48.],[98.,53.]]))
        mapped,mu,_=convert_to_par(d,m,de,pred,err)
        for i in range(2):
            pa,pb=d['par_close'][i];ratio=mapped['beta'][i,0]*pa/pb
            self.assertAlmostEqual(ratio,4.)
            cash=(pa+ratio*pb)*mu['ridge_6'][i,0]+pa-ratio*pb
            ra,rb=m.close[i];expected=(ra+ratio*rb)*.01+ra-ratio*rb
            self.assertAlmostEqual(cash,expected)
        np.testing.assert_array_equal(mapped['forecast_times'],[0,3600])

    def test_exit_uses_reference_source_hour_not_new_par_bar(self):
        self.de['forecast_times']=self.d['times']-3600
        r=self.run_replay()
        first_close=next(f['time'] for f in r['fills'] if f['reduce_only'])
        self.assertEqual(first_close,self.d['times'][0]+5*3600)

    def test_quote_maturity_and_funding_use_remaining_forecast_time(self):
        at=int(self.d['times'][0]*1000);self.de['forecast_times']=self.d['times']-1800
        streams={s:stream(times=(at,at+1000,at+2000)) for s in ('ETH','BTC')}
        for symbol,st in streams.items():
            for _,_,ctx in st['ctx']:ctx['funding']='.0001' if symbol=='ETH' else '0'
        view=dict(symbol='ETH',decision=at,side=1,requested_qty=1.,requested_price=100.)
        life=json.loads((HERE.parent/'pair_execution_calibration_20261017/lifecycle_protocol.json').read_text())
        r=decide(view,['ETH','BTC'],0,'ridge_6',streams,{'ETH':4,'BTC':4},self.d,self.de,self.pred,self.p['symbols'],self.cfg,life,life['scenarios']['base'])
        self.assertEqual(r['status'],'selected');self.assertEqual(r['forecast_maturity_ms'],at+int(5.5*3600000))
        self.assertAlmostEqual(r['expected_funding_cash'],-100*.0001*5.5)

    def test_future_truncation_changes_neither_features_nor_predictions(self):
        m=self.reference();cfg=json.loads((HERE/'protocol.json').read_text());cfg['fit_dates']=cfg['fit_dates'][:1]
        cfg['horizons_hours']=[24]
        cut=ts('2026-10-02T12:00:00');mask=m.times<=cut
        short=Market(*(getattr(m,k)[mask] for k in ('times','close','volume','trades','funding','observed')),m.symbols)
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            de,pred,err,records=fit_predict(m,cfg,Path(td)/'full')
            dd,pp,ee,rr=fit_predict(short,cfg,Path(td)/'prefix')
        for k in ('x','beta','correlation','fresh'):np.testing.assert_allclose(de[k][mask],dd[k],rtol=0,atol=0,equal_nan=True)
        for k in pred:
            np.testing.assert_allclose(pred[k][mask],pp[k],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(err[k][mask],ee[k],rtol=0,atol=0,equal_nan=True)
        self.assertEqual(records,rr)
        self.assertTrue(all(r['last_training_maturity']<r['calibration_start']<r['last_calibration_maturity']<r['fit_time'] for r in records))
        self.assertTrue(any(r['training_rows']>1000 for r in records))


if __name__=='__main__':unittest.main()
