import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from .data import HERE,OUT,ts
from .engine import replay
from .model import design,fit_predict

class DirectTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'protocol.json').read_text());self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        self.cfg['pairs']=[['ETH','BTC']];self.cfg['pair_max_gross']=1000
        t=np.arange(ts('2026-10-08T12:00:00'),ts('2026-10-10T12:00:00')+300,300);n=len(t)
        self.d=dict(times=t,par_close=np.full((n,10),100.),reference_close=np.full((n,10),90.),
            par_volume=np.full((n,10),10000.),par_trades=np.full((n,10),50.),funding=np.zeros((n,10)),
            funding_observed=np.tile((t%3600==0)[:,None],(1,10)),known_hourly_funding=np.zeros((n,10)),
            first=np.tile([.001]+[0.]*9,(n,1)),second=np.tile([.001]+[0.]*9,(n,1)),retention=np.ones((n,10)),
            sigma=np.tile(np.eye(10)*1e-7,(n,1,1)))
        self.de=dict(pairs=[[1,0]],beta=np.ones((n,1)),correlation=np.ones((n,1)),fresh=np.ones((n,1),bool))
        self.pred=np.full((n,1),.02)
    def run_replay(self,arm='ridge_6',overlay=False):
        return replay(self.d,self.de,self.pred,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)
    def test_one_leg_abort_and_later_close(self):
        self.d['par_volume'][1,1]=0
        r=self.run_replay();self.assertGreater(r['metrics']['aborted_pairs'],0)
        e=next(e for e in r['events'] if e['event']=='one_leg_abort')
        self.assertTrue(any(f['reduce_only'] and f['time']>e['time'] for f in r['fills']))
    def test_repair_is_reduce_only_then_completed(self):
        self.d['par_volume'][1,1]=25
        r=self.run_replay();self.assertGreater(r['metrics']['hedge_repairs'],0);self.assertGreater(r['metrics']['completed_pairs'],0)
        a=next(e['time'] for e in r['events'] if e['event']=='repair_requested');b=next(e['time'] for e in r['events'] if e['event']=='balanced_after_repair')
        self.assertTrue(all(f['reduce_only'] for f in r['fills'] if a<f['time']<=b))
    def test_baseline_retry_is_delayed_and_limited(self):
        self.d['par_volume'][1,0]=0
        r=self.run_replay('first_strategy');fills=r['fills'];self.assertTrue(fills)
        self.assertEqual(fills[0]['time'],int(self.d['times'][3]))
        self.assertTrue(all(f['source_available']<=f['decision_time']<f['time'] for f in fills))
    def test_funding_only_hourly_on_existing_inventory(self):
        self.d['funding'][self.d['times']%3600==0,0]=.0001
        r=self.run_replay('first_strategy')
        for e in r['funding_events']:
            self.assertEqual(e['time']%3600,0);self.assertAlmostEqual(e['signed_amount'],-e['qty'][0]*90*.0001)
    def test_no_signal_overlay_exact_and_final_accounting(self):
        self.pred[:]=0
        a=self.run_replay('second_strategy');b=self.run_replay('ridge_6',overlay=True)
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])
        m=b['metrics'];self.assertAlmostEqual(m['net_pnl'],m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'],places=7)
    def test_future_truncation_changes_neither_features_nor_predictions(self):
        with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        cfg=json.loads((HERE/'protocol.json').read_text());cfg['fit_dates']=cfg['fit_dates'][:1];cfg['horizons_hours']=[6]
        cut=ts('2026-10-09T12:00:00');mask=d['times']<=cut;short={k:v[mask] for k,v in d.items()}
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            full,fp,fr=fit_predict(d,cfg,self.p['symbols'],Path(td)/'full')
            pre,pp,pr=fit_predict(short,cfg,self.p['symbols'],Path(td)/'prefix')
        for k in ('x','beta','correlation','fresh'):np.testing.assert_allclose(full[k][mask],pre[k],rtol=0,atol=0,equal_nan=True)
        for k in pp:np.testing.assert_allclose(fp[k][mask],pp[k],rtol=0,atol=0,equal_nan=True)
        self.assertEqual(fr,pr);self.assertTrue(all(r.get('last_label_maturity',0)<r['fit_time'] for r in fr))
        self.assertTrue(any('coef' in r for r in fr))
        self.assertTrue(np.isfinite(pp['ridge_6']).any())

if __name__=='__main__':unittest.main()
