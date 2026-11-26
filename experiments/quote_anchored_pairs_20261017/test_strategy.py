import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import experiments.par_direct_forecast_20261017.test_direct as fixtures
from experiments.par_direct_forecast_20261017.data import ts, OUT as INPUT
from .model import labels, fit_predict
from .engine import replay
from .pricing import reservation_limits, value_at_entry
from .quote_probe import decide
from .probe_lifecycle import audit_probe
from experiments.pair_execution_calibration_20261017.test_lifecycle import stream

HERE=Path(__file__).resolve().parent


class StrategyTests(fixtures.DirectTests):
    def setUp(self):
        super().setUp()
        cfg=json.loads((HERE/'protocol.json').read_text())
        cfg['pairs']=self.cfg['pairs'];cfg['pair_max_gross']=1000;self.cfg=cfg

    def run_replay(self,arm='ridge_6',overlay=False):
        return replay(self.d,self.de,self.pred,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def test_forecast_label_anchors_at_source_not_unknown_entry(self):
        self.d['par_close'][1,1]=150
        self.d['par_close'][12,1]=110
        y,m=labels(self.d,self.de,1)
        self.assertAlmostEqual(y[0,0],500.)
        self.assertEqual(m[0],self.d['times'][0]+3600)

    def test_both_reservation_limits_preserve_the_cash_buffer(self):
        q=np.array([2.,-3.]);anchors=np.array([100.,50.]);previews=np.array([100.1,49.95])
        for fee,impact in ((9,10),(12,20)):
            r=reservation_limits(q,anchors,previews,.02,-.1,fee,impact,5,100)
            self.assertIsNotNone(r)
            at=value_at_entry(q,anchors,r['limits'],.02,-.1,fee,impact)
            self.assertGreaterEqual(at['expected_net_cash']+1e-9,r['buffer_cash'])

    def test_price_movement_debits_remaining_edge_not_a_new_forecast(self):
        a=value_at_entry([1.,-1.],[100.,100.],[100.,100.],.01,0.,9,10)
        b=value_at_entry([1.,-1.],[100.,100.],[101.,99.],.01,0.,9,10)
        self.assertAlmostEqual(a['expected_net_cash']-b['expected_net_cash'],2.)
        self.assertLess(b['expected_net_cash'],0)

    def test_unfavorable_leg_does_not_retroactively_cancel_other_leg(self):
        self.d['par_close'][1,1]=110
        r=self.run_replay();first=self.d['times'][1]
        self.assertFalse(any(f['asset']==1 and f['time']==first for f in r['fills']))
        self.assertTrue(any(f['asset']==0 and f['time']==first for f in r['fills']))
        self.assertTrue(any(e['event']=='one_leg_abort' for e in r['events']))
        self.assertTrue(any(f['reduce_only'] and f['time']>first for f in r['fills']))

    def test_exit_matches_source_forecast_maturity(self):
        r=self.run_replay()
        close=next(f['time'] for f in r['fills'] if f['reduce_only'])
        self.assertEqual(close,self.d['times'][0]+6*3600)

    def test_limits_fixed_before_future_price_or_volume_is_known(self):
        a=self.run_replay()
        self.d['par_close'][1,1]=110;self.d['par_volume'][1,0]=0
        b=self.run_replay()
        da=next(x for x in a['decisions'] if x.get('pair_entry'))
        db=next(x for x in b['decisions'] if x.get('pair_entry'))
        self.assertEqual(da['entry_limits'],db['entry_limits'])
        self.assertEqual(da['expected_net_cash'],db['expected_net_cash'])

    def quote_case(self):
        at=int(self.d['times'][0]*1000)
        streams={s:stream(times=(at,at+1000,at+2000,at+3000)) for s in ('ETH','BTC')}
        for st in streams.values():
            for _,_,ctx in st['ctx']:ctx['funding']='0'
        view=dict(symbol='ETH',decision=at,side=1,requested_qty=1.,requested_price=100.)
        life=json.loads((HERE.parent/'pair_execution_calibration_20261017/lifecycle_protocol.json').read_text())
        return at,streams,view,life

    def test_quote_selection_reprices_entry_before_any_fill_is_known(self):
        at,streams,view,life=self.quote_case();sc=life['scenarios']['base']
        fn=lambda:decide(view,['ETH','BTC'],0,'ridge_6',streams,{'ETH':4,'BTC':4},self.d,self.de,self.pred,self.p['symbols'],self.cfg,life,sc)
        first=fn();self.assertEqual(first['status'],'selected')
        self.assertEqual(first['forecast_maturity_ms'],at+6*3600000)
        view['requested_price']=105.
        self.assertEqual(fn()['status'],'expected_edge_below_cost')

    def test_hedge_limit_does_not_follow_price_after_maker_fill(self):
        at,streams,view,life=self.quote_case();sc=life['scenarios']['base'];dec={'ETH':4,'BTC':4}
        chosen=decide(view,['ETH','BTC'],0,'ridge_6',streams,dec,self.d,self.de,self.pred,self.p['symbols'],self.cfg,life,sc)
        self.assertEqual(chosen['status'],'selected')
        streams['BTC']['book'][2]=(at+2000,0,dict(time=at+2000,levels=[[dict(px='99.5',sz='10')],[dict(px='99.7',sz='10')]]))
        probe=dict(chosen,lead_fill=dict(time=at+1000,side=1,qty=1.,price=100.,coin='par:ETH'),
            forecast_aligned={'ridge':True},tape='synthetic',phase='test')
        r=audit_probe(probe,streams,dec,self.d,self.p['symbols'],life,sc)
        self.assertEqual(r['hedge_status'],'unfilled_within_limit')
        self.assertLess(r['net_pnl'],0)

    def test_future_truncation_changes_neither_features_nor_predictions(self):
        with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        cfg=json.loads((HERE/'protocol.json').read_text());cfg['fit_dates']=cfg['fit_dates'][:1];cfg['horizons_hours']=[6]
        mask=d['times']<=ts('2026-10-09T12:00:00');short={k:v[mask] for k,v in d.items()}
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            de,pred,records=fit_predict(d,cfg,self.p['symbols'],Path(td)/'full')
            small,sp,sr=fit_predict(short,cfg,self.p['symbols'],Path(td)/'prefix')
        for k in ('x','beta','correlation','fresh'):np.testing.assert_allclose(de[k][mask],small[k],rtol=0,atol=0,equal_nan=True)
        for k in pred:np.testing.assert_allclose(pred[k][mask],sp[k],rtol=0,atol=0,equal_nan=True)
        self.assertEqual(records,sr)
        self.assertTrue(any(r.get('rows',0)>1000 for r in records))
        self.assertTrue(all(r.get('last_label_maturity',0)<r['fit_time'] for r in records))
        self.assertIn('trade_age_minutes_a',de['names'])


if __name__=='__main__':unittest.main()
