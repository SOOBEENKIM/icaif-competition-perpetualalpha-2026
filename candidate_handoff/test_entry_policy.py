import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .entry_policy import BASE,NAME,configured_policy,summarize_candles,validate_history
from .portfolio import target_from_frame
from .producer import validate
from competition_live.profile import ROOT,load_profile
from competition_live.runtime import fingerprint

class EntryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.profile=load_profile();self.symbols=self.profile['trading']['symbols'];self.source=1792670400

    def raw(self):
        raw={}
        for symbol in self.symbols:
            raw[symbol]=[dict(s='pas:'+symbol,i='1m',t=(self.source-h*3600-60)*1000,
                T=(self.source-h*3600)*1000-1,c=str(100+h),v='1',n=1) for h in [24,3,0]]
        return raw

    def history(self,raw=None):
        h=summarize_candles(raw or self.raw(),self.symbols,self.source)
        h['received_at']=self.source+30
        return h

    def test_completed_candle_prefix_matches_no_future_and_all_symbols(self):
        raw=self.raw();expected=self.history(raw)
        for s in self.symbols:
            raw[s].append(dict(s='pas:'+s,i='1m',t=self.source*1000,
                T=(self.source+60)*1000-1,c='100000000',v='99999',n=99))
        self.assertEqual(expected,self.history(raw))
        validate_history(expected,self.symbols,self.source,self.source+40)
        self.assertEqual(expected['prices'],[[124.]*10,[103.]*10,[100.]*10])

    def test_sparse_no_trade_candles_cannot_make_stale_prices_fresh(self):
        raw=self.raw()
        for rows in raw.values():rows[-1].update(v='0',n=0)
        h=self.history(raw)
        self.assertEqual(h['age_seconds'][-1],[10800.]*10)
        validate_history(h,self.symbols,self.source,self.source+40)

    def test_future_and_inconsistent_history_is_rejected(self):
        for field,value in [('close_completed_at',self.source+60),('last_active_close_at',self.source+60)]:
            h=self.history();h[field][-1][0]=value
            with self.assertRaises(ValueError):validate_history(h,self.symbols,self.source,self.source+40)
        h=self.history();h['received_at']=self.source+50
        with self.assertRaises(ValueError):validate_history(h,self.symbols,self.source,self.source+40)

    def test_policy_contract_prevents_silent_reversion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.assertEqual(configured_policy(p),BASE)
            (p/'team_execution_contract.json').write_text(json.dumps({'required_execution_policy':NAME}))
            with self.assertRaises(ValueError):configured_policy(p)
            (p/'execution_policy.json').write_text(json.dumps({'policy':NAME}))
            self.assertEqual(configured_policy(p),NAME)

    def test_packet_policy_and_history_are_bound_to_digest_and_expected_revision(self):
        from universe_repair.test_integration import IntegrationTests
        packet=IntegrationTests().packet();body=packet['body'];source=body['frame']['source_time']
        self.source=source
        body['execution_policy']=NAME;body['frame']['execution_policy']=NAME
        body['frame']['venue_history']=self.history()
        body['published_at']=source+40;packet['sha256']=fingerprint(body)
        validate(packet,self.profile,source+130,expected_policy=NAME)
        with self.assertRaises(ValueError):validate(packet,self.profile,source+130,expected_policy=BASE)
        body['frame']['venue_history']['prices'][0][0]+=1
        with self.assertRaises(ValueError):validate(packet,self.profile,source+130,expected_policy=NAME)

    def test_live_adapter_equals_selected_controller_on_saved_pas_decisions(self):
        from pas_performance_research.selected import controller
        from pas_strategy_audit.data import load
        out=ROOT/'reports/pas_performance_research_20261022/extension'
        _,data=load(out);f=dict(np.load(out/'candidate_inputs.npz'))
        events={r['source_time']:r for r in json.loads((out/'events.json').read_text())}
        checks=0
        for i,source in enumerate(f['times']):
            if source%21600 or source-24*3600<data['times'][0]/1000:continue
            points=np.array([source-24*3600,source-3*3600,source])
            ix=np.searchsorted(data['times']/1000,points,side='right')-1
            frame=dict(source_time=int(source),execution_policy=NAME,
                anchor_weights=f['anchor'][i].tolist(),protected=f['protected'][i].tolist(),
                price_forecast_168h=f['forecast'][i].tolist(),reference_funding_forecast_168h=f['carry'][i].tolist(),
                sigma=events[source]['sigma'],volume=events[source]['volume'],trades=events[source]['trades'],risk_aversion=5.,
                venue_history=dict(price_times=points.tolist(),prices=data['close'][ix].tolist(),age_seconds=(data['age_ms'][ix]/1000).tolist()))
            prices=np.nan_to_num(data['close'][ix[-1]],nan=1.);qty=np.zeros(10)
            full=controller([source],f['forecast'][i:i+1],f['carry'][i:i+1],f['anchor'][i:i+1],f['protected'][i:i+1],
                price_times=data['times']/1000,prices=data['close'],age=data['age_ms']/1000)
            expected=full.propose(source,prices,qty,50000.,f['anchor'][i],np.asarray(frame['sigma']),
                np.asarray(frame['volume']),np.asarray(frame['trades']),self.profile['trading'],buffer=.2)
            got,trace=target_from_frame(frame,prices,qty,50000.,self.profile['trading'])
            np.testing.assert_array_equal(got['target_qty'],expected['target_qty'])
            self.assertEqual(trace['entry_filter_pass'],full.trace[-1]['entry_filter_pass']);checks+=1
        self.assertGreaterEqual(checks,6)

    def test_new_policy_keeps_held_units_and_risk_reductions_both_sides(self):
        from pas_economic_allocation.lifecycle import LifecycleController
        frame=dict(source_time=self.source,execution_policy=NAME,venue_history=self.history(),
            price_forecast_168h=[0.]*10,reference_funding_forecast_168h=[0.]*10,anchor_weights=[0.]*10,
            protected=[False]*10,risk_aversion=5.,sigma=(np.eye(10)*1e-8).tolist(),volume=[1.]*10,trades=[1.]*10)
        def parent(self,*args,**kwargs):self.trace.append({});return {'target_qty':[5.,-5.,1.,-1.,0.,0.,0.,0.,0.,0.]}
        with patch.object(LifecycleController,'propose',parent):
            got,_=target_from_frame(frame,np.full(10,100.),np.array([3.,-3.,2.,-2.,0.,0.,0.,0.,0.,0.]),50000.,self.profile['trading'])
        # Falling price permits a long addition but retains an existing short;
        # both reductions survive regardless of entry timing.
        np.testing.assert_allclose(got['target_qty'][:4],[5.,-3.,1.,-1.],atol=1e-5)

if __name__=='__main__':unittest.main()
