import copy,json,unittest
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts
from .signals import intent_state,usable_signals,proposals
from .engine import replay

HERE=Path(__file__).resolve().parent


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'par_protocol.json').read_text());self.cfg['pairs']=[['ETH','BTC']];self.cfg['pair_max_gross']=1000
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        t=np.arange(ts('2026-10-08T12:00:00'),ts('2026-10-10T12:00:00')+300,300);n=len(t)
        self.d=dict(times=t,par_close=np.full((n,10),100.),reference_close=np.full((n,10),100.),
            par_volume=np.full((n,10),10000.),par_trades=np.full((n,10),50.),funding=np.zeros((n,10)),
            funding_observed=np.tile((t%3600==0)[:,None],(1,10)),known_hourly_funding=np.zeros((n,10)),
            first=np.tile([.001]+[0.]*9,(n,1)),second=np.tile([.001]+[0.]*9,(n,1)),retention=np.ones((n,10)),
            sigma=np.tile(np.eye(10)*1e-7,(n,1,1)))
        self.d['par_close'][t>=t[0]+6*3600,1]=100*np.exp(.051)
        z=((np.log(self.d['par_close'][:,1])-np.log(self.d['par_close'][:,0]))-.05)/.01
        directions=np.where(t<t[0]+6*3600,1,0)[:,None]
        terminal=np.tile(100*np.exp([.025,-.025]),(n,1,1))
        self.de=dict(pairs=[[1,0]],beta=np.ones((n,1)),mean=np.full((n,1),.05),sd=np.full((n,1),.01),
            z=z[:,None],directions=directions,terminal_prices=terminal,correlation=np.ones((n,1)),
            signal_times=np.full((n,1),t[0]),forecast_times=t//3600*3600,
            mean_log_prices=np.full((n,10),np.log(100.)))

    def account(self,arm='fixed_delayed_persistent_complete_168',overlay=False):
        return replay(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def choices(self,i=0):
        return proposals(i,self.d,self.de,None,168,self.cfg,self.p,self.cfg['scenarios']['base'],np.zeros(10),1e6,np.zeros(10),np.array([1]),'fixed_delayed_persistent_complete_168')

    def test_intent_waits_for_liquidity_without_inventing_source_volume(self):
        self.d['par_volume'][:24,:2]=0;r=self.account()
        entry=next(e for e in r['events'] if e['event']=='pair_intent')
        self.assertEqual(entry['time'],int(self.d['times'][24]));self.assertEqual(entry['forecast_source'],int(self.d['times'][0]))
        self.assertTrue(all(f['time']>entry['time'] for f in r['fills']));self.assertEqual(r['metrics']['completed_pairs'],1)

    def test_one_hour_intent_cannot_enter_after_two_hours(self):
        self.d['par_volume'][:24,:2]=0;self.de['directions'][12:]=0
        r=self.account();self.assertEqual(r['fills'],[]);self.assertFalse(r['metrics']['eligible'])

    def test_source_terminal_price_stays_fixed_and_current_price_reduces_edge(self):
        a=self.choices()[0];self.d['par_close'][0,1]=101;b=self.choices()[0]
        self.assertEqual(a['terminal_prices'],b['terminal_prices']);self.assertLess(b['expected_net_bps'],a['expected_net_bps'])
        self.d['par_close'][0,1]=106;self.assertEqual(self.choices(),[])

    def test_mean_reached_before_entry_cancels_source_permanently(self):
        used={};self.de['z'][0,0]=0
        a=usable_signals(0,self.d,self.de,self.cfg,'fixed_delayed_persistent_complete_168',{},used)
        self.assertEqual(a[0],0)
        b=usable_signals(1,self.d,self.de,self.cfg,'fixed_delayed_persistent_complete_168',{},used)
        self.assertEqual(b[0],0)

    def test_zero_fill_retries_stop_at_three(self):
        self.d['par_volume'][1::3,:2]=0
        r=self.account();intents=[e for e in r['events'] if e['event']=='pair_intent']
        self.assertEqual(len(intents),3);self.assertEqual(r['fills'],[])
        self.assertEqual([e['attempt'] for e in r['events'] if e['event']=='entry_unfilled'],[1,2,3])

    def test_single_leg_completion_pays_cost_without_increasing_first_leg(self):
        self.d['par_close'][1,0]=97
        r=self.account();entries=[f for f in r['fills'] if not f['reduce_only']]
        self.assertEqual([f['symbol'] for f in entries],['ETH','BTC'])
        self.assertEqual(entries[1]['time']-entries[0]['time'],300)
        self.assertTrue(all(f['fee']>0 for f in r['fills']))

    def test_source_expiration_does_not_cancel_required_exit(self):
        self.de['directions'][12:]=0
        self.d['par_volume'][72:96,:2]=0
        r=self.account();exits=[f for f in r['fills'] if f['reduce_only']]
        self.assertTrue(exits);self.assertTrue(all(f['time']>=self.d['times'][97] for f in exits))
        self.assertEqual(r['metrics']['residual_gross'],0)

    def test_no_signal_overlay_reproduces_incumbent_ledger(self):
        self.de['directions'][:]=0;a=self.account('second_strategy');b=self.account(overlay=True)
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])

    def test_funding_is_known_before_decision_and_charged_only_on_actual_position(self):
        self.d['funding'][self.d['times']%3600==0,1]=.0001
        r=self.account()
        for e in r['events']:
            if e['event']=='pair_intent':self.assertLess(e['funding_known_through'],e['time'])
        for f in r['funding_events']:
            self.assertEqual(f['time']%3600,0);self.assertAlmostEqual(f['signed_amount'],-f['qty'][1]*100*.0001)
        m=r['metrics'];self.assertAlmostEqual(m['net_pnl'],m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'],places=7)

    def test_future_truncation_preserves_source_intentions(self):
        cfg=json.loads((HERE/'par_protocol.json').read_text())
        with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in z.files}
        cut=ts('2026-10-09T12:00:00');mask=d['times']<=cut;rmask=ref['market_times']<=cut
        a=intent_state(d,ref,cfg,self.p['symbols'],6)
        b=intent_state({k:v[mask] for k,v in d.items()},{k:v[rmask] for k,v in ref.items()},cfg,self.p['symbols'],6)
        for k in a:
            if k!='pairs':np.testing.assert_allclose(a[k][mask],b[k],rtol=0,atol=0,equal_nan=True)
        valid=a['directions']!=0;clock=np.broadcast_to(d['times'][:,None],valid.shape)
        self.assertTrue(np.all(a['signal_times'][valid]<=clock[valid]));self.assertTrue(np.all(clock[valid]-a['signal_times'][valid]<6*3600))


if __name__=='__main__':unittest.main()
