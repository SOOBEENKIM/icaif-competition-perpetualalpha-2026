import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.test_direct import DirectTests
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts
from .signals import design,crossing,held_state,usable_signals,projected_return
from .engine import replay

HERE=Path(__file__).resolve().parent


class AdaptiveTests(DirectTests):
    def setUp(self):
        super().setUp()
        self.cfg=json.loads((HERE/'protocol.json').read_text());self.cfg['pairs']=[['ETH','BTC']];self.cfg['pair_max_gross']=1000
        n=len(self.d['times']);t=self.d['times'];meanlog=np.full((n,10),np.log(100.))
        meanlog[t<t[0]+6*3600,1]+=.05
        source=t//3600*3600;signal=np.zeros((n,1),int);signal[t<t[0]+3600]=1
        self.de.update(mean=np.full((n,1),.05),sd=np.full((n,1),.01),z=np.full((n,1),-5.),
            mean_log_prices=meanlog,forecast_times=source,standard=signal,delayed=signal.copy())

    def run_replay(self,arm='adaptive_standard_168',overlay=False):
        return replay(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def pair(self):
        return dict(ids=[1,0],beta=1.,entry_mean=.05,entry_sd=.01,adaptive_mean=True,
                    direction=1,forecast_source=int(self.d['times'][0]),first_fill=int(self.d['times'][0]))

    def test_mean_can_adapt_but_hedge_and_risk_stay_frozen(self):
        p=self.pair();i=72
        adaptive=held_state(p,i,self.d,self.de,self.cfg)
        fixed=held_state({**p,'adaptive_mean':False},i,self.d,self.de,self.cfg)
        self.assertEqual(adaptive['reason'],'mean_cross');self.assertIsNone(fixed['reason'])
        dd=copy.deepcopy(self.de);dd['beta'][i]=9;dd['sd'][i]=100
        self.assertEqual(adaptive,held_state(p,i,self.d,dd,self.cfg))

    def test_stop_lock_releases_only_after_mean_cross(self):
        locks={0:self.pair()};self.assertEqual(usable_signals(0,self.d,self.de,self.cfg,'adaptive_standard_168',locks,{})[0],0)
        self.assertIn(0,locks)
        usable_signals(72,self.d,self.de,self.cfg,'adaptive_standard_168',locks,{})
        self.assertNotIn(0,locks)

    def test_used_signal_does_not_reenter(self):
        self.d['par_volume'][1,1]=0
        r=self.run_replay()
        self.assertEqual(sum(e['event']=='pair_intent' for e in r['events']),1)

    def test_exit_request_persists_until_delayed_liquidity(self):
        t=self.d['times'];mask=(t>=t[0]+6*3600)&(t<t[0]+8*3600)
        self.d['par_volume'][mask,:2]=0
        r=self.run_replay();exits=[f for f in r['fills'] if f['reduce_only']]
        self.assertTrue(exits);self.assertGreaterEqual(exits[0]['time'],t[0]+8*3600+300)
        self.assertEqual(r['metrics']['completed_pairs'],1);self.assertEqual(r['metrics']['residual_gross'],0)

    def test_crossing_is_directional_and_convergence_projection_consistent(self):
        self.assertEqual(crossing(-2,-4,3),1);self.assertEqual(crossing(2,4,3),-1)
        self.assertEqual(crossing(-4,-2,3,True),1);self.assertEqual(crossing(4,2,3,True),-1)
        self.assertEqual(crossing(-4,1,3,True),0)
        beta=2.;move=.06/(1+beta)
        self.assertAlmostEqual(projected_return(.06,beta),(np.exp(move)-1-beta*(np.exp(-move)-1))/(1+beta))

    def test_time_stop_tightens_to_zero(self):
        p=self.pair();i=0
        p['first_fill']=int(self.d['times'][i])-126*3600
        a=held_state(p,i,self.d,self.de,self.cfg);self.assertEqual(a['stop_bound'],3.)
        p['first_fill']=int(self.d['times'][i])-168*3600
        b=held_state(p,i,self.d,self.de,self.cfg);self.assertEqual(b['reason'],'time_stop');self.assertEqual(b['stop_bound'],0.)

    def test_no_signal_overlay_exact_and_final_accounting(self):
        self.de['standard'][:]=0
        a=self.run_replay('second_strategy');b=self.run_replay(overlay=True)
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])
        m=b['metrics'];self.assertAlmostEqual(m['net_pnl'],m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'],places=7)

    def test_future_truncation_changes_neither_features_nor_predictions(self):
        cfg=json.loads((HERE/'protocol.json').read_text())
        with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in z.files}
        at=ts('2026-10-09T12:00:00');mask=d['times']<=at;rmask=ref['market_times']<=at
        full=design(d,ref,cfg,self.p['symbols'])
        short=design({k:v[mask] for k,v in d.items()},{k:v[rmask] for k,v in ref.items()},cfg,self.p['symbols'])
        for k,v in full.items():
            if k!='pairs':np.testing.assert_allclose(v[mask],short[k],rtol=0,atol=0,equal_nan=True)
        self.assertTrue(np.all(full['forecast_times']<=d['times']))


if __name__=='__main__':unittest.main()
