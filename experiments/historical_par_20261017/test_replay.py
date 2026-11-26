import copy
import json
import unittest
import numpy as np
from strategy.agent import TAIL
from .data import HERE,OUT,ts
from .engine import run
from .model import build


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'protocol.json').read_text())
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        self.cfg.update(pairs=[['ETH','BTC']],holding_hours=6,terminal_reduce_hours=2,pair_max_gross=1000)
        t=np.arange(ts('2026-10-01T12:00:00'),ts('2026-10-03T12:00:00')+3600,3600);n=len(t)
        self.d=dict(times=t,close=np.full((n,10),100.),oracle_proxy=np.full((n,10),90.),
            volume=np.full((n,10),10000.),trades=np.full((n,10),50.),funding=np.zeros((n,10)),
            first=np.tile([.001]+[0.]*9,(n,1)),second=np.tile([.001]+[0.]*9,(n,1)),retention=np.ones((n,10)),
            sigma=np.tile(np.eye(10)*1e-7,(n,1,1)))
        self.s=dict(beta=np.ones((n,1)),correlation=np.ones((n,1)),ridge=np.zeros((n,1)),
            funding_forecast=np.tile([-.02,.02]+[0.]*8,(n,1)))

    def replay(self,arm='pairs_carry',scenario='base'):
        return run(self.d,self.s,self.cfg,self.p,arm,scenario,int(self.d['times'][0]),int(self.d['times'][-1]))

    def test_next_hour_and_zero_volume(self):
        self.d['volume'][1,0]=0
        r=self.replay('first_strategy')
        self.assertFalse(any(f['time']==self.d['times'][1] for f in r['fills']))
        self.assertTrue(all(f['time']>f['decision_time'] for f in r['fills']))
        self.assertTrue(r['fills'])

    def test_actual_funding_uses_oracle_proxy_and_existing_inventory(self):
        self.d['funding'][:,0]=.0001
        r=self.replay('first_strategy')
        self.assertGreater(len(r['funding_events']),0)
        for e in r['funding_events']:
            self.assertAlmostEqual(e['signed_amount'],-e['qty'][0]*90*.0001)
        self.assertGreater(r['funding_events'][0]['time'],r['fills'][0]['time'])

    def test_partial_pair_unwinds_at_future_price(self):
        self.d['volume'][1,1]=0
        r=self.replay()
        self.assertGreater(r['metrics']['aborted_pairs'],0)
        self.assertGreater(r['metrics']['one_leg_hours'],0)
        first=[e for e in r['events'] if e['event']=='entry_unbalanced_abort'][0]
        closes=[f for f in r['fills'] if f['reduce_only'] and f['time']>first['time']]
        self.assertTrue(closes)
        self.assertGreater(r['metrics']['fees'],0)

    def test_terminal_liquidity_is_not_invented(self):
        self.d['volume'][-3:]=0
        r=self.replay('first_strategy')
        self.assertGreater(r['metrics']['residual_gross'],0)
        self.assertGreater(r['metrics']['liquidation_cost_estimate'],0)

    def test_no_edge_ensemble_matches_second(self):
        self.s['funding_forecast'][:]=0;self.d['oracle_proxy'][:]=100
        a=self.replay('second_strategy');b=self.replay('second_plus_basis_carry')
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])

    def test_equity_attribution_and_daily_compounding(self):
        self.d['close'][:,0]=np.linspace(100,105,len(self.d['times']))
        r=self.replay('first_strategy');m=r['metrics']
        self.assertAlmostEqual(m['net_pnl'],m['price_move_pnl']-m['fees']-m['execution_cost']+m['funding'],places=7)
        self.assertAlmostEqual(np.prod([1+x['return_'] for x in r['daily']])-1,m['roi'],places=12)
        self.assertEqual(m['completed_days'],2)

    def test_forecast_prefix_invariance_and_label_purge(self):
        if not (OUT/'inputs.npz').exists():self.skipTest('prepared history required')
        with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        cfg=json.loads((HERE/'protocol.json').read_text())
        full,models=build(d,cfg,self.p['symbols']);cut=ts('2026-10-08T12:00:00')
        prefix={k:v.copy() for k,v in d.items()}
        for key in ('times','close','volume','trades','funding','oracle_proxy','first','second','retention','sigma'):
            prefix[key]=prefix[key][d['times']<=cut]
        prefix['ref_times']=d['ref_times'][d['ref_times']<=cut]
        prefix['ref_close']=d['ref_close'][d['ref_times']<=cut]
        prefix['funding_times']=d['funding_times'][d['funding_times']<=cut]
        prefix['funding_history']=d['funding_history'][d['funding_times']<=cut]
        short,_=build(prefix,cfg,self.p['symbols'])
        for key in full:np.testing.assert_allclose(full[key][:len(short[key])],short[key],rtol=0,atol=0,equal_nan=True)
        self.assertTrue(all(m['last_label_maturity']<m['fit_time'] for m in models))


if __name__=='__main__':unittest.main()
