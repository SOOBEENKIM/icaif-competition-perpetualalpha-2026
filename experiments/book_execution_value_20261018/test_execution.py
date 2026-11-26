import copy
import unittest
import numpy as np
from .source import make_plan,features
from .episodes import episode
from experiments.basis_carry_20261017.probe import asof


class BookExecutionTests(unittest.TestCase):
    def setUp(self):
        self.now=1000;self.p=dict(initial=1e6,symbols=['A','B'],execution=dict(decimals=[2,2],minimum_order_value=10.),
            risk=dict(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,asset_notional=150000.,daily_volatility=.005))
        self.spec=dict(gross_ceiling=120000.,visible_depth_band_bps=10.,maximum_reconciliation_seconds=60)
        self.sc=dict(depth_fraction=1.,fee_bps=9.,latency_ms=1000,exit_impact_bps=10.)
        self.cfg=dict(book_max_age_ms=15000,context_max_age_ms=3000,future_exit_observation_grace_ms=5000,
            completion_ack_delay_ms=1000,pair_balance_tolerance=.05)
        self.marks=np.array([100.,50.]);self.sigma=np.eye(2)*1e-6;self.streams={}
        for symbol,mid in zip(['A','B'],self.marks):
            entries=[];contexts=[]
            for at in [1000000,1001000,1002000,1003000]:
                entries.append((at,0,dict(time=at,levels=[[dict(px=str(mid*.999),sz='1000')],[dict(px=str(mid*1.001),sz='1000')]])))
                contexts.append((at,0,dict(oraclePx=str(mid),markPx=str(mid),funding='0')))
            self.streams[symbol]=dict(book=entries,ctx=contexts,book_times=[e[0] for e in entries],ctx_times=[e[0] for e in contexts],resets=[])
        self.quotes={s:asof(v,1000000,self.cfg) for s,v in self.streams.items()}

    def plan(self,fraction=1.,buffer=10.):
        return make_plan(['A','B'],[0,1],1.,self.quotes,self.now,self.marks,self.sigma,self.p,self.sc,self.spec,1,fraction,buffer)

    def test_size_is_based_on_source_depth_and_lots(self):
        full=self.plan();quarter=self.plan(.25)
        np.testing.assert_allclose(np.array(full['target'])*.25,quarter['target'])
        self.assertAlmostEqual(full['intended_gross']/4,quarter['intended_gross'])

    def test_limit_is_source_vwap_plus_fixed_buffer(self):
        plan=self.plan(buffer=0.)
        self.assertEqual(plan['entry_limits'],{'0':plan['entry_preview'][0],'1':plan['entry_preview'][1]})

    def test_future_book_cannot_change_source_features(self):
        plan=self.plan();x,names=features(plan,self.quotes,self.now,self.marks,self.sigma,self.p,self.sc,0)
        changed=copy.deepcopy(self.streams);changed['A']['book'][1][2]['levels'][1][0]['px']='10000'
        source={s:asof(v,1000000,self.cfg) for s,v in changed.items()}
        y,_=features(plan,source,self.now,self.marks,self.sigma,self.p,self.sc,0)
        np.testing.assert_array_equal(x,y);self.assertEqual(len(x),len(names))

    def test_balanced_entry_is_not_a_realized_profitable_trade(self):
        r=episode(self.plan(.25),self.streams,self.quotes,self.now,1100000,self.marks,self.sigma,self.p,self.sc,self.cfg,self.spec)
        self.assertEqual(r['outcome'],2);self.assertIsNone(r['realized_cash_pnl'])
        self.assertFalse(r['complete_horizon_exit_observed']);self.assertAlmostEqual(r['entry_cost_return'],0.,places=10)

    def test_unknown_arrival_is_not_zero_cash_or_no_fill(self):
        streams=copy.deepcopy(self.streams)
        for key in ['book','ctx','book_times','ctx_times']:streams['B'][key]=streams['B'][key][:1]
        r=episode(self.plan(.25),streams,self.quotes,self.now,1100000,self.marks,self.sigma,self.p,self.sc,self.cfg,self.spec)
        self.assertEqual(r['outcome'],3);self.assertIsNone(r['entry_cost_return']);self.assertIsNone(r['realized_cash_pnl'])
        self.assertEqual(r['available_at'],1060)

    def test_observed_zero_fill_is_zero_cash(self):
        streams=copy.deepcopy(self.streams)
        for s in streams:
            for at,epoch,book in streams[s]['book'][1:]:
                book['levels'][1][0]['px']=str(float(book['levels'][1][0]['px'])*1.02)
                book['levels'][0][0]['px']=str(float(book['levels'][0][0]['px'])*.98)
        r=episode(self.plan(.25,0.),streams,self.quotes,self.now,1100000,self.marks,self.sigma,self.p,self.sc,self.cfg,self.spec)
        self.assertEqual(r['outcome'],0);self.assertEqual(r['realized_cash_pnl'],0.)


if __name__=='__main__':unittest.main()
