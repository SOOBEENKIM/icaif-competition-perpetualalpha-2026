import copy
import json
import unittest
import numpy as np
from .lifecycle import audit_probe, HERE


def stream(times=(1000,2000,3000), size=10., bid=99.9, ask=100.1):
    books=[(t,0,dict(time=t,levels=[[dict(px=str(bid),sz=str(size))],[dict(px=str(ask),sz=str(size))]])) for t in times]
    ctx=[(t,0,dict(oraclePx='100')) for t in times]
    return dict(book=books,book_times=list(times),ctx=ctx,ctx_times=list(times),resets=[])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'lifecycle_protocol.json').read_text());self.sc=self.cfg['scenarios']['base']
        times=np.arange(0,30*3600+300,300);shape=(len(times),2)
        self.d=dict(times=times,par_close=np.full(shape,100.),par_volume=np.full(shape,100.),
                    reference_close=np.full(shape,100.),funding=np.zeros(shape),funding_observed=np.ones(shape,bool))
        self.streams={'A':stream(),'B':stream()};self.decimals={'A':2,'B':2}
        self.probe=dict(lead='A',hedge='B',pair=['A','B'],beta=1.,hours=1,tape='synthetic',phase='test',
            quote_decision=0,forecast_aligned={'ridge':True,'tree':False,'blend':False},status='old_probe',
            lead_fill=dict(time=1000,side=1,qty=1.,price=100.,coin='par:A'))

    def run_probe(self):
        return audit_probe(self.probe,self.streams,self.decimals,self.d,['A','B'],self.cfg,self.sc)

    def test_both_legs_close_with_costs_and_delayed_candle_exit(self):
        r=self.run_probe()
        self.assertEqual(r['hedge_status'],'full');self.assertEqual(r['residual_gross'],0)
        self.assertLess(r['net_pnl'],0);self.assertGreater(r['fees'],.29)
        self.assertGreater(r['exit_delay_minutes'],0);self.assertLess(r['accounting_error'],1e-8)
        self.assertTrue(any(f['source']=='modeled_candle_exit' for p in r['positions'] for f in p['fills']))

    def test_zero_volume_keeps_inventory_and_fees_not_zero_profit(self):
        self.d['par_volume'][:]=0
        r=self.run_probe()
        self.assertEqual(r['status'],'unresolved_inventory');self.assertEqual(r['residual_gross'],200)
        self.assertGreater(r['entry_fees'],0);self.assertLess(r['net_after_reserve'],r['net_pnl'])

    def test_below_minimum_hedge_preserves_lead_then_unwinds(self):
        self.probe['lead_fill']['qty']=.05
        r=self.run_probe()
        self.assertEqual(r['hedge_status'],'hedge_below_minimum')
        self.assertEqual(r['residual_gross'],0);self.assertGreater(r['entry_fees'],0)
        self.assertLess(r['net_pnl'],0);self.assertEqual(len(r['positions'][1]['fills']),0)

    def test_partial_hedge_is_paid_and_both_actual_quantities_unwound(self):
        self.streams['B']=stream(size=.4)
        r=self.run_probe()
        self.assertEqual(r['hedge_status'],'partial');self.assertAlmostEqual(r['filled_hedge'],.4)
        self.assertFalse(r['balanced']);self.assertEqual(r['residual_gross'],0)
        self.assertAlmostEqual(r['positions'][1]['fills'][0]['signed_qty'],-.4)
        self.assertLess(r['net_pnl'],0)

    def test_fixed_limit_rejects_price_jump(self):
        self.streams['B']['book'][1]=(2000,0,dict(time=2000,levels=[[dict(px='95',sz='10')],[dict(px='95.2',sz='10')]]))
        r=self.run_probe()
        self.assertEqual(r['hedge_status'],'unfilled_within_limit');self.assertEqual(r['filled_hedge'],0)
        self.assertGreater(r['positions'][0]['fees'],0)

    def test_missing_hedge_is_flagged_and_not_removed(self):
        self.streams['B']=stream(times=(1000,))
        r=self.run_probe()
        self.assertTrue(r['unobserved_hedge_outcome']);self.assertEqual(r['hedge_status'],'missing_hedge_arrival')
        self.assertGreaterEqual(r['exit_decision'],7000);self.assertGreater(r['fees'],0)

    def test_funding_charged_on_prior_quantity_with_correct_sign(self):
        self.d['funding'][:,0]=.001;self.d['funding'][:,1]=-.002
        r=self.run_probe()
        self.assertAlmostEqual(r['funding'],-.3)
        self.assertEqual({f['time'] for f in r['funding_events']},{3600000})
        self.assertLess(r['accounting_error'],1e-8)

    def test_exit_quantity_cannot_exceed_bar_capacity(self):
        self.d['par_volume'][:]=1.
        r=self.run_probe()
        for p in r['positions']:
            for f in p['fills']:
                if f['source']=='modeled_candle_exit':self.assertLessEqual(abs(f['signed_qty']),.1+1e-10)
        self.assertEqual(r['residual_gross'],0);self.assertGreater(r['exit_delay_minutes'],40)

    def test_final_lot_is_not_stranded_by_float_subtraction(self):
        self.decimals={'A':4,'B':4};self.probe['lead_fill']['qty']=.0015
        self.streams={'A':stream(times=(1000,)),'B':stream(times=(1000,))}
        self.d['par_volume'][1,0]=.014
        r=self.run_probe();pos=r['positions'][0]
        self.assertEqual(r['residual_gross'],0)
        self.assertEqual([abs(f['signed_qty']) for f in pos['fills'][1:]],[.0014,.0001])
        self.assertEqual(pos['units'],0)


if __name__=='__main__':unittest.main()
