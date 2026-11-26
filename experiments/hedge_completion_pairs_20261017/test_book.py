import copy,unittest
from experiments.pair_execution_calibration_20261017.test_lifecycle import LifecycleTests,stream
from .book_lifecycle import audit_probe
from .book_completion import plan


class BookCompletionTests(LifecycleTests):
    def setUp(self):
        super().setUp()
        self.cfg.update(completion_enabled=True,completion_ack_delay_ms=1000,completion_buffer_bps=5)
        self.probe.update(hedge_ratio_units=1.,hedge_limit=99.8,forecast_maturity_ms=3600000,
            anchor_prices=[100.,100.],forecast=.02,direction=1)
        for st in self.streams.values():
            for _,_,ctx in st['ctx']:ctx['funding']='0'

    def run_probe(self):return audit_probe(self.probe,self.streams,self.decimals,self.d,['A','B'],self.cfg,self.sc)

    def test_fixed_limit_rejects_price_jump(self):
        self.cfg['completion_enabled']=False
        super().test_fixed_limit_rejects_price_jump()

    def completion_streams(self,last=5000):
        self.streams={s:stream(times=tuple(range(1000,last+1,1000))) for s in ['A','B']}
        for st in self.streams.values():
            for _,_,ctx in st['ctx']:ctx['funding']='0'
        for i,(t,ep,b) in enumerate(self.streams['B']['book']):
            if t>=2000:b['levels'][0][0]['px']='99.5'

    def test_observed_unfilled_hedge_gets_one_budgeted_retry(self):
        self.completion_streams();r=self.run_probe()
        self.assertEqual(r['hedge_status'],'completed_full');self.assertTrue(r['balanced'])
        c=r['completion'];self.assertEqual(c['decision_time'],3000)
        self.assertGreaterEqual(c['minimum_expected_cash_at_limits'],c['buffer_cash']-1e-8)
        self.assertEqual(r['positions'][1]['fills'][0]['time'],4000)
        self.assertLess(r['accounting_error'],1e-8)

    def test_missing_first_response_does_not_create_second_order(self):
        self.streams['B']=stream(times=(1000,))
        r=self.run_probe();self.assertIsNone(r['completion']);self.assertTrue(r['unobserved_hedge_outcome'])

    def test_missing_second_response_stays_unknown(self):
        self.completion_streams(last=3000);r=self.run_probe()
        self.assertEqual(r['hedge_status'],'missing_completion_arrival');self.assertTrue(r['unobserved_hedge_outcome'])

    def test_second_order_cannot_use_its_future_price_to_set_limit(self):
        self.completion_streams()
        self.streams['B']['book'][3][2]['levels'][0][0]['px']='90'
        self.streams['B']['book'][4][2]['levels'][0][0]['px']='90'
        r=self.run_probe();self.assertEqual(r['completion']['status'],'unfilled')
        self.assertGreater(r['completion']['limit'],90);self.assertFalse(r['balanced'])

    def test_settled_lead_funding_enters_cash_budget(self):
        self.completion_streams()
        fill={**self.probe['lead_fill'],'time':3599000}
        streams={s:stream(times=(3600000,3601000)) for s in ['A','B']}
        for st in streams.values():
            for _,_,ctx in st['ctx']:ctx['funding']='0'
        probe={**self.probe,'forecast_maturity_ms':7200000}
        self.d['funding'][12,0]=.001
        budget,_=plan(probe,fill,3601000,1.,streams,self.decimals,self.cfg,self.sc,self.d,['A','B'])
        self.assertAlmostEqual(budget['accrued_lead_funding'],-.1)


if __name__=='__main__':unittest.main()
