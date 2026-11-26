import copy,json,unittest
import numpy as np
from .prepare import ROOT,OUT,START,END
from .engine import simulate,execute
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from pas_strategy_audit.data import load


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from competition_live.profile import load_profile
        cls.p=load_profile()['trading'];cls.sc=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.)

    def test_minimum_order_applies_to_request_not_partial_fill(self):
        a=SharedAccount(50000.,self.p['symbols']);prices=np.full(10,100.)
        q=np.zeros(10);q[1]=1.
        intent=dict(time=1000,source=0,qty=q,source_prices=prices,expires=100000)
        row=execute(a,intent,dict(time=2000,coin='pas:ETH',buy_price=100.,sell_price=100.,qty=.1),
            prices,np.eye(10)*1e-6,self.p,self.sc)
        self.assertTrue(row['minimum_request_pass']);self.assertGreater(row['filled'],0.)
        self.assertLess(row['filled']*row['price'],10.)

    def test_untradeable_request_is_not_turned_into_an_order(self):
        a=SharedAccount(50000.,self.p['symbols']);prices=np.full(10,100.);q=np.zeros(10);q[1]=.01
        row=execute(a,dict(time=1000,source=0,qty=q,source_prices=prices,expires=10000),
            dict(time=2000,coin='pas:ETH',buy_price=100.,sell_price=100.,qty=100.),prices,np.eye(10)*1e-6,self.p,self.sc)
        self.assertEqual(row['filled'],0.)

    def test_no_prior_trade_volume_gate_and_actual_fill_after_decision(self):
        a=SharedAccount(50000.,self.p['symbols']);prices=np.full(10,100.);q=np.zeros(10);q[0]=1.
        row=execute(a,dict(time=1000,source=0,qty=q,source_prices=prices,expires=10000),
            dict(time=2000,coin='pas:BTC',buy_price=100.,sell_price=100.,qty=1.),prices,np.eye(10)*1e-6,self.p,self.sc)
        self.assertEqual(row['filled'],.01);self.assertLess(a.physical.fills[0]['decision_time'],a.physical.fills[0]['time'])

    def test_future_poison_does_not_change_earlier_decisions_or_pnl(self):
        cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
        signals=json.loads((OUT/'events.json').read_text());tape=json.loads((OUT/'trade_tape.json').read_text());fund=json.loads((OUT/'funding_tape.json').read_text())
        cut=(START+18*3600)*1000;bad=copy.deepcopy(d);bad['close'][bad['times']>cut]*=1.5
        bt=copy.deepcopy(tape)
        for t in bt:
            if t['time']>cut:
                for k in ['buy_price','sell_price','vwap']:t[k]*=1.5
                t['qty']*=4
        bf=copy.deepcopy(fund)
        for f in bf:
            if f['time']>cut:f['cash_per_unit']*=100
        a=simulate(cfg,d,signals,tape,fund,self.p,self.sc)
        b=simulate(cfg,bad,signals,bt,bf,self.p,self.sc)
        for field in ['decisions','snapshots','orders']:
            self.assertEqual([x for x in a[field] if x['time']<=cut],[x for x in b[field] if x['time']<=cut])

    def test_market_data_coverage_and_pair_exclusion_are_proven(self):
        coverage=json.loads((OUT/'tape_coverage.json').read_text())
        for row in coverage['assets']:
            self.assertAlmostEqual(row['fraction'],1.,places=9);self.assertEqual(row['missing_active_minutes'],0)
        pairs=json.loads((OUT/'pair_forecasts.json').read_text());f=[x for r in pairs for x in r['forecasts'] if x['model']=='decision_gate']
        self.assertEqual(len(f),96);self.assertTrue(all(x['economic_available'] and x['economic_action']==0 for x in f))

    def test_frozen_forecast_uses_only_available_reference_prefix(self):
        from pas_economic_allocation import model
        protocol=json.loads((ROOT/'reports/pas_adaptive_development_20261021/protocol.json').read_text())
        data=dict(np.load(ROOT/protocol['pointer']['inputs'],allow_pickle=False))
        fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text())
        x,u,carry=model.features(data);full=model.predict(x,u,fit)*fit['attenuation']
        for source in (START,START+86400,END-3600):
            stop=int(np.searchsorted(data['market_times'],source))+1
            prefix={k:v[:stop].copy() for k,v in data.items()}
            a,b,c=model.features(prefix)
            np.testing.assert_allclose(model.predict(a,b,fit)[-1]*fit['attenuation'],full[stop-1],rtol=0,atol=1e-12)
            np.testing.assert_allclose(c[-1],carry[stop-1],rtol=0,atol=1e-12)
        self.assertLess(fit['last_training_maturity'],START)
        self.assertLess(fit['last_calibration_maturity'],START)


if __name__=='__main__':unittest.main()
