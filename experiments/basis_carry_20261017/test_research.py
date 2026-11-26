import json,unittest
from pathlib import Path
import numpy as np
from .probe import sweep,asof,expected_net,funding_events_between,decompose,HOUR
from .funding import fit_ar1,predict
from .score import score,transfer_adjusted_return

CFG=json.loads(Path(__file__).with_name('protocol.json').read_text())

class ResearchTests(unittest.TestCase):
    def test_limit_rejects_expensive_levels_and_preserves_partial_fill(self):
        levels=[dict(px='100',sz='1'),dict(px='101',sz='2')]
        q,v=sweep(levels,2,1,1,2,100.1)
        self.assertEqual(q,1);self.assertEqual(v,100)
        q,v=sweep(levels,2,.5,1,2,100.1)
        self.assertEqual(q,.5);self.assertEqual(v,100)
    def test_depth_haircut_respects_lot_grid(self):
        q,_=sweep([dict(px='100',sz='.0003')],.0003,.5,1,4)
        self.assertEqual(q,.0001)
    def test_no_prorated_funding_and_hour_boundary(self):
        self.assertEqual(funding_events_between(1,30*60000),[])
        self.assertEqual(funding_events_between(30*60000+1,HOUR+1),[HOUR])
        self.assertEqual(funding_events_between(HOUR+1,HOUR+30*60000),[])
    def test_negative_funding_benefits_long_and_haircut_applies(self):
        zero=expected_net(1,100,100,100,0,0,1,9,.5,.5)
        negative=expected_net(1,100,100,100,0,-.001,1,9,.5,.5)
        self.assertAlmostEqual(negative-zero,5.)
    def test_oracle_drift_is_not_basis_alpha(self):
        d=decompose(1,2,99,109,100,110,0,0)
        self.assertEqual(d['basis_pnl'],0);self.assertEqual(d['oracle_direction_pnl'],20);self.assertEqual(d['net_pnl'],20)
    def test_costs_on_actual_entry_and_exit_values(self):
        d=decompose(-1,2,100,90,100,90,1,9)
        self.assertAlmostEqual(d['fees'],.342);self.assertAlmostEqual(d['net_pnl'],20.658)
    def test_disconnect_invalidates_pre_disconnect_snapshot(self):
        s=dict(book=[(10000,0,dict(time=10000,levels=[[dict(px='99',sz='1')],[dict(px='101',sz='1')]]))],
          ctx=[(10000,0,dict(oraclePx='100'))],book_times=[10000],ctx_times=[10000],resets=[10500])
        self.assertIsNotNone(asof(s,10400,CFG));self.assertIsNone(asof(s,10600,CFG))
    def test_constant_funding_forecast_has_correct_horizon_units(self):
        m=fit_ar1(np.full(168,.0001))
        self.assertAlmostEqual(predict(m,.0001,24,'ar1'),.0024)
    def test_transfers_are_not_profit(self):
        self.assertAlmostEqual(transfer_adjusted_return(1100000,1000000,100000),0)
    def test_score_uses_horizon_sample_std_and_no_intraday_vol(self):
        z=score(.01,[-.001,.001],trading_days=1,start_sample=True)
        self.assertAlmostEqual(z['volatility_horizon'],.002)
        self.assertAlmostEqual(z['financial_score'],.015/.063)
        self.assertAlmostEqual(z['score'],.8*.015/.063)
    def test_one_return_is_ineligible_and_losses_score_below_point_one(self):
        self.assertFalse(score(.1,[.01],trading_days=1,start_sample=True)['eligible'])
        z=score(-.01,[0.,0.],trading_days=1,start_sample=True)
        self.assertLess(z['financial_score'],.1)
    def test_missing_days_adjust_volatility_only(self):
        a=score(.02,[.01,-.01],[1,4],trading_days=1,start_sample=True)
        self.assertAlmostEqual(a['volatility_horizon'],.015)
        self.assertEqual(a['return_fraction'],.02)

if __name__=='__main__':unittest.main()
