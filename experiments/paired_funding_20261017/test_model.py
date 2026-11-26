import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .model import HOUR,panel,features,labels,fit_ridge,ridge_predict,paired_differential,carry_diagnostic,ar1_forecast


class Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
        rng=np.random.default_rng(9);n=400;times=1790000000000//HOUR*HOUR+np.arange(n)*HOUR
        self.p=dict(times=times,rates=rng.normal(0,.2,(n,3)),seen=np.repeat((times+50)[:,None],3,axis=1),symbols=['A','B','C'])

    def test_target_excludes_already_paid_funding(self):
        p=copy.deepcopy(self.p);p['rates'][:]=1.;p['rates'][80]=1000.
        y,maturity=labels(p,24)
        np.testing.assert_array_equal(y[80],np.ones(3)*24)
        np.testing.assert_array_equal(maturity[80],p['seen'][104])

    def test_missing_future_funding_is_not_zero_filled(self):
        p=copy.deepcopy(self.p);p['rates'][100,0]=np.nan
        y,_=labels(p,24)
        self.assertTrue(np.isnan(y[80,0]));self.assertTrue(np.isfinite(y[80,1]))

    def test_future_observations_cannot_change_existing_features(self):
        p=copy.deepcopy(self.p);original=features(p);p['rates'][251:]=10000
        np.testing.assert_allclose(original[:251],features(p)[:251],equal_nan=True)

    def test_future_labels_cannot_change_past_fit(self):
        p=copy.deepcopy(self.p);at=int(p['times'][250])+1000
        original=fit_ridge(p,features(p),72,at,self.cfg)
        p['rates'][251:]=10000
        other=fit_ridge(p,features(p),72,at,self.cfg)
        self.assertEqual(original,other)
        self.assertLess(original['last_mature_label'],at)

    def test_funding_sign_and_gross_notional_normalization(self):
        diff=paired_differential(30.,0.,2.)
        self.assertEqual(diff,10.)
        row=carry_diagnostic(30.,20.,18.,5.)
        self.assertEqual(row['direction'],-1.)
        self.assertEqual(row['realized_carry_bps'],20.)
        self.assertEqual(row['fee_only_net_bps'],2.)
        self.assertFalse(carry_diagnostic(20.,20.,18.,5.)['active'])

    def test_reversed_funding_is_loss_despite_positive_prediction(self):
        row=carry_diagnostic(-30.,20.,18.,5.)
        self.assertEqual(row['direction'],1.)
        self.assertEqual(row['fee_only_net_bps'],-38.)

    def test_ar1_units_and_known_information(self):
        p=copy.deepcopy(self.p);p['rates'][:]=.1
        at=int(p['times'][250])+1000
        self.assertAlmostEqual(ar1_forecast(p,250,0,72,at,self.cfg),7.2)
        p['rates'][251:]=10000
        self.assertAlmostEqual(ar1_forecast(p,250,0,72,at,self.cfg),7.2)

    def test_conflicting_settlements_are_rejected(self):
        rows={'A':[dict(time=HOUR+50,fundingRate='.001'),dict(time=HOUR+80,fundingRate='.002')]}
        with self.assertRaises(ValueError):panel(rows,['A'])

    def test_late_settlement_not_claimed_available(self):
        rows={'A':[dict(time=HOUR+1001,fundingRate='.001')]}
        with self.assertRaises(ValueError):panel(rows,['A'])

    def test_forecast_clipping_uses_training_range(self):
        x=features(self.p);at=int(self.p['times'][250])+1000
        model=fit_ridge(self.p,x,24,at,self.cfg)
        value=ridge_predict(model,np.ones(x.shape[-1])*1e9,24)
        self.assertLessEqual(value,model['bounds'][1]*24)
        self.assertGreaterEqual(value,model['bounds'][0]*24)


if __name__=='__main__':unittest.main()
