import copy,json,unittest
from pathlib import Path
import numpy as np
from .model import calibrate,mapped_states

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((HERE/'protocol.json').read_text())
        with np.load(ROOT/'experiments/daily_tree_pairs_20261017/artifacts/reference_forecasts.npz') as z:
            cls.arrays={k:z[k].copy() for k in z.files}
        cls.pred,cls.records=calibrate(cls.arrays,cls.spec)

    def test_strict_maturity_sampling_and_slope_bounds(self):
        self.assertTrue(self.records)
        for r in self.records:
            self.assertLess(r['last_maturity'],r['time'])
            self.assertEqual(r['first_source']%86400,0);self.assertEqual(r['last_source']%86400,0)
            self.assertGreaterEqual(r['rows'],12)
            self.assertGreaterEqual(r['affine_slope'],0);self.assertLessEqual(r['affine_slope'],1)
            self.assertLessEqual(r['current_original_fit_time'],r['time'])

    def test_exact_blend_and_missing_forecast_suppression(self):
        np.testing.assert_allclose(self.pred['cmix'],(self.pred['ctree']+self.arrays['prediction_direct'])/2,
                                   atol=1e-14,equal_nan=True)
        self.assertTrue(np.isnan(self.pred['ctree'][:1000]).all())
        self.assertTrue(np.isnan(self.pred['cmix'][:1000]).all())

    def test_unmatured_labels_and_future_forecasts_do_not_change_current_output(self):
        cutoff=int(np.datetime64('2026-09-01T12:00:00','s').astype('int64'))
        changed=copy.deepcopy(self.arrays)
        unavailable=changed['maturity']>=cutoff
        changed['target'][unavailable]=3.9
        changed['terminal_fresh'][unavailable]=False
        changed['prediction_tree'][changed['times']>cutoff]*=-1
        pred,records=calibrate(changed,self.spec)
        before=changed['times']<=cutoff
        for k in pred:np.testing.assert_array_equal(pred[k][before],self.pred[k][before])
        self.assertEqual([r for r in records if r['time']<=cutoff],[r for r in self.records if r['time']<=cutoff])

    def test_current_terminal_freshness_cannot_remove_current_forecast(self):
        at=int(np.flatnonzero(self.arrays['times']==np.datetime64('2026-09-01T12:00:00','s').astype('int64'))[0])
        changed=copy.deepcopy(self.arrays);changed['terminal_fresh'][at:]=False
        pred,_=calibrate(changed,self.spec)
        np.testing.assert_array_equal(pred['ctree'][:at+1],self.pred['ctree'][:at+1])

    def test_missing_mature_endpoint_is_excluded(self):
        chosen=next(r for r in self.records if r['rows']>=30 and r['pair_index']==0)
        now=chosen['time'];changed=copy.deepcopy(self.arrays)
        ix=np.flatnonzero(changed['times']==chosen['last_source'])[0]
        changed['terminal_fresh'][ix,0]=False
        _,records=calibrate(changed,self.spec)
        changed_record=next(r for r in records if r['time']==now and r['pair_index']==0)
        self.assertEqual(changed_record['rows'],chosen['rows']-1)

    def test_future_coefficient_stamp_is_rejected(self):
        changed=copy.deepcopy(self.arrays)
        i,k=np.argwhere(np.isfinite(changed['prediction_tree']))[0]
        changed['fit_stamps'][i,k]=changed['times'][i]+1
        with self.assertRaisesRegex(ValueError,'unavailable'):
            calibrate(changed,self.spec)

    def test_mapping_preserves_source_risk_units(self):
        with np.load(ROOT/'experiments/daily_tree_pairs_20261017/artifacts/reference_tree_24_state.npz') as z:
            original={k:z[k].copy() for k in z.files}
        states=mapped_states(self.arrays['times'],None,original,self.pred,self.arrays)
        for state in states.values():
            for k in ['source_prices','signal_times','beta','vol','pairs']:
                np.testing.assert_array_equal(state[k],original[k])
            self.assertTrue((state['directions'][~np.isfinite(state['mu'])]==0).all())


if __name__=='__main__':unittest.main()
