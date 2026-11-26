import copy,hashlib,json,unittest
from pathlib import Path
import numpy as np
import xgboost as xgb
from .model import infer,fit

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


class DailyTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((HERE/'protocol.json').read_text())
        cls.direct=json.loads((ROOT/'experiments/rolling_feature_pairs_20261017/protocol.json').read_text())
        cls.symbols=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['trading_protocol']['symbols']
        with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
            cls.reference={k:z[k].copy() for k in ['market_times','close','volume','trades']}
        cls.states,cls.records,cls.arrays=infer(cls.reference['market_times'],cls.reference,cls.symbols,
            cls.spec['pairs'],cls.direct,cls.spec['tree'])

    def test_direct_control_is_unchanged(self):
        with np.load(ROOT/'experiments/rolling_feature_pairs_20261017/artifacts/reference_direct_24_state.npz') as z:
            for k in ['mu','beta','vol']:
                np.testing.assert_allclose(self.states['direct'][k],z[k],atol=1e-14,rtol=1e-12,equal_nan=True)
            for k in ['directions','source_prices','signal_times']:
                np.testing.assert_array_equal(self.states['direct'][k],z[k])

    def test_fixed_equal_mix_and_risk_inputs(self):
        np.testing.assert_allclose(self.states['mix']['mu'],(self.states['direct']['mu']+self.states['tree']['mu'])/2,
            atol=1e-14,equal_nan=True)
        for kind in ('tree','mix'):
            for k in ['directions','source_prices','signal_times','beta','vol','pairs']:
                np.testing.assert_array_equal(self.states[kind][k],self.states['direct'][k])

    def test_daily_training_strict_maturity_and_weight_mass(self):
        self.assertEqual({r['pair_index'] for r in self.records},{0,1})
        for r in self.records:
            self.assertEqual(r['fit_time']%86400,0)
            self.assertLess(r['last_maturity'],r['fit_time'])
            self.assertGreaterEqual(r['rows'],200)
            self.assertAlmostEqual(r['tree_weight_sum'],r['rows'],places=10)
            self.assertEqual(r['tree_rounds'],80)
            self.assertEqual(r['tree_initial_intercept'],r['direct'][0])

    def test_future_mutation_preserves_earlier_predictions_and_tree_hashes(self):
        cutoff=int(np.datetime64('2026-09-01T12:00:00','s').astype('int64'))
        changed=copy.deepcopy(self.reference);after=changed['market_times']>cutoff
        changed['close'][after]*=1.2
        states,records,_=infer(changed['market_times'],changed,self.symbols,self.spec['pairs'],self.direct,self.spec['tree'])
        for k in self.states:
            np.testing.assert_allclose(states[k]['mu'][~after],self.states[k]['mu'][~after],atol=1e-14,rtol=1e-12,equal_nan=True)
        self.assertEqual([r for r in records if r['fit_time']<=cutoff],
                         [r for r in self.records if r['fit_time']<=cutoff])

    def test_roundtrip_saved_model_and_unmatured_label_rejected(self):
        rng=np.random.default_rng(20261017);x=rng.normal(size=(240,20));y=x[:,0]*x[:,1]
        now=200*86400;maturity=now-np.arange(240,0,-1)*21600
        record,booster,blob=fit(x,y,maturity,now,self.direct,self.spec['tree'])
        restored=xgb.Booster();restored.load_model(bytearray(blob))
        matrix=xgb.DMatrix(x,nthread=1)
        np.testing.assert_array_equal(booster.predict(matrix),restored.predict(matrix))
        self.assertEqual(record['tree_sha256'],hashlib.sha256(blob).hexdigest())
        bad=maturity.copy();bad[-1]=now
        with self.assertRaisesRegex(ValueError,'Unmatured'):
            fit(x,y,bad,now,self.direct,self.spec['tree'])

    def test_insufficient_training_produces_no_model(self):
        self.assertIsNone(fit(np.zeros((50,20)),np.zeros(50),np.arange(50),1000,self.direct,self.spec['tree']))


if __name__=='__main__':unittest.main()
