import unittest
import numpy as np
from pair_value_learning.study import INPUT,read
from .study import OUT as FEATURES
from .realized_study import OUT as LABELS
from . import realized_ranker,execution_value


class ExecutionLearningCausalityTests(unittest.TestCase):
    def test_future_execution_labels_cannot_change_daily_models(self):
        source=1785542400
        d=dict(np.load(INPUT));f=dict(np.load(FEATURES/'features.npz'))
        n=int(np.searchsorted(f['times'],source+86400));total=len(f['times'])
        d={k:(v[:n].copy() if v.ndim and len(v)==total else v.copy()) for k,v in d.items()}
        f={k:(v[:n].copy() if v.ndim and len(v)==total else v.copy()) for k,v in f.items()}
        labels=[r for r in read(LABELS/'execution_labels.json') if r['index']<n]
        poisoned=[dict(r,utility=1e9) if r['maturity']>=source else dict(r) for r in labels]
        for model in [realized_ranker,execution_value]:
            aa,ab,ac=model.build(d,f,labels,source)
            ba,bb,bc=model.build(d,f,poisoned,source)
            self.assertTrue(ac)
            self.assertEqual(ab,bb)
            for key in aa:np.testing.assert_array_equal(aa[key],ba[key])
            for a,b in zip(ac,bc):self.assertEqual(bytes(a[-1].save_raw()),bytes(b[-1].save_raw()))


if __name__=='__main__':unittest.main()
