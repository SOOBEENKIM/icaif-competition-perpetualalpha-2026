import unittest
import numpy as np
from .estimator_audit import variance_scaling


class EstimatorTests(unittest.TestCase):
    def test_random_walk_and_stationary_process_are_distinguished(self):
        rng=np.random.default_rng(8301); increments=rng.normal(size=100000)
        path=np.cumsum(increments)
        h=variance_scaling(path)
        self.assertTrue(.47<h<.53,h)
        self.assertLess(abs(variance_scaling(increments)),.02)

    def test_translation_and_scale_do_not_change_memory(self):
        x=np.cumsum(np.random.default_rng(1).normal(size=1500))
        self.assertAlmostEqual(variance_scaling(x),variance_scaling(10*x+100),places=12)

    def test_constant_or_insufficient_data_are_not_a_pass(self):
        self.assertIsNone(variance_scaling(np.ones(1500)))
        self.assertIsNone(variance_scaling(np.arange(20)))


if __name__=='__main__':unittest.main()
