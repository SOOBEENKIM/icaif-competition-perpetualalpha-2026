"""Correct only the original test assertion's Python-list abs() typo.

The initial frozen test file/log is retained. Production model/policy and
registered trial sources remain byte-for-byte unchanged.
"""
import unittest
import numpy as np
from . import test_invariants
from .policy import balanced_inventory


class ValidatedInvariants(test_invariants.Invariants):
    def test_imbalance_repair_only_reduces(self):
        ok, q = balanced_inventory([8., -10.], [10., -20.], .05)
        self.assertFalse(ok)
        np.testing.assert_array_equal(q, [5., -10.])
        self.assertTrue(np.all(abs(q) <= np.abs([8., -10.])))
        ok, q = balanced_inventory([8., 0.], [10., -20.], .05)
        self.assertFalse(ok)
        np.testing.assert_array_equal(q, [0., 0.])


if __name__ == '__main__':
    unittest.main()
