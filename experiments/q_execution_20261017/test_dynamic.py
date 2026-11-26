import unittest
import numpy as np
from .dynamic import budget_path


class BudgetTests(unittest.TestCase):
    def test_current_and_future_changes_cannot_affect_current_decision(self):
        times=np.arange(300)*3600;eq=np.exp(np.arange(300)*.0001);valid=np.ones(300,bool)
        a,_=budget_path(times,eq,valid)
        cut=228;changed=eq.copy();changed[cut:]=.01
        b,_=budget_path(times,changed,valid)
        np.testing.assert_array_equal(a[:cut+1],b[:cut+1])
        self.assertEqual(a[cut],4.)

    def test_loss_or_missing_history_uses_original_budget(self):
        times=np.arange(300)*3600;eq=np.exp(-np.arange(300)*.0001)
        a,_=budget_path(times,eq,np.ones(300,bool));self.assertTrue((a==1).all())
        b,_=budget_path(times,eq[::-1],np.zeros(300,bool));self.assertTrue((b==1).all())

    def test_only_changes_at_daily_boundary(self):
        times=np.arange(300)*3600;eq=np.exp(np.arange(300)*.0001)
        a,_=budget_path(times,eq,np.ones(300,bool))
        ix=np.flatnonzero(np.diff(a)!=0)+1
        self.assertTrue(np.all(times[ix]%86400==43200))


if __name__=='__main__':unittest.main()
