import unittest
import numpy as np
from .cashflow_overlay import entry_eligible

class EntryTests(unittest.TestCase):
    def test_unoccupied_long_cannot_reuse_zero_sign_entry_pass(self):
        t=np.array([-86400.,-10800.,0.]);px=np.array([[100.],[101.],[102.]])
        self.assertFalse(entry_eligible(0,np.array([1.]),t,px,np.zeros_like(px))[0])
        self.assertTrue(entry_eligible(0,np.array([-1.]),t,px,np.zeros_like(px))[0])

    def test_stale_close_never_qualifies_addition(self):
        t=np.array([-86400.,-10800.,0.]);px=np.array([[100.],[101.],[102.]])
        age=np.zeros_like(px);age[-1]=7200
        self.assertFalse(entry_eligible(0,np.array([-1.]),t,px,age)[0])

    def test_future_quote_cannot_change_decision(self):
        t=np.array([-86400.,-10800.,0.,3600.]);px=np.array([[100.],[101.],[102.],[1.]])
        a=entry_eligible(0,np.array([1.]),t,px,np.zeros_like(px))
        b=entry_eligible(0,np.array([1.]),t[:-1],px[:-1],np.zeros_like(px[:-1]))
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
