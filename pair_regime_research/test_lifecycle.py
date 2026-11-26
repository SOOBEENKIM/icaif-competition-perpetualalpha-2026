import unittest
import numpy as np
from .lifecycle import verdict

class LifecycleTests(unittest.TestCase):
    def state(self,source,relation=True,regime=True):
        return dict(signal_times=np.array([[source]]),relationship_allowed=np.array([[relation]]),regime_allowed=np.array([[regime]]))
    def test_old_engine_state_has_no_new_veto(self):
        self.assertEqual(verdict({},0,{},100),dict(block_renewal=False,exit_reason=None))
    def test_two_distinct_consecutive_sources_request_paid_exit(self):
        pair=dict(pair_index=0)
        x=verdict(pair,0,self.state(3600,relation=False),3605);self.assertTrue(x['block_renewal']);self.assertIsNone(x['exit_reason'])
        x=verdict(pair,0,self.state(3600,relation=False),3700);self.assertEqual(x['bad_sources'],1)
        x=verdict(pair,0,self.state(7200,relation=False),7210);self.assertEqual(x['exit_reason'],'relationship_invalidated')
    def test_recovery_resets_counter(self):
        pair=dict(pair_index=0)
        verdict(pair,0,self.state(3600,regime=False),3600)
        verdict(pair,0,self.state(7200),7200)
        self.assertIsNone(verdict(pair,0,self.state(10800,regime=False),10800)['exit_reason'])
    def test_missing_source_is_not_consecutive_confirmation(self):
        pair=dict(pair_index=0)
        verdict(pair,0,self.state(3600,regime=False),3600)
        self.assertIsNone(verdict(pair,0,self.state(10800,regime=False),10800)['exit_reason'])
        self.assertTrue(verdict(pair,0,self.state(10800),14401)['block_renewal'])
    def test_future_source_rejected(self):
        with self.assertRaises(ValueError):verdict(dict(pair_index=0),0,self.state(3600),3599)
if __name__=='__main__':unittest.main()
