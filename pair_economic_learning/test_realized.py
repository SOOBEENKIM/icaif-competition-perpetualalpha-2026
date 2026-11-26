import unittest
import numpy as np
from .realized_ranker import training_mask
from .realized_labels import HOURS


class ExecutionLabelTests(unittest.TestCase):
    def test_completed_price_target_is_not_completed_execution_label(self):
        source=1800000000
        origins=source-np.array([72,95,96,97])*3600
        self.assertEqual(training_mask(origins,source).tolist(),[False,False,False,True])

    def test_future_label_never_in_training(self):
        source=1800000000;t=np.arange(source-100*86400,source+10*86400,86400)
        eligible=training_mask(t,source)
        self.assertTrue(np.all(t[eligible]+HOURS*3600<source))
        self.assertTrue(np.all(t[eligible]>=source-90*86400))


if __name__=='__main__':unittest.main()
