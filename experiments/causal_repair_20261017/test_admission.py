import copy
import unittest
from .admission import economic_admission


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.base={'metrics':{'complete_days':60,'roi':.002,'daily_std_sample':.001,'max_drawdown_magnitude':.01},'gates':{'accounting':True,'risk':True}}

    def test_predictive_or_risk_success_cannot_override_negative_costed_return(self):
        candidate=copy.deepcopy(self.base);candidate['metrics'].update(roi=-.003,daily_std_sample=.0005)
        self.assertFalse(economic_admission(candidate,self.base)['admitted'])

    def test_positive_but_inferior_candidate_is_not_called_improvement(self):
        candidate=copy.deepcopy(self.base);candidate['metrics']['roi']=.001
        self.assertFalse(economic_admission(candidate,self.base)['admitted'])

    def test_higher_return_at_higher_risk_is_not_silently_admitted(self):
        candidate=copy.deepcopy(self.base);candidate['metrics'].update(roi=.003,daily_std_sample=.002)
        self.assertFalse(economic_admission(candidate,self.base)['admitted'])

    def test_genuine_joint_improvement_passes(self):
        candidate=copy.deepcopy(self.base);candidate['metrics'].update(roi=.003,daily_std_sample=.0008,max_drawdown_magnitude=.008)
        self.assertTrue(economic_admission(candidate,self.base)['admitted'])


if __name__=='__main__':unittest.main()
