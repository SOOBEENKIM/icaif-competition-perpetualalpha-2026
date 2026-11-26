"""Promotion must not hide invalid data, exit losses, or a stronger control."""
import unittest
from .full_forward_review import decision


class ForwardDecisionTests(unittest.TestCase):
    e=dict(primary='candidate',controls=['incumbent','execution_control'])
    def rows(self):
        return [dict(scenario=s,arm=a,valid=True,net_pnl=p+1,net_after_terminal_exit=p,cash_rounding_bound=.001)
                for s in ['base','stress'] for a,p in [('incumbent',1.),('execution_control',2.),('candidate',3.)]]

    def test_complete_positive_advantage_required_in_both_conditions(self):
        rows=self.rows();self.assertTrue(all(r['passed'] for r in decision(rows,self.e)))
        rows[-1]['valid']=False
        self.assertFalse(all(r['passed'] for r in decision(rows,self.e)))

    def test_positive_mark_pnl_cannot_hide_exit_loss(self):
        rows=self.rows();rows[-1]['net_after_terminal_exit']=-1.
        self.assertFalse(decision(rows,self.e)[1]['passed'])

    def test_second_control_cannot_be_omitted_after_result(self):
        rows=self.rows();rows[-2]['net_after_terminal_exit']=4.
        self.assertFalse(decision(rows,self.e)[1]['passed'])

    def test_rounding_uncertainty_cannot_be_called_an_improvement(self):
        rows=self.rows();rows[-1]['net_after_terminal_exit']=2.001
        self.assertFalse(decision(rows,self.e)[1]['passed'])


if __name__=='__main__':unittest.main()
