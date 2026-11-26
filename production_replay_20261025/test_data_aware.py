import copy
import unittest
from . import test_books as fixtures
from .test_paced_replay import Recorder
from .paced_book_replay import run_book_paced
from .data_aware_book_replay import run_book_data_aware


class SlowWorkRecorder(Recorder):
    def step_candidate(self,now,*args):
        super().step_candidate(now,*args)
        self.account.executor.now+=30.


class DataPauseTests(unittest.TestCase):
    def case(self,gap_end=None):
        v,p,t=fixtures.RecordedBookTests().case();p.__class__=Recorder
        for symbol,s in v.streams.items():
            book=s['book'][0][2];ctx=s['ctx'][0][2]
            s['book']=[((t+i)*1000,1,dict(book,time=(t+i)*1000)) for i in range(181)]
            s['ctx']=[((t+i)*1000,1,ctx.copy()) for i in range(181) if gap_end is None or not 90<=i<gap_end]
            s['book_times']=[x[0] for x in s['book']];s['ctx_times']=[x[0] for x in s['ctx']]
            if gap_end is not None:s['resets'].append((t+90)*1000)
        return v,p,t

    def test_missing_inputs_pause_all_decision_code(self):
        v,p,t=self.case(93)
        r=run_book_data_aware(v.streams,[],v.profile,v.sc,t,t+180,p,'incumbent',[])
        self.assertEqual([e['time']-t for e in r['events']],[0])
        self.assertEqual(r['input_pauses'],[t+90])
        self.assertTrue(r['input_gap_audit'][0]['bounded'])
        self.assertEqual(r['input_gap_audit'][0]['surrounding_span_seconds'],4)
        self.assertEqual(r['failures'],[])

    def test_extended_gap_remains_a_failure(self):
        v,p,t=self.case(181)
        r=run_book_data_aware(v.streams,[],v.profile,v.sc,t,t+180,p,'incumbent',[])
        self.assertFalse(r['input_gap_audit'][0]['bounded'])
        self.assertEqual(r['failures'][0]['kind'],'extended_unobserved_input_gap')

    def test_complete_inputs_leave_policy_and_accounting_unchanged(self):
        v,p,t=self.case()
        args=(v.streams,[],v.profile,v.sc,t,t+180,p,'incumbent',[])
        a=run_book_paced(*args);b=run_book_data_aware(*args)
        self.assertEqual(b.pop('input_pauses'),[])
        self.assertEqual(b.pop('input_gap_audit'),[])
        self.assertEqual(a,b)

    def test_slow_submission_work_does_not_trigger_immediate_next_iteration(self):
        v,p,t=self.case();p.__class__=SlowWorkRecorder
        r=run_book_data_aware(v.streams,[],v.profile,v.sc,t,t+180,p,'incumbent',[])
        self.assertEqual([e['time']-t for e in r['events']],[0,120])


if __name__=='__main__':unittest.main()
