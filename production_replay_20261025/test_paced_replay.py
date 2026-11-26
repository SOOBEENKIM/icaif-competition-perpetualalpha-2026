import unittest
from unittest.mock import patch
import numpy as np
from candidate_handoff.portfolio import CandidatePortfolio
from . import test_books as fixtures
from . import paced_book_replay as paced
from .book_replay import run_book


class Recorder(CandidatePortfolio):
    def step_candidate(self,now,*args):self.events.append(dict(kind='test_evaluation',time=now))


class SingleOrder(CandidatePortfolio):
    def step_candidate(self,now,streams,*args):
        if getattr(self,'placed',False):return
        self.submit_baseline(now,np.array([1.]+[0.]*9),streams,np.full(10,100.),np.eye(10)*1e-6,False)
        self.placed=True


class CadenceTests(unittest.TestCase):
    def run_case(self,phase):
        v,p,t=fixtures.RecordedBookTests().case();p.__class__=Recorder
        for symbol,s in v.streams.items():
            book=s['book'][0][2];ctx=s['ctx'][0][2]
            s['book']=[((t+i)*1000,1,dict(book,time=(t+i)*1000)) for i in range(181)]
            s['ctx']=[((t+i)*1000,1,ctx.copy()) for i in range(181)]
            s['book_times']=[x[0] for x in s['book']];s['ctx_times']=[x[0] for x in s['ctx']]
        with patch.object(paced,'PHASE_SECONDS',phase):
            r=paced.run_book_paced(v.streams,[],v.profile,v.sc,t,t+180,p,'incumbent',[])
        return t,r

    def test_ninety_second_cadence_is_not_fifteen_seconds(self):
        t,r=self.run_case(0)
        self.assertEqual([e['time']-t for e in r['events']],[0,90])
        self.assertEqual(r['snapshots'][-1]['time'],t+180)

    def test_phase_shift_keeps_same_initial_and_terminal_valuation(self):
        t,r=self.run_case(45)
        self.assertEqual([e['time']-t for e in r['events']],[45,135])
        self.assertEqual(r['snapshots'][0]['time'],t)
        self.assertEqual(r['snapshots'][-1]['time'],t+180)

    def test_fifteen_second_setting_is_identical_through_real_fill_account(self):
        v,p,t=fixtures.RecordedBookTests().case();p.__class__=SingleOrder
        a=run_book(v.streams,[],v.profile,v.sc,t,t+10,p,'incumbent',[])
        with patch.object(paced,'LOOP_SECONDS',15),patch.object(paced,'PHASE_SECONDS',0):
            b=paced.run_book_paced(v.streams,[],v.profile,v.sc,t,t+10,p,'incumbent',[])
        self.assertTrue(a['fills']);self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
