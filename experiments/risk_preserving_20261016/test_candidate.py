import json,tempfile,unittest
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from replay_lab.ema_rebuild.engine import Controller
from .policy import RiskCapController
from .runtime import SignalFrame,SignalFile,Supervisor

class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        self.price=np.ones(10)*100;self.sigma=np.eye(10)*.000001
        self.activity=np.ones(10)*1000
        self.t=1792130400

    def call(self,control,qty,raw,retention=None):
        kwargs={} if retention is None else {'retention':retention}
        return control.propose(self.t,self.price,qty,1e6,raw,self.sigma,self.activity,self.activity,self.p,buffer=.2,**kwargs)

    def test_risk_reduction_bypasses_20_percent_hold_buffer(self):
        q=np.zeros(10);q[0]=400;raw=np.zeros(10);raw[0]=.036
        old=self.call(Controller(),q,raw)
        new=self.call(RiskCapController(),q,raw,np.full(10,.9))
        self.assertEqual(old['target_qty'][0],400)
        self.assertEqual(new['target_qty'][0],360)
        self.assertTrue(new['cap_overrode_soft_constraint'][0])

    def test_cap_bypasses_soft_turnover_but_never_increases(self):
        q=np.zeros(10);q[:3]=1000;raw=np.zeros(10);raw[:3]=.01
        self.p['risk']['normal_turnover_fraction']=.001
        new=self.call(RiskCapController(),q,raw,np.full(10,.5))
        self.assertTrue(np.all(np.abs(new['target_qty'])<=abs(raw)*1e6/self.price+1e-9))

    def test_inactive_cap_matches_incumbent(self):
        q=np.zeros(10);q[0]=400;raw=np.zeros(10);raw[0]=.036
        a=self.call(Controller(),q,raw);b=self.call(RiskCapController(),q,raw,np.ones(10))
        self.assertEqual(a['target_qty'],b['target_qty'])

    def test_short_cap_and_state_restore(self):
        q=np.zeros(10);q[0]=-400;raw=np.zeros(10);raw[0]=-.036
        c=RiskCapController();a=self.call(c,q,raw,np.full(10,.9))
        self.assertEqual(a['target_qty'][0],-360)
        self.assertIsNone(self.call(RiskCapController(c.snapshot()),q,raw,np.full(10,.9)))

    def test_invalid_gate_does_not_consume_cursor(self):
        c=RiskCapController();q=np.zeros(10);raw=np.ones(10)*.01
        self.assertIsNone(self.call(c,q,raw,np.full(10,1.1)))
        self.assertIsNotNone(self.call(c,q,raw,np.ones(10)))

    def frame(self,t=None):
        return SignalFrame(t or self.t,float(t or self.t)+1,[.02]*10,[.5]*10,self.sigma.tolist(),
                           self.activity.tolist(),self.activity.tolist(),'a'*64)

    def test_stale_signal_cannot_open_but_can_reduce_actual_risk(self):
        with tempfile.TemporaryDirectory() as d:
            f=SignalFile(Path(d)/'signal.json');f.publish(self.frame())
            s=Supervisor(self.p,f,Path(d)/'journal.jsonl')
            now=self.t+3602
            flat={'time':now,'equity':1e6,'qty':[0.]*10,'prices':self.price.tolist()}
            self.assertIsNone(s.observe(flat,now)['intent'])
            q=np.zeros(10);q[0]=2000
            flat.update(time=now+30,qty=q.tolist())
            e=s.observe(flat,now+30)
            self.assertTrue(e['intent']['reduce_only'])
            self.assertLess(e['intent']['target_qty'][0],2000)
            self.assertTrue(all(q>=0 for q in e['intent']['target_qty']))

    def test_corrupt_publication_keeps_risk_checks_blocks_increases(self):
        with tempfile.TemporaryDirectory() as d:
            f=SignalFile(Path(d)/'signal.json');f.publish(self.frame())
            s=Supervisor(self.p,f,Path(d)/'journal.jsonl')
            now=self.t+10;account={'time':now,'equity':1e6,'qty':[0.]*10,'prices':self.price.tolist()}
            s.observe(account,now)
            f.path.write_text('{}');account.update(time=now+30,equity=900000)
            e=s.observe(account,now+30)
            self.assertIsNotNone(e['signal_error']);self.assertEqual(e['intent']['target_qty'],[0.]*10)
            self.assertTrue(e['halted'])

    def test_future_signal_rejected_and_journal_uses_actual_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            f=SignalFile(Path(d)/'signal.json');f.publish(self.frame(self.t+3600))
            s=Supervisor(self.p,f,Path(d)/'journal.jsonl')
            q=[.4682,0,59.95]+[0.]*7
            e=s.observe({'time':self.t+10,'equity':1e6,'qty':q,'prices':self.price.tolist()},self.t+10)
            self.assertIsNone(e['intent']);self.assertFalse(e['fresh_signal'])
            self.assertEqual(e['actual_qty'],q)

    def test_stale_account_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            s=Supervisor(self.p,SignalFile(Path(d)/'absent'),Path(d)/'journal.jsonl')
            with self.assertRaises(ValueError):
                s.observe({'time':self.t-20,'equity':1e6,'qty':[0.]*10,'prices':self.price.tolist()},self.t)

    def test_completed_incumbent_round_not_repeated_after_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            f=SignalFile(Path(d)/'signal.json');f.publish(self.frame())
            state={'control':{'controller':{'last_decision':self.t}},'last_normal':self.t,
                   'peak':1000131.,'last_tick':self.t+2,'halted':False}
            s=Supervisor(self.p,f,Path(d)/'journal.jsonl',state)
            event=s.observe({'time':self.t+10,'equity':1e6,'qty':[0.]*10,'prices':self.price.tolist()},self.t+10)
            self.assertIsNone(event['intent']);self.assertEqual(s.peak,1000131.)

if __name__=='__main__':unittest.main()
