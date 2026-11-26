import copy,unittest
from unittest.mock import patch
import numpy as np
from candidate_handoff import test_risk_priority as fixtures
from competition_live.profile import START
from competition_live.runtime import fingerprint
from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
from .hourly_portfolio import adopt

class HourlyTests(unittest.TestCase):
    def case(self):
        case=list(fixtures.RiskPriorityTests().setup_case());case[0]=adopt(case[0]);case[1]=case[0].account
        return case
    def tick(self,c):
        p,a,t,orders,stream,packet,target,profile=c
        with patch('production_replay_20261025.hourly_portfolio.target_from_frame',return_value=(dict(target_qty=target.tolist()),{})),patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)
    def new_hour(self,c):
        import hashlib,json
        c[2][0]=START+3600+120;body=c[5]['body'];source=START+3600
        body['published_at']=c[2][0];body['frame']['source_time']=source
        old=body['original_signal'];old['published_at']=c[2][0]
        event=old['publication'];event['received_ms']=int(c[2][0]*1000)
        frame=event['data']['frame'];frame['source_time']=source;frame['published_time']=c[2][0]
        event['data']['publication_sha256']=hashlib.sha256(json.dumps({k:frame[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()
        c[5]['sha256']=fingerprint(body)
    def test_new_hour_is_eligible_without_waiting_six_hours(self):
        c=self.case();self.tick(c);self.new_hour(c);self.tick(c)
        self.assertEqual(c[0].last_decision,START+3600)
        self.assertEqual(c[0].base_task['expires'],START+3600+900)
        self.assertEqual(sum(e['kind']=='hourly_fresh_decision' for e in c[0].events),2)
    def test_old_hour_cannot_create_new_decision(self):
        c=self.case();self.tick(c);c[2][0]=START+3600+120;self.tick(c)
        self.assertEqual(c[0].last_decision,START)
        self.assertIsNone(c[0].base_task)
    def test_late_publication_cannot_be_backdated_into_window(self):
        c=self.case();self.new_hour(c);c[2][0]=START+3600+901;self.tick(c)
        self.assertNotEqual(c[0].last_decision,START+3600)
    def test_same_hour_does_not_generate_duplicate_intent(self):
        c=self.case();self.tick(c);c[2][0]+=20;self.tick(c);c[2][0]+=20;self.tick(c)
        self.assertEqual(sum(e['kind']=='hourly_fresh_decision' for e in c[0].events),1)
        self.assertEqual([x.coin for x in c[3]],['pas:BTC','pas:ETH'])
    def test_risk_cut_has_priority_and_cannot_be_reinflated(self):
        c=self.case();c[6][0]=.02;self.tick(c)
        self.assertEqual([x.coin for x in c[3]],['pas:BTC'])
        self.assertAlmostEqual(c[1].physical.qty[0],.005)
        self.assertAlmostEqual(c[0].base_task['target'][0],.005)
    def test_expired_task_cannot_open_short(self):
        c=self.case();self.tick(c);c[2][0]=START+901;self.tick(c)
        self.assertEqual(c[1].physical.qty[1],0.)
        self.assertIsNone(c[0].base_task)
    def test_adoption_preserves_all_owned_reserved_financial_state(self):
        c=self.case();p=c[0];p.account.reserved.qty[0]=-.03;new=adopt(p)
        for name in ['physical','reserved']:
            for field in ['qty','entry','cash','realized','fees','funding']:
                np.testing.assert_equal(getattr(getattr(new.account,name),field),getattr(getattr(p.account,name),field))
        self.assertIsNone(new.base_task);self.assertEqual(new.last_decision,p.last_decision)

if __name__=='__main__':unittest.main()
