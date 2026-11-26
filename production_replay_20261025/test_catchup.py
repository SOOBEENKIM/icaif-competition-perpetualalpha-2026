import copy,hashlib,json,unittest
from unittest.mock import patch
import numpy as np
from candidate_handoff import test_risk_priority as fixtures
from competition_live.profile import START
from competition_live.runtime import fingerprint
from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
from .catchup_portfolio import adopt


class CatchupTests(unittest.TestCase):
    def case(self):
        c=list(fixtures.RiskPriorityTests().setup_case());c[0]=adopt(c[0]);c[1]=c[0].account;return c

    def source(self,c,source,delay):
        c[2][0]=source+delay;b=c[5]['body'];b['published_at']=c[2][0];b['frame']['source_time']=source
        old=b['original_signal'];old['published_at']=c[2][0];event=old['publication'];event['received_ms']=int(c[2][0]*1000)
        f=event['data']['frame'];f['source_time']=source;f['published_time']=c[2][0]
        event['data']['publication_sha256']=hashlib.sha256(json.dumps({k:f[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()
        c[5]['sha256']=fingerprint(b)

    def tick(self,c):
        p,a,t,orders,stream,packet,target,profile=c
        with patch('production_replay_20261025.catchup_portfolio.target_from_frame',return_value=(dict(target_qty=target.tolist()),{})),patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)

    def test_missing_boundary_can_catch_up_using_next_fresh_hour(self):
        c=self.case();self.source(c,START+3600,120);self.tick(c)
        self.assertEqual(c[0].last_decision,START)
        e=next(x for x in c[0].events if x['kind']=='fresh_block_catchup_decision')
        self.assertEqual(e['source'],START+3600);self.assertEqual(e['time'],START+3720)

    def test_current_21minute_lag_is_not_backdated_or_silently_missed(self):
        c=self.case();self.source(c,START,1260);self.tick(c)
        e=next(x for x in c[0].events if x['kind']=='fresh_block_catchup_decision')
        self.assertEqual(e['time'],START+1260);self.assertEqual(c[0].base_task['entry_expires'],START+4860)

    def test_second_fresh_source_in_same_block_cannot_duplicate_decision(self):
        c=self.case();self.source(c,START+3600,120);self.tick(c);self.source(c,START+7200,120);self.tick(c)
        self.assertEqual(sum(x['kind']=='fresh_block_catchup_decision' for x in c[0].events),1)

    def test_previous_block_source_cannot_supply_new_allocation(self):
        c=self.case();self.source(c,START,120);self.tick(c);c[2][0]=START+21600+120
        # Unavailable old packet is omitted by the runner, never fed as fresh.
        p,a,t,orders,stream,packet,target,profile=c
        with patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},None,profile)
        self.assertEqual(sum(x['kind']=='fresh_block_catchup_decision' for x in p.events),1)

    def test_risk_reduction_still_has_priority(self):
        c=self.case();c[6][0]=.02;self.source(c,START+3600,120);self.tick(c)
        self.assertAlmostEqual(c[1].physical.qty[0],.005)
        self.assertAlmostEqual(c[0].base_task['target'][0],.005)

    def test_adoption_preserves_owned_cash_and_reserves(self):
        c=self.case();p=c[0];p.account.reserved.qty[0]=-.03;q=adopt(p)
        for name in ['physical','reserved']:
            for f in ['qty','entry','cash','realized','fees','funding']:np.testing.assert_equal(getattr(getattr(p.account,name),f),getattr(getattr(q.account,name),f))
        self.assertEqual(p.last_decision,q.last_decision)

if __name__=='__main__':unittest.main()
