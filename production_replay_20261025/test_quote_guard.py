import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from competition_live.live_account import LiveAccount
from candidate_handoff import test_risk_priority as fixtures
from .quote_guard import QuoteAccount, adopt, freeze_sides


class QuoteGuardTests(unittest.TestCase):
    def account(self,held=0.):
        a=QuoteAccount.__new__(QuoteAccount)
        a.owners={'base':SimpleNamespace(qty=np.array([held]))}
        a.frozen_side_prices=np.array([[99.,101.]])
        return a

    def seal(self,a,target,limit=None):
        intent=dict(packet_id='base:1',owner_targets={'base':np.array([target])})
        if limit is not None:intent['owner_limits']={'base':{'0':limit}}
        with patch.object(LiveAccount,'seal',side_effect=lambda r,p,s:r):
            return a.seal(intent,{}, {})

    def test_entry_cap_uses_correct_executable_side(self):
        self.assertAlmostEqual(self.seal(self.account(),1)['owner_limits']['base']['0'],101.101)
        self.assertAlmostEqual(self.seal(self.account(),-1)['owner_limits']['base']['0'],98.901)

    def test_existing_economic_limit_never_loosened(self):
        self.assertEqual(self.seal(self.account(),1,100.5)['owner_limits']['base']['0'],100.5)
        self.assertEqual(self.seal(self.account(),-1,99.5)['owner_limits']['base']['0'],99.5)

    def test_no_decision_quote_cannot_create_exposure(self):
        a=self.account();a.frozen_side_prices=None
        with self.assertRaises(ValueError):self.seal(a,1.)

    def test_reduction_does_not_require_entry_quote(self):
        a=self.account(1.);a.frozen_side_prices=None
        r=self.seal(a,0.)
        self.assertEqual(r['owner_targets']['base'][0],0.)

    def test_saved_quotes_do_not_follow_mutated_book(self):
        q={'BTC':dict(book=dict(levels=[[dict(px='99')],[dict(px='101')]]))}
        anchors=freeze_sides(['BTC'],q);q['BTC']['book']['levels'][1][0]['px']='110'
        self.assertEqual(anchors.tolist(),[[99.,101.]])

    def test_integration_preserves_risk_cap_priority(self):
        c=list(fixtures.RiskPriorityTests().setup_case());c[0]=adopt(c[0]);c[1]=c[0].account
        p,a,t,orders,stream,packet,target,profile=c
        target[0]=.02
        with patch('production_replay_20261025.quote_guard.target_from_frame',return_value=(dict(target_qty=target.tolist()),{})),patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in p.p['symbols']},packet,profile)
        self.assertAlmostEqual(a.physical.qty[0],.005)
        self.assertAlmostEqual(p.base_task['target'][0],.005)
        self.assertIn('frozen_side_prices',p.base_task)


if __name__=='__main__':unittest.main()
