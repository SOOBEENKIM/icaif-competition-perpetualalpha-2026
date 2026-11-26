"""Leakage barriers and same-inventory wiring of the second frozen test."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import replication, replication_evaluate
from .full_forward_review import prepare
from .quote_history import Activated as HistoryActivated
from .quote_study import Activated as GuardActivated
from . import test_books as fixtures
from .qualification_review import qualification
from .test_qualification_review import fixture


class ReplicationTests(unittest.TestCase):
    def test_all_arms_share_initial_cash_and_inventory_and_leave_seed_unchanged(self):
        _,seed,t=fixtures.RecordedBookTests().case()
        seed.account.physical.fill(2,-.1,100.,.009,t-60)
        seed.account.owners['base'].fill(2,-.1,100.,.009,t-60)
        seed.account.reserved.qty[2]=.2
        original=copy.deepcopy(seed)
        experiment=dict(name='quote_history',start=t)
        for arm in ['incumbent','frozen_side_quote','quote_inputs']:
            p,direct=prepare(seed,experiment,arm)
            self.assertEqual(direct,'incumbent')
            self.assertIsNot(p,seed)
            np.testing.assert_array_equal(p.account.physical.qty,original.account.physical.qty)
            np.testing.assert_array_equal(p.account.reserved.qty,original.account.reserved.qty)
            self.assertEqual(p.account.physical.shadow_cash,original.account.physical.shadow_cash)
            if arm=='quote_inputs':self.assertIsInstance(p,HistoryActivated)
            if arm=='frozen_side_quote':self.assertIsInstance(p,GuardActivated)
            p.account.reserved.qty[0]=123.
        np.testing.assert_array_equal(seed.account.reserved.qty,original.account.reserved.qty)

    def test_public_outcomes_cannot_be_collected_before_fixed_end(self):
        with patch.object(replication_evaluate.time,'time',return_value=replication.END):
            with self.assertRaisesRegex(ValueError,'early'):
                replication_evaluate.collect({}, {})

    def test_freeze_is_impossible_after_start(self):
        with patch.object(replication.time,'time',return_value=replication.START):
            with self.assertRaisesRegex(ValueError,'before'):
                replication.freeze()

    def test_execution_model_error_can_block_raw_profit_improvement(self):
        groups,capacity,fidelity=fixture()
        fidelity['realistic_phase45']['absolute_exit_pnl_mismatch']=100.
        result=qualification(groups,capacity,fidelity)
        self.assertFalse(result['short_conditional_trial_eligible'])
        self.assertFalse(result['live_release_created'])

    def test_report_keeps_every_arm_phase_and_cost(self):
        groups,capacity,fidelity=fixture()
        for g in groups.values():
            for r in g['rows']:
                r.update(fill_count=0,valid=True,net_pnl=1.,net_after_terminal_exit=.5,fees=.1,funding=0.)
        result=dict(comparisons=groups,verdict=qualification(groups,capacity,fidelity))
        with tempfile.TemporaryDirectory() as temp,patch.object(replication_evaluate,'OUT',Path(temp)):
            replication_evaluate.render(result)
            text=(Path(temp)/'report.html').read_text()
        self.assertEqual(text.count('<tr>'),13)
        self.assertIn('10-26 00:05',text)
        self.assertIn('실제 수익이 아닙니다',text)


if __name__=='__main__':unittest.main()
