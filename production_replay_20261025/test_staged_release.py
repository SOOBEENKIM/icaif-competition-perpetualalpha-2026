import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from execution_revision_20261024.guarded_portfolio import GuardedAccount,GuardedPortfolio
from . import staged_runner
from .protocol import ROOT,OUT,sha


class StagedReleaseTests(unittest.TestCase):
    def case(self,directory):
        folder=Path(directory);(folder/'qualification_review').mkdir()
        dependency=folder/'comparison.json';dependency.write_text('{"synthetic_test":true}')
        evidence=folder/'qualification_review/results.json'
        evidence.write_text(json.dumps(dict(experiments=[dict(primary='fixed_entry_1h',
            verdict=dict(short_conditional_trial_eligible=True))],source_results={'comparison.json':sha(dependency)})))
        p=GuardedPortfolio.__new__(GuardedPortfolio);p.account=GuardedAccount.__new__(GuardedAccount)
        manifest=dict(schema='production_replay_20261025_qualified_release_v1',
            source_checkpoint_is_live_reconciled=True,short_paper_gate_passed=True,
            qualification_sha256=sha(evidence),candidate_arm='fixed_entry_1h',
            portfolio_class='GuardedPortfolio',execution_revision=p.execution_revision,
            sources={'production_replay_20261025/staged_runner.py':sha(ROOT/'production_replay_20261025/staged_runner.py')})
        return folder,manifest,p,dependency

    def verify(self,folder,manifest,portfolio):
        (folder/'research_release.json').write_text(json.dumps(manifest))
        with patch.object(staged_runner,'OUT',folder),patch.object(staged_runner,'private_load',return_value={'portfolio':portfolio}):
            return staged_runner.verify_release(folder)

    def test_synthetic_release_requires_matching_completed_evidence(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            folder,m,p,_=self.case(directory)
            self.assertEqual(self.verify(folder,m,p)['candidate_arm'],'fixed_entry_1h')

    def test_boolean_pass_alone_is_insufficient(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            folder,m,p,_=self.case(directory);m.pop('qualification_sha256')
            with self.assertRaisesRegex(ValueError,'evidence'):self.verify(folder,m,p)

    def test_changed_results_cannot_reuse_old_qualification(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            folder,m,p,dependency=self.case(directory);dependency.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'dependency changed'):self.verify(folder,m,p)

    def test_a_different_candidate_cannot_borrow_the_winner_evidence(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            folder,m,p,_=self.case(directory);m['candidate_arm']='quote_inputs'
            with self.assertRaisesRegex(ValueError,'did not pass'):self.verify(folder,m,p)

    def test_research_seed_is_not_a_live_handover(self):
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            folder,m,p,_=self.case(directory);m['source_checkpoint_is_live_reconciled']=False
            with self.assertRaisesRegex(ValueError,'research seed'):self.verify(folder,m,p)


if __name__=='__main__':unittest.main()
