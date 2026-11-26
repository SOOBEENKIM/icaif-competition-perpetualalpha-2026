import json
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .registry import Registry, EvaluationRejected, sha


class EvaluationControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reg = Registry(self.root / 'shared.sqlite3')
        self.reg.import_audit({'metadata_inventory_complete': True, 'exposures': [
            {'start': 100, 'end': 500, 'reason': 'earlier experiment', 'evidence': ['old/test.npz']}]})
        self.model = self.root / 'model.json'
        self.model.write_text('{"weight":0.5}')
        self.plan = dict(evaluation_kind='independent_latest_period', train_start=100,
            train_end=300, validation_start=300, validation_end=501,
            test_start=600, test_end=900, max_train_label_maturity=299,
            preprocessing_fit_last_time=299, selection_last_time=500,
            model_knowledge_last_time=299, comparators=['original_v10'],
            success_rule='positive net ROI > control, dailySTD <= control; report stress too',
            artifacts={str(self.model): sha(self.model)})

    def test_renamed_files_and_new_run_names_do_not_reset_exposure(self):
        for dirname in ('first', 'copied', 'new_seed', 'new_model', 'new_venue'):
            (self.root / dirname).mkdir()
            with self.assertRaisesRegex(EvaluationRejected, 'Previously exposed'):
                Registry(self.reg.path).preflight(400, 800)

    def test_uninitialized_registry_rejects_unknown_provenance(self):
        with self.assertRaisesRegex(EvaluationRejected, 'provenance unknown'):
            Registry(self.root / 'empty.sqlite3').preflight(600, 900)

    def test_global_latest_period_rejects_earlier_repartition(self):
        with self.assertRaises(EvaluationRejected):
            self.reg.preflight(20, 80)

    def test_label_purge_and_train_only_preprocessing(self):
        for key in ('max_train_label_maturity', 'preprocessing_fit_last_time'):
            with self.assertRaisesRegex(EvaluationRejected, 'Training labels or preprocessing'):
                self.reg.freeze({**self.plan, key: 300}, now=550)

    def test_future_pretraining_and_selection_are_rejected(self):
        for key in ('model_knowledge_last_time', 'selection_last_time'):
            with self.assertRaises(EvaluationRejected):
                self.reg.freeze({**self.plan, key: 600}, now=550)

    def test_freeze_must_precede_test(self):
        with self.assertRaisesRegex(EvaluationRejected, 'before its test starts'):
            self.reg.freeze(self.plan, now=600)

    def test_partial_period_is_rejected_before_loader_runs(self):
        trial = self.reg.freeze(self.plan, now=550)
        calls = []
        with self.assertRaisesRegex(EvaluationRejected, 'not finished'):
            self.reg.evaluate(trial, lambda p: calls.append(p), now=899)
        self.assertEqual(calls, [])
        self.assertEqual(self.reg.evaluate(trial, lambda p: {'roi': -0.1}, now=900), {'roi': -0.1})

    def test_mutated_model_rejected_before_data_access(self):
        trial = self.reg.freeze(self.plan, now=550)
        self.model.write_text('{"weight":0.6}')
        calls = []
        with self.assertRaisesRegex(EvaluationRejected, 'artifact changed'):
            self.reg.evaluate(trial, lambda p: calls.append(p), now=900)
        self.assertEqual(calls, [])

    def test_new_exposure_after_freeze_invalidates_trial(self):
        trial = self.reg.freeze(self.plan, now=550)
        self.reg.import_audit({'metadata_inventory_complete': True, 'exposures': [
            {'start': 600, 'end': 700, 'reason': 'another analysis opened prices', 'evidence': ['copy']}]})
        with self.assertRaises(EvaluationRejected):
            self.reg.consume(trial, now=900)

    def test_model_changed_during_evaluation_cannot_publish_result(self):
        trial = self.reg.freeze(self.plan, now=550)
        def mutate(_):
            self.model.write_text('{"weight":0.9}')
            return {'roi': 1.0}
        with self.assertRaisesRegex(EvaluationRejected, 'artifact changed'):
            self.reg.evaluate(trial, mutate, now=900)
        with self.reg.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM result').fetchone()[0], 0)
        with self.assertRaises(EvaluationRejected):
            self.reg.consume(trial, now=900)

    def test_failed_execution_cannot_reset_holdout(self):
        trial = self.reg.freeze(self.plan, now=550)
        def fail(_):
            raise RuntimeError('inference failed after loading data')
        with self.assertRaises(RuntimeError):
            self.reg.evaluate(trial, fail, now=900)
        with self.assertRaisesRegex(EvaluationRejected, 'already opened'):
            self.reg.consume(trial, now=900)
        with self.assertRaises(EvaluationRejected):
            self.reg.preflight(650, 1000)

    def test_cannot_register_multiple_nominees_for_same_test(self):
        trial = self.reg.freeze(self.plan, now=550)
        self.assertEqual(trial, self.reg.freeze(self.plan, now=551))
        with self.assertRaisesRegex(EvaluationRejected, 'Overlapping'):
            self.reg.freeze({**self.plan, 'new_study_name': 'another_try'}, now=551)

    def test_concurrent_access_has_only_one_winner(self):
        trial = self.reg.freeze(self.plan, now=550)
        def attempt(_):
            try:
                Registry(self.reg.path).consume(trial, now=900)
                return True
            except EvaluationRejected:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(attempt, range(2))), 1)


if __name__ == '__main__':
    unittest.main()
