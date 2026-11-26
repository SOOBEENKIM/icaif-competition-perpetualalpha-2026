import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from validation_rebuild.model import train, predict, publication
from replay_lab.evaluation_control.registry import Registry, EvaluationRejected, sha


class ModelBoundaryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(26)
        n = 180*24
        self.data = dict(market_times=1735689600+np.arange(n)*3600,
            close=100*np.exp(np.cumsum(rng.normal(0, .004, (n, 10)), axis=0)),
            funding=np.zeros((n, 10)), volume=np.ones((n, 10))*100, trades=np.ones((n, 10)))
        self.spec = dict(horizon=24, execution_delay=1, sample_hours=24,
            half_life_days=45, ridge_penalty=.1, minimum_rows=200,
            target_clip=5, forecast_clip=2)
        self.cutoff = int(self.data["market_times"][120*24])
        self.start = self.cutoff-90*86400

    def test_validation_and_test_values_cannot_change_training(self):
        a = train(self.data, self.spec, self.start, self.cutoff)
        changed = copy.deepcopy(self.data)
        later = changed["market_times"] >= self.cutoff
        changed["close"][later] = np.nan
        changed["funding"][later] = 999
        b = train(changed, self.spec, self.start, self.cutoff)
        for key in ("center", "scale", "beta"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(all(r["maturity"] < self.cutoff for r in a["label_rows"]))
        self.assertTrue(all(r["maturity"] == r["source"]+25*3600 for r in a["label_rows"]))

    def test_predict_never_fits_or_constructs_labels_and_preserves_prefix(self):
        model = train(self.data, self.spec, self.start, self.cutoff)
        before = copy.deepcopy(model)
        with patch("validation_rebuild.model.fit", side_effect=AssertionError("refit")), \
             patch("validation_rebuild.model.labels", side_effect=AssertionError("labels")):
            _, full = predict(self.data, model)
            prefix = {k: v[:-48] for k,v in self.data.items()}
            _, part = predict(prefix, model)
        np.testing.assert_allclose(full[:-48], part, equal_nan=True)
        for key in ("center", "scale", "beta"):
            np.testing.assert_array_equal(before[key], model[key])

    def test_publication_reaches_actual_portfolio_input_without_retiming(self):
        event = dict(received_ms=201000, data=dict(
            frame=dict(source_time=100, published_time=200, weights=[.1, .2]),
            incumbent=[.1, .2], candidate=[.1, .2]))
        result = publication(event, [.01, -.03])
        self.assertEqual(result["data"]["incumbent"], [.01, -.03])
        self.assertEqual(result["data"]["frame"]["weights"], [.01, -.03])
        self.assertEqual(result["data"]["frame"]["published_time"], 200)
        self.assertEqual(result["received_ms"], 201000)
        self.assertEqual(event["data"]["incumbent"], [.1, .2])
        self.assertEqual(result["data"]["publication_sha256"], hashlib.sha256(
            json.dumps(result["data"]["frame"], sort_keys=True).encode()).hexdigest())


class EvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.registry = Registry(root/"registry.sqlite3")
        self.registry.import_audit(dict(metadata_inventory_complete=True,
            exposures=[dict(start=0, end=100, reason="seen", evidence=["audit"])]))
        self.file = root/"candidate.json"; self.file.write_text("{}")
        self.plan = dict(train_start=0, train_end=40, validation_start=40,
            validation_end=90, test_start=200, test_end=300,
            max_train_label_maturity=39, preprocessing_fit_last_time=39,
            selection_last_time=100, model_knowledge_last_time=100,
            artifacts={str(self.file): sha(self.file)}, success_rule="PnL and Q",
            comparators=["incumbent"], evaluation_kind="independent_latest_period")

    def test_seen_period_rejected_even_after_renaming_venue(self):
        with self.assertRaises(EvaluationRejected):
            self.registry.preflight(50, 80)

    def test_boundary_crossing_label_is_rejected(self):
        p = dict(self.plan, max_train_label_maturity=40)
        with self.assertRaises(EvaluationRejected):
            self.registry.freeze(p, now=150)

    def test_early_access_rejected_before_loader_and_single_use(self):
        ident = self.registry.freeze(self.plan, now=150)
        called = []
        with self.assertRaises(EvaluationRejected):
            self.registry.evaluate(ident, lambda p: called.append(1), now=250)
        self.assertEqual(called, [])
        self.registry.evaluate(ident, lambda p: {"net_pnl": -1}, now=301)
        with self.assertRaises(EvaluationRejected):
            self.registry.evaluate(ident, lambda p: called.append(1), now=302)
        self.assertEqual(called, [])

    def test_changed_source_rejected_before_loader(self):
        ident = self.registry.freeze(self.plan, now=150)
        self.file.write_text('{"changed":true}')
        with self.assertRaises(EvaluationRejected):
            self.registry.evaluate(ident, lambda p: self.fail("loaded"), now=301)


if __name__ == "__main__":
    unittest.main()
