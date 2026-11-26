import copy
import json
import tempfile
import gzip
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from competition_live.coordinator import Portfolio
from validation_rebuild.tape import TapeWindow
from validation_rebuild.model import publication
from validation_rebuild import forward
from validation_rebuild.registry import Registry, EvaluationRejected, sha
from competition_live.profile import ROOT, load_profile
from experiments.basis_carry_20261017.probe import asof


class ForwardIntegrationTests(unittest.TestCase):
    def test_lookahead_cannot_publish_future_signal_or_use_future_book(self):
        cfg = json.loads((ROOT/"competition_live/strategy_plan.json").read_text())["book_protocol"]
        rows = []
        for t, price in [(1000, 100), (2000, 200)]:
            rows.extend([
                dict(received_ms=t, channel="l2Book", data=dict(coin="pas:BTC", time=t,
                     levels=[[dict(px=str(price-.01),sz="1")],[dict(px=str(price+.01),sz="1")]])),
                dict(received_ms=t, channel="activeAssetCtx",
                     data=dict(coin="pas:BTC",ctx=dict(markPx=str(price),oraclePx=str(price),funding="0")))])
        rows.append(dict(received_ms=2500,channel="strategySignal",data={}))
        tape = TapeWindow(rows); streams = tape.ensure(1000)
        self.assertEqual(asof(streams["BTC"],1000,cfg)["mid"],100)
        self.assertEqual(list(tape.publications(1000)),[])
        self.assertEqual(len(list(tape.publications(2500))),1)

    def test_disconnect_invalidates_prior_book_even_if_age_is_short(self):
        cfg = json.loads((ROOT/"competition_live/strategy_plan.json").read_text())["book_protocol"]
        rows = [
            dict(received_ms=1000,channel="l2Book",data=dict(coin="pas:BTC",time=1000,
                levels=[[dict(px="99",sz="1")],[dict(px="101",sz="1")]])),
            dict(received_ms=1000,channel="activeAssetCtx",data=dict(coin="pas:BTC",
                ctx=dict(markPx="100",oraclePx="100",funding="0"))),
            dict(received_ms=1500,channel="connection",data=dict(state="disconnected"))]
        tape = TapeWindow(rows); streams = tape.ensure(1500)
        self.assertIsNone(asof(streams["BTC"],1500,cfg))

    def test_candidate_target_reaches_coordinator_and_produces_short_fill(self):
        p = load_profile()
        plan = json.loads((ROOT/"competition_live/strategy_plan.json").read_text())
        plan.update(paper_start_unix=0, capture_end_unix=7*86400,
                    end_liquidation_seconds=600, trading_protocol=p["trading"])
        frame = dict(source_time=0, published_time=1, weights=[0.]*10, retention=[1.]*10,
                     sigma=(np.eye(10)*1e-6).tolist(), volume=[1e6]*10,
                     trades=[1000]*10, source_hash="x"*64)
        event = dict(channel="strategySignal", received_ms=2000,
                     data=dict(frame=frame, incumbent=[0.]*10, candidate=[0.]*10))
        corrected = publication(event, [0., 0., -.02]+[0.]*7)
        events = []
        for sec in range(29, 42):
            for symbol in p["trading"]["symbols"]:
                events.extend([
                    dict(received_ms=sec*1000, channel="l2Book", data=dict(coin="pas:"+symbol,
                        time=sec*1000, levels=[[dict(px="99.99", sz="1000000")],
                                              [dict(px="100.01", sz="1000000")]])),
                    dict(received_ms=sec*1000, channel="activeAssetCtx", data=dict(coin="pas:"+symbol,
                        ctx=dict(markPx="100", oraclePx="100", funding="0")))])
        tape = TapeWindow(events)
        streams = tape.ensure(30000)
        baseline = Portfolio("first_plus_decision_gate", "base", plan, lambda *args: None)
        candidate = Portfolio("first_plus_decision_gate", "base", plan, lambda *args: None)
        baseline.step(30, streams, event, [])
        candidate.step(30, streams, corrected, [])
        self.assertEqual(len(baseline.account.physical.fills), 0)
        self.assertLess(candidate.account.physical.qty[2], 0)
        self.assertTrue(all(x["decision_time"] < x["time"]
                            for x in candidate.account.physical.fills))
        self.assertFalse(candidate.account.unknown)
        self.assertFalse(candidate.fatal)

    def test_actual_forward_entrypoint_rejects_before_raw_or_network_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/"model.json"; model.write_text("{}")
            registry = Registry(root/"r.sqlite3")
            registry.import_audit(dict(metadata_inventory_complete=True,
                exposures=[dict(start=0, end=100, reason="seen", evidence=[])]))
            plan = dict(train_start=0, train_end=40, validation_start=40, validation_end=90,
                test_start=200, test_end=300, max_train_label_maturity=39,
                preprocessing_fit_last_time=39, selection_last_time=100, model_knowledge_last_time=100,
                artifacts={str(model): sha(model)}, comparators=["incumbent"], success_rule="PnL and Q",
                evaluation_kind="independent_latest_period")
            ident = registry.freeze(plan, now=150)
            (root/"registration.json").write_text(json.dumps(dict(trial_id=ident)))
            with patch.object(forward, "FWD", root), patch.object(forward, "Registry", return_value=registry), \
                 patch("_sfinx_global_evaluation_registry.time.time", return_value=250), \
                 patch.object(forward, "outcome", side_effect=AssertionError("opened")):
                with self.assertRaises(EvaluationRejected):
                    forward.evaluate_once()

    def test_full_outcome_path_with_synthetic_received_books_and_settlements(self):
        p = load_profile(); start = 43200; end = start+86400
        venue = json.loads((ROOT/"competition_live/strategy_plan.json").read_text())
        venue.update(paper_start_unix=start, capture_end_unix=end,
                     end_liquidation_seconds=7200, step_seconds=3600, trading_protocol=p["trading"])
        records = []
        for now in range(start, end+1, 3600):
            # Publish a finished source at the boundary; known synthetic inputs.
            frame = dict(source_time=now, published_time=now, weights=[0.]*10, retention=[1.]*10,
                sigma=(np.eye(10)*1e-6).tolist(), volume=[1e6]*10, trades=[1000]*10, source_hash="x"*64)
            data = dict(frame=frame, incumbent=[0.]*10, candidate=[0.]*10,
                        candidate_weights=[0.,0.,-.02]+[0.]*7, shared_forecasts=[],
                        publication_sha256=hashlib.sha256(json.dumps(frame,sort_keys=True).encode()).hexdigest())
            records.append(dict(received_ms=now*1000, channel="strategySignal", data=data))
            for sec in range(now-2, now+11):
                for symbol in p["trading"]["symbols"]:
                    records.extend([
                        dict(received_ms=sec*1000, channel="l2Book", data=dict(coin="pas:"+symbol,
                             time=sec*1000, levels=[[dict(px="99.99",sz="1000000")],
                                                   [dict(px="100.01",sz="1000000")]])),
                        dict(received_ms=sec*1000, channel="activeAssetCtx", data=dict(coin="pas:"+symbol,
                             ctx=dict(markPx="100",oraclePx="100",funding="0")))])
        records.sort(key=lambda e:e["received_ms"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"raw").mkdir()
            with gzip.open(root/"raw/fixture.jsonl.gz","wt") as f:
                for row in records:
                    f.write(json.dumps(row)+"\n")
            plan = dict(test_start=start, test_end=end, venue_plan=venue,
                        artifacts={str(ROOT/"competition_live/config.json"):sha(ROOT/"competition_live/config.json")})
            def funding(symbol, lo, hi):
                return [dict(coin="pas:"+symbol,time=t*1000,fundingRate="0")
                        for t in range(lo+3600,hi+1,3600)]
            with patch.object(forward,"FWD",root), patch.object(forward,"public_funding",side_effect=funding):
                result = forward.outcome(plan)
            self.assertEqual(len(result["rows"]),4)
            candidate = [r for r in result["rows"] if r["variant"]=="fixed_fit_candidate"]
            self.assertTrue(all(r["valid"] for r in candidate), candidate)
            self.assertTrue(all(r["fills"]>=2 and r["fees"]>0 for r in candidate))
            self.assertTrue(all(len(r["daily"])==2 for r in candidate))
            self.assertFalse(result["passed"])  # One completed return cannot establish Q.


if __name__ == "__main__":
    unittest.main()
