"""One predeclared historical diagnostic, with a separate prospective holdout."""
import argparse
import copy
import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from competition_live.profile import ROOT, BUNDLE, load_profile, verify_deployment
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from all_market_alpha.model import labels
from .model import train, predict, build

OUT = ROOT / "reports/validation_rebuild_20261021"
FIRST = ROOT / "reports/all_market_alpha_20261020"
LAST = ROOT / "reports/all_market_alpha_wave5_20261020"
ARMS = ["incumbent", "previous_rolling_candidate", "fixed_fit_candidate"]


def native(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, default=native, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def audit():
    """Metadata inventory: never open price arrays from a reserved outcome tape."""
    old = json.loads((ROOT / "replay_lab/output/evaluation_control/exposure_audit.json").read_text())
    exposures = copy.deepcopy(old["exposures"])
    evidence = [
        ("experiments/older_reference_data_20261017/event_audit/summary.json",
         "Older Binance history used to construct and inspect pair training labels"),
        ("experiments/causal_repair_20261017/artifacts/data_receipts.json",
         "External history used by prior causal-repair experiment"),
        ("reports/all_market_alpha_wave5_20261020/protocol.json",
         "All August/September/October audits explicitly seen before fifth candidate batch"),
    ]
    inventory = []
    for rel, reason in evidence:
        p = ROOT / rel
        obj = json.loads(p.read_text())
        if "pairs" in obj:
            intervals = [(r["first_available"], r["last_available"]) for r in obj["pairs"]]
        elif "datasets" in obj:
            intervals = [(r["start"], r["end"]) for r in obj["datasets"].values()]
        else:
            intervals = [(ts("2026-08-01T00:00:00Z"), ts("2026-10-20T13:00:00Z"))]
        for start, end in intervals:
            exposures.append(dict(start=start, end=end, reason=reason, evidence=[rel]))
        inventory.append(dict(path=rel, sha256=sha(p), intervals=intervals))
    # Production metadata is exposure too; downloading elsewhere cannot reset it.
    pointers = sorted((ROOT / "practice_live/updates").glob("*/manifest.json"))
    last = json.loads((ROOT / "practice_live/latest.json").read_text())
    cutoff = ts(last["completed_bar_utc"])
    exposures.append(dict(start=ts("2026-03-18T14:00:00Z"), end=cutoff,
                          reason="Conservative full retained reference-panel exposure",
                          evidence=[last["inputs"], "practice_live/latest.json"]))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  metadata_inventory_complete=True,
                  scope="Known retained project data; not a claim about every possible external archive",
                  exposures=exposures, evidence_inventory=inventory,
                  production_manifests_count=len(pointers), latest_known_data_exposure=cutoff,
                  existing_complete_ten_asset_untouched_past_test_found=False,
                  historical_independence=False)
    write(OUT / "exposure_audit.json", result)
    return result


def prepare():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Study already frozen; do not replace it")
    OUT.mkdir(exist_ok=True)
    audit()
    source = json.loads((FIRST / "protocol.json").read_text())
    calibration = json.loads((LAST / "candidate_parity.json").read_text())["calibration"]
    cutoff = ts("2026-08-01T00:00:00Z")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS,
        train_start=cutoff-90*86400, train_end=cutoff,
        validation_start=ts("2026-08-01T12:00:00Z"),
        validation_end=ts("2026-09-01T12:00:00Z"),
        retrospective_start=ts("2026-09-01T12:00:00Z"),
        retrospective_end=ts("2026-10-15T12:00:00Z"),
        classification="Previously exposed chronological diagnostic, NOT an independent final test",
        historical_independence=False, candidate="fixed_fit_candidate",
        selection="Candidate named before results; no parameter grid, no re-selection after any diagnostic",
        model=source["model"], policy=source["policy"], calibration=calibration,
        refit_contract={
            "new_auxiliary_ridge": "Fit once using labels strictly before Aug1; no validation/test refits",
            "incumbent_core": "Existing pretrained/frozen artifacts unchanged",
            "shared_pair_layer": "Original causal rolling pair/gate rule unchanged in every arm; this is NOT an all-components-static experiment",
            "candidate_structure": "Inherited wave5 choices already depend on exposed historical results; independent claim prohibited"},
        common_capital=dict(team=100000., strategy=50000., reserve=50000.),
        source_pointer=source["pointer"], input_sha256=sha(ROOT/source["pointer"]["inputs"]),
        deployment_hashes=verify_deployment()["files"],
        primary_forward_candidate="fixed_fit_candidate",
        success_rule="Positive net PnL and higher PnL and Q than incumbent in both base/stress; "
                     "valid timing/accounting/risk and complete data. Report failure and no re-selection.",
        live_orders=0, live_changed=False)
    write(OUT / "protocol.json", protocol)  # before fitting and evaluation
    data = dict(np.load(ROOT/source["pointer"]["inputs"], allow_pickle=False))
    model = train(data, protocol["model"], protocol["train_start"], cutoff)
    write(OUT / "model.json", model)
    old = dict(np.load(FIRST/"targets.npz", allow_pickle=False))
    previous = dict(np.load(LAST/"targets.npz", allow_pickle=False))
    weights, trace = build(data, old["incumbent"], old["sigma"],
                          load_profile()["trading"], protocol, calibration, model)
    np.savez_compressed(OUT/"targets.npz", times=data["market_times"], sigma=old["sigma"],
        incumbent=old["incumbent"], previous_rolling_candidate=previous["cost_qualified_reverse"],
        fixed_fit_candidate=weights, forecast=trace["forecast"])
    # Delete the future AND poison it: fitting and earlier predictions must not move.
    changed = {k: v.copy() for k, v in data.items()}
    after = changed["market_times"] >= cutoff
    changed["close"][after] *= 7
    changed["funding"][after] += 1
    other = train(changed, protocol["model"], protocol["train_start"], cutoff)
    for key in ("beta", "center", "scale"):
        np.testing.assert_array_equal(model[key], other[key])
    n = len(data["market_times"])-48
    prefix = {k: v[:n] for k, v in data.items()}
    pweights, _ = build(prefix, old["incumbent"][:n], old["sigma"][:n],
                        load_profile()["trading"], protocol, calibration, model)
    np.testing.assert_allclose(weights[:n], pweights, atol=1e-12)
    write(OUT/"causality.json", dict(fits=1, rows=model["rows"],
        last_label_maturity=model["last_maturity"], cutoff=cutoff,
        all_labels_before_cutoff=model["last_maturity"] < cutoff,
        future_poison_cannot_change_fit=True, future_deletion_target_prefix_invariant=True,
        auxiliary_prediction_calls_fit=False, historical_independence=False))
    paths = [*Path(__file__).parent.glob("*.py"), OUT/"protocol.json", OUT/"model.json",
             OUT/"targets.npz", ROOT/source["pointer"]["inputs"]]
    write(OUT/"source_hashes.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    print(json.dumps(dict(stage="prepared", rows=model["rows"],
                          last_label_maturity=model["last_maturity"], refits=0)), flush=True)


def verify():
    for rel, digest in json.loads((OUT/"source_hashes.json").read_text()).items():
        if sha(ROOT/rel) != digest:
            raise ValueError("Frozen historical study changed: "+rel)
    verify_deployment()


def summarize(r, name, phase, scenario):
    m = r["metrics"]
    daily = [dict(time=s["time"], equity=s["equity"]+50000)
             for s in r["snapshots"] if s["time"]%86400 == 43200]
    score = evaluate(daily, trading_days=len({f["time"]//86400 for f in r["fills"]}))
    curve = np.array([s["equity"]+50000 for s in r["snapshots"]])
    row = dict(variant=name, phase=phase, scenario=scenario,
        net_pnl=m["net_pnl"], terminal_adjusted_pnl=m["pnl_after_liquidation_reserve"],
        **score, fees=m["fees"], funding=m["funding"], execution_cost=m["execution_cost"],
        fills=m["fills"], completed_pairs=m["completed_pairs"],
        base_pnl=m["base_owned_pnl"], pair_pnl=m["pair_owned_pnl"],
        max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
        symbols=sorted({f["symbol"] for f in r["fills"]}),
        short_entries=sum(f["signed_qty"]<0 and not f["reduce_only"] for f in r["fills"]),
        accounting_error=m["maximum_accounting_error"], risk_breach_bars=m["risk_breach_bars"],
        timing_pass=all(f["source_available"] <= f["decision_time"] < f["time"] for f in r["fills"]))
    if not row["timing_pass"] or row["accounting_error"] >= 1e-6:
        raise ValueError("Invalid replay accounting/timing")
    return row


def historical():
    verify()
    if (OUT/"reference_results.json").exists():
        raise FileExistsError("Historical evaluation already recorded")
    p = load_profile()
    d0 = dict(np.load(BUNDLE/"data/reference.npz", allow_pickle=False))
    de = dict(np.load(BUNDLE/"models/reference_gate_state.npz", allow_pickle=False))
    w = dict(np.load(OUT/"targets.npz", allow_pickle=False))
    ix = np.searchsorted(w["times"], d0["times"])
    np.testing.assert_array_equal(w["times"][ix], d0["times"])
    expected = json.loads((BUNDLE/"expected.json").read_text())
    rows = []
    for phase in ("validation", "comparison", "continuous"):
        windows = [r for r in expected["runs"] if r["venue"]=="reference"
                   and r["arm"]==p["candidate"] and r["phase"]==phase]
        for window in windows:
            for name in ARMS:
                d = dict(d0); d["first"] = w[name][ix]
                cfg = copy.deepcopy(p["configs"]["reference"])
                cfg["conditional_completion"] = {"risk_aversion": 0.}
                r = replay(d, de, None, cfg, p["trading"], p["candidate"],
                           window["scenario"], window["start"], window["end"])
                row = summarize(r, name, phase, window["scenario"])
                dest = OUT/"reference"/f"{name}_{phase}_{window['scenario']}.json.gz"
                dest.parent.mkdir(exist_ok=True)
                with gzip.open(dest, "wt") as f:
                    json.dump(r, f, default=native)
                rows.append(row)
                print(json.dumps(row), flush=True)
    write(OUT/"reference_results.json", rows)
    protocol = json.loads((OUT/"protocol.json").read_text())
    data = dict(np.load(ROOT/protocol["source_pointer"]["inputs"], allow_pickle=False))
    model = json.loads((OUT/"model.json").read_text())
    f, forecast = predict(data, model)
    y, maturity, valid = labels(data, f, 24, 1)
    diagnostics = []
    for name, start, end in [
        ("validation", protocol["validation_start"], protocol["validation_end"]),
        ("reused_later_period", protocol["retrospective_start"], protocol["retrospective_end"])]:
        mask = ((data["market_times"] >= start) & (maturity < end)
                & (data["market_times"]%86400 == 0))
        ok = valid & mask[:, None] & np.isfinite(forecast) & np.isfinite(y)
        actual = y*f["unit"]
        diagnostics.append(dict(period=name, observations=int(ok.sum()),
            mae_bps=float(np.mean(abs(forecast[ok]-actual[ok]))*10000),
            zero_forecast_mae_bps=float(np.mean(abs(actual[ok]))*10000),
            direction_accuracy=float(np.mean(np.sign(forecast[ok])==np.sign(actual[ok])))))
    write(OUT/"forecast_diagnostics.json", diagnostics)
    verify()


def pas():
    verify()
    from pas_strategy_audit.data import load
    from strategy_recheck.pas import simulate
    p = load_profile(); w = dict(np.load(OUT/"targets.npz", allow_pickle=False))
    spec = json.loads((OUT/"protocol.json").read_text())
    history = dict(np.load(ROOT/spec["source_pointer"]["inputs"], allow_pickle=False))
    cfg, data = load(ROOT/"reports/isolated_repairs_20261020/pas")
    publications = {}
    for path in (ROOT/"practice_live/updates").glob("*/signals.json"):
        r = json.loads(path.read_text())
        t = ts(r["input_completed_bar_utc"])
        at = int(datetime.fromisoformat(r["created_at_utc"]).timestamp()*1000)
        if cfg["start_ms"] <= t*1000 < cfg["cutoff_ms"]:
            publications[t] = min(at, publications.get(t, at))
    rows = []
    for name in ARMS:
        events = [dict(time=publications[int(t)], source_time=int(t),
                       weights=w[name][i].tolist(), sigma=w["sigma"][i].tolist(),
                       volume=history["volume"][i].tolist(), trades=history["trades"][i].tolist())
                  for i,t in enumerate(w["times"])
                  if cfg["start_ms"] <= t*1000 < cfg["cutoff_ms"]]
        for scenario, assumptions in {
            "base": dict(fee_bps=9., impact_bps=10., participation=.01, receipt_multiplier=1.),
            "stress": dict(fee_bps=12., impact_bps=20., participation=.0025, receipt_multiplier=.5)}.items():
            r = simulate(cfg, data, events, p["trading"], assumptions)
            rows.append(dict(variant=name, scenario=scenario, **r["metrics"]))
            dest = OUT/"pas"/f"{name}_{scenario}.json.gz"; dest.parent.mkdir(exist_ok=True)
            with gzip.open(dest, "wt") as f:
                json.dump(r, f, default=native)
    write(OUT/"pas_results.json", rows)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "historical", "pas"])
    args = parser.parse_args()
    {"prepare": prepare, "historical": historical, "pas": pas}[args.stage]()
