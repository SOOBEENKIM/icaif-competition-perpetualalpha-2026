"""Conditional PAS evaluation after model/selection freeze; no fitting."""
import gzip
import json
from datetime import datetime
import numpy as np
from competition_live.profile import ROOT,load_profile
from pas_strategy_audit.data import load
from strategy_recheck.pas import simulate
from .protocol import OUT as FIRST,PRIOR
from .wave3 import OUT as THIRD
from .wave5 import OUT,ARMS
from .prepare import native
from .candidate import build_weights


def main():
    assert (OUT/"nomination.json").exists()
    spec=json.loads((FIRST/"protocol.json").read_text())
    history=dict(np.load(ROOT/spec["pointer"]["inputs"],allow_pickle=False))
    saved=dict(np.load(OUT/"targets.npz",allow_pickle=False))
    calibration=dict(
        agreement_scale=json.loads((THIRD/"calibration.json").read_text())["agreement_scale"],
        final_scale=json.loads((OUT/"calibration.json").read_text())["scales"]["cost_qualified_reverse"])
    weights,trace=build_weights(history,saved["incumbent"],saved["sigma"],
                               load_profile()["trading"],spec,calibration)
    np.testing.assert_allclose(weights,saved["cost_qualified_reverse"],atol=1e-13,rtol=1e-11)
    # Verify full candidate causality, including all target post-processing.
    n=len(weights)-48
    short,_=build_weights({k:v[:n] for k,v in history.items()},saved["incumbent"][:n],
        saved["sigma"][:n],load_profile()["trading"],spec,calibration)
    np.testing.assert_allclose(weights[:n],short,atol=1e-13,rtol=1e-11)
    (OUT/"candidate_parity.json").write_text(json.dumps(dict(
        max_error=float(abs(weights-saved["cost_qualified_reverse"]).max()),
        full_candidate_prefix_invariance=True,calibration=calibration),indent=2))
    cfg,data=load(PRIOR/"pas")
    publications={}
    for path in (ROOT/"practice_live/updates").glob("*/signals.json"):
        r=json.loads(path.read_text())
        t=int(datetime.fromisoformat(r["input_completed_bar_utc"]).timestamp())
        at=int(datetime.fromisoformat(r["created_at_utc"]).timestamp()*1000)
        if cfg["start_ms"]<=t*1000<cfg["cutoff_ms"]:
            publications[t]=min(at,publications.get(t,at))
    scenarios=dict(base=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
                   stress=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5))
    rows=[];(OUT/"pas").mkdir(exist_ok=True)
    for name in ARMS:
        events=[]
        for i,t in enumerate(saved["times"]):
            t=int(t)
            if cfg["start_ms"]<=t*1000<cfg["cutoff_ms"]:
                assert t in publications
                events.append(dict(time=publications[t],source_time=t,weights=saved[name][i].tolist(),
                    sigma=saved["sigma"][i].tolist(),volume=history["volume"][i].tolist(),
                    trades=history["trades"][i].tolist()))
        for scenario,assumptions in scenarios.items():
            result=simulate(cfg,data,events,load_profile()["trading"],assumptions)
            row=dict(variant=name,scenario=scenario,**result["metrics"])
            assert row["maximum_accounting_error"]<1e-6 and row["timing_pass"]
            with gzip.open(OUT/"pas"/f"{name}_{scenario}.json.gz","wt") as f:
                json.dump(result,f,default=native)
            rows.append(row);print(json.dumps(row),flush=True)
    (OUT/"pas_results.json").write_text(json.dumps(rows,indent=2))
    current=[dict(symbol=s,incumbent_weight=float(saved["incumbent"][-1,j]),
        candidate_weight=float(weights[-1,j]),quality=float(trace["quality"][-1,j]),
        predicted_24h_return=float(trace["forecast"][-1,j]),
        economic_auxiliary_weight=float(trace["economic"][-1,j]),
        momentum_weight=float(trace["momentum"][-1,j]))
        for j,s in enumerate(load_profile()["trading"]["symbols"])]
    (OUT/"latest_target_trace.json").write_text(json.dumps(dict(
        source_time=int(saved["times"][-1]),assets=current,orders_sent=0,
        note="Source locked to historical Oct20 22:00 KST input. Not current exchange positions."),indent=2))


if __name__=="__main__":
    main()
