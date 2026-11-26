import argparse
import copy
import gzip
import json
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .prepare import native
from .protocol import OUT,freeze,sha


def nominate(rows):
    eligible=[]
    ix={(r["variant"],r["scenario"]):r for r in rows if r["phase"]=="validation"}
    for name in freeze()["arms"][1:]:
        improvements=[]
        for sc in ("base","stress"):
            a,b=ix["incumbent",sc],ix[name,sc]
            if (b["net_pnl"]>a["net_pnl"] and b["Q"]>a["Q"] and b["risk_breach_bars"]==0
                    and b["timing_pass"] and b["accounting_error"]<1e-6):
                improvements.append(b["net_pnl"]-a["net_pnl"])
        if len(improvements)==2:
            eligible.append(dict(variant=name,worst_pnl_advantage=min(improvements)))
    eligible.sort(key=lambda x:(-x["worst_pnl_advantage"],x["variant"]))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                nominee=eligible[0]["variant"] if eligible else None,
                eligible=eligible,selection_period="August only; reused history",
                historical_independence=False)


def run(phases):
    spec=freeze();profile=load_profile()
    hashes=json.loads((OUT/"source_freeze.json").read_text())
    for path,digest in hashes.items():
        assert sha(ROOT/path)==digest,path
    d0=dict(np.load(BUNDLE/"data/reference.npz",allow_pickle=False))
    de=dict(np.load(BUNDLE/"models/reference_gate_state.npz",allow_pickle=False))
    w=dict(np.load(OUT/"targets.npz",allow_pickle=False))
    ix=np.searchsorted(w["times"],d0["times"])
    np.testing.assert_array_equal(w["times"][ix],d0["times"])
    np.testing.assert_allclose(w["incumbent"][ix],d0["first"],atol=1e-13,rtol=1e-11)
    expected=json.loads((BUNDLE/"expected.json").read_text())
    windows=[r for r in expected["runs"] if r["venue"]=="reference"
             and r["arm"]==profile["candidate"] and r["phase"] in phases]
    dest=OUT/"reference";dest.mkdir(exist_ok=True)
    path=OUT/"reference_results.json"
    rows=json.loads(path.read_text()) if path.exists() else []
    for window in windows:
        for name in spec["arms"]:
            key=(name,window["phase"],window["scenario"])
            if any((r["variant"],r["phase"],r["scenario"])==key for r in rows):
                continue
            d=dict(d0);d["first"]=w[name][ix]
            cfg=copy.deepcopy(profile["configs"]["reference"])
            cfg["conditional_completion"]={"risk_aversion":0.}
            r=replay(d,de,None,cfg,profile["trading"],profile["candidate"],
                     window["scenario"],window["start"],window["end"])
            m=r["metrics"]
            daily=[dict(time=s["time"],equity=s["equity"]+50000) for s in r["snapshots"]
                   if s["time"]%86400==43200]
            score=evaluate(daily,trading_days=len({f["time"]//86400 for f in r["fills"]}))
            curve=np.array([s["equity"]+50000 for s in r["snapshots"]])
            row=dict(variant=name,phase=window["phase"],scenario=window["scenario"],
                start=window["start"],end=window["end"],net_pnl=m["net_pnl"],
                terminal_adjusted_pnl=m["pnl_after_liquidation_reserve"],
                Q=score["Q"],volatility=score["volatility"],score=score["score"],
                return_fraction=score["return_fraction"],
                max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
                mean_notional=float(np.mean([s["gross"]*s["equity"] for s in r["snapshots"]])),
                fees=m["fees"],execution_cost=m["execution_cost"],funding=m["funding"],
                base_pnl=m["base_owned_pnl"],pair_pnl=m["pair_owned_pnl"],
                fills=m["fills"],completed_pairs=m["completed_pairs"],
                traded_symbols=sorted({f["symbol"] for f in r["fills"]}),
                base_traded_symbols=sorted({f["symbol"] for f in r["owner_fills"]["base"]}),
                short_entries=sum(f["signed_qty"]<0 and not f["reduce_only"] for f in r["fills"]),
                risk_breach_bars=m["risk_breach_bars"],accounting_error=m["maximum_accounting_error"],
                timing_pass=all(f["source_available"]<=f["decision_time"]<f["time"] for f in r["fills"]))
            assert row["accounting_error"]<1e-6 and row["timing_pass"]
            with gzip.open(dest/f"{name}_{window['phase']}_{window['scenario']}.json.gz","wt") as f:
                json.dump(r,f,default=native)
            rows.append(row);path.write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    if phases==["validation"]:
        selection=nominate(rows)
        selection["result_sha256"]=sha(path)
        target=OUT/"nomination.json"
        if target.exists():
            raise FileExistsError("Do not replace a frozen nomination")
        target.write_text(json.dumps(selection,indent=2))
        print("NOMINATION",json.dumps(selection),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("phase",choices=["development","audit"])
    mode=arg.parse_args().phase
    if mode=="audit" and not (OUT/"nomination.json").exists():
        raise RuntimeError("Freeze development nomination first")
    run(["validation"] if mode=="development" else ["comparison","continuous"])
