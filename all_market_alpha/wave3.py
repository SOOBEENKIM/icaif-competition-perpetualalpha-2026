"""Registered signal-agreement allocation; includes pre-evaluation risk matching."""
import copy
import gzip
import json
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_deployment
from competition_live.scoring import evaluate
from replay_lab.common_comparison.engine import project
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .protocol import OUT as FIRST,PRIOR,sha
from .wave2 import vacant_overlay
from .model import features
from .prepare import native

OUT=ROOT/"reports/all_market_alpha_wave3_20261020"
ARMS=["incumbent","agreement","agreement_all","agreement_matched","agreement_all_matched"]


def freeze():
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if path.exists():return json.loads(path.read_text())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        rationale="The incumbent base creates positive returns, but its slow states can disagree with recent multi-horizon momentum. "
                  "Vary conviction symmetrically, without choosing asset names or fitting thresholds to evaluation PnL.",
        formula="At completed UTC day: multiplier=1+0.5*sign(incumbent weight)*tanh(mean(z_72,z_168,z_336)/2). "
                "Bound 0.5..1.5. Apply multiplier after incumbent weights; do not normalize after attenuation. "
                "Daily multiplier held until next day, then existing downward account caps. This is a new allocation policy.",
        all_market="agreement_all additionally uses the predeclared wave2 vacant_confirmed 25%-size signal "
                   "where incumbent is flat; all ten assets eligible. Core pair overlay remains exactly the incumbent two pairs.",
        risk_matching="For matched arms use one scalar <=1, computed ONLY from July1–July29 mean predicted base daily volatility, "
                      "to match incumbent risk. Do not estimate this scalar on August/September/PAS.",
        selection="Only agreement_all_matched is eligible: August PnL AND Q exceed incumbent in base AND stress, "
                  "with zero timing/accounting/risk violations. All other arms are attribution controls. Freeze before later audits.",
        prior_trials="All wave1/wave2 failures retained; historical results reused, not an independent holdout.",
        historical_independence=False,orders_sent=0,live_deployment_changed=False,
        deployment_hashes=verify_deployment()["files"])
    path.write_text(json.dumps(result,indent=2));return result


def allocate(base,quality,times):
    out=np.zeros_like(base);scale=np.ones(base.shape[1])
    for i,t in enumerate(times):
        if t%86400==0:
            scale=1+.5*np.sign(base[i])*quality[i]
        out[i]=base[i]*scale
    return out


def prepare():
    freeze()
    first=dict(np.load(FIRST/"targets.npz",allow_pickle=False))
    ptr=json.loads((FIRST/"protocol.json").read_text())["pointer"]
    data=dict(np.load(ROOT/ptr["inputs"],allow_pickle=False))
    f=features(data);quality=np.nan_to_num(np.tanh(f["x"][:,:,1:4].mean(-1)/2),nan=0)
    t=first["times"];base=first["incumbent"]
    a=allocate(base,quality,t);b=vacant_overlay(a,first["multiscale"],t,True)
    start=int(datetime(2026,7,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,7,29,tzinfo=timezone.utc).timestamp())
    calibration=(t>=start)&(t<end)
    sigma=first["sigma"]
    def risk(w):
        return np.sqrt(np.maximum(0,24*np.einsum("ni,nij,nj->n",w[calibration],sigma[calibration],w[calibration]))).mean()
    reference=risk(base);sa=min(1.,reference/max(risk(a),1e-12));sb=min(1.,reference/max(risk(b),1e-12))
    result=dict(times=t,sigma=sigma,incumbent=base,agreement=a,
                agreement_all=b,agreement_matched=a*sa,agreement_all_matched=b*sb)
    p=load_profile()["trading"]
    for name in ARMS[1:]:
        for i in range(168,len(t)):
            result[name][i]=project(result[name][i],p["initial"],sigma[i],p["risk"])
        result[name][:168]=0
    # The calibration cutoff is strictly before every evaluation.
    assert end<int(datetime(2026,8,1,tzinfo=timezone.utc).timestamp())
    np.savez_compressed(OUT/"targets.npz",**result)
    (OUT/"calibration.json").write_text(json.dumps(dict(start=start,end=end,
        incumbent_mean_predicted_daily_risk=reference,agreement_scale=sa,agreement_all_scale=sb,
        agreement_risk_before=risk(a),agreement_all_risk_before=risk(b),
        coefficients_fit_to_evaluation_returns=False),indent=2))
    (OUT/"source_freeze.json").write_text(json.dumps({
        str(ROOT/"all_market_alpha/wave3.py"):sha(ROOT/"all_market_alpha/wave3.py"),
        str(ROOT/"all_market_alpha/wave2.py"):sha(ROOT/"all_market_alpha/wave2.py"),
        str(ROOT/"all_market_alpha/model.py"):sha(ROOT/"all_market_alpha/model.py"),
        str(FIRST/"targets.npz"):sha(FIRST/"targets.npz"),
        str(OUT/"targets.npz"):sha(OUT/"targets.npz"),
    },indent=2))
    print("wave3 prepared",sa,sb,flush=True)


def run(phases):
    freeze();p=load_profile()
    for path,digest in json.loads((OUT/"source_freeze.json").read_text()).items():
        assert sha(path)==digest,path
    original=dict(np.load(BUNDLE/"data/reference.npz",allow_pickle=False))
    state=dict(np.load(BUNDLE/"models/reference_gate_state.npz",allow_pickle=False))
    target=dict(np.load(OUT/"targets.npz",allow_pickle=False))
    ix=np.searchsorted(target["times"],original["times"])
    np.testing.assert_array_equal(target["times"][ix],original["times"])
    windows=[r for r in json.loads((BUNDLE/"expected.json").read_text())["runs"]
             if r["venue"]=="reference" and r["arm"]==p["candidate"] and r["phase"] in phases]
    path=OUT/"reference_results.json";rows=json.loads(path.read_text()) if path.exists() else []
    (OUT/"reference").mkdir(exist_ok=True)
    for w in windows:
        for name in ARMS:
            if any(r["variant"]==name and r["phase"]==w["phase"] and r["scenario"]==w["scenario"] for r in rows):continue
            d=dict(original);d["first"]=target[name][ix]
            cfg=copy.deepcopy(p["configs"]["reference"]);cfg["conditional_completion"]={"risk_aversion":0.}
            result=replay(d,state,None,cfg,p["trading"],p["candidate"],w["scenario"],w["start"],w["end"])
            m=result["metrics"]
            daily=[dict(time=s["time"],equity=s["equity"]+50000) for s in result["snapshots"] if s["time"]%86400==43200]
            sc=evaluate(daily,trading_days=len({f["time"]//86400 for f in result["fills"]}))
            curve=np.array([s["equity"]+50000 for s in result["snapshots"]])
            row=dict(variant=name,phase=w["phase"],scenario=w["scenario"],start=w["start"],end=w["end"],
                net_pnl=m["net_pnl"],terminal_adjusted_pnl=m["pnl_after_liquidation_reserve"],
                Q=sc["Q"],volatility=sc["volatility"],score=sc["score"],return_fraction=sc["return_fraction"],
                max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
                mean_notional=float(np.mean([s["gross"]*s["equity"] for s in result["snapshots"]])),
                fees=m["fees"],execution_cost=m["execution_cost"],funding=m["funding"],
                base_pnl=m["base_owned_pnl"],pair_pnl=m["pair_owned_pnl"],
                fills=m["fills"],completed_pairs=m["completed_pairs"],
                traded_symbols=sorted({f["symbol"] for f in result["fills"]}),
                base_traded_symbols=sorted({f["symbol"] for f in result["owner_fills"]["base"]}),
                short_entries=sum(f["signed_qty"]<0 and not f["reduce_only"] for f in result["fills"]),
                risk_breach_bars=m["risk_breach_bars"],accounting_error=m["maximum_accounting_error"],
                timing_pass=all(f["source_available"]<=f["decision_time"]<f["time"] for f in result["fills"]))
            assert row["accounting_error"]<1e-6 and row["timing_pass"]
            with gzip.open(OUT/"reference"/f"{name}_{w['phase']}_{w['scenario']}.json.gz","wt") as file:
                json.dump(result,file,default=native)
            rows.append(row);path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    if phases==["validation"]:
        idx={(r["variant"],r["scenario"]):r for r in rows}
        name="agreement_all_matched"
        passed=all(idx[name,s]["net_pnl"]>idx["incumbent",s]["net_pnl"] and
                   idx[name,s]["Q"]>idx["incumbent",s]["Q"] and idx[name,s]["risk_breach_bars"]==0
                   for s in ("base","stress"))
        selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee=name if passed else None,
                       eligible=[name] if passed else [],historical_independence=False,result_sha256=sha(path))
        if (OUT/"nomination.json").exists():raise FileExistsError("Frozen nomination exists")
        (OUT/"nomination.json").write_text(json.dumps(selection,indent=2))
        print("NOMINATION",json.dumps(selection),flush=True)


if __name__=="__main__":
    import sys
    mode=sys.argv[1]
    if mode=="prepare":prepare()
    elif mode=="development":run(["validation"])
    elif mode=="audit":
        assert (OUT/"nomination.json").exists()
        run(["comparison","continuous"])
    else:raise ValueError(mode)
