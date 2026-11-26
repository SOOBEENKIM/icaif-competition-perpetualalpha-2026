"""Confirmed opposite-trend lifecycle; registered before reading later audits."""
import copy
import gzip
import json
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_deployment
from competition_live.scoring import evaluate
from replay_lab.common_comparison.engine import project
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .protocol import OUT as FIRST,sha
from .wave3 import OUT as THIRD
from .model import features
from .prepare import native

OUT=ROOT/"reports/all_market_alpha_wave4_20261020"
ARMS=["incumbent","confirmed_exit","confirmed_reverse"]


def freeze():
    OUT.mkdir(exist_ok=True);path=OUT/"protocol.json"
    if path.exists():return json.loads(path.read_text())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        rationale="Wave3 can attenuate a wrong-way incumbent holding but cannot reverse it. "
                  "Introduce an explicit same-rule-for-all-assets lifecycle with persistent opposite-trend evidence.",
        reversal="Use wave3 agreement_all_matched as the starting target. At each six-hour decision, if the "
                 "last 24 hourly quality values all have incumbent-opposite sign and absolute quality >=0.5, "
                 "exit or reverse (separate arms). In reverse arm use the already-fixed multiscale target's direction "
                 "and the smaller of incumbent and multiscale absolute weights. Exchange executor must close before opening opposite.",
        normalization="Additional scalar <=1 calculated only July1–July29 to match incumbent mean predicted base risk. "
                      "No post-attenuation leverage restoration.",
        selection="Only confirmed_reverse is eligible. August PnL AND Q exceed incumbent in BOTH base and stress, "
                  "with zero timing/accounting/risk violations. Freeze before reading any later results.",
        historical_independence=False,
        prior_trials="All three previous batches retained. No untouched-test or multiple-testing-adjusted significance claim.",
        pair_scope="Retained pair model still two fixed pairs. Directional opportunity and confirmed reversal cover all ten.",
        live_deployment_changed=False,orders_sent=0,deployment_hashes=verify_deployment()["files"])
    path.write_text(json.dumps(result,indent=2));return result


def reverse_targets(base,combined,auxiliary,quality,times,mode):
    out=np.zeros_like(base);held=combined[0].copy()
    for i,t in enumerate(times):
        if t%21600==0:
            held=combined[i].copy()
            if i>=23:
                sign=np.sign(base[i])
                contrary=np.all(sign*quality[i-23:i+1]<=-.5,axis=0)&(sign!=0)
                replacement=np.zeros(base.shape[1]) if mode=="exit" else (
                    np.sign(auxiliary[i])*np.minimum(abs(base[i]),abs(auxiliary[i])))
                # A flat or incoherent auxiliary cannot create an opposite order.
                replacement=np.where(sign*replacement<0,replacement,0)
                held=np.where(contrary,replacement,held)
        out[i]=held
    return out


def prepare():
    freeze()
    first=dict(np.load(FIRST/"targets.npz",allow_pickle=False))
    third=dict(np.load(THIRD/"targets.npz",allow_pickle=False))
    ptr=json.loads((FIRST/"protocol.json").read_text())["pointer"]
    data=dict(np.load(ROOT/ptr["inputs"],allow_pickle=False))
    quality=np.nan_to_num(np.tanh(features(data)["x"][:,:,1:4].mean(-1)/2),nan=0)
    base=first["incumbent"];t=first["times"];sigma=first["sigma"]
    result=dict(times=t,sigma=sigma,incumbent=base)
    start=int(datetime(2026,7,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,7,29,tzinfo=timezone.utc).timestamp())
    calibration=(t>=start)&(t<end)
    def risk(w):
        return np.sqrt(np.maximum(0,24*np.einsum("ni,nij,nj->n",
            w[calibration],sigma[calibration],w[calibration]))).mean()
    p=load_profile()["trading"];scales={}
    for mode in ("exit","reverse"):
        name="confirmed_"+mode
        a=reverse_targets(base,third["agreement_all_matched"],first["multiscale"],quality,t,mode)
        scale=min(1.,risk(base)/max(risk(a),1e-12));scales[name]=scale
        a*=scale
        for i in range(168,len(t)):a[i]=project(a[i],p["initial"],sigma[i],p["risk"])
        a[:168]=0;result[name]=a
        stop=len(t)-48
        prefix=reverse_targets(base[:stop],third["agreement_all_matched"][:stop],
            first["multiscale"][:stop],quality[:stop],t[:stop],mode)*scale
        for i in range(168,len(prefix)):prefix[i]=project(prefix[i],p["initial"],sigma[i],p["risk"])
        prefix[:168]=0
        np.testing.assert_allclose(a[:stop],prefix,atol=1e-12)
    np.savez_compressed(OUT/"targets.npz",**result)
    (OUT/"calibration.json").write_text(json.dumps(dict(start=start,end=end,scales=scales,
        prefix_invariance=True,threshold_selected_on_PAS=False),indent=2))
    (OUT/"source_freeze.json").write_text(json.dumps({
        str(ROOT/"all_market_alpha/wave4.py"):sha(ROOT/"all_market_alpha/wave4.py"),
        str(ROOT/"all_market_alpha/model.py"):sha(ROOT/"all_market_alpha/model.py"),
        str(FIRST/"targets.npz"):sha(FIRST/"targets.npz"),
        str(THIRD/"targets.npz"):sha(THIRD/"targets.npz"),
        str(OUT/"targets.npz"):sha(OUT/"targets.npz"),
    },indent=2))
    print("wave4 prepared",scales,flush=True)


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
        name="confirmed_reverse"
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
