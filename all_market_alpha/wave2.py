"""Second registered development batch: preserve, then diversify the incumbent."""
import copy
import gzip
import json
from datetime import datetime,timezone
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_deployment
from competition_live.scoring import evaluate
from replay_lab.alpha_rebuild.common import limits
from replay_lab.common_comparison.engine import project
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .protocol import OUT as FIRST,PRIOR,sha
from .prepare import native

OUT=ROOT/"reports/all_market_alpha_wave2_20261020"
ARMS=["incumbent","vacant_daily","vacant_confirmed","reversion_expanded","combined","size_control"]


def freeze():
    OUT.mkdir(exist_ok=True)
    path=OUT/"protocol.json"
    if path.exists():return json.loads(path.read_text())
    prior=json.loads((FIRST/"reference_results.json").read_text())
    assert all(r["phase"]=="validation" for r in prior)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        previous_development=str(FIRST/"reference_results.json"),
        previous_development_sha256=sha(FIRST/"reference_results.json"),
        rationale="Wave1 replacement loses incumbent directional returns and incurs excessive new turnover. "
                  "Test additive opportunities only where the original baseline is flat, and daily instead of six-hour resizing.",
        policies={
            "vacant_daily":"Incumbent plus 0.25 of the frozen all-market multiscale target only in assets where incumbent target is zero; refresh auxiliary target at completed UTC day.",
            "vacant_confirmed":"Same, but auxiliary signal sign must have persisted for all preceding 24 completed hours.",
            "reversion_expanded":"Original long-trend/carry policy unchanged. Extend the SAME 7-day aligned mean-reversion policy from crypto to all assets.",
            "combined":"Expanded mean-reversion plus vacant daily auxiliary. No named-asset preference in new opportunity layer.",
            "size_control":"1.25 times incumbent, diagnostic only. Same account caps. Ineligible for selection.",
        },
        pairs="Exact incumbent two-pair overlay retained, not expanded.",
        selection="August only, both base/stress PnL and Q beat incumbent, zero account/timing/risk violations. "
                  "Highest minimum PnL advantage, then name. Size control ineligible. Freeze before all later audits.",
        anti_overfit="Prior results reused. No untouched-history claim. All candidates and failures retained; no PAS tuning.",
        live_deployment_changed=False,orders_sent=0,
        deployment_hashes=verify_deployment()["files"])
    path.write_text(json.dumps(result,indent=2))
    return result


def vacant_overlay(base, auxiliary, times, confirmed=False):
    out=np.zeros_like(base);held=np.zeros(base.shape[1])
    for i,t in enumerate(times):
        if t%86400==0:
            held=auxiliary[i]*.25
            if confirmed:
                if i<23:held[:]=0
                else:
                    stable=np.all(np.sign(auxiliary[i-23:i+1])==np.sign(auxiliary[i]),axis=0)
                    held=np.where(stable,held,0)
        # Inherited baseline has priority immediately, not only once a day.
        out[i]=base[i]+np.where(abs(base[i])<1e-12,held,0)
    return out


def prepare():
    spec=freeze();first=dict(np.load(FIRST/"targets.npz",allow_pickle=False))
    old=dict(np.load(PRIOR/"incumbent_trace.npz",allow_pickle=False))
    all_=dict(np.load(PRIOR/"risk_short_exit_all_trace.npz",allow_pickle=False))
    base=first["incumbent"];times=first["times"]
    expanded=limits(old["legacy_sized_trend"]+.5*all_["reversion_weight"])*old["final_scale"]
    result=dict(times=times,sigma=first["sigma"],incumbent=base,
        vacant_daily=vacant_overlay(base,first["multiscale"],times),
        vacant_confirmed=vacant_overlay(base,first["multiscale"],times,True),
        reversion_expanded=expanded,
        combined=vacant_overlay(expanded,first["multiscale"],times),
        size_control=base*1.25)
    p=load_profile()["trading"]
    for name in ARMS[1:]:
        for i in range(168,len(times)):
            result[name][i]=project(result[name][i],p["initial"],first["sigma"][i],p["risk"])
        result[name][:168]=0
        assert np.isfinite(result[name]).all()
    np.savez_compressed(OUT/"targets.npz",**result)
    (OUT/"source_freeze.json").write_text(json.dumps({
        str(ROOT/"all_market_alpha/wave2.py"):sha(ROOT/"all_market_alpha/wave2.py"),
        str(FIRST/"targets.npz"):sha(FIRST/"targets.npz"),
        str(PRIOR/"incumbent_trace.npz"):sha(PRIOR/"incumbent_trace.npz"),
        str(PRIOR/"risk_short_exit_all_trace.npz"):sha(PRIOR/"risk_short_exit_all_trace.npz"),
        str(OUT/"targets.npz"):sha(OUT/"targets.npz"),
    },indent=2))
    print("wave2 prepared",flush=True)


def run(phases):
    spec=freeze();profile=load_profile()
    for path,digest in json.loads((OUT/"source_freeze.json").read_text()).items():
        assert sha(path)==digest,path
    d0=dict(np.load(BUNDLE/"data/reference.npz",allow_pickle=False))
    state=dict(np.load(BUNDLE/"models/reference_gate_state.npz",allow_pickle=False))
    target=dict(np.load(OUT/"targets.npz",allow_pickle=False))
    ix=np.searchsorted(target["times"],d0["times"])
    np.testing.assert_array_equal(target["times"][ix],d0["times"])
    expected=json.loads((BUNDLE/"expected.json").read_text())
    windows=[r for r in expected["runs"] if r["venue"]=="reference" and
             r["arm"]==profile["candidate"] and r["phase"] in phases]
    path=OUT/"reference_results.json";rows=json.loads(path.read_text()) if path.exists() else []
    (OUT/"reference").mkdir(exist_ok=True)
    for window in windows:
        for name in ARMS:
            if any(r["variant"]==name and r["phase"]==window["phase"] and
                   r["scenario"]==window["scenario"] for r in rows):continue
            data=dict(d0);data["first"]=target[name][ix]
            cfg=copy.deepcopy(profile["configs"]["reference"])
            cfg["conditional_completion"]={"risk_aversion":0.}
            result=replay(data,state,None,cfg,profile["trading"],profile["candidate"],
                          window["scenario"],window["start"],window["end"])
            m=result["metrics"]
            daily=[dict(time=s["time"],equity=s["equity"]+50000)
                   for s in result["snapshots"] if s["time"]%86400==43200]
            score=evaluate(daily,trading_days=len({f["time"]//86400 for f in result["fills"]}))
            curve=np.array([s["equity"]+50000 for s in result["snapshots"]])
            row=dict(variant=name,phase=window["phase"],scenario=window["scenario"],
                start=window["start"],end=window["end"],net_pnl=m["net_pnl"],
                terminal_adjusted_pnl=m["pnl_after_liquidation_reserve"],
                Q=score["Q"],volatility=score["volatility"],score=score["score"],
                return_fraction=score["return_fraction"],
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
            with gzip.open(OUT/"reference"/f"{name}_{window['phase']}_{window['scenario']}.json.gz","wt") as f:
                json.dump(result,f,default=native)
            rows.append(row);path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    if phases==["validation"]:
        index={(r["variant"],r["scenario"]):r for r in rows}
        eligible=[]
        for name in ARMS[1:-1]:
            delta=[]
            for sc in ("base","stress"):
                a,b=index["incumbent",sc],index[name,sc]
                if (b["net_pnl"]>a["net_pnl"] and b["Q"]>a["Q"] and
                    b["risk_breach_bars"]==0 and b["timing_pass"]):delta.append(b["net_pnl"]-a["net_pnl"])
            if len(delta)==2:eligible.append(dict(variant=name,worst_pnl_advantage=min(delta)))
        eligible.sort(key=lambda x:(-x["worst_pnl_advantage"],x["variant"]))
        selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),eligible=eligible,
                       nominee=eligible[0]["variant"] if eligible else None,
                       historical_independence=False,result_sha256=sha(path))
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
