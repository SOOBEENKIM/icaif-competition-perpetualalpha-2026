"""Cost-qualified all-market diversification and confirmed reversal."""
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
from .wave2 import vacant_overlay
from .wave4 import reverse_targets
from .model import features
from .prepare import native

OUT=ROOT/"reports/all_market_alpha_wave5_20261020"
ARMS=["incumbent","cost_qualified","cost_qualified_reverse"]


def freeze():
    OUT.mkdir(exist_ok=True);path=OUT/"protocol.json"
    if path.exists():return json.loads(path.read_text())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        rationale="Unconditional auxiliary trends cost more than they add. Reuse the ALREADY fixed pooled-ridge "
                  "expected-return gate instead of changing thresholds or increasing capital. Retain matched-risk agreement allocation.",
        core="Exact wave3 agreement_matched.",
        auxiliary="Exact wave1 pooled_ridge targets (all 10 assets, same model, 38bp roundtrip and signal confidence). "
                  "Use the fixed 25% vacant_daily rule. No parameter refit to PAS and no named-asset restriction.",
        reversal="Separate arm applies exact wave4 24h confirmed opposite-trend reversal using unchanged multiscale direction and size. "
                 "No change of reversal threshold after seeing evaluation results.",
        selection="Only cost_qualified_reverse eligible: August PnL and Q exceed incumbent in base AND stress, zero violations. "
                  "Both later-period and continuous results reported even when they fail. They are reused-history development evidence.",
        risk_matching="Scalar <=1 on July1–July29 predicted base daily risk, applied after all combinations.",
        prior_trials="All earlier candidate outcomes, including failed later audits, were seen. "
                     "This batch is adaptive research and must not be called independent OOS validation.",
        historical_independence=False,orders_sent=0,live_deployment_changed=False,
        pair_universe="Still original BTC/ETH and BTC/HYPE pair overlay; all-market evaluation is in the directional model.",
        deployment_hashes=verify_deployment()["files"])
    path.write_text(json.dumps(result,indent=2));return result


def prepare():
    freeze()
    first=dict(np.load(FIRST/"targets.npz",allow_pickle=False))
    third=dict(np.load(THIRD/"targets.npz",allow_pickle=False))
    ptr=json.loads((FIRST/"protocol.json").read_text())["pointer"]
    data=dict(np.load(ROOT/ptr["inputs"],allow_pickle=False))
    quality=np.nan_to_num(np.tanh(features(data)["x"][:,:,1:4].mean(-1)/2),nan=0)
    base=first["incumbent"];t=first["times"];sigma=first["sigma"]
    a=vacant_overlay(third["agreement_matched"],first["pooled_ridge"],t)
    b=reverse_targets(base,a,first["multiscale"],quality,t,"reverse")
    result=dict(times=t,sigma=sigma,incumbent=base,cost_qualified=a,cost_qualified_reverse=b)
    start=int(datetime(2026,7,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,7,29,tzinfo=timezone.utc).timestamp())
    calibration=(t>=start)&(t<end)
    def risk(w):
        return np.sqrt(np.maximum(0,24*np.einsum("ni,nij,nj->n",
            w[calibration],sigma[calibration],w[calibration]))).mean()
    p=load_profile()["trading"];scales={}
    for name in ARMS[1:]:
        scale=min(1.,risk(base)/max(risk(result[name]),1e-12));scales[name]=scale
        result[name]*=scale
        for i in range(168,len(t)):
            result[name][i]=project(result[name][i],p["initial"],sigma[i],p["risk"])
        result[name][:168]=0
    np.savez_compressed(OUT/"targets.npz",**result)
    (OUT/"calibration.json").write_text(json.dumps(dict(start=start,end=end,scales=scales),indent=2))
    files=[ROOT/"all_market_alpha"/(x+".py") for x in ["wave5","wave4","wave3","wave2","model"]]
    files += [FIRST/"targets.npz",THIRD/"targets.npz",OUT/"targets.npz"]
    (OUT/"source_freeze.json").write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
    print("wave5 prepared",scales,flush=True)


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
        name="cost_qualified_reverse"
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
