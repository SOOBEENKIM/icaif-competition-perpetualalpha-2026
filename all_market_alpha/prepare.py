import json
import numpy as np
from competition_live.profile import ROOT,load_profile
from .protocol import OUT,PRIOR,freeze,sha
from .model import features,rolling_forecast,targets


def native(v):
    return v.tolist() if hasattr(v,"tolist") else v.item()


def main():
    spec=freeze()
    for path,digest in spec["input_hashes"].items():
        assert sha(ROOT/path)==digest,path
    d=dict(np.load(ROOT/spec["pointer"]["inputs"],allow_pickle=False))
    old=dict(np.load(PRIOR/"targets.npz",allow_pickle=False))
    f=features(d);forecast,models=rolling_forecast(d,f,spec["model"])
    result=dict(times=d["market_times"],sigma=old["sigma"],incumbent=old["incumbent"])
    for name in ("multiscale","pooled_ridge"):
        result[name],result[name+"_state"]=targets(d,f,forecast,old["sigma"],
                                                  load_profile()["trading"],spec["policy"],name)
    result["equal_ensemble"]=(result["multiscale"]+result["pooled_ridge"])/2
    result["forecast"]=forecast
    np.savez_compressed(OUT/"targets.npz",**result)
    (OUT/"model_fits.json").write_text(json.dumps(models,default=native))
    # Append-only future data must not alter already-published forecasts/targets.
    stop=len(d["market_times"])-48
    prefix={k:v[:stop] for k,v in d.items()}
    pf=features(prefix);pp,_=rolling_forecast(prefix,pf,spec["model"])
    np.testing.assert_allclose(forecast[:stop],pp,equal_nan=True,atol=1e-12)
    for name in ("multiscale","pooled_ridge"):
        w,_=targets(prefix,pf,pp,old["sigma"][:stop],load_profile()["trading"],spec["policy"],name)
        np.testing.assert_allclose(result[name][:stop],w,atol=1e-12)
    np.testing.assert_allclose(result["incumbent"],old["incumbent"])
    (OUT/"source_freeze.json").write_text(json.dumps({
        str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"all_market_alpha").glob("*.py"))
    },indent=2))
    (OUT/"causality.json").write_text(json.dumps(dict(prefix_hours=48,
        forecast_prefix_invariant=True,targets_prefix_invariant=True,
        every_training_maturity_before_fit=all(m["last_maturity"]<m["fit_time"] for m in models),
        number_fits=len(models)),indent=2))
    print("prepared",len(d["market_times"]),"hours",len(models),"mature rolling fits",flush=True)


if __name__=="__main__":
    main()
