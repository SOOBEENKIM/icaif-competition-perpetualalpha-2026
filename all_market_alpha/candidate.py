"""Reproducible candidate target builder; no order submission or live mutation."""
import copy
import hashlib
import json
import numpy as np
from replay_lab.common_comparison.engine import project
from .model import features,rolling_forecast,targets
from .wave2 import vacant_overlay
from .wave3 import allocate
from .wave4 import reverse_targets


def build_weights(history,incumbent,sigma,trading,spec,calibration):
    times=history["market_times"]
    shape=history["close"].shape
    if not all(np.isfinite(calibration[k]) and 0<calibration[k]<=1
               for k in ("agreement_scale","final_scale")):
        raise ValueError("Risk matching may not increase exposure")
    if not np.isfinite(history["close"]).all() or np.any(history["close"]<=0):
        raise ValueError("Invalid source prices")
    if incumbent.shape!=shape or sigma.shape!=(shape[0],shape[1],shape[1]):
        raise ValueError("Misaligned candidate inputs")
    if shape[1]!=len(trading["symbols"]) or not np.all(np.diff(times)==3600):
        raise ValueError("Complete aligned hourly universe required")
    f=features(history)
    forecast,models=rolling_forecast(history,f,spec["model"])
    momentum,momentum_state=targets(history,f,forecast,sigma,trading,spec["policy"],"multiscale")
    economic,economic_state=targets(history,f,forecast,sigma,trading,spec["policy"],"pooled_ridge")
    quality=np.nan_to_num(np.tanh(f["x"][:,:,1:4].mean(-1)/2),nan=0)
    agreement=allocate(incumbent,quality,times)*calibration["agreement_scale"]
    for i in range(168,len(times)):
        agreement[i]=project(agreement[i],trading["initial"],sigma[i],trading["risk"])
    agreement[:168]=0
    combined=vacant_overlay(agreement,economic,times)
    result=reverse_targets(incumbent,combined,momentum,quality,times,"reverse")
    result*=calibration["final_scale"]
    for i in range(168,len(times)):
        result[i]=project(result[i],trading["initial"],sigma[i],trading["risk"])
    result[:168]=0
    if not np.isfinite(result).all():
        raise ValueError("Invalid candidate weights")
    return result,dict(forecast=forecast,quality=quality,economic=economic,
        momentum=momentum,economic_state=economic_state,momentum_state=momentum_state,
        model_fits=models,agreement=agreement,combined=combined)


def publication(original,weights):
    """Transform an already-validated input envelope, retaining its timing."""
    result=copy.deepcopy(original)
    frame=result["data"]["frame"]
    weights=np.asarray(weights,float)
    if weights.shape!=np.asarray(frame["weights"]).shape or not np.isfinite(weights).all():
        raise ValueError("Invalid publication weights")
    frame["weights"]=weights.tolist()
    frame["strategy_variant"]="all_market_cost_qualified_reverse"
    result["data"]["candidate"]=weights.tolist()
    result["data"]["publication_sha256"]=hashlib.sha256(
        json.dumps(frame,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result
