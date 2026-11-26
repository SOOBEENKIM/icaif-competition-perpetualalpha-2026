"""A once-fitted auxiliary model; prediction never calls labels or fit."""
import copy
import numpy as np
from all_market_alpha.model import features, labels, fit, targets
from all_market_alpha.wave2 import vacant_overlay
from all_market_alpha.wave3 import allocate
from all_market_alpha.wave4 import reverse_targets
from replay_lab.common_comparison.engine import project


def train(history, spec, start, cutoff):
    times = np.asarray(history["market_times"])
    # Slice before feature and target construction, not merely before fitting.
    prefix = {k: v[times < cutoff] for k, v in history.items()}
    t = prefix["market_times"]
    f = features(prefix)
    y, maturity, terminal = labels(prefix, f, spec["horizon"], spec["execution_delay"])
    mask = ((t >= start) & (maturity < cutoff)
            & (t % (spec["sample_hours"] * 3600) == 0))
    valid = (np.isfinite(f["x"]).all(-1) & np.isfinite(y)
             & f["fresh"] & terminal & mask[:, None])
    ii, jj = np.where(valid)
    model = fit(f["x"][ii, jj], y[ii, jj], t[ii], maturity[ii], cutoff, spec)
    if model is None:
        raise ValueError("Insufficient past-only fitting observations")
    model.update(train_start=int(start), cutoff=int(cutoff),
                 feature_last_time=int(t[ii].max()),
                 asset_counts=np.bincount(jj, minlength=y.shape[1]).tolist(),
                 label_rows=[dict(source=int(t[i]), maturity=int(maturity[i]), asset=int(j))
                             for i, j in zip(ii, jj)],
                 forecast_clip=spec["forecast_clip"])
    return model


def predict(history, model):
    f = features(history)
    prediction = np.clip(
        model["beta"][0] + ((f["x"] - model["center"]) / model["scale"]) @ model["beta"][1:],
        -model["forecast_clip"], model["forecast_clip"]) * f["unit"]
    prediction[history["market_times"] < model["cutoff"]] = np.nan
    return f, prediction


def build(history, incumbent, sigma, trading, spec, calibration, model):
    """Keep wave5 policy fixed; replace only its daily-refitted auxiliary fit."""
    if not all(np.isfinite(calibration[k]) and 0 < calibration[k] <= 1
               for k in ("agreement_scale", "final_scale")):
        raise ValueError("Calibration may not increase risk")
    f, forecast = predict(history, model)
    momentum, _ = targets(history, f, forecast, sigma, trading, spec["policy"], "multiscale")
    economic, _ = targets(history, f, forecast, sigma, trading, spec["policy"], "pooled_ridge")
    quality = np.nan_to_num(np.tanh(f["x"][:, :, 1:4].mean(-1) / 2), nan=0)
    t = history["market_times"]
    agreement = allocate(incumbent, quality, t) * calibration["agreement_scale"]
    for i in range(168, len(t)):
        agreement[i] = project(agreement[i], trading["initial"], sigma[i], trading["risk"])
    agreement[:168] = 0
    combined = vacant_overlay(agreement, economic, t)
    result = reverse_targets(incumbent, combined, momentum, quality, t, "reverse")
    result *= calibration["final_scale"]
    for i in range(168, len(t)):
        result[i] = project(result[i], trading["initial"], sigma[i], trading["risk"])
    result[:168] = 0
    if not np.isfinite(result).all():
        raise ValueError("Invalid frozen-model targets")
    return result, dict(forecast=forecast, economic=economic, quality=quality)


def publication(original, weights):
    """Portfolio reads data.incumbent: update all three weight representations."""
    import hashlib
    import json
    out = copy.deepcopy(original)
    w = np.asarray(weights, float)
    if w.shape != np.asarray(out["data"]["incumbent"]).shape or not np.isfinite(w).all():
        raise ValueError("Invalid candidate publication")
    for key in ("incumbent", "candidate"):
        out["data"][key] = w.tolist()
    out["data"]["frame"]["weights"] = w.tolist()
    out["data"]["publication_sha256"] = hashlib.sha256(
        json.dumps(out["data"]["frame"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    return out
