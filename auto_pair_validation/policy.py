"""Causal forecast calibration and monitoring, identical for every pair.

Selection is based on previously published predictions, not fitted training
residuals or a named asset list. Statistical margins are robustness heuristics,
not a confidence guarantee across dependent observations and 45 comparisons.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class ValidationSpec:
    calibration_days: int = 60
    calibration_minimum: int = 20
    monitoring_days: int = 30
    monitoring_minimum: int = 20
    monitoring_minimum_trades: int = 3
    coefficient_prior_rows: float = 12.
    half_life_days: float = 30.
    uncertainty_multiplier: float = 1.
    proxy_roundtrip_bps: float = 38.
    entry_buffer_bps: float = 5.
    hac_lags: int = 2


def fit_calibration(times, predictions, target, maturity, source, spec=ValidationSpec()):
    """One row per UTC day: forecast and target normalized at THEIR source.

    Callers supply out-of-time forecasts. Cropping before all numeric work
    ensures a future suffix, including malformed future labels, has no effect.
    """
    times, maturity = np.asarray(times), np.asarray(maturity)
    ix = (times >= source-spec.calibration_days*86400) & (maturity < source) & (times % 86400 == 0)
    x, y = np.asarray(predictions)[ix], np.asarray(target)[ix]
    stamps = maturity[ix]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, stamps = x[valid], y[valid], stamps[valid]
    if len(x) < spec.calibration_minimum:
        return None
    w = np.exp2(-(source-stamps)/(spec.half_life_days*86400)); w *= len(w)/w.sum()
    z = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(z.T@(w[:,None]*z) + np.eye(2)*spec.coefficient_prior_rows)
    theta = bread@(z.T@(w*y))
    # Shrink a noisy raw forecast, never mechanically amplify its slope.
    theta[1] = np.clip(theta[1], 0., 1.)
    residual = y-z@theta
    scores = z*(w*residual)[:,None]
    meat = scores.T@scores
    # Add only true neighboring days; missing observations are not adjacent.
    for lag in range(1, spec.hac_lags+1):
        pairs = np.argwhere((stamps[:,None]-stamps[None,:]) == lag*86400)
        if len(pairs):
            cross = scores[pairs[:,0]].T@scores[pairs[:,1]]
            meat += (1-lag/(spec.hac_lags+1))*(cross+cross.T)
    covariance = bread@meat@bread
    vals, vecs = np.linalg.eigh(covariance)
    covariance = (vecs*np.maximum(vals,0))@vecs.T
    return dict(theta=theta.tolist(), covariance=covariance.tolist(), rows=len(x),
                fit_source=int(source), last_label_maturity=int(stamps.max()),
                first_label_maturity=int(stamps.min()))


def calibrate(model, prediction, unit):
    if model is None or not np.isfinite([prediction, unit]).all() or unit <= 0:
        return None
    z = np.array([1., prediction/unit])
    mean = float(z@model['theta'])*unit
    error = float(np.sqrt(max(0., z@np.asarray(model['covariance'])@z)))*unit
    return mean, error


def current_action(mean, uncertainty, spec=ValidationSpec()):
    threshold = (spec.proxy_roundtrip_bps+spec.entry_buffer_bps)/10000
    if not np.isfinite([mean,uncertainty]).all() or uncertainty < 0:
        return 0
    return int(np.sign(mean)) if abs(mean)-spec.uncertainty_multiplier*uncertainty > threshold else 0


def monitor(times, published_mean, published_uncertainty, target, maturity,
            source, spec=ValidationSpec()):
    """Evaluate the already-published calibrated policy on matured daily rows.

    No-trade days contribute zero payoff. MAE is calculated on all available
    daily forecasts, not just the hindsight winning trades. Proxy payoff is
    NOT executable account PnL: funding, partial fills and renewals are replayed
    separately with the incumbent execution model.
    """
    times, maturity = np.asarray(times), np.asarray(maturity)
    use = (times >= source-spec.monitoring_days*86400) & (maturity < source) & (times % 86400 == 0)
    x, se, y = [np.asarray(v)[use] for v in (published_mean,published_uncertainty,target)]
    stamps = times[use]
    good = np.isfinite(x)&np.isfinite(se)&np.isfinite(y)&(se>=0)
    x,se,y,stamps = x[good],se[good],y[good],stamps[good]
    actions = np.array([current_action(a,b,spec) for a,b in zip(x,se)])
    payoff = actions*y-abs(actions)*spec.proxy_roundtrip_bps/10000
    reasons = []
    if len(x) < spec.monitoring_minimum: reasons.append('insufficient_mature_monitoring_days')
    if np.count_nonzero(actions) < spec.monitoring_minimum_trades: reasons.append('insufficient_past_trade_signals')
    if not len(x):
        return dict(eligible=False,reasons=reasons,rows=0,trades=0,last_label_maturity=None,
                    prediction_mae=None,zero_mae=None,mean_net_proxy=None,mean_uncertainty=None)
    mae, zero = float(np.mean(abs(x-y))),float(np.mean(abs(y)))
    if mae >= zero: reasons.append('forecast_not_better_than_zero')
    # Calendar-day HAC standard error; missing days don't become neighbors.
    centered = payoff-payoff.mean(); meat=float(centered@centered)
    for lag in range(1,spec.hac_lags+1):
        pairs=np.argwhere((stamps[:,None]-stamps[None,:])==lag*86400)
        if len(pairs):meat+=2*(1-lag/(spec.hac_lags+1))*float(centered[pairs[:,0]]@centered[pairs[:,1]])
    se_mean=float(np.sqrt(max(0.,meat))/len(x))
    mean=float(payoff.mean())
    if mean-spec.uncertainty_multiplier*se_mean <= 0:reasons.append('past_net_proxy_not_robustly_positive')
    return dict(eligible=not reasons,reasons=reasons,rows=len(x),trades=int(np.count_nonzero(actions)),
                last_label_maturity=int((stamps+86400).max()),prediction_mae=mae,zero_mae=zero,
                mean_net_proxy=mean,mean_uncertainty=se_mean)


def apply_entry_gate(state, arm):
    """All 45 stable IDs and held forecasts survive deselection."""
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    if arm=='expanded_only':
        out['mu']=out['raw_mu'];out['economic_available']=out['gate_available']
        out['economic_action']=np.where(out['gate_available'],out['gate_action'],0)
    else:
        available=out['calibration_available'] & out['directions'].astype(bool)
        if arm=='validated_all45':available &= out['validated']
        elif arm!='calibrated_only':raise ValueError('Unknown experiment arm')
        out['economic_available']=available
        out['economic_action']=np.where(available,out['calibrated_action'],0)
    return out
