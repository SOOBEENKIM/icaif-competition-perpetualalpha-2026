"""All-pair causal forecasts and daily eligibility; no exchange dependencies.

Retains the incumbent direct ridge, SPO+ gate and source-unit convention.
Selection can block NEW entries, never delete a held pair's direct forecast.
"""
import copy
import hashlib
import json
import numpy as np
from .selection import PAIRS, IDS, SYMBOLS, SelectionSpec, select, validate_snapshot
import competition_live  # Register the immutable incumbent implementation.
from experiments.dense_history_pairs_20261017.model import design, labels
from experiments.rolling_feature_pairs_20261017.model import fit as direct_fit
from experiments.decision_gate_pairs_20261017.model import fit as gate_fit, action


def prepare_features(d):
    validate_snapshot(d)
    f = design(d['market_times'], d['close'], d['volume'], d['trades'], SYMBOLS, PAIRS)
    y, maturity = labels(d['market_times'], d['close'], f, 24)
    terminal = np.zeros_like(f['fresh'])
    terminal[:-24] = f['fresh'][24:]
    return f, y, maturity, terminal


def fit_day(d, f, target, maturity, terminal, source, direct_spec, gate_spec,
            selection_spec=SelectionSpec(), additional_gate_ids=()):
    selection = select(d, source, selection_spec)
    t = d['market_times']
    window = ((t >= source-direct_spec['lookback_days']*86400) & (maturity < source)
              & (t % (direct_spec['sample_hours']*3600) == 0))
    finite = np.isfinite(f['x']).all(-1) & f['fresh'] & terminal & np.isfinite(target)
    unit = f['vol']*np.sqrt(24)
    finite &= np.isfinite(unit) & (unit > 0)
    result = dict(source=int(source), selection=selection, direct={}, gate={}, failures=[])
    gate_ids = set(np.flatnonzero(selection['selected'])) | set(additional_gate_ids)
    for k in range(len(PAIRS)):
        ix = np.flatnonzero(window & finite[:, k])
        # The retained 'direct' column regresses y on x; its coefficients do
        # not depend on the compatibility argument for residual predictions.
        m = direct_fit(f['x'][ix, k], target[ix, k], np.zeros(len(ix)), maturity[ix], source, direct_spec)
        if m is not None:
            result['direct'][k] = m
        if k in gate_ids:
            try:
                g = gate_fit(f['x'][ix, k], target[ix, k], unit[ix, k], maturity[ix], source, gate_spec)
                if g is not None:
                    result['gate'][k] = g
            except ArithmeticError as error:
                # Convergence failure is recorded and prevents new entries.
                result['failures'].append(dict(pair=list(PAIRS[k]), reason=str(error)))
    return result


def predict_hour(f, i, model, direct_spec, gate_spec):
    n = len(PAIRS)
    mu = np.full(n, np.nan); scores = np.full(n, np.nan)
    directions = np.zeros(n, bool); available = np.zeros(n, bool); actions = np.zeros(n, int)
    for k, m in model['direct'].items():
        if not f['fresh'][i, k] or not np.isfinite(f['x'][i, k]).all():
            continue
        unit = f['vol'][i, k]*np.sqrt(24)
        if not np.isfinite(unit) or unit <= 0:
            continue
        z = np.r_[1., (f['x'][i, k]-m['center'])/m['scale']]
        mu[k] = float(np.clip(z @ m['direct'], -direct_spec['prediction_clip'], direct_spec['prediction_clip'])*unit)
        directions[k] = True
        if k in model['gate']:
            g = model['gate'][k]
            z = np.r_[1., (f['x'][i, k]-g['center'])/g['scale']]
            scores[k] = float(z @ g['coefficient'])
            cost = 2*(gate_spec['fee_bps']+gate_spec['impact_bps'])/10000/unit
            actions[k] = int(action(np.asarray(scores[k]), cost))
            available[k] = True
    return dict(mu=mu, directions=directions, gate_available=available,
                gate_action=actions, gate_score=scores)


def snapshot_hash(d):
    h = hashlib.sha256()
    for key in ('market_times', 'close', 'volume', 'trades'):
        h.update(key.encode()); h.update(np.ascontiguousarray(d[key]).tobytes())
    return h.hexdigest()


class AutomaticPairSignals:
    """Staged publication adapter with the existing coordinator's forecast schema.

    One contiguous prefix ending at source is supplied AFTER its actual receipt.
    Restarting mid-day cannot backdate a missed daily fit: warm it from a saved
    midnight snapshot (and its receipt) first, or wait for the next midnight.
    The class cannot load keys, place orders or modify a live profile.
    """
    def __init__(self, direct_spec, gate_spec, selection_spec=SelectionSpec(), latency_seconds=5):
        if direct_spec['horizon'] != 24 or gate_spec['horizon'] != 24:
            raise ValueError('Retained horizon must be 24 hours')
        self.direct_spec = copy.deepcopy(direct_spec); self.gate_spec = copy.deepcopy(gate_spec)
        self.spec = selection_spec; self.latency = latency_seconds
        self.model = None; self.last_source = -1; self.last_hash = None; self.last_result = None
        self.audit = []

    def ingest(self, d, source, received_at):
        t = validate_snapshot(d)
        if t[-1] != source or not np.isfinite(received_at) or received_at < source or received_at >= source+3600:
            raise ValueError('Snapshot/receipt must identify the current completed source hour')
        if source < self.last_source:
            raise ValueError('Out-of-order publication')
        digest = snapshot_hash(d)
        if source == self.last_source:
            if digest != self.last_hash:
                raise ValueError('Published source was revised')
            return copy.deepcopy(self.last_result)
        f, y, maturity, terminal = prepare_features(d)
        day = source//86400*86400
        if self.model is not None and self.model['source'] != day:
            self.model = None
        if source % 86400 == 0:
            self.model = fit_day(d, f, y, maturity, terminal, source, self.direct_spec, self.gate_spec, self.spec)
            self.model['received_at'] = received_at
            self.audit.append(dict(source=source, source_hash=digest, received_at=received_at,
                                   selection=self.model['selection'], failures=self.model['failures']))
        result = []
        if self.model is not None:
            p = predict_hour(f, len(t)-1, self.model, self.direct_spec, self.gate_spec)
            for k, pair in enumerate(PAIRS):
                if not p['directions'][k]:
                    continue
                eligible = self.model['selection']['selected'][k]
                ids = IDS[k]; anchors = d['close'][-1, list(ids)]
                mu = float(p['mu'][k])
                result.append(dict(model='decision_gate', pair=list(pair), ids=list(ids), horizon=24,
                    forecast_source=int(source), available_time=received_at+self.latency,
                    source_prices=anchors.tolist(), source_beta=float(f['beta'][-1, k]),
                    source_vol=float(f['vol'][-1, k]), mu=mu,
                    terminal_prices=(anchors*np.array([1+mu, 1-mu])).tolist(),
                    coefficient_fit_source=self.model['source'], source_hash=digest,
                    economic_available=bool(eligible and p['gate_available'][k]),
                    economic_action=int(p['gate_action'][k]) if eligible else 0,
                    decision_score=float(p['gate_score'][k]) if np.isfinite(p['gate_score'][k]) else None,
                    gate_fit_source=self.model['source'], selection_source=self.model['source'],
                    selection_eligible=bool(eligible), stable_pair_index=k))
        self.last_source = source; self.last_hash = digest; self.last_result = copy.deepcopy(result)
        return result


def state_for_arm(state, arm):
    """Entry-only selection mask. Forecasts for existing inventory remain intact."""
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    if arm == 'automatic_45':
        mask = out['selection_eligible']
    elif arm == 'fixed_two_rebuilt':
        control = np.array([pair in (('BTC', 'ETH'), ('BTC', 'HYPE')) for pair in PAIRS])
        mask = np.broadcast_to(control, out['mu'].shape)
    else:
        raise ValueError('Unknown comparison arm')
    out['economic_available'] = out['gate_available'] & mask
    out['economic_action'] = np.where(out['economic_available'], out['gate_action'], 0)
    return out
