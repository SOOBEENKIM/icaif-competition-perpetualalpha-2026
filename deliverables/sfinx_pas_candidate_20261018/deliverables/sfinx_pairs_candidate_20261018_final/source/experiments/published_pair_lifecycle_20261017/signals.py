"""Causal forecasts from immutable, captured hourly publication envelopes.

No raw capture is opened here. The caller must enforce its registered release
time before passing events. Daily coefficients are fitted from the actual
midnight publication once; later revised snapshots cannot silently refit them.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.dense_history_pairs_20261017.model import design, labels
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions, fit_correction
from experiments.rolling_feature_pairs_20261017.model import fit


FRAME_KEYS = ('source_time', 'published_time', 'weights', 'retention', 'sigma', 'volume', 'trades', 'source_hash')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def validate_envelope(event):
    if event['channel'] != 'strategySignal':
        raise ValueError('Expected captured strategy publication')
    data = event['data']
    frame = {key: data['frame'][key] for key in FRAME_KEYS}
    digest = digest_bytes(json.dumps(frame, sort_keys=True, allow_nan=False).encode())
    if digest != data['publication_sha256']:
        raise ValueError('Publication frame digest mismatch')
    source, published = frame['source_time'], frame['published_time']
    if source % 3600 or not np.isfinite([source, published, event['received_ms']]).all():
        raise ValueError('Invalid publication clock')
    if not source <= published <= event['received_ms']/1000:
        raise ValueError('Publication unavailable at capture receipt')
    for key in ('weights', 'retention', 'volume', 'trades'):
        value = np.asarray(frame[key])
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError('Invalid publication vector')
    covariance = np.asarray(frame['sigma'])
    if covariance.shape != (10, 10) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
        raise ValueError('Invalid publication covariance')
    if np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError('Indefinite publication covariance')
    return frame


def load_snapshot(root, event):
    frame = validate_envelope(event)
    root = Path(root).resolve()
    run = (root/event['data']['source_run']).resolve()
    if not run.is_relative_to(root/'practice_live/updates'):
        raise ValueError('Publication input outside immutable update directory')
    path = run/'inputs.npz'
    if digest_bytes(path.read_bytes()) != frame['source_hash']:
        raise ValueError('Captured input digest mismatch')
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key].copy() for key in ('market_times', 'close', 'volume', 'trades')}
    if int(result['market_times'][-1]) != frame['source_time']:
        raise ValueError('Snapshot contains a different source hour')
    return frame, result


class PublishedSignals:
    def __init__(self, symbols, pairs, models, direct_spec, bias_spec, latency_seconds=5):
        self.symbols, self.pairs, self.models = list(symbols), copy.deepcopy(pairs), models
        self.direct_spec, self.bias_spec = direct_spec, bias_spec
        self.latency_seconds = latency_seconds
        self.daily = {}
        self.daily_source = None
        self.last_source = -1
        self.last_digest = None
        self.last_result = None
        self.audit = []

    def ingest_file(self, root, event):
        _, snapshot = load_snapshot(root, event)
        return self.ingest(event, snapshot)

    def ingest(self, event, snapshot):
        frame = validate_envelope(event)
        source = int(frame['source_time'])
        if source < self.last_source:
            raise ValueError('Out-of-order source publication')
        if source == self.last_source:
            if frame['source_hash'] != self.last_digest:
                raise ValueError('Source hour was revised after first publication')
            return copy.deepcopy(self.last_result)
        t = snapshot['market_times']
        if not len(t) or t[-1] != source or not np.all(np.diff(t) == 3600):
            raise ValueError('Snapshot must end exactly at the source hour')
        for key in ('close', 'volume', 'trades'):
            value = snapshot[key]
            if value.shape != (len(t), len(self.symbols)) or not np.isfinite(value).all():
                raise ValueError('Invalid source market matrix')
            invalid = np.any(value <= 0) if key == 'close' else np.any(value < 0)
            if invalid:
                raise ValueError('Invalid source prices or activity')
        h = self.direct_spec['horizon']
        features = design(t, snapshot['close'], snapshot['volume'], snapshot['trades'], self.symbols, self.pairs)
        raw = frozen_predictions(features, self.models, self.pairs, h)
        target, maturity = labels(t, snapshot['close'], features, h)
        terminal = np.zeros_like(features['fresh'])
        terminal[:-h] = features['fresh'][h:]
        finite = features['fresh'] & np.isfinite(raw) & np.isfinite(features['x']).all(-1)
        eligible = finite & terminal & np.isfinite(target)
        day = source//86400*86400
        if self.daily_source != day:
            # Missing the daily publication cannot be repaired by fitting from
            # a later, potentially revised input file and backdating its model.
            self.daily = {}
            self.daily_source = None
        if source % (self.direct_spec['refit_hours']*3600) == 0:
            self.daily_source = source
            window = (t >= source-self.direct_spec['lookback_days']*86400) & (maturity < source)
            window &= t % (self.direct_spec['sample_hours']*3600) == 0
            for k in range(len(self.pairs)):
                ii = np.flatnonzero(window & eligible[:, k])
                model = fit(features['x'][ii, k], target[ii, k], raw[ii, k], maturity[ii], source, self.direct_spec)
                if model is not None:
                    self.daily[k] = model
            self.audit.append(dict(source=source, kind='daily_fit', source_hash=frame['source_hash'],
                                   receipt_ms=event['received_ms'], models=copy.deepcopy(self.daily)))
        results = []
        for k, pair in enumerate(self.pairs):
            ids = list(map(int, features['pairs'][k]))
            predictions = {}
            if finite[-1, k]:
                unit = features['vol'][-1, k]*np.sqrt(h)
                if k in self.daily:
                    model = self.daily[k]
                    z = np.r_[1., (features['x'][-1, k]-model['center'])/model['scale']]
                    # Preserve multiplication order used by frozen historical direct.
                    predictions['direct'] = float(np.clip(z@model['direct'], -self.direct_spec['prediction_clip'], self.direct_spec['prediction_clip']) * features['vol'][-1, k] * np.sqrt(h))
                window = ((t % (h*3600) == 0) & (maturity < source)
                          & (maturity >= source-self.bias_spec['lookback_days']*86400))
                ii = np.flatnonzero(window & features['fresh'][:, k] & np.isfinite(target[:, k]) & np.isfinite(raw[:, k]))
                correction = fit_correction(raw[ii, k], target[ii, k], maturity[ii], source, self.bias_spec)
                if correction is not None:
                    predictions['bias'] = float(np.clip(raw[-1, k]+correction['bias'], -self.bias_spec['normalized_target_clip'], self.bias_spec['normalized_target_clip'])*unit)
            for model, mu in predictions.items():
                anchors = snapshot['close'][-1, ids]
                results.append(dict(model=model, pair=pair, ids=ids, horizon=h, forecast_source=source,
                    available_time=event['received_ms']/1000+self.latency_seconds,
                    source_prices=anchors.tolist(), source_beta=float(features['beta'][-1, k]),
                    source_vol=float(features['vol'][-1, k]), mu=mu,
                    terminal_prices=(anchors*np.array([1+mu, 1-mu])).tolist(),
                    coefficient_fit_source=self.daily_source if model == 'direct' else source,
                    source_hash=frame['source_hash']))
        self.last_source, self.last_digest = source, frame['source_hash']
        self.last_result = copy.deepcopy(results)
        return results
