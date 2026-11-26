"""Reconstruct the retained entry gate without changing direct holding forecasts."""
import copy

import numpy as np

from experiments.published_pair_lifecycle_20261017.signals import PublishedSignals, validate_envelope
from experiments.dense_history_pairs_20261017.model import design, labels
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.decision_gate_pairs_20261017.model import fit, action


class DecisionGateSignals(PublishedSignals):
    def __init__(self, symbols, pairs, models, direct_spec, bias_spec, gate_spec, latency_seconds=5):
        super().__init__(symbols, pairs, models, direct_spec, bias_spec, latency_seconds)
        self.gate_spec = copy.deepcopy(gate_spec)
        if gate_spec['horizon'] != direct_spec['horizon'] or gate_spec['refit_hours'] != 24:
            raise ValueError('Adapter requires the retained common horizon and daily gate refit')
        self.gate_daily = {}
        self.gate_day = None
        self.gate_audit = []
        self.combined_result = None

    def ingest(self, event, snapshot):
        frame = validate_envelope(event)
        source = int(frame['source_time'])
        duplicate = source == self.last_source
        # Parent performs publication, snapshot, revision and causal checks.
        original = super().ingest(event, snapshot)
        if duplicate:
            return copy.deepcopy(self.combined_result)
        spec = self.gate_spec
        times = snapshot['market_times']
        h = spec['horizon']
        features = design(times, snapshot['close'], snapshot['volume'], snapshot['trades'], self.symbols, self.pairs)
        raw = frozen_predictions(features, self.models, self.pairs, h)
        target, maturity = labels(times, snapshot['close'], features, h)
        unit = features['vol'] * np.sqrt(h)
        terminal = np.zeros_like(features['fresh'])
        terminal[:-h] = features['fresh'][h:]
        valid = (np.isfinite(features['x']).all(-1) & features['fresh'] & np.isfinite(raw)
                 & np.isfinite(unit) & (unit > 0))
        train = valid & terminal & np.isfinite(target)
        day = source // 86400 * 86400
        if self.gate_day != day:
            self.gate_daily = {}
            self.gate_day = None
        if source % (spec['refit_hours'] * 3600) == 0:
            self.gate_day = source
            window = ((times >= source - spec['lookback_days'] * 86400) & (maturity < source)
                      & (times % (spec['sample_hours'] * 3600) == 0))
            for k in range(len(self.pairs)):
                ii = np.flatnonzero(window & train[:, k])
                model = fit(features['x'][ii, k], target[ii, k], unit[ii, k], maturity[ii], source, spec)
                if model is not None:
                    self.gate_daily[k] = model
            self.gate_audit.append(dict(source=source, source_hash=frame['source_hash'],
                receipt_ms=event['received_ms'], models=copy.deepcopy(self.gate_daily)))
        result = copy.deepcopy(original)
        for direct in original:
            if direct['model'] != 'direct':
                continue
            k = self.pairs.index(direct['pair'])
            available = k in self.gate_daily and bool(valid[-1, k])
            score = None
            chosen = 0
            if available:
                model = self.gate_daily[k]
                z = np.r_[1., (features['x'][-1, k] - model['center']) / model['scale']]
                score = float(z @ model['coefficient'])
                cost = 2 * (spec['fee_bps'] + spec['impact_bps']) / 10000 / unit[-1, k]
                chosen = int(action(np.asarray(score), cost))
            # Direct mu, source prices, beta, volatility, terminal prices and
            # availability remain identical. The action score is never a price.
            result.append(dict(direct, model='decision_gate', economic_available=available,
                economic_action=chosen, decision_score=score,
                gate_fit_source=self.gate_day if available else None))
        self.combined_result = copy.deepcopy(result)
        return result
