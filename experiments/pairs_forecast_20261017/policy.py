"""Paired inventory state machine: balanced partial fills, holds, paid exits."""
import numpy as np
from replay_lab.common_comparison.engine import risk_breaches, rounded


def balanced_inventory(actual, planned, tolerance):
    """Return target that only reduces the overfilled leg, never invents fills."""
    actual, planned = np.asarray(actual), np.asarray(planned)
    if np.any(actual * planned <= 0) or np.any(abs(planned) < 1e-12):
        return False, np.zeros_like(actual)
    ratios = abs(actual / planned)
    if min(ratios) / max(ratios) >= 1 - tolerance:
        return True, actual.copy()
    return False, np.sign(actual) * np.minimum(abs(actual), abs(planned) * min(ratios))


class ForecastController:
    def __init__(self, market, prediction, beta, vol, cfg, horizon, scenario, last):
        self.m, self.prediction, self.beta, self.vol = market, prediction, beta, vol
        self.cfg, self.horizon, self.scenario, self.last = cfg, horizon, scenario, last
        self.pairs = [[market.symbols.index(s) for s in pair] for pair in cfg['pairs']]
        self.states, self.cooldown, self.events = {}, {}, []

    def propose(self, time, prices, qty, equity, raw, sigma, volume, trades, protocol, **kwargs):
        i = self.m.exact(time); s = self.cfg['strategy']; target = qty.copy()
        forced = bool(kwargs.get('halted') or risk_breaches(qty, equity, prices, sigma, protocol['risk']))
        ending = i >= self.last - 2; used = set()
        for k, state in list(self.states.items()):
            ids = self.pairs[k]; actual = qty[ids]
            if not np.any(actual):
                self.events.append(dict(time=time, pair=k, event='flat_confirmed'))
                self.cooldown[k] = i + s['cooldown_hours']; del self.states[k]; continue
            used.update(ids)
            if state['phase'] in ('entering', 'balancing'):
                balanced, reduced = balanced_inventory(actual, state['planned'], s['balanced_fill_tolerance'])
                if balanced:
                    state.update(phase='holding', entry_prices=prices[ids].copy(), actual=actual.copy())
                    self.events.append(dict(time=time, pair=k, event='balanced_open', qty=actual.tolist()))
                elif i - state['entry'] <= s['maximum_balance_hours'] and np.all(reduced):
                    state['phase'] = 'balancing'; target[ids] = reduced
                    self.events.append(dict(time=time, pair=k, event='reduce_imbalance', actual=actual.tolist(), target=reduced.tolist()))
                    continue
                else:
                    state['phase'] = 'closing'
                    self.events.append(dict(time=time, pair=k, event='unhedged_abort'))
            reason = None
            if forced: reason = 'account_risk'
            elif ending: reason = 'evaluation_end'
            elif state['phase'] == 'closing': reason = 'closing_retry'
            elif i - state['entry'] >= self.horizon: reason = 'horizon'
            elif abs(self.beta[i, k] / state['beta'] - 1) > s['hedge_drift_exit']: reason = 'hedge_instability'
            elif state['phase'] == 'holding':
                pair_return = float(actual @ (prices[ids] - state['entry_prices'])) / max(float(abs(actual) @ state['entry_prices']), 1e-10)
                if pair_return < -s['stop_sigma'] * state['entry_vol'] * np.sqrt(self.horizon): reason = 'spread_stop'
                elif time % (s['decision_hours'] * 3600) == 0 and state['direction'] * self.prediction[i, k] < 0: reason = 'forecast_reversal'
            if reason:
                if state['phase'] != 'closing': self.events.append(dict(time=time, pair=k, event='exit_intent', reason=reason))
                state['phase'] = 'closing'; target[ids] = 0.
        if not forced and not ending and time % (s['decision_hours'] * 3600) == 0:
            candidates = []
            cost = 2 * (protocol['execution']['fee_bps'] + self.scenario['impact_bps']) / 10000
            for k, ids in enumerate(self.pairs):
                mu = self.prediction[i, k]
                if k in self.states or self.cooldown.get(k, 0) > i or not np.isfinite(mu): continue
                if np.any(volume[ids] <= 0) or np.any(trades[ids] <= 0): continue
                if abs(mu) > cost + s['entry_buffer_bps'] / 10000:
                    candidates.append((abs(mu) - cost, k, mu))
            for edge, k, mu in sorted(candidates, reverse=True):
                ids = self.pairs[k]
                if len(self.states) >= s['maximum_pairs'] or used.intersection(ids): continue
                beta = self.beta[i, k]
                if not np.isfinite(beta): continue
                direction = int(np.sign(mu))
                weights = direction * np.array([1., -beta]) / (1 + beta) * s['pair_gross']
                factor = 1.
                for j, w in zip(ids, weights):
                    factor = min(factor, .9 * self.scenario['participation'] * volume[j] * prices[j] / (abs(w) * equity))
                remaining = max(0., protocol['risk']['normal_turnover_fraction'] - float(abs((target - qty) * prices / equity).sum()))
                factor = min(factor, remaining / max(float(abs(weights).sum()), 1e-12))
                units = weights * factor * equity / prices[ids]
                lo, hi = 0., 1.
                for _ in range(28):
                    mid = (lo + hi) / 2; trial = target.copy(); trial[ids] = units * mid
                    if risk_breaches(trial, equity, prices, sigma, protocol['risk']): hi = mid
                    else: lo = mid
                units *= lo * protocol['risk']['sizing_headroom']
                units = np.array([rounded(v, protocol['execution']['decimals'][j]) for v, j in zip(units, ids)])
                if np.any(abs(units) * prices[ids] < protocol['execution']['minimum_order_value']): continue
                target[ids] = units; used.update(ids)
                self.states[k] = dict(phase='entering', planned=units.copy(), beta=beta, direction=direction,
                                     entry=i + 1, entry_vol=self.vol[i, k])
                self.events.append(dict(time=time, pair=k, event='entry_intent', predicted_return=float(mu),
                    net_edge=float(edge), beta=float(beta), target=units.tolist()))
        if np.allclose(target, qty, rtol=0, atol=1e-12): return None
        return dict(target_qty=target.tolist(), risk_forced=forced, defensive=forced)
