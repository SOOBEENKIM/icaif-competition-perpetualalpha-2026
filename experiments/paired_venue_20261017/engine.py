"""Visible-depth paired execution with independent legs and paid repair.

This module has no signing or exchange client. All inventory changes are fills
from the existing conservative Paper broker, not desired target quantities.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import numpy as np
from experiments.execution_retry_20261016.engine import Paper, lot_floor
from experiments.pairs_forecast_20261017.policy import balanced_inventory


def book_walk(levels, quantity, fraction=1., limit=None, side=1):
    remaining, value = quantity, 0.
    for row in levels:
        px = float(row['px'])
        if limit is not None and side * (px - limit) > 1e-9:
            break
        size = min(remaining, float(row['sz']) * fraction)
        value += size * px; remaining -= size
        if remaining <= 1e-10:
            return value / quantity
    return None


def quote_pair(coins, snapshots, metadata, beta, direction, horizon, cfg, scenario,
               gross=None, fixed_quantity=None):
    """Forecast value using executable entry/depth and spread-paying exit.

    Oracle is a valuation reference, never an executable venue. Funding is an
    uncertain forecast; no future realized rate is inspected here. Any capacity
    scale applies to BOTH legs, including the depth needed for an exit now.
    """
    if not math.isfinite(beta) or not .25 <= beta <= 4 or direction not in (-1, 1):
        return None
    gross = cfg['target_pair_gross'] if gross is None else gross
    oracle = np.array([snapshots[c]['oracle'] for c in coins])
    signs = np.array([direction, -direction])
    planned = gross * np.array([1., beta]) / (1 + beta) / oracle
    fraction = scenario['visible_depth_fraction']
    if fixed_quantity is not None:
        planned = np.abs(fixed_quantity)
        signs = np.sign(fixed_quantity)
    capacity_scale = 1.
    for c, quantity, sign in zip(coins, planned, signs):
        book = snapshots[c]['book']
        for side in (sign, -sign):
            levels = book['levels'][1 if side > 0 else 0]
            if not levels:
                return None
            best = float(levels[0]['px'])
            limit = best * (1 + side * cfg['limit_bps'] / 10000)
            capacity = sum(float(r['sz']) * fraction for r in levels if side * (float(r['px']) - limit) <= 1e-9)
            capacity_scale = min(capacity_scale, capacity / max(quantity, 1e-12))
    # Valuing existing inventory must never silently shrink the position.
    if fixed_quantity is not None and capacity_scale < 1 - 1e-8:
        return None
    quantities = np.array([lot_floor(q * capacity_scale, metadata[c]['szDecimals']) for c, q in zip(coins, planned)])
    if np.any(quantities * oracle < cfg['minimum_leg_notional']):
        return None
    ratios = quantities / planned
    if min(ratios) / max(ratios) < 1 - cfg['balanced_fill_tolerance']:
        return None
    entry, exits, roundtrip_now = [], [], []
    for c, q, side in zip(coins, quantities, signs):
        snap = snapshots[c]; levels = snap['book']['levels']
        en = book_walk(levels[1 if side > 0 else 0], q, fraction)
        ex_now = book_walk(levels[0 if side > 0 else 1], q, fraction)
        if en is None or ex_now is None:
            return None
        expected_mid = snap['mid'] + cfg['basis_convergence'] * (snap['oracle'] - snap['mid'])
        ex = expected_mid + ex_now - snap['mid']
        entry.append(en); exits.append(ex); roundtrip_now.append(ex_now)
    entry, exits, roundtrip_now = map(np.array, (entry, exits, roundtrip_now))
    qty = quantities * signs
    rates = np.array([float(snapshots[c]['ctx']['funding']) for c in coins])
    carry = float(-qty @ (oracle * rates) * horizon)
    carry = carry * cfg['positive_funding_haircut'] if carry > 0 else carry
    fee = float(quantities @ (entry + exits) * scenario['fee_bps'] / 10000)
    price = float(qty @ (exits - entry))
    notional = float(quantities @ entry)
    instant = float(qty @ (roundtrip_now - entry) - quantities @ (entry + roundtrip_now) * scenario['fee_bps'] / 10000)
    return dict(coins=list(coins), beta=float(beta), direction=int(direction), horizon_hours=int(horizon),
                target_qty=qty.tolist(), entry_preview=entry.tolist(), exit_forecast=exits.tolist(),
                gross_notional=notional, capacity_scale=float(capacity_scale),
                predicted_price_pnl=price, predicted_funding=carry, predicted_fees=fee,
                predicted_net_pnl=price + carry - fee, predicted_net_bps=(price + carry - fee) / notional * 10000,
                instant_roundtrip_bps=instant / notional * 10000,
                funding_is_forecast=True, future_exit_depth_is_unknown=True)


@dataclass
class PairExecution:
    paper: Paper
    config: dict
    state: dict | None = None
    events: list = field(default_factory=list)

    def enter(self, proposal, now):
        if self.state is not None or self.paper.pending or any(abs(q) > 1e-10 for q in self.paper.qty.values()):
            raise ValueError('Pair account is not flat or previous orders are unresolved')
        if proposal['predicted_net_bps'] <= self.config['entry_buffer_bps']:
            raise ValueError('No cost-adjusted entry edge')
        self.state = dict(phase='entering', proposal=deepcopy(proposal), requested_at=now,
                          deadline=now + self.config['maximum_unbalanced_seconds'] * 1000,
                          expires=now + proposal['horizon_hours'] * 3600000,
                          repair=0)
        self.paper.request_target({c: float(q) for c, q in zip(proposal['coins'], proposal['target_qty'])}, now,
                                  self.state['deadline'], f'pair:{now}')
        self.events.append(dict(time=now, event='entry_intent', proposal=proposal))

    def close(self, now, reason):
        if self.state is None:
            return
        self.state['close_requested'] = reason
        # Resolve in-flight fills before constructing reduce-only targets.
        if self.paper.pending:
            return
        if self.state['phase'] != 'closing':
            self.state.update(phase='closing', exit_deadline=now + self.config['maximum_exit_seconds'] * 1000)
            self.events.append(dict(time=now, event='exit_intent', reason=reason))
        elif now < self.state.get('next_exit_attempt', 0):
            return
        self.state['next_exit_attempt'] = now + 30000
        self.paper.tasks = {}
        self.paper.request_target({c: 0. for c in self.state['proposal']['coins']}, now,
                                  self.state['exit_deadline'], f'exit:{now}')

    def tick(self, now, books, contexts):
        # The caller updates depth from each observed book before this method.
        self.paper.apply_pending(now, books)
        state = self.state
        if state is not None and not self.paper.pending:
            coins = state['proposal']['coins']
            actual = np.array([self.paper.qty.get(c, 0.) for c in coins])
            if state.get('close_requested') and state['phase'] != 'closing':
                self.close(now, state['close_requested'])
            elif state['phase'] in ('entering', 'balancing'):
                submitted = any(a.get('submitted', -1) >= state['requested_at'] for a in self.paper.attempts)
                balanced, target = balanced_inventory(actual, state['proposal']['target_qty'], self.config['balanced_fill_tolerance'])
                if balanced:
                    self.paper.tasks = {}; state.update(phase='holding', actual_qty=actual.tolist(), opened=now)
                    self.events.append(dict(time=now, event='balanced_open', qty=actual.tolist()))
                elif submitted and not np.any(actual):
                    self.paper.tasks = {}; self.state = None
                    self.events.append(dict(time=now, event='entry_unfilled'))
                elif submitted and (now >= state['deadline'] or not np.all(target)):
                    self.close(now, 'unhedged_abort')
                elif submitted:
                    if now < state.get('next_repair_attempt', 0):
                        self.paper.tick(now, books, contexts)
                        return
                    state['phase'] = 'balancing'; state['repair'] += 1
                    state['next_repair_attempt'] = now + 30000
                    self.paper.tasks = {}
                    self.paper.request_target({c: float(q) for c, q in zip(coins, target)}, now, state['deadline'], f'repair:{now}')
                    self.events.append(dict(time=now, event='reduce_imbalance', qty=actual.tolist(), target=target.tolist()))
            elif state['phase'] == 'holding' and now >= state['expires']:
                self.close(now, 'horizon')
            elif state['phase'] == 'closing':
                if not np.any(abs(actual) > 1e-10):
                    self.paper.tasks = {}; self.state = None
                    self.events.append(dict(time=now, event='flat_confirmed'))
                elif now > state['exit_deadline']:
                    if not state.get('failed'):
                        self.paper.fatal.append('Pair exit deadline expired with residual inventory')
                        self.events.append(dict(time=now, event='unresolved_exit', qty=actual.tolist()))
                        state['failed'] = True
                else:
                    self.close(now, state['close_requested'])
        self.paper.tick(now, books, contexts)


def broker(metadata, scenario):
    protocol = dict(candidate=dict(attempts_per_coin_phase=1, retry_seconds=30, initial_quote_drift_bps=10),
                    common=dict(quote_limit_bps=10, asset_notional_limit=150000, minimum_order_notional=10))
    return Paper('baseline', protocol, scenario, metadata)
