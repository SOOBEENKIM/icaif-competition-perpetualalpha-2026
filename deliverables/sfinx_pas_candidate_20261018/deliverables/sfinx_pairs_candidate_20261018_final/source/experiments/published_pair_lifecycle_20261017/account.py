"""Received-book execution with exact owner limits and consumed-depth accounting.

This is an additive correction to the frozen research adapter. It does not
modify a live strategy, an earlier experiment, or any reserved capture.
"""
import bisect
import copy
import numpy as np
from experiments.shared_execution_pairs_20261017.account_v2 import (
    SharedAccount, difference, sum_targets,
)
from experiments.basis_carry_20261017.probe import asof, sweep
from replay_lab.common_comparison.engine import rounded


def arrival(stream, earliest, submitted, cfg):
    i = bisect.bisect_left(stream['book_times'], earliest)
    while i < len(stream['book_times']):
        at = stream['book_times'][i]
        if at > earliest + cfg['future_exit_observation_grace_ms']:
            break
        snapshot = asof(stream, at, cfg)
        if snapshot is not None and snapshot['book']['time'] >= submitted:
            values = [float(r[k]) for levels in snapshot['book']['levels']
                      for r in levels for k in ('px', 'sz')]
            if np.isfinite(values).all() and all(v >= 0 for v in values):
                return at, snapshot
        i += 1
    return None


class BookAccount(SharedAccount):
    def __init__(self, initial, symbols):
        super().__init__(initial, symbols)
        self.consumed = {}
        self.last_packet_arrival_ms = -1

    def seal(self, intent, p, sc):
        intent = copy.deepcopy(intent)
        n = len(self.physical.qty)
        for name, target in intent['owner_targets'].items():
            if name not in self.owners or len(target) != n or not np.isfinite(target).all():
                raise ValueError('Invalid owner target')
        normalized = {}
        for owner, limits in intent.get('owner_limits', {}).items():
            if owner not in self.owners:
                raise ValueError('Unknown limit owner')
            normalized[owner] = {}
            for key, value in limits.items():
                index = int(key)
                if str(key) != str(index) or not 0 <= index < n:
                    raise ValueError('Invalid limit asset identity')
                name = str(index)
                if name in normalized[owner]:
                    raise ValueError('Duplicate integer/string limit identity')
                if value is not None and (not np.isfinite(value) or value <= 0):
                    raise ValueError('Invalid economic order limit')
                normalized[owner][name] = value
        intent['owner_limits'] = normalized
        return super().seal(intent, p, sc)

    def remaining_levels(self, symbol, book, side, fraction):
        levels = []
        for row in book['levels'][1 if side > 0 else 0]:
            key = (symbol, book['time'], int(side), float(row['px']))
            available = max(0., float(row['sz']) * fraction - self.consumed.get(key, 0.))
            levels.append(dict(px=row['px'], sz=str(available)))
        return levels

    def consume(self, symbol, book, side, quantity, fraction, decimals, limit):
        factor = 10 ** decimals
        for row in self.remaining_levels(symbol, book, side, fraction):
            price = float(row['px'])
            if limit is not None and side * (price - limit) > 1e-10:
                break
            size = min(quantity, np.floor(float(row['sz']) * factor + 1e-9) / factor)
            key = (symbol, book['time'], int(side), price)
            self.consumed[key] = self.consumed.get(key, 0.) + size
            quantity = difference(quantity, size)
            if quantity <= 1e-10:
                return
        raise AssertionError('Filled quantity exceeded remaining displayed depth')

    def execute_books(self, intent, streams, marks, sigma, p, sc, cfg, *, on_time=None):
        submitted = intent['decision_time'] * 1000
        if submitted < self.last_packet_arrival_ms:
            raise ValueError('New packet precedes a prior independent-leg receipt')
        self._take(intent)
        marks = np.asarray(marks, float).copy()
        requested = sum_targets(intent['owner_targets'])
        events, orders = [], []
        for j, symbol in enumerate(self.physical.symbols):
            if all(abs(v[j]) < 1e-12 for v in self.needs(intent).values()):
                continue
            received = arrival(streams[symbol], submitted + sc['latency_ms'], submitted, cfg) if symbol in streams else None
            if received is None:
                external = difference(requested[j], self.physical.qty[j]) != 0
                if external:
                    self.unknown.add(symbol)
                orders.append(dict(symbol=symbol, status='UNKNOWN_ARRIVAL' if external else 'INTERNAL_MARK_UNOBSERVED', filled_qty=None))
                continue
            at, snapshot = received
            events.append((at, j, snapshot))
        if on_time is None and any(at // 3600000 != submitted // 3600000 for at, _, _ in events):
            raise ValueError('Funding callback required across a settlement boundary')
        for at, j, snapshot in sorted(events, key=lambda e: (e[0], e[1])):
            if on_time is not None:
                on_time(at, self)
            self.last_packet_arrival_ms = max(self.last_packet_arrival_ms, at)
            for k, symbol in enumerate(self.physical.symbols):
                current = asof(streams[symbol], at, cfg) if symbol in streams else None
                if current is not None:
                    marks[k] = float(current['ctx'].get('markPx', current['mid']))
            symbol = self.physical.symbols[j]
            self.cross(intent, j, float(snapshot['mid']), at / 1000)
            old = self.physical.qty[j]
            delta = difference(requested[j], old)
            if not delta:
                continue
            side = np.sign(delta)
            decimals = p['execution']['decimals'][j]
            request = abs(rounded(abs(delta), decimals))
            full_close = old * side < 0 and request == abs(old)
            if request * marks[j] < p['execution']['minimum_order_value'] and not full_close:
                orders.append(dict(symbol=symbol, status='BELOW_MINIMUM', filled_qty=0., arrival_ms=at))
                continue
            limit = self.net_limit(intent, j, side)
            levels = self.remaining_levels(symbol, snapshot['book'], side, sc['depth_fraction'])
            got, price = sweep(levels, request, 1., side, decimals, limit)
            if got:
                got = self.feasible_size(j, side, got, price, marks, sigma, p, sc['fee_bps'])
                if got:
                    price = sweep(levels, got, 1., side, decimals, limit)[1]
                    self.consume(symbol, snapshot['book'], side, got, sc['depth_fraction'], decimals, limit)
                    nf = len(self.physical.fills)
                    self.physical_fill(j, side * got, price, sc['fee_bps'], at / 1000,
                        decision_time=intent['decision_time'], source_available=intent['source_available'],
                        source_close=intent['source_close'], risk_forced=intent['risk_forced'])
                    for fill in self.physical.fills[nf:]:
                        self.allocate(intent, fill, sc['fee_bps'])
            orders.append(dict(symbol=symbol, status='FULL' if got >= request - 1e-10 else 'PARTIAL_OR_UNFILLED',
                               filled_qty=got, requested_qty=request, arrival_ms=at, limit=limit))
            self.reconcile(marks)
        return dict(orders=orders, unknown_symbols=sorted(self.unknown), orders_sent=0)
