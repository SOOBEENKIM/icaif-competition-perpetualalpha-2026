"""Deterministic public-book replay. This module cannot submit orders."""
from copy import deepcopy
from dataclasses import dataclass, field
import math

import numpy as np

from bot.execution import prepare_limit, rounded_price
from decimal import Decimal as D
from practice_live.live_logic import phase_delta


def lot_floor(value, decimals):
    return float(D(str(max(value, 0))).quantize(D(10) ** -decimals, rounding='ROUND_DOWN'))


class Depth:
    """One independent hypothetical order book, with conservative replenishment."""
    def __init__(self, fraction=1.0):
        self.fraction = fraction
        self.raw, self.free = {}, {}
        self.book_time = {}

    def update(self, coin, book):
        if int(book['time']) <= self.book_time.get(coin, -1):
            return
        self.book_time[coin] = int(book['time'])
        for side, levels in enumerate(book['levels']):
            key = coin, side
            old, free = self.raw.get(key, {}), self.free.get(key, {})
            new = {float(x['px']): float(x['sz']) * self.fraction for x in levels}
            available = {}
            for price, size in new.items():
                available[price] = min(size, max(0., free.get(price, 0.) + size - old[price])) if price in old else size
            self.raw[key], self.free[key] = new, available

    def consume(self, coin, buy, size, limit, decimals=8):
        levels = self.free.get((coin, 1 if buy else 0), {})
        result = []
        for price in sorted(levels, reverse=not buy):
            if (buy and price > limit) or (not buy and price < limit):
                continue
            amount = lot_floor(min(size, levels[price]), decimals)
            if amount > 1e-12:
                levels[price] -= amount
                size -= amount
                result.append((amount, price))
            if size <= 1e-12:
                break
        return result


@dataclass
class Paper:
    name: str
    protocol: dict
    scenario: dict
    metadata: dict
    cash: float = 1_000_000.
    qty: dict = field(default_factory=dict)
    fills: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    funding_events: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    tasks: dict = field(default_factory=dict)
    fees: float = 0.
    funding: float = 0.
    fatal: list = field(default_factory=list)
    equity_series: list = field(default_factory=list)

    def __post_init__(self):
        self.depth = Depth(self.scenario['visible_depth_fraction'])

    def equity(self, contexts, price='markPx'):
        return self.cash + sum(q * float(contexts[c]['ctx'][price]) for c, q in self.qty.items() if abs(q) > 1e-12)

    def request_target(self, target, now, deadline, round_id, terminal=False):
        if self.pending:
            self.fatal.append('Target replaced with unresolved simulated order')
            return
        self.tasks = {}
        for coin, desired in target.items():
            self.tasks[coin] = {'target': desired, 'round': round_id, 'created': now, 'deadline': deadline,
                                'anchor': {}, 'attempts': {'reduce': 0, 'increase': 0},
                                'next': {'reduce': now, 'increase': now}, 'terminal': terminal}

    def apply_pending(self, now, books):
        keep = []
        for order in self.pending:
            if now < order['arrives']:
                keep.append(order)
                continue
            coin = order['coin']
            book = books.get(coin)
            if book is None or now - book['received_ms'] > 15000 or not -2000 <= now - book['data']['time'] <= 15000:
                self.attempts.append({**order, 'status': 'unfilled_stale_arrival_book'})
                continue
            amount = order['size']
            if order['phase'] == 'reduce':
                amount = min(amount, abs(self.qty.get(coin, 0.)))
            matches = self.depth.consume(coin, order['buy'], amount, order['limit'], self.metadata[coin]['szDecimals'])
            filled = 0.
            for q, price in matches:
                signed = q if order['buy'] else -q
                fee = q * price * self.scenario['fee_bps'] / 10000
                self.cash -= signed * price + fee
                self.qty[coin] = self.qty.get(coin, 0.) + signed
                if abs(self.qty[coin]) < 1e-10:
                    self.qty[coin] = 0.
                self.fees += fee
                filled += q
                self.fills.append({**order, 'time': now, 'qty': signed, 'px': price, 'fee': fee})
            self.attempts.append({**order, 'filled': filled, 'status': 'filled' if abs(filled - amount) < 1e-9 else 'partial_or_unfilled'})
        self.pending = keep

    def tick(self, now, books, contexts, active=None):
        self.apply_pending(now, books)
        maximum = 1 if self.name == 'baseline' else self.protocol['candidate']['attempts_per_coin_phase']
        common = self.protocol['common']
        for phase in ('reduce', 'increase'):
            for coin, task in self.tasks.items():
                if self.fatal or now > task['deadline'] or now < task['next'][phase] or task['attempts'][phase] >= maximum:
                    continue
                if any(o['coin'] == coin for o in self.pending):
                    continue
                current = self.qty.get(coin, 0.)
                delta = phase_delta(current, task['target'], phase)
                if abs(delta) < 1e-10:
                    continue
                b, c = books.get(coin), contexts.get(coin)
                if not b or not c or max(now - b['received_ms'], now - c['received_ms']) > 15000 or not -2000 <= now - b['data']['time'] <= 15000:
                    continue
                levels = b['data']['levels']
                if not levels[0] or not levels[1]:
                    continue
                buy = delta > 0
                best = float(levels[1 if buy else 0][0]['px'])
                if float(levels[0][0]['px']) >= float(levels[1][0]['px']):
                    continue
                sign = 1 if buy else -1
                limit = best * (1 + sign * common['quote_limit_bps'] / 10000)
                task['anchor'].setdefault(phase, best)
                if self.name != 'baseline':
                    initial = task['anchor'][phase] * (1 + sign * self.protocol['candidate']['initial_quote_drift_bps'] / 10000)
                    limit = min(limit, initial) if buy else max(limit, initial)
                decimals = self.metadata[coin]['szDecimals']
                limit = float(rounded_price(D(str(limit)), decimals, buy))
                visible = sum(float(x['sz']) for x in levels[1 if buy else 0]
                              if (buy and float(x['px']) <= limit) or (not buy and float(x['px']) >= limit))
                size = min(abs(delta), visible, common['asset_notional_limit'] / limit)
                if phase == 'increase':
                    # Paper account uses its own 1x collateral; actual wallet capacity
                    # is recorded separately and is not transferable to a hypothetical account.
                    eq = self.equity(contexts)
                    used = sum(abs(q) * float(contexts[c]['ctx']['markPx']) for c, q in self.qty.items() if q)
                    size = min(size, max(0., eq - used) / limit)
                    if active is not None:
                        capacity = active.get(coin)
                        if not capacity or now - capacity['received_ms'] > 15000:
                            continue
                        size = min(size, float(capacity['data']['maxTradeSzs'][0 if buy else 1]))
                size = lot_floor(size, decimals)
                task['next'][phase] = now + self.protocol['candidate']['retry_seconds'] * 1000
                task['attempts'][phase] += 1
                if size * limit < common['minimum_order_notional']:
                    self.attempts.append({'coin': coin, 'phase': phase, 'time': now, 'status': 'no_executable_size'})
                    continue
                # Exercise the exact production price/lot/notional validation seam.
                prepare_limit(coin, buy, size, limit, sz_decimals=decimals, mark_price=c['ctx']['markPx'],
                              max_notional=str(common['asset_notional_limit']), tif='Ioc', reduce_only=phase == 'reduce')
                ident = f"{task['round']}:{coin}:{phase}:{task['attempts'][phase]}"
                self.pending.append({'coin': coin, 'phase': phase, 'buy': buy, 'size': size, 'limit': limit,
                                     'submitted': now, 'arrives': now + self.scenario['latency_seconds'] * 1000,
                                     'id': ident, 'anchor': task['anchor'][phase]})

    def charge_funding(self, when, rate_map, contexts):
        for coin, q in self.qty.items():
            if abs(q) < 1e-12:
                continue
            c = contexts.get(coin)
            if coin not in rate_map or not c or when - c['received_ms'] > 15000:
                self.fatal.append(f'Missing funding/oracle at {when}:{coin}')
                continue
            amount = -q * float(c['ctx']['oraclePx']) * rate_map[coin]
            self.cash += amount
            self.funding += amount
            self.funding_events.append({'time': when, 'coin': coin, 'qty': q, 'payment': amount})

    def snapshot(self, now, contexts):
        held = [c for c, q in self.qty.items() if abs(q) > 1e-12]
        if any(c not in contexts or now - contexts[c]['received_ms'] > 15000 for c in held):
            return None
        mark, oracle = self.equity(contexts), self.equity(contexts, 'oraclePx')
        gross = sum(abs(q) * float(contexts[c]['ctx']['markPx']) for c, q in self.qty.items() if q) / mark
        notionals = [q * float(contexts[c]['ctx']['markPx']) for c, q in self.qty.items() if q]
        net = abs(sum(notionals)) / mark
        if (not np.isfinite([mark, oracle, gross]).all() or mark <= 0 or gross > .6 + 1e-9
                or net > .4 + 1e-9 or any(abs(v) > min(.15 * mark, 150000) + 1e-6 for v in notionals)):
            self.fatal.append('Invalid equity or gross risk breach')
        row = {'time': now, 'mark_equity': mark, 'oracle_equity': oracle, 'gross': gross}
        self.equity_series.append(row)
        return row

    def result(self, contexts):
        row = self.equity_series[-1]
        eq = [x['mark_equity'] for x in self.equity_series]
        peak, drawdown = 1_000_000., 0.
        for value in eq:
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
        independent_cash = 1_000_000. - sum(f['qty'] * f['px'] + f['fee'] for f in self.fills) + sum(f['payment'] for f in self.funding_events)
        assert math.isclose(independent_cash, self.cash, abs_tol=1e-6)
        ids = [x['id'] for x in self.attempts if 'id' in x]
        assert len(ids) == len(set(ids))
        return {'arm': self.name, 'mark_net_pnl': row['mark_equity'] - 1_000_000.,
                'oracle_net_pnl': row['oracle_equity'] - 1_000_000., 'fees': self.fees, 'funding': self.funding,
                'return_percent': (row['mark_equity'] / 1_000_000. - 1) * 100,
                'max_drawdown_percent': drawdown * 100,
                'fill_count': len(self.fills), 'order_count': len(ids),
                'pending_orders': len(self.pending), 'residual_positions': {c: q for c, q in self.qty.items() if abs(q) > 1e-10},
                'flat': all(abs(q) < 1e-10 for q in self.qty.values()), 'fatal': self.fatal,
                'daily_volatility_percent': None}
