"""Preserve the frozen policy while attaching historical PAS quotes.

No network, key loading, service control, checkpoint writes, or order client.
The caller supplies public archive streams and fresh execution streams. History
only supplies the three registered feature points; current execution quotes
remain those supplied by the existing worker.
"""
import copy
import math

import numpy as np

from candidate_handoff.migration import wire
from experiments.basis_carry_20261017.probe import asof
from production_replay_20261025.quote_guard import QuoteAccount
from production_replay_20261025.quote_history import QuoteHistoryPortfolio


def attach_history(current, archive, symbols, source, now, cfg):
    if (not all(math.isfinite(v) for v in (source, now))
            or source % 3600 or source > now):
        raise ValueError('A completed source hour is required')
    if set(current) != set(symbols):
        raise ValueError('Current quote universe differs from the policy')
    result = copy.deepcopy(current)
    evidence = []
    for symbol in symbols:
        live = current[symbol]
        stamps = []
        for field in ('book', 'ctx'):
            rows = live[field]
            times = live[field + '_times']
            if not rows or times != [r[0] for r in rows] or times != sorted(times):
                raise ValueError('Malformed current quote stream')
            if times[-1] > now * 1000 or times[0] <= source * 1000:
                raise ValueError('Fresh worker observations must follow the source boundary')
            stamps += times
        fresh_start = min(stamps)
        s = result[symbol]
        old = archive.get(symbol)
        added = dict(book=[], ctx=[])
        resets = []
        for i, point in enumerate((source - 86400, source - 10800, source)):
            q = asof(old, point * 1000, cfg) if old is not None else None
            row = dict(symbol=symbol, point=point, available=q is not None)
            if q is not None:
                # Both observations receive the same isolated epoch. Resets
                # prevent one feature point borrowing the other point's half.
                epoch = -i - 1
                at = min(q['received'].values())
                resets.append(at)
                for field in ('book', 'ctx'):
                    added[field].append((q['received'][field], epoch,
                                         copy.deepcopy(q[field])))
                row.update(mid=q['mid'], received=copy.deepcopy(q['received']))
            evidence.append(row)
        # A missing fresh execution observation must never borrow historical
        # depth/context. It still fails the existing as-of validation.
        s['resets'] = sorted(set(s.get('resets', []) + resets + [fresh_start]))
        for field in ('book', 'ctx'):
            s[field] = added[field] + s[field]
            s[field + '_times'] = [r[0] for r in s[field]]
        before, after = asof(live, now * 1000, cfg), asof(s, now * 1000, cfg)
        if before != after:
            raise ValueError('Attaching history changed the executable observation')
    return result, evidence


def prepare_unsigned(seed):
    a = seed.account
    if any(getattr(a, k, None) is not None for k in
           ('executor', 'checkpoint', 'collect_fills', 'refresh_market')) or seed.settlement is not None:
        raise ValueError('Use a sanitized unsigned copy, never a connected portfolio')
    if seed.fatal or seed.halted or a.pending_orders or a.unknown:
        raise ValueError('Unresolved state cannot be reset by strategy migration')
    if seed.pair is not None:
        raise ValueError('Active pair migration was not covered by this frozen trial')
    p = copy.deepcopy(seed)
    p.__class__ = QuoteHistoryPortfolio
    p.quote_mode = 'quote_inputs'
    p.base_task = None
    p.account.__class__ = QuoteAccount
    p.account.frozen_side_prices = None
    for session in p.sessions:
        session.account = p.account
        session.owner = p.account.owners['pair']
    for name in ('physical', 'reserved'):
        np.testing.assert_array_equal(getattr(a, name).qty, getattr(p.account, name).qty)
        assert getattr(a, name).shadow_cash == getattr(p.account, name).shadow_cash
    return p


def wire_staged(portfolio, bridge, executor, checkpoint, collect_fills, refresh_market):
    """Explicit dependency wiring; callers of this module never acquire keys.

    The original wire() resets the account type. Reapply the same price guard
    after every connection, including reconnection; preserve pending task data.
    """
    if not isinstance(portfolio, QuoteHistoryPortfolio) or portfolio.quote_mode != 'quote_inputs':
        raise ValueError('Unexpected staged candidate')
    wire(portfolio, bridge, executor, checkpoint, collect_fills, refresh_market)
    portfolio.account.__class__ = QuoteAccount
    portfolio.account.frozen_side_prices = None
    return portfolio
