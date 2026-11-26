"""Independent, fixed-endpoint executable-depth audit of paper portfolios.

Does not submit orders or edit a frozen candidate/evaluator. Observed books are
conditional liquidity, not guaranteed fills. Unclosed quantities invalidate a
full liquidation valuation; they are never silently valued at a free exit.
"""
import copy
import math
import numpy as np
from experiments.basis_carry_20261017.probe import asof, sweep
from .book_replay import BookVenue


def consume(capacity, quantity, side, limit):
    """Reproduce already recorded paper consumption at eligible price levels."""
    remaining = float(quantity)
    for px in sorted(capacity, reverse=side < 0):
        if side * (px - limit) > 1e-10:
            break
        displayed, available = capacity[px]
        taken = min(remaining, available)
        capacity[px] = (displayed, available - taken)
        remaining -= taken
        if remaining <= 1e-10:
            return
    if remaining > 1e-8:
        raise ValueError('Recorded paper fill exceeds reconstructed capacity')


def terminal_value(streams, profile, scenario, snapshot, orders=()):
    at = float(snapshot['time'])
    venue = BookVenue(streams, profile, scenario, at, at, [])
    # Honor the evaluator's consumption/replenishment accounting. Starting
    # inventory predates capture and is already reflected in observed depth.
    for order in sorted(orders, key=lambda x: x['arrival']):
        if order['arrival'] > at:
            continue
        side = order['side']
        cap = venue.depth_at(order['symbol'], int(order['arrival'] * 1000))[1 if side > 0 else 0]
        consume(cap, order['filled'], side, order['limit'])
    legs = []
    adjustment = 0.
    complete = True
    for j, symbol in enumerate(profile['trading']['symbols']):
        qty, mark = float(snapshot['qty'][j]), float(snapshot['marks'][j])
        if abs(qty) < 1e-12:
            continue
        quote = asof(streams[symbol], int(at * 1000), venue.cfg)
        if quote is None:
            complete = False
            legs.append(dict(symbol=symbol, qty=qty, filled=0., residual=qty,
                             reason='missing_fresh_endpoint_book'))
            continue
        side = -1 if qty > 0 else 1
        cap = venue.depth_at(symbol, int(at * 1000))[1 if side > 0 else 0]
        levels = [dict(px=str(px), sz=str(v[1])) for px, v in
                  sorted(cap.items(), reverse=side < 0)]
        filled, px = sweep(levels, abs(qty), 1., side,
                           profile['trading']['execution']['decimals'][j])
        residual = math.copysign(max(0., abs(qty) - filled), qty)
        complete &= abs(residual) < 1e-8
        fee = filled * px * scenario['fee_bps'] / 10000 if filled else 0.
        change = math.copysign(filled, qty) * (px - mark) - fee if filled else 0.
        adjustment += change
        legs.append(dict(symbol=symbol, qty=qty, filled=filled, residual=residual,
                         exit_price=px, mark=mark, fee=fee, adjustment=change,
                         book_received_ms=quote['received']['book']))
    return dict(time=at, complete=bool(complete), legs=legs,
                priced_partial_adjustment=adjustment,
                complete_liquidation_equity=snapshot['equity'] + adjustment if complete else None,
                conditional_observed_depth=True, actual_orders_sent=0)


def value_at(streams, profile, scenario, result, at):
    """Mark both arms at exactly the same time, including intervening fills.

    A sequential order can shift the next sampling tick. Use actual simulated
    arrival milliseconds, not rounded Ledger seconds or nearest-row returns.
    The initial snapshot is before trading; conservation is checked separately.
    """
    first = result['snapshots'][0]
    if at < first['time']:
        raise ValueError('Requested endpoint precedes first valued state')
    successful = [o for o in result['orders'] if o['filled']]
    if len(successful) != len(result['fills']):
        raise ValueError('Cannot unambiguously match fills to paper order arrivals')
    qty = np.array(first['qty'], float)
    shadow = first['equity'] - qty @ np.array(first['marks'])
    fees, funding = first['fees'], first['funding']
    for order, fill in zip(successful, result['fills']):
        if (profile['trading']['symbols'][fill['asset']] != order['symbol'] or
            abs(fill['signed_qty'] - order['side']*order['filled']) > 1e-8 or
            abs(fill['price'] - order['price']) > 1e-8):
            raise ValueError('Fill/order sequence does not reconcile')
        if order['arrival'] <= first['time']:
            raise ValueError('Initial snapshot already contains a replay fill')
        if order['arrival'] <= at:
            qty[fill['asset']] += fill['signed_qty']
            shadow -= fill['signed_qty']*fill['price'] + fill['fee']
            fees += fill['fee']
    for row in result['funding']:
        if first['time'] < row['time']/1000 <= at:
            cash = float(row['delta']['usdc']); shadow += cash; funding += cash
    venue = BookVenue(streams, profile, scenario, at, at, [])
    marks = venue.marks()
    if marks is None:
        raise ValueError('No fresh common endpoint marks')
    return dict(time=at, qty=qty.tolist(), marks=marks.tolist(), equity=float(shadow+qty@marks),
                fees=fees, funding=funding, notional=float(abs(qty*marks).sum()))


def compare_at_common_end(streams, profile, scenario, results, start, end):
    """Require the same exact endpoint in every arm, with causal order costs."""
    output = []
    for arm, r in results.items():
        try:
            a, b = (value_at(streams, profile, scenario, r, at) for at in [start, end])
            # Midpoint and final recorded snapshots independently check the
            # prefix reconstruction without using future quantities at start.
            for sample in [r['snapshots'][len(r['snapshots'])//2], r['snapshots'][-1]]:
                restored = value_at(streams, profile, scenario, r, sample['time'])
                if abs(restored['equity']-sample['equity']) > 1e-6 or not np.allclose(restored['qty'],sample['qty'],atol=1e-8,rtol=0):
                    raise ValueError('Endpoint reconstruction does not conserve the saved ledger')
        except ValueError as error:
            output.append(dict(arm=arm, valid=False, reason=str(error)))
            continue
        entry = terminal_value(streams, profile, scenario, a, r['orders'])
        exit = terminal_value(streams, profile, scenario, b, r['orders'])
        invalid = bool(r['failures'] or r['risk_violations'] or r['unknown'] or
                       r['pending_orders'] or r['fatal'] or r['accounting_error'] > 1e-6)
        complete = entry['complete'] and exit['complete']
        output.append(dict(arm=arm, valid=complete and not invalid,
                           net_pnl=b['equity'] - a['equity'], initial_exit=entry, terminal_exit=exit,
                           liquidation_basis_pnl=(exit['complete_liquidation_equity'] - entry['complete_liquidation_equity']) if complete else None,
                           net_after_terminal_exit=b['equity'] + exit['priced_partial_adjustment'] - a['equity'] if exit['complete'] else None,
                           failures=len(r['failures']), risk_violations=len(r['risk_violations']),
                           accounting_error=r['accounting_error'], actual_orders_sent=0))
    return output
