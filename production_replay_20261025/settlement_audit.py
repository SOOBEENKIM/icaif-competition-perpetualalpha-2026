"""Reconstruct per-unit settlement cash from public account funding records.

Labels only: these observations must never enter the earlier decision stream.
Intersect rounding intervals instead of assuming a displayed oracle is exact.
"""
import json
import math
import time
from datetime import datetime, timezone
import numpy as np
from .protocol import ROOT, OUT, save


def cash_units(records, tolerance=1e-6):
    groups = {}
    seen = set()
    for wallet, row in records:
        d = row['delta']; q = float(d.get('szi', 0.)); coin = d.get('coin', '')
        if not q or not coin.startswith('pas:'):
            continue
        key = (coin, row['time'] // 3600000)
        identity = (wallet, row['time'], coin)
        if identity in seen:
            continue
        seen.add(identity)
        groups.setdefault(key, []).append((q, float(d['usdc']), float(d['fundingRate'])))
    units = {}; rejected = []
    for key, rows in groups.items():
        q, cash, rate = np.asarray(rows).T
        ends = np.array([(min((c-tolerance)/s, (c+tolerance)/s),
                          max((c-tolerance)/s, (c+tolerance)/s)) for s,c in zip(q,cash)])
        lo, hi = float(ends[:,0].max()), float(ends[:,1].min())
        if lo > hi or np.ptp(rate) > 1e-12:
            rejected.append(dict(coin=key[0], hour=key[1], reason='inconsistent_settlement_records'))
            continue
        value = float(np.clip(q @ cash / (q @ q), lo, hi))
        bound = max(value-lo, hi-value)
        if rate[0] and -value/rate[0] <= 0:
            rejected.append(dict(coin=key[0], hour=key[1], reason='nonpositive_inferred_oracle'))
            continue
        units[key] = dict(coin=key[0], hour=key[1], cash_per_unit=value,
                          cash_per_unit_lower=lo, cash_per_unit_upper=hi,
                          maximum_unit_error=bound, observations=len(rows), rate=float(rate[0]))
    return list(units.values()), rejected


def path_cash_sensitivity(result, seed_fills, units, symbols, start, end, receipt_multiplier,
                          expected_settlements):
    """Reprice the recorded path; explicitly NOT a fresh policy rerun."""
    all_fills = [*seed_fills, *result['fills']]
    total = 0.; error = 0.; events = []; missing = []
    available = {(u['coin'], u['hour']) for u in units}
    for f in expected_settlements:
        at = f['time']/1000
        if not start < at <= end:
            continue
        j = symbols.index(f['coin'][4:])
        qty = sum(x['signed_qty'] for x in all_fills if x['asset'] == j and x['time'] < at)
        if qty and float(f['fundingRate']) and (f['coin'], f['time']//3600000) not in available:
            missing.append(dict(coin=f['coin'], time=at, qty=qty))
    for u in units:
        at = u['hour'] * 3600
        if not start < at <= end:
            continue
        j = symbols.index(u['coin'][4:])
        qty = sum(f['signed_qty'] for f in all_fills if f['asset'] == j and f['time'] < at)
        cash = qty * u['cash_per_unit']
        if cash > 0:
            cash *= receipt_multiplier
        total += cash; error += abs(qty) * u['maximum_unit_error']
        events.append(dict(coin=u['coin'], time=at, qty=qty, cash=cash))
    return dict(funding=total if not missing else None, observed_partial_funding=total,
                missing=missing, complete=not missing, maximum_rounding_error=error, events=events,
                path_held_fixed=True, policy_feedback_replayed=False)


def collect_after_end():
    end = 1792915500
    while time.time() < end + 65:
        save(OUT/'settlement_check/status.json', dict(state='waiting_for_fixed_end', at=time.time(), orders_sent=0))
        time.sleep(min(30, max(1, end+65-time.time())))
    from .retrospective import request
    wallets = json.loads((OUT/'reserved_eight_hours/protocol.json').read_text())['wallets_for_funding_cash_only']
    dest = OUT/'settlement_check'; records = []
    for wallet in wallets:
        rows = request(dict(type='userFunding', user=wallet, startTime=1792897200000, endTime=end*1000+999), dest/'raw'/f'{wallet}.json')
        if len(rows) >= 500:
            raise ValueError('Funding pagination required before using these observations')
        records.extend((wallet, r) for r in rows)
    units, rejected = cash_units(records)
    save(dest/'cash_units.json', units); save(dest/'rejected.json', rejected)
    save(dest/'status.json', dict(state='complete', at=time.time(), units=len(units), rejected=len(rejected), orders_sent=0))


if __name__ == '__main__':
    collect_after_end()
