"""Margin-only integer leverage selection; never changes any order/target.

Own and teammate virtual margin reservations are calculated separately even if
their positions offset physically. Shared occupied markets retain their setting.
The exchange's fresh maxTradeSzs remains authoritative after setting updates.
"""
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from .book import ReconciliationError

MODE = 'minimum_required_margin_v1'
DEFAULT = dict(mode=MODE, maximum_leverage=3, margin_fraction=0.8,
               change_shared_market=False, resize_positions=False)


def validate_policy(policy):
    if (policy.get('mode') != MODE or policy.get('maximum_leverage') != 3
            or policy.get('margin_fraction') != 0.8
            or policy.get('change_shared_market') is not False
            or policy.get('resize_positions') is not False):
        raise ReconciliationError('Unrecognized margin-only leverage policy')
    return {k: policy[k] for k in DEFAULT}


def load_policy(directory):
    directory = Path(directory)
    contract_path = directory / 'team_execution_contract.json'
    required = json.loads(contract_path.read_text()).get('required_leverage_policy') if contract_path.exists() else None
    path = directory / 'leverage_policy.json'
    if not path.exists():
        if required:
            raise ReconciliationError('Required leverage policy is missing')
        return None
    policy = validate_policy(json.loads(path.read_text()))
    if required != MODE:
        raise ReconciliationError('Leverage policy lacks matching execution contract')
    return policy


def audit_writer(directory):
    """Durably record intent before a setting change and confirmation after it."""
    directory = Path(directory)
    def record(event):
        from competition_live.runtime import json_save
        event = dict(at=time.time(), **event)
        payload = json.dumps(event, allow_nan=False) + '\n'
        fd = os.open(directory / 'leverage_events.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        json_save(directory / 'leverage_status.json', event)
    return record


def leverage_value(value):
    if (not isinstance(value, dict) or value.get('type') != 'cross'
            or isinstance(value.get('value'), bool) or not isinstance(value.get('value'), int)
            or value['value'] < 1):
        raise ReconciliationError('Automatic leverage requires an existing cross margin setting')
    return value['value']


def choose(*, plan, book, marks, leverages, exchange_max, policy):
    """Select one market's minimum feasible integer, with other settings fixed.

    For an existing margin-budget breach, a non-worsening reservation remains
    eligible. This avoids obstructing risk reductions merely because a separate
    sleeve was already over its budget. Physical venue capacity is checked later.
    """
    policy = validate_policy(policy)
    if set(book.owners) != {'ours', 'teammate'} or plan['owner'] != 'ours':
        raise ReconciliationError('Unexpected owner in leverage policy')
    book.validate_plan(plan)
    n = len(book.symbols); marks = np.asarray(marks, float); lev = np.asarray(leverages, float)
    if (marks.shape != (n,) or lev.shape != (n,) or not np.isfinite(marks).all()
            or not np.isfinite(lev).all() or np.any(marks <= 0) or np.any(lev < 1)
            or np.any(lev != np.floor(lev))):
        raise ReconciliationError('Invalid margin inputs')
    if (isinstance(exchange_max, bool) or not isinstance(exchange_max, int) or exchange_max < 1):
        raise ReconciliationError('Invalid exchange leverage ceiling')
    j = book.symbols.index(plan['coin'][4:]); current = int(lev[j])
    if current > exchange_max:
        raise ReconciliationError('Observed leverage exceeds exchange maximum')
    delta = float(plan['delta']); limit = float(plan['price'])
    if not np.isfinite([delta, limit]).all() or limit <= 0:
        raise ReconciliationError('Invalid pending order')
    old = np.stack([book.owners[k].qty for k in ('ours', 'teammate')])
    after = old.copy(); after[0, j] += delta
    eq = np.array([book.owners[k].equity(marks) for k in ('ours', 'teammate')])
    # Reserve conservative paid slippage and incumbent's 9bp fee; never credit a
    # hypothetical favorable fill before it happens. Actual preflight checks fees.
    eq[0] -= max(0., delta * (limit - marks[j])) + abs(delta) * limit * .0009
    if not np.isfinite(eq).all() or np.any(eq <= 0):
        raise ReconciliationError('Nonpositive virtual sleeve equity')
    valuation = marks.copy(); valuation[j] = max(marks[j], limit)
    before_margin = (abs(old) * valuation / lev).sum(axis=1)
    budget = np.maximum(eq * policy['margin_fraction'], before_margin)
    locked = bool(abs(old[1, j]) > 1e-12)
    candidates = [current] if locked else list(range(1, min(policy['maximum_leverage'], exchange_max) + 1))
    for level in candidates:
        selected = lev.copy(); selected[j] = level
        margin = (abs(after) * valuation / selected).sum(axis=1)
        if np.all(margin <= budget + 1e-8):
            return dict(mode=MODE, coin=plan['coin'], selected_leverage=int(level),
                        observed_leverage=current, changed=bool(level != current),
                        shared_market_locked=locked, margin_budget=budget.tolist(),
                        margin_after=margin.tolist(), owner_equity=eq.tolist(),
                        own_actual_gross_leverage=float(abs(after[0]) @ marks / eq[0]),
                        physical_margin_estimate=float(abs(after.sum(axis=0)) @ (valuation / selected)),
                        delta_unchanged=delta, cloid=plan['cloid'], book_version=book.version,
                        reason='preserve_shared_setting' if locked else 'minimum_required_margin',
                        resized_positions=False)
    raise ReconciliationError('No permitted leverage fits separate owner margin budgets; target not resized')


def decide(plan, book, active, state, meta, contexts, public, policy):
    coins = ['pas:' + s for s in book.symbols]
    if ([m['name'] for m in meta['universe']] != coins or len(contexts) != len(coins)):
        raise ReconciliationError('Market identity/order changed')
    positions = {r['position']['coin']: r['position'] for r in state['positions']['assetPositions']}
    lev = []
    for j, coin in enumerate(coins):
        if coin == plan['coin']:
            value = active.get('leverage')
        elif coin in positions:
            value = positions[coin].get('leverage')
        elif any(abs(o.qty[j]) > 1e-12 for o in book.owners.values()):
            # A physically net-flat market can still have two exposed owners.
            value = public.post({'type': 'activeAssetData', 'user': public.profile['team_wallet'], 'coin': coin}).get('leverage')
        else:
            value = {'type': 'cross', 'value': 1}  # zero inventory, contributes zero margin
        lev.append(leverage_value(value))
    j = coins.index(plan['coin']); spec = meta['universe'][j]
    if spec.get('onlyIsolated') or spec.get('isDelisted') or spec.get('marginMode') in ('noCross', 'strictIsolated'):
        raise ReconciliationError('Unsupported margin mode')
    maximum = spec['maxLeverage']
    tables = dict(meta.get('marginTables', [])); table = tables.get(spec.get('marginTableId'))
    if table:
        # Use the strictest published tier instead of assuming the first tier
        # holds at the post-order size. This can only reduce the allowed choice.
        tiers = table.get('marginTiers', [])
        if not tiers: raise ReconciliationError('Empty margin tiers')
        maximum = min([maximum] + [r['maxLeverage'] for r in tiers])
    return choose(plan=plan, book=book, marks=[float(c['markPx']) for c in contexts],
                  leverages=lev, exchange_max=maximum, policy=policy)
