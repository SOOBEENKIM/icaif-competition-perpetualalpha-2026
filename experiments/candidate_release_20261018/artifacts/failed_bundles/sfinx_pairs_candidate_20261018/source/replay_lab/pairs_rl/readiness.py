"""Fail-closed opportunity audit, recomputed from training data before every fit.

The thresholds are operational floors, not estimates of statistical power. A
successful check permits an experiment; it does not certify a profitable agent.
"""
import numpy as np

from .engine import simulate_rules
from ..common_comparison.models import timestamp


MINIMUM_ROUNDTRIPS = 20
MINIMUM_FORMATIONS = 2
MINIMUM_PER_HALF = 5


def completed_trades(run, formations):
    """Count unique, fully filled two-leg entries that subsequently become flat.

    Abort/flatten cycles are disclosed but excluded. Count inventory, not repeated
    exit requests, hourly observations, or randomized replays of the same trade.
    The simulator prohibits overlapping pairs on shared assets.
    """
    assets = {r['pair']: r['assets'] for f in formations for r in f['rows']}
    snapshots = run['snapshots']
    by_time = {s['time']: i for i, s in enumerate(snapshots)}
    decisions = {d['decision_time']: d for d in run['decisions']}
    partial = {(e['pair'], e['time']) for e in run['events']
               if e['event'] == 'incomplete_pair_flatten_next_hour'}
    entries = [e for e in run['events'] if e['event'] == 'entry_intent']
    completed, rejected, open_entries = [], [], []
    for e in entries:
        t = e['time'] + 3600
        ids = assets[e['pair']]
        decision = decisions.get(e['time'])
        k = by_time.get(t)
        reason = None
        if decision is None or k is None:
            reason = 'no_execution_snapshot'
        else:
            actual = np.array(snapshots[k]['qty'])[ids]
            target = np.array(decision['target_qty'])[ids]
            if (e['pair'], t) in partial:
                reason = 'partial_pair_aborted'
            elif not np.all(actual != 0) or not np.allclose(actual, target, rtol=0, atol=1e-9):
                reason = 'not_both_legs_filled'
        if reason:
            rejected.append(dict(pair=e['pair'], time=t, reason=reason))
            continue
        end = next((s['time'] for s in snapshots[k+1:]
                    if np.all(np.abs(np.array(s['qty'])[ids]) < 1e-10)), None)
        f = [f for f in formations if f['close_index'] is not None and f['time'] <= e['time']]
        formation_time = f[-1]['time'] if f else None
        record = dict(pair=e['pair'], decision_time=e['time'], entry_time=t,
                      exit_time=end, formation_time=formation_time)
        (completed if end is not None else open_entries).append(record)
    return dict(entry_intents=len(entries), completed=completed, rejected=rejected, open_entries=open_entries)


def assess(run, formations):
    trades = completed_trades(run, formations)
    completed = trades['completed']
    midpoint = (run['start'] + run['end']) / 2
    halves = [sum(t['entry_time'] < midpoint for t in completed),
              sum(t['entry_time'] >= midpoint for t in completed)]
    active = len({t['formation_time'] for t in completed if t['formation_time'] is not None})
    gates = dict(
        enough_completed_roundtrips=len(completed) >= MINIMUM_ROUNDTRIPS,
        multiple_entry_formations=active >= MINIMUM_FORMATIONS,
        both_chronological_halves=min(halves) >= MINIMUM_PER_HALF,
        accounting_time_risk=all(run['gates'].values()),
        observed_position_funding=run['metrics']['missing_position_hours'] == 0,
    )
    return dict(passed=all(gates.values()), gates=gates,
                reasons=[k for k, v in gates.items() if not v],
                completed_roundtrips=len(completed), active_entry_formations=active,
                chronological_half_counts=halves, trades=trades, metrics=run['metrics'],
                requirements=dict(completed_roundtrips=MINIMUM_ROUNDTRIPS,
                                  entry_formations=MINIMUM_FORMATIONS, per_half=MINIMUM_PER_HALF),
                interpretation='Minimum experimental data coverage only; not statistical power or profit certification.')


def preflight(market, formations, protocol):
    """Replay the supplied, causal training formations. Never trust a passed JSON."""
    end = market.exact(timestamp(protocol['periods']['train_end']))
    if not formations:
        raise ValueError('training readiness: no formation history')
    indices = [f['close_index'] for f in formations]
    if indices != sorted(set(indices)) or any(i < 1 or i >= end for i in indices):
        raise ValueError('training readiness: unordered or non-training formation')
    # Bind timestamps to actual market indices; callers cannot relabel old trades.
    forms = [{**f, 'time': int(market.times[f['close_index']])} for f in formations]
    run = simulate_rules(market, forms, protocol, int(market.times[indices[0]]), int(market.times[end]))
    return assess(run, forms), run


def require_ready(report):
    if not report['passed']:
        raise ValueError('training readiness failed: ' + ', '.join(report['reasons']) +
                         f"; completed roundtrips={report['completed_roundtrips']}")
