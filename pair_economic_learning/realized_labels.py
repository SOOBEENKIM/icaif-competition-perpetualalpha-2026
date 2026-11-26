"""Counterfactual labels from the unchanged paid, independent-leg executor.

Each experiment starts flat, permits exactly one source opportunity and then
disables new sources. Labels mature after 96 hours, including 24 hours for
closing a 72-hour trade. They are conditional candle simulations, not fills.
"""
import copy
import numpy as np
from auto_pair_selection.selection import IDS, PAIRS
from competition_live.profile import BUNDLE, load_profile
from pair_economic_learning.study import OUT as FEATURES

HOURS = 96


def state(d, f):
    t = d['times']; n = len(t)
    assert np.array_equal(t, f['times'][:n])
    available = f['valid'][:n].copy()
    mu = f['ar'][:n].copy()
    return dict(times=t, dense=np.array(True), pairs=np.asarray(IDS),
        signal_times=np.broadcast_to(t[:, None], mu.shape).copy(),
        source_prices=d['reference_close'].copy(), forecast_times=t.copy(),
        beta=f['beta'][:n].copy(), vol=f['vol'][:n].copy(), mu=mu,
        directions=available.astype(int), economic_available=available,
        economic_action=np.where(available & (abs(mu) > .0038),
            np.sign(np.nan_to_num(mu)), 0).astype(int))


def initialize():
    global DATA, STATE, PROFILE, CONFIG
    DATA = dict(np.load(BUNDLE/'data/reference.npz'))
    f = dict(np.load(FEATURES/'features.npz'))
    STATE = state(DATA, f)
    PROFILE = load_profile()
    CONFIG = copy.deepcopy(PROFILE['configs']['reference'])
    CONFIG['dense_pairs'] = [list(x) for x in PAIRS]
    CONFIG['conditional_completion'] = {'risk_aversion': 0.}


def slice_data(data, index):
    n = len(data['times'])
    return {k: (v[index:index+HOURS+1].copy() if v.ndim and
        len(v) == n else v.copy()) for k, v in data.items()}


def episode(index, pair, scenario, *, retain=False):
    from pair_regime_research.replay import replay
    d = slice_data(DATA, index); s = slice_data(STATE, index)
    assert len(d['times']) == HOURS+1
    s.pop('times')
    # This affects only new source admission; completion/unwind of actual
    # holdings still use the original sealed source and execution rules.
    s['directions'][:] = 0
    s['directions'][0, pair] = STATE['directions'][index, pair]
    r = replay(d, s, None, CONFIG, PROFILE['trading'],
        'pairs_dense_hourlyhedge_individual_72', scenario,
        int(d['times'][0]), int(d['times'][-1]))
    m = r['metrics']
    assert m['maximum_accounting_error'] < 1e-6
    assert sum(e['event'] == 'pair_intent' for e in r['events']) <= 1
    assert all(f['decision_time'] < f['time'] for f in r['fills'])
    row = dict(index=index, pair=pair, scenario=scenario,
        source=int(d['times'][0]), maturity=int(d['times'][-1]),
        utility=m['pnl_after_liquidation_reserve'], net_pnl=m['net_pnl'],
        fees=m['fees'], funding=m['funding'], impact=m['execution_cost'],
        completed=m['completed_pairs'], residual=m['residual_gross'],
        fills=m['fills'], breaches=m['risk_breach_bars'],
        one_leg=any(e['event']=='one_leg_completion_pending' for e in r['events']),
        repaired=any(e['event']=='repair_requested' for e in r['events']))
    return (row, r) if retain else row


def origin(index):
    rows = []
    for pair in np.flatnonzero(STATE['economic_action'][index]):
        for scenario in ['base', 'stress']:
            rows.append(episode(int(index), int(pair), scenario))
    return rows
