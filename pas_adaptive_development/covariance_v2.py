"""Contract correction: never defer a change in the original signal's quantity.

The v1 family and its results remain immutable. This revision is a semantic fix,
not a parameter search: all hedge/risk constants and evaluation windows persist.
"""
import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .study import ROOT, OUT as PARENT, sha, save, native
from . import covariance as original

OUT = ROOT / 'reports/pas_covariance_v2_20261021'
ARMS = ['incumbent', 'size_only', 'hedged', 'hedged_restored']


def construct(base, sigma):
    """Partial minimum-variance hedge; occupied signal positions are retained."""
    owned = np.flatnonzero(abs(base) > 1e-12)
    empty = np.flatnonzero(abs(base) <= 1e-12)
    result = base.copy()
    if len(owned) and len(empty):
        hedge = -.5 * np.linalg.solve(
            sigma[np.ix_(empty, empty)], sigma[np.ix_(empty, owned)] @ base[owned])
        bound = .5 * abs(base).sum()
        if abs(hedge).sum() > bound:
            hedge *= bound / abs(hedge).sum()
        result[empty] = hedge
    before, after = float(base @ sigma @ base), float(result @ sigma @ result)
    # A valid covariance makes the hedge reduce variance. No leverage-up step.
    if after > before and after > 0:
        result *= np.sqrt(before / after)
    return result


def targets(base, sigma, times):
    result = np.zeros_like(base)
    held = np.zeros(base.shape[1])
    previous = np.zeros(base.shape[1])
    for i, t in enumerate(times):
        if i >= 168 and t % 21600 == 0:
            if t % 86400 == 0 or not np.array_equal(base[i], previous):
                held = construct(base[i], sigma[i])
                previous = base[i].copy()
        result[i] = held
    return result


def prepare():
    from competition_live.profile import load_profile, verify_deployment
    from replay_lab.common_comparison.engine import project
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Already frozen')
    p = json.loads((original.OUT / 'protocol.json').read_text())
    p.update(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS,
        revision='Preserve EVERY original target quantity change on the six-hour decision grid; all hedge/risk constants unchanged.',
        reason='v1 deferred 41 same-sign quantity changes over the input history until midnight. This conflicts with retaining the original signal quantities.',
        cadence='Daily covariance refresh OR any original target quantity change, on the original six-hour grid.',
        parent_protocol_sha=sha(original.OUT / 'protocol.json'),
        deployment=verify_deployment()['files'])
    save(OUT / 'protocol.json', p)
    z = dict(np.load(PARENT / 'targets.npz', allow_pickle=False))
    trading = load_profile()['trading']
    w = {k:z[k] for k in ['incumbent', 'size_only']}
    w['hedged'] = targets(z['incumbent'], z['sigma'], z['times'])
    w['hedged_restored'] = w['hedged'] * 1.44117372816997
    for name in ['hedged', 'hedged_restored']:
        for i in range(168, len(z['times'])):
            w[name][i] = project(w[name][i], trading['initial'], z['sigma'][i], trading['risk'])
    np.savez_compressed(OUT / 'targets.npz', times=z['times'], sigma=z['sigma'], **w)
    save(OUT / 'source_freeze.json', {str(f.relative_to(ROOT)):sha(f) for f in
        [Path(__file__), OUT / 'protocol.json', OUT / 'targets.npz']})
    print('Prepared corrected covariance family', flush=True)


def run():
    from competition_live.profile import load_profile, BUNDLE, verify_deployment
    from experiments.decision_gate_pairs_20261017.portfolio import replay
    from pas_allocation_study.study import summarize
    p = json.loads((OUT / 'protocol.json').read_text())
    for rel, h in json.loads((OUT / 'source_freeze.json').read_text()).items():
        assert sha(ROOT / rel) == h
    profile = load_profile()
    d0 = dict(np.load(BUNDLE / 'data/reference.npz', allow_pickle=False))
    states = dict(np.load(BUNDLE / 'models/reference_gate_state.npz', allow_pickle=False))
    z = dict(np.load(OUT / 'targets.npz', allow_pickle=False))
    ix = np.searchsorted(z['times'], d0['times'])
    np.testing.assert_array_equal(z['times'][ix], d0['times'])
    np.testing.assert_allclose(z['incumbent'][ix], d0['first'], atol=1e-12)
    windows = [r for r in json.loads((BUNDLE/'expected.json').read_text())['runs']
               if r['venue']=='reference' and r['arm']==profile['candidate']]
    rows = []
    for win in windows:
        for arm in ARMS:
            path = OUT/'reference'/f"{arm}_{win['phase']}_{win['scenario']}.json.gz"
            if path.exists():
                with gzip.open(path, 'rt') as f:
                    r = json.load(f)
            elif arm in ['incumbent', 'size_only']:
                with gzip.open(original.OUT/'reference'/path.name, 'rt') as f:
                    r = json.load(f)
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path, 'wt') as f:
                    json.dump(r, f, default=native)
            else:
                d = dict(d0); d['first'] = z[arm][ix]
                cfg = copy.deepcopy(profile['configs']['reference'])
                cfg['conditional_completion'] = {'risk_aversion':0.}
                r = replay(d, states, None, cfg, profile['trading'], profile['candidate'],
                           win['scenario'], win['start'], win['end'])
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path, 'wt') as f:
                    json.dump(r, f, default=native)
            row = summarize(r, arm, win['phase'], win['scenario'])
            assert row['timing_pass'] and row['accounting_error'] < 1e-6
            rows.append(row); save(OUT/'reference_results.json', rows)
            print(json.dumps(row), flush=True)
    assert verify_deployment()['files'] == p['deployment']


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
