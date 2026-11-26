"""Evaluate one predeclared past-formation pair universe; never deploy."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.decision_gate_pairs_20261017.portfolio import replay
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as PAR_INPUT, save, sha
from .formation import select, mask_entries

HERE = Path(__file__).resolve().parent
OUT = HERE/'artifacts'
GATE = HERE.parent/'decision_gate_pairs_20261017/artifacts'
SECTOR = HERE.parent/'decision_sector_pairs_20261018/artifacts'
STABILITY = HERE.parent/'lifecycle_stability_pairs_20261018/artifacts'


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()))


def readrun(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def main():
    OUT.mkdir(exist_ok=False); (OUT/'runs').mkdir()
    spec = json.loads((HERE/'protocol.json').read_text())
    tests = (HERE/'test_results.txt').read_text()
    assert 'Ran 6 tests' in tests and tests.rstrip().endswith('OK')
    parent = json.loads((SECTOR/'results.json').read_text())
    windows = parent['windows']; rows = parent['rows'].copy()
    configs = {v: json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text())
               for v in ['reference', 'par']}
    configs['par_extended'] = copy.deepcopy(configs['par'])
    refpath = ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'
    extpath = ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz'
    oldpaths = {}
    for v, phase in windows:
        for sc in ['base', 'stress']:
            for label, directory, arm in [('two', GATE, spec['control']), ('six', SECTOR, spec['sector_control'])]:
                oldpaths[v, phase, sc, label] = directory/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz'
    continuous = {sc: STABILITY/'runs'/f"reference-continuous-{spec['control']}-{sc}.json.gz" for sc in ['base','stress']}
    sources = json.loads((SECTOR/'frozen.json').read_text())['sources'].copy()
    files = [*HERE.glob('*.py'), HERE/'protocol.json', HERE/'test_results.txt', refpath, extpath,
             PAR_INPUT/'inputs.npz', SECTOR/'frozen.json', SECTOR/'results.json', *oldpaths.values(), *continuous.values()]
    files += [base/f'{v}_{suffix}_state.npz' for v in configs for base, suffix in [(GATE,'gate'),(SECTOR,'sector')]]
    files += [ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json' for v in ['reference','par']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files})
    verify(sources)
    prospective = json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    save(OUT/'frozen.json', dict(at_unix=time.time(), sources=sources, protected=protected(),
         previous_goal_turn_classification=spec['previous_goal_turn_classification']))
    rd, _, trading = inputs(configs['reference'])
    formation = select(load(refpath), list(trading['symbols']), spec)
    save(OUT/'formation.json', formation)
    formation_hash = sha(OUT/'formation.json')
    print('FORMATION', json.dumps(formation), flush=True)
    data = dict(reference=rd, par=load(PAR_INPUT/'inputs.npz'), par_extended=load(extpath))
    controls = {v: load(GATE/f'{v}_gate_state.npz') for v in configs}
    sectors = {v: load(SECTOR/f'{v}_sector_state.npz') for v in configs}
    masked = {}
    for v, state in sectors.items():
        assert [[trading['symbols'][int(i)] for i in pair] for pair in state['pairs']] == spec['pairs']
        masked[v] = mask_entries(state, formation['mask'])
        for key in state:
            if key != 'economic_available':
                np.testing.assert_array_equal(masked[v][key], state[key])
        np.savez_compressed(OUT/f'{v}_state.npz', **masked[v])
    originals = {k: readrun(path) for k, path in oldpaths.items()}
    exact = []

    def evaluate(v, phase, sc, label, old):
        cfg = copy.deepcopy(configs[v]); cfg['conditional_completion'] = {'risk_aversion':0.}
        if label != 'two': cfg['dense_pairs'] = spec['pairs']
        state = {'two': controls, 'six': sectors, 'candidate': masked}[label][v]
        arm = {'two':spec['control'], 'six':spec['sector_control'], 'candidate':spec['candidate']}[label]
        run = plain(replay(data[v], state, None, cfg, trading, arm, sc, old['start'], old['end']))
        if label != 'candidate':
            for key in old:
                assert run[key] == old[key], (v, phase, sc, label, key)
            exact.append(dict(venue=v, phase=phase, scenario=sc, control=label))
            print('EXACT',v,phase,sc,label,flush=True)
        else:
            with gzip.open(OUT/'runs'/f'{v}-{phase}-{arm}-{sc}.json.gz','wt') as f:
                json.dump(run, f, allow_nan=False)
            rows.append(dict(venue=v, phase=phase, arm=arm, scenario=sc, metrics=run['metrics']))
            m=run['metrics']
            print('RUN',v,phase,sc,'PnL',m['net_pnl'],'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)
        return run

    def window(v, phase):
        for sc in ['base','stress']:
            for label in ['two','six','candidate']:
                old = originals[v,phase,sc,'two' if label=='candidate' else label]
                evaluate(v,phase,sc,label,old)

    window('reference','validation')
    lookup = {(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    checks=[]
    for sc in ['base','stress']:
        a=lookup['reference','validation',spec['candidate'],sc]
        b=lookup['reference','validation',spec['control'],sc]
        checks.append(dict(scenario=sc,pnl_difference=a['net_pnl']-b['net_pnl'],
          q_difference=a['financial_score']-b['financial_score'], no_risk_breach=a['risk_breach_bars']==0,
          above_native_pnl_q=all(a['net_pnl']>lookup['reference','validation',n,sc]['net_pnl'] and
             a['financial_score']>lookup['reference','validation',n,sc]['financial_score'] for n in ['first_strategy','second_strategy']),
          completed_pairs=a['completed_pairs']))
    promote = (all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and
                  c['no_risk_breach'] and c['above_native_pnl_q'] for c in checks)
               and any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks))
    selection=dict(selected=spec['candidate'] if promote else spec['control'],promote_graph=bool(promote),
                   checks=checks,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection); selection_hash=sha(OUT/'selection.json')
    for v,phase in windows:
        if (v,phase)!=('reference','validation'): window(v,phase)
    for sc,path in continuous.items():
        old=readrun(path)
        evaluate('reference','continuous',sc,'two',old)
        evaluate('reference','continuous',sc,'candidate',old)
    assert sha(OUT/'selection.json')==selection_hash and sha(OUT/'formation.json')==formation_hash
    verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,windows=windows,formation=formation,selection=selection,
         new_account_replays=14,exact_controls=exact,tests_passed=6,protected=protected(),
         prospective_sources=len(prospective),historical_independence=False,
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False))
    print('DONE formation graph: 14 new accounts, 26 exact controls',flush=True)


if __name__=='__main__':main()
