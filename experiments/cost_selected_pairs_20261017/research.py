"""Frozen costed expert selection; later evaluation cannot reselect."""
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.research import load
from experiments.hourly_refresh_pairs_20261017.policy import replay
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha, ts
from .model import infer

HERE = Path(__file__).resolve().parent
HOURLY = HERE.parent/'hourly_refresh_pairs_20261017'
CAUSAL = HERE.parent/'causal_expert_pairs_20261017'


def main():
    out = HERE/'artifacts'; out.mkdir(exist_ok=False); (out/'runs').mkdir()
    spec = json.loads((HERE/'protocol.json').read_text())
    expert_spec = json.loads((HOURLY/'model_protocol.json').read_text())
    cfgs = {v: json.loads((HOURLY/f'{v}_protocol.json').read_text()) for v in ['reference', 'par']}
    parent = json.loads((CAUSAL/'artifacts/results.json').read_text())
    sources = json.loads((CAUSAL/'artifacts/frozen.json').read_text())['sources'].copy()
    sources.update(json.loads((HOURLY/'prepared/frozen.json').read_text())['sources'])
    for rel, digest in sources.items():
        assert sha(ROOT/rel) == digest, rel
    controls = ['first_strategy', 'second_strategy', *[f'{base}_dense_hourlyhedge_{kind}_24'
                for base in ['first', 'second'] for kind in [*spec['experts'], 'equal', 'imse', 'stack']]]
    rows = [r for r in parent['rows'] if r['arm'] in controls]
    assert len(rows) == 128
    paths = [*HERE.glob('*.py'), HERE/'protocol.json', HERE/'test_results.txt',
             CAUSAL/'artifacts/frozen.json', CAUSAL/'artifacts/results.json', HOURLY/'prepared/frozen.json']
    for venue in cfgs:
        paths.extend(HOURLY/'prepared'/f'{venue}_{kind}_24_state.npz' for kind in spec['experts'])
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    save(out/'frozen.json', dict(at_unix=time.time(), sources=sources, protected=protected(),
                               historical_independence=False, orders_sent=0,
                               previous_goal_turn='progress_causal_experts_48_accounts_and_actual_book_failure_completed'))
    rd, _, p = inputs(cfgs['reference']); data = {'reference': rd, 'par': load(INPUT/'inputs.npz')}
    refs = {'reference': load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
            'par': load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    models = {}
    for pair in expert_spec['pairs']:
        model = json.loads((ROOT/expert_spec['model_directory']/('_'.join(pair)+'_24.json')).read_text())
        assert model['fit_available'] == ts(expert_spec['model_available'])
        models[tuple(pair), 24] = model
    states = {}; manifests = []
    for venue, reference in refs.items():
        features = design(reference['market_times'], reference['close'], reference['volume'], reference['trades'],
                          p['symbols'], cfgs[venue]['dense_pairs'])
        ss, cc, records, arrays = infer(data[venue]['times'], reference, features, models,
                                       cfgs[venue]['dense_pairs'], expert_spec, spec)
        for kind, state in cc.items():
            old = load(HOURLY/'prepared'/f'{venue}_{kind}_24_state.npz')
            for key in old:
                np.testing.assert_array_equal(state[key], old[key], err_msg=f'{venue}-{kind}-{key}')
            states[venue, kind] = state
        for kind, state in ss.items():
            np.savez_compressed(out/f'{venue}_{kind}_24_state.npz', **state)
            states[venue, kind] = state
        np.savez_compressed(out/f'{venue}_issued_forecasts.npz', **arrays)
        save(out/f'{venue}_weight_audit.json', records)
        assert all(r['last_maturity'] is None or r['last_maturity'] < r['time'] for r in records)
        manifests.append(dict(venue=venue, exact_expert_states=4, selection_decisions=len(records),
                              selected_sources={kind: sum(r['choices'][kind]['selected'] >= 0 for r in records) for kind in spec['candidates']}))
        print('PREPARED', venue, manifests[-1], flush=True)
    save(out/'preparation.json', manifests)
    verification = [('reference', 'validation', 'first', 'base'), ('reference', 'comparison', 'second', 'stress'),
                    ('par', 'validation', 'first', 'base'), ('par', 'comparison', 'second', 'stress')]
    verified = []
    for venue, phase, base, scenario in verification:
        arm = f'{base}_dense_hourlyhedge_bias_24'
        with gzip.open(HOURLY/'artifacts/runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz', 'rt') as f:
            old = json.load(f)
        run = replay(data[venue], states[venue, 'bias'], None, cfgs[venue], p, arm, scenario, old['start'], old['end'])
        for key in ['fills', 'events', 'metrics', 'snapshots']:
            assert run[key] == old[key], (venue, phase, key)
        verified.append(dict(venue=venue, phase=phase, arm=arm, scenario=scenario))
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r['metrics'] for r in rows}
    arms = [f'{base}_dense_hourlyhedge_{kind}_24' for base in ['first', 'second'] for kind in spec['candidates']]

    def record(venue, phase, arm, scenario):
        cfg = cfgs[venue]; kind = arm.split('_')[-2]
        start = ts(cfg['validation_start'] if phase == 'validation' else cfg['comparison_start'])
        end = ts(cfg['comparison_start'] if phase == 'validation' else cfg['end'])
        run = replay(data[venue], states[venue, kind], None, cfg, p, arm, scenario, start, end)
        with gzip.open(out/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz', 'wt') as f:
            json.dump(run, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        m = run['metrics']; lookup[venue, phase, arm, scenario] = m
        matched = arm.replace(f'_{kind}_24', '_bias_24'); old = lookup[venue, phase, matched, scenario]
        rows.append(dict(venue=venue, phase=phase, arm=arm, scenario=scenario, metrics=m,
                         matched=matched, delta_hourly_bias_pnl=m['net_pnl']-old['net_pnl'],
                         delta_hourly_bias_q=m['financial_score']-old['financial_score']))
        print(venue, phase, arm, scenario, 'PnL', round(m['net_pnl'], 3), 'Q', round(m['financial_score'], 8),
              'pairs', m['completed_pairs'], 'delta_bias', round(m['net_pnl']-old['net_pnl'], 3), flush=True)

    for arm in arms:
        for scenario in cfgs['reference']['scenarios']:
            record('reference', 'validation', arm, scenario)
    ranking = []
    for arm in arms:
        checks = []; margins = []
        for scenario in cfgs['reference']['scenarios']:
            m = lookup['reference', 'validation', arm, scenario]
            checks.extend([m['net_pnl'] > 0, m['completed_pairs'] >= 3, m['risk_breach_bars'] == 0])
            for base in ['first_strategy', 'second_strategy']:
                b = lookup['reference', 'validation', base, scenario]
                checks.extend([m['net_pnl'] > b['net_pnl'], m['financial_score'] > b['financial_score']])
                margins.append(m['net_pnl']-b['net_pnl'])
        ranking.append(dict(arm=arm, eligible=all(checks), minimum_native_pnl_advantage=min(margins)))
    ranking.sort(key=lambda r: (-r['eligible'], -r['minimum_native_pnl_advantage'], r['arm']))
    selection = dict(selected=ranking[0]['arm'], eligible=ranking[0]['eligible'], ranking=ranking,
                     frozen_before_later_account_replay=True, historical_independence=False)
    save(out/'selection.json', selection); selection_hash = sha(out/'selection.json')
    for venue, phase in [('reference', 'comparison'), ('par', 'validation'), ('par', 'comparison')]:
        for arm in arms:
            for scenario in cfgs[venue]['scenarios']:
                record(venue, phase, arm, scenario)
    assert sha(out/'selection.json') == selection_hash
    for rel, digest in sources.items():
        assert sha(ROOT/rel) == digest, rel
    save(out/'results.json', dict(rows=rows, selection=selection, new_account_replays=32,
                                 exact_control_replays=verified, reused_control_accounts=128, exact_expert_states=8,
                                 protected=protected(), orders_sent=0, historical_independence=False,
                                 automatic_live_replacement=False))
    print('DONE 32 new accounts; 4 exact controls; eligible', selection['eligible'], flush=True)


if __name__ == '__main__':
    main()
