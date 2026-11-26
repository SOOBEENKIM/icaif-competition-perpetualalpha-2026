"""Account and episode audit for the predeclared fixed-graph gate ablation."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .research import ROOT, HERE, OUT, QUEUE, GRAPH, readrun, save, sha, verify, protected


def pair_attribution(run):
    output = defaultdict(lambda: defaultdict(float))
    for episode in run['episodes']:
        key = '/'.join(episode['pair'])
        for field in ['net_pnl', 'fees', 'funding', 'realized']:
            output[key][field] += episode[field]
        output[key]['episodes'] += 1
    return {key: dict(value) for key, value in output.items()}


def verify_account(run, allowed):
    snapshots = {s['time']: s for s in run['snapshots']}
    queued = 'episodes' in run
    for snapshot in snapshots.values():
        np.testing.assert_allclose(np.array(snapshot['base_qty']) + snapshot['pair_qty'],
                                   snapshot['qty'], rtol=0, atol=1e-10)
        if queued:
            np.testing.assert_allclose(np.array(snapshot['active_pair_qty']) + snapshot['residual_qty'],
                                       snapshot['pair_qty'], rtol=0, atol=1e-10)
            assert not np.any((np.asarray(snapshot['active_pair_qty']) != 0) &
                              (np.asarray(snapshot['residual_qty']) != 0))
    entries = [e for e in run['events'] if e['event'] == 'pair_intent']
    assert all(tuple(e['pair']) in allowed for e in entries)
    assert run['metrics']['risk_breach_bars'] == 0
    assert run['metrics']['maximum_accounting_error'] < 1e-6
    if not queued:
        return dict(queued=False, entries=len(entries), attribution=None)

    assert abs(sum(e['net_pnl'] for e in run['episodes']) - run['metrics']['pair_owned_pnl']) < 1e-6
    for field in ['fees', 'funding']:
        expected = (sum(f['fee'] for owner in ['pair', 'residual'] for f in run['owner_fills'][owner])
                    if field == 'fees' else
                    sum(f['signed_amount'] + f['adverse_amount'] for owner in ['pair', 'residual']
                        for f in run['owner_funding'][owner]))
        assert abs(sum(e[field] for e in run['episodes']) - expected) < 1e-6
    for fill in run['owner_fills']['residual']:
        if fill['allocation_type'] == 'residual_transfer':
            continue
        assert fill['episode_id'] is not None
        assert abs(fill['position_after']) <= abs(fill['position_before']) + 1e-10
        assert fill['position_after'] * fill['position_before'] >= 0
    concurrent = []
    for event in run['events']:
        if event['event'] == 'residual_handoff':
            assert any(d['execution_time'] == event['time'] and d['active_exit_attempt']
                       for d in run['decisions'])
        if event['event'] == 'pair_intent':
            residual = np.asarray(snapshots[event['time']]['residual_qty'])
            assert not np.any(residual[event['ids']])
            if np.any(residual):
                concurrent.append(dict(time=event['time'], pair=event['pair']))
    completed = sum(int(e.get('balanced', False)) for e in run['events'] if e['event'] == 'closed')
    completed += sum(int(e['residual']['balanced']) for e in run['events'] if e['event'] == 'residual_closed')
    assert completed == run['metrics']['completed_pairs']
    return dict(queued=True, entries=len(entries), concurrent=concurrent,
                attribution=pair_attribution(run), episode_accounting_verified=True)


def main():
    spec = json.loads((HERE / 'protocol.json').read_text())
    results = json.loads((OUT / 'results.json').read_text())
    formation = json.loads((GRAPH / 'formation.json').read_text())
    allowed = {tuple(pair) for pair in formation['selected_pairs']}
    sources = json.loads((OUT / 'frozen.json').read_text())['sources'].copy()
    paths = sorted((OUT / 'runs').glob('*.json.gz'))
    controls = [QUEUE / 'runs' / f"{v}-{phase}-{spec['queued_control']}-{sc}.json.gz"
                for v, phase in results['windows'] for sc in ['base', 'stress']]
    files = [Path(__file__), OUT / 'results.json', OUT / 'selection.json', *paths, *controls]
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in files})
    verify(sources)
    save(OUT / 'audit_frozen.json', dict(at_unix=time.time(), sources=sources, protected=protected()))
    rows = []
    for path in paths:
        run = readrun(path)
        rows.append(dict(file=str(path.relative_to(ROOT)), **verify_account(run, allowed)))
    effects = []
    for venue, phase in results['windows']:
        for scenario in ['base', 'stress']:
            direct = readrun(OUT / 'runs' / f"{venue}-{phase}-{spec['candidate']}-{scenario}.json.gz")
            gated = readrun(QUEUE / 'runs' / f"{venue}-{phase}-{spec['queued_control']}-{scenario}.json.gz")
            verify_account(gated, allowed)
            attributions = [pair_attribution(run) for run in [direct, gated]]
            changes = {}
            for pair in sorted(set(attributions[0]) | set(attributions[1])):
                changes[pair] = {field: attributions[1].get(pair, {}).get(field, 0.) -
                                attributions[0].get(pair, {}).get(field, 0.)
                                for field in ['net_pnl', 'fees', 'funding', 'realized', 'episodes']}
            account_gain = gated['metrics']['net_pnl'] - direct['metrics']['net_pnl']
            base_gain = gated['metrics']['base_owned_pnl'] - direct['metrics']['base_owned_pnl']
            assert abs(account_gain - base_gain - sum(v['net_pnl'] for v in changes.values())) < 1e-6
            effects.append(dict(venue=venue, phase=phase, scenario=scenario,
                gated_minus_direct_pnl=account_gain,
                gated_minus_direct_q=(gated['metrics']['financial_score'] - direct['metrics']['financial_score']
                    if gated['metrics']['financial_score'] is not None else None),
                base_owned_pnl_difference=base_gain, pair_differences=changes))
    assert len(rows) == 28 and len(effects) == 14
    verify(sources)
    save(OUT / 'audit.json', dict(accounts_verified=len(rows), rows=rows, gate_effects=effects,
        selected_pairs=formation['selected_pairs'], risk_breach_bars=0, protected=protected(),
        orders_sent=0, reserved_raw_read=False, historical_independence=False))
    print('AUDIT', len(rows), 'new accounts;', len(effects), 'matched gate comparisons', flush=True)


if __name__ == '__main__':
    main()
