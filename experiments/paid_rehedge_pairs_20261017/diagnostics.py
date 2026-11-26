"""Reconcile actual hedge adjustment paths; never alter candidate selection."""
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha
from experiments.complementary_alpha_20261017.research import protected
from .research import control_directory

HERE = Path(__file__).resolve().parent
ART = HERE/'artifacts'


def read(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def main():
    results = json.loads((ART/'results.json').read_text())
    frozen = json.loads((ART/'frozen.json').read_text())
    for rel, digest in frozen['sources'].items():
        assert sha(ROOT/rel) == digest, rel
    cfgs = {v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference', 'par']}
    rd, _, protocol = inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:
        pd = {k:z[k].copy() for k in z.files}
    data = {'reference':rd, 'par':pd}; rows = []; accounting = 0.
    for row in results['rows']:
        arm = row['arm']
        if '_renewhedge_' not in arm:
            continue
        venue, phase, scenario = row['venue'], row['phase'], row['scenario']
        matched = arm.replace('_renewhedge_', '_renew_')
        key = f'{venue}-{phase}-{arm}-{scenario}'
        prior_key = f'{venue}-{phase}-{matched}-{scenario}'
        run = read(ART/'runs'/f'{key}.json.gz')
        prior = read(control_directory(matched)/'runs'/f'{prior_key}.json.gz')
        audits = {name:episode_attribution(r, data[venue], cfgs[venue]['scenarios'][scenario])
                  for name, r in [('adjusted', run), ('no_resize', prior)]}
        totals = {name:{k:sum(e[k] for e in a['episodes'])
                        for k in ['price_move', 'execution_cost', 'fees', 'funding', 'net_pnl', 'fills']}
                  for name, a in audits.items()}
        accounting = max(accounting, *(a['accounting_error'] for a in audits.values()))
        delta = dict(
            pair_price_change=totals['adjusted']['price_move']-totals['no_resize']['price_move'],
            pair_execution_cost_saved=totals['no_resize']['execution_cost']-totals['adjusted']['execution_cost'],
            pair_fees_saved=totals['no_resize']['fees']-totals['adjusted']['fees'],
            pair_funding_change=totals['adjusted']['funding']-totals['no_resize']['funding'],
            pair_net_change=totals['adjusted']['net_pnl']-totals['no_resize']['net_pnl'],
            base_owned_change=run['metrics']['base_owned_pnl']-prior['metrics']['base_owned_pnl'],
            account_net_change=run['metrics']['net_pnl']-prior['metrics']['net_pnl'],
            q_change=run['metrics']['financial_score']-prior['metrics']['financial_score'],
        )
        assert abs(delta['pair_net_change']-sum(delta[k] for k in [
            'pair_price_change', 'pair_execution_cost_saved', 'pair_fees_saved', 'pair_funding_change'])) < 1e-6
        assert abs(delta['account_net_change']-delta['pair_net_change']-delta['base_owned_change']) < 1e-6
        plans = [e for e in run['events'] if e['event'] == 'hedge_adjustment_planned']
        packets = [d for d in run['decisions'] if d.get('rehedge_order')]
        outcomes = [e for e in run['events'] if e.get('outcome') in ['renewed', 'unfilled', 'unwind']]
        assert len(packets) == len(outcomes)
        gross_errors = []
        for packet, outcome in zip(packets, outcomes):
            plan = next(e for e in plans if e['time'] == packet['decision_time'])
            assert packet['decision_time'] < outcome['time']
            assert packet['owner_limits']['pair'] == plan['entry_limits']
            # Asset IDs come from configured market symbols, not pair-column indices.
            ids = [protocol['symbols'].index(symbol) for symbol in plan['pair']]
            got = np.array([sum(f['signed_qty'] for f in run['owner_fills']['pair']
                                if f['asset'] == j and packet['decision_time'] < f['time'] <= outcome['time']) for j in ids])
            np.testing.assert_allclose(got, outcome['actual_delta'], rtol=0, atol=1e-8)
            gross_errors.append(abs(plan['target_gross']/min(plan['current_gross'], cfgs[venue]['pair_max_gross'])-1))
        rows.append(dict(
            venue=venue, phase=phase, arm=arm, scenario=scenario, matched=matched,
            delta=delta, totals=totals, planned_adjustments=len(plans), submitted_adjustments=len(packets),
            outcomes=dict(Counter(e['outcome'] for e in outcomes)),
            rejection_reasons=dict(Counter(e['reason'] for e in run['events'] if e['event'] in ['renewal_rejected', 'adjustment_packet_rejected', 'rehedge_unwind'])),
            maximum_target_gross_error=max(gross_errors, default=0.), episodes=audits['adjusted']['episodes'],
        ))
    save(ART/'attribution.json', dict(rows=rows, maximum_episode_accounting_error=accounting,
                                    post_evaluation_only=True, protected=protected(), orders_sent=0))
    for row in rows:
        if row['venue'] == 'reference' and row['phase'] == 'comparison' and row['arm'].startswith('first'):
            print(row['arm'], row['scenario'], 'delta', row['delta']['account_net_change'],
                  'Q delta', row['delta']['q_change'], 'outcomes', row['outcomes'], 'rejected', row['rejection_reasons'], flush=True)


if __name__ == '__main__':
    main()
