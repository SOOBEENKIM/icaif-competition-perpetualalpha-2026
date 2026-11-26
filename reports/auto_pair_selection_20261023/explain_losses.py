"""Post-result attribution only. Does not fit, select or change any strategy."""
import collections
import gzip
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent


def attribute(arm, scenario):
    with gzip.open(OUT / 'reference' / f'{arm}_continuous_{scenario}.json.gz', 'rt') as f:
        run = json.load(f)
    spec = json.loads((OUT / 'protocol.json').read_text())
    data = np.load(ROOT / spec['input'], allow_pickle=False)
    times, prices = data['market_times'], data['close']
    episodes, active = [], None
    for e in run['events']:
        if e['event'] == 'pair_intent':
            assert active is None
            active = dict(e)
        elif e['event'] in ('closed', 'entry_unfilled'):
            assert active is not None and e['pair'] == active['pair']
            active.update(end=e['time'], balanced=e.get('balanced', False), aborted=e.get('aborted', False), closed=True)
            episodes.append(active)
            active = None
        elif e['event'] in ('balanced_open', 'completion_balanced', 'balanced_after_repair') and active is not None:
            active['balanced'] = True
    if active is not None:
        active.update(end=run['snapshots'][-1]['time'], closed=False,
                      balanced=active.get('balanced', False), aborted=False)
        episodes.append(active)
    else:
        assert np.all(np.asarray(run['snapshots'][-1]['pair_qty']) == 0)
    prior_qty = {s['time'] + 3600: np.asarray(s['pair_qty']) for s in run['snapshots']}
    rows = []
    for e in episodes:
        fills = [f for f in run['owner_fills']['pair'] if e['time'] < f['time'] <= e['end']]
        funding = sum(-float(prior_qty[f['time']] @ (np.asarray(f['prices']) * np.asarray(f['rates'])))
                      for f in run['funding_events'] if e['time'] < f['time'] <= e['end'])
        realized = sum(f['realized'] for f in fills)
        fees = sum(f['fee'] for f in fills)
        q = np.zeros(10)
        for f in fills:
            q[f['asset']] += f['signed_qty']
        mark_index = int(np.searchsorted(times, e['end']))
        assert times[mark_index] == e['end']
        net = -sum(f['signed_qty']*f['price'] for f in fills) - fees + funding + float(q@prices[mark_index])
        if e['closed']:
            assert np.max(abs(q)) < 1e-8 and abs(net-realized+fees-funding) < 1e-6
        source = int(e['forecast_source'])
        i = int(np.searchsorted(times, source))
        j = int(np.searchsorted(times, source + 24 * 3600))
        assert times[i] == source and times[j] == source + 24 * 3600
        ret = prices[j, e['ids']] / prices[i, e['ids']] - 1
        actual = float((ret[0] - e['source_beta'] * ret[1]) / (1 + e['source_beta']))
        forecast = e['source_price_alpha']
        rows.append(dict(pair=e['pair'], start=e['time'], end=e['end'], balanced=e['balanced'],
                         aborted=e['aborted'], closed=e['closed'], owner_fills=len(fills), realized_after_execution=realized,
                         fees=fees, funding=funding, net_pnl=net,
                         predicted_24h_return=forecast, realized_source_24h_return=actual,
                         chosen_direction=e['direction'], direction_correct=bool(e['direction']*actual > 0)))
    assert sum(x['owner_fills'] for x in rows) == len(run['owner_fills']['pair'])
    total = sum(x['net_pnl'] for x in rows)
    assert abs(total - run['metrics']['pair_owned_pnl']) < 1e-6
    groups = collections.defaultdict(list)
    for row in rows:
        groups['/'.join(row['pair'])].append(row)
    grouped = [dict(pair=k, attempts=len(v), balanced=sum(x['balanced'] for x in v),
                    net_pnl=sum(x['net_pnl'] for x in v), fees=sum(x['fees'] for x in v),
                    funding=sum(x['funding'] for x in v)) for k, v in groups.items()]
    summary = dict(attempts=len(rows), direction_correct=sum(x['direction_correct'] for x in rows),
                   source_24h_prediction_mae_bps=1e4*np.mean([abs(x['predicted_24h_return']-x['realized_source_24h_return']) for x in rows]),
                   source_24h_zero_prediction_mae_bps=1e4*np.mean([abs(x['realized_source_24h_return']) for x in rows]),
                   pair_net_pnl=total,
                   completed_balanced_net_pnl=sum(x['net_pnl'] for x in rows if x['balanced']),
                   never_balanced_net_pnl=sum(x['net_pnl'] for x in rows if not x['balanced']))
    return dict(arm=arm, scenario=scenario, summary=summary,
                pairs=sorted(grouped, key=lambda x:x['net_pnl']), episodes=rows,
                account_metrics=run['metrics'])


if __name__ == '__main__':
    result = dict(scope='Post-result diagnostic on reused reference history, not independent performance or PAS fills.',
                  notes=['Episode net includes simulated execution prices, fees, effective funding and internal ownership transfers.',
                         '24h direction/MAE uses each entry intention source and fixed source beta, not later holding renewals.',
                         'Entry-selected observations overlap and are not independent. Do not treat this as universe-wide prediction accuracy.',
                         'Completion means balanced lifecycle closed; unsuccessful entry episodes are listed separately.'],
                  runs=[attribute(a,s) for s in ('base','stress') for a in ('incumbent_fixed_two','automatic_45')])
    with gzip.open(OUT / 'reference/incumbent_fixed_two_continuous_base.json.gz', 'rt') as f:
        incumbent = json.load(f)
    exclusions = []
    for e in incumbent['events']:
        if e['event'] != 'pair_intent':
            continue
        model = json.loads((OUT / 'models' / f"{e['time']//86400*86400}.json").read_text())
        selection = next(x for x in model['selection']['pairs'] if x['pair'] == e['pair'])
        exclusions.append(dict(source=e['time'], pair=e['pair'], selected=selection['selected'],
                               reasons=selection['reasons'],
                               raw_block_betas=[b['beta'] for b in selection['blocks']],
                               incumbent_source_beta=e['source_beta']))
    result['incumbent_entries_under_new_formation_filter'] = exclusions
    (OUT / 'loss_attribution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for r in result['runs']:
        print(r['arm'], r['scenario'], r['summary'])
        for x in r['pairs']:
            print(x)
