"""Audit inactive switches and distinguish holding from an unfinished exit."""
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape
from .research import ROOT,HERE,OUT,GRAPH,readrun,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',*sorted((OUT/'runs').glob('*.json.gz')),
           GRAPH/'books/results.json',ROOT/'experiments/basis_carry_20261017/protocol.json']
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    files += [ROOT/p for p in allowed]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    save(OUT/'audit_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    rows=[]
    for path in sorted((OUT/'runs').glob('*.json.gz')):
        run=readrun(path);counts=Counter(d['reason'] for d in run['switch_decisions'])
        # Performance fields remain identical to no-switch when all requests
        # are rejected; compare complete accounts, not just rounded PnL.
        u=next(u for u,arm in spec['candidates'].items() if run['arm']==arm)
        prefix=path.name.split('-'+run['arm'])[0]
        if u=='graph':parent=GRAPH/'runs'/f"{prefix}-{spec['universes'][u]}-{run['scenario']}.json.gz"
        else:
            from .research import GATE,STABILITY
            folder=STABILITY if prefix=='reference-continuous' else GATE
            parent=folder/'runs'/f"{prefix}-{spec['universes'][u]}-{run['scenario']}.json.gz"
        old=readrun(parent);same=True
        for key in old:
            if key in ['arm','metrics']:continue
            same &= run[key]==old[key]
        same &= all(run['metrics'][k]==v for k,v in old['metrics'].items())
        rows.append(dict(file=str(path.relative_to(ROOT)),reasons=dict(counts),
                         requests=run['metrics']['switch_requests'],same_account_path=bool(same)))
    # The motivating overlap was observed in the previous experiment. Inspect
    # its entire close process instead of inferring an active HOLD from duration.
    old=readrun(GRAPH/'runs'/'reference-comparison-first_dense_hourlyhedge_formationgraph_24-stress.json.gz')
    entries=[e for e in old['events'] if e['event']=='pair_intent']
    episode=next(e for e in entries if e['pair']==['EWY','EWJ'])
    close_request=next(e for e in old['events'] if e['time']>episode['time'] and e['event']=='exit_requested')
    closed=next(e for e in old['events'] if e['time']>close_request['time'] and e['event']=='closed')
    orders=[o for o in old['orders'] if close_request['time']<o['time']<=closed['time'] and o['symbol'] in episode['pair']]
    from .research import GATE
    control=readrun(GATE/'runs'/'reference-comparison-first_dense_hourlyhedge_decisiongate_24-stress.json.gz')
    rival=next(e for e in control['events'] if e['event']=='pair_intent' and close_request['time']<e['time']<closed['time'])
    snapshots={s['time']:s for s in old['snapshots']}
    closing=dict(entry_time=episode['time'],exit_requested_at=close_request['time'],exit_reason=close_request['reason'],
       flat_at=closed['time'],hours_waiting_for_flat=(closed['time']-close_request['time'])/3600,
       rival_entry_time=rival['time'],rival_expected_net_cash=rival['expected_net_cash'],
       exit_already_requested_before_rival_hours=(rival['time']-close_request['time'])/3600,
       held_qty_at_rival=snapshots[rival['time']]['pair_qty'],orders=orders,
       correction='The old ETF was already being liquidated 9h before the retained BTC/HYPE opportunity. Its EWJ exit took 11h under the unchanged participation/liquidity model. A later opportunity comparator cannot treat that residual inventory as flat. The prior overlap finding alone did not identify a missing HOLD-versus-switch decision as the cause.')
    metadata={t:load_tape(ROOT/t)[1] for t in allowed}
    books=json.loads((GRAPH/'books/results.json').read_text());coverage=[]
    for trial in books['trials']:
        if trial['model']!='formation_graph':continue
        fills=trial['known_fills'];first_fill=min(f['time'] for f in fills)
        end=metadata[trial['tape']]['end_ms']/1000
        earliest=first_fill+spec['minimum_holding_seconds']
        coverage.append(dict(tape=trial['tape'],scenario=trial['scenario'],entry_time=trial['time'],
          first_fill=first_fill,last_quote_time=end,earliest_allowed_switch=earliest,
          observable_minutes_after_entry=(end-first_fill)/60,
          minimum_holding_window_observed=end>=earliest,full_switch_pnl=None))
    verify(sources)
    save(OUT/'audit.json',dict(account_paths=rows,all_accounts_unchanged=all(r['same_account_path'] for r in rows),
        requests=sum(r['requests'] for r in rows),refined_liquidation_diagnosis=closing,
        old_book_coverage=coverage,full_quote_switch_pnl=None,full_quote_switch_q=None,
        protected=protected(),orders_sent=0,reserved_raw_read=False,historical_independence=False))
    print('AUDIT',sum(r['requests'] for r in rows),'switches;',sum(r['same_account_path'] for r in rows),
          'unchanged accounts; actual close waited',closing['hours_waiting_for_flat'],'h',flush=True)


if __name__=='__main__':main()
