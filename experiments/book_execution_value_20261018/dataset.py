"""Historical L2 execution labels, with fixed source-only size alternatives."""
import gzip,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.venue_execution_value_20261018.dataset import setup as old_setup,ROOT,save,sha,verify,protected
from experiments.par_direct_forecast_20261017.model import design
from experiments.basis_carry_20261017.probe import load_tape,asof
from .source import make_plan,features
from .episodes import episode

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def setup():
    _,cfg,p,d,_=old_setup();spec=json.loads((HERE/'protocol.json').read_text())
    de=design(d,spec,p['symbols'])
    books=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    return spec,cfg,p,d,de,books


def main():
    test=(HERE/'test_results.txt').read_text();assert 'Ran 6 tests' in test and test.rstrip().endswith('OK')
    OUT.mkdir(exist_ok=False);(OUT/'episodes').mkdir()
    spec,cfg,p,d,de,books=setup()
    sources=json.loads((ROOT/'experiments/venue_hurdle_value_20261018/artifacts/conclusion_frozen.json').read_text())['sources'].copy();verify(sources)
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',*(ROOT/t for t in spec['tapes'])]
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    rows=[];coverage=[];counts=Counter();names=None
    for ti,tape in enumerate(spec['tapes']):
        streams,meta=load_tape(ROOT/tape);coverage.append(meta)
        first=math.ceil(meta['start_ms']/1000/spec['cadence_seconds'])*spec['cadence_seconds']
        with gzip.open(OUT/'episodes'/f'tape_{ti}.jsonl.gz','wt') as log:
            for now in range(first,meta['end_ms']//1000+1,spec['cadence_seconds']):
                i=int(np.searchsorted(d['times'],now,side='right')-1)
                if i<0 or now-d['times'][i]>=300:continue
                for k,raw_ids in enumerate(de['pairs']):
                    ids=list(map(int,raw_ids));beta=float(de['beta'][i,k])
                    if not np.isfinite(beta) or not .1<=beta<=10 or de['correlation'][i,k]<spec['minimum_correlation']:continue
                    pair=spec['pairs'][k];quotes={s:asof(streams[s],now*1000,books) if s in streams else None for s in pair}
                    if any(v is None for v in quotes.values()):continue
                    marks=d['par_close'][i].copy()
                    for symbol,j in zip(pair,ids):marks[j]=quotes[symbol]['mid']
                    for scenario,sc in books['scenarios'].items():
                        for direction in [-1,1]:
                            for fraction in spec['size_fractions']:
                                for buffer in spec['limit_buffers_bps']:
                                    plan=make_plan(pair,ids,beta,quotes,now,marks,d['sigma'][i],p,sc,spec,direction,fraction,buffer)
                                    if plan is None:counts['source_inadmissible']+=1;continue
                                    x,columns=features(plan,quotes,now,marks,d['sigma'][i],p,sc,k)
                                    if names is not None:assert names==columns
                                    names=columns
                                    result=episode(plan,streams,quotes,now,meta['end_ms'],marks,d['sigma'][i],p,sc,books,spec)
                                    if result is None:counts['arrival_window_unobserved']+=1;continue
                                    identity=f'{ti}:{now}:{k}:{scenario}:{direction}:{fraction}:{buffer}'
                                    row=dict(sample_id=identity,tape=tape,tape_index=ti,source=now,pair_index=k,scenario=scenario,
                                        direction=direction,size_fraction=fraction,limit_buffer_bps=buffer,features=x.tolist(),
                                        intended_gross=plan['intended_gross'],**{n:result[n] for n in ['outcome','available_at','held_fraction','entry_cost_return','realized_cash_pnl','state','fees']})
                                    rows.append(row);counts[f'outcome_{result["outcome"]}']+=1
                                    log.write(json.dumps(dict(sample_id=identity,plan=plan,result=result),allow_nan=False)+'\n')
        print('TAPE',ti,'rows',len(rows),dict(counts),flush=True)
    verify(sources)
    save(OUT/'labels.json',dict(rows=rows,counts=dict(counts),feature_names=names,coverage=coverage,
        unique_source_times=len({r['source'] for r in rows}),dependent_size_side_scenario_replicas=True,
        labels_are_execution_not_full_trade_returns=True,historical_independence=False,reserved_raw_read=False,orders_sent=0))
    save(HERE/'work_state.json',dict(status='progress_actual_book_execution_labels_generated',rows=len(rows),
        next_action='Fit fixed execution models on first two tapes, freeze validation judgment on third, then evaluate later tapes and source-plan selection.',
        goal_complete=False,no_progress_streak=0,orders_sent=0,live_strategy_changed=False))
    print('DONE actual book labels',len(rows),dict(counts),flush=True)


if __name__=='__main__':main()
