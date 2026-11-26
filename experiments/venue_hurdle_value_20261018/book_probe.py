"""Fixed source IOC plans against received historical books: entry feasibility only.

Do not use a fictitious terminal price, manufacture a paid exit, or convert an
unknown arrival into an unfilled order. Each accepted source is an independent
flat-account probe, not one simultaneous executable portfolio or an ROI series.
"""
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT, HERE, OUT, setup, load, save, sha, verify, protected
from .entry import proposals
from experiments.basis_carry_20261017.probe import load_tape, asof
from experiments.published_pair_lifecycle_20261017.account import BookAccount


def main():
    out=OUT/'books';out.mkdir(exist_ok=False)
    spec,cfg,p,d,de=setup()
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    settings=ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json'
    books=json.loads(settings.read_text())
    models=json.loads((OUT/'models.json').read_text())
    sources=json.loads((OUT/'evaluation_frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),settings,OUT/'results.json',*(ROOT/t for t in allowed)]
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,probe='independent_entry_only',orders_sent=0))
    trials=[];summaries=[];coverage=[]
    for tape in allowed:
        streams,metadata=load_tape(ROOT/tape);coverage.append(metadata)
        first=math.ceil(metadata['start_ms']/1000/spec['sample_seconds'])*spec['sample_seconds']
        for horizon in spec['horizons_hours']:
            for kind in ['tree','hurdle']:
                for scenario,sc in cfg['scenarios'].items():
                    counts=Counter()
                    for now in range(first,metadata['end_ms']//1000+1,spec['sample_seconds']):
                        i=int(np.searchsorted(d['times'],now))
                        if i>=len(d['times']) or d['times'][i]!=now:continue
                        counts['scheduled_sources']+=1
                        available=[m for m in models if m['horizon']==horizon and m['fit_time']<=now]
                        model=max(available,key=lambda m:m['fit_time']) if available else None
                        rcfg=copy.deepcopy(cfg)
                        rcfg.update(hurdle_model_kind=kind,hurdle_models={(horizon,now):model} if model else {})
                        choices=proposals(i,d,de,None,horizon,rcfg,p,sc,np.zeros(10),p['initial'],
                            np.zeros(10),de['directions'][i],f'first_dense_hurdle_{kind}_{horizon}')
                        counts['positive_plans']+=len(choices)
                        chosen=None;quotes=None
                        for proposal in choices:
                            observed={s:asof(streams[s],now*1000,books) if s in streams else None for s in proposal['pair']}
                            if any(v is None for v in observed.values()):
                                counts['missing_fresh_pair_book']+=1;continue
                            chosen=proposal;quotes=observed;break
                        if chosen is None:continue
                        account=BookAccount(p['initial'],p['symbols']);marks=d['par_close'][i].copy()
                        for symbol,j in zip(chosen['pair'],chosen['ids']):
                            marks[j]=float(quotes[symbol]['ctx'].get('markPx',quotes[symbol]['mid']))
                        target=np.zeros(10);target[chosen['ids']]=chosen['target']
                        packet=dict(packet_id=f'{now}:{kind}:{horizon}',decision_time=now,
                            source_close=now,source_available=now,source_prices=d['par_close'][i].tolist(),
                            owner_targets={'base':np.zeros(10).tolist(),'pair':target.tolist()},
                            owner_limits={'pair':chosen['entry_limits']},risk_forced=False,sigma=d['sigma'][i])
                        book_sc=books['scenarios'][scenario]
                        packet=account.seal(packet,p,book_sc)
                        def no_unobserved_funding(at, ledger):
                            assert int(at)//3600000==now//3600, 'Entry probe crosses funding boundary'
                        result=account.execute_books(packet,streams,marks,d['sigma'][i],p,book_sc,books,on_time=no_unobserved_funding)
                        q=account.owners['pair'].qty[chosen['ids']]
                        ratios=q/np.asarray(chosen['target'])
                        if result['unknown_symbols']:status='unknown_arrival_requires_reconciliation'
                        elif not np.any(q):status='observed_unfilled'
                        elif min(ratios)>0 and abs(ratios[0]-ratios[1])/max(ratios)<=cfg['pair_balance_tolerance']:
                            status='balanced_entry_exit_unobserved'
                        else:status='partial_entry_completion_and_exit_unobserved'
                        counts[status]+=1
                        trials.append(dict(tape=tape,time=now,kind=kind,horizon=horizon,scenario=scenario,
                            plan=chosen,status=status,orders=result,qty=q.tolist(),fills=account.physical.fills,
                            fees=account.physical.fees,mark_drift_bps=((marks[chosen['ids']]/d['par_close'][i,chosen['ids']]-1)*10000).tolist(),
                            account_return=None,financial_score=None,full_lifecycle_observed=False,orders_sent=0))
                    summaries.append(dict(tape=tape,kind=kind,horizon=horizon,scenario=scenario,counts=dict(counts)))
        print('BOOK_TAPE',tape,'cumulative trials',len(trials),flush=True)
    verify(sources)
    save(out/'results.json',dict(trials=trials,summaries=summaries,coverage=coverage,
        statuses=dict(Counter(r['status'] for r in trials)),orders_sent=0,live_strategy_changed=False,
        independent_flat_entry_probes=True,full_portfolio_return=None,financial_score=None,
        completion_and_maturity_exit_not_evaluated=True,not_full_horizon_performance=True,
        frozen_candle_value_not_recalibrated_to_books=True,historical_independence=False,reserved_raw_read=False,protected=protected()))
    print('DONE entry books',len(trials),dict(Counter(r['status'] for r in trials)),flush=True)


if __name__=='__main__':main()
