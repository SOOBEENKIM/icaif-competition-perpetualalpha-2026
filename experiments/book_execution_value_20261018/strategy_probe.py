"""Unchanged reference alpha with source-book size and execution-risk choices."""
import copy,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,setup,save,sha,verify,protected
from .source import gross_capacity,features
from .episodes import episode
from .model import predict,score_plan
from experiments.dense_history_pairs_20261017.research import load
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry


def source_intent(state,i,k,d,p,direction):
    ids=list(map(int,state['pairs'][k]));anchors=state['source_prices'][i,ids];mu=float(state['mu'][i,k])
    return dict(pair=[p['symbols'][j] for j in ids],ids=ids,direction=direction,source_beta=float(state['beta'][i,k]),
        source_prices=anchors.tolist(),terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist(),
        forecast_source=int(state['signal_times'][i,k]),horizon=24,account_marks=d['par_close'][i].tolist())


def enriched(plan,intent,quotes,p,buffer):
    result=copy.deepcopy(plan);ids=intent['ids'];mids=np.array([quotes[s]['mid'] for s in intent['pair']])
    anchors=np.asarray(intent['source_prices']);beta=intent['source_beta']*anchors[0]/anchors[1]*mids[1]/mids[0]
    limits={str(o['asset']):o['limit'] for o in result['orders']}
    if buffer==0:
        limits={str(j):float(px) for j,px in zip(ids,plan['entry_preview'])}
        for order in result['orders']:order['limit']=limits[str(order['asset'])]
    return dict(result,pair=intent['pair'],ids=ids,beta=float(beta),direction=intent['direction'],
        entry_limits=limits,forecast_source=intent['forecast_source'],horizon=24,
        reverse_preview=plan['exit_preview'],intended_gross=float(abs(np.array(plan['target']))@mids),limit_buffer_bps=buffer)


def main():
    dest=OUT/'strategy';dest.mkdir(exist_ok=False)
    spec,cfg,p,d,de,books=setup();books.update(book_intent_lifetime_hours=1,funding_forecast_hours=24)
    gatepath=ROOT/'experiments/decision_gate_pairs_20261017/artifacts/par_gate_state.npz';state=load(gatepath)
    priorpath=ROOT/'experiments/decision_gate_pairs_20261017/artifacts/books/results.json';prior=json.loads(priorpath.read_text())
    models={'validation':json.loads((OUT/'initial_model.json').read_text()),'later':json.loads((OUT/'later_model.json').read_text())}
    sources=json.loads((OUT/'evaluation_frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),gatepath,priorpath,OUT/'initial_model.json',OUT/'later_model.json',OUT/'results.json']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,reference_forecasts_unchanged=True,
        candidate_sizes=[.25,.5,1.],comparison='independent source decisions; no full horizon equity curve'))
    trials=[];summaries=[];exact=0;source_plans=[]
    for tape in spec['tapes']:
        streams,meta=load_tape(ROOT/tape)
        for old in prior['rows']:
            if old['tape']!=tape:continue
            now=old['time'];i=int(np.searchsorted(d['times'],now,side='right')-1)
            if 'pair' not in old:continue
            ids=[p['symbols'].index(s) for s in old['pair']]
            k=next(k for k,pair in enumerate(state['pairs']) if list(pair)==ids)
            intent=source_intent(state,i,k,d,p,old['direction'])
            quotes={s:asof(streams[s],now*1000,books) for s in intent['pair']}
            plan=plan_entry(intent,quotes,now,books,p,books['scenarios'][old['scenario']],np.zeros(10),p['initial'],d['sigma'][i])
            if old['model']=='gate' and (not state['economic_available'][i,k] or state['economic_action'][i,k]!=old['direction']):
                plan={**plan,'accepted':False,'original_reason':plan['reason'],'reason':'decision_gate_rejected'}
            assert plan==old['plan'],(tape,now,old['model'],old['scenario'])
            exact+=1
        phase='validation' if tape in spec['validation_tapes'] else 'later' if tape in spec['later_tapes'] else None
        if phase is None:continue
        model=models[phase];first=math.ceil(meta['start_ms']/1000/spec['cadence_seconds'])*spec['cadence_seconds']
        for scenario,sc in books['scenarios'].items():
            counts=Counter()
            for now in range(first,meta['end_ms']//1000+1,spec['cadence_seconds']):
                i=int(np.searchsorted(d['times'],now,side='right')-1)
                if i<0 or i>=len(state['directions']) or now-d['times'][i]>=300:continue
                alternatives=[];originals=[];counts['source_times']+=1
                for k,raw_ids in enumerate(state['pairs']):
                    if not state['directions'][i,k]:continue
                    for direction in [-1,1]:
                        intent=source_intent(state,i,k,d,p,direction);ids=intent['ids']
                        quotes={s:asof(streams[s],now*1000,books) if s in streams else None for s in intent['pair']}
                        if any(v is None for v in quotes.values()):counts['missing_fresh_book']+=1;continue
                        if not state['economic_available'][i,k] or state['economic_action'][i,k]!=direction:
                            counts['decision_gate_rejected']+=1;continue
                        marks=d['par_close'][i].copy()
                        for symbol,j in zip(intent['pair'],ids):marks[j]=quotes[symbol]['mid']
                        anchors=np.asarray(intent['source_prices']);beta=intent['source_beta']*anchors[0]/anchors[1]*marks[ids[1]]/marks[ids[0]]
                        prepared=[dict(entry_levels=quotes[s]['book']['levels'][1 if side>0 else 0],side=side) for s,side in zip(intent['pair'],direction*np.array([1.,-1.]))]
                        cap=gross_capacity(ids,beta,prepared,marks,d['sigma'][i],p,sc,spec)
                        seen=set()
                        for fraction in spec['size_fractions']:
                            local=copy.deepcopy(books)
                            if fraction<1:local['pair_max_gross']=cap*fraction
                            plan=plan_entry(intent,quotes,now,local,p,sc,np.zeros(10),p['initial'],d['sigma'][i])
                            counts[f'size_{fraction}:{plan["reason"]}']+=1
                            source_plans.append(dict(tape=tape,time=now,scenario=scenario,pair=intent['pair'],direction=direction,size_fraction=fraction,plan=plan))
                            if not plan['accepted']:continue
                            pair_index=spec['pairs'].index(intent['pair'])
                            for buffer in spec['limit_buffers_bps']:
                                candidate=enriched(plan,intent,quotes,p,buffer)
                                identity=(tuple(candidate['target']),tuple(candidate['entry_limits'].values()))
                                if identity in seen:continue
                                seen.add(identity)
                                x,names=features(candidate,quotes,now,marks,d['sigma'][i],p,sc,pair_index)
                                learned=predict(model,x,now);constant=predict(model,x,now,True)
                                candidate.update(size_fraction=fraction,execution_prediction=learned,
                                    learned_value=score_plan(candidate,learned),constant_value=score_plan(candidate,constant))
                                record=(candidate,quotes,marks)
                                alternatives.append(record)
                                if fraction==1 and buffer==10:originals.append(record)
                for arm in ['original_max','size_only','constant_execution','learned_execution']:
                    choices=originals if arm=='original_max' else alternatives
                    key='learned_value' if arm=='learned_execution' else 'constant_value' if arm=='constant_execution' else None
                    if key:
                        choices=[r for r in choices if r[0][key]['value']>r[0]['intended_gross']*books['entry_buffer_bps']/10000]
                    if not choices:continue
                    plan,quotes,marks=max(choices,key=lambda r:r[0][key]['value'] if key else r[0]['expected_net_cash'])
                    result=episode(plan,streams,quotes,now,meta['end_ms'],marks,d['sigma'][i],p,sc,books,spec)
                    counts[f'{arm}:selected']+=1
                    if result is None:counts[f'{arm}:unobserved_arrival_window']+=1;continue
                    trials.append(dict(tape=tape,time=now,phase=phase,scenario=scenario,arm=arm,plan=plan,result=result))
                    counts[f'{arm}:outcome_{result["outcome"]}']+=1
            summaries.append(dict(tape=tape,phase=phase,scenario=scenario,counts=dict(counts)))
        print('STRATEGY',tape,'trials',len(trials),'exact',exact,flush=True)
    verify(sources)
    save(dest/'results.json',dict(trials=trials,summaries=summaries,source_plans=source_plans,exact_old_plans=exact,
        selected_unique_source_opportunities=len({(r['time'],tuple(r['plan']['pair']),r['plan']['direction']) for r in trials}),
        full_horizon_return=None,financial_score=None,independent_hypothetical_decisions=True,
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,protected=protected()))
    print('DONE strategy',len(trials),'exact old plans',exact,flush=True)


if __name__=='__main__':main()
