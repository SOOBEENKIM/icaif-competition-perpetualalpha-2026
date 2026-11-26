"""Chronological mixture experiment with exact previous shared-account controls."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,PARENT,setup,load,save,sha,verify,protected
from .model import fit
from .portfolio import replay
from experiments.venue_execution_value_20261018.research import plain,readrun
from experiments.par_direct_forecast_20261017.data import ts

CONTROL='first_dense_hourlyhedge_decisiongate_24'


def cache_for(models,times,dates):
    cache={}
    for model in models:
        now=model['fit_time'];index=dates.index(now);until=dates[index+1] if index+1<len(dates) else int(times[-1])+1
        for at in times[(times>=now)&(times<until)]:cache[model['horizon'],int(at)]=model
    return cache


def main():
    spec,cfg,p,d,de=setup();rows=json.loads((OUT/'labels.json').read_text())['rows']
    assert (HERE/'dataset.log').read_text().rstrip().startswith('DONE labels')
    (OUT/'runs').mkdir(exist_ok=False)
    prior=json.loads((PARENT/'results.json').read_text());gate=ROOT/'experiments/decision_gate_pairs_20261017/artifacts/par_extended_gate_state.npz'
    sources=json.loads((OUT/'dataset_frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),OUT/'labels.json',OUT/'state.npz',PARENT/'value_models.json',PARENT/'price_predictions.npz',PARENT/'results.json',gate]
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'evaluation_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    models=[];dates=[ts(t) for t in spec['fit_dates']]
    for h in spec['horizons_hours']:
        for now in dates:
            model=fit(rows,now,h,spec)
            if model is not None:models.append(model)
            print('FIT',h,now,None if model is None else dict(initial=model['initial_rows'],cash=model['mature_rows'],classes=model['class_counts']),flush=True)
    save(OUT/'models.json',models)
    cache=cache_for(models,d['times'],dates)
    old_cache=cache_for(json.loads((PARENT/'value_models.json').read_text()),d['times'],dates)
    prices=load(PARENT/'price_predictions.npz');gate_state=load(gate)
    base_cfg=copy.deepcopy(cfg);base_cfg.pop('venue_cash');base_cfg['conditional_completion']={'risk_aversion':0.}
    exact=[]
    for row in prior['rows']:
        phase,arm,sc=row['phase'],row['arm'],row['scenario']
        old=readrun(PARENT/'runs'/f'{phase}-{arm}-{sc}.json.gz')
        if arm in ['first_strategy','second_strategy',CONTROL]:rcfg=base_cfg;state=gate_state
        else:
            rcfg=copy.deepcopy(cfg);rcfg.update(venue_model_kind='price' if '_price_' in arm else 'value',venue_models=old_cache,venue_price_predictions=prices);state=de
        run=plain(replay(d,state,None,rcfg,p,arm,sc,old['start'],old['end']))
        for key in ['metrics','events','orders','fills','owner_fills','snapshots','decisions']:assert run[key]==old[key],(phase,arm,sc,key)
        exact.append(dict(phase=phase,arm=arm,scenario=sc));print('EXACT',phase,arm,sc,flush=True)
    new=[];lookup={(r['phase'],r['arm'],r['scenario']):r['metrics'] for r in prior['rows']}
    arms=[f'first_dense_hurdle_{kind}_{h}' for h in spec['horizons_hours'] for kind in spec['kinds']]
    def record(phase,arm,sc,start,end):
        rcfg=copy.deepcopy(cfg);rcfg.update(hurdle_model_kind=arm.split('_')[-2],hurdle_models=cache)
        run=replay(d,de,None,rcfg,p,arm,sc,ts(start),ts(end))
        with gzip.open(OUT/'runs'/f'{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(plain(run),f,allow_nan=False)
        m=run['metrics'];new.append(dict(phase=phase,arm=arm,scenario=sc,metrics=m));lookup[phase,arm,sc]=m
        print('RUN',phase,arm,sc,'PnL',m['net_pnl'],'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)
    phase,start,end=spec['windows'][0]
    for arm in arms:
        for sc in ['base','stress']:record(phase,arm,sc,start,end)
    ranking=[]
    for arm in arms:
        checks=[];h=arm.rsplit('_',1)[1]
        for sc in ['base','stress']:
            m=lookup['validation',arm,sc];controls=[lookup['validation',a,sc] for a in ['first_strategy','second_strategy',CONTROL]]
            old=lookup['validation',f'first_dense_venue_value_{h}',sc]
            checks.append(dict(scenario=sc,positive_pnl=m['net_pnl']>0,no_risk_breach=m['risk_breach_bars']==0,
                pnl_advantage=min(m['net_pnl']-c['net_pnl'] for c in controls),q_advantage=min(m['financial_score']-c['financial_score'] for c in controls),
                prior_value_pnl_advantage=m['net_pnl']-old['net_pnl'],prior_value_q_advantage=m['financial_score']-old['financial_score']))
        promote=all(c['positive_pnl'] and c['no_risk_breach'] and c['pnl_advantage']>0 and c['q_advantage']>0 for c in checks)
        ranking.append(dict(arm=arm,checks=checks,promote=promote,minimum_pnl_advantage=min(c['pnl_advantage'] for c in checks)))
    ranking.sort(key=lambda r:(-r['minimum_pnl_advantage'],r['arm']))
    selection=dict(selected=next((r['arm'] for r in ranking if r['promote']),None),ranking=ranking,
        frozen_before_later_accounts=True,historical_independence=False,not_official_submission_requirement=True)
    save(OUT/'selection.json',selection);digest=sha(OUT/'selection.json')
    for phase,start,end in spec['windows'][1:]:
        for arm in arms:
            for sc in ['base','stress']:record(phase,arm,sc,start,end)
    assert sha(OUT/'selection.json')==digest;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=prior['rows']+new,new_rows=new,selection=selection,exact_controls=exact,new_account_replays=len(new),
        model_bundles=len(models),tests_passed=7,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,protected=protected()))
    save(HERE/'work_state.json',dict(status='progress_hurdle_and_tree_account_comparison_completed',selection=selection,
        next_action='Audit calibration and selected cash outcomes; test the unchanged source plans on permitted old books and retain or reject.',
        new_account_replays=len(new),exact_controls=len(exact),orders_sent=0,live_strategy_changed=False,goal_complete=False,no_progress_streak=0))
    print('DONE mixture accounts',len(new),'controls',len(exact),'selected',selection['selected'],flush=True)


if __name__=='__main__':main()
