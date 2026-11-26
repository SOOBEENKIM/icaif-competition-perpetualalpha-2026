"""Matched price-label and execution-value accounts, with chronological selection."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,setup,load,save,sha,ts,verify,protected
from .portfolio import replay
from .model import fit
from .control_audit_v2 import compare
from experiments.par_direct_forecast_20261017.model import labels as price_labels

GATE=ROOT/'experiments/decision_gate_pairs_20261017/artifacts'
NATIVE=ROOT/'experiments/shared_execution_pairs_20261017/artifacts'
CONTROL='first_dense_hourlyhedge_decisiongate_24'


def readrun(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def plain(x):
    return json.loads(json.dumps(x,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))


def price_models(d,de,spec):
    old=json.loads((ROOT/'experiments/par_direct_forecast_20261017/protocol.json').read_text())
    x=de['x'];times=d['times'];dates=[ts(s) for s in spec['fit_dates']]
    predictions={f'ridge_{h}':np.full(de['beta'].shape,np.nan) for h in spec['horizons_hours']};records=[]
    for h in spec['horizons_hours']:
        target,maturity=price_labels(d,de,h)
        for j,at in enumerate(dates):
            until=dates[j+1] if j+1<len(dates) else int(times[-1])+1
            usable=np.isfinite(x).all(-1)&de['fresh']&np.isfinite(target)&(de['beta']>=.1)&(de['beta']<=10)
            mask=usable&((times>=at-old['training_window_days']*86400)&(maturity<at)&(times%old['fit_sampling_seconds']==0))[:,None]
            ii,jj=np.where(mask)
            if len(ii)<old['minimum_training_rows']:continue
            xx=x[ii,jj];yy=target[ii,jj];bounds=np.quantile(yy,[.01,.99]);yy=np.clip(yy,*bounds)
            mean=xx.mean(0);scale=np.maximum(xx.std(0),1e-6);z=np.c_[np.ones(len(xx)),(xx-mean)/scale]
            penalty=np.eye(z.shape[1])*old['ridge_penalty'];penalty[0,0]=0
            coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@yy/len(z))
            vi,vj=np.where(np.isfinite(x).all(-1)&((times>=at)&(times<until))[:,None])
            estimates=np.clip(np.c_[np.ones(len(vi)),(x[vi,vj]-mean)/scale]@coef,*bounds)
            predictions[f'ridge_{h}'][vi,vj]=estimates/10000
            records.append(dict(horizon=h,fit_time=at,rows=len(ii),last_maturity=int(maturity[ii].max()),
                                 mean=mean.tolist(),scale=scale.tolist(),coefficient=coef.tolist(),bounds=bounds.tolist()))
    original=load(ROOT/'experiments/par_direct_forecast_20261017/artifacts/predictions.npz')
    stop=int(np.searchsorted(times,ts('2026-10-16T12:00:00')))
    for name,value in predictions.items():np.testing.assert_array_equal(value[:stop],original[name][:stop],err_msg=name)
    return predictions,records


def main():
    spec,cfg,p,d,de=setup();dataset=json.loads((OUT/'labels.json').read_text())
    assert (HERE/'dataset.log').read_text().rstrip().splitlines()[-1].startswith('DONE venue labels')
    assert not list((OUT/'runs').iterdir()), 'Do not overwrite existing runs'
    assert not (OUT/'results.json').exists()
    verify(json.loads((OUT/'evaluation_frozen.json').read_text())['sources'])
    controls=[]
    for phase in ['validation','comparison']:
        for arm in ['first_strategy','second_strategy']:
            for sc in ['base','stress']:
                controls.append(('par',phase,arm,sc,NATIVE/'runs'/f'par-{phase}-{arm}-{sc}.json.gz'))
    gate_result=json.loads((GATE/'results.json').read_text())
    for venue,phase in gate_result['windows']:
        if venue=='reference':continue
        for sc in ['base','stress']:controls.append((venue,phase,CONTROL,sc,GATE/'runs'/f'{venue}-{phase}-{CONTROL}-{sc}.json.gz'))
    files=[Path(__file__),HERE/'control_audit_v2.py',HERE/'test_control_audit_v2.py',HERE/'control_v2_test_results.txt',OUT/'value_models.json',OUT/'fit_status.json',OUT/'price_models.json',OUT/'price_predictions.npz',OUT/'labels.json',OUT/'state.npz',ROOT/'experiments/par_direct_forecast_20261017/artifacts/predictions.npz',
           ROOT/'experiments/par_direct_forecast_20261017/protocol.json',GATE/'results.json',*[c[-1] for c in controls],
           GATE/'par_gate_state.npz',GATE/'par_extended_gate_state.npz']
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy();sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'evaluation_resume_v3_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    price=load(OUT/'price_predictions.npz');price_records=json.loads((OUT/'price_models.json').read_text())
    models=json.loads((OUT/'value_models.json').read_text());fit_status=json.loads((OUT/'fit_status.json').read_text())
    cache={};dates=[ts(t) for t in spec['fit_dates']]
    for model in models:
        now=model['fit_time'];index=dates.index(now);until=dates[index+1] if index+1<len(dates) else int(d['times'][-1])+1
        for at in d['times'][(d['times']>=now)&(d['times']<until)]:cache[model['horizon'],int(at)]=model
    base_cfg=copy.deepcopy(cfg);base_cfg.pop('venue_cash');base_cfg['conditional_completion']={'risk_aversion':0.}
    gate_states={v:load(GATE/f'{v}_gate_state.npz') for v in ['par','par_extended']}
    exact=[]
    for venue,phase,arm,sc,path in controls:
        old=readrun(path);fresh=plain(replay(d,gate_states[venue],None,base_cfg,p,arm,sc,old['start'],old['end']))
        normalized_fields=compare(fresh,old)
        exact.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,added_order_annotations=normalized_fields));print('EXACT',venue,phase,arm,sc,flush=True)
    arms=['first_strategy','second_strategy',CONTROL]+[f'first_dense_venue_{kind}_{h}' for h in spec['horizons_hours'] for kind in ['price','value']]
    rows=[];lookup={}
    def record(phase,arm,sc,start,end):
        if arm in ['first_strategy','second_strategy',CONTROL]:rcfg=base_cfg;state=gate_states['par_extended']
        else:
            rcfg=copy.deepcopy(cfg);rcfg.update(venue_model_kind='price' if '_price_' in arm else 'value',
                                               venue_models=cache,venue_price_predictions=price);state=de
        run=replay(d,state,None,rcfg,p,arm,sc,start,end)
        with gzip.open(OUT/'runs'/f'{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        m=run['metrics'];rows.append(dict(phase=phase,arm=arm,scenario=sc,metrics=m));lookup[phase,arm,sc]=m
        print('RUN',phase,arm,sc,'PnL',round(m['net_pnl'],8),'Q',m['financial_score'],'complete',m['completed_pairs'],flush=True)
    phase,start,end=spec['windows'][0]
    for arm in arms:
        for sc in ['base','stress']:record(phase,arm,sc,ts(start),ts(end))
    ranking=[]
    for horizon in spec['horizons_hours']:
        arm=f'first_dense_venue_value_{horizon}';checks=[]
        for sc in ['base','stress']:
            a=lookup['validation',arm,sc];references=[lookup['validation',c,sc] for c in ['first_strategy','second_strategy',CONTROL]]
            b=lookup['validation',f'first_dense_venue_price_{horizon}',sc]
            checks.append(dict(scenario=sc,positive_pnl=a['net_pnl']>0,no_risk_breach=a['risk_breach_bars']==0,
                pnl_advantage=min(a['net_pnl']-r['net_pnl'] for r in references),q_advantage=min(a['financial_score']-r['financial_score'] for r in references),
                matched_price_pnl_advantage=a['net_pnl']-b['net_pnl'],matched_price_q_advantage=a['financial_score']-b['financial_score']))
        promote=all(c['positive_pnl'] and c['no_risk_breach'] and c['pnl_advantage']>0 and c['q_advantage']>0 for c in checks)
        ranking.append(dict(arm=arm,horizon=horizon,checks=checks,promote=promote,minimum_pnl_advantage=min(c['pnl_advantage'] for c in checks)))
    ranking.sort(key=lambda r:(r['minimum_pnl_advantage'],r['horizon']),reverse=True)
    selection=dict(selected=next((r['arm'] for r in ranking if r['promote']),None),ranking=ranking,
                   frozen_before_later_replays=True,historical_independence=False,sample_flags_not_official_requirements=True)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for phase,start,end in spec['windows'][1:]:
        for arm in arms:
            for sc in ['base','stress']:record(phase,arm,sc,ts(start),ts(end))
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(rows=rows,selection=selection,new_account_replays=42,exact_controls=exact,
         value_fits=len(models),price_ridge_fits=len(price_records),price_prefix_exact=True,fit_status=fit_status,tests_passed=12,
         protected=protected(),prospective_sources=len(prospective),orders_sent=0,live_strategy_changed=False,
         historical_independence=False,reserved_raw_read=False))
    save(HERE/'work_state.json',dict(status='progress_par_execution_value_accounts_completed',selection=selection,
         next_action='Audit zero/partial/closed outcomes and cost attribution, inspect old-book entry feasibility, then retain or reject with exact historical scope.',
         new_accounts=42,exact_controls=len(exact),orders_sent=0,live_strategy_changed=False,goal_complete=False,no_progress_streak=0))
    print('DONE par execution-value comparison',flush=True)


if __name__=='__main__':main()
