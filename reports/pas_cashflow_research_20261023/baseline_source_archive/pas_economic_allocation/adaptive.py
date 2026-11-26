"""Causal risk-budget adaptation of the repaired anchor, not forecast refitting."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .anchored import AnchoredController,RATIO
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,OUT as PRIOR,save,sha,native

OUT=ROOT/'reports/pas_economic_adaptive_20261022'


def budget(times,prices,funding,weights):
    r=np.zeros_like(weights);r[1:]=prices[1:]/prices[:-1]-1
    lag=np.vstack([np.zeros(weights.shape[1]),weights[:-1]])
    proxy=(lag*r).sum(1)-(lag*funding).sum(1)-.0019*abs(weights-lag).sum(1)
    # This is an allocation feature, not reported ledger PnL. Actual backtest
    # still pays every entry/exit fee and uses delayed independent fills.
    factor=np.ones(len(times));trace=[];daily=[];last=None;held=1.
    for i,t in enumerate(times):
        if t%86400==0:
            if last is not None and i-last==24:daily.append(float(proxy[last+1:i+1].sum()))
            last=i
            if len(daily)>=30:
                x=np.array(daily[-90:]);ew=np.exp2(-np.arange(len(x)-1,-1,-1)/30);ew/=ew.sum()
                mean=float(ew@x);sd=float(np.sqrt(ew@((x-mean)**2)))
                conviction=max(0.,float(np.tanh(np.sqrt(30)*mean/max(sd,1e-8))))
                held=.5+1.5*conviction
                trace.append(dict(time=int(t),last_reward_time=int(t),mean=mean,sd=sd,factor=held))
        factor[i]=held
    return factor,trace


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    parent=json.loads((PRIOR/'protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),pointer=parent['pointer'],
      nominee='adaptive_augmented',arms=['incumbent','size_only','adaptive_anchor','adaptive_augmented'],
      hypothesis='Uniformly increasing the risk budget amplified a weak later period. Scale the repaired anchor BEFORE its final risk gate using only previously completed costed daily proxy returns; no later-period switch or fitted dates.',
      formula='Pre-existing30day-half-life90day-history evidence rule. Factor0.5+1.5*max(tanh(sqrt(30)*mean/std),0) times original scale. Range0.5-2 of original, NOT mandatory full investment. This changes capital allocation, not forecasting accuracy.',
      gate='The causal scalar multiplies pre-gate and post-gate amounts identically; final protected caps remain binding. No probability cutoff selected from outcomes.',
      model='Reuse sealed economic forecast and July-selected lambda5 without refitting.',
      promotion='PnL and Q above incumbent in full and later periods under both costs; no accounting/timing/hard risk breaches. Same-size repaired anchor and original size-only are mandatory diagnostics; preserve all failures.',
      independence=False,history_reused=True,changed_after_budget_results=True,
      final_risk_inputs='reports/strategy_repairs_20261021/final_risk_targets.npz',
      live_changed=False,orders_sent=0,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    r=dict(np.load(ROOT/p['final_risk_inputs']));d=dict(np.load(ROOT/p['pointer']['inputs']))
    ix=np.searchsorted(d['market_times'],r['times']);np.testing.assert_array_equal(d['market_times'][ix],r['times'])
    f,trace=budget(r['times'],d['close'][ix],d['funding'][ix],r['final_mix_risk']*RATIO)
    np.savez_compressed(OUT/'targets.npz',times=r['times'],weights=r['final_mix_risk']*RATIO*f[:,None],protected=r['protected'],factor=f)
    save(OUT/'budget_trace.json',trace)
    files=[Path(__file__),Path(__file__).with_name('anchored.py'),Path(__file__).with_name('replay.py'),
           OUT/'protocol.json',OUT/'targets.npz',ECONOMIC/'forecasts.npz',ROOT/p['pointer']['inputs']]
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ECONOMIC/'forecasts.npz'));ix=np.searchsorted(z['times'],d['times'])
    a=dict(np.load(OUT/'targets.npz'));ia=np.searchsorted(a['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(a['times'][ia],d['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    lam=json.loads((ECONOMIC/'selected.json').read_text())['selected']['risk_aversion']
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    controls=json.loads((ROOT/'reports/pas_covariance_v2_20261021/reference_results.json').read_text());rows=[]
    for w in windows:
        rows.extend([r for r in controls if r['arm'] in ['incumbent','size_only'] and r['phase']==w['phase'] and r['scenario']==w['scenario']])
        for arm in p['arms'][2:]:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                c=AnchoredController(d['times'],z['forecast'][ix],z['carry'][ix],a['weights'][ia],a['protected'][ia],lam,augment=arm.endswith('augmented'))
                result=replay(d,s,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                result['allocation_trace']=c.trace
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as f:json.dump(result,f,default=native,allow_nan=False)
            row=summarize(result,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
