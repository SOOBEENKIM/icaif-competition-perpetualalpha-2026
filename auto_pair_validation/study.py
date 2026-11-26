"""Predeclared matched ablations with the incumbent live files untouched."""
import copy
import gzip
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from auto_pair_selection.study import ROOT, OUT as PRIOR, read, save, sha, verify as verify_prior
from auto_pair_selection.selection import PAIRS, IDS
from auto_pair_selection.signals import prepare_features, predict_hour
from competition_live.profile import BUNDLE, load_profile
from experiments.rolling_feature_pairs_20261017.model import fit as direct_fit
from experiments.decision_gate_pairs_20261017.model import fit as gate_fit
from .policy import ValidationSpec, fit_calibration, calibrate, current_action, monitor, apply_entry_gate

OUT=ROOT/'reports/auto_pair_validation_20261023'


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    old=verify_prior()
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        primary='validated_all45',arms=['expanded_only','calibrated_only','validated_all45'],
        baselines=['incumbent_fixed_two','automatic_45'],
        input=old['input'],input_sha256=old['input_sha256'],
        spec=asdict(ValidationSpec()),direct_spec=old['direct_spec'],gate_spec=old['gate_spec'],
        raw_fit_start=1777593600, # May1; missing training history means unavailable.
        gate_fit_start=old['start'],evaluation_start=old['start'],
        universe=[list(p) for p in PAIRS],
        selection='No named-pair, raw-correlation, raw-beta or retrospective profit exclusion. All45 receive daily independent ridge fits. Existing source-beta clipping and current liquidity/execution checks remain unchanged.',
        raw_predictions='Every past forecast is from the daily model fitted before that forecast. Training uses source+24h maturity strictly before daily fit, existing90d/6h/recency parameters.',
        calibration='Every day use up to60 previously forecast daily observations with strictly mature labels, at least20. Fit shrunk affine normalized-return calibration. Slope constrained0..1. HAC2 mean-estimate uncertainty; no forecast-success guarantee.',
        monitoring='Every day assess up to30 previously published calibrated daily forecasts, not a newly fitted model on its own calibration rows. Require20 valid days,3 nonzero signals, MAE below zero forecast, mean proxy net return above one HAC2 standard error. Proxy38bp cost, no fabricated fills.',
        current='Current calibrated return magnitude minus one mean-estimation standard error must exceed38bp proxy roundtrip plus5bp buffer. Current venue prices, depth, lots, funding, risk and fees remain an additional veto in the actual planner.',
        held='Entry gate never drops IDs, held forecasts or exit management. Paid renewal and hedge/stop lifecycle unchanged; calibrated forecasts replace means in calibrated arms, raw fallback only while calibration unavailable and NO new entries allowed.',
        fixed='Same incumbent pullback_any3 base,50k owned budget+50k reserve,one active pair,unchanged costs/risk and all original six reference replay windows.',
        hypotheses=['Removing the hand-chosen formation thresholds restores consideration of previously blocked old opportunities.',
                    'Calibration of already-published forecasts reduces exaggerated edge estimates.',
                    'Rolling monitoring abstains when past forecast skill and after-cost proxy performance are insufficient.'],
        history_reused=True,independent_test=False,no_parameter_search=True,
        evaluation='All arms, all windows, both costs; preserve failed arms. No selection of a new primary after results. PAS signals reported separately; no reference-return substitution for PAS fills.',
        promotion='No automatic live deployment. Require primary PnL/Q improvement over incumbent across six conditions, no timing/accounting/risk violations, and further PAS execution validation.',
        orders_sent=0,live_changed=False)
    save(OUT/'protocol.json',p)
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [*Path(__file__).parent.glob('*.py'),OUT/'protocol.json']})
    print('FROZEN primary=validated_all45; no parameter search',flush=True)


def verify():
    verify_prior()
    for f,digest in read(OUT/'freeze.json').items():assert sha(ROOT/f)==digest,f
    p=read(OUT/'protocol.json');assert sha(ROOT/p['input'])==p['input_sha256']
    return p


def fit_raw(d,f,y,maturity,terminal,source,direct_spec,gate_spec,fit_gates):
    t=d['market_times'];unit=f['vol']*np.sqrt(24)
    window=(t>=source-direct_spec['lookback_days']*86400)&(maturity<source)&(t%(direct_spec['sample_hours']*3600)==0)
    finite=np.isfinite(f['x']).all(-1)&f['fresh']&terminal&np.isfinite(y)&np.isfinite(unit)&(unit>0)
    model=dict(source=int(source),direct={},gate={},failures=[])
    for k in range(len(PAIRS)):
        ix=np.flatnonzero(window&finite[:,k])
        m=direct_fit(f['x'][ix,k],y[ix,k],np.zeros(len(ix)),maturity[ix],source,direct_spec)
        if m is not None:model['direct'][k]=m
        if fit_gates:
            try:g=gate_fit(f['x'][ix,k],y[ix,k],unit[ix,k],maturity[ix],source,gate_spec)
            except ArithmeticError as error:
                model['failures'].append(dict(pair=list(PAIRS[k]),reason=str(error)));g=None
            if g is not None:model['gate'][k]=g
    return model


def build():
    p=verify();spec=ValidationSpec(**p['spec'])
    d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times'];n=len(t);q=len(PAIRS)
    f,y,maturity,terminal=prepare_features(d);unit=f['vol']*np.sqrt(24)
    actual=y*unit;actual[~terminal]=np.nan
    raw_mu=np.full((n,q),np.nan);mean=raw_mu.copy();se=raw_mu.copy()
    calibrated_action=np.zeros((n,q),int);validation=np.zeros((n,q),bool)
    calibrated=np.zeros((n,q),bool);directions=np.zeros((n,q),int)
    gate_available=np.zeros((n,q),bool);gate_action=np.zeros((n,q),int)
    raw_stamps=np.full((n,q),-1,np.int64);cal_stamps=raw_stamps.copy()
    audits=[]
    for day in np.unique(t[t>=p['raw_fit_start']]//86400*86400):
        started=time.monotonic();cache=OUT/'models'/f'{day}.json'
        if cache.exists():
            model=read(cache)
            for key in ('direct','gate'):model[key]={int(k):v for k,v in model[key].items()}
        else:
            model=fit_raw(d,f,y,maturity,terminal,int(day),p['direct_spec'],p['gate_spec'],day>=p['gate_fit_start'])
            save(cache,model)
        calibrators={};checks=[]
        for k in range(q):
            # raw_mu contains only forecasts actually published by this loop.
            fresh_target=np.where(terminal[:,k],y[:,k],np.nan)
            c=fit_calibration(t,raw_mu[:,k]/unit[:,k],fresh_target,maturity,int(day),spec)
            if c is not None:calibrators[k]=c
            checks.append(monitor(t,mean[:,k],se[:,k],actual[:,k],maturity,int(day),spec))
        for i in np.flatnonzero((t>=day)&(t<day+86400)):
            forecast=predict_hour(f,i,model,p['direct_spec'],p['gate_spec'])
            raw_mu[i]=forecast['mu'];directions[i]=forecast['directions']
            gate_available[i]=forecast['gate_available'];gate_action[i]=forecast['gate_action']
            raw_stamps[i]=np.where(forecast['directions'],day,-1)
            for k in range(q):
                v=calibrate(calibrators.get(k),raw_mu[i,k],unit[i,k])
                if v is None:continue
                mean[i,k],se[i,k]=v;calibrated[i,k]=True;cal_stamps[i,k]=day
                calibrated_action[i,k]=current_action(*v,spec)
                validation[i,k]=checks[k]['eligible']
        audit=dict(source=int(day),raw_models=len(model['direct']),gate_models=len(model['gate']),
                   failures=model['failures'],calibrators=calibrators,
                   pairs=[dict(pair=list(pair),pair_index=k,**checks[k]) for k,pair in enumerate(PAIRS)])
        for m in model['direct'].values():assert m['last_maturity']<day
        for c in calibrators.values():assert c['last_label_maturity']<day
        for c in checks:assert c['last_label_maturity'] is None or c['last_label_maturity']<day
        audits.append(audit)
        print(datetime.fromtimestamp(int(day),timezone.utc).date(),'raw',len(model['direct']),
              'calibrated',len(calibrators),'validated',sum(x['eligible'] for x in checks),
              'seconds',round(time.monotonic()-started,2),flush=True)
    state=dict(dense=np.array(True),pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],(n,q)),
               source_prices=d['close'],beta=f['beta'],vol=f['vol'],forecast_times=t,
               raw_mu=raw_mu,mu=np.where(calibrated,mean,raw_mu),directions=directions,
               calibration_available=calibrated,calibrated_mean=mean,mean_uncertainty=se,
               calibrated_action=calibrated_action,validated=validation,
               gate_available=gate_available,gate_action=gate_action,raw_fit_source=raw_stamps,
               calibration_fit_source=cal_stamps)
    np.savez_compressed(OUT/'states.npz',times=t,**state)
    np.savez_compressed(OUT/'labels.npz',times=t,actual=actual,maturity=maturity,unit=unit)
    save(OUT/'monitoring_history.json',audits)
    old=dict(np.load(PRIOR/'states.npz',allow_pickle=False));valid=t>=p['evaluation_start']
    np.testing.assert_array_equal(old['times'],t)
    np.testing.assert_allclose(raw_mu[valid],old['mu'][valid],rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(directions[valid],old['directions'][valid])
    gate_mask=old['gate_available']&valid[:,None]
    np.testing.assert_array_equal(gate_action[gate_mask],old['gate_action'][gate_mask])
    save(OUT/'parity.json',dict(all45_raw_forecasts_unchanged=True,old_gate_actions_unchanged_where_previously_fitted=True,
                              no_future_training_or_monitoring_labels=True,orders_sent=0))
    verify()


def reference():
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=verify();assert read(OUT/'parity.json')['all45_raw_forecasts_unchanged']
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(OUT/'states.npz',allow_pickle=False));st=state.pop('times');ii=np.searchsorted(st,d['times'])
    np.testing.assert_array_equal(st[ii],d['times'])
    mapped={k:(v[ii] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(st) else v) for k,v in state.items()}
    raw=dict(np.load(ROOT/p['input'],allow_pickle=False))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
    windows=[w for w in read(BUNDLE/'expected.json')['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    rows=[]
    for w in windows:
        for arm in p['arms']:
            cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
            cfg['dense_pairs']=[list(x) for x in PAIRS];signals=apply_entry_gate(mapped,arm)
            path=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if path.exists():
                with gzip.open(path,'rt') as fp:r=json.load(fp)
            else:
                c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][ix],
                  z['all_market_signed_core_v1_protected'][ix],price_times=raw['market_times'],prices=raw['close'])
                r=replay(d,signals,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            row=summarize(r,arm,w['phase'],w['scenario'])
            row['entered_pairs']=sorted({tuple(x['pair']) for x in r['events'] if x['event']=='pair_intent'})
            assert row['timing_pass'] and row['accounting_error']<1e-6
            rows.append(row);save(OUT/'reference_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
