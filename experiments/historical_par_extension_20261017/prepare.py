"""Append validated inputs and saved forecasts; do not revise frozen history."""
import json
import numpy as np
import xgboost as xgb
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from experiments.performance_attribution_20261016.agent import BudgetMatchedAgent
from experiments.dense_history_pairs_20261017.model import design,labels,map_state
from experiments.calibrated_daily_tree_pairs_20261017.model import calibrate,mapped_states
from .collect import HERE,ROOT,OUT,RAW,ts,sha,save,load

OLD=ROOT/'experiments/par_direct_forecast_20261017/artifacts'
TREE=ROOT/'experiments/daily_tree_pairs_20261017/artifacts'
CAL=ROOT/'experiments/calibrated_daily_tree_pairs_20261017/artifacts'

def candle_values(rows,times,coin):
    by_time={}
    for row in rows:
        if row['s']!=coin or int(row['T'])+1!=int(row['t'])+300000:
            raise ValueError('Invalid candle identity or interval')
        stamp=(int(row['T'])+1)//1000
        if stamp in by_time:raise ValueError('Duplicate candle')
        by_time[stamp]=row
    result=np.full((len(times),3),np.nan)
    for i,stamp in enumerate(times):
        if int(stamp) not in by_time:raise ValueError(f'Missing completed candle {coin} {stamp}')
        row=by_time[int(stamp)];result[i]=[float(row[k]) for k in ['c','v','n']]
    if not np.isfinite(result).all() or np.any(result[:,0]<=0) or np.any(result[:,1:]<0):
        raise ValueError('Invalid OHLCV values')
    return result

def funding_values(rows,times,coin):
    # Exact settlement boundary; never borrow a record received after cutoff.
    mapping={}
    for row in rows:
        stamp=int(row['time'])
        if row['coin']!=coin or stamp%3600000:
            raise ValueError('Funding not exactly on the inherited hourly settlement grid')
        if stamp//1000 in mapping:raise ValueError('Duplicate funding settlement')
        mapping[stamp//1000]=float(row['fundingRate'])
    try:result=np.array([mapping[int(t)] for t in times])
    except KeyError as exc:raise ValueError(f'Missing actual funding {coin} {exc}') from exc
    if not np.isfinite(result).all():raise ValueError('Nonfinite settled funding')
    return result

def exact_prefix(new,old):
    for k,v in old.items():
        np.testing.assert_array_equal(new[k][:len(v)] if v.ndim else new[k],v,err_msg=k)

def prepare_data():
    cfg=json.loads((HERE/'protocol.json').read_text());snapshot=ROOT/cfg['reference_run']
    manifest=json.loads((snapshot/'manifest.json').read_text());verification=json.loads((snapshot/'verification.json').read_text())
    assert sha(snapshot/'inputs.npz')==manifest['inputs_sha256']
    assert sha(snapshot/'targets.npz')==verification['targets_sha256']
    ref=load(snapshot/'inputs.npz');targets=load(snapshot/'targets.npz')
    old_ref=load(ROOT/'practice_live/updates/20261017T050027043588Z/inputs.npz')
    exact_prefix(ref,old_ref)
    assert ref['market_times'][-1]==ts(cfg['end'])
    p=json.loads((TAIL/'baseline.json').read_text())['protocol'];symbols=p['symbols']
    market=Market(*(ref[k] for k in ['market_times','close','volume','trades','funding','observed']),tuple(symbols))
    assert np.array_equal(targets['times'],market.times)
    agent=BudgetMatchedAgent()
    for i in np.flatnonzero((np.arange(len(market.times))>=1008)&(market.times%21600==0)):
        context=np.log(market.close[i-167:i+1].T).astype('float32')
        agent.incumbent.agent.signal.cached[(int(market.times[i]),context.tobytes())]=targets['chronos_forecast'][i].copy()
    agent.prepare_history(market,{k:v for k,v in ref.items() if k not in ['market_times','funding','observed']},funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(agent.incumbent.weights,targets['default'])
    old=load(OLD/'inputs.npz');old_n=len(old['times']);end=ts(cfg['end'])
    times=np.arange(old['times'][0],end+300,300);new_times=times[old_n:]
    assert len(new_times)==288 and old['times'][-1]==ts(cfg['old_end'])
    d={k:np.concatenate([v,np.zeros((len(new_times),)+v.shape[1:],dtype=v.dtype)]) for k,v in old.items()}
    d['times']=times
    hour=load(ROOT/'experiments/historical_par_20261017/artifacts/inputs.npz')
    ft=np.arange(hour['funding_times'][0],end+3600,3600)
    rates=np.full((len(ft),len(symbols)),np.nan);rates[:len(hour['funding_times'])]=hour['funding_history']
    audit=[]
    from experiments.par_direct_forecast_20261017.collect import MAPPING
    overlap=times[(times>ts(cfg['fetch_start']))&(times<=old['times'][-1])]
    for j,symbol in enumerate(symbols):
        for venue in ['par','reference']:
            rows=json.loads((RAW/f'{venue}_{symbol}_5m.json').read_text())['response']
            coin='par:'+symbol if venue=='par' else MAPPING[symbol]
            values=candle_values(rows,np.r_[overlap,new_times],coin)
            for column,key in enumerate(['close','volume','trades']):
                np.testing.assert_array_equal(values[:len(overlap),column],old[f'{venue}_{key}'][np.searchsorted(old['times'],overlap),j])
                d[f'{venue}_{key}'][old_n:,j]=values[len(overlap):,column]
        hours=ft[ft>=ts(cfg['fetch_start'])]
        values=funding_values(json.loads((RAW/f'funding_{symbol}.json').read_text())['response'],hours,'par:'+symbol)
        prior=hours<=old['times'][-1];indices=np.searchsorted(ft,hours)
        np.testing.assert_array_equal(values[prior],rates[indices[prior],j])
        rates[indices[~prior],j]=values[~prior]
        audit.append(dict(symbol=symbol,appended_bars=len(new_times),appended_funding=int((~prior).sum()),overlap_bars=len(overlap),active_bars=int((d['par_volume'][old_n:,j]>0).sum())))
    assert np.isfinite(rates).all()
    ix=np.searchsorted(market.times,new_times,side='right')-1
    for key,values in [('first',agent.incumbent.weights),('second',agent.weights),('retention',agent.retention),('sigma',agent.sigma)]:
        original_ix=np.searchsorted(market.times,old['times'],side='right')-1
        np.testing.assert_array_equal(values[original_ix],old[key],err_msg=key)
        d[key][old_n:]=values[ix]
    d['signal_time'][old_n:]=market.times[ix]
    for i,t in enumerate(new_times,start=old_n):
        prior=np.searchsorted(ft,t,side='left')-1
        history=rates[prior-23:prior+1];assert history.shape==(24,len(symbols)) and np.isfinite(history).all()
        d['known_hourly_funding'][i]=.5*rates[prior]+.5*history.mean(0)
        if t%3600==0:
            d['funding'][i]=rates[np.searchsorted(ft,t)];d['funding_observed'][i]=True
    exact_prefix(d,old)
    np.savez_compressed(OUT/'inputs.npz',**d)
    save(OUT/'data_audit.json',dict(coverage=audit,old_prefix_exact=True,reference_prefix_exact=True,native_target_reproduction_exact=True,
        reference_snapshot=str(snapshot.relative_to(ROOT)),reference_sha256=sha(snapshot/'inputs.npz'),targets_sha256=sha(snapshot/'targets.npz'),funding_missing=0))
    print('PREPARED 288 new 5m bars, 24 funding hours x10; old inputs/targets exact',flush=True)
    return d,ref,p

def prepare_states(d,ref,p):
    spec=json.loads((ROOT/'experiments/daily_tree_pairs_20261017/protocol.json').read_text())
    cal_spec=json.loads((CAL.parent/'protocol.json').read_text());h=24
    old=load(TREE/'par_forecasts.npz');rt=ref['market_times'];n=len(old['times'])
    np.testing.assert_array_equal(rt[:n],old['times'])
    # No new fit boundary: only inference from the same saved Oct17 00UTC fits.
    if any(t%86400==0 for t in rt[n:]):raise ValueError('Extension crossed a daily refit boundary')
    features=design(rt,ref['close'],ref['volume'],ref['trades'],p['symbols'],spec['pairs'])
    target,maturity=labels(rt,ref['close'],features,h)
    terminal=np.zeros_like(features['fresh']);terminal[:-h]=features['fresh'][h:]
    arrays=dict(times=rt,target=target,maturity=maturity,unit=features['vol']*np.sqrt(h),fresh=features['fresh'],terminal_fresh=terminal)
    for k in ['unit','fresh','maturity']:np.testing.assert_array_equal(arrays[k][:n],old[k])
    known=np.isfinite(old['target']);np.testing.assert_array_equal(target[:n][known],old['target'][known])
    for k in ['prediction_direct','prediction_tree','prediction_mix']:
        arrays[k]=np.concatenate([old[k],np.full((len(rt)-n,old[k].shape[1]),np.nan)])
    arrays['fit_stamps']=np.concatenate([old['fit_stamps'],np.full((len(rt)-n,old['fit_stamps'].shape[1]),-1,np.int64)])
    records=json.loads((TREE/'par_fit_audit.json').read_text());used=[]
    for k in range(len(spec['pairs'])):
        r=max((r for r in records if r['pair_index']==k),key=lambda r:r['fit_time'])
        assert r['last_maturity']<r['fit_time']<=rt[n-1]
        path=TREE/'models/par'/r['model_file'];assert sha(path)==r['tree_sha256']
        booster=xgb.Booster();booster.load_model(path)
        for i in range(n-1,len(rt)):
            if not features['fresh'][i,k] or not np.isfinite(features['x'][i,k]).all():continue
            x=features['x'][i,k];z=np.r_[1.,(x-r['center'])/r['scale']]
            direct=float(np.clip(z@r['direct'],-4,4))*arrays['unit'][i,k]
            tree=float(np.clip(booster.predict(xgb.DMatrix(x[None,:],nthread=1))[0],-4,4))*arrays['unit'][i,k]
            if i<n:
                assert direct==old['prediction_direct'][i,k] and tree==old['prediction_tree'][i,k]
            else:
                arrays['prediction_direct'][i,k]=direct;arrays['prediction_tree'][i,k]=tree
                arrays['prediction_mix'][i,k]=(direct+tree)/2;arrays['fit_stamps'][i,k]=r['fit_time']
        used.append(dict(pair_index=k,model_file=str(path.relative_to(ROOT)),sha256=sha(path),fit_time=r['fit_time'],last_maturity=r['last_maturity']))
    forecasts,audit=calibrate(arrays,cal_spec)
    old_cal=load(CAL/'par_forecasts.npz')
    for kind in forecasts:np.testing.assert_array_equal(forecasts[kind][:n],old_cal['prediction_'+kind])
    direct=map_state(d['times'],ref,features,arrays['prediction_direct'],features['pairs'],source_hours=1,lifetime_hours=1)
    states={'direct':direct,**mapped_states(d['times'],None,direct,forecasts,arrays)}
    for kind,state in states.items():
        oldstate=load((TREE if kind=='direct' else CAL)/f'par_{kind}_24_state.npz')
        for key,value in oldstate.items():
            prefix=state[key][:len(value)] if np.ndim(value)>0 and key!='pairs' else state[key]
            np.testing.assert_array_equal(prefix,value,err_msg=f'{kind}/{key}')
        np.savez_compressed(OUT/f'{kind}_state.npz',**state)
    np.savez_compressed(OUT/'forecasts.npz',**arrays,**{f'prediction_{k}':v for k,v in forecasts.items()})
    save(OUT/'model_audit.json',dict(saved_models_reused=used,refits=0,new_source_hours=len(rt)-n,old_forecasts_exact=True,old_mapped_states_exact=True))
    save(OUT/'calibration_audit.json',audit)
    print('PREPARED 7 new hourly forecasts; saved coefficients reused, original predictions exact',flush=True)
    return states
