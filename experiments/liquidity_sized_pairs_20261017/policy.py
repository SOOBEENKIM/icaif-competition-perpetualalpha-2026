"""Causal initial capacity only; the original executor retains actual volume."""
import numpy as np
from experiments.dense_history_pairs_20261017.policy import proposals as original

def capacity_volume(volume,trades,window):
    volume=np.asarray(volume,float);trades=np.asarray(trades,float)
    if volume.shape!=trades.shape or volume.ndim!=2 or not isinstance(window,int) or window<2:
        raise ValueError('Aligned market arrays and integer history window >=2 required')
    if np.any(volume[np.isfinite(volume)]<0) or np.any(trades[np.isfinite(trades)]<0):
        raise ValueError('Negative observed volume/trades')
    observed=np.isfinite(volume)&np.isfinite(trades)
    effective=np.where(observed,np.where(trades>0,volume,0.),np.nan)
    result=np.full(volume.shape,np.nan)
    if len(volume)<window:return result
    windows=np.lib.stride_tricks.sliding_window_view(effective,window,axis=0)
    valid=np.isfinite(windows).all(-1)
    median=np.median(windows,axis=-1)
    result[window-1:]=np.where(valid,np.minimum(effective[window-1:],median),np.nan)
    return result

def prepare(d,cfg):
    if 'entry_capacity' not in cfg:return d
    step=np.diff(d['times'])
    if not len(step) or np.any(step!=step[0]) or step[0] not in (300,3600):
        raise ValueError('Contiguous completed five-minute or hourly observations required')
    return {**d,'entry_capacity_volume':capacity_volume(d['par_volume'],d['par_trades'],cfg['entry_capacity']['window_bars'])}

def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    if 'entry_capacity' not in cfg:
        return original(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    if not de.get('dense'):raise ValueError('This capacity experiment is only for the fixed dense pair states')
    capacity=d['entry_capacity_volume'];active=np.array(directions,copy=True)
    for k,ids in enumerate(de['pairs']):
        if not np.isfinite(capacity[i,ids]).all():active[k]=0
    proposed=original(i,{**d,'par_volume':capacity},de,pred,h,cfg,p,sc,qty,equity,reserved,active,arm,risk_qty=risk_qty)
    for row in proposed:
        ids=row['ids'];row.update(capacity_window_bars=cfg['entry_capacity']['window_bars'],
            observed_volume_units=d['par_volume'][i,ids].tolist(),entry_capacity_volume_units=capacity[i,ids].tolist())
        assert np.all(abs(np.asarray(row['target']))<=sc['participation']*capacity[i,ids]+1e-10)
    return proposed
