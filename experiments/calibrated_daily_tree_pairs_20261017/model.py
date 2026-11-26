"""Past-matured calibration of reconstructed source-time daily tree forecasts."""
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model import fit_correction


def calibrate(arrays,spec):
    times=arrays['times'];maturity=arrays['maturity'];unit=arrays['unit'];h=spec['horizon']
    if not np.all(np.diff(times)==3600) or not np.array_equal(maturity,times+h*3600):
        raise ValueError('Contiguous hourly sources and exact fixed-horizon maturities required')
    issued=arrays['prediction_tree'];target=arrays['target'];stamps=arrays['fit_stamps']
    if any(arrays[k].shape!=issued.shape for k in ['target','unit','fresh','terminal_fresh','fit_stamps','prediction_direct']):
        raise ValueError('Unaligned pair source matrices')
    current=np.isfinite(issued)&np.isfinite(unit)&(unit>0)&arrays['fresh']
    if np.any(current&((stamps<0)|(stamps>times[:,None]))):
        raise ValueError('Forecast coefficient fit was unavailable at its source')
    raw=np.divide(issued,unit,out=np.full(issued.shape,np.nan),where=np.isfinite(unit)&(unit>0))
    valid=current&arrays['terminal_fresh']&np.isfinite(target)
    out=np.full(issued.shape,np.nan);records=[]
    sampling=times%(h*3600)==0
    for i,now in enumerate(times):
        window=sampling&(maturity<now)&(maturity>=now-spec['lookback_days']*86400)
        for k in range(issued.shape[1]):
            if not current[i,k]:continue
            ix=np.flatnonzero(window&valid[:,k])
            record=fit_correction(raw[ix,k],target[ix,k],maturity[ix],int(now),spec)
            if record is None:continue
            if not np.all(np.diff(times[ix])>=h*3600):raise ValueError('Overlapping calibration source labels')
            value=record['affine_intercept']+record['affine_slope']*raw[i,k]
            out[i,k]=np.clip(value,-spec['normalized_target_clip'],spec['normalized_target_clip'])*unit[i,k]
            records.append(dict(time=int(now),pair_index=k,first_source=int(times[ix[0]]),last_source=int(times[ix[-1]]),
                current_original_fit_time=int(stamps[i,k]),original_normalized_prediction=float(raw[i,k]),
                **record))
    # A missing calibrated tree suppresses the mixture too; it is not a zero
    # forecast that accidentally leaves a basis-only trade enabled.
    mixture=(out+arrays['prediction_direct'])/2
    return {'ctree':out,'cmix':mixture},records


def mapped_states(times,reference,control_state,forecasts,arrays):
    rt=arrays['times'];ix=np.searchsorted(rt,times,side='right')-1
    if np.any(ix<0) or np.any(times-rt[ix]>=3600):raise ValueError('No current hourly source')
    if not np.array_equal(control_state['signal_times'],np.broadcast_to(rt[ix,None],control_state['signal_times'].shape)):
        raise ValueError('Mapped source times differ from retained state')
    states={}
    for kind,forecast in forecasts.items():
        state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in control_state.items()}
        state['mu']=forecast[ix].copy()
        state['directions']=(np.isfinite(state['mu'])&(control_state['directions']>0)).astype(int)
        states[kind]=state
    return states
