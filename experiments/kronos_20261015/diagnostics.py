"""Matured prediction labels; never imported by model inference or policy."""
import json
import numpy as np
from scipy.stats import rankdata
from replay_lab.common_comparison.models import timestamp
from .data import HERE, DAY, HOUR, PRICE_COLS, daily_bars, load, make_plan, write
from .run import forecast_files


def corr(x,y,rank=False):
    x,y=np.asarray(x).ravel(),np.asarray(y).ravel()
    if rank:x,y=rankdata(x),rankdata(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def stats(pred,true):
    nz=true!=0
    return dict(observations=int(pred.size),pearson=corr(pred,true),spearman=corr(pred,true,True),
                mae=float(np.abs(pred-true).mean()),mse=float(np.square(pred-true).mean()),
                zero_return_mae=float(np.abs(true).mean()),zero_return_mse=float(np.square(true).mean()),
                directional_accuracy=float(np.mean(np.sign(pred[nz])==np.sign(true[nz]))) if nz.any() else None,
                positive_target_fraction=float(np.mean(true[nz]>0)) if nz.any() else None)


def evaluate(records,bars,step,horizon,start,end,symbols):
    predictions=[];truth=[];origins=[];path_p=[];path_y=[];cross=[]
    for t,f in records.items():
        if t<start or t+horizon*step>end:continue
        i=np.searchsorted(bars['known'],t)
        ids=np.arange(i+1,i+1+horizon)
        if ids[-1]>=len(bars['known']):continue
        if not np.array_equal(bars['known'][ids],t+np.arange(1,horizon+1)*step):
            raise ValueError('Noncontiguous future labels')
        full=f['forecast']
        forecast=full[:,:,3] if full.ndim==3 else full
        actual=bars['close'][ids].T
        pred=(forecast.mean(1) if step==DAY else forecast[:,-1])/f['last']-1
        true=(actual.mean(1) if step==DAY else actual[:,-1])/f['last']-1
        predictions.append(pred);truth.append(true);origins.append(t)
        cross.append(corr(pred,true,True))
        if full.ndim==3:
            path_p.append(full/f['last'][:,None,None]-1)
            path_y.append(np.stack([bars[k][ids].T for k in PRICE_COLS],axis=-1)/f['last'][:,None,None]-1)
    if not origins:raise ValueError('No matured labels')
    pred,true=np.stack(predictions),np.stack(truth)
    out=dict(origins=len(origins),first_origin=origins[0],last_origin=origins[-1],
             last_label_maturity=origins[-1]+horizon*step,phase_end=end,
             pooled_return=stats(pred,true),per_asset={s:stats(pred[:,j],true[:,j]) for j,s in enumerate(symbols)},
             mean_cross_section_rank_ic=float(np.mean([v for v in cross if v is not None])),
             cross_section_rank_ic_days=sum(v is not None for v in cross),
             overlapping_daily_targets=step==DAY,
             independent_samples_claimed=False)
    if path_p:
        pp,yy=np.stack(path_p),np.stack(path_y)
        # Within each asset/origin/channel: path correlation, then average.
        # Do not pool raw dollar levels from assets with different price scales.
        ic=[];ric=[]
        for a in range(len(pp)):
            for j in range(10):
                for c in range(4):
                    v=corr(pp[a,j,:,c],yy[a,j,:,c]);r=corr(pp[a,j,:,c],yy[a,j,:,c],True)
                    if v is not None:ic.append(v)
                    if r is not None:ric.append(r)
        out['normalized_ohlc_path']=dict(mae=float(np.abs(pp-yy).mean()),mse=float(np.square(pp-yy).mean()),
            constant_price_mae=float(np.abs(yy).mean()),constant_price_mse=float(np.square(yy).mean()),
            mean_within_path_ic=float(np.mean(ic)) if ic else None,
            mean_within_path_rank_ic=float(np.mean(ric)) if ric else None,valid_path_channels=len(ic),
            convention='Normalize by current close; within-path OHLC correlation then average; not original benchmark normalization.')
    return out


def main():
    plan=make_plan();m,z,p=load();daily=daily_bars(z)
    start=timestamp(plan['calibration']['start']);end=timestamp(plan['comparison']['end'])
    results={}
    for kind in ['daily','hourly','chronos_daily']:
        bars=z if kind=='hourly' else daily
        data,_=forecast_files(kind,plan,bars,start,end)
        step,horizon=(HOUR,12) if kind=='hourly' else (DAY,10)
        results[kind]={phase:evaluate(data,bars,step,horizon,timestamp(plan[phase]['start']),timestamp(plan[phase]['end']),m.symbols)
                       for phase in ['calibration','validation','comparison']}
    write(HERE/'prediction_metrics.json',dict(results=results,used_for_policy_selection=False,
          labels_require_full_horizon_inside_phase=True,
          interpretation='Directional accuracy compared with positive-target fraction; random-walk price baseline included. Overlap and small samples prevent independent significance claims.'))
    print(json.dumps({k:v['comparison']['pooled_return'] for k,v in results.items()}))


if __name__=='__main__':main()
