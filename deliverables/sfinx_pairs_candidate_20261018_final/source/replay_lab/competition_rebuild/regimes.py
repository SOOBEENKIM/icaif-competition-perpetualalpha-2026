"""Independent JM -> XGBoost adaptation of Shu/Yu/Mulvey; no copied source."""
import numpy as np
import pandas as pd
import xgboost as xgb
from .common import OUT,load,write,stamp,covariance_path,run
from .models import rolling


def regime_features(m):
    ids=np.flatnonzero(m.times%21600==0);prices=m.close[ids]
    ret=np.diff(np.log(prices),axis=0,prepend=np.log(prices[:1]))
    means=[];risks=[]
    for half in (20,40,84):
        means.append(pd.DataFrame(ret).ewm(halflife=half,adjust=False).mean().to_numpy())
        risks.append(np.maximum(np.sqrt(pd.DataFrame(np.minimum(ret,0)**2).ewm(halflife=half,adjust=False).mean().to_numpy()),1e-5))
    x=np.stack([np.log(risks[0]),np.log(risks[2]),*means,*[a/b for a,b in zip(means,risks)]],axis=-1)
    return ids,x,ret


def dp_path(x,centres,penalty):
    loss=.5*((x[:,None]-centres[None])**2).sum(2);n=len(x)
    d=loss[0].copy();back=np.zeros((n,2),int)
    for t in range(1,n):
        options=d[:,None]+penalty*(1-np.eye(2));back[t]=options.argmin(0)
        d=loss[t]+options.min(0)
    states=np.zeros(n,int);states[-1]=d.argmin()
    for t in range(n-1,0,-1):states[t-1]=back[t,states[t]]
    return states,float(d.min()),d-d.min()


def fit_jump(xx,rr,penalty=10.):
    mean=xx.mean(0);std=np.maximum(xx.std(0),1e-6);x=(xx-mean)/std
    best=None
    for feature in (2,3,4,7):
        order=np.argsort(x[:,feature]);centres=x[order[[len(x)//4,3*len(x)//4]]].copy()
        old=None
        for _ in range(50):
            states,value,cost=dp_path(x,centres,penalty)
            if old is not None and np.array_equal(states,old):break
            old=states.copy()
            for k in (0,1):
                if np.any(states==k):centres[k]=x[states==k].mean(0)
        states,value,cost=dp_path(x,centres,penalty)
        if best is None or value<best['objective']:
            sums=[float(rr[states==k].sum()) for k in (0,1)]
            bull=int(np.argmax(sums))
            best=dict(mean=mean,std=std,centres=centres,states=states,bull=bull,cost=cost,objective=value,penalty=penalty,
                      counts=[int((states==k).sum()) for k in (0,1)],returns=sums)
    return best


def fit_classifier(xx,states,bull):
    y=(states[1:]==bull).astype(float)
    if len(np.unique(y))==1:return float(y[0])
    return xgb.train(dict(objective='binary:logistic',max_depth=2,eta=.05,min_child_weight=10,reg_lambda=10,
                         seed=20261013,nthread=1,tree_method='hist'),xgb.DMatrix(xx[:-1],label=y),num_boost_round=60)


def forecasts(m,stage):
    ids,x,r=regime_features(m);pred={k:np.full_like(m.close,.5) for k in ('jm','jm_xgb')};records=[]
    first=np.searchsorted(ids,1440);smooth={'jm':np.full(10,.5),'jm_xgb':np.full(10,.5)}
    alpha=1-np.exp(np.log(.5)/4)
    for begin in range(first,len(ids),56):
        finish=min(begin+56,len(ids));aa=max(28,begin-360);rec=dict(first_prediction_time=int(m.times[ids[begin]]),
          last_training_time=int(m.times[ids[begin-1]]),assets=[])
        for j in range(10):
            state=fit_jump(x[aa:begin,j],r[aa:begin,j]);classifier=fit_classifier(x[aa:begin,j],state['states'],state['bull'])
            xx=x[begin:finish,j];q=np.full(len(xx),classifier) if isinstance(classifier,float) else classifier.predict(xgb.DMatrix(xx))
            if not isinstance(classifier,float):
                target=stage/'models'/f'{begin}-{j}.json';target.parent.mkdir(exist_ok=True,parents=True);classifier.save_model(target)
            cost=state['cost'].copy()
            for k in range(begin,finish):
                point=(x[k,j]-state['mean'])/state['std']
                loss=.5*((state['centres']-point)**2).sum(1)
                cost=loss+(cost[:,None]+10.*(1-np.eye(2))).min(0);cost-=cost.min()
                probs={'jm':float(cost.argmin()==state['bull']),'jm_xgb':float(q[k-begin])}
                for key,val in probs.items():
                    smooth[key][j]=(1-alpha)*smooth[key][j]+alpha*val
                    pred[key][ids[k]:min(ids[k]+6,len(m.times)),j]=smooth[key][j]
            saved={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in state.items() if key!='states'};saved['asset']=j;rec['assets'].append(saved)
        records.append(rec);write(stage/'training.json',dict(folds=records,labels='Next six-hour training regime; no labels outside training boundary',all_history_reused=True))
        print('regime fold',len(records),rec['first_prediction_time'],flush=True)
    np.savez_compressed(stage/'forecasts.npz',**pred)
    return pred


def paths(m,prob,base):
    out=np.zeros_like(base);gate=base.copy();state=np.zeros(10,bool)
    ret=np.diff(np.log(m.close),axis=0,prepend=np.log(m.close[:1]));sigma=np.maximum(rolling(ret,168,'std')*np.sqrt(24),.003)
    held=np.zeros(10)
    for i in range(1440,len(base)):
        if m.times[i]%21600==0:
            state=np.where(prob[i]>.6,True,np.where(prob[i]<.4,False,state))
            fresh=(m.trades[i]>0)&(m.volume[i]>0)
            close=(~state)&(held!=0);held[close]=0
            enter=state&fresh&(held==0);held[enter]=np.minimum(.015,.00012/sigma[i,enter])
        out[i]=held
        # Reduce the incumbent only when the regime opposes its direction.
        conflict=((base[i]>0)&(~state))|((base[i]<0)&state)
        gate[i]=base[i]*np.where(conflict,.5,1.)
    return out,gate


def main():
    stage=OUT/'regimes';stage.mkdir(exist_ok=True)
    if (stage/'forecasts.npz').exists():raise RuntimeError('Do not overwrite completed forecasts')
    if (stage/'plan.json').exists():
        write(stage/'restart.json',dict(time=stamp(),reason='Fix ndarray serialization in training audit. No model, data, settings, or selection changes; first run stopped before PnL evaluation.'))
    else:write(stage/'plan.json',dict(created_at=stamp(),source='https://arxiv.org/html/2406.09578v2',
      hypothesis='Persistent asset-specific regimes may be easier to predict than noisy24hour returns.',
      paper_method='Jump-model temporal clustering, next-period XGBoost classification, exponential probability smoothing, regime-conditioned allocation.',
      changes='Six-hour sampling;5/10/21calendar-day feature halflives;90day training cap;14day refit;no external macro features;fixedlambda10;depth2/60trees;24hour probability half-life;0.6/0.4hysteresis;shared perpetual ledger. Independent adaptation, not reproduction.',
      trials=['jm_standalone','jm_xgb_standalone','v7_jm_half','v7_jm_xgb_half','v7_jm_gate','v7_jm_xgb_gate'],
      calibration='Same June1-July29 calibration and95%risk target as earlier stages',
      nomination='Net developmentROI>v7,STD<=v7,gates;rank worst21day excess;no later selection.',
      all_history_reused=True,prior_later_history_seen=True,live_orders_sent=0))
    m,z,b,p=load();sigmas=covariance_path(m,p);base=np.load(OUT/'expected_targets.npy');pred=forecasts(m,stage)
    models={'v7_control':base}
    for key,prob in pred.items():
        direct,gate=paths(m,prob,base);models[key+'_standalone']=direct;models['v7_'+key+'_half']=(base+direct)/2;models['v7_'+key+'_gate']=gate
    np.savez_compressed(stage/'raw_targets.npz',**models)
    evaluate_models(m,p,sigmas,models,stage)


def evaluate_models(m,p,sigma,models,stage):
    idx=lambda date:m.exact(int(np.datetime64(date,'s').astype('int64')))
    a=idx('2026-06-01T00:00:00');cut=idx('2026-07-29T00:00:00');rows=[]
    def evaluate(name,w,phase,start,end,scenario='base'):
        r=run(m,p,w,name,start,end,sigma,scenario,stage/'runs'/phase)
        row={k:r[k] for k in ('policy','scenario','start','end','metrics','gates')};row['phase']=phase;rows.append(row)
        write(stage/'running_results.json',rows);print(phase,name,scenario,round(r['metrics']['roi']*100,6),round(r['metrics']['daily_vol']*100,6),flush=True)
        return r
    base=models['v7_control'];ref=evaluate('v7_control',base,'calibration',a,cut);scaled={'v7_control':base};scales={'v7_control':1.}
    for name,w in models.items():
        if name=='v7_control':continue
        r=evaluate(name,w,'unscaled_calibration',a,cut);s=min(3.,.95*ref['metrics']['daily_vol']/max(r['metrics']['daily_vol'],1e-9))
        r=evaluate(name,w*s,'calibration',a,cut)
        if r['metrics']['daily_vol']>ref['metrics']['daily_vol']:
            s*=.95*ref['metrics']['daily_vol']/r['metrics']['daily_vol'];evaluate(name,w*s,'calibration_corrected',a,cut)
        scaled[name]=w*s;scales[name]=s
    write(stage/'scales.json',scales);np.savez_compressed(stage/'targets.npz',**scaled)
    phases={'development':(1439,cut)}
    for k in range(3):
        aa=idx('2026-05-18T00:00:00')+k*504;phases[f'window{k+1}']=(aa,aa+504)
    for phase,(aa,bb) in phases.items():
        for name,w in scaled.items():
            for scenario in (('base','stress') if phase=='development' else ('base',)):evaluate(name,w,phase,aa,bb,scenario)
    index={(r['phase'],r['policy'],r['scenario']):r for r in rows};ref=index['development','v7_control','base'];eligible=[]
    for name in scaled:
        if name=='v7_control':continue
        r=index['development',name,'base'];s=index['development',name,'stress']
        if r['metrics']['roi']>max(0,ref['metrics']['roi']) and r['metrics']['daily_vol']<=ref['metrics']['daily_vol'] and all(r['gates'].values()) and all(s['gates'].values()):
            differences=[index[f'window{k}',name,'base']['metrics']['roi']-index[f'window{k}','v7_control','base']['metrics']['roi'] for k in (1,2,3)]
            eligible.append(dict(policy=name,window_excess=differences,worst_window_excess=min(differences)))
    eligible.sort(key=lambda r:-r['worst_window_excess'])
    write(stage/'selection.json',dict(time=stamp(),nominee=eligible[0]['policy'] if eligible else None,eligible=eligible,all_history_reused=True,prior_later_history_seen=True))
    for phase,(aa,bb) in {'later_reused':(cut,len(m.times)-1),'whole_reused':(1439,len(m.times)-1),
      'last40_reused':(idx('2026-08-26T00:00:00'),idx('2026-10-05T16:00:00'))}.items():
        for name,w in scaled.items():
            for scenario in ('base','stress'):evaluate(name,w,phase,aa,bb,scenario)
    write(stage/'results.json',dict(completed_at=stamp(),routes=rows,live_orders_sent=0))

if __name__=='__main__':main()
