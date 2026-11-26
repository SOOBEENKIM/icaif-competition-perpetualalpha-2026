"""Risk-scaled coherent forecasts, with explicit uncertainty ablations."""
import numpy as np


def project(mean,beta,pairs,valid,unit):
    kk=np.flatnonzero(valid&np.isfinite(mean)&np.isfinite(beta)&(beta>0)&np.isfinite(unit)&(unit>0))
    out=np.array(mean,copy=True)
    if not len(kk):return out,dict(pairs=0,rank=0,noise=0.,stein_scale=1.,removed_energy=0.,original_energy=0.)
    w=np.zeros((len(kk),10))
    for row,k in enumerate(kk):
        a,b=pairs[k];w[row,a]=1/(1+beta[k]);w[row,b]=-beta[k]/(1+beta[k])
    w/=unit[kk,None];normalized=mean[kk]/unit[kk]
    u,s,_=np.linalg.svd(w,full_matrices=False);rank=int(np.sum(s>max(w.shape)*np.finfo(float).eps*s[0]));basis=u[:,:rank]
    coherent=basis@(basis.T@normalized);res=normalized-coherent;df=len(kk)-rank
    noise=float(res@res/df) if df>0 else 0.
    stein=float(max(0.,1-max(0,rank-2)*noise/max(float(coherent@coherent),1e-15)))
    out[kk]=coherent*unit[kk]
    return out,dict(pairs=len(kk),rank=rank,noise=noise,stein_scale=stein,removed_energy=float(res@res),original_energy=float(normalized@normalized))


def build(source):
    arms=['coherent_scaled_72','scaled_stein_72','scaled_consensus_72']
    states={a:{k:np.array(v,copy=True) for k,v in source.items()} for a in arms};records=[]
    for i,t in enumerate(source['times']):
        valid=source['economic_available'][i].copy();raw=source['mu'][i]
        projected,audit=project(raw,source['beta'][i],source['pairs'],valid,source['vol'][i]*np.sqrt(72))
        for arm,out in states.items():
            mu=projected*audit['stein_scale'] if arm=='scaled_stein_72' else projected
            allowed=valid.copy()
            if arm=='scaled_consensus_72':
                allowed &= raw*projected>0
                mu=np.sign(projected)*np.minimum(abs(projected),abs(raw))
            out['mu'][i]=mu;out['economic_action'][i]=np.where(allowed&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
        if audit['pairs']:records.append(dict(source=int(t),**audit))
    return states,records
