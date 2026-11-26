"""Project relative forecasts onto a common vector of asset returns.

For source-fixed beta, each true pair return is a row of W times the same
future asset-return vector. A forecast component orthogonal to col(W) is
impossible. Removing it cannot increase total squared error on that complete
aligned cohort. This says nothing by itself about direction accuracy or PnL.
"""
import numpy as np


def project(mean,beta,pairs,valid):
    kk=np.flatnonzero(valid&np.isfinite(mean)&np.isfinite(beta)&(beta>0))
    out=np.array(mean,copy=True)
    if not len(kk):return out,dict(pairs=0,rank=0,removed_energy=0.,original_energy=0.)
    w=np.zeros((len(kk),10))
    for row,k in enumerate(kk):
        a,b=pairs[k];w[row,a]=1/(1+beta[k]);w[row,b]=-beta[k]/(1+beta[k])
    u,s,_=np.linalg.svd(w,full_matrices=False)
    threshold=max(w.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    rank=int(np.sum(s>threshold));basis=u[:,:rank]
    out[kk]=basis@(basis.T@mean[kk]);res=mean[kk]-out[kk]
    assert np.max(abs(w.T@res),initial=0)<1e-10
    return out,dict(pairs=len(kk),rank=rank,removed_energy=float(res@res),original_energy=float(mean[kk]@mean[kk]))


def build(source):
    out={k:np.array(v,copy=True) for k,v in source.items()};rows=[]
    for i,t in enumerate(out['times']):
        valid=source['economic_available'][i].copy()
        if 'relationship_allowed' in source:valid &= source['relationship_allowed'][i]
        if 'regime_allowed' in source:valid &= source['regime_allowed'][i]
        mu,audit=project(source['mu'][i],source['beta'][i],source['pairs'],valid)
        out['mu'][i]=mu;out['economic_action'][i]=np.where(valid&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
        if audit['pairs']:rows.append(dict(source=int(t),**audit))
    return out,rows
