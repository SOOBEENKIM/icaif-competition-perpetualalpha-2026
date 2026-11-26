"""Source-day clustered uncertainty for the retained per-pair ridge.

The covariance describes estimation uncertainty, not total future return risk
or simultaneous coverage over45 dependent candidates. Raw labels are allowed
only after maturity; unclipped residuals expose tails omitted by training clip.
"""
import numpy as np


def covariance(x,y,times,maturity,source,model,spec):
    if np.any(maturity>=source):raise ValueError('Future target in uncertainty fit')
    if len(x)<spec['minimum_training_rows']:return None
    z=np.c_[np.ones(len(x)),(x-model['center'])/model['scale']]
    w=np.exp2(-(source-maturity)/(spec['half_life_days']*86400));w/=w.sum()
    penalty=np.eye(z.shape[1])*spec['ridge_penalty'];penalty[0,0]=spec['intercept_prior_rows']/len(x)
    bread=np.linalg.inv(z.T@(w[:,None]*z)+penalty)
    scores=z*(w*(y-z@model['direct']))[:,None]
    first=int(times.min()//86400);last=int(times.max()//86400)
    clustered=np.zeros((last-first+1,z.shape[1]))
    np.add.at(clustered,times//86400-first,scores)
    meat=clustered.T@clustered
    for lag in [1,2]:
        cross=clustered[lag:].T@clustered[:-lag]
        meat+=(1-lag/3)*(cross+cross.T)
    cov=bread@meat@bread
    vals,vecs=np.linalg.eigh(cov);cov=(vecs*np.maximum(vals,0))@vecs.T
    return dict(matrix=cov.tolist(),first_maturity=int(maturity.min()),last_maturity=int(maturity.max()),
                source=int(source),rows=len(x),source_day_clusters=len(clustered))


def conservative(raw,uncertainty,multiplier):
    return np.sign(raw)*np.maximum(0,abs(raw)-multiplier*uncertainty)
