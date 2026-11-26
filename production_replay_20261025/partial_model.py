"""All-pair partial-cointegration research; no live imports or order sender.

Clegg/Krauss supplies the permanent+AR1 state equations. Joint beta likelihood,
chronological 21-day predictive screening and fixed-source return calibration
are this experiment's adaptations, not a reproduction of published returns.
The selection screen is NOT a calibrated cointegration significance test.
"""
import ctypes
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Spec:
    fit_hours: int = 60*24
    validation_hours: int = 21*24
    refresh_hours: int = 7*24
    horizon: int = 72
    minimum_coverage: float = .5
    minimum_validation_labels: int = 4
    beta_min: float = .1
    beta_max: float = 10.
    half_life_min: float = 3.
    half_life_max: float = 168.
    burn_hours: int = 24
    cost_floor: float = .0038
    calibration_prior: float = 2.


_LIB=None
def _library():
    global _LIB
    if _LIB is not None:return _LIB
    src=Path(__file__).with_name('partial_filter.c')
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    build=src.parent/'_build';build.mkdir(exist_ok=True)
    target=build/f'partial_filter_{digest}.so'
    if not target.exists():
        temporary=target.with_suffix('.tmp.so')
        subprocess.run(['cc','-shared','-O3','-fPIC',str(src),'-o',str(temporary),'-lm'],check=True)
        temporary.replace(target)
    lib=ctypes.CDLL(str(target));lib.pci_filter.restype=None
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.pci_filter.argtypes=[f64,f64,u8,ctypes.c_size_t,*([ctypes.c_double]*4),f64]
    _LIB=lib;return lib


def filtered(y,x,valid,params):
    y=np.ascontiguousarray(y,dtype=np.float64);x=np.ascontiguousarray(x,dtype=np.float64)
    valid=np.ascontiguousarray(valid,dtype=np.uint8)
    if y.ndim!=1 or x.shape!=y.shape or valid.shape!=y.shape:raise ValueError('Bad filter shapes')
    beta,rho,qr,qm=map(float,params)
    if not 0<=rho<1 or min(qr,qm)<=0:raise ValueError('Invalid transition/noise')
    if not np.isfinite(y[valid.astype(bool)]).all() or not np.isfinite(x[valid.astype(bool)]).all():raise ValueError('Nonfinite observation')
    out=np.empty((len(y),4));_library().pci_filter(y,x,valid,len(y),beta,rho,qr,qm,out)
    return out


def python_filter(y,x,valid,params):
    """Small independent matrix reference used only for kernel regression tests."""
    beta,rho,qr,qm=params;state=np.zeros(2);cov=np.diag([1.,qm/(1-rho*rho)])
    trans=np.diag([1.,rho]);noise=np.diag([qr,qm]);rows=[];begun=False
    for yy,xx,good in zip(y,x,valid):
        nll=np.nan
        if begun:state=trans@state;cov=trans@cov@trans.T+noise
        if good:
            z=yy-beta*xx
            if not begun:state[0]=z;begun=True
            else:
                e=z-state.sum();s=cov.sum()+1e-12;k=cov.sum(axis=1)/s
                nll=.5*(np.log(2*np.pi*s)+e*e/s);state+=k*e
                j=np.eye(2)-k[:,None]*np.ones((1,2));cov=j@cov@j.T+np.outer(k,k)*1e-12
        rows.append([*state,cov[1,1],nll] if begun else [np.nan]*4)
    return np.array(rows)


def null_model(y,x,valid,spec=Spec()):
    ii=np.flatnonzero(valid);gap=np.diff(ii).astype(float)
    dy=np.diff(y[ii]);dx=np.diff(x[ii]);den=np.sum(dx*dx/gap)
    beta=float(np.clip(np.sum(dx*dy/gap)/max(den,1e-18),spec.beta_min,spec.beta_max))
    q=max(float(np.mean((dy-beta*dx)**2/gap)),1e-12)
    return beta,q


def null_nll(y,x,valid,beta,q):
    """Random-walk conditional density with actual elapsed hours across gaps."""
    out=np.full(len(y),np.nan);ii=np.flatnonzero(valid);gap=np.diff(ii)
    e=np.diff(y[ii]-beta*x[ii]);variance=q*gap
    out[ii[1:]]=.5*(np.log(2*np.pi*variance)+e*e/variance)
    return out


def fit(y,x,valid,spec=Spec()):
    if len(y)!=spec.fit_hours or np.mean(valid)<spec.minimum_coverage:return None
    rb,rq=null_model(y,x,valid,spec)
    def unpack(theta):return float(theta[0]),float(theta[1]),float(np.exp(theta[2])),float(np.exp(theta[3]))
    def loss(theta):
        a=filtered(y,x,valid,unpack(theta))[spec.burn_hours:,3]
        return float(np.nansum(a))
    bounds=[(spec.beta_min,spec.beta_max),(.05,.9995),(np.log(1e-12),np.log(.1)),(np.log(1e-12),np.log(.1))]
    starts=[[rb,rho,np.log(max(rq*.5,1e-12)),np.log(max(rq*.5,1e-12))] for rho in [.8,.98]]
    fits=[minimize(loss,z,method='L-BFGS-B',bounds=bounds,options={'maxiter':120,'ftol':1e-9}) for z in starts]
    good=[r for r in fits if r.success and np.isfinite(r.fun)]
    if not good:return None
    best=min(good,key=lambda r:r.fun);beta,rho,qr,qm=unpack(best.x)
    return dict(params=[beta,rho,qr,qm],null=[rb,rq],nll=float(best.fun),
        null_nll=float(np.nansum(null_nll(y,x,valid,rb,rq)[spec.burn_hours:])),
        half_life=float(-np.log(2)/np.log(rho)),converged=True,
        mean_reverting_variance_fraction=float(2*qm/(2*qm+(1+rho)*qr)),
        beta_at_bound=bool(beta<=spec.beta_min*1.001 or beta>=spec.beta_max*.999))


def expected_return(state,model,horizon):
    beta,rho,qr,qm=model['params']
    # Approximation to the fixed-entry-share return, subsequently calibrated
    # against actual simple-return labels, NEVER moving-residual contraction.
    return (rho**horizon-1.)*state[:,1]/(1.+beta)


def validate(y,x,valid,model,spec=Spec()):
    n=spec.fit_hours;v=spec.validation_hours;h=spec.horizon
    if len(y)!=n+v:raise ValueError('Fit/validation boundary mismatch')
    a=filtered(y,x,valid,model['params']);rb,rq=model['null']
    null=null_nll(y,x,valid,rb,rq)
    ll_valid=np.isfinite(a[n:,3])&np.isfinite(null[n:])
    delta=float(np.sum(null[n:][ll_valid])-np.sum(a[n:,3][ll_valid]))
    # All labels mature strictly BEFORE the next fit/decision boundary.
    origins=np.arange(n,n+v-h,h);origins=origins[valid[origins]&valid[origins+h]]
    bt=model['params'][0]
    pred=expected_return(a[origins],model,h)
    actual=(np.expm1(y[origins+h]-y[origins])-bt*np.expm1(x[origins+h]-x[origins]))/(1+bt)
    variance=(h*model['params'][2]+model['params'][3]*(1-model['params'][1]**(2*h))/(1-model['params'][1]**2))/(1+bt)**2
    unit=max(variance,1e-10)
    slope=float(np.clip(np.sum(pred*actual)/unit/(np.sum(pred*pred)/unit+spec.calibration_prior),0.,1.))
    mse=float(np.mean((pred-actual)**2)) if len(pred) else None
    null_mse=float(np.mean(actual**2)) if len(pred) else None
    reasons=[]
    if len(origins)<spec.minimum_validation_labels:reasons.append('few_disjoint_validation_labels')
    if not delta>0:reasons.append('random_walk_density_better')
    if mse is None or not mse<null_mse:reasons.append('zero_return_prediction_better')
    if slope<=0:reasons.append('nonpositive_fixed_hedge_calibration')
    if model['beta_at_bound']:reasons.append('hedge_at_optimization_bound')
    if not spec.half_life_min<=model['half_life']<=spec.half_life_max:reasons.append('inadmissible_half_life')
    return dict(eligible=not reasons,reasons=reasons,validation_log_score_advantage=delta,
        labels=len(origins),origin_offsets=origins.tolist(),last_maturity_offset=int(max(origins+h)) if len(origins) else None,
        uncalibrated_mse=mse,zero_mse=null_mse,calibration=slope)
