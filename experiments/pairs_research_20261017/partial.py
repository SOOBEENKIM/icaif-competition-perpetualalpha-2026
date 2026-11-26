"""Permanent/transitory spread decomposition, an explicitly older foundation.

Clegg-Krauss partial cointegration combined with 2025/26 cost-aware finite
horizon timing. Fixed past-only OLS beta is an adaptation; joint beta MLE and
the paper's calibrated likelihood-ratio significance test are not reproduced.
"""
import json,copy,gzip,time
import numpy as np
from scipy import optimize
from .models import stopping_tables
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from .policy import PairsController
from .research import HERE,ROOT,save,sha
from experiments.cost_horizon_20261016.research import data,ts,SecondController
from experiments.complementary_alpha_20261017.research import protected
from experiments.performance_attribution_20261016.engine import simulate
from experiments.q_execution_20261017.metrics import official_metrics,paired_uncertainty


def fit_partial(residual):
    y=np.asarray(residual,float);delta=np.diff(y);variance=max(float(np.nanvar(delta)),1e-10)
    kf=KalmanFilter(k_endog=1,k_states=2,k_posdef=2)
    kf.bind(np.ascontiguousarray(y[:,None]));kf.design=np.array([[1.,1.]])
    kf.selection=np.eye(2);kf.obs_cov=np.array([[1e-12]]);kf.loglikelihood_burn=1
    def set_params(theta):
        qr,qm=np.exp(theta[:2]);rho=theta[2]
        kf.transition=np.diag([1.,rho]);kf.state_cov=np.diag([qr,qm])
        kf.initialize_known(np.array([y[0],0.]),np.diag([variance,qm/(1-rho*rho)]))
        return qr,qm,rho
    def loss(theta):set_params(theta);return -float(kf.loglike())
    starts=[np.array([np.log(variance*.5),np.log(variance*.5),rho]) for rho in (.8,.97)]
    fits=[optimize.minimize(loss,start,method='L-BFGS-B',bounds=[(np.log(1e-12),np.log(.1))]*2+[(.05,.9995)],options={'maxiter':100}) for start in starts]
    best=min(fits,key=lambda f:f.fun);qr,qm,rho=set_params(best.x);result=kf.filter()
    finite=delta[np.isfinite(delta)];rwvar=max(float(np.mean(finite**2)),1e-12)
    rw_nll=.5*float(np.sum(np.log(2*np.pi*rwvar)+finite**2/rwvar))
    return dict(q_rw=float(qr),q_mr=float(qm),rho=float(rho),sd=float(np.sqrt(qm/(1-rho*rho))),
        half_life=float(-np.log(2)/np.log(rho)),mr_variance_fraction=float(2*qm/(2*qm+(1+rho)*qr)),
        fitted_state=result.filtered_state[:,-1].copy(),fitted_cov=result.filtered_state_cov[:,:,-1].copy(),
        aic=float(6+2*best.fun),rw_aic=float(2+2*rw_nll),converged=bool(best.success))


def filter_step(state,cov,value,f):
    trans=np.diag([1.,f['rho']]);prediction=trans@state
    p=trans@cov@trans.T+np.diag([f['q_rw'],f['q_mr']]);innovation=value-prediction.sum()
    variance=p.sum()+1e-12;gain=p.sum(axis=1)/variance
    updated=prediction+gain*innovation
    # Joseph update retains symmetry and positive semidefiniteness numerically.
    matrix=np.eye(2)-gain[:,None]@np.ones((1,2))
    return updated,matrix@p@matrix.T+np.outer(gain,gain)*1e-12


def formations(m,cfg,window):
    names=list(m.symbols);logs=np.log(m.close);out=[]
    for i in range(window,len(m.times)):
        if m.times[i]%(168*3600):continue
        models=[];diagnostics=[]
        for pair in cfg['universe']:
            a,b=[names.index(s) for s in pair];lo=i-window
            mask=(m.volume[lo:i,a]>0)&(m.volume[lo:i,b]>0)&(m.trades[lo:i,a]>0)&(m.trades[lo:i,b]>0)
            if mask.mean()<.8:
                diagnostics.append(dict(key='/'.join(pair),eligible=False,reason='less_than_80pct_joint_observations'));continue
            # Fits only available prices; timestamps remain hourly and missing
            # observations become missing updates, never compressed elapsed time.
            y,x=logs[lo:i,a],logs[lo:i,b];alpha,beta=np.linalg.lstsq(np.c_[np.ones(mask.sum()),x[mask]],y[mask],rcond=None)[0]
            if not .25<=beta<=4:
                diagnostics.append(dict(key='/'.join(pair),eligible=False,reason='hedge_ratio'));continue
            residual=y-alpha-beta*x
            residual=np.where(mask,residual,np.nan)
            if not np.isfinite(residual[0]):
                diagnostics.append(dict(key='/'.join(pair),eligible=False,reason='missing_initial'));continue
            f=fit_partial(residual)
            f.update(key='/'.join(pair),assets=[a,b],kind='log',formation_index=i,trained_through=int(m.times[i-1]),
                alpha=float(alpha),beta=float(beta),mean=0.,relative_funding=float(np.mean(m.funding[max(lo,i-168):i,a]-beta*m.funding[max(lo,i-168):i,b])/(1+beta)))
            f['eligible']=bool(f['converged'] and 3<=f['half_life']<=168 and f['mr_variance_fraction']>=.5 and f['aic']<f['rw_aic']-2)
            diagnostics.append(f)
            if f['eligible']:models.append(f)
        out.append(dict(index=i,time=int(m.times[i]),models=models,copulas=[],diagnostics=diagnostics))
    return out


class PartialController(PairsController):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.filters={}
    def signal(self,f,i,direction=0,held=0):
        key=f['key'],f['formation_index'];a,b=f['assets']
        if key not in self.filters:
            self.filters[key]=[f['formation_index']-1,np.array(f['fitted_state']),np.array(f['fitted_cov'])]
        cursor,state,cov=self.filters[key]
        for j in range(cursor+1,i+1):
            if np.any(self.m.volume[j,[a,b]]<=0) or np.any(self.m.trades[j,[a,b]]<=0):
                transition=np.diag([1.,f['rho']]);state=transition@state
                cov=transition@cov@transition.T+np.diag([f['q_rw'],f['q_mr']]);continue
            value=np.log(self.m.close[j,a])-f['alpha']-f['beta']*np.log(self.m.close[j,b])
            state,cov=filter_step(state,cov,value,f)
        self.filters[key]=[i,state,cov]
        # Reuse audited entry/exit and risk mechanisms with only the transitory
        # component as the expected-reversion signal. The hedge beta stays fixed.
        effective=dict(f,alpha=float(np.log(self.m.close[i,a])-f['beta']*np.log(self.m.close[i,b])-state[1]))
        signal=super().signal(effective,i,direction,held)
        signal['transitory']=float(state[1]);signal['permanent']=float(state[0])
        return signal


def main():
    out=HERE/'partial_artifacts';out.mkdir(exist_ok=True);(out/'runs').mkdir(exist_ok=True)
    if (out/'frozen.json').exists():raise FileExistsError('Frozen partial trial already exists')
    cfg=json.loads((HERE/'protocol.json').read_text());inp,a,p,m=data();p=copy.deepcopy(p)
    p['scenarios']['execution_stress']=dict(impact_bps=20.,participation=.0025,adverse_funding_bps_hour=0.)
    p['scenarios']['funding_stress']=dict(impact_bps=20.,participation=.0025,adverse_funding_bps_hour=1.)
    trial=dict(created_after_primary_failure=True,variants=['ou_504','ou_1440','stopping_504','stopping_1440'],
        selection='positive base and execution-stress PnL, engineering pass, choose largest base PnL; diagnostic only if none eligible',
        state='one random walk plus one AR1, past-only Kalman filter; no backward smoothing',screen='AIC better than RW by 2; MR variance >=0.5; half-life3..168; beta.25..4; >=80% joint fresh observations',
        evidence='Model-based screen, not a calibrated partial-cointegration significance test',same_history_reused=True,
        source={str(f.relative_to(ROOT)):sha(f) for f in [HERE/'partial.py',HERE/'policy.py',HERE/'models.py',HERE/'protocol.json']},protected=protected(),at=time.time())
    save(out/'frozen.json',trial)
    forms={}
    for w in cfg['formation_hours']:
        forms[w]=formations(m,cfg,w);save(out/f'formations_{w}.json',forms[w]);print('PARTIAL FORMS',w,len(forms[w]),sum(len(f['models']) for f in forms[w]),flush=True)
    rows=[];runs={}
    def replay(name,phase,scenario):
        family,window=name.rsplit('_',1);begin,end=cfg['periods'][phase];first,last=m.exact(ts(begin)),m.exact(ts(end))
        control=PartialController(m,forms[int(window)],cfg,family,p['scenarios'][scenario],last)
        r=simulate(m,np.zeros_like(a['candidate']),a['sigma'],p,first,last,scenario,control)
        r.update(policy='partial_'+name,phase=phase,official=official_metrics(r),events=control.events)
        r['metrics']['daily_std_sample']=r['official']['daily_sample_std']
        r['metrics']['partial_entry_aborts']=sum(e['event']=='partial_entry_abort' for e in control.events)
        r['metrics']['pair_entry_intents']=sum(e['event']=='entry_intent' for e in control.events)
        label=f'{phase}-{name}-{scenario}'
        with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append({k:r[k] for k in ('policy','phase','scenario','metrics','official','gates')});runs[label]=r
        print(label,'PnL',round(r['metrics']['net_pnl'],2),'Q',r['official']['financial_score'],'fills',r['metrics']['fills'],flush=True)
        return r
    ranking=[]
    for name in trial['variants']:
        b=replay(name,'selection','base');s=replay(name,'selection','execution_stress')
        ranking.append(dict(name=name,pnl=b['metrics']['net_pnl'],Q=b['official']['financial_score'],eligible=bool(b['metrics']['roi']>0 and s['metrics']['roi']>0 and all(b['gates'].values()) and all(s['gates'].values()))))
    ranking.sort(key=lambda r:(-r['eligible'],-r['pnl'],-r['Q'],r['name']));selected=ranking[0]['name']
    selection=dict(selected=selected,eligible=ranking[0]['eligible'],ranking=ranking,at=time.time(),frozen_before_comparison=True);save(out/'selection.json',selection)
    comparisons=[]
    for phase in ('comparison','recent'):
        for scenario in ('base','execution_stress','funding_stress'):
            r=replay(selected,phase,scenario)
            with gzip.open(HERE/'artifacts'/'runs'/f'{phase}-second-{scenario}.json.gz','rt') as f:b=json.load(f)
            comparisons.append(dict(phase=phase,scenario=scenario,pnl_difference=r['metrics']['net_pnl']-b['metrics']['net_pnl'],Q_difference=r['official']['financial_score']-b['official']['financial_score'],uncertainty=paired_uncertainty(r['official'],b['official'])))
    save(out/'results.json',dict(selection=selection,rows=rows,comparisons=comparisons,protected=protected(),historical_independence=False,live_orders_sent=0,live_replacement_approved=False))

if __name__=='__main__':main()
