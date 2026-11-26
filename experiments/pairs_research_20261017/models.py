"""Past-only formation, copula conditioning and finite-horizon liquidation.

These are explicitly documented adaptations, not exact paper reproductions.
EG uses MacKinnon cointegration p-values, not an ordinary residual ADF p-value.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests


def segmented_area(path):
    """Unsigned areas between a piecewise linear path and its end-point chord."""
    p=np.asarray(path,float)-path[0]
    chord=p[-1]; cross=p[:,0]*chord[1]-p[:,1]*chord[0]
    polygon=[p[0]]; total=0.
    def area(v):
        v=np.asarray(v);return abs(float(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))))/2
    for k in range(1,len(p)):
        if cross[k-1]*cross[k]<0:
            lam=cross[k-1]/(cross[k-1]-cross[k]);hit=p[k-1]+lam*(p[k]-p[k-1])
            polygon.append(hit);total+=area(polygon);polygon=[hit]
        polygon.append(p[k])
        if cross[k]==0 and k>0:
            total+=area(polygon);polygon=[p[k]]
    if len(polygon)>1:total+=area(polygon)
    return total


def estimate(y,x,mask,step=6):
    take=np.flatnonzero(mask)[::step]
    if len(take)<60:return None
    yy,xx=y[take],x[take]
    a,b=np.linalg.lstsq(np.c_[np.ones(len(xx)),xx],yy,rcond=None)[0]
    residual=y-a-b*x;sd=float(np.std(residual[mask],ddof=1))
    if sd<1e-10:return None
    adjacent=mask[:-1]&mask[1:]
    past=residual[:-1][adjacent];future=residual[1:][adjacent]
    if len(past)<60:return None
    offset,rho=np.linalg.lstsq(np.c_[np.ones(len(past)),past],future,rcond=None)[0]
    half=float(-np.log(2)/np.log(rho)) if 0<rho<1 else None
    mean=float(offset/(1-rho)) if half is not None else 0.
    # Both halves are strictly within the formation period.
    betas=[]
    for ix in np.array_split(take,2):
        betas.append(float(np.linalg.lstsq(np.c_[np.ones(len(ix)),x[ix]],y[ix],rcond=None)[0][1]))
    stability=abs(betas[1]-betas[0])/max(abs(b),1e-8)
    p=float(coint(yy,xx,trend='c',maxlag=min(12,len(take)//10),autolag='aic')[1])
    return dict(alpha=float(a),beta=float(b),sd=sd,rho=float(rho),mean=mean,
                half_life=half,p=p,beta_relative_change=float(stability),samples=len(take))


def eligible(f,screen):
    return bool(f and f['q']<=screen['fdr_q'] and screen['beta_min']<=f['beta']<=screen['beta_max']
        and f['half_life'] is not None and screen['half_life_min']<=f['half_life']<=screen['half_life_max']
        and f['beta_relative_change']<=screen['maximum_beta_relative_change'])


def copula_fit(values):
    """Rank margins; Gaussian and Student-t (df grid) selected by AIC."""
    n=len(values);u=np.column_stack([stats.rankdata(values[:,j])/(n+1) for j in range(2)])
    fits=[]
    for df in (None,3.,5.,10.,30.):
        z=stats.norm.ppf(u) if df is None else stats.t.ppf(u,df)
        def objective(rho):
            det=1-rho*rho;q=(z[:,0]**2-2*rho*z[:,0]*z[:,1]+z[:,1]**2)/det
            if df is None:ll=-.5*np.log(det)-.5*q+.5*(z*z).sum(1)
            else:
                joint=special.gammaln((df+2)/2)-special.gammaln(df/2)-np.log(df*np.pi)-.5*np.log(det)-(df+2)/2*np.log1p(q/df)
                ll=joint-stats.t.logpdf(z[:,0],df)-stats.t.logpdf(z[:,1],df)
            return -float(ll.sum())
        opt=optimize.minimize_scalar(objective,bounds=(-.98,.98),method='bounded')
        fits.append(dict(df=df,rho=float(opt.x),aic=2*(1 if df is None else 2)+2*float(opt.fun)))
    best=min(fits,key=lambda f:f['aic'])
    return dict(**best,margins=[np.sort(values[:,j]) for j in range(2)],fits=fits)


def conditional(c,values):
    u=np.array([np.clip((np.searchsorted(c['margins'][j],values[j],side='right')+.5)/(len(c['margins'][j])+1),1e-5,1-1e-5) for j in range(2)])
    rho,df=c['rho'],c['df'];z=stats.norm.ppf(u) if df is None else stats.t.ppf(u,df)
    if df is None:return stats.norm.cdf((z-rho*z[::-1])/np.sqrt(1-rho*rho))
    scale=np.sqrt((df+z[::-1]**2)*(1-rho*rho)/(df+1))
    return stats.t.cdf((z-rho*z[::-1])/scale,df+1)


def stopping_tables(f,cost,horizon=72,carry=(0.,0.)):
    """Discrete OU single-cycle stopping approximation, per gross notional.

    Terminal holding value pays the exit cost. Flat state may wait; opening is
    allowed once, closing ends that cycle. Not the continuous-time paper solver.
    """
    z=np.linspace(-6,6,241);dz=z[1]-z[0];rho=f['rho']
    innovation=max(np.sqrt(1-rho*rho),.02)
    boundaries=np.r_[-np.inf,(z[:-1]+z[1:])/2,np.inf]
    trans=np.diff(stats.norm.cdf((boundaries[None,:]-rho*z[:,None])/innovation),axis=1)
    gain=f['sd']/(1+f['beta'])*(rho-1)*z
    hold=np.full((horizon+1,2,len(z)),-cost);flat=np.zeros((horizon+1,len(z)))
    enter=np.zeros((horizon+1,2,len(z)));continue_value=np.zeros_like(enter)
    for h in range(1,horizon+1):
        cont=np.stack([d*gain-carry[k]+trans@hold[h-1,k] for k,d in enumerate((-1,1))])
        continue_value[h]=cont;hold[h]=np.maximum(-cost,cont)
        enter[h]=cont-cost
        flat[h]=np.maximum(trans@flat[h-1],np.max(enter[h],axis=0))
    return dict(grid=z,hold=hold,flat=flat,entry=enter,continuation=continue_value,transition=trans)


def build_formations(m,cfg,lookback):
    logs=np.log(m.close);names=list(m.symbols);result=[]
    # Anchor to an absolute week, not input length: prefix recomputation is exact.
    for i in range(lookback,len(m.times)):
        if m.times[i]%(cfg['refit_hours']*3600)!=0:continue
        lo=i-lookback;forms=[]
        for pair in cfg['universe']:
            a,b=[names.index(s) for s in pair];mask=(m.volume[lo:i,a]>0)&(m.volume[lo:i,b]>0)&(m.trades[lo:i,a]>0)&(m.trades[lo:i,b]>0)
            f=estimate(logs[lo:i,a],logs[lo:i,b],mask,cfg['sampling_hours'])
            if f:
                f.update(key='/'.join(pair),assets=[a,b],kind='log',formation_index=i,
                    trained_through=int(m.times[i-1]),relative_funding=float(np.mean(m.funding[max(lo,i-168):i,a]-f['beta']*m.funding[max(lo,i-168):i,b])/(1+f['beta'])))
                areas=[segmented_area(logs[j-cfg['strategy']['signature_hours']+1:j+1,[a,b]]) for j in range(lo+60,i,6)]
                f['signature_threshold']=float(np.mean(areas)) if areas else 0.
            forms.append(f)
        # Correct across the declared family at each formation, including failures.
        q=multipletests([f['p'] if f else 1. for f in forms],method='fdr_bh')[1]
        accepted=[];diagnostics=[]
        for f,qq,pair in zip(forms,q,cfg['universe']):
            if f:
                f['q']=float(qq);f['eligible']=eligible(f,cfg['screen']);diagnostics.append({k:v for k,v in f.items()})
                if f['eligible']:accepted.append(f)
            else:diagnostics.append(dict(key='/'.join(pair),eligible=False,reason='insufficient_joint_observations'))
        # Reference-asset price-level spreads: buying S1 and selling S2 exactly
        # cancels reference units. Coin quantities are -beta1 and +beta2.
        refrows=[];legs=[]
        for group in cfg['reference_groups']:
            r,a,b=[names.index(s) for s in group];mask=np.all(m.volume[lo:i,[r,a,b]]>0,axis=1)&np.all(m.trades[lo:i,[r,a,b]]>0,axis=1)
            fs=[estimate(m.close[lo:i,r],m.close[lo:i,j],mask,cfg['sampling_hours']) for j in (a,b)]
            legs.extend(fs);refrows.append((group,(r,a,b),mask,fs))
        qs=multipletests([f['p'] if f else 1. for f in legs],method='fdr_bh')[1]
        for f,qv in zip(legs,qs):
            if f:f['q']=float(qv)
        copulas=[]
        for group,(r,a,b),mask,fs in refrows:
            # Raw-price slopes have units; log-beta bounds do not apply here.
            ok=all(f and f['q']<=cfg['screen']['fdr_q'] and f['beta']>0 and f['half_life'] is not None
                   and 3<=f['half_life']<=168 and f['beta_relative_change']<=1 for f in fs)
            if ok:
                values=np.column_stack([m.close[lo:i,r]-f['alpha']-f['beta']*m.close[lo:i,j] for f,j in zip(fs,(a,b))])
                copulas.append(dict(key='ref:'+('/'.join(group)),kind='reference',assets=[a,b],reference=r,
                    legs=fs,copula=copula_fit(values[mask][::cfg['sampling_hours']]),formation_index=i,trained_through=int(m.times[i-1])))
        result.append(dict(index=i,time=int(m.times[i]),models=accepted,copulas=copulas,diagnostics=diagnostics,
                           reference_diagnostics=[dict(group=g,legs=fs) for g,_,_,fs in refrows]))
    return result
