"""Causal venue basis and relative trend experts. No labels from the future."""
from itertools import combinations
import numpy as np
from .protocol import SYMBOLS
PAIRS=list(combinations(range(10),2))
HORIZON=24

def ema(x,span):
 out=float(x[0]);a=2/(span+1)
 for v in x[1:]:out=a*float(v)+(1-a)*out
 return out

def basis_projection(b):
 """Local equilibrium fit shrunk toward rho=1 (no convergence)."""
 b=np.asarray(b,float)
 return basis_transitions(b[:-1],b[1:],b[-1])

def basis_transitions(x,y,current):
 x=np.asarray(x,float);dy=np.asarray(y,float)-x;mean=float(x.mean()) if len(x) else 0.;xc=x-mean
 ss=float(xc@xc);penalty=24*float(np.var(x))
 if len(x)<36 or ss<1e-16:return 0.,dict(n=len(x),rho=1.,uncertainty=0.,raw=0.)
 slope=float(xc@dy/(ss+penalty));rho=float(np.clip(1+slope,.8,1.02))
 resid=dy-(rho-1)*xc
 se=float(np.std(resid,ddof=1)/np.sqrt(ss+penalty))
 raw=(rho**HORIZON-1)*(current-mean)
 uncertainty=abs(HORIZON*rho**(HORIZON-1)*(current-mean))*se
 value=float(np.sign(raw)*max(0.,abs(raw)-uncertainty))
 return float(np.clip(value,-.1,.1)),dict(n=len(x),rho=rho,uncertainty=uncertainty,raw=float(raw))

def estimates(source,ref,d,carry,arm):
 """All reference/PAS rows are sliced at source before any statistics."""
 ri=np.searchsorted(ref['market_times'],source,side='right')
 rt=ref['market_times'][:ri];rp=ref['close'][:ri];rv=ref['volume'][:ri]
 if not len(rt) or rt[-1]!=source:return [],[]
 end=np.searchsorted(d['times'],source*1000,side='right')
 times=d['times'][:end];pas=d['close'][:end];age=d['age_ms'][:end]
 if not len(times) or times[-1]!=source*1000:return [],[]
 ht=rt[-97:];hi=np.searchsorted(times,ht*1000,side='right')-1
 validtime=hi>=0;hi=np.maximum(hi,0)
 hp=pas[hi];fresh=(age[hi]<=3600000)&validtime[:,None]&np.isfinite(hp)&(hp>0)
 basis=np.log(np.divide(hp,rp[-len(ht):],out=np.ones_like(hp),where=fresh))
 rr=np.diff(np.log(rp[-721:]),axis=0)
 active=(rv[-len(rr):]>0)&(rv[-len(rr)-1:-1]>0)
 forecasts=[];audit=[]
 for j,k in PAIRS:
  pair=[SYMBOLS[j],SYMBOLS[k]];ok=active[:,j]&active[:,k]&np.isfinite(rr[:,j])&np.isfinite(rr[:,k])
  row=dict(pair=pair,source=int(source),reference_count=int(ok.sum()),eligible=False)
  if ok.sum()<120:row['reason']='reference_observations';audit.append(row);continue
  a=rr[ok,j];b=rr[ok,k];cov=np.cov(a,b);beta=float(cov[0,1]/max(cov[1,1],1e-16))
  corr=float(cov[0,1]/max(np.sqrt(cov[0,0]*cov[1,1]),1e-16));row.update(beta=beta,correlation=corr)
  if not .1<=beta<=10 or corr<.2:row['reason']='factor_relationship';audit.append(row);continue
  good=fresh[:,j]&fresh[:,k]
  # Each included transition is one hour; an old gap need not erase all earlier evidence.
  adjacent=good[:-1]&good[1:]&(np.diff(ht)==3600)
  if not good[-1] or adjacent.sum()<36:row['reason']='pas_basis_history';audit.append(row);continue
  x=basis[:,j]-beta*basis[:,k]
  correction,fit=basis_transitions(x[:-1][adjacent],x[1:][adjacent],x[-1])
  rel=rr[:,j]-beta*rr[:,k]
  trend=float(np.clip(.25*HORIZON*(ema(rel,6)+ema(rel,24))/2,-.1,.1))
  move=correction if arm=='basis_reversion' else trend if arm=='relative_trend' else .5*(correction+trend)
  if arm=='relative_reversal':move=-trend
  if arm=='pas_spread_reversion':
   levels=np.log(hp[:,j])-beta*np.log(hp[:,k])
   move,spread_fit=basis_transitions(levels[:-1][adjacent],levels[1:][adjacent],levels[-1])
   rho=spread_fit['rho']
   if not 0<rho<1 or -np.log(2)/np.log(rho)>72:move=0.
   row['pas_spread_fit']=spread_fit
  weights=np.array([1.,-beta])/(1+beta)
  offsets=move/(1+beta*beta)*np.array([1.,-beta])
  anchors=hp[-1,[j,k]];terminal=anchors*np.exp(offsets)
  gains=[]
  for side in (-1,1):
   cash=-side*weights*carry[[j,k]]
   net=float(side*weights@(terminal/anchors-1)+np.where(cash>0,.5*cash,cash).sum())
   gains.append(net)
  ix=int(np.argmax(gains));direction=(-1,1)[ix] if gains[ix]>.0043 else 0
  vol=max(float(np.std(rel[ok],ddof=1)/(1+beta)),.0001)
  row.update(eligible=True,reason='forecast_ready',basis=fit,basis_correction=correction,trend=trend,
             expected_price_return=float(weights@(terminal/anchors-1)),best_net_before_cost=max(gains),direction=direction)
  audit.append(row)
  forecasts.append(dict(model='decision_gate',pair=pair,ids=[j,k],forecast_source=int(source),
   available_time=float(source+240),horizon=HORIZON,source_prices=anchors.tolist(),
   terminal_prices=terminal.tolist(),source_beta=beta,source_vol=vol,mu=float(weights@(terminal/anchors-1)),
   economic_available=True,economic_action=direction,reference_relationship=corr,
   research_arm=arm,uses_pas_prices=True))
 return forecasts,audit
